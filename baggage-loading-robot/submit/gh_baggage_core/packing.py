"""Vectorized sliding-window search for the best (x, y) landing spot for a
given footprint on a container's heightmap, using a heightmap/skyline
"place lowest & flattest" heuristic (a discretized variant of the classic
extreme-point / deepest-bottom-left-fill family of 3D bin-packing
constructors).
"""
from __future__ import annotations

import math

import numpy as np

from .container_state import ContainerState

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:  # pragma: no cover - very old numpy fallback
    sliding_window_view = None


def _windows(arr: np.ndarray, fw: int, fh: int) -> np.ndarray:
    if sliding_window_view is not None:
        return sliding_window_view(arr, (fw, fh))
    n0, n1 = arr.shape[0] - fw + 1, arr.shape[1] - fh + 1
    out = np.empty((n0, n1, fw, fh), dtype=arr.dtype)
    for i in range(n0):
        for j in range(n1):
            out[i, j] = arr[i:i + fw, j:j + fh]
    return out


def best_position(state: ContainerState, footprint_x: float, footprint_y: float, item_height: float,
                   avoid_soft_top: bool, avoid_priority_top: bool) -> dict | None:
    """Search the grid for the best anchor to place a `footprint_x x footprint_y`
    x `item_height` box. Returns None if it cannot fit anywhere."""
    n = state.grid_n
    fw = max(1, int(math.ceil(footprint_x / max(state.cell_w, 1e-6))))
    fh = max(1, int(math.ceil(footprint_y / max(state.cell_h, 1e-6))))
    if fw > n or fh > n:
        return None

    hmap = state.height_grid
    top_windows = _windows(hmap, fw, fh)
    top = top_windows.max(axis=(2, 3))
    bottom = top_windows.min(axis=(2, 3))
    flat = top - bottom

    fits_ceiling = (top + item_height) <= (state.ceiling_z + 1e-6)
    if not fits_ceiling.any():
        return None

    conflict = np.zeros_like(fits_ceiling, dtype=bool)
    if avoid_soft_top:
        conflict |= _windows(state.top_soft, fw, fh).any(axis=(2, 3))
    if avoid_priority_top:
        conflict |= _windows(state.top_prioritized, fw, fh).any(axis=(2, 3))

    valid = fits_ceiling & ~conflict
    used_conflict_relax = False
    if not valid.any():
        valid = fits_ceiling
        used_conflict_relax = True
        if not valid.any():
            return None

    # Primary: minimize resulting stack height (keeps the load low / stable
    # and leaves headroom for later items). Secondary: prefer a flat support
    # surface (reduces tip-over risk on settle). Tertiary: mild bottom-left
    # bias for a tidy, deterministic layout.
    ix_grid, iy_grid = np.meshgrid(np.arange(top.shape[0]), np.arange(top.shape[1]), indexing="ij")
    score = top * 1000.0 + flat * 10.0 + (ix_grid + iy_grid) * 1e-4
    score = np.where(valid, score, np.inf)
    ix, iy = np.unravel_index(np.argmin(score), score.shape)

    x_center = state.x_min + (ix + fw / 2.0) * state.cell_w
    y_center = state.y_min + (iy + fh / 2.0) * state.cell_h
    z_center = float(top[ix, iy]) + item_height / 2.0

    return {
        "x": x_center,
        "y": y_center,
        "z": z_center,
        "top": float(top[ix, iy]),
        "flat": float(flat[ix, iy]),
        "conflict": bool(conflict[ix, iy]) or used_conflict_relax,
        "fw": fw,
        "fh": fh,
    }
