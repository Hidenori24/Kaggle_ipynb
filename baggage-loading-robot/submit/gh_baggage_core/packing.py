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


def _sliding_max_axis0(arr: np.ndarray, fw: int) -> np.ndarray:
    """max over every length-fw window along axis 0, keeping axis 1 intact."""
    if sliding_window_view is not None:
        return sliding_window_view(arr, fw, axis=0).max(axis=-1)
    n0 = arr.shape[0] - fw + 1
    out = np.empty((n0, arr.shape[1]), dtype=arr.dtype)
    for i in range(n0):
        out[i] = arr[i:i + fw].max(axis=0)
    return out


# Items enter through the door (local y_min side) and slide straight in
# along +Y at roughly a constant X/Z until they reach their resting spot
# (see the simulator's PlacementValidator.check_transport_path -- the X
# correction afterwards is only a few millimeters). So any already-placed
# item sitting in the same X-columns, between the door and our candidate
# spot, and taller than our own landing height, would be hit during entry.
#
# Gliding along a surface that is level with (or lower than) our own
# landing height is completely normal -- most items rest flush on the bare
# floor, which is "in the way" of every deeper cell in exactly this sense,
# without being an obstruction at all. So the corridor only counts as
# blocked when something in it is genuinely *taller* than where we're
# landing; this small epsilon exists purely for floating point safety, not
# as a required physical standoff (that's already provided by the AABB
# padding in ContainerState._build_from_packed_items).
PATH_CLEARANCE = 0.005

# `flat` is the height difference between the highest and lowest cell in the
# footprint -- a large value means the item would only really touch down on
# a small sliver (e.g. the corner of a shorter neighbor) while the rest of
# its base hangs unsupported in the air. In practice that reliably tips the
# item over during the physics settle step, which -- exactly like a blocked
# entry path -- ends the whole episode (place_item failing marks the step
# terminated). So "reasonably flat support" is treated as a near-hard
# requirement, on the same footing as a clear entry path.
FLAT_TOLERANCE = 0.03

# Large, roughly-equal penalties for the three ways a candidate can plausibly
# get this whole episode terminated (blocked entry path, unsupported/tipping
# perch, forced sideways detour through the chamfered corner). The base score
# below is `top * 1000` (top is in meters, rarely above ~2), so this must be
# well past that scale to reliably dominate any plausible top/flat
# difference -- otherwise a low-but-blocked spot can out-score a slightly
# higher, genuinely clear one, which defeats the whole point of tracking
# these risks. Unlike a hard filter, it still returns *something* even when
# every option on the grid carries some risk.
RISK_PENALTY = 5000.0

# How strongly a deeper (further-from-the-door) landing spot is preferred
# over a shallower one at the same height, on the same top*1000 scale. Large
# enough to reroute around a modest stack (comparable to ~15cm of extra
# height) rather than immediately reverting to "closest to the door wins"
# the moment anything is already placed.
DEEP_BIAS_WEIGHT = 150.0

# Previously-placed items rarely settle perfectly flat (tiny tilts from the
# physics settle step are normal), so a target that assumes their recorded
# top height exactly is occasionally a millimeter or two optimistic. A small
# vertical buffer on every landing (not just the floor, which already gets
# one via ContainerState.floor_z) keeps that from turning into a graze.
LANDING_CLEARANCE = 0.025


def _path_block_height(state: ContainerState, fw: int, n_iy: int) -> np.ndarray:
    """For every (ix, iy) anchor, the tallest obstruction in the entry
    corridor (same X-columns, door side up to iy-1) that a new item would
    have to clear. -inf where iy == 0 (nothing between the door and here)."""
    cummax_y = np.maximum.accumulate(state.height_grid, axis=1)
    block_by_col = _sliding_max_axis0(cummax_y, fw)  # shape (n-fw+1, n)
    path_block = np.full((block_by_col.shape[0], n_iy), -np.inf)
    if n_iy > 1:
        path_block[:, 1:n_iy] = block_by_col[:, 0:n_iy - 1]
    return path_block


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
    landing_bottom = top + LANDING_CLEARANCE

    # The applicable ceiling can vary across the footprint (e.g. a shelf
    # covering only part of the container), so use whatever is most
    # restrictive within the window.
    window_ceiling = _windows(state.ceiling_grid, fw, fh).min(axis=(2, 3))
    fits_ceiling = (landing_bottom + item_height) <= (window_ceiling + 1e-6)
    if not fits_ceiling.any():
        return None

    path_block = _path_block_height(state, fw, top.shape[1])
    path_clear = path_block <= (landing_bottom + PATH_CLEARANCE)

    conflict = np.zeros_like(fits_ceiling, dtype=bool)
    if avoid_soft_top:
        conflict |= _windows(state.top_soft, fw, fh).any(axis=(2, 3))
    if avoid_priority_top:
        conflict |= _windows(state.top_prioritized, fw, fh).any(axis=(2, 3))

    # Require flatter support the higher up the landing is: a settling
    # discrepancy of a few mm on the item below is a bigger deal on top of
    # a tall, complex, multi-item stack (more torque, more accumulated
    # drift from what this heightmap assumes) than it is resting near the
    # floor.
    stable = flat <= (FLAT_TOLERANCE / (1.0 + top))
    in_corner_keepout = _windows(state.corner_keepout, fw, fh).any(axis=(2, 3))

    # `fits_ceiling` is a genuine hard constraint (there is no way to make an
    # over-height placement valid). Everything else is a strong-but-soft
    # penalty: we always want *a* candidate back, even if every option on
    # the grid carries some risk, since returning None drops the item from
    # consideration entirely.
    #
    # Deep-first bias: prefer larger iy (further from the door) over
    # smaller, at a weight big enough to actually compete with `top` (unlike
    # a mere tie-break). A single item colliding ends the whole episode and
    # torches every remaining item's contribution to the score, so it is
    # worth deliberately accepting a noticeably higher landing spot to keep
    # the door-side lane clear for later, deeper-targeted items -- this is
    # what makes fill-from-the-back-forward hold up beyond an empty
    # container, where `top` differences are still small enough for it to
    # matter, instead of collapsing back into fill-from-the-door the moment
    # any stacking is involved.
    # `flat` is weighted close to `top`'s own scale (not the ~10x-weaker
    # tie-break it used to be): once every candidate is at least somewhat
    # risky -- which RISK_PENALTY alone can't prevent, it's a flat additive
    # offset that cancels out between two already-unstable options -- this
    # is what actually decides "least bad" in favor of the smallest
    # unsupported gap (most likely to survive settling) rather than just the
    # lowest height.
    #
    # The same gap is also more dangerous the higher up it is (more torque,
    # further to fall, and a physics-settle drift of a few mm on the item
    # below is a larger fraction of a small footprint than a large one) --
    # and it's exactly at height that real settled positions drift furthest
    # from what this heightmap assumes. So weight `flat` more heavily as
    # `top` grows, to keep a real but close call from tipping toward the
    # riskier option under that drift.
    flat_weight = 500.0 * (1.0 + top)
    n_iy = max(top.shape[1] - 1, 1)
    ix_grid, iy_grid = np.meshgrid(np.arange(top.shape[0]), np.arange(top.shape[1]), indexing="ij")
    deep_bias = -(iy_grid / n_iy) * DEEP_BIAS_WEIGHT
    score = top * 1000.0 + flat * flat_weight + deep_bias + ix_grid * 1e-4
    score = score + (~path_clear) * RISK_PENALTY
    score = score + (~stable) * RISK_PENALTY
    score = score + in_corner_keepout * RISK_PENALTY
    score = score + conflict * 0.5
    score = np.where(fits_ceiling, score, np.inf)
    ix, iy = np.unravel_index(np.argmin(score), score.shape)

    x_center = state.x_min + (ix + fw / 2.0) * state.cell_w
    y_center = state.y_min + (iy + fh / 2.0) * state.cell_h
    z_center = float(landing_bottom[ix, iy]) + item_height / 2.0

    return {
        "x": x_center,
        "y": y_center,
        "z": z_center,
        "top": float(top[ix, iy]),
        "flat": float(flat[ix, iy]),
        "conflict": bool(conflict[ix, iy]),
        "path_blocked": not bool(path_clear[ix, iy]),
        "unstable": not bool(stable[ix, iy]),
        "in_corner_keepout": bool(in_corner_keepout[ix, iy]),
        "fw": fw,
        "fh": fh,
    }
