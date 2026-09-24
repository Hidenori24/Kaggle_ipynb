"""Exact-geometry extreme-point placement search.

`packing.py`'s search scores every cell of a discretized heightmap grid
(`ContainerState.grid_n = 40` cells across the container -- a few
centimeters per cell on a typical container). Every candidate position is
snapped to that grid, and a footprint's cell count is rounded *up*
(`math.ceil`), so a real item can lose several centimeters of usable space
in each direction to pure rounding -- multiplied across dozens of items,
this is a real and measurable share of the container's volume, not just
numerical noise. `ContainerState` already tracks the *exact* (unpadded)
AABB of every placed item in `item_aabbs`, and `packing.py` already has
exact clearance/chamfer checks (`_exact_aabb_clear`, chamfer plane math) --
it just only ever uses them as a last-resort retry on top of the grid's
coarse answer, not as the primary search.

This module runs that primary search directly against the exact geometry
instead, using the classic extreme-point method for 3D bin packing
(Crainic, Perboli & Tadei, 2008): after placing a box, its far corners (in
X and in Y) become new candidate anchor points for the *next* item, on the
reasoning that any efficient packing has some box flush against some
other box's (or the container's) corner. Candidates are generated in O(n)
from the n already-placed boxes (two candidates per box, plus the
container's own origin), not the O(n^2) full cross product.

Every candidate's landing height, support, clearance and path-block checks
are computed against `item_aabbs` for *all* candidate anchors at once, via
numpy broadcasting over an (anchors x items) matrix, rather than a Python
loop per anchor that itself loops over every item -- the same O(anchors *
items) amount of work as a naive nested loop, but done as a handful of
vectorized array ops instead of Python-level iteration, which is the
difference between single-digit milliseconds and multiple seconds once a
container holds a couple hundred items (measured: a pure Python-loop
version took ~9s to rank a single step's pool against 250 placed items --
well past the online policy's 5.5s time budget).
"""
from __future__ import annotations

import numpy as np

from .container_state import ContainerState
from .packing import (
    ASPECT_HEIGHT_SCALE, ASPECT_RATIO_BASE_LIMIT, ASPECT_RATIO_MIN_LIMIT,
    CHAMFER_INCLUSION_MARGIN, CHAMFER_SAFETY_MARGIN, CONTACT_TOLERANCE,
    EXACT_CHECK_SAFETY_MARGIN, HEIGHT_SUPPORT_SCALE, LANDING_CLEARANCE,
    MIN_CORE_SUPPORT_FRACTION, MIN_SUPPORT_FRACTION, PATH_CLEARANCE,
)

# Same shape as packing.py's RISK_PENALTY family: large enough to always
# lose to any genuinely clear candidate, small enough that two comparably
# risky candidates still get ranked by their other merits (height, support)
# rather than being ties at infinity.
_RISK_PENALTY = 5000.0


def _candidate_anchors(state: ContainerState) -> list[tuple[float, float]]:
    """Extreme-point candidates for an item's (x_min, y_min) corner: the
    container's own origin, plus each existing box's far-X and far-Y
    corners projected onto the floor plane (see module docstring). A box
    resting flush against one of these, in XY, either sits beside the
    generating box at floor level or -- if its footprint still overlaps
    that box in the other axis -- lands stacked on top of it once the
    landing-height lookup below finds the real overlap height; either way
    falls out of the same candidate list without needing separate
    "stack here" points.
    """
    anchors = {(state.x_min, state.y_min)}
    if state.cut_x > 0.0:
        # The container's own origin corner sits inside the chamfer's
        # keepout band (see best_position_exact's "no exact chamfer data"
        # fallback and _apply_cut_corner_keepout in container_state.py):
        # on an otherwise-empty container that's the *only* anchor this
        # would otherwise offer, so without this the search would wrongly
        # come back empty rather than starting just outside the band,
        # exactly like the grid search's own first available column does.
        anchors.add((state.x_min + state.cut_x, state.y_min))
    for lo, hi in state.item_aabbs:
        anchors.add((hi[0], lo[1]))
        anchors.add((lo[0], hi[1]))
    return list(anchors)


def best_position_exact(state: ContainerState, footprint_x: float, footprint_y: float,
                         item_height: float, avoid_soft_top: bool, avoid_priority_top: bool,
                         require_safe: bool = False) -> dict | None:
    """Exact-geometry counterpart to `packing.best_position` -- same
    scoring philosophy (lowest safe landing height, with the same
    stability/aspect-ratio/keepout/path-clearance risk terms), same return
    contract, but searching real extreme-point candidates against
    `state.item_aabbs` instead of a discretized heightmap. Returns None if
    the footprint fits nowhere.
    """
    if footprint_x > (state.x_max - state.x_min) or footprint_y > (state.y_max - state.y_min):
        return None

    anchors = _candidate_anchors(state)
    A = np.asarray(anchors, dtype=np.float64)
    x0 = np.clip(A[:, 0], state.x_min, state.x_max - footprint_x)
    y0 = np.clip(A[:, 1], state.y_min, state.y_max - footprint_y)
    valid = (x0 >= state.x_min - 1e-9) & (y0 >= state.y_min - 1e-9)
    x1, y1 = x0 + footprint_x, y0 + footprint_y
    x_center, y_center = x0 + footprint_x / 2.0, y0 + footprint_y / 2.0

    n_items = len(state.item_aabbs)
    if n_items:
        los = np.asarray([item[0] for item in state.item_aabbs], dtype=np.float64)
        his = np.asarray([item[1] for item in state.item_aabbs], dtype=np.float64)
        # Footprint-vs-item XY overlap, for every (anchor, item) pair at
        # once -- reused below for landing height, path-block height and
        # support area, so it's computed exactly once.
        overlap_x = (his[None, :, 0] > x0[:, None]) & (los[None, :, 0] < x1[:, None])
        overlap_y = (his[None, :, 1] > y0[:, None]) & (los[None, :, 1] < y1[:, None])
        overlap = overlap_x & overlap_y

        landing_z = np.where(overlap, his[None, :, 2], -np.inf).max(axis=1)
        landing_z = np.maximum(landing_z, state.floor_z)

        # Path-block: same X-column overlap as `overlap_x`, but only items
        # that sit strictly before this footprint's own near (door-side)
        # edge -- see PATH_CLEARANCE in packing.py.
        path_overlap = overlap_x & (los[None, :, 1] < y0[:, None])
        path_block = np.where(path_overlap, his[None, :, 2], -np.inf).max(axis=1)
    else:
        landing_z = np.full(len(anchors), state.floor_z)
        path_block = np.full(len(anchors), -np.inf)

    landing_bottom = landing_z + LANDING_CLEARANCE

    effective_ceiling = np.full(len(anchors), state.ceiling_z)
    if state.has_shelf:
        shelf_ceiling = max(state.height / 2.0 - 0.05, state.floor_z + 0.05)
        shelf_mask = y1 > 0.0
        effective_ceiling[shelf_mask] = np.minimum(effective_ceiling[shelf_mask], shelf_ceiling)
    fits_ceiling = (landing_bottom + item_height) <= effective_ceiling

    if state._chamfer_normal is not None:
        n = state._chamfer_normal
        px, py, pz = state._chamfer_point
        hx, hy, hz = footprint_x / 2.0, footprint_y / 2.0, item_height / 2.0
        z_center = landing_bottom + hz
        dot = (
            n[0] * (x_center - px) + n[1] * (y_center - py) + n[2] * (z_center - pz)
            + abs(n[0]) * hx + abs(n[1]) * hy + abs(n[2]) * hz
        )
        chamfer_ok = dot <= (CHAMFER_INCLUSION_MARGIN - CHAMFER_SAFETY_MARGIN)
    else:
        # No exact chamfer data: stay out of the cut band entirely.
        chamfer_ok = x0 >= state.x_min + state.cut_x

    in_corner_keepout = x0 < state.x_min + state.cut_x

    if n_items:
        m = EXACT_CHECK_SAFETY_MARGIN
        separated = (
            (x1[:, None] + m <= los[None, :, 0]) | (his[None, :, 0] + m <= x0[:, None])
            | (y1[:, None] + m <= los[None, :, 1]) | (his[None, :, 1] + m <= y0[:, None])
            | (landing_bottom[:, None] + item_height + m <= los[None, :, 2])
            | (his[None, :, 2] + m <= landing_bottom[:, None])
        )
        clear = separated.all(axis=1)

        path_clear = (path_block + PATH_CLEARANCE) <= landing_bottom

        on_floor = landing_z <= state.floor_z + 1e-9
        within_tol = (
            (his[None, :, 2] >= landing_z[:, None] - CONTACT_TOLERANCE)
            & (his[None, :, 2] <= landing_z[:, None] + CONTACT_TOLERANCE)
        )

        def _support(bx0, bx1, by0, by1):
            ox = np.minimum(bx1[:, None], his[None, :, 0]) - np.maximum(bx0[:, None], los[None, :, 0])
            oy = np.minimum(by1[:, None], his[None, :, 1]) - np.maximum(by0[:, None], los[None, :, 1])
            cell_area = np.clip(ox, 0.0, None) * np.clip(oy, 0.0, None)
            covered = np.where(within_tol, cell_area, 0.0).sum(axis=1)
            full_area = (bx1 - bx0) * (by1 - by0)
            frac = np.divide(covered, full_area, out=np.ones_like(covered), where=full_area > 1e-12)
            frac = np.clip(frac, 0.0, 1.0)
            return np.where(on_floor, 1.0, frac)

        support_fraction = _support(x0, x1, y0, y1)
        cx0, cx1 = x0 + footprint_x * 0.25, x1 - footprint_x * 0.25
        cy0, cy1 = y0 + footprint_y * 0.25, y1 - footprint_y * 0.25
        core_support_fraction = _support(cx0, cx1, cy0, cy1)
    else:
        clear = np.ones(len(anchors), dtype=bool)
        path_clear = np.ones(len(anchors), dtype=bool)
        support_fraction = np.ones(len(anchors))
        core_support_fraction = np.ones(len(anchors))

    required_support = np.clip(MIN_SUPPORT_FRACTION + landing_z * HEIGHT_SUPPORT_SCALE,
                                MIN_SUPPORT_FRACTION, 0.95)
    required_core = np.clip(MIN_CORE_SUPPORT_FRACTION + landing_z * HEIGHT_SUPPORT_SCALE,
                             MIN_CORE_SUPPORT_FRACTION, 0.95)
    aspect_ratio = item_height / max(min(footprint_x, footprint_y), 1e-6)
    required_aspect_ratio = np.clip(
        ASPECT_RATIO_BASE_LIMIT - landing_z * ASPECT_HEIGHT_SCALE, ASPECT_RATIO_MIN_LIMIT, ASPECT_RATIO_BASE_LIMIT,
    )
    stable = (
        (support_fraction >= required_support) & (core_support_fraction >= required_core)
        & (aspect_ratio <= required_aspect_ratio)
    )

    # `item_aabbs` doesn't carry per-box is_soft/is_prioritized flags (only
    # the heightmap's parallel `top_soft`/`top_prioritized` grids do), so
    # this samples those grids at each footprint's own cell range instead
    # of re-deriving box identity -- exact-vs-grid mismatches only matter
    # here as a soft scoring nudge (TOP_CONFLICT_PENALTY-equivalent), never
    # as a hard accept/reject, so grid resolution is fine for it. Kept as a
    # small per-anchor loop: each grid-window lookup is O(1)-ish (a few
    # cells), not O(items), so it doesn't reintroduce the quadratic cost
    # the rest of this function avoids.
    conflict = np.zeros(len(anchors), dtype=bool)
    if avoid_soft_top or avoid_priority_top:
        for i in range(len(anchors)):
            if landing_z[i] <= state.floor_z + 1e-9:
                continue
            ix0, ix1, iy0, iy1 = state._grid_index_range(x0[i], x1[i], y0[i], y1[i])
            region_soft = state.top_soft[ix0:ix1, iy0:iy1]
            region_prioritized = state.top_prioritized[ix0:ix1, iy0:iy1]
            if (avoid_soft_top and region_soft.any()) or (avoid_priority_top and region_prioritized.any()):
                conflict[i] = True

    mask = valid & fits_ceiling & chamfer_ok & clear
    if require_safe:
        mask = mask & stable & ~in_corner_keepout & path_clear

    score = landing_z * 1000.0
    score = score + (1.0 - support_fraction) * 500.0 * (1.0 + landing_z)
    score = score + np.where(path_clear, 0.0, _RISK_PENALTY)
    score = score + np.where(stable, 0.0, _RISK_PENALTY)
    score = score + np.where(in_corner_keepout, _RISK_PENALTY, 0.0)
    score = score + np.where(conflict, 0.5, 0.0)
    score = score + x0 * 1e-4
    score = np.where(mask, score, np.inf)

    if not np.isfinite(score).any():
        return None
    i = int(np.argmin(score))

    return {
        "x": float(x_center[i]), "y": float(y_center[i]), "z": float(landing_bottom[i] + item_height / 2.0),
        "top": float(landing_z[i]), "flat": 0.0, "shadow_area": 0.0,
        "support_fraction": float(support_fraction[i]), "core_support_fraction": float(core_support_fraction[i]),
        "conflict": bool(conflict[i]), "path_blocked": not bool(path_clear[i]),
        "unstable": not bool(stable[i]), "in_corner_keepout": bool(in_corner_keepout[i]),
        "fw": 0, "fh": 0,
    }
