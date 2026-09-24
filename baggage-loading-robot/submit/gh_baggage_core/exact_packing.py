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
container's own origin), not the O(n^2) full cross product, keeping this
cheap enough for the online policy's per-step time budget.

Landing height and support are computed by exact overlap against
`item_aabbs` rather than a per-column grid maximum, so two items that
happen to differ by a millimeter no longer get rounded onto the same
"layer" (or split across two) purely by grid quantization.
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
    that box in the other axis -- lands stacked on top of it once
    `_landing_z` looks up the real overlap height; either way falls out of
    the same candidate list without needing separate "stack here" points.
    """
    anchors = {(state.x_min, state.y_min)}
    if state.cut_x > 0.0:
        # The container's own origin corner sits inside the chamfer's
        # keepout band (see best_position_exact's "no exact chamfer data"
        # fallback and _apply_cut_corner_keepout in container_state.py):
        # on an otherwise-empty container that's the *only* anchor
        # _candidate_anchors would otherwise offer, so without this the
        # search would wrongly come back empty rather than starting just
        # outside the band, exactly like the grid search's own first
        # available column does.
        anchors.add((state.x_min + state.cut_x, state.y_min))
    for lo, hi in state.item_aabbs:
        anchors.add((hi[0], lo[1]))
        anchors.add((lo[0], hi[1]))
    return list(anchors)


def _landing_z(state: ContainerState, x0: float, x1: float, y0: float, y1: float) -> float:
    """The height this footprint would actually settle at: the tallest top
    face among already-placed boxes whose XY footprint overlaps
    [x0, x1] x [y0, y1], or the bare floor if nothing does."""
    best = state.floor_z
    for lo, hi in state.item_aabbs:
        if hi[0] <= x0 or lo[0] >= x1 or hi[1] <= y0 or lo[1] >= y1:
            continue
        if hi[2] > best:
            best = hi[2]
    return best


def _path_block_height(state: ContainerState, x0: float, x1: float, y0: float) -> float:
    """The tallest obstruction between the door (y_min) and this footprint's
    own near edge (y0), in the same X-columns -- see PATH_CLEARANCE in
    packing.py. -inf if nothing is in the way."""
    best = -np.inf
    for lo, hi in state.item_aabbs:
        if hi[0] <= x0 or lo[0] >= x1:
            continue
        if lo[1] >= y0:
            continue  # at or past this footprint's own near edge, not "in the way"
        if hi[2] > best:
            best = hi[2]
    return best


def _support_fraction(state: ContainerState, x0: float, x1: float, y0: float, y1: float,
                       landing_z: float) -> float:
    """Fraction of this footprint's area actually resting on something at
    (near) `landing_z`, by exact overlap area rather than a grid-cell
    count -- see CONTACT_TOLERANCE in packing.py for why only faces within
    that tolerance of the landing height count as load-bearing."""
    area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    if area <= 0.0:
        return 0.0
    if landing_z <= state.floor_z + 1e-9:
        return 1.0  # resting on the bare floor: fully supported by definition
    covered = 0.0
    for lo, hi in state.item_aabbs:
        if hi[2] < landing_z - CONTACT_TOLERANCE or hi[2] > landing_z + CONTACT_TOLERANCE:
            continue
        ox = min(x1, hi[0]) - max(x0, lo[0])
        oy = min(y1, hi[1]) - max(y0, lo[1])
        if ox > 0.0 and oy > 0.0:
            covered += ox * oy
    return min(covered / area, 1.0)


def best_position_exact(state: ContainerState, footprint_x: float, footprint_y: float,
                         item_height: float, avoid_soft_top: bool, avoid_priority_top: bool,
                         require_safe: bool = False) -> dict | None:
    """Exact-geometry counterpart to `packing.best_position` -- same
    scoring philosophy (lowest safe landing height, with the same
    stability/aspect-ratio/keepout risk terms), same return contract, but
    searching real extreme-point candidates against `state.item_aabbs`
    instead of a discretized heightmap. Returns None if the footprint
    fits nowhere.
    """
    if footprint_x > (state.x_max - state.x_min) or footprint_y > (state.y_max - state.y_min):
        return None

    best_score = np.inf
    best = None
    for ax, ay in _candidate_anchors(state):
        x0 = min(max(ax, state.x_min), state.x_max - footprint_x)
        y0 = min(max(ay, state.y_min), state.y_max - footprint_y)
        if x0 < state.x_min - 1e-9 or y0 < state.y_min - 1e-9:
            continue  # footprint doesn't fit in the container at all
        x1, y1 = x0 + footprint_x, y0 + footprint_y
        x_center, y_center = x0 + footprint_x / 2.0, y0 + footprint_y / 2.0

        # `landing_z` is the true resting height (top of whatever's directly
        # underneath); `landing_bottom` adds the same settle-drift buffer
        # packing.py's grid search applies to every candidate (see
        # LANDING_CLEARANCE there) before checking ceiling/chamfer/clearance
        # or reporting where the item's own box actually sits.
        landing_z = _landing_z(state, x0, x1, y0, y1)
        landing_bottom = landing_z + LANDING_CLEARANCE

        effective_ceiling = state.ceiling_z
        if state.has_shelf and y1 > 0.0:
            effective_ceiling = min(effective_ceiling, max(state.height / 2.0 - 0.05, state.floor_z + 0.05))
        if landing_bottom + item_height > effective_ceiling:
            continue

        if state._chamfer_normal is not None:
            n = state._chamfer_normal
            px, py, pz = state._chamfer_point
            hx, hy, hz = footprint_x / 2.0, footprint_y / 2.0, item_height / 2.0
            z_center = landing_bottom + hz
            dot = (
                n[0] * (x_center - px) + n[1] * (y_center - py) + n[2] * (z_center - pz)
                + abs(n[0]) * hx + abs(n[1]) * hy + abs(n[2]) * hz
            )
            if dot > (CHAMFER_INCLUSION_MARGIN - CHAMFER_SAFETY_MARGIN):
                continue
        elif x0 < state.x_min + state.cut_x:
            continue  # no exact chamfer data: stay out of the cut band entirely

        m = EXACT_CHECK_SAFETY_MARGIN
        clear = True
        for lo, hi in state.item_aabbs:
            if (x1 + m <= lo[0] or hi[0] + m <= x0
                    or y1 + m <= lo[1] or hi[1] + m <= y0
                    or landing_bottom + item_height + m <= lo[2] or hi[2] + m <= landing_bottom):
                continue
            clear = False
            break
        if not clear:
            continue

        in_corner_keepout = x0 < state.x_min + state.cut_x

        # See PATH_CLEARANCE in packing.py: an item slides in from the door
        # along +Y at roughly constant X/Z, so anything tall enough sitting
        # between the door and this footprint's own near edge is a real
        # transport-path risk, not just a landing-spot one.
        path_block = _path_block_height(state, x0, x1, y0)
        path_clear = (path_block + PATH_CLEARANCE) <= landing_bottom

        support_fraction = _support_fraction(state, x0, x1, y0, y1, landing_z)
        cx0, cx1 = x0 + footprint_x * 0.25, x1 - footprint_x * 0.25
        cy0, cy1 = y0 + footprint_y * 0.25, y1 - footprint_y * 0.25
        core_support_fraction = _support_fraction(state, cx0, cx1, cy0, cy1, landing_z)

        required_support = np.clip(MIN_SUPPORT_FRACTION + landing_z * HEIGHT_SUPPORT_SCALE,
                                    MIN_SUPPORT_FRACTION, 0.95)
        required_core = np.clip(MIN_CORE_SUPPORT_FRACTION + landing_z * HEIGHT_SUPPORT_SCALE,
                                 MIN_CORE_SUPPORT_FRACTION, 0.95)
        aspect_ratio = item_height / max(min(footprint_x, footprint_y), 1e-6)
        required_aspect_ratio = np.clip(
            ASPECT_RATIO_BASE_LIMIT - landing_z * ASPECT_HEIGHT_SCALE, ASPECT_RATIO_MIN_LIMIT, ASPECT_RATIO_BASE_LIMIT,
        )
        stable = (
            support_fraction >= required_support and core_support_fraction >= required_core
            and aspect_ratio <= required_aspect_ratio
        )

        # `item_aabbs` doesn't carry per-box is_soft/is_prioritized flags
        # (only the heightmap's parallel `top_soft`/`top_prioritized` grids
        # do), so this samples those grids at the footprint's own cell range
        # instead of re-deriving box identity -- exact-vs-grid mismatches
        # only matter here as a soft scoring nudge (TOP_CONFLICT_PENALTY-
        # equivalent), never as a hard accept/reject, so grid resolution is
        # fine for it.
        conflict = False
        if landing_z > state.floor_z + 1e-9:
            ix0, ix1, iy0, iy1 = state._grid_index_range(x0, x1, y0, y1)
            region_soft = state.top_soft[ix0:ix1, iy0:iy1]
            region_prioritized = state.top_prioritized[ix0:ix1, iy0:iy1]
            if (avoid_soft_top and region_soft.any()) or (avoid_priority_top and region_prioritized.any()):
                conflict = True

        if require_safe and (not stable or in_corner_keepout or not path_clear):
            continue

        score = landing_z * 1000.0
        score += (1.0 - support_fraction) * 500.0 * (1.0 + landing_z)
        score += (0.0 if path_clear else _RISK_PENALTY)
        score += (0.0 if stable else _RISK_PENALTY)
        score += (_RISK_PENALTY if in_corner_keepout else 0.0)
        score += (0.5 if conflict else 0.0)
        score += x0 * 1e-4

        if score < best_score:
            best_score = score
            best = {
                "x": x_center, "y": y_center, "z": landing_bottom + item_height / 2.0,
                "top": landing_z, "flat": 0.0, "shadow_area": 0.0,
                "support_fraction": support_fraction, "core_support_fraction": core_support_fraction,
                "conflict": conflict, "path_blocked": not path_clear,
                "unstable": not stable, "in_corner_keepout": in_corner_keepout,
                "fw": 0, "fh": 0,
            }

    return best
