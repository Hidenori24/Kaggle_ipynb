"""Vectorized sliding-window search for the best (x, y) landing spot for a
given footprint on a container's heightmap, using a heightmap/skyline
"place lowest & flattest" heuristic (a discretized variant of the classic
extreme-point / deepest-bottom-left-fill family of 3D bin-packing
constructors) gated by a coarse static-equilibrium stability check (see
the comment on CONTACT_TOLERANCE / MIN_SUPPORT_FRACTION below) rather than
a bare height-range threshold.
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
# spot, and about as tall as (or taller than) our own landing height, would
# be grazed or hit during entry.
#
# Gliding along a surface that is level with (or lower than) our own
# landing height is completely normal -- most items rest flush on the bare
# floor, which is "in the way" of every deeper cell in exactly this sense,
# without being an obstruction at all. But "in the way" has to leave real
# clearance, not just avoid literal overlap: the validator's own transport
# check (`_move_item`) uses `getClosestPoints(..., distance=safety_margin)`,
# which flags *any* approach within that margin, not only a true collision
# (confirmed against the real simulator -- a reproduced failure had a
# genuine positive separation of 1.28cm, still inside the evaluation
# config's 1.5cm safety_margin, and still ended the episode). The AABB
# padding in ContainerState._build_from_packed_items only widens an
# obstruction's X/Y footprint, not the *height* recorded for it, so it
# provides no such margin here -- this has to enforce it directly: the
# corridor only counts as clear once the tallest obstruction sits at least
# PATH_CLEARANCE below our own landing height, not merely level with or a
# hair above it.
PATH_CLEARANCE = 0.015

# `flat` (the height difference between the highest and lowest cell in the
# footprint) is kept as a secondary scoring signal below, but the actual
# go/no-go stability call is a coarse static-equilibrium check instead: a
# rigid box settles flush onto whatever is tallest within its footprint, so
# only the cells within CONTACT_TOLERANCE of that top height actually bear
# any load -- everywhere else is a gap the box bridges over. Two boxes can
# have the same `flat` (max-min height range) with very different amounts
# of *actual* contact area, and `flat` alone says nothing about *where*
# that contact is relative to the box's own center of mass (dead center,
# always, for a uniform box) -- a small contact patch tucked in one corner
# tips even if the range itself looks tame. So stability requires both:
#   - enough of the footprint is actually in contact (SUPPORT_FRACTION), and
#   - the box's own center isn't hanging over a gap (CORE_SUPPORT_FRACTION,
#     checked over the middle half of the footprint specifically).
# Both requirements are tightened with height for the same reason `flat`
# used to be: more torque, and more accumulated drift between this
# heightmap and the real settled geometry, the higher up the landing is.
CONTACT_TOLERANCE = 0.02
MIN_SUPPORT_FRACTION = 0.6
MIN_CORE_SUPPORT_FRACTION = 0.5
HEIGHT_SUPPORT_SCALE = 0.25

# Support-fraction/core-support-fraction can both read a perfect 1.0 --
# resting flush on a fully flat, fully-covered surface -- and the box can
# *still* tip in the real settle step (confirmed against the real
# simulator: two observed failures with support_fraction == core ==
# 1.0 and real displacement/angle far past the validator's threshold, one
# of them a near-total flip). Perfect footprint contact says nothing about
# whether the box's own center of mass stays over that footprint once it's
# perturbed -- a box far taller than its own base tips over a flat floor
# just as easily as over a gap, the same way a pencil balanced on its tip
# doesn't need an uneven table to fall. This is an orthogonal risk from
# everything above (which only ever looks at the *landing surface*, never
# the item's own shape), so it needs its own check rather than folding into
# support_fraction's own threshold -- raising that threshold can never flag
# a candidate that's already sitting at the 1.0 ceiling.
#
# `aspect_ratio` is the box's own height divided by its narrower footprint
# dimension in *this* orientation -- the ratio that determines how far the
# item's center of mass can shift laterally (roughly proportional to base
# width) before it moves outside the base (roughly proportional to height)
# and gravity takes over, i.e. real tip-over physics, not a heuristic
# stand-in for it. A cube (ratio 1) is included as still "safe"; the limit
# tightens with landing height for the same reason the support thresholds
# above do (settle-step perturbation and heightmap/reality drift both grow
# with height) down to ASPECT_RATIO_MIN_LIMIT. Since a box has 3 candidate
# orientations for "which face is down" and only the tallest one or two
# tend to violate this, the practical effect is steering the search toward
# laying a tall item on its side when the floor space for that is
# available, rather than standing it upright out of sheer habit (nothing
# before this ever compared orientations on stability grounds at all --
# only on the resulting landing height and footprint area).
ASPECT_RATIO_BASE_LIMIT = 1.5
ASPECT_RATIO_MIN_LIMIT = 1.0
ASPECT_HEIGHT_SCALE = 0.3

# Three real-simulator failures (see DESIGN.md) all had support_fraction
# == core_support_fraction == 1.0 and flat == 0.0 -- a heightmap-perfect,
# fully flat, fully covered landing -- yet still displaced far past the
# validator's threshold in the real settle step. Inspecting the real
# packed items behind each failure (not just the heightmap) showed why:
# the "flat" surface was a patchwork of *multiple separate* items that
# happened to share the exact same recorded top height (routine when
# several identical items land at the same layer), not one single rigid
# support. `height_grid` only ever stores "how tall," never "whose," so
# nothing above can tell a single pedestal from a seam between two
# independently-settled ones -- each contributing item has its own small
# settle tolerance, so a box bridging the seam is resting on two things
# that can each move a little, not one that can't, the same kind of
# heightmap blind spot ASPECT_RATIO_BASE_LIMIT exists to catch for the
# item's own shape, just on the support side. A blanket fix already tried
# for this same trio (scaling LANDING_CLEARANCE with height) regressed
# broadly across the benchmark suite (see DESIGN.md) by adding pointless
# buffer under perfectly fine single-item stacks too -- this instead only
# fires for the specific shape those failures had in common: bridging a
# real seam.
#
# SUPPORT_SEAM_HEIGHT_MIN gates it off near the floor, where several items
# abutting side by side is completely normal, not a stacking risk -- but
# it turns out to matter well beyond that: at the failures' own height
# range (0.3), this flagged far more seams than it saved, regressing the
# 8-scenario benchmark suite broadly (mean fill_score 13.40 -> 12.23,
# worst single case -6.18) -- most seams a couple of layers up are
# perfectly fine, and treating them all as a hard risk crowded the search
# into worse compromises elsewhere far more often than it prevented a real
# failure. Raised to 0.7 (in from the 0.75-1.1m the three anecdotes
# actually failed at, rather than tuned to fit them exactly) restores a
# clean non-regression (13.40 -> 13.45, zero regressions, one real gain)
# while presumably keeping less of the protection those anecdotes
# motivated this for -- deliberately conservative given how easily the
# lower threshold went net-negative, over chasing the original 3 cases
# exactly.
SUPPORT_SEAM_HEIGHT_MIN = 0.7
SUPPORT_SEAM_TOLERANCE = CONTACT_TOLERANCE

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

# Weight on `item_height * shadow_area` (see `_shadowed_clear_floor`): how
# strongly a placement is discouraged from permanently sealing off clear
# floor behind it, in its own X-lane, for anything shorter that might need
# it later. Started at DEEP_BIAS_WEIGHT's own order of magnitude (150), but
# against the real simulator that was too weak to move a tall/narrow item
# off an otherwise-cheaper shallow spot -- `top`'s own differences between
# candidates routinely swamp a 150-scale nudge. Raised until it actually
# changed the real outcome on a reproduced failure case (see DESIGN.md):
# a tall stack landing shallow, sealing a lane a later, shorter item needed.
SHADOW_BLOCK_WEIGHT = 600.0

# How high a cell can already be stacked and still count as "clear" for
# `_shadowed_clear_floor` -- deliberately more than a hair above the floor
# (not just literally-untouched cells), since a cell one modest layer deep
# is still perfectly usable by another short item, and multi-layer lanes
# are exactly where sealing off deeper space by mistake tends to happen
# (a tall stack landing shallow in a lane that already has a layer or two
# behind it).
SHADOW_LOW_THRESHOLD = 0.3

# Previously-placed items rarely settle perfectly flat (tiny tilts from the
# physics settle step are normal), so a target that assumes their recorded
# top height exactly is occasionally a millimeter or two optimistic. A small
# vertical buffer on every landing (not just the floor, which already gets
# one via ContainerState.floor_z) keeps that from turning into a graze.
LANDING_CLEARANCE = 0.025

# The validator's own `check_inclusion` requires, for every container-wall
# plane, dot = n.(center - point) + |n|.half_extents <= inclusion_margin
# (-0.005 m in the evaluation config). Every wall except the chamfer is
# already handled exactly by ContainerState's scalar x/y/z bounds (their
# SAFETY_MARGIN-padded cell-edge windows are exact regardless of item size).
# Only the chamfer's diagonal plane needs this direct check, using the same
# formula -- plus the same SAFETY_MARGIN the other walls use, since we're
# working from a coarse heightmap grid rather than continuous coordinates.
CHAMFER_INCLUSION_MARGIN = -0.005
CHAMFER_SAFETY_MARGIN = 0.012

# Defense-in-depth on top of the heightmap/padding model above: that model
# collapses every item's true 3D shape into a single per-column max height
# at this grid's own (coarse) resolution, which can occasionally let a
# candidate through that's actually closer to a real item than the
# validator's own safety_margin allows (a gap that looks clear at grid
# resolution can still be a real-world graze). `_exact_aabb_clear` checks
# the chosen candidate's real (unpadded) box against every already-placed
# item's real (unpadded) box directly, using this same real-world margin,
# and rejects/retries if it's actually too close. This can only ever make a
# candidate *more* conservative, never less, so it can't invalidate any of
# the heightmap-based scoring -- it's a last-resort correction, not a
# replacement for it.
EXACT_CHECK_SAFETY_MARGIN = 0.015
EXACT_CHECK_MAX_RETRIES = 20


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


def _shadowed_clear_floor(state: ContainerState, fw: int, fh: int, n0: int, n1: int) -> np.ndarray:
    """For every (ix, iy) anchor (shape matches `top`: (n0, n1)), how much
    still-clear floor area, in the *same X-columns*, sits strictly deeper
    (larger Y) than this window -- i.e. floor space this placement would
    permanently wall off. `_path_block_height` is this same corridor looked
    at from the door's side (what blocks *this* item on the way in); this
    is the mirror view (what *this* item, once tall and sitting shallow,
    would in turn block for anything arriving later needing the same lane).

    A single X-column can only ever be entered from the door (see
    `check_transport_path`: items always spawn at the door and travel
    straight up the Y axis at roughly a fixed X) -- so once something taller
    than a future item occupies a shallow Y position in that column, every
    deeper cell behind it in the same column is unusable by anything short
    enough to need this floor, for the rest of the episode. That's a much
    longer-lived cost than merely "risky right now" (`in_corner_keepout`,
    `path_blocked` etc.), so it's scored as its own term rather than folded
    into those.

    "Still usable" isn't limited to literally-untouched floor: a cell that
    already carries one modest layer is just as capable of taking another
    short item on top as bare floor is, so `SHADOW_LOW_THRESHOLD` treats
    anything below that as fair game rather than only exact-floor-height
    cells -- otherwise this would only ever catch the very first layer of
    stacking and miss multi-layer lanes exactly like the one that produced
    the real collision this was written to address (a tall stack landing
    shallow, sealing off a deeper lane at a *second* layer's height).
    """
    n = state.grid_n
    still_clear = state.height_grid <= (state.floor_z + SHADOW_LOW_THRESHOLD)
    # suffix_count[x, y] = number of still-clear cells at row >= y in column x.
    suffix_count = np.cumsum(still_clear[:, ::-1], axis=1).astype(np.float64)[:, ::-1]
    padded = np.concatenate([suffix_count, np.zeros((n, 1))], axis=1)  # index n -> 0

    # Row strictly behind window-row `iy` (0-based anchor of a size-fh
    # window) is `iy + fh`.
    behind_rows = np.arange(n1) + fh  # shape (n1,), values in [fh, n]
    behind_by_col = padded[:, behind_rows]  # shape (n, n1)

    # Sum over the fw columns this footprint actually occupies.
    col_cumsum = np.concatenate([np.zeros((1, n1)), np.cumsum(behind_by_col, axis=0)], axis=0)
    return col_cumsum[fw:fw + n0] - col_cumsum[0:n0]  # shape (n0, n1)


def _support_seam_count(state: ContainerState, fw: int, fh: int, top: np.ndarray) -> np.ndarray:
    """For every (ix, iy) window (shape matches `top`), how many distinct
    already-placed items' real top faces sit within SUPPORT_SEAM_TOLERANCE
    of that window's own landing height `top` and overlap its XY footprint
    -- see SUPPORT_SEAM_HEIGHT_MIN above. 0 or 1 means resting on the bare
    floor or a single item; >= 2 means the box would bridge a seam between
    separately-settled items that only coincidentally share a height."""
    n0, n1 = top.shape
    if not state.item_aabbs:
        return np.zeros((n0, n1), dtype=np.int64)
    los = np.asarray([item[0] for item in state.item_aabbs])
    his = np.asarray([item[1] for item in state.item_aabbs])
    x_center = state.x_min + (np.arange(n0) + fw / 2.0) * state.cell_w
    y_center = state.y_min + (np.arange(n1) + fh / 2.0) * state.cell_h
    hx = fw * state.cell_w / 2.0
    hy = fh * state.cell_h / 2.0

    ax0 = (x_center - hx)[:, None, None]
    ax1 = (x_center + hx)[:, None, None]
    ay0 = (y_center - hy)[None, :, None]
    ay1 = (y_center + hy)[None, :, None]
    top3 = top[:, :, None]

    overlap_x = (his[None, None, :, 0] > ax0) & (los[None, None, :, 0] < ax1)
    overlap_y = (his[None, None, :, 1] > ay0) & (los[None, None, :, 1] < ay1)
    at_height = (
        (his[None, None, :, 2] >= top3 - SUPPORT_SEAM_TOLERANCE)
        & (his[None, None, :, 2] <= top3 + SUPPORT_SEAM_TOLERANCE)
    )
    contributing = overlap_x & overlap_y & at_height
    return contributing.sum(axis=2)


def _exact_aabb_clear(state: ContainerState, x_center: float, y_center: float,
                       footprint_x: float, footprint_y: float, z0: float, z1: float) -> bool:
    """True iff the box [x_center +/- footprint_x/2, y_center +/- footprint_y/2,
    z0..z1] keeps at least EXACT_CHECK_SAFETY_MARGIN clear of every already
    -placed item's real (unpadded) box in `state.item_aabbs`, using a
    standard separating-axis test (axis-aligned boxes: a clear gap along
    any single axis is a sound, if not tight, proof of real 3D separation)."""
    if not state.item_aabbs:
        return True
    hx, hy = footprint_x / 2.0, footprint_y / 2.0
    ax0, ax1 = x_center - hx, x_center + hx
    ay0, ay1 = y_center - hy, y_center + hy
    los = np.array([item[0] for item in state.item_aabbs])
    his = np.array([item[1] for item in state.item_aabbs])
    m = EXACT_CHECK_SAFETY_MARGIN
    separated = (
        (ax1 + m <= los[:, 0]) | (his[:, 0] + m <= ax0)
        | (ay1 + m <= los[:, 1]) | (his[:, 1] + m <= ay0)
        | (z1 + m <= los[:, 2]) | (his[:, 2] + m <= z0)
    )
    return bool(separated.all())


def _chamfer_fits(state: ContainerState, n0: int, n1: int, fw: int, fh: int,
                   z_center: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    """Exact per-window inclusion check against the chamfer plane, replacing
    the coarse "whole band blocked" approximation `ContainerState` used to
    apply directly to the heightmap. Uses the same half-space formula the
    simulator's own validator does (see the CHAMFER_* constants above):
    a box is on the legal side of the plane iff the *farthest* corner of its
    AABB, projected onto the plane's outward normal, doesn't cross it.
    """
    n = state._chamfer_normal
    px, py, pz = state._chamfer_point
    x_center = state.x_min + (np.arange(n0) + fw / 2.0) * state.cell_w
    y_center = state.y_min + (np.arange(n1) + fh / 2.0) * state.cell_h
    dot = (
        n[0] * (x_center[:, None] - px)
        + n[1] * (y_center[None, :] - py)
        + n[2] * (z_center - pz)
        + abs(n[0]) * hx + abs(n[1]) * hy + abs(n[2]) * hz
    )
    return dot <= (CHAMFER_INCLUSION_MARGIN - CHAMFER_SAFETY_MARGIN)


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

    if state._chamfer_normal is not None:
        fits_ceiling &= _chamfer_fits(
            state, top.shape[0], top.shape[1], fw, fh,
            landing_bottom + item_height / 2.0, footprint_x / 2.0, footprint_y / 2.0, item_height / 2.0,
        )

    if not fits_ceiling.any():
        return None

    path_block = _path_block_height(state, fw, top.shape[1])
    path_clear = (path_block + PATH_CLEARANCE) <= landing_bottom

    conflict = np.zeros_like(fits_ceiling, dtype=bool)
    if avoid_soft_top:
        conflict |= _windows(state.top_soft, fw, fh).any(axis=(2, 3))
    if avoid_priority_top:
        conflict |= _windows(state.top_prioritized, fw, fh).any(axis=(2, 3))

    # Static-equilibrium-style stability check (see the comment on the
    # constants above): a cell only counts as load-bearing if it's within
    # CONTACT_TOLERANCE of the resting height `top`, and we require both
    # enough total contact area and that the box's own center isn't
    # hanging over a gap.
    contact_mask = top_windows >= (top[:, :, None, None] - CONTACT_TOLERANCE)
    support_fraction = contact_mask.mean(axis=(2, 3))

    core_x0 = fw // 4
    core_x1 = core_x0 + max(1, fw - 2 * core_x0)
    core_y0 = fh // 4
    core_y1 = core_y0 + max(1, fh - 2 * core_y0)
    core_support_fraction = contact_mask[:, :, core_x0:core_x1, core_y0:core_y1].mean(axis=(2, 3))

    required_support = np.clip(MIN_SUPPORT_FRACTION + top * HEIGHT_SUPPORT_SCALE, MIN_SUPPORT_FRACTION, 0.95)
    required_core = np.clip(MIN_CORE_SUPPORT_FRACTION + top * HEIGHT_SUPPORT_SCALE, MIN_CORE_SUPPORT_FRACTION, 0.95)

    # See ASPECT_RATIO_BASE_LIMIT above: a tip-over risk that's independent
    # of how well-supported the footprint is, so it's checked separately
    # rather than folded into the support-fraction thresholds (which a
    # perfect 1.0 support_fraction would always clear regardless). This is
    # the same scalar for the whole grid -- footprint/height are fixed for
    # this call -- so it gates `stable` uniformly rather than varying by
    # landing spot.
    aspect_ratio = item_height / max(min(footprint_x, footprint_y), 1e-6)
    required_aspect_ratio = np.clip(
        ASPECT_RATIO_BASE_LIMIT - top * ASPECT_HEIGHT_SCALE, ASPECT_RATIO_MIN_LIMIT, ASPECT_RATIO_BASE_LIMIT,
    )
    aspect_ok = aspect_ratio <= required_aspect_ratio

    # See SUPPORT_SEAM_HEIGHT_MIN above: a heightmap-perfect landing can
    # still bridge a seam between two separately-settled items that happen
    # to share a height. Only checked high enough up that several items
    # merely abutting on the floor doesn't count.
    seam_risk = np.zeros(top.shape, dtype=bool)
    if (top >= SUPPORT_SEAM_HEIGHT_MIN).any():
        seam_count = _support_seam_count(state, fw, fh, top)
        seam_risk = (top >= SUPPORT_SEAM_HEIGHT_MIN) & (seam_count >= 2)

    stable = (
        (support_fraction >= required_support) & (core_support_fraction >= required_core)
        & aspect_ok & ~seam_risk
    )
    in_corner_keepout = _windows(state.corner_keepout, fw, fh).any(axis=(2, 3))

    # How much still-clear floor, in this item's own X-columns, would this
    # placement permanently wall off (see `_shadowed_clear_floor`): every
    # X-column is only ever entered from the door, so a tall item sitting
    # shallow (small Y) makes everything behind it in that column
    # unreachable by anything shorter for the rest of the episode, even
    # though nothing about *this* placement looks risky on its own (it's
    # not a `path_blocked`/`unstable` call -- it's a future one).
    shadow_area = _shadowed_clear_floor(state, fw, fh, top.shape[0], top.shape[1]) * (state.cell_w * state.cell_h)

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
    score = score + item_height * shadow_area * SHADOW_BLOCK_WEIGHT
    score = score + (~path_clear) * RISK_PENALTY
    score = score + (~stable) * RISK_PENALTY
    score = score + in_corner_keepout * RISK_PENALTY
    score = score + conflict * 0.5
    score = np.where(fits_ceiling, score, np.inf)
    ix, iy = np.unravel_index(np.argmin(score), score.shape)

    # See EXACT_CHECK_SAFETY_MARGIN above: verify the top candidate against
    # every real item box directly, and reject/retry (never accept without
    # checking) if the heightmap approximation let through something
    # actually too close.
    for _ in range(EXACT_CHECK_MAX_RETRIES):
        if not np.isfinite(score[ix, iy]):
            return None
        x_center = state.x_min + (ix + fw / 2.0) * state.cell_w
        y_center = state.y_min + (iy + fh / 2.0) * state.cell_h
        z_bottom = float(landing_bottom[ix, iy])
        if _exact_aabb_clear(state, x_center, y_center, footprint_x, footprint_y, z_bottom, z_bottom + item_height):
            break
        score[ix, iy] = np.inf
        ix, iy = np.unravel_index(np.argmin(score), score.shape)
    else:
        return None

    z_center = z_bottom + item_height / 2.0

    return {
        "x": x_center,
        "y": y_center,
        "z": z_center,
        "top": float(top[ix, iy]),
        "flat": float(flat[ix, iy]),
        "shadow_area": float(shadow_area[ix, iy]),
        "support_fraction": float(support_fraction[ix, iy]),
        "core_support_fraction": float(core_support_fraction[ix, iy]),
        "conflict": bool(conflict[ix, iy]),
        "path_blocked": not bool(path_clear[ix, iy]),
        "unstable": not bool(stable[ix, iy]),
        "in_corner_keepout": bool(in_corner_keepout[ix, iy]),
        "fw": fw,
        "fh": fh,
    }


# How many of the lowest-landing candidates `best_effort_position` will try
# against the exact box check before giving up and returning its single
# best (lowest) guess unverified. Mirrors EXACT_CHECK_MAX_RETRIES's role in
# `best_position`, just sized for a colder path: this only ever runs once
# `best_position` has already failed for every orientation in every
# container (see its docstring), so it can afford to look harder than the
# 20 retries the hot path budgets for.
BEST_EFFORT_MAX_CANDIDATES = 200


def best_effort_position(state: ContainerState, footprint_x: float, footprint_y: float,
                          item_height: float) -> dict:
    """Last-resort landing spot for when `best_position` finds nowhere
    valid *anywhere* -- every orientation, every container (see
    `rank_placements`'s "Returns [] if nothing fits anywhere for anyone").
    That only happens when the tallest obstruction under every possible
    footprint window already leaves no room under that window's own
    ceiling, i.e. a real, height-budget dead end, not something a better
    (x, y) choice under the same ceiling constraint could fix.

    The caller (`Policy._fallback_action`) still has to hand the
    environment *some* placement, so this drops the one constraint that
    dead end is actually about -- `fits_ceiling` -- and returns the single
    lowest-landing spot on the grid instead, still checked against every
    real item's exact box (same margin `best_position`'s own retry loop
    uses) so it never trades a now-likely inclusion/ceiling failure for a
    still-avoidable collision. An inclusion/ceiling failure ends the
    episode exactly like any other failure does (see env.py) -- no worse
    than today -- but a bare, unverified guess measured against the real
    simulator landed squarely on top of four already-packed items at once
    (a collision distance of -5cm to -8.6cm, not a near miss), which this
    is written specifically to stop being the near-certain outcome.
    """
    n = state.grid_n
    fw = max(1, min(n, int(math.ceil(footprint_x / max(state.cell_w, 1e-6)))))
    fh = max(1, min(n, int(math.ceil(footprint_y / max(state.cell_h, 1e-6)))))

    top_windows = _windows(state.height_grid, fw, fh)
    top = top_windows.max(axis=(2, 3))
    landing_bottom = (top + LANDING_CLEARANCE).reshape(-1)
    order = np.argsort(landing_bottom)

    fallback_x = fallback_y = fallback_z_bottom = None
    for rank, flat_idx in enumerate(order):
        ix, iy = np.unravel_index(int(flat_idx), top.shape)
        x_center = state.x_min + (ix + fw / 2.0) * state.cell_w
        y_center = state.y_min + (iy + fh / 2.0) * state.cell_h
        z_bottom = float(landing_bottom[flat_idx])
        if fallback_x is None:
            fallback_x, fallback_y, fallback_z_bottom = x_center, y_center, z_bottom
        if rank >= BEST_EFFORT_MAX_CANDIDATES:
            break
        if _exact_aabb_clear(state, x_center, y_center, footprint_x, footprint_y, z_bottom, z_bottom + item_height):
            return {"x": x_center, "y": y_center, "z": z_bottom + item_height / 2.0}

    # Every candidate tried was still too close to something real: hand
    # back the single lowest spot anyway (unverified) rather than nothing
    # -- still strictly the same landing height a from-scratch heightmap
    # search would have picked, unlike the old flat (0, 0) guess.
    return {"x": fallback_x, "y": fallback_y, "z": fallback_z_bottom + item_height / 2.0}
