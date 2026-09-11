"""Shared "pick the best (item, orientation, container, position)" search.

Used both by the online policy (searching over the currently-visible pool)
and the offline planner (searching over every item not yet placed, with a
much larger time budget). Keeping this in one place means the offline dry
run and the online policy make the same judgment calls, so the sequence the
planner commits to is one the online phase can actually follow.
"""
from __future__ import annotations

import time

from .container_state import ContainerState
from .geometry import NUM_ORIENTATIONS, oriented_dims
from .packing import best_position

PRIORITY_CONTAINER_VIOLATION_PENALTY = 1000.0
PRIORITY_CONTAINER_RESERVE_PENALTY = 0.05
TOP_CONFLICT_PENALTY = 0.5
PATH_BLOCKED_PENALTY = 100.0
UNSTABLE_PENALTY = 100.0
CORNER_KEEPOUT_PENALTY = 60.0
FOOTPRINT_BONUS_SCALE = 0.001

# cog_score ("重心スコア") rewards the whole load's center of gravity sitting
# low, weighted by each item's mass -- entirely orthogonal to what the rest
# of this scoring already optimizes (fill/stability/path-safety don't care
# which item ends up where, only that *something* lands safely). Each
# item's own best spot already comes out lowest-first (packing.py's `top`
# term), so the lever available here is *which* item gets to claim
# whatever's the best spot available *this step*, before something else
# grabs it and a heavy item is left to stack higher later: give heavier
# items a small bonus so they tend to be selected earlier, while the
# container still has more low floor space to offer.
#
# This is a flat per-item nudge, not scaled by the candidate's own height:
# it must stay far too small to ever flip a real placement-quality
# difference (a materially lower or safer spot always has to win on its
# own terms), since those differences guard against ending the whole
# episode outright (see env.py: any single failed placement terminates
# it) -- no amount of cog benefit is worth risking that. It only breaks
# ties between options that are otherwise comparable.
MASS_PRIORITY_WEIGHT = 0.001


def rank_placements(states: list[ContainerState], candidates: list[dict], deadline: float | None = None):
    """Search every (candidate, orientation, container) combination and
    return each candidate item's own best placement, sorted best-first.

    `candidates` should already be ordered largest-first: under a time
    budget we bail out between candidates, so that ordering determines which
    items got a fair evaluation.

    Each entry is (score, candidate_idx, container_idx, orn_idx, result,
    (dl, dw, dh)). Returns [] if nothing fits anywhere for anyone.
    """
    has_priority_container = any(s.is_prioritized for s in states)
    multi_container = len(states) > 1

    per_candidate_best: dict[int, tuple] = {}
    for candidate_idx, item in enumerate(candidates):
        length, width, height = item["length"], item["width"], item["height"]
        is_soft = bool(item.get("is_soft", False))
        is_prioritized = bool(item.get("is_prioritized", False))

        for orn_idx in range(NUM_ORIENTATIONS):
            dl, dw, dh = oriented_dims(length, width, height, orn_idx)

            for c_idx, state in enumerate(states):
                container_penalty = 0.0
                if is_prioritized and has_priority_container and not state.is_prioritized:
                    container_penalty += PRIORITY_CONTAINER_VIOLATION_PENALTY
                if (
                    not is_prioritized
                    and has_priority_container
                    and state.is_prioritized
                    and multi_container
                ):
                    container_penalty += PRIORITY_CONTAINER_RESERVE_PENALTY

                result = best_position(
                    state, dl, dw, dh,
                    avoid_soft_top=not is_soft,
                    avoid_priority_top=not is_prioritized,
                )
                if result is None:
                    continue

                mass = float(item.get("mass", 1.0) or 1.0)
                score = result["top"] + container_penalty - MASS_PRIORITY_WEIGHT * mass
                if result["conflict"]:
                    score += TOP_CONFLICT_PENALTY
                if result["path_blocked"]:
                    score += PATH_BLOCKED_PENALTY
                if result["unstable"]:
                    score += UNSTABLE_PENALTY
                if result["in_corner_keepout"]:
                    score += CORNER_KEEPOUT_PENALTY
                score -= FOOTPRINT_BONUS_SCALE * (dl * dw)

                entry = (score, candidate_idx, c_idx, orn_idx, result, (dl, dw, dh))
                current = per_candidate_best.get(candidate_idx)
                if current is None or score < current[0]:
                    per_candidate_best[candidate_idx] = entry

        if deadline is not None and time.perf_counter() > deadline:
            break

    return sorted(per_candidate_best.values(), key=lambda entry: entry[0])


def choose_placement(states: list[ContainerState], candidates: list[dict], deadline: float | None = None):
    """The single best (candidate, orientation, container, position)
    combination, or None if nothing fits anywhere. See rank_placements."""
    ranked = rank_placements(states, candidates, deadline)
    return ranked[0] if ranked else None


def largest_first_order(items: list[dict]) -> list[int]:
    return sorted(
        range(len(items)),
        key=lambda i: -(items[i]["length"] * items[i]["width"] * items[i]["height"]),
    )


DEAD_END_PENALTY = 200.0


def pick_with_lookahead(states: list[ContainerState], candidates: list[dict], ranked: list[tuple],
                         deadline: float | None, branch: int, steps: int):
    """Among the top `branch` first-moves in `ranked`, prefer the one whose
    greedy continuation over the rest of `candidates` (not the future
    stream -- whatever the caller can already see) racks up the least
    additional risk over `steps` further picks, rather than just the
    locally cheapest first move. Falls back to ranked[0] if nothing beats
    it (or there's only one candidate to begin with).

    This mutates nothing in `states`; each branch works on its own
    ContainerState.clone().
    """
    if len(ranked) <= 1 or branch <= 1 or steps <= 0:
        return ranked[0]

    best_branch = None
    for first_move in ranked[:branch]:
        if deadline is not None and time.perf_counter() > deadline:
            break
        score, candidate_idx, c_idx, orn_idx, result, (dl, dw, dh) = first_move
        cumulative = score

        cloned_states = [s.clone() for s in states]
        placed_item = candidates[candidate_idx]
        top_z = result["z"] + dh / 2.0
        cloned_states[c_idx].place_virtual(
            result["x"], result["y"], dl, dw, top_z,
            is_soft=bool(placed_item.get("is_soft", False)),
            is_prioritized=bool(placed_item.get("is_prioritized", False)),
            dh=dh,
        )
        remaining = [c for i, c in enumerate(candidates) if i != candidate_idx]

        depth = 0
        while remaining and depth < steps and (deadline is None or time.perf_counter() < deadline):
            sub_order = largest_first_order(remaining)
            sub_candidates = [remaining[i] for i in sub_order]
            sub_best = choose_placement(cloned_states, sub_candidates, deadline)
            if sub_best is None:
                cumulative += DEAD_END_PENALTY
                break
            sub_score, sub_cand_idx, sub_c_idx, _sub_orn, sub_result, (sdl, sdw, sdh) = sub_best
            cumulative += sub_score
            remaining_idx = sub_order[sub_cand_idx]
            sub_item = remaining.pop(remaining_idx)
            sub_top_z = sub_result["z"] + sdh / 2.0
            cloned_states[sub_c_idx].place_virtual(
                sub_result["x"], sub_result["y"], sdl, sdw, sub_top_z,
                is_soft=bool(sub_item.get("is_soft", False)),
                is_prioritized=bool(sub_item.get("is_prioritized", False)),
                dh=sdh,
            )
            depth += 1

        if best_branch is None or cumulative < best_branch[0]:
            best_branch = (cumulative, first_move)

    return best_branch[1] if best_branch is not None else ranked[0]
