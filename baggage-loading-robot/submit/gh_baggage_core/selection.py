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

                score = result["top"] + container_penalty
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
