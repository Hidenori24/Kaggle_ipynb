"""Offline planner: simulate the whole pack before the conveyor even starts.

When the harness enables offline optimization, `Agent.optimize()` receives
every item up front and gets a much larger time budget (180s) than the
online `policy()` (8-10s *per call*). Just as importantly, the online phase
in that mode only ever sees ONE item at a time (`lookahead_k == 1` --
see the simulator README's task table), so `optimize()`'s return value is
the *only* place any real sequencing decision can be made; the online phase
has no freedom left to pick a different item once the stream starts.

Rather than a static sort (ordering.py's largest-first/soft-last heuristic),
this runs the same search the online policy uses
(selection.choose_placement) as a dry run: repeatedly picking, from every
item not yet placed, the best (item, orientation, container, position)
combination, and recording that item's index into the plan. This is a
proper greedy best-fit construction with full information, instead of a
sort followed by a policy that can only ever act on whichever single item
the plan hands it next.

Small drift between this dry run's assumed coordinates and the real
physics-settled state doesn't matter: the online policy still independently
recomputes actual placements against the live observation every step. Only
the resulting *order* this function returns is used.
"""
from __future__ import annotations

import time

from .container_state import ContainerState
from .selection import choose_placement, largest_first_order

# Well under the 180s optimization_timeout so a slower/loaded evaluation
# host can't push us over it; on timeout the harness falls back to leaving
# the stream in its original order, which is much worse than even a
# partially-completed plan, so we always return *something* covering every
# item -- see the `remaining` fallback below.
TIME_BUDGET_SECONDS = 150.0


def plan_order(container_list: list[dict], item_list: list[dict]) -> list[int] | None:
    """Return a full permutation of every item's index, or None if planning
    isn't possible at all (caller should fall back to a simpler heuristic)."""
    if not container_list or not item_list:
        return None

    deadline = time.perf_counter() + TIME_BUDGET_SECONDS
    states = [ContainerState(c) for c in container_list]
    remaining = list(item_list)
    order: list[int] = []

    while remaining:
        if time.perf_counter() > deadline:
            # Out of planning time: append whatever's left in the simple
            # largest-first heuristic order rather than leaving it out.
            order.extend(item["index"] for item in _fallback_ordered(remaining))
            return order

        candidate_order = largest_first_order(remaining)
        candidates = [remaining[i] for i in candidate_order]
        best = choose_placement(states, candidates, deadline)

        if best is None:
            # Nothing fits anywhere in either container any more (both
            # effectively full): nothing we could plan for the rest would
            # be trustworthy either, so just hand back what's left.
            order.extend(item["index"] for item in _fallback_ordered(remaining))
            return order

        _, candidate_idx, c_idx, orn_idx, result, (dl, dw, dh) = best
        remaining_idx = candidate_order[candidate_idx]
        item = remaining.pop(remaining_idx)
        order.append(item["index"])

        top_z = result["z"] + dh / 2.0
        states[c_idx].place_virtual(
            result["x"], result["y"], dl, dw, top_z,
            is_soft=bool(item.get("is_soft", False)),
            is_prioritized=bool(item.get("is_prioritized", False)),
        )

    return order


def _fallback_ordered(items: list[dict]) -> list[dict]:
    order = largest_first_order(items)
    return [items[i] for i in order]
