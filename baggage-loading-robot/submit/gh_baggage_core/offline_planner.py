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

Once that construction finishes, whatever's left of the 150s budget goes
into a second pass: classic construct-then-refine, `_local_search_improve`
tries swapping pairs of items in the constructed order and keeps a swap
only when replaying the *whole* order with it (`_total_order_cost`) is a
real, measured improvement. Evaluating one fixed order this way is far
cheaper than choosing one (each item only competes against itself, not
every other remaining item), which is what leaves room for this second
pass at all within the same budget the construction already uses.
"""
from __future__ import annotations

import random
import time

from .container_state import ContainerState
from .selection import largest_first_order, pick_with_lookahead, rank_placements

# Well under the 180s optimization_timeout so a slower/loaded evaluation
# host can't push us over it; on timeout the harness falls back to leaving
# the stream in its original order, which is much worse than even a
# partially-completed plan, so we always return *something* covering every
# item -- see the `remaining` fallback below.
TIME_BUDGET_SECONDS = 150.0

# Same lookahead the online policy uses (see policy.py), applied here too:
# a move that looks best in isolation but leaves the next item nowhere good
# to go loses to a slightly costlier one that doesn't. Kept modest rather
# than "as deep as the 180s budget allows" -- unlike the online policy's
# pool (<=40), this dry run's remaining-item count can be up to 80, and
# each unit of lookahead roughly multiplies the per-step cost, so pushing it
# much further risks running out of planning time before the deadline
# check below can gracefully hand off the tail to the plain heuristic order.
LOOKAHEAD_BRANCH = 2
LOOKAHEAD_STEPS = 1


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
        ranked = rank_placements(states, candidates, deadline)

        if not ranked:
            # Nothing fits anywhere in either container any more (both
            # effectively full): nothing we could plan for the rest would
            # be trustworthy either, so just hand back what's left.
            order.extend(item["index"] for item in _fallback_ordered(remaining))
            return order

        best = pick_with_lookahead(
            states, candidates, ranked, deadline,
            branch=LOOKAHEAD_BRANCH, steps=LOOKAHEAD_STEPS,
        )

        _, candidate_idx, c_idx, orn_idx, result, (dl, dw, dh) = best
        remaining_idx = candidate_order[candidate_idx]
        item = remaining.pop(remaining_idx)
        order.append(item["index"])

        top_z = result["z"] + dh / 2.0
        states[c_idx].place_virtual(
            result["x"], result["y"], dl, dw, top_z,
            is_soft=bool(item.get("is_soft", False)),
            is_prioritized=bool(item.get("is_prioritized", False)),
            dh=dh,
        )

    order = _local_search_improve(container_list, item_list, order, deadline)
    return order


def _fallback_ordered(items: list[dict]) -> list[dict]:
    order = largest_first_order(items)
    return [items[i] for i in order]


# A dead end partway through a fixed order costs vastly more than any
# ordinary placement's own score ever does (top*1000-ish at most): a real
# failure this early ends the whole episode, torching every item after it
# (see env.py). Scaled by how many items never got the chance to place at
# all, so a fixed order that fails on item 5 of 80 scores far worse than
# one that fails on item 75 of 80 -- both are real failures, but the
# earlier one is a strictly worse outcome, and the local search needs that
# distinction to actually prefer delaying failure over "any failure is
# equally infinite."
ORDER_FAILURE_PENALTY = 100000.0

# How many swap attempts the local search gets, on top of just running out
# of the deadline first (whichever comes first) -- a large but finite cap
# so a fast machine with lots of spare time budget doesn't spin forever
# once it's stopped finding real improvements.
LOCAL_SEARCH_MAX_ATTEMPTS = 2000


def _total_order_cost(container_list: list[dict], ordered_items: list[dict], deadline: float | None) -> float:
    """The same per-item score `plan_order`'s own construction already
    computes (landing height plus the same path/stability/priority risk
    penalties -- see selection.rank_placements), just replayed for a
    *fixed* order instead of choosing one: each item only ever competes
    against itself (a 1-candidate rank_placements call), never against
    the other remaining items, so this is O(n) placement searches for the
    whole order, not the O(n^2) rank_placements/pick_with_lookahead pattern
    `plan_order`'s construction needs to actually pick that order in the
    first place. That gap is exactly what makes a local-search refinement
    pass over an already-built order affordable within the same time
    budget the construction itself used.
    """
    states = [ContainerState(c) for c in container_list]
    total = 0.0
    for i, item in enumerate(ordered_items):
        if deadline is not None and time.perf_counter() > deadline:
            total += ORDER_FAILURE_PENALTY * (len(ordered_items) - i)
            return total
        ranked = rank_placements(states, [item], deadline)
        if not ranked:
            total += ORDER_FAILURE_PENALTY * (len(ordered_items) - i)
            return total
        score, _candidate_idx, c_idx, _orn_idx, result, (dl, dw, dh) = ranked[0]
        total += score
        top_z = result["z"] + dh / 2.0
        states[c_idx].place_virtual(
            result["x"], result["y"], dl, dw, top_z,
            is_soft=bool(item.get("is_soft", False)),
            is_prioritized=bool(item.get("is_prioritized", False)),
            dh=dh,
        )
    return total


def _local_search_improve(container_list: list[dict], item_list: list[dict],
                           order: list[int], deadline: float | None) -> list[int]:
    """Classic construct-then-refine: `plan_order`'s own greedy dry run is
    the construction, this is the refinement -- a 2-opt-style local search
    (try swapping two items' positions in the sequence; keep the swap only
    if replaying the *whole* order with it is a real, measured improvement
    over not swapping, exactly `_total_order_cost` above) over whatever
    time budget the construction didn't use. Never returns anything worse
    than the order it was handed: every kept swap is independently
    verified to lower the same score construction already optimizes for,
    so unlike deepening the lookahead itself (tried and rejected -- see
    DESIGN.md, where a "more informed" deeper look turned out to be a
    *less* reliable signal than the shallow one it was meant to correct),
    this can't be fooled by an unreliable estimator: the full replay it
    checks against is the exact same ground truth the construction itself
    already trusts, not a fresh heuristic guessing at it.
    """
    n = len(order)
    if n < 2 or deadline is None:
        return order

    index_to_item = {item["index"]: item for item in item_list}
    ordered_items = [index_to_item[idx] for idx in order]
    best_cost = _total_order_cost(container_list, ordered_items, deadline)

    rng = random.Random(0)
    attempts = 0
    while attempts < LOCAL_SEARCH_MAX_ATTEMPTS and time.perf_counter() < deadline:
        attempts += 1
        i, j = rng.sample(range(n), 2)
        ordered_items[i], ordered_items[j] = ordered_items[j], ordered_items[i]
        new_cost = _total_order_cost(container_list, ordered_items, deadline)
        if new_cost < best_cost:
            best_cost = new_cost
        else:
            ordered_items[i], ordered_items[j] = ordered_items[j], ordered_items[i]

    return [item["index"] for item in ordered_items]
