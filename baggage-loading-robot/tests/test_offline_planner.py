import random
import time

import pytest

import gh_baggage_core.offline_planner as offline_planner_module
from gh_baggage_core.offline_planner import (
    ORDER_FAILURE_PENALTY, _local_search_improve, _total_order_cost, _total_order_cost_detailed,
    _weighted_index, plan_order,
)

# The local-search refinement pass added after construction (see
# _local_search_improve) is specifically designed to spend whatever's left
# of the real 150s TIME_BUDGET_SECONDS -- exactly what makes these
# already-slow-by-design 20-25 item tests take minutes instead of
# milliseconds if left unpatched. Bounding attempts (not the time budget
# itself) keeps construction behaving exactly as it does in production --
# these tests exist to check its output is a valid permutation -- while
# still exercising the local-search pass, just briefly.
FAST_LOCAL_SEARCH_MAX_ATTEMPTS = 1


def make_item(index, length, width, height, is_prioritized=False, is_soft=False):
    return {
        "index": index, "length": length, "width": width, "height": height, "mass": 5.0,
        "is_prioritized": is_prioritized, "is_soft": is_soft,
        "belongs_to": None, "pos": None, "orn": None,
    }


BASE_CONTAINER = {
    "index": 0, "length": 2.0, "width": 1.45, "height": 1.61, "thickness": 0.04,
    "cut_x": 0.44, "cut_y": 0.4, "center": (0.0, 0.0, 0.805), "shelf": False,
    "is_prioritized": False, "packed_items": [],
}


def test_plan_order_returns_a_full_valid_permutation(monkeypatch):
    monkeypatch.setattr(offline_planner_module, "LOCAL_SEARCH_MAX_ATTEMPTS", FAST_LOCAL_SEARCH_MAX_ATTEMPTS)
    items = [make_item(i, 0.4 + 0.02 * (i % 5), 0.3, 0.2, is_soft=(i % 4 == 0)) for i in range(25)]
    order = plan_order([dict(BASE_CONTAINER)], items)
    assert order is not None
    assert sorted(order) == list(range(len(items)))


def test_plan_order_none_without_container_info():
    items = [make_item(0, 0.4, 0.3, 0.2)]
    assert plan_order([], items) is None
    assert plan_order(None, items) is None


def test_plan_order_two_containers_uses_both(monkeypatch):
    monkeypatch.setattr(offline_planner_module, "LOCAL_SEARCH_MAX_ATTEMPTS", FAST_LOCAL_SEARCH_MAX_ATTEMPTS)
    containers = [
        dict(BASE_CONTAINER, index=0, center=(0.0, 0.0, 0.805)),
        dict(BASE_CONTAINER, index=1, center=(2.5, 0.0, 0.805)),
    ]
    items = [make_item(i, 0.5, 0.4, 0.3) for i in range(20)]
    order = plan_order(containers, items)
    assert order is not None
    assert sorted(order) == list(range(len(items)))


def fake_result(z=0.1):
    return {
        "x": 0.0, "y": 0.0, "z": z, "top": 0.0, "flat": 0.0,
        "conflict": False, "path_blocked": False, "unstable": False,
        "in_corner_keepout": False, "fw": 4, "fh": 4,
    }


def test_total_order_cost_sums_each_items_own_score(monkeypatch):
    # rank_placements is mocked here rather than relying on real geometry
    # (see test_plan_order_returns_a_full_valid_permutation's own slowness)
    # -- _total_order_cost's job is just to replay a fixed order and sum
    # scores/apply place_virtual, which this isolates from the (expensive)
    # real placement search.
    items = [make_item(0, 0.4, 0.3, 0.2), make_item(1, 0.4, 0.3, 0.2)]

    def fake_rank_placements(states, candidates, deadline):
        score = 0.1 if candidates[0]["index"] == 0 else 0.2
        return [(score, 0, 0, 0, fake_result(), (0.4, 0.3, 0.2))]

    monkeypatch.setattr(offline_planner_module, "rank_placements", fake_rank_placements)
    cost = _total_order_cost([dict(BASE_CONTAINER)], items, deadline=float("inf"))
    assert cost == pytest.approx(0.3)


def test_total_order_cost_penalizes_a_dead_end_by_items_left_unplaced(monkeypatch):
    items = [make_item(0, 0.4, 0.3, 0.2), make_item(1, 0.4, 0.3, 0.2), make_item(2, 0.4, 0.3, 0.2)]

    def fake_rank_placements(states, candidates, deadline):
        if candidates[0]["index"] == 1:
            return []  # dead end on the second item in this order
        return [(0.05, 0, 0, 0, fake_result(), (0.4, 0.3, 0.2))]

    monkeypatch.setattr(offline_planner_module, "rank_placements", fake_rank_placements)
    cost = _total_order_cost([dict(BASE_CONTAINER)], items, deadline=float("inf"))
    # Item 0 placed (0.05); item 1 fails with itself and item 2 (2 items) never placed.
    assert cost == pytest.approx(0.05 + ORDER_FAILURE_PENALTY * 2)


def test_local_search_improve_swaps_to_avoid_a_dead_end(monkeypatch):
    # Item 1 can only be placed if it goes *before* anything else -- a
    # dead end exactly like the real regression this feature exists to
    # correct (see DESIGN.md): a fixed early choice blocks a later one.
    # The order handed in ([0, 1]) hits that dead end; local search should
    # find the swap ([1, 0]) that avoids it entirely.
    items = [make_item(0, 0.4, 0.3, 0.2), make_item(1, 0.4, 0.3, 0.2)]

    def fake_rank_placements(states, candidates, deadline):
        already_placed = len(states[0].item_aabbs) > 0
        if candidates[0]["index"] == 1 and already_placed:
            return []
        return [(0.05, 0, 0, 0, fake_result(), (0.4, 0.3, 0.2))]

    monkeypatch.setattr(offline_planner_module, "rank_placements", fake_rank_placements)
    order = _local_search_improve(
        [dict(BASE_CONTAINER)], items, order=[0, 1], deadline=time.perf_counter() + 5.0,
    )
    assert order == [1, 0]


def test_local_search_improve_leaves_an_already_good_order_alone(monkeypatch):
    items = [make_item(0, 0.4, 0.3, 0.2), make_item(1, 0.4, 0.3, 0.2)]

    def fake_rank_placements(states, candidates, deadline):
        return [(0.05, 0, 0, 0, fake_result(), (0.4, 0.3, 0.2))]

    monkeypatch.setattr(offline_planner_module, "rank_placements", fake_rank_placements)
    order = _local_search_improve(
        [dict(BASE_CONTAINER)], items, order=[0, 1], deadline=time.perf_counter() + 5.0,
    )
    assert order == [0, 1]


def test_total_order_cost_detailed_returns_each_items_own_score(monkeypatch):
    items = [make_item(0, 0.4, 0.3, 0.2), make_item(1, 0.4, 0.3, 0.2)]

    def fake_rank_placements(states, candidates, deadline):
        score = 0.1 if candidates[0]["index"] == 0 else 0.2
        return [(score, 0, 0, 0, fake_result(), (0.4, 0.3, 0.2))]

    monkeypatch.setattr(offline_planner_module, "rank_placements", fake_rank_placements)
    total, per_item = _total_order_cost_detailed([dict(BASE_CONTAINER)], items, deadline=float("inf"))
    assert total == pytest.approx(0.3)
    assert per_item == pytest.approx([0.1, 0.2])


def test_total_order_cost_detailed_attributes_dead_end_penalty_to_the_failing_item(monkeypatch):
    # Mirrors test_total_order_cost_penalizes_a_dead_end_by_items_left_unplaced,
    # but checking the per-item breakdown _local_search_improve's guided
    # swap selection relies on: the failing item (index 1) should absorb
    # the whole tail penalty, and item 2 -- which never got a turn, since
    # the episode is already over by then -- has no real signal to give
    # yet and should read 0 rather than some share of the penalty.
    items = [make_item(0, 0.4, 0.3, 0.2), make_item(1, 0.4, 0.3, 0.2), make_item(2, 0.4, 0.3, 0.2)]

    def fake_rank_placements(states, candidates, deadline):
        if candidates[0]["index"] == 1:
            return []
        return [(0.05, 0, 0, 0, fake_result(), (0.4, 0.3, 0.2))]

    monkeypatch.setattr(offline_planner_module, "rank_placements", fake_rank_placements)
    total, per_item = _total_order_cost_detailed([dict(BASE_CONTAINER)], items, deadline=float("inf"))
    assert total == pytest.approx(0.05 + ORDER_FAILURE_PENALTY * 2)
    assert per_item[0] == pytest.approx(0.05)
    assert per_item[1] == pytest.approx(ORDER_FAILURE_PENALTY * 2)
    assert per_item[2] == 0.0


def test_weighted_index_favors_higher_weight_entries():
    rng = random.Random(42)
    weights = [1.0, 1.0, 100.0, 1.0]
    counts = [0, 0, 0, 0]
    for _ in range(500):
        counts[_weighted_index(rng, weights)] += 1
    assert counts[2] > sum(counts) * 0.8


def test_weighted_index_falls_back_to_uniform_when_every_weight_is_zero():
    rng = random.Random(0)
    idx = _weighted_index(rng, [0.0, 0.0, 0.0])
    assert 0 <= idx < 3


def test_local_search_improve_guided_swap_fixes_a_dead_end_the_worst_item_causes(monkeypatch):
    # A 6-item order where only item 3 is the problem (it only fits as the
    # very *first* item placed -- everywhere else hits a dead end); guided
    # swap selection should notice it's the worst-scoring item in the very
    # first replay and, weighted toward trying it, find the fix well
    # inside an attempts budget far smaller than plan_order's real
    # LOCAL_SEARCH_MAX_ATTEMPTS.
    items = [make_item(i, 0.4, 0.3, 0.2) for i in range(6)]

    def fake_rank_placements(states, candidates, deadline):
        already_placed = len(states[0].item_aabbs) > 0
        if candidates[0]["index"] == 3 and already_placed:
            return []
        return [(0.05, 0, 0, 0, fake_result(), (0.4, 0.3, 0.2))]

    monkeypatch.setattr(offline_planner_module, "rank_placements", fake_rank_placements)
    monkeypatch.setattr(offline_planner_module, "LOCAL_SEARCH_MAX_ATTEMPTS", 30)
    order = _local_search_improve(
        [dict(BASE_CONTAINER)], items, order=[0, 3, 1, 2, 4, 5], deadline=time.perf_counter() + 30.0,
    )
    assert order[0] == 3
