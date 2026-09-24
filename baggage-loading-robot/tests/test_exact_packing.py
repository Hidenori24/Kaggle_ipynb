import pytest

from gh_baggage_core.container_state import ContainerState
from gh_baggage_core.exact_packing import best_position_exact
from gh_baggage_core.packing import LANDING_CLEARANCE, PATH_CLEARANCE

BASE_CONTAINER = {
    "index": 0,
    "length": 2.0,
    "width": 1.5,
    "height": 1.6,
    "thickness": 0.04,
    "cut_x": 0.4,
    "cut_y": 0.4,
    "center": (0.0, 0.0, 0.8),
    "shelf": False,
    "is_prioritized": False,
    "packed_items": [],
}


def test_best_position_exact_lands_on_floor_when_empty():
    state = ContainerState(dict(BASE_CONTAINER))
    result = best_position_exact(state, footprint_x=0.3, footprint_y=0.3, item_height=0.2,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert abs(result["top"] - state.floor_z) < 1e-6
    assert abs(result["z"] - (state.floor_z + LANDING_CLEARANCE + 0.1)) < 1e-6
    assert not result["unstable"]
    assert not result["path_blocked"]


def test_best_position_exact_stacks_on_top_of_existing_item():
    container = dict(BASE_CONTAINER)
    container["packed_items"] = [
        {
            "length": 1.8, "width": 1.3, "height": 0.3,
            "pos": (0.0, 0.0, 0.04 + 0.15),
            "orn": (0.0, 0.0, 0.0, 1.0),
            "is_soft": False,
            "is_prioritized": False,
        }
    ]
    state = ContainerState(container)
    result = best_position_exact(state, footprint_x=0.3, footprint_y=0.3, item_height=0.2,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert result["top"] > 0.3


def test_best_position_exact_avoids_soft_top_when_alternative_exists():
    container = dict(BASE_CONTAINER)
    container["packed_items"] = [
        {
            "length": 0.5, "width": 0.5, "height": 0.2,
            "pos": (-0.5, 0.0, 0.04 + 0.1),
            "orn": (0.0, 0.0, 0.0, 1.0),
            "is_soft": True,
            "is_prioritized": False,
        }
    ]
    state = ContainerState(container)
    result = best_position_exact(state, footprint_x=0.3, footprint_y=0.3, item_height=0.2,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert not result["conflict"]
    assert abs(result["top"] - state.floor_z) < 1e-6


def test_best_position_exact_returns_none_when_item_too_tall():
    state = ContainerState(dict(BASE_CONTAINER))
    result = best_position_exact(state, footprint_x=0.3, footprint_y=0.3, item_height=10.0,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is None


def test_best_position_exact_returns_none_when_footprint_too_wide():
    state = ContainerState(dict(BASE_CONTAINER))
    too_wide = (state.x_max - state.x_min) + 1.0
    result = best_position_exact(state, footprint_x=too_wide, footprint_y=0.3, item_height=0.2,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is None


def test_best_position_exact_flags_a_tall_narrow_box_even_with_perfect_support():
    state = ContainerState(dict(BASE_CONTAINER))
    result = best_position_exact(state, footprint_x=0.2, footprint_y=0.2, item_height=0.6,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert result["support_fraction"] == 1.0
    assert result["core_support_fraction"] == 1.0
    assert result["unstable"]


def test_best_position_exact_accepts_a_short_wide_box_with_perfect_support():
    state = ContainerState(dict(BASE_CONTAINER))
    result = best_position_exact(state, footprint_x=0.2, footprint_y=0.2, item_height=0.15,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert result["support_fraction"] == 1.0
    assert not result["unstable"]


def test_best_position_exact_rejects_a_candidate_too_close_to_a_real_item():
    state = ContainerState(dict(BASE_CONTAINER))
    corner_x = state.x_min + 0.1
    corner_y = state.y_min + 0.1
    state.item_aabbs.append((
        (corner_x - 0.1, corner_y - 0.1, state.floor_z),
        (corner_x + 0.1, corner_y + 0.1, state.floor_z + 0.2),
    ))

    result = best_position_exact(state, footprint_x=0.2, footprint_y=0.2, item_height=0.2,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    too_close_x = corner_x - 0.115 <= result["x"] <= corner_x + 0.115
    too_close_y = corner_y - 0.115 <= result["y"] <= corner_y + 0.115
    assert not (too_close_x and too_close_y)


def test_best_position_exact_returns_none_when_the_only_candidates_are_all_too_close():
    state = ContainerState(dict(BASE_CONTAINER))
    state.item_aabbs.append((
        (state.x_min - 1.0, state.y_min - 1.0, -10.0),
        (state.x_max + 1.0, state.y_max + 1.0, 10.0),
    ))
    result = best_position_exact(state, footprint_x=0.2, footprint_y=0.2, item_height=0.2,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is None


def test_best_position_exact_avoids_the_chamfer_wedge():
    container = dict(BASE_CONTAINER)
    container["n_vecs"] = [
        (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0),
        (-0.7071, 0.0, -0.7071), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
    ]
    container["points"] = [
        (0.0, 0.0, 0.04), (0.96, 0.0, 0.8), (0.0, 0.0, 1.56), (-0.96, 0.0, 0.8),
        (-0.96, 0.0, 0.44), (0.0, -0.71, 0.8), (0.0, 0.71, 0.8),
    ]
    state = ContainerState(container)
    result = best_position_exact(state, footprint_x=0.15, footprint_y=0.15, item_height=0.1,
                                  avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    # Must not land low in the wedge band right at x_min.
    assert not (result["x"] < state.x_min + state.cut_x and result["z"] < 0.5)


def test_best_position_exact_requires_a_real_transport_margin():
    # A tall item sits near the door, spanning the full width. A second,
    # thin marker item -- well clear of both the obstruction and the
    # footprint window on every axis -- exists only to give the
    # extreme-point search a candidate anchor further from the door (mirrors
    # packing.py's test_path_clearance_requires_a_real_margin: the obstacle
    # is real but too short of the real transport margin to actually clear,
    # so a footprint landing past it must still be flagged path_blocked,
    # even though there's no literal 3D overlap with anything).
    container = dict(BASE_CONTAINER, cut_x=0.0, cut_y=0.0)
    state = ContainerState(container)
    footprint_y = 0.2
    far_y0 = state.y_min + 0.3
    gap = PATH_CLEARANCE / 2.0
    obstruction_top = state.floor_z + LANDING_CLEARANCE - gap
    state.item_aabbs.append((
        (state.x_min, state.y_min, state.floor_z),
        (state.x_max, state.y_min + 0.1, obstruction_top),
    ))
    marker_top = state.floor_z + 0.005
    state.item_aabbs.append((
        (state.x_min, far_y0 - 0.01, state.floor_z),
        (state.x_max, far_y0, marker_top),
    ))
    result = best_position_exact(state, footprint_x=state.x_max - state.x_min, footprint_y=footprint_y,
                                  item_height=0.2, avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert result["path_blocked"]


def test_best_position_exact_require_safe_rejects_unstable_and_blocked_candidates():
    state = ContainerState(dict(BASE_CONTAINER))
    result = best_position_exact(state, footprint_x=0.2, footprint_y=0.2, item_height=0.6,
                                  avoid_soft_top=True, avoid_priority_top=True, require_safe=True)
    # An empty container never forces a tall/narrow candidate; require_safe
    # must reject every candidate this unstable rather than accept the
    # only unstable option available.
    assert result is None
