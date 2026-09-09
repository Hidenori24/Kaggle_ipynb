from gh_baggage_core.container_state import ContainerState
from gh_baggage_core.packing import best_position

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


def test_best_position_lands_on_floor_when_empty():
    state = ContainerState(dict(BASE_CONTAINER), grid_n=24)
    result = best_position(state, footprint_x=0.3, footprint_y=0.3, item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert abs(result["top"] - state.floor_z) < 1e-6
    assert abs(result["z"] - (state.floor_z + 0.1)) < 1e-6


def test_best_position_stacks_on_top_of_existing_item():
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
    state = ContainerState(container, grid_n=24)
    result = best_position(state, footprint_x=0.3, footprint_y=0.3, item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    # Should land on top of the big base item, not find floor space beside it
    assert result["top"] > 0.3


def test_best_position_avoids_soft_top_when_alternative_exists():
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
    state = ContainerState(container, grid_n=24)
    result = best_position(state, footprint_x=0.3, footprint_y=0.3, item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert not result["conflict"]
    # Landed on the (still-empty) floor elsewhere, not on top of the soft item
    assert abs(result["top"] - state.floor_z) < 1e-6


def test_best_position_returns_none_when_item_too_tall():
    state = ContainerState(dict(BASE_CONTAINER), grid_n=16)
    result = best_position(state, footprint_x=0.3, footprint_y=0.3, item_height=10.0,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is None
