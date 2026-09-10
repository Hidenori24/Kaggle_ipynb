from gh_baggage_core.container_state import ContainerState
from gh_baggage_core.packing import CONTACT_TOLERANCE, LANDING_CLEARANCE, best_position

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
    assert abs(result["z"] - (state.floor_z + LANDING_CLEARANCE + 0.1)) < 1e-6


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


def test_stability_check_catches_off_center_support_a_flat_range_check_would_miss():
    # A "picture frame" of raised cells around a hollow, unraised center:
    # the height *range* between the highest and lowest cell here is
    # substantial (would trip a naive max-min-height "flat" threshold too),
    # but the interesting case is that the raised border alone already
    # covers most of the footprint's *area* (a plain support-fraction
    # check, with no notion of *where* the support is, could wrongly call
    # this fine) while the box's own center -- where its weight actually
    # bears down -- hangs directly over the unsupported hollow.
    state = ContainerState(dict(BASE_CONTAINER), grid_n=8)
    border_top = state.floor_z + 0.3
    state.height_grid[:, :] = state.floor_z
    state.height_grid[0:2, :] = border_top
    state.height_grid[6:8, :] = border_top
    state.height_grid[:, 0:2] = border_top
    state.height_grid[:, 6:8] = border_top
    # Sanity check the fixture: border cells clearly outnumber the hollow.
    assert (state.height_grid == border_top).mean() > 0.6

    footprint = (state.x_max - state.x_min), (state.y_max - state.y_min)
    result = best_position(state, footprint_x=footprint[0], footprint_y=footprint[1], item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert result["support_fraction"] > 0.6
    assert result["core_support_fraction"] < 0.5
    assert result["unstable"]


def test_stability_check_accepts_a_small_step_that_still_covers_the_center():
    # The mirror image: most of the footprint (including dead center) sits
    # at one height, with only a small strip noticeably lower -- a large
    # max-min "flat" range, but genuinely well supported underneath the
    # box's own center of mass, unlike the case above.
    state = ContainerState(dict(BASE_CONTAINER), grid_n=8)
    state.height_grid[:, :] = state.floor_z + 0.3
    state.height_grid[0:1, :] = state.floor_z  # a thin low strip at one edge

    footprint = (state.x_max - state.x_min), (state.y_max - state.y_min)
    result = best_position(state, footprint_x=footprint[0], footprint_y=footprint[1], item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert result["flat"] > CONTACT_TOLERANCE * 2  # a real height-range gap
    assert not result["unstable"]


def test_best_position_avoids_placement_that_requires_crossing_a_tall_item():
    # A tall item sits right at the door, spanning the full width. Anything
    # placed deeper in (larger local y) in the same x-columns would have to
    # slide straight through it to enter -- best_position must not treat
    # that as a clean, unpenalized spot when a genuinely clear one exists.
    container = dict(BASE_CONTAINER)
    container["packed_items"] = [
        {
            "length": 0.3, "width": 0.3, "height": 0.8,
            "pos": (0.0, -0.55, 0.04 + 0.4),  # near the door (y_min side)
            "orn": (0.0, 0.0, 0.0, 1.0),
            "is_soft": False,
            "is_prioritized": False,
        }
    ]
    state = ContainerState(container, grid_n=24)
    result = best_position(state, footprint_x=0.25, footprint_y=0.25, item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    assert not result["path_blocked"]
    # It should have picked an x-column that isn't behind the tall blocker.
    assert not (-0.15 <= result["x"] <= 0.15)
