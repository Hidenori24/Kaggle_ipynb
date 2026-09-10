import numpy as np

from gh_baggage_core.container_state import ContainerState
from gh_baggage_core.packing import CONTACT_TOLERANCE, LANDING_CLEARANCE, _shadowed_clear_floor, best_position

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


def test_best_position_rejects_a_candidate_too_close_to_a_real_item_the_heightmap_missed():
    # A phantom real item placed directly into item_aabbs (bypassing
    # _mark_occupied/height_grid entirely) simulates a case the coarse
    # heightmap+padding approximation let slip through -- the exact check
    # is the only thing standing between this and an unsafe candidate.
    state = ContainerState(dict(BASE_CONTAINER), grid_n=24)
    # Without any occupancy, the door-clearing deep-bias plus the ix
    # tie-break make the leftmost (smallest ix), deepest (largest iy)
    # anchor the natural pick -- put the phantom exactly there.
    corner_x = state.x_min + 0.1
    corner_y = state.y_max - 0.1
    state.item_aabbs.append((
        (corner_x - 0.1, corner_y - 0.1, state.floor_z),
        (corner_x + 0.1, corner_y + 0.1, state.floor_z + 0.2),
    ))

    result = best_position(state, footprint_x=0.2, footprint_y=0.2, item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is not None
    # Must not land within the phantom item's real (margin-padded) box.
    too_close_x = corner_x - 0.115 <= result["x"] <= corner_x + 0.115
    too_close_y = corner_y - 0.115 <= result["y"] <= corner_y + 0.115
    assert not (too_close_x and too_close_y)


def test_best_position_returns_none_when_the_only_candidates_are_all_too_close():
    state = ContainerState(dict(BASE_CONTAINER), grid_n=6)
    # A phantom item spanning the whole floor footprint and the whole
    # height range: no window the (tiny) grid can offer separates from it
    # on any axis, regardless of what height the (untouched) height_grid
    # reports.
    state.item_aabbs.append((
        (state.x_min - 1.0, state.y_min - 1.0, -10.0),
        (state.x_max + 1.0, state.y_max + 1.0, 10.0),
    ))
    result = best_position(state, footprint_x=0.2, footprint_y=0.2, item_height=0.2,
                            avoid_soft_top=True, avoid_priority_top=True)
    assert result is None


def test_exact_chamfer_check_still_blocks_the_true_wedge_corner():
    from gh_baggage_core.packing import _chamfer_fits

    container = dict(BASE_CONTAINER)
    container["n_vecs"] = [
        (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0),
        (-0.7071, 0.0, -0.7071), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
    ]
    container["points"] = [
        (0.0, 0.0, 0.04), (0.96, 0.0, 0.8), (0.0, 0.0, 1.56), (-0.96, 0.0, 0.8),
        (-0.96, 0.0, 0.44), (0.0, -0.71, 0.8), (0.0, 0.71, 0.8),
    ]
    state = ContainerState(container, grid_n=16)

    # A single window landing right at the wedge tip (x close to x_min, low
    # z) must be rejected...
    wedge_z = np.array([[0.1]])
    assert not _chamfer_fits(state, n0=1, n1=1, fw=1, fh=1, z_center=wedge_z,
                              hx=0.02, hy=0.02, hz=0.02).all()

    # ...but the same footprint higher up in the band, clear of the wedge,
    # must still be accepted (this is exactly the volume the old crude
    # "block the whole band" approximation used to waste).
    clear_z = np.array([[0.9]])
    assert _chamfer_fits(state, n0=1, n1=1, fw=1, fh=1, z_center=clear_z,
                          hx=0.02, hy=0.02, hz=0.02).all()


def test_shadowed_clear_floor_counts_only_still_clear_cells_strictly_behind():
    # A tiny, fully hand-controlled grid so the counts are easy to verify by
    # hand: grid_n=6, fw=2, fh=2 -> n0=n1=5. Row/col 0 is the door side.
    state = ContainerState(dict(BASE_CONTAINER), grid_n=6)
    state.height_grid[:, :] = state.floor_z
    # Occupy (raise) a couple of cells so they no longer count as "clear".
    state.height_grid[2, 4] = state.floor_z + 0.5
    state.height_grid[3, 5] = state.floor_z + 0.5

    shadow = _shadowed_clear_floor(state, fw=2, fh=2, n0=5, n1=5)
    assert shadow.shape == (5, 5)

    # Window anchored at (ix=2, iy=0) spans x in [2,4), y in [0,2); "behind"
    # (y >= 2) in those same x-columns is a 2x4 block (rows 2..5), minus the
    # two occupied cells above -> 8 - 2 = 6 clear cells.
    assert shadow[2, 0] == 6.0

    # A window anchored right at the back (iy=4, the last possible fh=2
    # window) has nothing behind it at all.
    assert (shadow[:, 4] == 0.0).all()

    # A window in x-columns with nothing raised anywhere behind it counts
    # the full remaining depth.
    assert shadow[0, 0] == 2 * 4  # fw=2 columns, 4 clear rows behind (y=2..5)


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
