from gh_baggage_core.container_state import ContainerState

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


def test_empty_container_floor_is_flat():
    state = ContainerState(dict(BASE_CONTAINER), grid_n=16)
    # Away from the chamfered-corner keepout band, the floor should be flat.
    interior = state.height_grid[4:-4, :]
    assert (interior == state.floor_z).all()


def test_cut_corner_keepout_raises_x_min_side_only():
    # The simulator always anchors the LD3 chamfer at local x_min (see
    # write_open_cut_corner_cup_obj / Container._create_small_shelf, both of
    # which build the cut at -length/2) -- never at x_max.
    state = ContainerState(dict(BASE_CONTAINER), grid_n=16)
    assert (state.height_grid[0, :] > state.floor_z).all()
    assert (state.corner_keepout[0, :]).all()
    assert (state.height_grid[-1, :] == state.floor_z).all()
    assert not state.corner_keepout[-1, :].any()


def test_packed_item_raises_heightmap_and_tags_top():
    container = dict(BASE_CONTAINER)
    container["packed_items"] = [
        {
            "length": 0.4, "width": 0.4, "height": 0.3,
            "pos": (0.0, 0.0, 0.04 + 0.15),  # sitting on the floor (thickness=0.04)
            "orn": (0.0, 0.0, 0.0, 1.0),
            "is_soft": True,
            "is_prioritized": False,
        }
    ]
    state = ContainerState(container, grid_n=16)
    # Somewhere near the center the height should now read ~0.04+0.3
    center_ix, center_iy = state.grid_n // 2, state.grid_n // 2
    assert state.height_grid[center_ix, center_iy] > state.floor_z + 0.2
    assert state.top_soft[center_ix, center_iy]
    assert not state.top_prioritized[center_ix, center_iy]


def _container_with_chamfer_plane():
    """A container carrying the same n_vecs/points half-space data the real
    observation provides. The chamfer plane here connects local
    (x_min, floor + cut_y) to (x_min + cut_x, floor), i.e. the same wedge
    `cut_x`/`cut_y` describe, with x_min/floor computed from BASE_CONTAINER's
    raw geometry (length=2.0, thickness=0.04 -> x_min=-0.96; thickness=0.04
    -> floor=0.04)."""
    container = dict(BASE_CONTAINER)
    container["n_vecs"] = [
        (0.0, 0.0, -1.0),   # floor
        (1.0, 0.0, 0.0),    # x_max wall
        (0.0, 0.0, 1.0),    # ceiling
        (-1.0, 0.0, 0.0),   # x_min wall
        (-0.7071, 0.0, -0.7071),  # chamfer (diagonal, the one we care about)
        (0.0, -1.0, 0.0),   # door
        (0.0, 1.0, 0.0),    # back
    ]
    container["points"] = [
        (0.0, 0.0, 0.04),
        (0.96, 0.0, 0.8),
        (0.0, 0.0, 1.56),
        (-0.96, 0.0, 0.8),
        (-0.96, 0.0, 0.44),
        (0.0, -0.71, 0.8),
        (0.0, 0.71, 0.8),
    ]
    return container


def test_chamfer_plane_is_parsed_from_n_vecs_and_points():
    state = ContainerState(_container_with_chamfer_plane(), grid_n=16)
    assert state._chamfer_normal is not None
    # Picks the diagonal entry, not one of the axis-aligned walls.
    assert (abs(state._chamfer_normal) > 0.1).sum() >= 2


def test_chamfer_plane_data_stops_the_crude_full_ceiling_block():
    # Without plane data (BASE_CONTAINER), the whole band is forced to
    # ceiling_z (see test_cut_corner_keepout_raises_x_min_side_only). With
    # exact plane data available, ContainerState should leave the heightmap
    # alone here -- the exact check happens later, in packing.best_position.
    state = ContainerState(_container_with_chamfer_plane(), grid_n=16)
    assert (state.height_grid[0, :] == state.floor_z).all()
    # The risk-scoring keepout flag is unrelated and still applies.
    assert state.corner_keepout[0, :].all()


def test_shelf_container_lowers_ceiling_on_the_back_half_only():
    # The shelf plank only occupies the back half of the depth (local y >= 0,
    # see Container._create_shelf); the door-side half keeps full height so
    # a shelf-equipped container isn't treated as having half the capacity
    # of an otherwise-identical one.
    with_shelf = dict(BASE_CONTAINER, shelf=True)
    without_shelf = dict(BASE_CONTAINER, shelf=False)
    s1 = ContainerState(with_shelf, grid_n=8)
    s2 = ContainerState(without_shelf, grid_n=8)
    assert s1.ceiling_z == s2.ceiling_z
    back_half = s1.ceiling_grid[:, s1.grid_n // 2:]
    front_half = s1.ceiling_grid[:, :s1.grid_n // 2]
    assert (back_half < s1.ceiling_z).all()
    assert (front_half == s1.ceiling_z).all()
    assert (s2.ceiling_grid == s2.ceiling_z).all()
