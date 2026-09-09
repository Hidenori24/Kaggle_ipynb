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


def test_shelf_container_has_lower_ceiling():
    with_shelf = dict(BASE_CONTAINER, shelf=True)
    without_shelf = dict(BASE_CONTAINER, shelf=False)
    s1 = ContainerState(with_shelf, grid_n=8)
    s2 = ContainerState(without_shelf, grid_n=8)
    assert s1.ceiling_z < s2.ceiling_z
