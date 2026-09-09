from gh_baggage_core.offline_planner import plan_order


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


def test_plan_order_returns_a_full_valid_permutation():
    items = [make_item(i, 0.4 + 0.02 * (i % 5), 0.3, 0.2, is_soft=(i % 4 == 0)) for i in range(25)]
    order = plan_order([dict(BASE_CONTAINER)], items)
    assert order is not None
    assert sorted(order) == list(range(len(items)))


def test_plan_order_none_without_container_info():
    items = [make_item(0, 0.4, 0.3, 0.2)]
    assert plan_order([], items) is None
    assert plan_order(None, items) is None


def test_plan_order_two_containers_uses_both():
    containers = [
        dict(BASE_CONTAINER, index=0, center=(0.0, 0.0, 0.805)),
        dict(BASE_CONTAINER, index=1, center=(2.5, 0.0, 0.805)),
    ]
    items = [make_item(i, 0.5, 0.4, 0.3) for i in range(20)]
    order = plan_order(containers, items)
    assert order is not None
    assert sorted(order) == list(range(len(items)))
