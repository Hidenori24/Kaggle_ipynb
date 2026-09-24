"""Tests for selection.rank_placements' mass-aware center-of-gravity bias.

The real evaluation's cog_score rewards the whole load's center of gravity
sitting low, weighted by each item's mass (see the simulator README: `mass`
"重心（cog_score）の計算に影響"). Since every item's own best spot already
comes out lowest-first, the only lever available at this level is *which*
item gets to claim a given low spot before something else takes it -- these
tests exercise exactly that choice.
"""
import pytest

from gh_baggage_core.container_state import ContainerState
from gh_baggage_core.selection import FLOOR_WASTE_WEIGHT, rank_placements

BASE_CONTAINER = {
    "index": 0, "length": 2.0, "width": 1.5, "height": 1.6, "thickness": 0.04,
    "cut_x": 0.0, "cut_y": 0.0, "center": (0.0, 0.0, 0.8), "shelf": False,
    "is_prioritized": False, "packed_items": [],
}


def make_item(index, mass, length=0.4, width=0.4, height=0.2):
    return {
        "index": index, "length": length, "width": width, "height": height,
        "mass": mass, "is_soft": False, "is_prioritized": False,
    }


def test_heavier_item_is_preferred_for_a_shared_low_spot():
    # Two same-sized items land at the identical (empty-floor) spot when
    # considered alone -- with equal `top`/penalties, the mass bias should
    # be what decides which one ranks first (claims the low spot this step).
    state = ContainerState(dict(BASE_CONTAINER), grid_n=24)
    heavy = make_item(0, mass=20.0)
    light = make_item(1, mass=1.0)

    ranked = rank_placements([state], [heavy, light], deadline=None)
    assert len(ranked) == 2
    best_candidate_idx = ranked[0][1]
    assert best_candidate_idx == 0  # the heavy item's candidate_idx


def test_mass_bias_does_not_override_a_genuinely_better_spot():
    # A light item with a strictly better (lower/safer) spot should still
    # win over a heavy item stuck with a worse one -- this is a tie-breaker,
    # not a license to prefer heavy items regardless of placement quality.
    state = ContainerState(dict(BASE_CONTAINER))
    # Occupy the floor everywhere except a small island, forcing anything
    # landing outside that island to stack on top of the floor-covering item.
    island = 0.15
    platform_top = state.floor_z + 0.5
    state.item_aabbs.append((
        (state.x_min, state.y_min + island, state.floor_z),
        (state.x_max, state.y_max, platform_top),
    ))
    state.item_aabbs.append((
        (state.x_min + island, state.y_min, state.floor_z),
        (state.x_max, state.y_min + island, platform_top),
    ))

    heavy_far = make_item(0, mass=50.0, length=1.0, width=1.0, height=0.1)  # can't use the island (too big)
    light_near = make_item(1, mass=1.0, length=0.1, width=0.1, height=0.05)  # fits the island easily, safe aspect ratio

    ranked = rank_placements([state], [heavy_far, light_near], deadline=None)
    assert len(ranked) == 2
    best_candidate_idx = ranked[0][1]
    assert best_candidate_idx == 1  # the light item's candidate_idx (genuinely lower top)


def test_flattest_item_is_preferred_for_the_bare_floor():
    # Nothing resting on the container floor can ever count toward
    # fill_score (see FLOOR_WASTE_WEIGHT), so the bottom layer should be
    # spent on whichever item wastes the least volume per unit of floor
    # area it covers -- that is, the flattest one. Same footprint and same
    # mass for both, so only height separates them.
    state = ContainerState(dict(BASE_CONTAINER), grid_n=24)
    flat = make_item(0, mass=5.0, length=0.4, width=0.4, height=0.1)
    tall = make_item(1, mass=5.0, length=0.4, width=0.4, height=0.5)

    ranked = rank_placements([state], [flat, tall], deadline=None)
    assert len(ranked) == 2
    assert ranked[0][1] == 0  # the flat item's candidate_idx


def test_floor_waste_penalty_applies_only_at_floor_level():
    # The same item, once landing on the bare floor and once landing on a
    # platform 0.3 higher. The elevated placement will actually count
    # toward fill_score, so it must not carry the floor penalty: the score
    # gap between the two should be the 0.3 of extra height *minus* the
    # penalty the floor placement alone pays.
    item = make_item(0, mass=5.0, length=0.4, width=0.4, height=0.2)

    on_floor = ContainerState(dict(BASE_CONTAINER))
    floor_score = rank_placements([on_floor], [item], deadline=None)[0][0]

    raised = ContainerState(dict(BASE_CONTAINER))
    platform_top = raised.floor_z + 0.3
    raised.item_aabbs.append((
        (raised.x_min, raised.y_min, raised.floor_z),
        (raised.x_max, raised.y_max, platform_top),
    ))
    raised_score = rank_placements([raised], [item], deadline=None)[0][0]

    # dh is 0.2 -- the flattest orientation of a 0.4 x 0.4 x 0.2 item.
    expected_gap = 0.3 - FLOOR_WASTE_WEIGHT * 0.2
    assert raised_score - floor_score == pytest.approx(expected_gap)
