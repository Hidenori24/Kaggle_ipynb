"""Tests for selection.rank_placements' mass-aware center-of-gravity bias.

The real evaluation's cog_score rewards the whole load's center of gravity
sitting low, weighted by each item's mass (see the simulator README: `mass`
"重心（cog_score）の計算に影響"). Since every item's own best spot already
comes out lowest-first, the only lever available at this level is *which*
item gets to claim a given low spot before something else takes it -- these
tests exercise exactly that choice.
"""
from gh_baggage_core.container_state import ContainerState
from gh_baggage_core.selection import rank_placements

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
    state = ContainerState(dict(BASE_CONTAINER), grid_n=24)
    # Occupy the floor everywhere except a small island, forcing anything
    # landing outside that island to stack on top of the floor-covering item.
    state.height_grid[:, :] = state.floor_z + 0.5
    state.height_grid[0:4, 0:4] = state.floor_z

    heavy_far = make_item(0, mass=50.0, length=1.0, width=1.0)  # can't use the island (too big)
    light_near = make_item(1, mass=1.0, length=0.1, width=0.1)  # fits the island easily

    ranked = rank_placements([state], [heavy_far, light_near], deadline=None)
    assert len(ranked) == 2
    best_candidate_idx = ranked[0][1]
    assert best_candidate_idx == 1  # the light item's candidate_idx (genuinely lower top)
