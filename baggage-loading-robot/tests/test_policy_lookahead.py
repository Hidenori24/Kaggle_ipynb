"""Tests for the online policy's shallow lookahead (Policy._pick_with_lookahead).

Rather than trying to organically construct a full pool/container scenario
where the numerics happen to disagree with pure greedy (fragile and hard to
predict by hand), these exercise the branch-selection logic directly with
small, controlled inputs: a "cheap now, dead end next" branch should lose to
a "slightly costlier now, but leaves the rest placeable" branch.
"""
from gh_baggage_core.container_state import ContainerState
from gh_baggage_core.policy import Policy

BASE_CONTAINER = {
    "index": 0, "length": 2.0, "width": 1.5, "height": 1.6, "thickness": 0.04,
    "cut_x": 0.4, "cut_y": 0.4, "center": (0.0, 0.0, 0.8), "shelf": False,
    "is_prioritized": False, "packed_items": [],
}


def make_item(index, length=0.4, width=0.4, height=0.2, is_soft=False, is_prioritized=False):
    return {
        "index": index, "length": length, "width": width, "height": height,
        "is_soft": is_soft, "is_prioritized": is_prioritized,
    }


def fake_result(x, y, z, top):
    return {
        "x": x, "y": y, "z": z, "top": top, "flat": 0.0,
        "conflict": False, "path_blocked": False, "unstable": False,
        "in_corner_keepout": False, "fw": 4, "fh": 4,
    }


def test_lookahead_avoids_branch_that_dead_ends_the_rest_of_the_pool(monkeypatch):
    state = ContainerState(dict(BASE_CONTAINER), grid_n=24)
    candidates = [make_item(0), make_item(1)]

    cheap_branch = (0.05, 0, 0, 0, fake_result(0.0, 0.0, 0.15, 0.05), (0.4, 0.4, 0.2))
    costly_branch = (0.20, 1, 0, 0, fake_result(0.3, 0.3, 0.3, 0.20), (0.4, 0.4, 0.2))
    ranked = [cheap_branch, costly_branch]

    # Force the "cheap" branch's continuation to hit a dead end, and the
    # "costly" branch's continuation to succeed cheaply -- this is exactly
    # the situation a purely greedy pick (ranked[0] == cheap_branch) gets
    # wrong, and what the lookahead exists to catch.
    import gh_baggage_core.policy as policy_module

    def fake_choose_placement(states, sub_candidates, deadline):
        # `candidates[0]` was placed by cheap_branch; only candidates[1]'s
        # item remains in that continuation.
        remaining_index = sub_candidates[0]["index"]
        if remaining_index == 1:
            return None  # cheap branch's continuation: nothing fits
        return (0.01, 0, 0, 0, fake_result(0.5, 0.5, 0.15, 0.01), (0.2, 0.2, 0.1))

    monkeypatch.setattr(policy_module, "choose_placement", fake_choose_placement)

    chosen = Policy._pick_with_lookahead([state], candidates, ranked, deadline=float("inf"))
    assert chosen is costly_branch


def test_lookahead_falls_back_to_ranked_first_when_no_branch_beats_it(monkeypatch):
    state = ContainerState(dict(BASE_CONTAINER), grid_n=24)
    candidates = [make_item(0)]
    only_branch = (0.05, 0, 0, 0, fake_result(0.0, 0.0, 0.15, 0.05), (0.4, 0.4, 0.2))
    ranked = [only_branch]

    chosen = Policy._pick_with_lookahead([state], candidates, ranked, deadline=float("inf"))
    assert chosen is only_branch
