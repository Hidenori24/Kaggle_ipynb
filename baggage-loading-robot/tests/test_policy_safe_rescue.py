"""Tests for the narrow single-item safe-placement rescue in policy.Agent._act
(see the comment above the rescue block in policy.py).

The item the ordinary search + lookahead already chose is never reconsidered
-- only *where* that same item lands can change, and only when its first
choice was flagged risky. These tests drive that logic directly by
monkeypatching rank_placements, since the interesting behavior is entirely
about how policy.py reacts to the two calls it makes (the ordinary one, then
the require_safe=True rescue), not the geometry search itself (already
covered in test_packing.py).
"""
import gh_baggage_core.policy as policy_module
from gh_baggage_core.policy import Policy

BASE_CONTAINER = {
    "index": 0, "length": 2.0, "width": 1.5, "height": 1.6, "thickness": 0.04,
    "cut_x": 0.4, "cut_y": 0.4, "center": (0.0, 0.0, 0.8), "shelf": False,
    "is_prioritized": False, "packed_items": [],
}


def make_item(index=0, length=0.4, width=0.4, height=0.2):
    return {
        "index": index, "length": length, "width": width, "height": height,
        "is_soft": False, "is_prioritized": False, "mass": 5.0,
    }


def fake_result(x, y, z, top, unstable=False, path_blocked=False, in_corner_keepout=False):
    return {
        "x": x, "y": y, "z": z, "top": top, "flat": 0.0,
        "conflict": False, "path_blocked": path_blocked, "unstable": unstable,
        "in_corner_keepout": in_corner_keepout, "fw": 4, "fh": 4,
    }


def make_observation():
    return {"container_list": [dict(BASE_CONTAINER)], "pool_list": [make_item(0)]}


def test_rescues_the_same_item_to_a_safe_spot_when_the_first_choice_is_risky(monkeypatch):
    risky = (0.05, 0, 0, 0, fake_result(0.0, 0.0, 0.15, 0.05, unstable=True), (0.4, 0.4, 0.2))
    safe = (0.30, 0, 0, 1, fake_result(0.2, 0.2, 0.40, 0.30), (0.4, 0.4, 0.2))

    def fake_rank_placements(states, candidates, deadline, floor_waste=True, require_safe=False):
        if require_safe:
            assert len(candidates) == 1  # rescue is single-item only
            return [safe]
        return [risky]

    monkeypatch.setattr(policy_module, "rank_placements", fake_rank_placements)
    action = Policy().act(make_observation())

    assert action["item_idx"] == 0  # same item -- selection never changed
    assert action["orientation"] == 1  # took the rescue's orientation...
    assert action["place_pos"][2] == 0.40  # ...and its (higher, safe) position


def test_keeps_the_risky_placement_when_no_safe_alternative_exists(monkeypatch):
    risky = (0.05, 0, 0, 0, fake_result(0.0, 0.0, 0.15, 0.05, unstable=True), (0.4, 0.4, 0.2))

    def fake_rank_placements(states, candidates, deadline, floor_waste=True, require_safe=False):
        if require_safe:
            return []  # nothing safe anywhere for this item
        return [risky]

    monkeypatch.setattr(policy_module, "rank_placements", fake_rank_placements)
    action = Policy().act(make_observation())

    assert action["item_idx"] == 0
    assert action["orientation"] == 0
    assert action["place_pos"][2] == 0.15  # unchanged: the original risky spot


def test_never_calls_the_rescue_when_the_first_choice_is_already_safe(monkeypatch):
    safe = (0.05, 0, 0, 0, fake_result(0.0, 0.0, 0.15, 0.05), (0.4, 0.4, 0.2))

    def fake_rank_placements(states, candidates, deadline, floor_waste=True, require_safe=False):
        if require_safe:
            raise AssertionError("rescue should never be attempted when nothing is flagged risky")
        return [safe]

    monkeypatch.setattr(policy_module, "rank_placements", fake_rank_placements)
    action = Policy().act(make_observation())

    assert action["item_idx"] == 0
    assert action["place_pos"][2] == 0.15
