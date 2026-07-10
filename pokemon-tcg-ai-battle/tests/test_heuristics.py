"""Unit tests for the pure `obs -> score/bool` helpers in `submission/main.py`.

`tests/test_policy.py` deliberately only exercises the real engine end-to-end
(never crashes, never illegal) -- these tests instead feed small synthetic
`obs` dicts directly to check specific branches of the scoring logic that are
rare enough in real self-play that waiting to observe them in a replay would
be slow and unreliable (e.g. "bench has exactly one Pokemon", "this exact
attack would be lethal"). They still use the real CARD_DB/ATTACK_DB loaded
from the bundled native engine -- only the board-state `obs` is synthetic.
"""

import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PATH = os.path.join(ROOT, "submission", "main.py")


@pytest.fixture(scope="module")
def sub():
    spec = importlib.util.spec_from_file_location("submission_main_heuristics", MAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _obs(active=None, bench=None, hand=None, discard=None, opp_active=None, opp_bench=None, your_index=0):
    """Minimal synthetic `obs["current"]` with just the fields the helpers
    under test actually read."""
    me = {"active": [active] if active else [], "bench": bench or [], "hand": hand or [], "discard": discard or []}
    opp = {"active": [opp_active] if opp_active else [], "bench": opp_bench or []}
    players = [me, opp] if your_index == 0 else [opp, me]
    return {"current": {"yourIndex": your_index, "players": players}}


# --- bench_is_thin -----------------------------------------------------------

def test_bench_is_thin_true_when_empty(sub):
    assert sub.bench_is_thin(_obs(bench=[])) is True


def test_bench_is_thin_true_at_one(sub):
    assert sub.bench_is_thin(_obs(bench=[{"id": 333}])) is True


def test_bench_is_thin_false_at_two(sub):
    assert sub.bench_is_thin(_obs(bench=[{"id": 333}, {"id": 333}])) is False


def test_bench_is_thin_false_when_no_current(sub):
    assert sub.bench_is_thin({"current": None}) is False


# --- active_in_danger ----------------------------------------------------

def test_active_in_danger_true_below_35_percent(sub):
    assert sub.active_in_danger(_obs(active={"hp": 30, "maxHp": 100})) is True


def test_active_in_danger_false_above_35_percent(sub):
    assert sub.active_in_danger(_obs(active={"hp": 50, "maxHp": 100})) is False


def test_active_in_danger_false_when_no_active(sub):
    assert sub.active_in_danger(_obs(active=None)) is False


# --- hand_has_pokemon ------------------------------------------------------

def test_hand_has_pokemon_true_for_basic(sub):
    # Riolu (333) is a Basic Pokemon in the current deck's card pool.
    assert sub.hand_has_pokemon(_obs(hand=[{"id": 333}])) is True


def test_hand_has_pokemon_false_for_evolution_only(sub):
    # Mega Lucario ex (678) evolves from Riolu -- it can't be played onto an
    # empty bench by itself, so a hand containing only it must not count.
    assert sub.hand_has_pokemon(_obs(hand=[{"id": 678}])) is False


def test_hand_has_pokemon_false_for_empty_hand(sub):
    assert sub.hand_has_pokemon(_obs(hand=[])) is False


# --- opponent_underprepared ------------------------------------------------

def test_opponent_underprepared_true_when_no_energy_no_bench(sub):
    assert sub.opponent_underprepared(_obs(opp_active={"energies": []}, opp_bench=[])) is True


def test_opponent_underprepared_false_when_opponent_has_energy(sub):
    assert sub.opponent_underprepared(_obs(opp_active={"energies": [6]}, opp_bench=[])) is False


def test_opponent_underprepared_false_when_opponent_has_bench(sub):
    assert sub.opponent_underprepared(_obs(opp_active={"energies": []}, opp_bench=[{"id": 333}])) is False


# --- attack_is_lethal -------------------------------------------------------

def test_attack_is_lethal_true_when_damage_exceeds_hp(sub):
    # Mega Brave (attackId 983): flat 270 damage.
    assert sub.attack_is_lethal(_obs(opp_active={"hp": 100, "energies": []}), 983) is True


def test_attack_is_lethal_false_when_damage_insufficient(sub):
    assert sub.attack_is_lethal(_obs(opp_active={"hp": 1000, "energies": []}), 983) is False


# --- _expected_discard_damage (Hammer-lanche-style "discard N now" text) --

def test_expected_discard_damage_matches_hammer_lanche_pattern(sub):
    text = ("Discard the top 6 cards of your deck, and this attack does 100 "
            "damage for each Basic {W} Energy card that you discarded in this way.")
    expected = 6 * sub.OWN_DECK_ENERGY_RATIO * 100
    assert sub._expected_discard_damage(text) == pytest.approx(expected)


def test_expected_discard_damage_zero_for_unrelated_text(sub):
    assert sub._expected_discard_damage("Draw a card.") == 0.0


def test_expected_discard_damage_zero_for_empty_text(sub):
    assert sub._expected_discard_damage(None) == 0.0


# --- _discard_pile_damage (Riptide-style "already in discard" text) -------

def test_discard_pile_damage_counts_actual_discard(sub):
    text = ("This attack does 20 damage for each Basic {W} Energy card in "
            "your discard pile. Then, shuffle those cards into your deck.")
    # 3 Basic Water Energy cards (id 3) plus 1 unrelated Supporter.
    obs = _obs(discard=[{"id": 3}, {"id": 3}, {"id": 3}, {"id": 1219}])
    assert sub._discard_pile_damage(obs, text) == 60.0


def test_discard_pile_damage_zero_with_empty_discard(sub):
    text = "This attack does 20 damage for each Basic {W} Energy card in your discard pile."
    assert sub._discard_pile_damage(_obs(discard=[]), text) == 0.0


def test_discard_pile_damage_none_for_unrelated_text(sub):
    # None (not 0.0) signals "this text doesn't match" so callers can fall
    # back to a different damage-estimation pattern instead of assuming zero.
    assert sub._discard_pile_damage(_obs(discard=[{"id": 3}]), "Draw a card.") is None


# --- attack_score integrates both damage-estimation patterns --------------

def test_attack_score_prefers_a_loaded_discard_pile(sub):
    # Riptide (attackId 1042) reports damage:0 in the raw AllAttack() data.
    score_empty = sub.attack_score(_obs(discard=[]), 1042)
    score_loaded = sub.attack_score(_obs(discard=[{"id": 3}, {"id": 3}, {"id": 3}]), 1042)
    assert score_loaded > score_empty


def test_attack_score_estimates_hammer_lanche_above_raw_zero(sub):
    # Hammer-lanche (attackId 1046) also reports damage:0; with our deck's
    # Energy density it should score well above a genuinely 0-damage move.
    score = sub.attack_score(_obs(), 1046)
    zero_damage_score = 60.0  # what a real 0-damage, non-lethal attack scores
    assert score > zero_damage_score
