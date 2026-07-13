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


# --- bench_is_empty -----------------------------------------------------

def test_bench_is_empty_true_when_empty(sub):
    assert sub.bench_is_empty(_obs(bench=[])) is True


def test_bench_is_empty_false_at_one(sub):
    # Deliberately stricter than bench_is_thin (0-1): a search-target
    # override checked against this measured better in A/B testing when
    # scoped to a literally empty bench, not "1 or fewer".
    assert sub.bench_is_empty(_obs(bench=[{"id": 333}])) is False


def test_bench_is_empty_false_when_no_current(sub):
    assert sub.bench_is_empty({"current": None}) is False


# --- score_option: OPT_CARD search-target preference at an empty bench ----

def test_score_option_prefers_basic_over_evolution_when_bench_empty(sub):
    # Riolu (333, Basic) vs Mega Lucario ex (678, evolution) as two search
    # targets (e.g. Ultra Ball) with the bench empty -- Riolu must outrank
    # it, since the evolution is dead weight with no Basic in play.
    obs = _obs(active={"id": 678, "hp": 340}, bench=[], hand=[{"id": 333}, {"id": 678}])
    sel = {"context": 7}
    basic_option = {"type": 3, "area": sub.AREA_HAND, "index": 0}
    evo_option = {"type": 3, "area": sub.AREA_HAND, "index": 1}
    assert sub.score_option(obs, sel, basic_option) > sub.score_option(obs, sel, evo_option)


def test_score_option_does_not_prefer_basic_when_bench_has_one(sub):
    # Same choice, but with 1 Pokemon already on the bench -- the override
    # must not fire (bench_is_empty, not bench_is_thin).
    obs = _obs(active={"id": 678, "hp": 340}, bench=[{"id": 333}], hand=[{"id": 333}, {"id": 678}])
    sel = {"context": 7}
    basic_option = {"type": 3, "area": sub.AREA_HAND, "index": 0}
    evo_option = {"type": 3, "area": sub.AREA_HAND, "index": 1}
    assert sub.score_option(obs, sel, evo_option) > sub.score_option(obs, sel, basic_option)


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


# --- apply_weakness_resistance ----------------------------------------------
# Card 678 (Mega Lucario ex, energyType=6/Fighting) is our attacker in all of
# these. Card 24 (Team Rocket's Kangaskhan ex) is weakness=6; card 80 (Iron
# Crown ex) is resistance=6; card 333 (Riolu) is weakness=5, i.e. a neutral
# matchup for a Fighting attacker. Multiplier/reduction values (x2, -30) were
# verified empirically against the native engine (see docs/ENGINE_NOTES.md).

def test_weakness_doubles_damage(sub):
    obs = _obs(active={"id": 678}, opp_active={"id": 24, "hp": 999})
    assert sub.apply_weakness_resistance(obs, 100) == 200


def test_resistance_reduces_damage_by_30(sub):
    obs = _obs(active={"id": 678}, opp_active={"id": 80, "hp": 999})
    assert sub.apply_weakness_resistance(obs, 100) == 70


def test_resistance_floors_at_zero(sub):
    obs = _obs(active={"id": 678}, opp_active={"id": 80, "hp": 999})
    assert sub.apply_weakness_resistance(obs, 10) == 0


def test_neutral_matchup_leaves_damage_unchanged(sub):
    obs = _obs(active={"id": 678}, opp_active={"id": 333, "hp": 999})
    assert sub.apply_weakness_resistance(obs, 100) == 100


def test_weakness_resistance_unchanged_when_our_active_unknown(sub):
    # Existing attack_is_lethal tests above omit our own active entirely --
    # apply_weakness_resistance must degrade to a no-op rather than crash.
    obs = _obs(opp_active={"id": 24, "hp": 999})
    assert sub.apply_weakness_resistance(obs, 100) == 100


def test_attack_is_lethal_true_only_thanks_to_weakness(sub):
    # Aura Jab (982): flat 130 damage. 130 < 200 (not lethal by the raw
    # field) but 130*2=260 >= 200 once Kangaskhan ex's Weakness to Fighting
    # is applied -- this exact gap (missed free kills) motivated the fix.
    obs = _obs(active={"id": 678}, opp_active={"id": 24, "hp": 200, "energies": []})
    assert sub.attack_is_lethal(obs, 982) is True


def test_attack_is_lethal_false_once_resistance_applied(sub):
    # Aura Jab (982): flat 130 >= 110 looks lethal, but Iron Crown ex resists
    # Fighting for -30, leaving 100 < 110 -- not actually lethal.
    obs = _obs(active={"id": 678}, opp_active={"id": 80, "hp": 110, "energies": []})
    assert sub.attack_is_lethal(obs, 982) is False


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


# --- card_value: OWN_DECK_EVOLUTION_BASES bonus -----------------------------
# The v3 deck runs two Basic Pokemon of the same HP (Riolu 70, Farfetch'd
# 70/id 123) -- a raw-HP valuation alone would score them identically,
# letting search/discard decisions divert some hits away from Riolu (the
# only one that actually opens the evolution line) toward Farfetch'd. This
# bonus was verified via A/B testing to matter: isolating it (same deck,
# fix vs no-fix) measured 57.8% vs 42.2% over 600 games -- see
# STRATEGY_REPORT.md Section 5 for the full evidence.

def test_own_deck_evolution_bases_contains_riolu(sub):
    assert "Riolu" in sub.OWN_DECK_EVOLUTION_BASES


def test_card_value_riolu_outranks_same_hp_farfetchd(sub):
    riolu = sub.CARD_DB[333]
    farfetchd = sub.CARD_DB[123]
    assert riolu["hp"] == farfetchd["hp"] == 70  # same raw stat, so the
    # bonus is the only thing that can break the tie
    assert sub.card_value(riolu) > sub.card_value(farfetchd)


def test_card_value_farfetchd_gets_no_evolution_bonus(sub):
    # Farfetch'd isn't referenced by any other deck card's evolvesFrom, so
    # it should score at its plain HP-based value with no bonus.
    farfetchd = sub.CARD_DB[123]
    assert sub.card_value(farfetchd) == farfetchd["hp"] / 10.0
