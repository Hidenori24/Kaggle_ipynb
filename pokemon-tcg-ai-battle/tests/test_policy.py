"""Tests run against the REAL engine (kaggle-environments' bundled `cabt`
native library) -- not mocks. The competition rules make a crash or an
illegal action an instant loss, so "never crashes, never returns an illegal
selection" is the property that matters most and the one these tests exist
to catch a regression in.

Run with:  pip install kaggle-environments==1.30.1 pytest && pytest tests/
"""

import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PATH = os.path.join(ROOT, "submission", "main.py")
DECK_PATH = os.path.join(ROOT, "submission", "deck.csv")


@pytest.fixture(scope="module")
def sub():
    spec = importlib.util.spec_from_file_location("submission_main", MAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_card_database_loaded(sub):
    # If this is empty, every value-based heuristic silently degrades to a
    # flat tie-break -- worth failing loudly on rather than discovering via
    # a mysteriously weak agent.
    assert len(sub.CARD_DB) > 1000
    assert len(sub.ATTACK_DB) > 1000


def test_deck_file_is_legal():
    with open(DECK_PATH, encoding="utf-8") as f:
        ids = [int(line.strip()) for line in f if line.strip() and not line.strip().startswith("#")]
    assert len(ids) == 60


def test_deck_ids_exist_in_card_database(sub):
    deck = sub.load_deck()
    for card_id in deck:
        assert card_id in sub.CARD_DB, f"deck.csv references unknown cardId {card_id}"


def test_default_deck_matches_deck_csv(sub):
    # The embedded fallback deck must stay in sync with deck.csv, or a
    # corrupted/missing deck.csv at submission time silently ships a
    # different (untested) deck.
    assert sub.DEFAULT_DECK == sub.load_deck()


def test_agent_never_crashes_and_stays_legal():
    """Play several full games against both baselines, both player slots,
    and assert our agent's status is always DONE (never TIMEOUT/ERROR/INVALID)."""
    from kaggle_environments import make
    from kaggle_environments.envs.cabt.cabt import first_agent, random_agent

    spec = importlib.util.spec_from_file_location("submission_main", MAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for opponent in (random_agent, first_agent):
        for my_slot in (0, 1):
            env = make("cabt")
            agents = [mod.agent, opponent] if my_slot == 0 else [opponent, mod.agent]
            result = env.run(agents)
            final = result[-1][my_slot]
            assert final["status"] == "DONE", f"agent ended with status={final['status']!r}"
            assert final["reward"] in (-1, 0, 1)
