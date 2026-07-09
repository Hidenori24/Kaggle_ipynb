"""Reproduces the deck A/B test that decided `submission/deck.csv`.

We tried two hand-built decks before keeping the sample deck that ships with
kaggle-environments. Both custom decks lost consistently in real self-play
(see notebooks/02_agent_evaluation.ipynb for the executed results with
charts). This script is kept so the comparison is reproducible and so future
deck ideas can be dropped in and A/B tested the same way.

Usage:
    python tools/build_deck.py --games 20
"""

import argparse
import time

from kaggle_environments import make
from kaggle_environments.envs.cabt.cabt import first_agent
import kaggle_environments.envs.cabt.cabt as cabtmod

# Candidate 1: every attacker is a Basic Pokemon with an all-Colorless energy
# cost, so the agent never needs evolution-sequencing or energy-color
# matching logic. Built on the hypothesis that "simpler board state = easier
# for a rule-based bot to pilot well" (a pattern reported by other teams'
# public write-ups for this competition).
KANGASKHAN_SWARM = (
    [756] * 4 + [24] * 4 + [472] * 4  # Mega Kangaskhan ex / Team Rocket's Kangaskhan ex / Kangaskhan
    + [1125] * 1 + [1121] * 4 + [1224] * 4 + [1236] * 4  # Master Ball / Ultra Ball / Cheren / Urbain
    + [3] * 35  # Basic {W} Energy (color is irrelevant, cost is all-Colorless)
)

# Candidate 2: the most prevalent real-ladder archetype at the time of
# writing ("Marnie's Grimmsnarl ex", ~38.6% meta share per public reports),
# a 3-stage Darkness-type evolution line with a cheap-for-its-power finisher
# (180 dmg for 2 energy).
GRIMMSNARL_EX = (
    [646] * 4 + [647] * 4 + [648] * 4  # Marnie's Impidimp / Morgrem / Grimmsnarl ex
    + [1125] * 1 + [1121] * 4 + [1224] * 4 + [1236] * 4
    + [7] * 35  # Basic {D} Energy
)

# Candidate 3 (kept as submission/deck.csv): the sample deck bundled with
# kaggle-environments itself -- a Water-type evolution line (Snover into
# Mega Abomasnow ex) backed by Kyogre as a second big basic attacker.
REFERENCE_DECK = list(cabtmod.deck)

# Candidate 4: same card pool as REFERENCE_DECK, but every non-energy card
# maxed out to its 4-copy limit (Secret Box stays at 1 -- it's an ACE SPEC,
# capped at one per deck by the engine) and the difference trimmed off Basic
# Energy. REFERENCE_DECK runs 33/60 (55%) energy against only 10 Pokemon --
# replay review showed games where we never saw a second Pokemon to bench
# all game, which a lower energy count / higher Pokemon-and-trainer density
# should reduce. No new cards, just recounting the same pool.
REFERENCE_DECK_LEAN = (
    [721] * 4 + [722] * 4 + [723] * 4
    + [1092] * 1 + [1121] * 4 + [1145] * 4 + [1163] * 4 + [1219] * 4 + [1227] * 4 + [1262] * 4
    + [3] * 23
)

CANDIDATES = {
    "kangaskhan_swarm": KANGASKHAN_SWARM,
    "grimmsnarl_ex": GRIMMSNARL_EX,
    "reference_abomasnow": REFERENCE_DECK,
    "reference_abomasnow_lean": REFERENCE_DECK_LEAN,
}


def make_deck_agent(deck, policy):
    """Factory returning a single-argument agent closure. Important: kaggle-environments
    inspects an agent callable's signature and will pass a second positional
    `configuration` argument if the function accepts one -- so a `def f(obs, deck=deck)`
    style default parameter silently gets clobbered. A plain closure avoids that."""
    def _agent(obs):
        return deck if obs.get("select") is None else policy(obs)
    return _agent


def check_legal(deck, opponent_deck=None):
    """A deck is only really validated by the engine accepting it at
    battle_start time (wrong length / duplicate-limit violations show up as
    `errorPlayer`). We just run one full game and check nobody gets INVALID."""
    opponent_deck = opponent_deck or REFERENCE_DECK
    env = make("cabt")
    result = env.run([make_deck_agent(opponent_deck, first_agent), make_deck_agent(deck, first_agent)])
    return result[-1][1]["status"] != "INVALID"


def compare(games, policy):
    for name, deck in CANDIDATES.items():
        assert len(deck) == 60, f"{name}: deck has {len(deck)} cards, must be 60"
        assert check_legal(deck), f"{name}: rejected as INVALID by the engine"

    print(f"All {len(CANDIDATES)} candidate decks are legal (60 cards, accepted by battle_start).\n")

    wins = {name: 0 for name in CANDIDATES}
    t0 = time.time()
    for name, deck in CANDIDATES.items():
        w = l = 0
        me = make_deck_agent(deck, policy)
        opp = make_deck_agent(REFERENCE_DECK, first_agent)
        for g in range(games):
            env = make("cabt")
            if g % 2 == 0:
                result = env.run([me, opp]); slot = 0
            else:
                result = env.run([opp, me]); slot = 1
            r = result[-1][slot]["reward"]
            w += 1 if (r or 0) > 0 else 0
            l += 1 if (r or 0) < 0 else 0
        wins[name] = w
        print(f"{name:22s} vs first_agent+reference_deck: {w}W-{l}L / {games}  ({w/games:.0%})")
    print(f"\n({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    import importlib.util
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("submission_main", os.path.join(root, "submission", "main.py"))
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)

    compare(args.games, sub.agent)
