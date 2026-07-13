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

# Candidate 5: a completely different archetype rather than a tweak of the
# reference deck. Riolu (Basic, 70 HP -- also a Buddy-Buddy Poffin target)
# into Mega Lucario ex (one evolution stage, same depth as the reference
# deck's Snover line, 340 HP). Both of Mega Lucario ex's attacks are
# single-Fighting-type-only (Aura Jab: 1 Energy/130 dmg, and it recycles
# discarded Energy onto the bench itself; Mega Brave: 2 Energy/270 dmg),
# so unlike Hammer-lanche this deck has no payoff for running a huge energy
# count -- Basic Energy is trimmed to a more conventional ~40% and the
# difference spent on Buddy-Buddy Poffin (searches 2 Basic <=70 HP Pokemon
# straight onto the bench -- a direct answer to the "empty bench, KO'd
# active, instant loss" pattern seen repeatedly in replay review) plus
# Boss's Orders / Judge for disruption.
MEGA_LUCARIO_EX = (
    [333] * 4 + [678] * 4
    + [1086] * 4 + [1121] * 4 + [1219] * 4 + [1227] * 4 + [1182] * 4 + [1213] * 4 + [1163] * 4
    + [6] * 24
)

# Candidate 6: same Riolu -> Mega Lucario ex core as candidate 5, unchanged --
# the goal here isn't a new archetype but raising how often we actually reach
# that evolution. Real-ladder replay review of 5 fresh losses with candidate 5
# showed several games where we never even got an Ultra Ball to find Riolu,
# let alone drew the evolution card once it was on the bench. Boss's Orders
# and Judge (pure disruption, no search) are swapped for two direct-search
# Supporters found by scanning the card pool for "search your deck for a
# Pokemon" text:
#   - Brock's Scouting: up to 2 Basic Pokemon OR 1 Evolution Pokemon --
#     confirmed by direct simulation (see tools/build_deck.py's git history /
#     STRATEGY_REPORT.md) that it offers Mega Lucario ex as a search target
#     despite being a megaEx card, not just a plain {ex}. Flexible enough to
#     either grab a second/third Riolu early or the evolution once Riolu is
#     already down.
#   - Cyrano: up to 3 Pokemon {ex} -- same megaEx-inclusion confirmed by
#     simulation. Only useful once a Riolu is already in play (it can't fetch
#     the Basic), but stacks extra evolution copies into hand fast.
# (Poke Pad was checked too and ruled out: its own text explicitly excludes
# any Pokemon with a Rule Box, which covers ex/megaEx, so it can never offer
# Mega Lucario ex.)
MEGA_LUCARIO_EX_V2 = (
    [333] * 4 + [678] * 4
    + [1086] * 4 + [1121] * 4 + [1219] * 4 + [1227] * 4 + [1210] * 4 + [1205] * 4 + [1163] * 4
    + [6] * 24
)

# Candidate 7 (kept as submission/deck.csv): same core as candidate 6, plus a
# second Basic Pokemon -- Farfetch'd (70 HP, matching Buddy-Buddy Poffin's
# <=70 HP search cap; Mach Cut: 1 colorless Energy for a guaranteed 30 dmg,
# no coin flip) -- to hedge against the deck's remaining failure mode:
# real-ladder loss data showed a large share of losses ending in an early
# "brick" with 0 Pokemon on the bench, traced to the deck running only 8
# Pokemon total (4 Riolu, 4 Mega Lucario ex).
#
# Two earlier attempts at this exact idea (a different insurance Basic,
# Sawk, and this same Farfetch'd swapped for Cyrano or Powerglass) were tried
# and rejected -- see STRATEGY_REPORT.md Section 6 for all 5 rejected
# variants. What made this one finally work: submission/main.py's
# card_value() scored every Basic Pokemon by raw HP alone, so a same-HP
# rival Basic tied with Riolu in search/discard comparisons, diverting some
# search hits away from the one card that actually opens the evolution line.
# OWN_DECK_EVOLUTION_BASES (computed dynamically from CARD_DB, not hardcoded
# to "Riolu") now gives that Basic a clear value bonus, breaking the tie.
# Petrel ("search a Trainer card") was cut to make room -- the least central
# search Trainer once Buddy-Buddy Poffin/Ultra Ball/Brock's Scouting/Cyrano
# are already in the deck (cutting Cyrano or Powerglass instead was tried
# first and both measured as clear regressions -- both turned out more
# load-bearing than Petrel).
MEGA_LUCARIO_EX_V3 = (
    [333] * 4 + [678] * 4
    + [1086] * 4 + [1121] * 4 + [1227] * 4 + [1210] * 4 + [1205] * 4 + [1163] * 4 + [123] * 4
    + [6] * 24
)

CANDIDATES = {
    "kangaskhan_swarm": KANGASKHAN_SWARM,
    "grimmsnarl_ex": GRIMMSNARL_EX,
    "reference_abomasnow": REFERENCE_DECK,
    "reference_abomasnow_lean": REFERENCE_DECK_LEAN,
    "mega_lucario_ex": MEGA_LUCARIO_EX,
    "mega_lucario_ex_v2": MEGA_LUCARIO_EX_V2,
    "mega_lucario_ex_v3": MEGA_LUCARIO_EX_V3,
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
