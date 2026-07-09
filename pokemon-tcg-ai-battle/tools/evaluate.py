"""Run real self-play matches against the official kaggle-environments "cabt"
engine to measure our agent's win rate. Requires `pip install kaggle-environments==1.30.1`
(see docs/ENGINE_NOTES.md for why this works fully offline, no Kaggle login needed).

Usage:
    python tools/evaluate.py --games 40 --opponent random
    python tools/evaluate.py --games 40 --opponent first
    python tools/evaluate.py --games 40 --opponent self
"""

import argparse
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_agent(path):
    spec = importlib.util.spec_from_file_location("submission_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", choices=["random", "first", "self"], default="random")
    parser.add_argument("--agent", default=os.path.join(ROOT, "submission", "main.py"))
    args = parser.parse_args()

    from kaggle_environments import make
    from kaggle_environments.envs.cabt.cabt import random_agent, first_agent

    my_agent = load_agent(args.agent)
    opponent = {"random": random_agent, "first": first_agent, "self": my_agent}[args.opponent]

    wins = losses = draws = errors = 0
    turns_list = []
    t0 = time.time()
    for g in range(args.games):
        env = make("cabt")
        # Alternate who goes in slot 0 to cancel out any first-move asymmetry.
        if g % 2 == 0:
            result = env.run([my_agent, opponent])
            my_slot = 0
        else:
            result = env.run([opponent, my_agent])
            my_slot = 1

        final = result[-1]
        status = final[my_slot]["status"]
        reward = final[my_slot]["reward"]
        turns_list.append(len(result))

        if status not in ("DONE",):
            errors += 1
        if reward is None:
            errors += 1
        elif reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        print(f"game {g+1:3d}/{args.games}: slot={my_slot} status={status} reward={reward} steps={len(result)}")

    dt = time.time() - t0
    n = args.games
    print()
    print(f"=== vs {args.opponent} over {n} games in {dt:.1f}s ===")
    print(f"wins={wins} ({wins/n:.1%})  losses={losses} ({losses/n:.1%})  draws={draws}  crashes/errors={errors}")
    print(f"avg steps per game: {sum(turns_list)/len(turns_list):.1f}")


if __name__ == "__main__":
    sys.exit(main())
