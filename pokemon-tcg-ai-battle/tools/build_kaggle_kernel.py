"""Generates a self-contained Kaggle Notebook (kernel) that runs the agent
and a self-play sanity check on Kaggle's own runtime.

Why a separate generated notebook instead of just uploading submission/main.py:
Kaggle Kernels run in an isolated environment with no access to the rest of
this git repo, so the notebook can't `import` submission/main.py by relative
path. This script inlines main.py's source and deck.csv's contents directly
into notebook cells so the result is fully self-contained and safe to push
with `kaggle kernels push`.

Usage:
    python tools/build_kaggle_kernel.py [--kaggle-username YOUR_USERNAME]

Then:
    tools/kaggle_push_kernel.sh
    tools/kaggle_kernel_status.sh
"""

import argparse
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "kaggle_kernel")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def detect_kaggle_username():
    try:
        out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "username" in line.lower():
                return line.split("-")[-1].strip()
    except Exception:
        pass
    return None


def build(kaggle_username):
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(ROOT, "submission", "main.py"), encoding="utf-8") as f:
        main_py_source = f.read()

    notebook = {
        "cells": [
            md("""# Pokemon TCG AI Battle — Agent + Self-Play Check (runs on Kaggle's runtime)

Auto-generated from [`submission/main.py`](https://github.com/) in the GitHub repo by
`tools/build_kaggle_kernel.py` — do not hand-edit the agent cell below, edit
`submission/main.py` in the repo and regenerate instead.

This notebook: (1) defines the exact same agent as the GitHub submission,
(2) installs the competition engine, (3) runs a handful of self-play games
against the engine's baseline agents right here on Kaggle's infrastructure,
so you can confirm the agent behaves the same way on Kaggle as it did in
local testing before spending one of the daily submission slots on it."""),
            code('!pip install -q "kaggle-environments==1.30.1"'),
            md("## Agent (inlined from `submission/main.py`)"),
            code(main_py_source),
            md("## Self-play sanity check"),
            code("""from kaggle_environments import make
from kaggle_environments.envs.cabt.cabt import random_agent, first_agent

def play(opponent, n=10):
    wins = losses = draws = crashes = 0
    for g in range(n):
        env = make("cabt")
        if g % 2 == 0:
            result = env.run([agent, opponent]); slot = 0
        else:
            result = env.run([opponent, agent]); slot = 1
        final = result[-1][slot]
        if final["status"] != "DONE":
            crashes += 1
            continue
        r = final["reward"]
        wins += r > 0
        losses += r < 0
        draws += r == 0
    print(f"{n} games -> {wins}W-{losses}L-{draws}D, crashes={crashes}")

print("vs random_agent:")
play(random_agent, n=10)
print("vs first_agent:")
play(first_agent, n=10)
"""),
            md("""## Next steps

- If this looks healthy (no crashes, win rate roughly matches
  `notebooks/02_agent_evaluation.ipynb` in the repo), the same `submission/main.py`
  + `submission/deck.csv` are ready to submit via `tools/kaggle_submit.sh` or
  the competition's "Submit Agent" page.
- This notebook can also be adapted into the Strategy-track write-up by
  copying in the analysis from `notebooks/01_card_pool_eda.ipynb` and
  `notebooks/02_agent_evaluation.ipynb`."""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = os.path.join(OUT_DIR, "notebook.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"wrote {nb_path}")

    username = kaggle_username or detect_kaggle_username() or "REPLACE_WITH_YOUR_KAGGLE_USERNAME"
    metadata = {
        "id": f"{username}/pokemon-tcg-ai-battle-agent",
        "title": "Pokemon TCG AI Battle Agent",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "keywords": ["pokemon", "tcg", "kaggle-environments"],
        "competition_sources": ["pokemon-tcg-ai-battle"],
    }
    meta_path = os.path.join(OUT_DIR, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {meta_path}")
    if username == "REPLACE_WITH_YOUR_KAGGLE_USERNAME":
        print("\nWARNING: could not auto-detect your Kaggle username "
              "(run `kaggle config view` after setting up ~/.kaggle/kaggle.json, "
              "or pass --kaggle-username). Edit kaggle_kernel/kernel-metadata.json's "
              "\"id\" field before pushing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-username", default=None)
    args = parser.parse_args()
    build(args.kaggle_username)
