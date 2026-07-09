#!/usr/bin/env bash
# Check submission status and leaderboard position. Requires the same
# ~/.kaggle/kaggle.json setup as tools/kaggle_submit.sh.
set -euo pipefail

COMPETITION="pokemon-tcg-ai-battle"

command -v kaggle >/dev/null || { echo "kaggle CLI not found. Run: pip install kaggle" >&2; exit 1; }

echo "=== Your submissions ($COMPETITION) ==="
kaggle competitions submissions -c "$COMPETITION"

echo
echo "=== Leaderboard (top) ==="
kaggle competitions leaderboard -c "$COMPETITION" --show
