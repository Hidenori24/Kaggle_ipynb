#!/usr/bin/env bash
# Package submission/{main.py,deck.csv} and submit to the competition.
#
# Setup (one-time, local):
#   pip install kaggle
#   # https://www.kaggle.com/settings -> API -> Create New Token -> save as ~/.kaggle/kaggle.json
#   chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   tools/kaggle_submit.sh ["submission message"]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPETITION="pokemon-tcg-ai-battle"
MESSAGE="${1:-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo manual) $(date -u +%FT%TZ)}"
ARCHIVE="$ROOT/submission.zip"

command -v kaggle >/dev/null || { echo "kaggle CLI not found. Run: pip install kaggle" >&2; exit 1; }

echo "Packaging submission/main.py + submission/deck.csv -> $ARCHIVE"
rm -f "$ARCHIVE"
(cd "$ROOT/submission" && zip -q "$ARCHIVE" main.py deck.csv)

echo "Submitting to $COMPETITION with message: $MESSAGE"
kaggle competitions submit -c "$COMPETITION" -f "$ARCHIVE" -m "$MESSAGE"

rm -f "$ARCHIVE"
echo "Done. Check status with: tools/kaggle_status.sh"
