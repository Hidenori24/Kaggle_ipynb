#!/usr/bin/env bash
# Check the run status of the pushed Kaggle Notebook and download its output
# (executed notebook + logs) so you can see the results without leaving the
# terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
META="$ROOT/kaggle_kernel/kernel-metadata.json"

command -v kaggle >/dev/null || { echo "kaggle CLI not found. Run: pip install kaggle" >&2; exit 1; }
[ -f "$META" ] || { echo "No $META. Run tools/build_kaggle_kernel.py / tools/kaggle_push_kernel.sh first." >&2; exit 1; }

KERNEL_ID=$(python3 -c "import json; print(json.load(open('$META'))['id'])")

echo "=== Status: $KERNEL_ID ==="
kaggle kernels status "$KERNEL_ID"

echo
echo "=== Downloading output to kaggle_kernel/output/ ==="
mkdir -p "$ROOT/kaggle_kernel/output"
kaggle kernels output "$KERNEL_ID" -p "$ROOT/kaggle_kernel/output" -o
echo "Done. See kaggle_kernel/output/ for the executed notebook and logs."
