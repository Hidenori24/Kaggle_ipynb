#!/usr/bin/env bash
# Build and push the Kaggle Notebook mirror of submission/main.py so it runs
# on Kaggle's own infrastructure (kaggle_kernel/, generated -- see
# tools/build_kaggle_kernel.py). Requires ~/.kaggle/kaggle.json (see
# tools/kaggle_submit.sh for setup).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v kaggle >/dev/null || { echo "kaggle CLI not found. Run: pip install kaggle" >&2; exit 1; }

python3 "$ROOT/tools/build_kaggle_kernel.py"

if grep -q REPLACE_WITH_YOUR_KAGGLE_USERNAME "$ROOT/kaggle_kernel/kernel-metadata.json"; then
    echo "Edit kaggle_kernel/kernel-metadata.json's \"id\" field with your Kaggle username, then re-run this script." >&2
    exit 1
fi

kaggle kernels push -p "$ROOT/kaggle_kernel"
echo "Pushed. Check status with: tools/kaggle_kernel_status.sh"
