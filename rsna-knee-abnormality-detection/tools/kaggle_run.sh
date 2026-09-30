#!/usr/bin/env bash
# Push notebooks/01_baseline.ipynb to Kaggle and run it there. Needs a configured kaggle CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=${KAGGLE_USERNAME:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")}
cp notebooks/01_baseline.ipynb kaggle_kernel/01_baseline.ipynb
sed -i "s#USERNAME/#${USER_NAME}/#" kaggle_kernel/kernel-metadata.json
kaggle kernels push -p kaggle_kernel
ID="${USER_NAME}/rsna-knee-baseline"
# Poll until the kernel finishes (max ~60 min).
for i in $(seq 1 120); do
  s=$(kaggle kernels status "$ID" 2>&1 || true); echo "$s"
  case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "kernel failed"; break;; esac
  sleep 30
done
mkdir -p out && kaggle kernels output "$ID" -p out || true
ls -la out
