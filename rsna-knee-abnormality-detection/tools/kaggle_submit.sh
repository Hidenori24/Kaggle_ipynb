#!/usr/bin/env bash
# Publish src as a Dataset, run 03_submit.ipynb on Kaggle (Internet OFF), then submit its output.
# Usage: kaggle_submit.sh "<submission message>"      (COMPETITION SUBMISSION -- consumes a daily slot)
set -euo pipefail
cd "$(dirname "$0")/.."
MSG=${1:?message required}
COMP=rsna-knee-abnormality-detection
USER_NAME=${KAGGLE_USERNAME:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")}
SLUG=rsna-knee-submit
python3 tools/kaggle_dataset.py rsna-knee-src src "$MSG" --only '*.py'
# the model dataset is produced by the training workflow; fail early if missing
kaggle datasets status "$USER_NAME/rsna-knee-model" >/dev/null || { echo "rsna-knee-model dataset missing: run the training workflow first"; exit 1; }
rm -f kaggle_kernel/*.ipynb; cp notebooks/03_submit.ipynb kaggle_kernel/03_submit.ipynb
python3 - "$USER_NAME" <<'PY'
import json, sys
u = sys.argv[1]
json.dump({"id": f"{u}/rsna-knee-submit", "title": "rsna-knee-submit", "code_file": "03_submit.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
           "enable_internet": False, "competition_sources": ["rsna-knee-abnormality-detection"],
           "dataset_sources": [f"{u}/rsna-knee-src", f"{u}/rsna-knee-model"], "kernel_sources": []},
          open("kaggle_kernel/kernel-metadata.json", "w"), indent=2)
PY
PUSH=$(kaggle kernels push -p kaggle_kernel 2>&1 | tee /dev/stderr)
VER=$(echo "$PUSH" | grep -oE 'ersion [0-9]+' | grep -oE '[0-9]+' | head -1)
[ -n "$VER" ] || { echo "could not parse kernel version"; exit 1; }
ID="$USER_NAME/$SLUG"
DONE=0
for i in $(seq 1 240); do
  s=$(kaggle kernels status "$ID" 2>&1 || true); echo "$s"
  case "$s" in *COMPLETE*) DONE=1; break;; *ERROR*|*CANCEL*) echo "kernel failed"; exit 1;; esac
  sleep 30
done
[ "$DONE" = 1 ] || { echo "timeout"; exit 1; }
kaggle competitions submit "$COMP" -k "$ID" -f submission.csv -v "$VER" -m "$MSG"
