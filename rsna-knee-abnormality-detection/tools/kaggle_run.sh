#!/usr/bin/env bash
# Push a notebook from notebooks/ to Kaggle and run it there.
# Usage: kaggle_run.sh [01_baseline|02_cnn_embed|04_finetune|05_label_diag|06_llm_labels].ipynb   (needs a configured kaggle CLI)
set -euo pipefail
cd "$(dirname "$0")/.."
NB=${1:-04_finetune.ipynb}
USER_NAME=${KAGGLE_USERNAME:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")}
SLUG="rsna-knee-$(basename "$NB" .ipynb | sed 's/^[0-9]*_//; s/_/-/g')"
GPU=false; case "$NB" in *cnn*|*finetune*|*llm*) GPU=true;; esac
rm -f kaggle_kernel/*.ipynb; cp "notebooks/$NB" "kaggle_kernel/$NB"
python3 - "$USER_NAME/$SLUG" "$SLUG" "$NB" "$GPU" <<'PY'
import json, sys
i, t, nb, gpu = sys.argv[1:]
json.dump({"id": i, "title": t, "code_file": nb, "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": gpu == "true", "enable_internet": True,
           "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [], "kernel_sources": []},
          open("kaggle_kernel/kernel-metadata.json", "w"), indent=2)
PY
PUSH_OUT=$(kaggle kernels push -p kaggle_kernel 2>&1) || true
echo "$PUSH_OUT"
if echo "$PUSH_OUT" | grep -qi "error"; then   # e.g. "Maximum batch GPU session count of 2 reached": do not read the status of an old run
  echo "kernel push failed -- nothing was started. If this says the GPU session limit was reached, stop a running session on Kaggle or wait for it to finish."
  exit 1
fi
ID="${USER_NAME}/${SLUG}"
# Poll until the kernel finishes (max ~5.8h).
FAILED=0
for i in $(seq 1 700); do
  s=$(kaggle kernels status "$ID" 2>&1 || true); echo "$s"
  case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "kernel failed"; FAILED=1; break;; esac
  sleep 30
done
mkdir -p out && kaggle kernels output "$ID" -p out || true
ls -la out
if [ "$FAILED" = 1 ]; then   # make the Actions run red and show where the notebook stopped
  echo "=== last lines of the Kaggle log (kernel: https://www.kaggle.com/code/$ID) ==="
  python3 - <<'PY' || true
import glob, json
for f in glob.glob("out/*.log"):
    try:
        rows = json.load(open(f))
        print("".join(r.get("data", "") for r in rows[-60:]))
    except Exception:
        print(open(f, errors="ignore").read()[-4000:])
PY
  exit 1
fi
