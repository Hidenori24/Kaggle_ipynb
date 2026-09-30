"""Create or version a private Kaggle Dataset from a directory.
Usage: kaggle_dataset.py <slug> <dir> <message> [--only 'glob,glob']   (needs kaggle CLI + credentials)
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

slug, src, msg = sys.argv[1:4]
only = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[4] == "--only" else ["*"]
user = os.environ.get("KAGGLE_USERNAME") or json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"]
tmp = Path(tempfile.mkdtemp())
for pat in only:
    for f in Path(src).glob(pat):
        if f.is_file():
            shutil.copy(f, tmp / f.name)
assert any(tmp.iterdir()), f"nothing to upload from {src}"
json.dump({"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}], "isPrivate": True},
          open(tmp / "dataset-metadata.json", "w"))
exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/{slug}"], capture_output=True, text=True).returncode == 0
cmd = ["kaggle", "datasets", "version", "-p", str(tmp), "-m", msg] if exists else ["kaggle", "datasets", "create", "-p", str(tmp)]
subprocess.run(cmd, check=True)
