"""Polls Kaggle's submission-status API and reports newly-observed results
as comments on a GitHub Issue.

Kaggle has no push/webhook API for "your submission was scored" -- the only
way to find out is to poll `kaggle competitions submissions`. This script is
meant to be run on a schedule (see
.github/workflows/pokemon-tcg-kaggle-watch.yml): each run fetches the current
submission list, compares it against what's already been posted as a comment
on the tracking issue (matched by an embedded fingerprint of the row, so a
status/score change produces a new comment instead of a duplicate), and
speaks up only about what's new.

Requires `gh` (GitHub CLI, preinstalled on GitHub-hosted runners) authenticated
via GH_TOKEN, and `kaggle` authenticated via ~/.kaggle/kaggle.json -- both are
already set up by the workflow before this script runs.

Usage:
    ISSUE_NUMBER=42 python tools/kaggle_watch.py
"""

import csv
import hashlib
import io
import json
import os
import subprocess
import sys

COMPETITION = "pokemon-tcg-ai-battle"


def fingerprint(row):
    """Stable short id for a submission row, sensitive to status/score
    changes (so "pending" -> "complete" for the same submission produces a
    new fingerprint and thus a new comment) but not to column ordering."""
    key = "|".join(row.get(k, "") for k in ("fileName", "date", "description", "status", "publicScore", "privateScore"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def fetch_submissions():
    out = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv"],
        capture_output=True, text=True, check=True,
    ).stdout
    return list(csv.DictReader(io.StringIO(out)))


def fetch_reported_fingerprints(issue_number):
    out = subprocess.run(
        ["gh", "issue", "view", str(issue_number), "--json", "comments", "--jq", ".comments[].body"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {line.split("kaggle-submission:")[1].strip(" -->\n")
            for line in out.splitlines() if "<!-- kaggle-submission:" in line}


def post_comment(issue_number, row, fp):
    score = row.get("publicScore") or "(no public score yet)"
    body = (
        f"<!-- kaggle-submission: {fp} -->\n"
        f"**{row.get('date', '?')}** — status: `{row.get('status', '?')}`, "
        f"public score: `{score}`\n"
        f"> {row.get('description', '')}"
    )
    subprocess.run(["gh", "issue", "comment", str(issue_number), "--body", body], check=True)


def main():
    issue_number = os.environ.get("ISSUE_NUMBER")
    if not issue_number:
        print("ISSUE_NUMBER not set", file=sys.stderr)
        return 1

    rows = fetch_submissions()
    if not rows:
        print("No submissions found.")
        return 0

    reported = fetch_reported_fingerprints(issue_number)
    new_count = 0
    for row in reversed(rows):  # oldest first, so comments land in chronological order
        fp = fingerprint(row)
        if fp in reported:
            continue
        post_comment(issue_number, row, fp)
        new_count += 1

    print(f"{len(rows)} submissions checked, {new_count} new comment(s) posted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
