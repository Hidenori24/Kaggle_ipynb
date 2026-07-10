"""Polls Kaggle's submission/episode APIs and reports newly-observed results
as comments on a GitHub Issue.

Kaggle has no push/webhook API for "your submission was scored" -- the only
way to find out is to poll. This script is meant to be run on a schedule
(see .github/workflows/pokemon-tcg-kaggle-watch.yml) and does two things:

1. Submission-level: polls `kaggle competitions submissions` and posts a
   comment whenever a submission's status/score changes (fingerprinted so
   pending -> complete produces a new comment instead of a duplicate).
2. Episode-level (this competition is a simulation/ladder, not a single
   scored prediction -- the public score alone doesn't say much): for the
   most recent submission, polls its individual match episodes via the
   `kagglesdk` python client (not exposed by the plain `kaggle` CLI table),
   aggregates win/loss/draw counts, opponents faced, and any crash/timeout/
   invalid-action errors since the last check, and downloads the replay
   JSON for any losses so they land in this run's GitHub Actions artifact --
   the same kind of file that's been manually uploaded for replay analysis
   elsewhere in this repo, just automated.

Requires `gh` (GitHub CLI, preinstalled on GitHub-hosted runners) authenticated
via GH_TOKEN, and `kaggle` authenticated via ~/.kaggle/kaggle.json -- both are
already set up by the workflow before this script runs.

Usage:
    ISSUE_NUMBER=42 python tools/kaggle_watch.py
"""

import csv
import hashlib
import io
import os
import subprocess
import sys

COMPETITION = "pokemon-tcg-ai-battle"
REPLAY_DIR = "kaggle_replays"  # uploaded as a workflow artifact by the caller


# ---------------------------------------------------------------------------
# Submission-level status/score reporting
# ---------------------------------------------------------------------------

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


def issue_comment_bodies(issue_number):
    return subprocess.run(
        ["gh", "issue", "view", str(issue_number), "--json", "comments", "--jq", ".comments[].body"],
        capture_output=True, text=True, check=True,
    ).stdout


def fetch_reported_fingerprints(issue_number):
    out = issue_comment_bodies(issue_number)
    return {line.split("kaggle-submission:")[1].strip(" -->\n")
            for line in out.splitlines() if "<!-- kaggle-submission:" in line}


def post_status_comment(issue_number, row, fp):
    score = row.get("publicScore") or "(no public score yet)"
    body = (
        f"<!-- kaggle-submission: {fp} -->\n"
        f"**{row.get('date', '?')}** — status: `{row.get('status', '?')}`, "
        f"public score: `{score}`\n"
        f"> {row.get('description', '')}"
    )
    subprocess.run(["gh", "issue", "comment", str(issue_number), "--body", body], check=True)


def report_submission_status(issue_number, rows):
    reported = fetch_reported_fingerprints(issue_number)
    new_count = 0
    for row in reversed(rows):  # oldest first, so comments land in chronological order
        fp = fingerprint(row)
        if fp in reported:
            continue
        post_status_comment(issue_number, row, fp)
        new_count += 1
    return new_count


# ---------------------------------------------------------------------------
# Episode-level win/loss/opponent/error reporting for the latest submission
# ---------------------------------------------------------------------------

def fetch_last_reported_episode_id(issue_number, submission_ref):
    marker = f"kaggle-episodes: submission_id={submission_ref} last_episode_id="
    seen = 0
    for line in issue_comment_bodies(issue_number).splitlines():
        if marker in line:
            try:
                seen = max(seen, int(line.split(marker)[1].strip(" -->\n")))
            except ValueError:
                pass
    return seen


def summarize_episodes(api, submission_ref, since_episode_id):
    from kagglesdk.competitions.types.competition_api_service import EpisodeState

    episodes = api.competition_list_episodes(submission_ref) or []
    candidates = sorted((e for e in episodes if e.id > since_episode_id), key=lambda e: e.id)

    finished_states = {EpisodeState.COMPLETED, EpisodeState.ERRORED}
    finished = [e for e in candidates if e.state in finished_states]
    pending_ids = [e.id for e in candidates if e.state not in finished_states]

    wins = losses = draws = 0
    error_counts = {}
    opponents = {}
    loss_episode_ids = []

    for ep in finished:
        mine = next((a for a in ep.agents if a.submission_id == submission_ref), None)
        opp = next((a for a in ep.agents if a.submission_id != submission_ref), None)
        if mine is None:
            continue
        opp_name = (opp.team_name if opp else None) or "?"
        opponents[opp_name] = opponents.get(opp_name, 0) + 1

        state_name = mine.state.name
        # In production, a cleanly-finished game reports the default/unset
        # EPISODE_AGENT_STATE_UNSPECIFIED, not an explicit
        # EPISODE_AGENT_STATE_COMPLETE (confirmed against the real API --
        # the first batch of 20 live episodes were all UNSPECIFIED despite
        # having sane win/loss rewards). Only the ERROR_* states mean the
        # agent actually crashed/timed out/was disqualified.
        if "ERROR" in state_name:
            error_counts[state_name] = error_counts.get(state_name, 0) + 1

        reward = mine.reward if mine.reward is not None else 0
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
            loss_episode_ids.append(ep.id)
        else:
            draws += 1

    # Never advance the watermark past a still-pending episode, or it would
    # never get picked up on a later run once it finishes.
    max_id = max((e.id for e in finished), default=since_episode_id)
    if pending_ids:
        max_id = min(max_id, min(pending_ids) - 1)
    max_id = max(max_id, since_episode_id)

    return dict(
        new_count=len(finished), wins=wins, losses=losses, draws=draws,
        error_counts=error_counts, opponents=opponents,
        loss_episode_ids=loss_episode_ids, max_id=max_id,
    )


def download_loss_replays(api, episode_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for eid in episode_ids:
        try:
            api.competition_episode_replay(eid, path=out_dir, quiet=True)
            saved.append(eid)
        except Exception as e:
            print(f"could not download replay for episode {eid}: {e}", file=sys.stderr)
    return saved


def post_episode_summary(issue_number, submission_ref, summary, saved_replay_ids):
    if summary["new_count"] == 0:
        return False
    wld = f"{summary['wins']}W-{summary['losses']}L-{summary['draws']}D"
    opp_lines = "\n".join(
        f"  - {name}: {n}" for name, n in sorted(summary["opponents"].items(), key=lambda kv: -kv[1])
    )
    err_lines = "\n".join(f"  - {name}: {n}" for name, n in summary["error_counts"].items())
    parts = [
        f"<!-- kaggle-episodes: submission_id={submission_ref} last_episode_id={summary['max_id']} -->",
        f"**{summary['new_count']} new episode(s)** for submission `{submission_ref}`: {wld}",
    ]
    if opp_lines:
        parts.append(f"対戦相手:\n{opp_lines}")
    if err_lines:
        parts.append(f"⚠️ 正常終了しなかった試合:\n{err_lines}")
    if saved_replay_ids:
        server, repo, run_id = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
        note = f"負け試合のリプレイJSON {len(saved_replay_ids)}件をこの実行のArtifactに保存しました"
        if server and repo and run_id:
            note += f": {server}/{repo}/actions/runs/{run_id}"
        parts.append(note)
    subprocess.run(["gh", "issue", "comment", str(issue_number), "--body", "\n\n".join(parts)], check=True)
    return True


def report_episodes(issue_number, latest_submission_row):
    ref = latest_submission_row.get("ref")
    if not ref:
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    since = fetch_last_reported_episode_id(issue_number, ref)
    summary = summarize_episodes(api, int(ref), since)
    saved = download_loss_replays(api, summary["loss_episode_ids"], REPLAY_DIR) if summary["loss_episode_ids"] else []
    if post_episode_summary(issue_number, ref, summary, saved):
        print(f"posted episode summary: {summary['new_count']} new episode(s), {len(saved)} loss replay(s) saved")
    else:
        print("no new finished episodes since last check")


# ---------------------------------------------------------------------------

def main():
    issue_number = os.environ.get("ISSUE_NUMBER")
    if not issue_number:
        print("ISSUE_NUMBER not set", file=sys.stderr)
        return 1

    rows = fetch_submissions()
    if not rows:
        print("No submissions found.")
        return 0

    new_status_count = report_submission_status(issue_number, rows)
    print(f"{len(rows)} submissions checked, {new_status_count} new status comment(s) posted.")

    latest = max(rows, key=lambda r: r.get("date", ""))
    try:
        report_episodes(issue_number, latest)
    except (SystemExit, Exception) as e:
        # Episode-level detail is a nice-to-have on top of the status
        # comment above, which has already been posted by this point --
        # don't let an auth hiccup or an unexpected API shape fail the run.
        print(f"episode summary failed (non-fatal): {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
