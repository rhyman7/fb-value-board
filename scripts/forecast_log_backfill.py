"""Rebuild the forecast log (nflmodel/forecastlog.py) from the git history of the boards' data files.

    python scripts/forecast_log_backfill.py --repo /path/to/full/clone [--schedule games.csv] [--out data/forecast_log]

Every commit that changed docs/data.json (NFL) or docs/cfb/data.json (college) is a board as it
was published. Oldest first, each one is fed to forecastlog.record with source="backfill", so a
row is kept whenever a game's forecast, line, weight or quarterback changed, exactly as the live
log does. A quarterback's status is known only where that board carried a qb_flag (NFL, from
Nov 9, 2026). Kickoff times missing from older boards come from later boards, docs/archive/ and,
for NFL weeks from before the archive, the nflverse games file passed as --schedule.

One-off (run Nov 11, 2026). It refuses to run when a league's log already exists: the log is
append-only and backfill rows must come before the live ones. Needs a clone with full history
(the Actions' checkouts are shallow).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import forecastlog as F   # noqa: E402

FILES = {"nfl": ("docs/data.json", "docs/archive"), "cfb": ("docs/cfb/data.json", "docs/cfb/archive")}


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def boards(repo: str, file: str):
    """(commit, parsed board) for every commit that changed the file, oldest first."""
    for sha in git(repo, "log", "--reverse", "--format=%H", "--", file).split():
        try:
            yield sha, json.loads(git(repo, "show", f"{sha}:{file}"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--schedule", help="nflverse games.csv, for NFL kickoff times the boards and archive lack")
    ap.add_argument("--out", default=str(F.DIR))
    a = ap.parse_args()
    out = Path(a.out)
    for league, (file, arch) in FILES.items():
        hist = [(sha, d) for sha, d in boards(a.repo, file) if (d.get("slate") or {}).get("games") and (d.get("meta") or {}).get("generated")]
        kick = {}
        if league == "nfl" and a.schedule:
            import pandas as pd
            from nflmodel import archive
            sch = pd.read_csv(a.schedule, usecols=["game_id", "gameday", "gametime"])
            kick = {r.game_id: archive.kickoff(r.gameday, r.gametime).strftime("%Y-%m-%dT%H:%MZ") for r in sch.itertuples()}
        for f in sorted((Path(a.repo) / arch).glob("*_w*.json")):
            for gid, g in (json.loads(f.read_text()).get("games") or {}).items():
                if g.get("kickoff"):
                    kick[str(g.get("id") or gid)] = g["kickoff"]
        for _, d in hist:
            for g in d["slate"]["games"]:
                if g.get("kickoff"):
                    kick[str(g["id"])] = g["kickoff"]
        seasons = sorted({int(d["meta"]["season"]) for _, d in hist})
        if any(F.path(league, s, out).exists() for s in seasons):
            print(f"{league}: a log already exists, nothing written")
            continue
        n = no_ko = 0
        for _, d in hist:
            games = d["slate"]["games"]
            no_ko += sum(1 for g in games if not g.get("kickoff") and str(g.get("id")) not in kick)
            n += F.record(games, league, int(d["meta"]["season"]), d["meta"]["generated"], F.status_from_flags(games),
                          root=out, source="backfill", kickoffs=kick)
        print(f"{league}: {len(hist)} published boards, {n} rows, {no_ko} game entries skipped for want of a kickoff time")


if __name__ == "__main__":
    main()
