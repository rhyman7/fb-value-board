"""Rebuild the archive from the board's own history: every committed docs/data.json,
oldest first, each treated as of the time it was generated. Run once.

    python experiments/backfill_archive.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import archive, data, players_data  # noqa: E402


def main():
    games = data.load_games()
    shas = subprocess.run(["git", "log", "--reverse", "--format=%H", "--", "docs/data.json"],
                          capture_output=True, text=True, check=True).stdout.split()
    for sha in shas:
        raw = subprocess.run(["git", "show", f"{sha}:docs/data.json"], capture_output=True, text=True)
        if raw.returncode:
            continue
        out = json.loads(raw.stdout)
        when = pd.Timestamp(out["meta"]["generated"].replace(" UTC", ""), tz="UTC")
        n = archive.snapshot(out, games, now=when)
        print(sha[:7], out["meta"]["generated"], "saved", n, "players" if out.get("players") else "")
    pg, _ = players_data.load_player_games(verbose=False)
    print("graded", archive.grade(games, pg))
    for w in archive.write_index():
        print(w)


if __name__ == "__main__":
    main()
