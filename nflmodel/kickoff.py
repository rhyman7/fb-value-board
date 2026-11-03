"""Kickoff refresh: a fresh line snapshot and a board rebuild shortly before each kickoff.

The Best bets card (bestbets.py) is graded as it stood at the last build before each game, and
the price tier is "take it now". GitHub's own schedules start late or not at all, so the
refreshes don't rely on them landing on the minute: the Kickoff refresh Action
(.github/workflows/kickoff.yml) starts well ahead, asks this module for the times, sleeps until
each one, starts a snapshot of that league (Market lines, workflow_dispatch) and then that
league's board rebuild (Update board / Update college board).

Times: games are grouped into kickoff windows (a game within WINDOW_MIN of the window's first
kickoff joins it), per league. Each window gets a refresh LEADS minutes before its first
kickoff: 35 (the card is live about 25 minutes before, time to place a bet) and 20 (as close to
the close as is still usable). Refreshes of the two leagues within MERGE_MIN of each other are
one refresh of both.

    python -m nflmodel.kickoff [--horizon MIN]   prints "epoch league label" per refresh still
                                                 ahead within the horizon, soonest first
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from . import config as C

LEADS = (35, 20)          # minutes before a window's first kickoff
WINDOW_MIN = 30           # games kicking off within this of the first one share its refreshes
MERGE_MIN = 5             # NFL and college refreshes this close become one refresh of both
PATHS = {"nfl": C.DOCS / "data.json", "cfb": C.DOCS / "cfb" / "data.json"}


def _parse(ko: str) -> dt.datetime:
    return dt.datetime.strptime(ko.replace("Z", ""), "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc)


def kickoffs(paths: dict | None = None) -> dict[str, list[dt.datetime]]:
    """Each league's game kickoffs (UTC) from the published boards."""
    out = {}
    for lg, p in (paths or PATHS).items():
        p = Path(p)
        if not p.exists():
            continue
        games = (json.loads(p.read_text()).get("slate") or {}).get("games") or []
        out[lg] = sorted({_parse(g["kickoff"]) for g in games if g.get("kickoff")})
    return out


def windows(times: list[dt.datetime]) -> list[dt.datetime]:
    """The first kickoff of each window."""
    firsts = []
    for t in sorted(times):
        if not firsts or (t - firsts[-1]).total_seconds() / 60 > WINDOW_MIN:
            firsts.append(t)
    return firsts


def plan(now: dt.datetime, horizon_min: float = 330, ko: dict | None = None) -> list[tuple[dt.datetime, str, str]]:
    """(when, league, label) for every refresh after now and within the horizon, soonest first.
    league is nfl, cfb or both."""
    ko = kickoffs() if ko is None else ko
    end = now + dt.timedelta(minutes=horizon_min)
    raw = []
    for lg, times in ko.items():
        for first in windows(times):
            for lead in LEADS:
                at = first - dt.timedelta(minutes=lead)
                if now < at <= end:
                    raw.append((at, lg, f"{lg} {first:%a %H:%MZ} -{lead}m"))
    raw.sort()
    out = []
    for at, lg, label in raw:
        if out and out[-1][1] != lg and out[-1][1] != "both" and (at - out[-1][0]).total_seconds() / 60 <= MERGE_MIN:
            out[-1] = (out[-1][0], "both", out[-1][2] + " + " + label)
            continue
        out.append((at, lg, label))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon", type=float, default=330, help="minutes ahead to plan (the job's time limit)")
    a = ap.parse_args(argv)
    for at, lg, label in plan(dt.datetime.now(dt.timezone.utc), a.horizon):
        print(int(at.timestamp()), lg, label.replace(" ", "_"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
