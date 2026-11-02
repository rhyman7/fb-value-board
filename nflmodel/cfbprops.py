"""Live college player-prop prices (DraftKings and FanDuel) for the Parlays page.

    python -m nflmodel.cfbprops day        # every game kicking off in the next 40 hours
    python -m nflmodel.cfbprops kick       # games kicking off in the next 5 hours

One file per game, data/props/history_cfb_live/<season>/<event_id>.csv, in the backfill's
columns (nflmodel/cfbpropbackfill.py), rewritten at each snapshot, so the last one before
kickoff stays as the game's closing prices (2026's rungs and TD prices get graded with
experiments/cfb_props_real_lines.py once the season is over). parlays.cfb_book_legs prices the
college legs from these files.

Snapshots: the daily run takes every game kicking off within 40 hours that hasn't been fetched
in the last 12; the Saturday runs before each kickoff wave take games within 5 hours not fetched
in the last 3. About three a game. Cost: up to 8 credits a game (one per market returned, the
two books are one region), nothing for a game with no props posted: about 650-800 a week
(27 of 43 games had props on a 2025 Saturday, experiments/out/cfb_props_probe_2025-11-01.json).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt

import pandas as pd

from . import config as C, oddsapi
from .cfbpropbackfill import MARKETS, SPORT
from .propbackfill import COLS

LIVE = C.DATA / "props" / "history_cfb_live"
MODES = {"day": (dt.timedelta(hours=40), dt.timedelta(hours=12)), "kick": (dt.timedelta(hours=5), dt.timedelta(hours=3))}
MAX_RUN = 900          # credits one run may spend at most (a full Saturday is about 40 games x 8)


def _fetched(out_dir) -> dict:
    """event_id -> when it was last asked for (fetched.csv; a game with nothing posted has no rows
    to carry the time, so the log does)."""
    p = out_dir / "fetched.csv"
    if not p.exists():
        return {}
    f = pd.read_csv(p, dtype=str)
    return {r.event_id: pd.Timestamp(r.at).to_pydatetime() for r in f.itertuples()}


def _note(out_dir, eid: str, at: dt.datetime) -> None:
    p = out_dir / "fetched.csv"
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["event_id", "at"])
        w.writerow([eid, at.strftime("%Y-%m-%dT%H:%M:%SZ")])


def _write(path, rows: list[dict], snapshot: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, snapshot=snapshot))


def season_of(t: dt.datetime) -> int:
    return t.year if t.month >= 7 else t.year - 1


def run(client, mode: str, now: dt.datetime | None = None, max_credits: int = MAX_RUN, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    now = now or dt.datetime.now(dt.timezone.utc)
    ahead, fresh = MODES[mode]
    out_dir = LIVE / str(season_of(now))
    spent, fetched, posted = 0, 0, 0
    asked = _fetched(out_dir)
    evs = client.events(SPORT)
    for e in sorted(evs, key=lambda e: e["commence_time"]):
        ko = pd.Timestamp(e["commence_time"]).to_pydatetime()
        if not (now < ko <= now + ahead):
            continue
        path = out_dir / f"{e['id']}.csv"
        last = asked.get(e["id"])
        if last is not None and now - last < fresh:
            continue
        if spent + len(MARKETS) > max_credits:
            say(f"  stopping: {spent} credits this run, cap {max_credits}")
            break
        try:
            ev = client.event_odds(e["id"], list(MARKETS), sport=SPORT)
        except oddsapi.BudgetError as err:
            say(f"  {err}")
            break
        except oddsapi.ApiError as err:
            say(f"  {e['away_team']} @ {e['home_team']}: {err}; skipped")
            continue
        spent += client.last_cost
        rows = oddsapi.flatten(ev) if ev else []
        if rows or not path.exists():         # an empty answer never wipes earlier prices
            _write(path, rows, now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        _note(out_dir, e["id"], now)
        fetched += 1
        posted += bool(rows)
        say(f"  {e['away_team']} @ {e['home_team']}: {len(rows)} prices, {client.last_cost} credits")
    return {"mode": mode, "events": len(evs), "fetched": fetched, "with_props": posted, "spent": spent}


def load(season: int, now: dt.datetime | None = None) -> pd.DataFrame:
    """Every live snapshot for games that haven't kicked off yet."""
    now = now or dt.datetime.now(dt.timezone.utc)
    parts = []
    for f in sorted((LIVE / str(season)).glob("*.csv")):
        if f.name == "fetched.csv":
            continue
        x = pd.read_csv(f)
        if len(x) and pd.Timestamp(x["commence"].iloc[0]).to_pydatetime() > now:
            parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=sorted(MODES))
    ap.add_argument("--max-credits", type=int, default=MAX_RUN)
    a = ap.parse_args()
    c = oddsapi.Client()
    if not c.ready:
        print("ODDS_API_KEY is not set: no college prop prices fetched.")
        return
    print(f"college props ({a.mode}): {run(c, a.mode, max_credits=a.max_credits)}")


if __name__ == "__main__":
    main()
