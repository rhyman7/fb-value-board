"""Past college player-prop prices (DraftKings and FanDuel) from The Odds API's archive, for
grading the college board's prop projections against real lines.

    python -m nflmodel.cfbpropbackfill --season 2025 --max-credits 32000 [--markets a,b,...]

One snapshot per game, ten minutes before kickoff. Weeks come from the college board's
schedule (data/cfb/cache/games_<season>.csv, regular season), newest first, so a run that hits
its credit cap leaves out the thin early-season weeks rather than the late ones. For each week
the archive's event list is read twice (1 credit each): two days before the week's first
kickoff, and at 12:00 UTC on its busiest day. Every listed event is asked for the markets; the
archive charges 10 per market returned, nothing for a game with no props.

Saved under data/props/history_cfb/<season>/: events.csv (every event seen) and
<event_id>.csv, one row per book x market x player x side x line (the NFL backfill's columns);
a header-only file is a game with no props, or one the archive answered 404 for. A file that
exists is never fetched again, so a stopped run restarts where it left off. Matching events to
the board's games happens when the history is graded (nflmodel.market.match_games).

Probe on Nov 1, 2026 (experiments/out/cfb_props_probe_2025-11-01.json): 27 of 43 games on
Saturday Nov 1, 2025 had props, 8 markets each, so about 2,200 credits a week with all eight.
"""
from __future__ import annotations

import argparse
import csv

import pandas as pd

from . import config as C, oddsapi
from .propbackfill import COLS, _iso

SPORT = "americanfootball_ncaaf"
HIST = C.DATA / "props" / "history_cfb"
MARKETS = ["player_pass_yds", "player_pass_tds", "player_rush_yds", "player_reception_yds", "player_anytime_td",
           "player_pass_yds_alternate", "player_rush_yds_alternate", "player_reception_yds_alternate"]
CLOSE_BEFORE = pd.Timedelta(minutes=10)


def weeks(season: int) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """(week, first kickoff, last kickoff, busiest day 12:00 UTC), newest week first."""
    g = pd.read_csv(C.DATA / "cfb" / "cache" / f"games_{season}.csv")
    g = g[g["postseason"] == 0].copy()
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True, errors="coerce")
    out = []
    for wk, x in g.dropna(subset=["kickoff"]).groupby("week"):
        busiest = x["kickoff"].dt.normalize().value_counts().idxmax() + pd.Timedelta(hours=12)
        out.append((int(wk), x["kickoff"].min(), x["kickoff"].max(), busiest))
    return sorted(out, key=lambda w: -w[0])


def _save(path, snapshot, ev) -> int:
    rows = oddsapi.flatten(ev) if ev else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, snapshot=snapshot))
    return len(rows)


def _events(out_dir) -> pd.DataFrame:
    p = out_dir / "events.csv"
    return pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame(columns=["event_id", "commence", "home_team", "away_team", "week"])


def run(client, season: int, max_credits: int, markets=MARKETS, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    out_dir = HIST / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    known = _events(out_dir)
    spent, fetched, empty, probe_ok = 0, 0, 0, False
    per_event = 10 * len(markets)
    for wk, first, last, busiest in weeks(season):
        listed_wk = known[known["week"] == str(wk)]
        if listed_wk.empty:   # list the week's events: two looks, 1 credit each
            seen = {}
            for at in (first - pd.Timedelta(days=2), busiest):
                if spent + 1 > max_credits:
                    return {"spent": spent, "fetched": fetched, "empty": empty, "complete": False}
                for e in client.historical_events(_iso(at), _iso(first - pd.Timedelta(hours=1)), _iso(last + pd.Timedelta(hours=1)), sport=SPORT):
                    seen[e["id"]] = {"event_id": e["id"], "commence": e["commence_time"], "home_team": e.get("home_team"),
                                     "away_team": e.get("away_team"), "week": str(wk)}
                spent += 1
            known = pd.concat([known, pd.DataFrame(list(seen.values()))], ignore_index=True).drop_duplicates("event_id")
            known.to_csv(out_dir / "events.csv", index=False)
            listed_wk = known[known["week"] == str(wk)]
            say(f"week {wk}: {len(listed_wk)} events listed")
        for e in listed_wk.itertuples():
            path = out_dir / f"{e.event_id}.csv"
            if path.exists():
                continue
            if spent + per_event > max_credits:
                say(f"  stopping: {spent} credits spent, cap {max_credits}")
                return {"spent": spent, "fetched": fetched, "empty": empty, "complete": False}
            at = pd.Timestamp(e.commence) - CLOSE_BEFORE
            try:
                ev = client.historical_event_odds(e.event_id, list(markets), _iso(at), sport=SPORT)
            except oddsapi.BudgetError as err:
                say(f"  {err}")
                return {"spent": spent, "fetched": fetched, "empty": empty, "complete": False}
            except oddsapi.ApiError as err:
                if "HTTP 404" in str(err):
                    _save(path, _iso(at), None)
                    empty += 1
                else:
                    say(f"  {e.away_team} @ {e.home_team}: {err}; skipped")
                continue
            cost = client.last_cost
            spent += cost
            if not probe_ok and cost > 0:
                if cost > per_event:
                    say(f"  first snapshot cost {cost} credits, expected at most {per_event}: stopping to check")
                    return {"spent": spent, "fetched": fetched, "empty": empty, "complete": False, "probe_cost": cost}
                probe_ok = True
            n = _save(path, (ev or {}).get("_snapshot") or "", ev)
            fetched += 1
            empty += n == 0
            say(f"  {e.away_team} @ {e.home_team}: {n} prices, {cost} credits (total {spent}, left {client.remaining})")
    return {"spent": spent, "fetched": fetched, "empty": empty, "complete": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--max-credits", type=int, required=True)
    ap.add_argument("--markets", default=",".join(MARKETS))
    a = ap.parse_args()
    c = oddsapi.Client()
    if not c.ready:
        raise SystemExit("ODDS_API_KEY is not set")
    res = run(c, a.season, a.max_credits, [m.strip() for m in a.markets.split(",") if m.strip()])
    print(f"college prop backfill {a.season}: {res}")


if __name__ == "__main__":
    main()
