"""Player props, NFL: price the books' lines, log everything, grade it later.

    python run_props.py flag      # price props for games kicking off within PROPS_OPEN_DAYS
                                  # (one snapshot per game) and log every side
    python run_props.py close     # closing prices for games starting soon that have flagged bets
    python run_props.py grade     # grade finished games, closing line value, weekly summary
    python run_props.py all       # grade, then flag, then close (the daily Actions run)

Needs ODDS_API_KEY in the environment for flag and close. Reads the week's player
projections from docs/data.json and the fitted distributions from data/prop_params.json
(written by experiments/props_calibration.py). See PROPS.md.

--fixture DIR   use saved event-odds JSON files (event_<id>.json) and events.json from DIR
                instead of the API (for tests and dry runs; logs nothing unless --log).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd

from nflmodel import config as C, oddsapi, propmarket as PM, proplog as PLOG
from nflmodel.props import PropModel

OPEN_DAYS = float(os.environ.get("PROPS_OPEN_DAYS", "3"))         # price a game this many days out, or nearer
CLOSE_MIN = float(os.environ.get("PROPS_CLOSE_MINUTES", "45"))    # closing snapshot within this many minutes of kickoff


class Fixture:
    """Stands in for oddsapi.Client, reading saved responses."""

    def __init__(self, d: Path):
        self.d, self.ready, self.used_this_run, self.remaining = d, True, 0, None

    def events(self):
        return json.loads((self.d / "events.json").read_text())

    def event_odds(self, event_id, markets, books=None):
        p = self.d / f"event_{event_id}.json"
        return json.loads(p.read_text()) if p.exists() else {}


def _when(s):
    return pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC")


def load_board():
    payload = json.loads((C.DOCS / "data.json").read_text())
    players = PM.board_players(payload)
    games = payload.get("slate", {}).get("games", [])
    meta = payload.get("meta", {})
    return payload, players, games, int(meta.get("season", 0)), int(meta.get("week", 0))


def markets() -> list[str]:
    m = os.environ.get("PROPS_MARKETS")
    return [x.strip() for x in m.split(",")] if m else PM.DEFAULT_MARKETS


def price_event(client, ev, mk, players, games, model, margin):
    body = client.event_odds(ev["id"], mk)
    rows = pd.DataFrame(oddsapi.flatten(body)) if body else pd.DataFrame()
    if rows.empty:
        return None, 0
    o = PM.match(rows, players, games)
    cons = PM.consensus(o, margin)
    return PM.price(cons, players, model), len(rows)


def cmd_flag(client, log_it=True, now=None):
    _, players, games, season, week = load_board()
    model = PropModel.load()
    if model is None:
        print("No data/prop_params.json: run experiments/props_calibration.py first.")
        return
    now = now or pd.Timestamp.now(tz="UTC")
    snaps = PLOG.snapshots()
    done = set(snaps[(snaps["kind"] == "flag") & (snaps["rows"].astype(int) > 0)]["event_id"])
    margin = PM.fit_one_sided_margin(PLOG.read_log())
    mk = markets()
    for ev in client.events():
        t = _when(ev["commence_time"])
        days = (t - now).total_seconds() / 86400
        if ev["id"] in done or not (0 < days <= OPEN_DAYS):
            continue
        try:
            priced, n = price_event(client, ev, mk, players, games, model, margin)
        except oddsapi.BudgetError as e:
            print(f"  {e}")
            break
        if priced is None or priced.empty:
            print(f"  {ev['away_team']} at {ev['home_team']}: no props posted yet")
            continue
        flagged = priced[priced["flagged"]]
        print(f"  {ev['away_team']} at {ev['home_team']}: {n} prices, {len(priced)} sides priced, {len(flagged)} flagged")
        for r in flagged.sort_values("edge", ascending=False).itertuples():
            print(f"    {r.player} {r.side} {r.line} {r.stat}: {r.price:+d} at {r.book}, ours {r.our_p:.3f}, "
                  f"market {r.fair:.3f}, edge {r.edge:+.3f}, EV {r.ev:+.3f}, stake {r.stake:.2%}")
        if log_it:
            PLOG.append_priced(priced, "flag", season, week)
            PLOG.record_snapshot(ev["id"], "flag", mk, len(priced))
    print(f"  credits used this run: {client.used_this_run}, left: {client.remaining}")


def cmd_close(client, log_it=True, now=None):
    _, players, games, season, week = load_board()
    model = PropModel.load()
    if model is None:
        return
    now = now or pd.Timestamp.now(tz="UTC")
    log = PLOG.read_log()
    if log.empty:
        print("  nothing flagged yet")
        return
    snaps = PLOG.snapshots()
    done = set(snaps[snaps["kind"] == "close"]["event_id"])
    flagged = log[(log["snapshot"] == "flag") & (log["flagged"] == 1)]
    margin = PM.fit_one_sided_margin(log)
    for ev in client.events():
        t = _when(ev["commence_time"])
        mins = (t - now).total_seconds() / 60
        if ev["id"] in done or not (0 < mins <= CLOSE_MIN):
            continue
        f = flagged[flagged["event_id"] == ev["id"]]
        if f.empty:
            continue
        mk = sorted(f["market"].unique())            # only the markets we bet: credits are scarce
        try:
            priced, n = price_event(client, ev, mk, players, games, model, margin)
        except oddsapi.BudgetError as e:
            print(f"  {e}")
            break
        if priced is None or priced.empty:
            continue
        print(f"  closing prices, {ev['away_team']} at {ev['home_team']}: {len(priced)} sides")
        if log_it:
            PLOG.append_priced(priced, "close", season, week)
            PLOG.record_snapshot(ev["id"], "close", mk, len(priced))
    print(f"  credits used this run: {client.used_this_run}, left: {client.remaining}")


def cmd_grade():
    n = PLOG.grade()
    s = PLOG.summarize()
    print(f"  graded {n} rows")
    if s:
        a = s["all"]
        print(f"  flagged bets so far: {a['bets']}, {a['record']}, {a['units_flat']:+.2f} units flat, "
              f"average closing line value (EV at the closing price) {a['avg_clv_ev']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["flag", "close", "grade", "all"])
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--log", action="store_true", help="with --fixture, write the log anyway")
    ap.add_argument("--now", help="pretend it is this UTC time (ISO), for dry runs")
    args = ap.parse_args()
    now = pd.Timestamp(args.now, tz="UTC") if args.now else None
    if args.fixture:
        client, log_it = Fixture(args.fixture), args.log
    else:
        client, log_it = oddsapi.Client(), True
        if args.cmd in ("flag", "close", "all") and not client.ready:
            print("ODDS_API_KEY is not set: skipping the price snapshots.")
            if args.cmd != "all":
                return 0
            client = None
    if args.cmd in ("grade", "all"):
        cmd_grade()
    if args.cmd in ("flag", "all") and client:
        cmd_flag(client, log_it, now)
    if args.cmd in ("close", "all") and client:
        cmd_close(client, log_it, now)
    return 0


if __name__ == "__main__":
    sys.exit(main())
