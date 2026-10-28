"""Player props, NFL: price the books' lines, log everything, grade it later.

    python run_props.py flag      # open and pricing snapshots (bets are decided at pricing,
                                  # kickoff PROPS_OPEN_DAYS away or less), every side logged
    python run_props.py close     # closing prices, every game starting within PROPS_CLOSE_MINUTES
    python run_props.py grade     # grade finished games, closing line value, weekly summary
    python run_props.py all       # grade, then flag, then close (the daily Actions run)
    python run_props.py backfill --season 2025 --max-credits 17500 --markets player_pass_yds,...
                                  # past prices from the historical endpoint (propbackfill.py)

Needs ODDS_API_KEY in the environment for everything but grade. Reads the week's player
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

from nflmodel import config as C, oddsapi, propboard, propmarket as PM, proplog as PLOG
from nflmodel.props import PropModel

OPEN_DAYS = float(os.environ.get("PROPS_OPEN_DAYS", "3"))         # price a game this many days out, or nearer
CLOSE_MIN = float(os.environ.get("PROPS_CLOSE_MINUTES", "45"))    # closing snapshot within this many minutes of kickoff


class Fixture:
    """Stands in for oddsapi.Client, reading saved responses."""

    def __init__(self, d: Path):
        self.d, self.ready, self.used_this_run, self.remaining, self.last_cost = d, True, 0, None, 0

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
    priced = PM.price(cons, players, model)
    alt = PM.with_new(players)
    if alt is not None and not priced.empty:          # the same props at the new projections
        pn = PM.price(cons, alt, model)
        if not pn.empty:
            k = ["game_id", "player_id", "market", "line", "side"]
            pn = pn[k + ["proj", "our_p", "our_push", "edge", "ev"]].rename(columns={"proj": "proj_new", "our_p": "our_p_new", "our_push": "our_push_new", "edge": "edge_new", "ev": "ev_new"})
            stat_new = {c[: -len("_proj_new")] for c in alt.columns if c.endswith("_proj_new")}
            pn = pn[pn.merge(priced[k + ["stat"]], on=k, how="left")["stat"].isin(stat_new).to_numpy()]
            priced = priced.merge(pn, on=k, how="left")
    return priced, len(rows)


def _snap(client, ev, kind, mk, players, games, model, margin, season, week, log_it):
    """Price one event and log it as snapshot `kind`. Returns rows priced (0: nothing posted)."""
    priced, n = price_event(client, ev, mk, players, games, model, margin)
    if priced is None or priced.empty:
        print(f"  {ev['away_team']} at {ev['home_team']} ({kind}): no props posted")
        return 0
    flagged = priced[priced["flagged"]] if kind == "flag" else priced.iloc[0:0]
    print(f"  {ev['away_team']} at {ev['home_team']} ({kind}): {n} prices, {len(priced)} sides priced"
          + (f", {len(flagged)} flagged" if kind == "flag" else ""))
    for r in flagged.sort_values("edge", ascending=False).itertuples():
        print(f"    {r.player} {r.side} {r.line} {r.stat}: {r.price:+d} at {r.book}, ours {r.our_p:.3f}, "
              f"market {r.fair:.3f}, edge {r.edge:+.3f}, EV {r.ev:+.3f}, stake {r.stake:.2%}")
    if log_it:
        if kind != "flag":                       # bets are decided at the pricing snapshot only
            priced = priced.assign(flagged=False, stake=0.0)
        PLOG.append_priced(priced, kind, season, week)
        PLOG.record_snapshot(ev["id"], kind, mk, len(priced))
    return len(priced)


def _setup():
    _, players, games, season, week = load_board()
    model = PropModel.load()
    if model is None:
        print("No data/prop_params.json: run experiments/props_calibration.py first.")
    return players, games, season, week, model


def cmd_flag(client, log_it=True, now=None):
    """The daily run. Each game gets an 'open' snapshot once props are posted (kickoff more
    than PROPS_OPEN_DAYS away) and one 'flag' snapshot, where bets are decided, on the first
    run with kickoff PROPS_OPEN_DAYS away or less."""
    players, games, season, week, model = _setup()
    if model is None:
        return
    now = now or pd.Timestamp.now(tz="UTC")
    snaps = PLOG.snapshots()
    have = {k: set(snaps[(snaps["kind"] == k) & (snaps["rows"].astype(int) > 0)]["event_id"]) for k in ("open", "flag")}
    margin = PM.fit_one_sided_margin(PLOG.read_log())
    mk = markets()
    for ev in client.events():
        days = (_when(ev["commence_time"]) - now).total_seconds() / 86400
        if days <= 0 or days > 8:
            continue
        kind = "flag" if days <= OPEN_DAYS else "open"
        if ev["id"] in have[kind]:
            continue
        try:
            _snap(client, ev, kind, mk, players, games, model, margin, season, week, log_it)
        except oddsapi.BudgetError as e:
            print(f"  {e}")
            break
    print(f"  credits used this run: {client.used_this_run}, left: {client.remaining}")


def cmd_close(client, log_it=True, now=None):
    """Closing prices for every game kicking off within PROPS_CLOSE_MINUTES, all markets."""
    players, games, season, week, model = _setup()
    if model is None:
        return
    now = now or pd.Timestamp.now(tz="UTC")
    snaps = PLOG.snapshots()
    done = set(snaps[snaps["kind"] == "close"]["event_id"])
    margin = PM.fit_one_sided_margin(PLOG.read_log())
    mk = markets()
    for ev in client.events():
        mins = (_when(ev["commence_time"]) - now).total_seconds() / 60
        if ev["id"] in done or not (0 < mins <= CLOSE_MIN):
            continue
        try:
            _snap(client, ev, "close", mk, players, games, model, margin, season, week, log_it)
        except oddsapi.BudgetError as e:
            print(f"  {e}")
            break
    print(f"  credits used this run: {client.used_this_run}, left: {client.remaining}")


def cmd_backfill(client, season: int, max_credits: int, mk: list[str]):
    from nflmodel import propbackfill
    res = propbackfill.run(client, season, mk, max_credits)
    print(f"  backfill {season}: {res}")


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
    ap.add_argument("cmd", choices=["flag", "close", "grade", "all", "backfill"])
    ap.add_argument("--season", type=int)
    ap.add_argument("--max-credits", type=int, default=0)
    ap.add_argument("--markets")
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--log", action="store_true", help="with --fixture, write the log anyway")
    ap.add_argument("--now", help="pretend it is this UTC time (ISO), for dry runs")
    args = ap.parse_args()
    now = pd.Timestamp(args.now, tz="UTC") if args.now else None
    if args.fixture:
        client, log_it = Fixture(args.fixture), args.log
    else:
        client, log_it = oddsapi.Client(), True
        if args.cmd in ("flag", "close", "all", "backfill") and not client.ready:
            print("ODDS_API_KEY is not set: skipping the price snapshots.")
            if args.cmd != "all":
                return 0
            client = None
    if args.cmd == "backfill":
        if client is None or not args.season or args.max_credits <= 0:
            print("backfill needs ODDS_API_KEY, --season and --max-credits")
            return 1
        mk = [m.strip() for m in args.markets.split(",")] if args.markets else __import__("nflmodel.propbackfill", fromlist=["x"]).YARDAGE
        cmd_backfill(client, args.season, args.max_credits, mk)
        return 0
    if args.cmd in ("grade", "all"):
        cmd_grade()
    if args.cmd in ("flag", "all") and client:
        cmd_flag(client, log_it, now)
    if args.cmd in ("close", "all") and client:
        cmd_close(client, log_it, now)
    if log_it:
        ix = propboard.build()          # the board's Props section (docs/props/)
        print(f"  board props: {len(ix['weeks'])} weeks on file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
