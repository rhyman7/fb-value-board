"""Game lines from several books, for both boards (nflmodel/market.py).

    python run_market.py snapshot            # one snapshot of NFL and college lines (6 credits)
    python run_market.py snapshot --league nfl
    python run_market.py derivatives --league nfl   # team totals and first-half lines, one event at a time
                                                    # (market.DERIV_BUDGET credits at most; nfl 70, cfb 160)
    python run_market.py status              # what is stored: games, books, latest snapshot
    python run_market.py alerts --out a.md --push p.txt
                                             # line movement vs the model at Pinnacle's newest line: docs/moves.json,
                                             # docs/cfb/moves.json, and alerts for new or changed sides
    python run_market.py backfill --league nfl --season 2024 --plan
                                             # past lines (marketbackfill.py): what a run would fetch and cost, free
    python run_market.py backfill --league nfl --season 2024 --max-credits 4500
                                             # fetch them (30 credits a snapshot with three markets)

Needs ODDS_API_KEY in the environment for snapshot. Rows go to data/market/<league>_<YYYY-MM>.csv
(only what changed since the last snapshot). See README "Market lines".

--fixture FILE   read a saved /sports/{sport}/odds response instead of calling the API
                 (for derivatives: a saved list of /events/{id}/odds responses).
--budget N       derivatives: credits this run may spend (default market.DERIV_BUDGET).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from nflmodel import config as C, market as M, oddsapi


class Fixture:
    def __init__(self, path: Path):
        self.path, self.ready, self.remaining, self.used_this_run, self.last_cost = path, True, None, 0, 0

    def sport_odds(self, sport, params):
        return json.loads(self.path.read_text())

    def events(self, sport):
        return [{k: e.get(k) for k in ("id", "commence_time", "home_team", "away_team")} for e in json.loads(self.path.read_text())]

    def event_odds(self, event_id, markets, books=None, sport=None):
        ev = next((e for e in json.loads(self.path.read_text()) if e.get("id") == event_id), {})
        self.last_cost = len({m["key"] for b in ev.get("bookmakers", []) for m in b.get("markets", []) if m["key"] in markets})
        self.used_this_run += self.last_cost
        return ev


def cmd_snapshot(args) -> int:
    client = Fixture(Path(args.fixture)) if args.fixture else oddsapi.Client()
    if not client.ready:
        print("ODDS_API_KEY is not set: no snapshot taken")
        return 0
    leagues = [args.league] if args.league else list(M.SPORTS)
    bad = 0
    for lg in leagues:
        if M.snapshot(client, lg) < 0:
            bad += 1
    if getattr(client, "used_this_run", 0):
        print(f"  credits used {client.used_this_run}, remaining {client.remaining}")
    return 0 if bad < len(leagues) else 1


def cmd_derivatives(args) -> int:
    """Team totals and first-half lines for the soonest games (nflmodel/market.snapshot_derivatives)."""
    client = Fixture(Path(args.fixture)) if args.fixture else oddsapi.Client()
    if not client.ready:
        print("ODDS_API_KEY is not set: no derivatives fetched")
        return 0
    leagues = [args.league] if args.league else list(M.SPORTS)
    bad = 0
    for lg in leagues:
        if M.snapshot_derivatives(client, lg, budget=args.budget) < 0:
            bad += 1
    if getattr(client, "used_this_run", 0):
        print(f"  credits used {client.used_this_run}, remaining {client.remaining}")
    return 0 if bad < len(leagues) else 1


def cmd_alternates(args) -> int:
    """Alternate spread and total ladders at DraftKings and FanDuel (market.snapshot_alternates)."""
    client = Fixture(Path(args.fixture)) if args.fixture else oddsapi.Client()
    if not client.ready:
        print("ODDS_API_KEY is not set: no alternates fetched")
        return 0
    leagues = [args.league] if args.league else list(M.SPORTS)
    bad = 0
    for lg in leagues:
        if M.snapshot_alternates(client, lg, budget=args.budget) < 0:
            bad += 1
    if getattr(client, "used_this_run", 0):
        print(f"  credits used {client.used_this_run}, remaining {client.remaining}")
    return 0 if bad < len(leagues) else 1


def cmd_status(args) -> int:
    for lg in M.SPORTS:
        rows = M.load(lg)
        if rows.empty:
            print(f"{lg}: nothing stored")
            continue
        last = rows["snapshot"].max()
        n_games = rows["event_id"].nunique()
        books = ", ".join(f"{b} {n}" for b, n in rows[rows["snapshot"] == last].groupby("book")["event_id"].nunique().items())
        print(f"{lg}: {len(rows)} rows, {n_games} games, latest snapshot {last} ({books})")
        dv = rows[rows["market"].isin(sum(M.DERIV_MARKETS.values(), []))]
        if len(dv):
            print(f"  derivatives: {len(dv)} rows, {dv['event_id'].nunique()} games, latest {dv['snapshot'].max()}")
    return 0


GAPS_STATE = C.DATA / "market" / "gaps_state.json"     # last status of every side alerted (league|game|market|side)
GAPS_LOG = C.DATA / "market" / "gaps.csv"              # every alert sent, append-only
GAPS_COLS = ["at", "league", "game", "pick", "market", "event", "status", "model", "opener", "now", "gap_open", "left", "closed", "tease"]


def _fresh_lines(lg: str, games: list[dict]) -> tuple[dict, str | None]:
    """Pinnacle's newest stored line per board game, {id: {"spread", "total"}} home-positive,
    and the newest snapshot time among them."""
    df = pd.DataFrame([{"game_id": g["id"], "home_team": g["home"], "away_team": g["away"], "kickoff": g.get("kickoff")} for g in games])
    names = None if lg == "nfl" else {**{g.get("home_name"): g["home"] for g in games}, **{g.get("away_name"): g["away"] for g in games}}
    q = M.quotes_for(df, lg, {k: v for k, v in (names or {}).items() if k}) if len(df) else {"latest": {}}
    out = {}
    for gid, books in (q.get("latest") or {}).items():
        sh = books.get(M.SHARP) or {}
        if sh.get("spread_line") is not None or sh.get("total_line") is not None:
            out[gid] = {"spread": sh.get("spread_line"), "total": sh.get("total_line")}
    return out, M.as_of({k: {M.SHARP: v[M.SHARP]} for k, v in (q.get("latest") or {}).items() if M.SHARP in v})


def _sv(e: dict, x: float) -> str:
    """A line from the side's own view: totals as they are, spreads as the side's handicap."""
    if e["market"] == "Total":
        return f"{x:g}"
    v = -x if e["side"] == "home" else x
    return "PK" if v == 0 else f"{v:+g}"


def _tease_txt(t: dict | None, e: dict | None = None) -> str:
    if not t:
        return ""
    to = f"{'Over' if e['side'] == 'over' else 'Under'} {t['line']:g}" if e and e["market"] == "Total" else f"{t['line']:+g}"
    return f"6-pt tease to {to} ({t['prob']:.0%}{', Wong' if t['wong'] else ''}{', clears -120' if t['clears'] else ''})"


def cmd_alerts(args) -> int:
    """Line movement against the model, after each snapshot (README "Line movement"). For
    both boards: every side the model is MOVE_GAP or more off the opener, measured at
    Pinnacle's newest line (sheet.line_moves), written to docs/moves.json and
    docs/cfb/moves.json so the boards' Line movement section is as fresh as the snapshot.
    A side with a backtested edge (sheet.bucket_record) that is new, or whose status has
    changed (Live -> Partly gone -> Missed), is an alert: logged to data/market/gaps.csv,
    written as Markdown to --out (the issue comment) and as a phone push to --push (first
    line the title)."""
    from nflmodel import sheet as SH
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    state = json.loads(GAPS_STATE.read_text()) if GAPS_STATE.exists() else {}
    events, live = [], {}
    for lg, path in (("nfl", C.DOCS / "data.json"), ("cfb", C.DOCS / "cfb" / "data.json")):
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        games = [g for g in (d.get("slate") or {}).get("games") or []
                 if g.get("kickoff") and pd.Timestamp(g["kickoff"]) > pd.Timestamp(now)]
        fresh, as_of = _fresh_lines(lg, games)
        rows = SH.line_moves(games, lg, d.get("openers"), fresh)
        (path.parent / "moves.json").write_text(json.dumps({"at": stamp, "lines_as_of": as_of, "board": (d.get("meta") or {}).get("generated"),
                                                            "rows": rows}, separators=(",", ":")))
        live[lg] = [r for r in rows if r["bucket"] and r["status"] != "Missed"]
        for r in rows:
            if not r["bucket"]:
                continue
            key = f"{lg}|{r['game']}|{r['market']}|{r['side']}"
            was = state.get(key)
            if was == r["status"] or (was is None and r["status"] == "Missed"):
                continue
            events.append({"league": lg, "event": "new" if was is None else f"{was} -> {r['status']}", **r})
            state[key] = r["status"]
        print(f"{lg}: {len(rows)} sides off the opener, {len(live[lg])} live with a backtested edge; lines as of {as_of}")
    GAPS_STATE.parent.mkdir(parents=True, exist_ok=True)
    GAPS_STATE.write_text(json.dumps(state, indent=0, sort_keys=True))
    if not events:
        print("no new alerts")
        return 0
    pd.DataFrame([{"at": stamp, "league": e["league"], "game": e["game"], "pick": e["pick"], "market": e["market"], "event": e["event"],
                   "status": e["status"], "model": e["model"], "opener": e["opener"], "now": e["now"], "gap_open": e["gap_open"],
                   "left": e["left"], "closed": e["closed"], "tease": _tease_txt(e["tease"], e)} for e in events],
                 columns=GAPS_COLS).to_csv(GAPS_LOG, mode="a", header=not GAPS_LOG.exists(), index=False)
    by = {lg: [e for e in events if e["league"] == lg] for lg in ("nfl", "cfb")}

    def line(e):
        mv = "unmoved" if not e["move"] else f"moved {abs(e['move']):g} {'toward' if e['move'] > 0 else 'against'}"
        return (f"{e['pick']} ({e['away']} @ {e['home']}): model {_sv(e, e['model'])}, opened {_sv(e, e['opener'])}, now {_sv(e, e['now'])} ({mv}); "
                f"{e['left']:g} left. {e['status']}" + (f" (was {e['event'].split(' -> ')[0]})" if e["event"] != "new" else "")
                + (f". {_tease_txt(e['tease'], e)}" if e["tease"] else ""))
    if args.push:
        head = ", ".join(f"{lg.upper()} {len(xs)}" for lg, xs in by.items() if xs)
        body = []
        for lg, xs in by.items():
            for e in xs[:12]:
                body.append(f"{lg.upper()} {line(e)}")
            if len(xs) > 12:
                body.append(f"{lg.upper()}: {len(xs) - 12} more on the board")
        Path(args.push).write_text(f"Line movement: {head}\n" + "\n".join(body) + "\n")
    if args.out:
        md = []
        for lg, xs in by.items():
            if not xs:
                continue
            md += [f"**{'NFL' if lg == 'nfl' else 'College'}** ({len(live[lg])} live with a backtested edge)", "",
                   "| Side | Game | Model | Opened | Now | Left | Status | Teaser leg |", "|---|---|---|---|---|---|---|---|"]
            md += [f"| **{e['pick']}** | {e['away']} @ {e['home']} | {_sv(e, e['model'])} | {_sv(e, e['opener'])} | {_sv(e, e['now'])} | {e['left']:g} | "
                   f"{e['status']}{'' if e['event'] == 'new' else ' (was ' + e['event'].split(' -> ')[0] + ')'} | {_tease_txt(e['tease'], e)} |" for e in xs]
            md.append("")
        Path(args.out).write_text("\n".join(md) + "\nThe model off the opener by the backtested threshold, at Pinnacle's newest line (spreads home-positive). "
                                  "Live: the gap at the current line still clears it; Partly gone: the line has come part of the way; Missed: it has reached the model. "
                                  "The edge is the line moving to the model, so bet a Live side early; once it has moved, the teaser leg is the way in. cc @rhyman7\n")
    print(f"{len(events)} alerts")
    return 0


def cmd_backfill(args) -> int:
    """Past Pinnacle, DraftKings and FanDuel lines from the historical endpoint (marketbackfill.py)."""
    from nflmodel import marketbackfill as MB
    if not args.league or not args.season:
        print("backfill needs --league and --season")
        return 1
    markets = [m.strip() for m in (args.markets or ",".join(MB.MARKETS)).split(",") if m.strip()]
    kinds = tuple(k.strip() for k in (args.kinds or "open,close").split(",") if k.strip())
    est = MB.estimate(args.league, args.season, markets, args.window, kinds)
    print(f"  plan: {est}")
    if args.plan:
        return 0
    if not args.max_credits:
        print("backfill needs --max-credits (or --plan)")
        return 1
    client = oddsapi.Client()
    if not client.ready:
        print("ODDS_API_KEY is not set: nothing fetched")
        return 0
    res = MB.run(client, args.league, args.season, args.max_credits, markets, window_min=args.window, kinds=kinds)
    print(f"  backfill {args.league} {args.season}: {res}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["snapshot", "derivatives", "alternates", "status", "alerts", "backfill"])
    p.add_argument("--league", choices=list(M.SPORTS))
    p.add_argument("--fixture")
    p.add_argument("--budget", type=int, help="derivatives: credits this run may spend")
    p.add_argument("--out", help="alerts: write new alerts as a Markdown table here")
    p.add_argument("--push", help="alerts: write the phone push here (first line the title)")
    p.add_argument("--season", type=int, help="backfill: season")
    p.add_argument("--max-credits", type=int, help="backfill: most credits this run may spend")
    p.add_argument("--markets", help="backfill: comma separated (default spreads,totals,h2h)")
    p.add_argument("--kinds", help="backfill: open,close (default both)")
    p.add_argument("--window", type=int, help="backfill: minutes of kickoffs one close covers (default nfl 30, cfb 120)")
    p.add_argument("--plan", action="store_true", help="backfill: only show what would be fetched and its cost")
    a = p.parse_args(argv)
    return {"snapshot": cmd_snapshot, "derivatives": cmd_derivatives, "alternates": cmd_alternates, "status": cmd_status, "alerts": cmd_alerts,
            "backfill": cmd_backfill}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
