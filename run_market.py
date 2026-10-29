"""Game lines from several books, for both boards (nflmodel/market.py).

    python run_market.py snapshot            # one snapshot of NFL and college lines (6 credits)
    python run_market.py snapshot --league nfl
    python run_market.py status              # what is stored: games, books, latest snapshot
    python run_market.py alerts              # opener alerts: model vs Pinnacle, from the published boards

Needs ODDS_API_KEY in the environment for snapshot. Rows go to data/market/<league>_<YYYY-MM>.csv
(only what changed since the last snapshot). See README "Market lines".

--fixture FILE   read a saved /sports/{sport}/odds response instead of calling the API.
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
        self.path, self.ready, self.remaining, self.used_this_run = path, True, None, 0

    def sport_odds(self, sport, params):
        return json.loads(self.path.read_text())


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
    return 0


def cmd_alerts(args) -> int:
    """Games where the published board's raw model is well off Pinnacle's current line:
    the model's value is early in the week (README "Against opening lines")."""
    out = []
    for lg, path in (("nfl", C.DOCS / "data.json"), ("cfb", C.DOCS / "cfb" / "data.json")):
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        games = (d.get("slate") or {}).get("games") or []
        if not games:
            continue
        quotes = {g["id"]: (g.get("books") or {}) for g in games}
        thr_s, thr_t = (2.0, 3.0) if lg == "nfl" else (3.0, 3.0)
        for g in games:
            q = quotes[g["id"]].get(M.SHARP) or {}
            if not q:
                continue
            raw = g.get("model") or {}
            if "spread_line" in q and raw.get("margin") is not None and abs(raw["margin"] - q["spread_line"]) >= thr_s:
                out.append({"league": lg, "game": g["id"], "market": "spread", "model": raw["margin"], "pinnacle": q["spread_line"],
                            "side": g["home"] if raw["margin"] > q["spread_line"] else g["away"]})
            if "total_line" in q and raw.get("total") is not None and abs(raw["total"] - q["total_line"]) >= thr_t:
                out.append({"league": lg, "game": g["id"], "market": "total", "model": raw["total"], "pinnacle": q["total_line"],
                            "side": "over" if raw["total"] > q["total_line"] else "under"})
    if not out:
        print("no alerts")
        return 0
    df = pd.DataFrame(out)
    print(df.to_string(index=False))
    path = C.DATA / "market" / "alerts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.assign(at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")).to_csv(path, mode="a", header=not path.exists(), index=False)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["snapshot", "status", "alerts"])
    p.add_argument("--league", choices=list(M.SPORTS))
    p.add_argument("--fixture")
    a = p.parse_args(argv)
    return {"snapshot": cmd_snapshot, "status": cmd_status, "alerts": cmd_alerts}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
