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
    """Games where the published board's raw model is well off Pinnacle's current line: the
    model's value is early in the week (README "Against opening lines"). New alerts (a side
    not alerted before) are appended to data/market/alerts.csv and written as a Markdown
    table to --out for the Action to post."""
    thr = {"nfl": (2.0, 3.0), "cfb": (3.0, 3.0)}      # spread, total
    out = []
    for lg, path in (("nfl", C.DOCS / "data.json"), ("cfb", C.DOCS / "cfb" / "data.json")):
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        for g in (d.get("slate") or {}).get("games") or []:
            q = (g.get("books") or {}).get(M.SHARP) or {}
            raw = g.get("model") or {}
            if not q or not raw:
                continue
            label = f"{g['away']} @ {g['home']} ({g['date']})"
            if "spread_line" in q and raw.get("margin") is not None and abs(raw["margin"] - q["spread_line"]) >= thr[lg][0]:
                home = raw["margin"] > q["spread_line"]
                out.append({"league": lg, "game": g["id"], "label": label, "market": "spread",
                            "side": f"{g['home']} {-q['spread_line']:+g}" if home else f"{g['away']} {q['spread_line']:+g}",
                            "model": round(raw["margin"], 1), "pinnacle": q["spread_line"], "gap": round(abs(raw["margin"] - q["spread_line"]), 1)})
            if "total_line" in q and raw.get("total") is not None and abs(raw["total"] - q["total_line"]) >= thr[lg][1]:
                over = raw["total"] > q["total_line"]
                out.append({"league": lg, "game": g["id"], "label": label, "market": "total",
                            "side": f"{'Over' if over else 'Under'} {q['total_line']:g}",
                            "model": round(raw["total"], 1), "pinnacle": q["total_line"], "gap": round(abs(raw["total"] - q["total_line"]), 1)})
    if not out:
        print("no alerts")
        return 0
    df = pd.DataFrame(out).sort_values(["league", "gap"], ascending=[True, False])
    path = C.DATA / "market" / "alerts.csv"
    seen = set()
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        seen = set(zip(old["league"], old["game"].astype(str), old["market"], old["side"]))
    new = df[[(r.league, str(r.game), r.market, r.side) not in seen for r in df.itertuples()]]
    print(df.to_string(index=False))
    if new.empty:
        print("no new alerts")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new.assign(at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")).to_csv(path, mode="a", header=not path.exists(), index=False)
    if args.out:
        lines = ["| League | Game | Side | Raw model | Pinnacle | Gap |", "|---|---|---|---|---|---|"]
        for r in new.itertuples():
            lines.append(f"| {r.league.upper()} | {r.label} | **{r.side}** | {r.model} | {r.pinnacle} | {r.gap} |")
        Path(args.out).write_text("\n".join(lines) + "\n\nRaw model vs Pinnacle's current line, 2+ points on NFL spreads, 3+ on college spreads and all totals. "
                                  "The model's record against openers is in the README; this is a lead to check, not a bet to take blind.\n")
    print(f"{len(new)} new alerts")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["snapshot", "status", "alerts"])
    p.add_argument("--league", choices=list(M.SPORTS))
    p.add_argument("--fixture")
    p.add_argument("--out", help="alerts: write new alerts as a Markdown table here")
    a = p.parse_args(argv)
    return {"snapshot": cmd_snapshot, "status": cmd_status, "alerts": cmd_alerts}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
