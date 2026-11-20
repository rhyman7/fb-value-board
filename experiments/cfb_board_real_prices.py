"""Today's college board number at real closing prices, 2024-25 (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_board_real_prices.py

T6 of the Project doc claude/plan-batch-tests-2026-11-20.md (written before the run). A diagnostic:
nothing changes. Committed before its first run.

At consensus closing lines and a flat -110 the board's number (the market-residual margin, live
since Nov 19) picks the winning side about 54% of the time at 1+ points and 56-57% at 2+. Those
seasons are in the model's selection history and the prices are not ones anyone could bet. The
market backfill (data/market/history/cfb/2024, 2025) holds Pinnacle's, DraftKings' and FanDuel's
closes.

Bets: the board number's side when it is T+ points off Pinnacle's closing spread (T = 1, 2),
  - at the better of DraftKings' and FanDuel's closing line for that side (ties: the better price)
  - at Pinnacle's closing line and price
graded on the result at the line and price taken. 2026 is added if the backfill holds it.

Reading, fixed before the run: return above 0% at DraftKings/FanDuel prices at both thresholds on
2024-25 = a lead for the January recheck (not a reason to stake: these seasons are in the model's
selection history). At or below 0% at either = the consensus-line record does not survive real
prices.

Writes experiments/out/cfb_board_real_prices.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import pinnacle_grade as PG  # noqa: E402  (the backfill as quotes per game)
import recent_accuracy_weight_test as R  # noqa: E402  (the board's frame)
from nflmodel import market as M, marketbackfill as MB, odds as O  # noqa: E402

OUT = HERE / "out" / "cfb_board_real_prices.json"
RETAIL = ("draftkings", "fanduel")


def bet(side: int, line: float, home_odds: float, away_odds: float, result: float) -> tuple[float, int]:
    """Profit per unit and outcome (1 win, -1 loss, 0 push) for the home (side 1) or away (-1)
    side at spread `line` (positive = home favoured by that many)."""
    cover = np.sign(result - line) * side
    odds = home_odds if side > 0 else away_odds
    dec = O.american_to_decimal(odds)
    return (dec - 1 if cover > 0 else -1.0 if cover < 0 else 0.0), int(cover)


def stats(rows: list[dict], key: str) -> dict:
    x = pd.DataFrame(rows)
    if x.empty or key + "_profit" not in x:
        return {"n": 0}
    x = x[x[key + "_profit"].notna()]
    n, w, l = len(x), int((x[key + "_res"] > 0).sum()), int((x[key + "_res"] < 0).sum())
    if n < 2:
        return {"n": int(n)}
    return {"n": int(n), "w": w, "l": l, "win": w / (w + l) if w + l else None,
            "ret": float(x[key + "_profit"].mean()), "se": float(x[key + "_profit"].std(ddof=1) / np.sqrt(n))}


def main() -> None:
    lg = PG.League("cfb")
    games = lg.games()
    f = R.cfb_frame()
    f = f[f.played & f.proj_margin.notna()]
    res = {"seasons": {}}
    all_rows = []
    for season in (2024, 2025, 2026):
        if MB.load("cfb", season).empty:
            res["seasons"][season] = "no backfill for this season"
            continue
        _, close_q = PG.quotes("cfb", season, games, lg.team_names(season))
        rows = []
        for r in f[f.season == season].itertuples():
            q = close_q.get(r.game_id) or {}
            pin = q.get(M.SHARP) or {}
            if not all(k in pin for k in PG.SPREAD_KEYS) or pd.isna(pin["spread_line"]):
                continue
            d = r.proj_margin - pin["spread_line"]
            if d == 0:
                continue
            side = int(np.sign(d))
            row = {"game_id": r.game_id, "season": season, "week": r.week, "gap": abs(d),
                   "consensus_gap": abs(r.proj_margin - r.spread_line) if pd.notna(r.spread_line) else np.nan}
            row["pin_profit"], row["pin_res"] = bet(side, pin["spread_line"], pin["home_spread_odds"], pin["away_spread_odds"], r.result)
            best = None
            for b in RETAIL:
                qb = q.get(b) or {}
                if not all(k in qb for k in PG.SPREAD_KEYS) or pd.isna(qb["spread_line"]):
                    continue
                odds = qb["home_spread_odds"] if side > 0 else qb["away_spread_odds"]
                key = (-side * qb["spread_line"], O.american_to_decimal(odds))     # a better line for the side first, then the price
                if best is None or key > best[0]:
                    best = (key, qb)
            if best is not None:
                qb = best[1]
                row["retail_profit"], row["retail_res"] = bet(side, qb["spread_line"], qb["home_spread_odds"], qb["away_spread_odds"], r.result)
                row["retail_line_vs_pin"] = side * (pin["spread_line"] - qb["spread_line"])
            # the same side graded the backtest's way: the consensus line rounded to the half point at -110
            if pd.notna(r.spread_line):
                half = np.floor(r.spread_line * 2 + 0.5) / 2
                c = np.sign(r.result - half) * side
                row["cons_profit"], row["cons_res"] = (100 / 110 if c > 0 else -1.0 if c < 0 else 0.0), int(c)
            rows.append(row)
        all_rows += rows
        res["seasons"][season] = {f"{t}+": {k: stats([x for x in rows if x["gap"] >= t], k) for k in ("retail", "pin", "cons")} for t in (0, 1, 2)}
    both = [x for x in all_rows if x["season"] in (2024, 2025)]
    res["2024-25"] = {f"{t}+": {k: stats([x for x in both if x["gap"] >= t], k) for k in ("retail", "pin", "cons")} for t in (0, 1, 2)}
    x = pd.DataFrame(both)
    if len(x):
        res["retail_line_better_than_pinnacle_pts"] = float(x["retail_line_vs_pin"].mean())
    a, b = res["2024-25"]["1+"]["retail"], res["2024-25"]["2+"]["retail"]
    if a.get("ret") is not None and b.get("ret") is not None:
        res["reading"] = ("Above 0% at DraftKings/FanDuel closing prices at both 1+ and 2+ on 2024-25: a lead for the January recheck, not a reason to stake."
                          if a["ret"] > 0 and b["ret"] > 0 else
                          "At or below 0% at DraftKings/FanDuel closing prices at 1+ or 2+: the consensus-line record does not survive real prices.")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")

    def show(s):
        return "-" if s.get("ret") is None else f"{s['w']}-{s['l']} {100 * s['win']:.1f}% {100 * s['ret']:+.1f}% ± {100 * s['se']:.1f}%"
    for t, r in res["2024-25"].items():
        print(f"2024-25 {t}: DK/FD {show(r['retail'])} | Pinnacle {show(r['pin'])} | consensus -110 {show(r['cons'])}")
    for s, r in res["seasons"].items():
        if isinstance(r, dict):
            print(s, {t: show(v["retail"]) for t, v in r.items()})
        else:
            print(s, r)
    print(res.get("reading"))


if __name__ == "__main__":
    main()
