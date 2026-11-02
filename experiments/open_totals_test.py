"""NFL totals graded at the opening total (2024 on).

    python experiments/open_totals_test.py

The totals bets were only ever graded at the closing total, because no historical NFL opening
totals were on hand. nfelo's games file (data/raw/nfelo_games.csv, already downloaded for the
opening spreads) carries total_line_open with prices from 2024. The spread edge, if any, is in
the first number (FEATURES.md, "Where the edge lives"); this asks the same of totals.

Runs: experiments/nfl_segments.py's cached production runs (the "early" run, quarterbacks
blind, which is what is known when the opener is posted, as for the opener spreads; and the full
run for comparison). Games whose team played the Monday night before are left out, as for the
opening spreads. The total model uses the recorded wind, which at the opener would only be a
forecast, so the result is also shown without games with 10+ mph wind.

Graded per gap (model total minus the opening total): record and return at the opener's price,
the same sides at the close, and closing line value as how far the total moved toward the model
(close - open, signed by the bet; in points). Standard errors clustered by game (one bet each).

Rule, set before running (Nov 2, 2026): a lead (for the live log, not the board) only if, at
3+ points, the return at the opener is above zero AND the line moves toward the model by more
than two standard errors. The board's totals bets change only on a larger sample and Ryan's say.
Writes experiments/out/open_totals_test.json.
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
import nfl_segments as S  # noqa: E402
from nflmodel import backtest, config as C  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "open_totals_test.json"
GAPS = (1, 2, 3, 4)


def summarize(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"bets": 0}
    se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    w, l, p = int((x.res > 0).sum()), int((x.res < 0).sum()), int((x.res == 0).sum())
    return {"bets": int(len(x)), "record": f"{w}-{l}-{p}", "return": round(float(x.profit.mean()), 4),
            "return_se": round(se(x.profit), 4), "close_record": f"{int((x.res_close > 0).sum())}-{int((x.res_close < 0).sum())}",
            "close_return": round(float(x.profit_close.mean()), 4),
            "move_toward_model": round(float(x.move.mean()), 3), "move_se": round(se(x.move), 3)}


def main():
    R = S.build(False)
    games = R["games"]
    n = pd.read_csv(C.RAW / "nfelo_games.csv", usecols=["game_id", "total_line_open", "over_price_open", "under_price_open"])
    n = n[n.total_line_open.notna()]
    monday = games.game_id.map(backtest._came_off_monday(games)).fillna(False).astype(bool)
    skip = set(games.loc[monday, "game_id"])
    res = {"source": "nfelo_games.csv total_line_open", "games_with_open_total": int(len(n))}
    for run in ("early", "frame"):
        f = R[run].merge(n, on="game_id")
        f = f[f.played & f.total_line.notna() & ~f.game_id.isin(skip)].copy()
        f["gap"] = f.model_total - f.total_line_open
        f = f[f.gap != 0]
        pick = np.sign(f.gap)
        f["res"] = np.sign(f.total - f.total_line_open) * pick
        f["profit"] = S._profit(f.res, np.where(pick > 0, f.over_price_open, f.under_price_open))
        f["res_close"] = np.sign(f.total - f.total_line) * pick
        f["profit_close"] = S._profit(f.res_close, np.where(pick > 0, f.over_odds, f.under_odds))
        f["move"] = (f.total_line - f.total_line_open) * pick
        calm = (f.indoor == 1) | (f.wind.fillna(0) < 10)
        r = {"games": int(len(f)), "seasons": sorted(int(s) for s in f.season.unique()),
             "model_miss": round(float((f.model_total - f.total).abs().mean()), 3),
             "open_miss": round(float((f.total_line_open - f.total).abs().mean()), 3),
             "close_miss": round(float((f.total_line - f.total).abs().mean()), 3)}
        for g in GAPS:
            r[f"gap_{g}+"] = summarize(f[f.gap.abs() >= g])
            r[f"gap_{g}+_no_wind"] = summarize(f[(f.gap.abs() >= g) & calm])
        r["gap_3+_overs"] = summarize(f[f.gap >= 3])
        r["gap_3+_unders"] = summarize(f[f.gap <= -3])
        r3 = r["gap_3+"]
        r["lead"] = bool(r3["bets"] and r3["return"] > 0 and r3["move_toward_model"] > 2 * r3["move_se"])
        res[run] = r
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
