"""An early-season rule for college spread bets (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_early_season_rule_test.py

Plan, written before the run: Project doc claude/plan-cfb-early-season-rule-test-2026-11-20.md.
This script was committed before its first run.

Question: do the college model's flagged spread bets do worse in weeks 1-4 than from week 5 on,
consistently enough to stop flagging them early in the season? (College sides are unstaked today,
so the rule would change flags and labels, not money bets. Totals run the other way and are not
part of this.)

Bets, graded at -110 at the line rounded to the half point, as experiments/cfb_segments.py does:
  close    today's board number (proj_margin, the market-residual margin) T+ points off the
           closing spread. Tuning 2017-19 and holdout 2021-25.
  opener   the bet-time run's raw model (last week's quarterbacks, a typical wind; short-week
           games left out, as the board's opener report does) T+ points off the opening spread,
           graded at the opener. Holdout 2021-25 only: there are no openers before 2021.
Each split into weeks 1-4 and weeks 5+.

Threshold T: 2 points. The board's number moves the line by a fitted amount, so 2+ may flag few
games. Counts only (no returns) are read first: T is the first of 2, 1, 0.5 at which all three
weeks-1-4 cells (tuning close, holdout close, holdout opener) hold at least 150 bets, and it is
used for every cell. If none does, T = 0.5 and the result is reported as underpowered.

Rule, set before running. The early-season rule is adopted only if all four hold:
  1. tuning, close: return per bet in weeks 5+ is higher than in weeks 1-4
  2. holdout, close: the same
  3. holdout, opener: the same
  4. holdout, close: the gap (weeks 5+ minus weeks 1-4) is at least 1 standard error of the difference
  - All four: adopted provisionally (plan and mockup before the board changes; real test 2027).
  - Some but not all: a trade-off for Ryan.
  - (1) failing: nothing goes further, nothing changes.
  - Reported, not used: 2026 so far, win rates, weeks 5-9 against 10+, the other thresholds, and
    closing line value for the opener bets.

Writes experiments/out/cfb_early_season_rule_test.json.
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
warnings.filterwarnings("ignore")
OUT = HERE / "out" / "cfb_early_season_rule_test.json"
TUNE, HOLD = (2017, 2019), (2021, 2025)
THRESHOLDS = (2.0, 1.0, 0.5)
MIN_BETS = 150
EARLY = 4


def frames():
    """(the board's walk-forward frame, the bet-time walk-forward frame, games)."""
    from cfbmodel import data, extra, model
    from cfbmodel.pipeline import CFBRatingEngine, PFFRET
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games["wind_source"] = np.where(games["played"], "recorded", "")
    inputs = extra.load_inputs(games, refresh_current=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games, extra=inputs["rated"])
    close, _ = model.walk_forward(PFFRET.add(model.build_frame(games, engine, inputs=inputs)))
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan, wind=np.nan)
    early, _ = model.walk_forward(PFFRET.add(model.build_frame(blind, engine)))
    return close, early, games


def graded(f: pd.DataFrame, mcol: str, line: str, close_line: str | None = None) -> pd.DataFrame:
    from cfbmodel import bets as B
    from nflmodel import odds as O
    dec = O.american_to_decimal(B.DEFAULT_PRICE)
    x = f[f.played & f[line].notna() & f[mcol].notna() & (f.fbs_game == 1)].copy()
    x["d"] = x[mcol] - x[line]
    pick = np.sign(x["d"])
    x["cover"] = np.sign(x["result"] - B.half(x[line])) * pick
    x["profit"] = np.where(x.cover > 0, dec - 1, np.where(x.cover < 0, -1.0, 0.0))
    if close_line:
        x["clv"] = (x[close_line] - x[line]) * pick
    return x[x.d != 0]


def stats(y: pd.DataFrame) -> dict:
    n = int(len(y))
    w, l = int((y.cover > 0).sum()), int((y.cover < 0).sum())
    out = {"n": n, "w": w, "l": l, "win": w / (w + l) if w + l else None,
           "ret": float(y.profit.mean()) if n else None,
           "se": float(y.profit.std(ddof=1) / np.sqrt(n)) if n > 1 else None}
    if "clv" in y and n:
        out["clv"] = float(y.clv.mean())
    return out


def cell(g: pd.DataFrame, lo: int, hi: int, t: float) -> dict:
    p = g[(g.season >= lo) & (g.season <= hi) & (g.d.abs() >= t)]
    return {"wk1-4": stats(p[p.week <= EARLY]), "wk5+": stats(p[p.week > EARLY]),
            "wk5-9": stats(p[(p.week > EARLY) & (p.week <= 9)]), "wk10+": stats(p[p.week >= 10])}


def gap(c: dict) -> tuple[float | None, float | None]:
    a, b = c["wk5+"], c["wk1-4"]
    if a["ret"] is None or b["ret"] is None or a["se"] is None or b["se"] is None:
        return None, None
    return a["ret"] - b["ret"], float(np.hypot(a["se"], b["se"]))


def main() -> None:
    from cfbmodel import backtest
    close, early, games = frames()
    gc = graded(close, "proj_margin", "spread_line")
    short = backtest._short_week(games)
    e = early[~early["game_id"].map(short).fillna(False).to_numpy(bool)]
    go = graded(e[e.spread_line.notna()], "model_margin", "open_spread", close_line="spread_line")

    # counts only, to fix the threshold
    counts = {}
    T = None
    for t in THRESHOLDS:
        c = {"tune_close": cell(gc, *TUNE, t)["wk1-4"]["n"], "hold_close": cell(gc, *HOLD, t)["wk1-4"]["n"],
             "hold_open": cell(go, *HOLD, t)["wk1-4"]["n"]}
        counts[str(t)] = c
        if T is None and min(c.values()) >= MIN_BETS:
            T = t
    underpowered = T is None
    T = T if T is not None else THRESHOLDS[-1]

    res = {"threshold": T, "underpowered": underpowered, "counts_wk1_4": counts, "cells": {}}
    for t in THRESHOLDS:
        res["cells"][str(t)] = {"tune_close": cell(gc, *TUNE, t), "hold_close": cell(gc, *HOLD, t), "hold_open": cell(go, *HOLD, t),
                                "2026_close": cell(gc, 2026, 2026, t), "2026_open": cell(go, 2026, 2026, t)}
    main_cells = res["cells"][str(T)]
    g1, _ = gap(main_cells["tune_close"])
    g2, se2 = gap(main_cells["hold_close"])
    g3, _ = gap(main_cells["hold_open"])
    checks = {"1_tune_close": g1 is not None and g1 > 0, "2_hold_close": g2 is not None and g2 > 0,
              "3_hold_open": g3 is not None and g3 > 0, "4_hold_close_1se": g2 is not None and se2 and g2 >= se2}
    res["gaps"] = {"tune_close": g1, "hold_close": g2, "hold_close_se": se2, "hold_open": g3}
    res["checks"] = checks
    if not checks["1_tune_close"]:
        res["verdict"] = "Check 1 fails: on 2017-19 the early weeks did not return less than the rest. Nothing goes further; nothing changes."
    elif all(checks.values()):
        res["verdict"] = "All four checks hold: the early-season rule is adopted provisionally (plan and mockup before the board changes; real test 2027)."
    else:
        res["verdict"] = f"A trade-off: {sum(checks.values())} of 4 checks hold. Ryan's call."
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")

    def show(s):
        if not s["n"]:
            return "-"
        return f"{s['w']}-{s['l']} {100 * s['win']:.1f}% {100 * s['ret']:+.1f}% ± {100 * s['se']:.1f}%" + (f" clv {s['clv']:+.2f}" if "clv" in s else "")
    print("weeks 1-4 bet counts by threshold:", counts, "-> T =", T, "(underpowered)" if underpowered else "")
    for name, c in main_cells.items():
        print(f"{name:11s} wk1-4 {show(c['wk1-4'])} | wk5+ {show(c['wk5+'])} | wk5-9 {show(c['wk5-9'])} | wk10+ {show(c['wk10+'])}")
    print("gaps (weeks 5+ minus weeks 1-4):", {k: (round(100 * v, 1) if v is not None else None) for k, v in res["gaps"].items()})
    print(checks)
    print(res["verdict"])


if __name__ == "__main__":
    main()
