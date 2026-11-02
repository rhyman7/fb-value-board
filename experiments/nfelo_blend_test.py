"""Our NFL spread model blended with nfelo's, graded at the opening spread.

    python experiments/nfelo_blend_test.py

nfelo (nfeloapp.com) publishes, per game since 2009, its Elo difference at the opener
(nfelo_dif_open: team Elo, home field, bye, divisional, surface, time-zone and QB adjustments) and
a model line regressed toward the market (nfelo_home_line_open). Two independently built models
often forecast better averaged than either alone, because their errors partly cancel.

Signals (home margin, points; positive = home better):
  ours        the board's early run (quarterbacks blind; nfl_segments.py's cache)
  nfelo       nfelo_dif_open / 25 (nfelo's own points conversion)
  blend(w)    (1 - w) x ours + w x nfelo, w chosen on 2014-2018 by average miss on the result

Graded as the board grades the opening spread (backtest.opening_report: games after a Monday
night left out): average miss on the final margin, and betting the signal's side when it is 2+
points off the opener (record, return at the opener's price, line move toward the signal).

Caution: nfelo's file is rebuilt with its current settings, which were tuned on these same
seasons, so its past numbers are partly in-sample; a blend that only just wins here would not.

Rule, set before running (Nov 2, 2026): a lead only if the blend lowers the average miss on the
holdout (2019 on) by more than two paired standard errors AND its 2+ opener bets return more than
ours on the holdout. Writes experiments/out/nfelo_blend_test.json.
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
OUT = HERE / "out" / "nfelo_blend_test.json"
TUNE = range(2014, C.HOLDOUT_FROM)


def bets(f: pd.DataFrame, sig: str, th: float = 2.0) -> dict:
    d = f[sig] - f.open_spread
    x = f[d.abs() >= th].copy()
    pick = np.sign(d[d.abs() >= th])
    x["res"] = np.sign(x.result - x.open_spread) * pick
    x["profit"] = S._profit(x.res, np.where(pick > 0, x.open_home_odds, x.open_away_odds))
    x["move"] = (x.spread_line - x.open_spread) * pick
    out = {}
    for name, m in (("tune", x.season.isin(TUNE)), ("hold", x.season >= C.HOLDOUT_FROM)):
        y = x[m]
        n = len(y)
        out[name] = {"bets": n, "record": f"{int((y.res > 0).sum())}-{int((y.res < 0).sum())}-{int((y.res == 0).sum())}",
                     "return": round(float(y.profit.mean()), 4), "return_se": round(float(y.profit.std(ddof=1) / np.sqrt(n)), 4),
                     "move_toward": round(float(y.move.mean()), 3)}
    return out


def main():
    R = S.build(False)
    games = R["games"]
    n = pd.read_csv(C.RAW / "nfelo_games.csv", usecols=["game_id", "nfelo_dif_open", "nfelo_home_line_open"])
    f = R["early"].merge(R["openers"], on="game_id").merge(n, on="game_id")
    monday = f.game_id.map(backtest._came_off_monday(games)).fillna(False).astype(bool)
    f = f[f.played & f.spread_line.notna() & f.open_spread.notna() & f.nfelo_dif_open.notna() & ~monday
          & (f.season >= C.FIRST_TEST_SEASON)].copy()
    f["ours"] = f.model_margin
    f["nfelo"] = f.nfelo_dif_open / 25.0
    f["nfelo_line"] = -f.nfelo_home_line_open
    tune = f.season.isin(TUNE)
    miss = lambda s: (f[s] - f.result).abs()
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    tm = [float(((1 - w) * f.ours[tune] + w * f.nfelo[tune] - f.result[tune]).abs().mean()) for w in ws]
    w = float(ws[int(np.argmin(tm))])
    f["blend"] = (1 - w) * f.ours + w * f.nfelo
    f["blend_half"] = 0.5 * f.ours + 0.5 * f.nfelo
    res = {"games": int(len(f)), "weight_on_nfelo_chosen_2014_2018": w,
           "tuning_miss_by_weight": dict(zip(map(str, ws), [round(v, 3) for v in tm])), "miss": {}, "bets_2plus": {}}
    hold = f.season >= C.HOLDOUT_FROM
    for s in ("ours", "nfelo", "nfelo_line", "blend", "blend_half", "open_spread", "spread_line"):
        res["miss"][s] = {"tune": round(float(miss(s)[tune].mean()), 3), "hold": round(float(miss(s)[hold].mean()), 3)}
    dd = (miss("blend") - miss("ours"))[hold]
    res["blend_minus_ours_hold"] = {"change": round(float(dd.mean()), 4), "se": round(float(dd.std(ddof=1) / np.sqrt(len(dd))), 4)}
    for s in ("ours", "nfelo", "blend", "blend_half"):
        res["bets_2plus"][s] = bets(f, s)
    b, o = res["bets_2plus"]["blend"]["hold"], res["bets_2plus"]["ours"]["hold"]
    # nfelo's QB adjustment uses the quarterback who actually started (not known at the opener
    # when it changed during the week): split by whether either team changed starters.
    qc = f.game_id.map(S.qb_change(games)).fillna(False).astype(bool)
    res["by_qb_change"] = {}
    for lab, m in (("qb_change", qc), ("no_qb_change", ~qc)):
        g = f[m]
        h = g[g.season >= C.HOLDOUT_FROM]
        res["by_qb_change"][lab] = {"games": int(len(g)),
                                    "hold_miss": {s: round(float((h[s] - h.result).abs().mean()), 3) for s in ("ours", "nfelo", "blend", "open_spread")},
                                    "bets_2plus": {s: bets(g, s) for s in ("ours", "nfelo")}}
    res["lead"] = bool(res["blend_minus_ours_hold"]["change"] < -2 * res["blend_minus_ours_hold"]["se"] and b["return"] > o["return"])
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "tuning_miss_by_weight"}, indent=1, default=float))


if __name__ == "__main__":
    main()
