"""nfelo's spreads as published before each game (its own archive, 2021 on), graded.

    python experiments/nfelo_published_test.py

experiments/nfelo_blend_test.py graded nfelo's rebuilt games file, which knows the quarterback who
actually started and uses settings tuned on the same seasons. nfelo also keeps an archive of the
spreads it published each week (output_data/historic_projected_spreads.csv, saved by
nflmodel/nfelolog.py to data/nfelo_log/): the model line before its pull toward the market
(home_line_pre_regression) and the line it published (home_closing_line_rounded_nfelo), with
the quarterbacks it assumed. These are as published, so grading them is a fair forward record.

Graded (home spread convention: negative = home favoured; results from nflverse):
  1. Average miss on the final margin: nfelo pre-regression, nfelo published, the board's early
     run, the opener, the close.
  2. Betting nfelo's side when its pre-regression line is 2+ / 3+ points off the opener (opener
     price) and off the close (closing price); and the same for the board's early run, on the
     same games, for comparison.

Rule, set before running (Nov 2, 2026): a lead only if nfelo's 2+ bets at the close return more
than zero by two standard errors (an edge against the closing line), or at the opener by two
standard errors with the line moving toward nfelo. Writes experiments/out/nfelo_published_test.json.
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
from nflmodel import config as C  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "nfelo_published_test.json"
ARCH = C.DATA / "nfelo_log" / "historic_projected_spreads.csv"


def grade(f: pd.DataFrame, sig: str, line: str, h_odds: str, a_odds: str, th: float) -> dict:
    """sig and line in home-margin points (positive = home better)."""
    d = f[sig] - f[line]
    x = f[d.abs() >= th].copy()
    pick = np.sign(d[d.abs() >= th])
    x["res"] = np.sign(x.result - x[line]) * pick
    x["profit"] = S._profit(x.res, np.where(pick > 0, x[h_odds], x[a_odds]))
    x["move"] = (x.close_m - x.open_m) * pick
    n = len(x)
    if not n:
        return {"bets": 0}
    return {"bets": n, "record": f"{int((x.res > 0).sum())}-{int((x.res < 0).sum())}-{int((x.res == 0).sum())}",
            "return": round(float(x.profit.mean()), 4), "return_se": round(float(x.profit.std(ddof=1) / np.sqrt(n)), 4),
            "move_toward": round(float(x.move.mean()), 3), "by_season": {int(s): round(float(g.profit.mean()), 3) for s, g in x.groupby("season")}}


def main():
    R = S.build(False)
    a = pd.read_csv(ARCH)
    a = a[a.home_line_pre_regression.notna()].drop_duplicates("game_id", keep="last")
    a["game_id"] = a.game_id.str.replace("OAK", "LV").str.replace("_LAR", "_LA")
    g = R["games"][["game_id", "season", "result", "spread_line", "home_spread_odds", "away_spread_odds"]]
    op = R["openers"]
    early = R["early"][["game_id", "model_margin"]]
    f = a.merge(g, on="game_id", suffixes=("_a", "")).merge(op, on="game_id", how="left").merge(early, on="game_id", how="left")
    f = f[f.result.notna()].copy()
    f["nfelo_pre"] = -f.home_line_pre_regression
    f["nfelo_pub"] = -f.home_closing_line_rounded_nfelo
    f["open_m"] = -f.home_line_open
    f["close_m"] = f.spread_line                      # nflverse: positive = home favoured
    f["ours"] = f.model_margin
    f["open_home_odds"] = f.open_home_odds.fillna(-110)
    f["open_away_odds"] = f.open_away_odds.fillna(-110)
    res = {"games": int(len(f)), "seasons": sorted(int(s) for s in f.season.unique()),
           "unmatched_archive_rows": int(len(a) - len(f)), "miss": {}, "bets": {}}
    for s in ("nfelo_pre", "nfelo_pub", "ours", "open_m", "close_m"):
        m = f[f[s].notna()]
        res["miss"][s] = round(float((m[s] - m.result).abs().mean()), 3)
    both = f[f.ours.notna()]
    res["miss_same_games"] = {s: round(float((both[s] - both.result).abs().mean()), 3) for s in ("nfelo_pre", "ours", "open_m", "close_m")}
    for th in (2, 3):
        for sig in ("nfelo_pre", "ours"):
            res["bets"][f"{sig} {th}+ off the opener"] = grade(both, sig, "open_m", "open_home_odds", "open_away_odds", th)
            res["bets"][f"{sig} {th}+ off the close"] = grade(both, sig, "close_m", "home_spread_odds", "away_spread_odds", th)
    c, o = res["bets"]["nfelo_pre 2+ off the close"], res["bets"]["nfelo_pre 2+ off the opener"]
    res["lead"] = bool((c.get("bets") and c["return"] > 2 * c["return_se"])
                       or (o.get("bets") and o["return"] > 2 * o["return_se"] and o["move_toward"] > 0))
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
