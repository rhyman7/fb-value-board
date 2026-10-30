"""Line movement as a model input, and as a filter on the board's spread bets (Oct 2026).

    python experiments/line_move_test.py

Two questions, both using the nfelo opening spread (data.load_openers):

1. Feature: move = closing spread - opening spread (positive = the market moved toward the
   home team), added to the margin model alone and scored walk-forward as
   experiments/feature_test.py does (tuning 2011-2018, holdout 2019 on; games without an
   opener get 0). Kept only if it lowers the raw model's average miss on the tuning seasons
   and holds on the holdout. Note what it would mean: this week's closing line as a feature,
   which the board does not do (README, "This week's line is never used as a feature").

2. Filter: the backtest's market-adjusted spread bets with EV above 0 (data/backtest_bets.csv,
   written by `python run.py`), bucketed by whether the line moved toward the bet's side,
   away from it, or not at all between the opener and the close. Win rate and return with
   one standard error per bucket, tuning and holdout.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")


def score(frame, mf):
    out, info = model.walk_forward(frame, margin_feats=mf)
    o = out[out.played & out.spread_line.notna()]
    res = {}
    for name, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        res[name] = np.mean(np.abs(x.model_margin - x.result))
    by_season = {int(s): float(np.mean(np.abs(x.model_margin - x.result))) for s, x in o.groupby("season")}
    return res, by_season, info


def rec(x: pd.DataFrame) -> str:
    w, l, p = int((x.outcome > 0).sum()), int((x.outcome < 0).sum()), int((x.outcome == 0).sum())
    n = len(x)
    if n == 0:
        return "no bets"
    roi, se = x.profit.mean(), x.profit.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return f"{w}-{l}-{p}  win {w / max(w + l, 1):.1%}  return {roi:+.1%} ± {se:.1%}  claimed EV {x.ev.mean():+.1%}"


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    openers = data.load_openers(refresh=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame = model.build_frame(games, eng, injuries=inj)
    frame = frame.merge(openers[["game_id", "open_spread"]], on="game_id", how="left")
    frame["move"] = (frame["spread_line"] - frame["open_spread"]).fillna(0.0)
    frame["move_abs"] = frame["move"].abs()
    have = frame.open_spread.notna() & frame.played & frame.spread_line.notna()
    print(f"openers for {int(have.sum())} of {int((frame.played & frame.spread_line.notna()).sum())} played games with a line")

    base, base_s, _ = score(frame, list(model.MARGIN_FEATS))
    print(f"base  margin tune {base['tune']:.3f} hold {base['hold']:.3f}")
    print("change in average miss when added (negative = better)")
    for cand in (["move"], ["move", "move_abs"]):
        r, r_s, info = score(frame, list(model.MARGIN_FEATS) + cand)
        better = sum(r_s[s] < base_s[s] for s in r_s if s >= 2014)
        coef = info[max(info)]["margin_coefs"]
        print(f"  margin {'+'.join(cand):14s} tune {r['tune'] - base['tune']:+.4f}  hold {r['hold'] - base['hold']:+.4f}"
              f"  seasons better since 2014: {better} of {sum(1 for s in r_s if s >= 2014)}  coef {[coef[c] for c in cand]}")

    # ---- the backtest's spread bets by line movement ------------------------------------
    path = C.DATA / "backtest_bets.csv"
    if not path.exists():
        print("no data/backtest_bets.csv: run `python run.py` first")
        return
    b = pd.read_csv(path)
    b = b[(b.source == "proj") & (b.market == "Spread") & (b.ev > 0)]
    b = b.merge(openers[["game_id", "open_spread"]], on="game_id", how="inner")
    b = b.merge(games[["game_id", "spread_line"]], on="game_id", how="left")
    move = b["spread_line"] - b["open_spread"]
    toward = np.where(b["side"] == "home", move, -move)      # positive: the line moved toward the bet's side
    b["bucket"] = np.select([toward > 0, toward < 0], ["moved toward the model's side", "moved away from it"], "did not move")
    print(f"\nmarket-adjusted spread bets with EV above 0 that have an opener: {len(b)}")
    for label, x in (("all seasons", b), ("tuning (2014-2018)", b[b.season < C.HOLDOUT_FROM]), ("holdout (2019 on)", b[b.season >= C.HOLDOUT_FROM])):
        print(f" {label}")
        for k in ("moved toward the model's side", "did not move", "moved away from it"):
            print(f"   {k:30s} {rec(x[x.bucket == k])}")
    big = b[np.abs(move) >= 1]
    print(" moves of a point or more")
    for k in ("moved toward the model's side", "moved away from it"):
        print(f"   {k:30s} {rec(big[big.bucket == k])}")


if __name__ == "__main__":
    main()
