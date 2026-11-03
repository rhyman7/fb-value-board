"""College rushing yards: the chain, the recent average, their plain average, or the calibration?

    python experiments/cfb_rush_blend_test.py

Ryan's question (Nov 3, 2026, Messiah Burch at Miami OH: chain 71.9, recent 103.6, projection
95.4): which of the simple numbers is closest to what the player actually ran for?

Rules, written before running:

1. Candidates, per running back per game, from the college walk-forward player table
   (cfbmodel.players.walk_forward, first_test 2017, the board's own backtest):
     chain      rush_yds_p: team carries x his share x his yards per carry x the defense
     recent     his recent per-game rushing yards (the calibration's second input)
     average    (chain + recent) / 2
     projection rush_yds_proj, the calibration regression's output (what the board shows)
2. Sample: group RB, appeared in the game, played, chain or projection >= 20 yards (the
   prop-relevant backs); 2017-2025.
3. Score: mean absolute miss against actual rushing yards, and bias (mean of projection minus
   actual). Tuning 2017-2019 and holdout 2021 on, standard errors clustered by game.
4. Decision: a candidate replaces the projection only if its miss is lower on both periods and
   lower on the holdout by 2+ standard errors (paired difference). Otherwise nothing changes.

Writes experiments/out/cfb_rush_blend_test.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
CACHE = Path("/tmp/cfb_pproj_all.pkl")


def projections():
    from cfbmodel import config as C, data, model, players, players_data
    from cfbmodel.ratings import CFBRatingEngine
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, engine))
    pg, pinfo = players_data.load_player_games(verbose=False)
    pproj = players.walk_forward(players.build(games, frame, pg, pinfo, None, None), first_test=C.FIRST_TEST_SEASON)
    pproj.to_pickle(CACHE)
    return pproj


def clustered_se(x: np.ndarray, groups: np.ndarray) -> float:
    s = pd.Series(x - x.mean()).groupby(groups).sum()
    return float(np.sqrt((s ** 2).sum()) / len(x))


def main():
    p = projections()
    d = p[(p["group"] == "RB") & (p["appeared"] == 1) & p["played"].astype(bool) & p["rush_yds"].notna()].copy()
    d["recent"] = (d["rush_yds_sum"] / d["one_sum"].replace(0, np.nan)).fillna(0.0)
    d["chain"] = d["rush_yds_p"]
    d["average"] = (d["chain"] + d["recent"]) / 2
    d["projection"] = d["rush_yds_proj"]
    d = d[(d["chain"] >= 20) | (d["projection"] >= 20)]
    periods = {"tuning 2017-19": d[d.season <= 2019], "holdout 2021+": d[d.season >= 2021]}
    rows = []
    for per, x in periods.items():
        base = np.abs(x["projection"] - x["rush_yds"]).to_numpy()
        for c in ("chain", "recent", "average", "projection"):
            err = np.abs(x[c] - x["rush_yds"]).to_numpy()
            diff = err - base
            rows.append({"period": per, "candidate": c, "n": len(x), "mean_abs_miss": err.mean(),
                         "bias": float((x[c] - x["rush_yds"]).mean()),
                         "vs_projection": diff.mean(), "se": clustered_se(diff, x["game_id"].to_numpy()),
                         "z": diff.mean() / clustered_se(diff, x["game_id"].to_numpy()) if c != "projection" else 0.0})
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "cfb_rush_blend_test.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    # the fitted weights, for the record
    for per, x in periods.items():
        X = np.column_stack([np.ones(len(x)), x["chain"], x["recent"]])
        w = np.linalg.lstsq(X, x["rush_yds"].to_numpy(float), rcond=None)[0]
        print(f"{per}: best in-sample fit = {w[0]:.1f} + {w[1]:.2f} x chain + {w[2]:.2f} x recent")


if __name__ == "__main__":
    main()
