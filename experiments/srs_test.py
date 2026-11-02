"""Does SRS (Simple Rating System) add anything to the game models?

    python experiments/srs_test.py

SRS, as Pro Football Reference computes it: each team's average point margin adjusted for its
opponents (least squares, margin = rating_home - rating_away, no home field, no shrinkage, every
game this season counts the same), and OSRS / DSRS the same for points scored and allowed. Here
it is recomputed before every week from that season's earlier games only (week 1: zero), so it
can be used walk-forward. The board already has a points rating built the same way with home
field, recency weighting, last season at lower weight and shrinkage (ratings.py, d_pts / s_pts).

Added to the board's own walk-forward models (NFL and college):
  margin model  + d_srs (home SRS minus away SRS)
  total model   + s_srs (both teams' OSRS minus both teams' DSRS)
and, for information, the points rating swapped out for SRS.

Rule, set before running (Nov 2, 2026): SRS is kept only if adding it lowers the raw model's average
miss in both the tuning seasons and the holdout, the holdout by at least 2 standard errors (paired,
game by game).

Writes experiments/out/srs_test.json.
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
OUT = HERE / "out" / "srs_test.json"


def srs_asof(g: pd.DataFrame) -> pd.DataFrame:
    """g: one row per game with season, order, home, away, hs, as_, played. Returns per game the
    home and away SRS, OSRS and DSRS from that season's games before the game's week."""
    out = []
    for season, s in g.groupby("season"):
        teams = sorted(set(s.home) | set(s.away))
        ix = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        done = s[s.played].sort_values("order")
        for order in sorted(s.order.unique()):
            prev = done[done.order < order]
            cur = s[s.order == order]
            if len(prev) < n // 2:          # too few games for a solve: zero (the week-1 case)
                r = o = d = np.zeros(n)
            else:
                h, a = prev.home.map(ix).to_numpy(), prev.away.map(ix).to_numpy()
                m = len(prev)
                X = np.zeros((m + 1, n))
                X[np.arange(m), h], X[np.arange(m), a] = 1, -1
                X[m, :] = 1                  # ratings sum to zero
                y = np.append((prev.hs - prev.as_).to_numpy(float), 0.0)
                r = np.linalg.lstsq(X, y, rcond=None)[0]
                # points: scored = avg + off_self - def_opp, each game twice
                Xp = np.zeros((2 * m + 2, 2 * n + 1))
                Xp[:2 * m, -1] = 1
                Xp[np.arange(m), h], Xp[np.arange(m), n + a] = 1, -1
                Xp[m + np.arange(m), a], Xp[m + np.arange(m), n + h] = 1, -1
                Xp[2 * m, :n], Xp[2 * m + 1, n:2 * n] = 1, 1
                yp = np.concatenate([prev.hs.to_numpy(float), prev.as_.to_numpy(float), [0.0, 0.0]])
                sol = np.linalg.lstsq(Xp, yp, rcond=None)[0]
                o, d = sol[:n], sol[n:2 * n]
            hi, ai = cur.home.map(ix).to_numpy(), cur.away.map(ix).to_numpy()
            out.append(pd.DataFrame({"game_id": cur.game_id.to_numpy(), "d_srs": r[hi] - r[ai],
                                     "s_srs": o[hi] + o[ai] - d[hi] - d[ai]}))
    return pd.concat(out, ignore_index=True)


def paired(base: pd.DataFrame, test: pd.DataFrame, pred: str, actual: str, holdout_from: int) -> dict:
    j = base[["game_id", "season", pred, actual]].merge(test[["game_id", pred]], on="game_id", suffixes=("", "_t")).dropna()
    j["d"] = (j[pred + "_t"] - j[actual]).abs() - (j[pred] - j[actual]).abs()
    res = {}
    for p, x in (("tune", j[j.season < holdout_from]), ("hold", j[j.season >= holdout_from])):
        res[p] = {"n": int(len(x)), "base_miss": round(float((x[pred] - x[actual]).abs().mean()), 4),
                  "change": round(float(x.d.mean()), 4), "se": round(float(x.d.std(ddof=1) / np.sqrt(len(x))), 4)}
    res["keep"] = bool(res["tune"]["change"] < 0 and res["hold"]["change"] <= -2 * res["hold"]["se"])
    return res


def run(league: str) -> dict:
    if league == "nfl":
        from nflmodel import config as C, data, extra, model
        from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
        games, tg, qb = data.load_all(refresh=False, verbose=False)
        te, qe = extra.load_extra(verbose=False)
        eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
        frame = model.build_frame(games, eng)
        g = games.rename(columns={"home_team": "home", "away_team": "away", "home_score": "hs", "away_score": "as_"})
        MF, TF, wf, hold = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS), model.walk_forward, C.HOLDOUT_FROM
    else:
        from cfbmodel import config as C, data, model
        from cfbmodel.ratings import CFBRatingEngine
        games, tg, qb = data.load_all(refresh=False, verbose=False)
        eng = CFBRatingEngine(tg, qb, games=games)
        frame = model.build_frame(games, eng)
        g = games.rename(columns={"home_team": "home", "away_team": "away", "home_score": "hs", "away_score": "as_"})
        MF, TF, wf, hold = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS), model.walk_forward, C.HOLDOUT_FROM
    g = g[["game_id", "season", "order", "home", "away", "hs", "as_", "played"]].copy()
    frame = frame.merge(srs_asof(g), on="game_id", how="left")
    frame[["d_srs", "s_srs"]] = frame[["d_srs", "s_srs"]].fillna(0.0)
    frame["act_margin"] = frame["home_score"] - frame["away_score"]
    frame["act_total"] = frame["home_score"] + frame["away_score"]

    def out(mf, tf):
        o, _ = wf(frame, margin_feats=mf, total_feats=tf)
        return o[o.played & o.model_margin.notna()]

    base = out(MF, TF)
    add = out(MF + ["d_srs"], TF + ["s_srs"])
    swap = out([f if f != "d_pts" else "d_srs" for f in MF], [f if f != "s_pts" else "s_srs" for f in TF])
    return {"margin, SRS added": paired(base, add, "model_margin", "act_margin", hold),
            "total, OSRS/DSRS added": paired(base, add, "model_total", "act_total", hold),
            "margin, points rating swapped for SRS": paired(base, swap, "model_margin", "act_margin", hold),
            "total, points rating swapped for OSRS/DSRS": paired(base, swap, "model_total", "act_total", hold),
            "holdout_from": hold}


def main():
    res = {lg: run(lg) for lg in ("nfl", "cfb")}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
