"""Role inputs and a machine-learning model for rushing and receiving projections.

    python experiments/players_roles_test.py [--proj path]

Starts from the player walk-forward (players_test.py's table: every projection made with
earlier seasons only). For each season S from 2014, a gradient-boosted model per stat is
trained on seasons before S and predicts S from:

  the current model     the volume x efficiency chain and its calibrated projection, the
                        player's recent average, his usage shares, the team's projected
                        passes and carries, implied points, margin, total, wind, the defense
  roles (nflmodel/roles.py)  snap share (decayed, last game, last two, trend), share of
                        carries and targets in close games, depth-chart rank, a new team
  background            age, draft pick, years in the league
  team play-calling     pace (plays per game) and neutral pass rate

Variants: the current model; boosting on the current model's inputs only; boosting with
everything. Scored on the same player-games (chosen on the current projection, before the
game): average miss, root-mean-square miss, bias, and the Brier score of P(over) at
half-point lines around the current projection, with prop widths fitted on 2014-2018.
Tuning 2014-2018, holdout 2019 on.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model, players as PL, players_data as PD, propdist as D, props as P, roles  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
STATS = {"carries": ("RB", "QB"), "rush_yds": ("RB", "QB"), "targets": ("WR", "TE", "RB"), "receptions": ("WR", "TE", "RB"), "rec_yds": ("WR", "TE", "RB")}
BASE_F = ["{s}_p", "{s}_proj", "{s}_recent", "tgt_share", "car_share", "tgt_share_raw", "car_share_raw", "proj_team_att", "proj_team_car",
          "imp_pts", "margin", "proj_total", "wind", "home", "def_ypc", "def_ypa", "def_ypt", "one_sum", "ypc", "ypt", "catch", "gl_car_rate", "rz_tgt_rate"]
ROLE_F = ["snap_avg", "snap_last", "snap_last2", "snap_trend", "car_close_avg", "tgt_close_avg", "new_team", "depth", "age", "draft_pick", "exp",
          "pace", "neutral_pass"]


def build(proj_path):
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    if proj_path and Path(proj_path).exists():
        proj = pd.read_pickle(proj_path)
    else:
        te, qe = extra.load_extra(verbose=False)
        inj = injuries.load_injuries(games, verbose=False)
        pr = injuries.load_pass_rush(games, verbose=False)
        eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
        frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
        pg, info = PD.load_player_games(verbose=False)
        av = PL.load_availability(verbose=False)
        table = PL.build(frame, pg, info, av, inj, pr, games=games)
        proj = PL.walk_forward(table)
    R = roles.load_all(games, verbose=False)
    F = roles.features(proj, R)
    proj = proj.merge(F, on=["game_id", "player_id"], how="left")
    for st in STATS:
        proj[st + "_recent"] = proj[st + "_sum"] / proj["one_sum"].replace(0, np.nan)
    return proj


def boost(train, test, stat, feats):
    X = lambda d: d[[f.format(s=stat) for f in feats]].to_numpy(float)
    y = train[stat].clip(lower=0).to_numpy(float)
    m = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                      min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(X(train), y)
    return m.predict(X(test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=None)
    args = ap.parse_args()
    proj = build(args.proj)
    played = proj["played"].astype(bool) & (proj["active"] == 1)
    variants = {"current": None, "boost, current inputs": BASE_F, "boost, all inputs": BASE_F + ROLE_F}
    preds = {k: proj.copy() for k in variants}
    seasons = sorted(proj["season"].unique())
    for stat, groups in STATS.items():
        for name, feats in variants.items():
            if feats is None:
                continue
            col = preds[name][stat + "_proj"].copy()
            for s in seasons:
                if s < 2014:
                    continue
                tr = proj[played & (proj["season"] < s) & proj["group"].isin(groups)]
                te = proj[(proj["season"] == s) & proj["group"].isin(groups)]
                if len(tr) < 1000 or te.empty:
                    continue
                col.loc[te.index] = boost(tr, te, stat, feats)
            preds[name][stat + "_proj"] = col
            print(f"  {stat}: {name} done", flush=True)
    rows = []
    for stat in ("rush_yds", "rec_yds", "carries", "receptions"):
        for g in STATS[stat]:
            base = preds["current"]
            sel = P.eligible(base, stat, g) & (base["season"] >= 2014)
            mu0 = base.loc[sel, stat + "_proj"].to_numpy()
            step = P.SPECS[stat].step
            L = (np.floor(mu0) + 0.5)[:, None] + step * np.array([-1, 0, 1])[None, :] if step < 5 else (np.floor(mu0 / 5) * 5 + 0.5)[:, None] + np.array([-10, 0, 10])[None, :]
            for name in variants:
                x = preds[name].loc[sel].reset_index(drop=True)
                tune = (x["season"] < C.HOLDOUT_FROM).to_numpy()
                spec = P.SPECS[stat]
                m = P.PropModel({}).fit(x[tune], stats=[spec.vol, stat] if spec.vol else [stat])
                for period, mask in (("2014-18", tune), ("2019-26", ~tune)):
                    d = x[mask].reset_index(drop=True)
                    Lp = L[mask]
                    y = d[stat].to_numpy(float)
                    rec = {"stat": stat, "group": g, "variant": name, "period": period, "n": len(d),
                           "mae": round(float(np.mean(np.abs(d[stat + "_proj"] - y))), 3),
                           "rmse": round(float(np.sqrt(np.mean((d[stat + "_proj"] - y) ** 2))), 3),
                           "bias": round(float(np.mean(d[stat + "_proj"] - y)), 3)}
                    if (stat, g) in m.counts or (stat, g) in m.yards:
                        over, _, _ = m.probs(d, stat, g, Lp, calibrate=False)
                        ok = Lp >= 0.5
                        rec["brier"] = round(D.brier(over[ok], (y[:, None] > Lp)[ok].astype(float)), 5)
                    rows.append(rec)
    r = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    r.to_csv(OUT / "players_roles_test.csv", index=False)
    pd.set_option("display.width", 220)
    for period in ("2014-18", "2019-26"):
        print(f"\n== {period}")
        print(r[r.period == period].drop(columns="period").to_string(index=False))
    # which role inputs matter (permutation-free proxy: correlation with the current model's miss)
    x = preds["current"][played & (preds["current"]["season"] < C.HOLDOUT_FROM) & preds["current"]["group"].isin(["RB"])]
    miss = x["rush_yds"] - x["rush_yds_proj"]
    print("\ncorrelation of RB rushing-yards miss with each role input (tuning seasons):")
    print(pd.Series({f: round(float(np.corrcoef(x[f].fillna(x[f].median()), miss)[0, 1]), 3) for f in ROLE_F}).to_string())
    proj.to_pickle(C.RAW / "players_proj_roles.pkl")


if __name__ == "__main__":
    main()
