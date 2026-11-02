"""Does the playing surface (grass vs artificial turf) add anything to the NFL game model?

    python experiments/surface_test.py

nfelo adds a "surface adjustment" to home field when the visitors play at home on the other kind
of surface. The board has no surface input. From nflverse's per-game `surface` (gaps filled from
the same stadium's other games), each game is grass (grass, dessograss hybrid) or turf (fieldturf,
sportturf, matrixturf, astroturf, a_turf, astroplay); each team's home surface in a season is the
surface of most of its home games that season.

Features:
  d_surf        away team off its home surface minus home team off its home surface (the home
                team is only ever "off" at neutral sites); positive helps the home team
  d_surf_t2g    the same, only for turf teams playing on grass
  d_surf_g2t    the same, only for grass teams playing on turf
  s_turf        the game is on turf (total model)

Rule, set before running (Nov 2, 2026): a feature is kept only if adding it lowers the raw
model's average miss in both the tuning seasons and the holdout (2019 on), the holdout by at
least 2 paired standard errors.

Writes experiments/out/surface_test.json.
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
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

OUT = HERE / "out" / "surface_test.json"
GRASS = {"grass", "dessograss"}


def paired(base, test, pred, actual, tune, hold):
    j = base[["game_id", "season", pred, actual]].merge(test[["game_id", pred]], on="game_id", suffixes=("", "_t")).dropna()
    j["d"] = (j[pred + "_t"] - j[actual]).abs() - (j[pred] - j[actual]).abs()
    res = {}
    for p, x in (("tune", j[j.season.isin(tune)]), ("hold", j[j.season >= hold])):
        res[p] = {"n": int(len(x)), "base_miss": round(float((x[pred] - x[actual]).abs().mean()), 4),
                  "change": round(float(x.d.mean()), 4), "se": round(float(x.d.std(ddof=1) / np.sqrt(len(x))), 4)}
    res["keep"] = bool(res["tune"]["change"] < 0 and res["hold"]["change"] <= -2 * res["hold"]["se"])
    return res


def surface_features(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = games.copy()
    s = g["surface"].astype("string").str.strip().str.lower().replace({"": pd.NA})
    g["turf"] = s.map(lambda x: np.nan if pd.isna(x) else float(x not in GRASS))
    by_stadium = g.groupby("stadium_id")["turf"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    filled = g["turf"].isna()
    g.loc[filled, "turf"] = g.loc[filled, "stadium_id"].map(by_stadium)
    home_games = g[(g.neutral == 0) & g.turf.notna()]
    home_surf = home_games.groupby(["home_team", "season"])["turf"].agg(lambda x: float(x.mean() >= 0.5))
    hs = [home_surf.get((t, s_), np.nan) for t, s_ in zip(g.home_team, g.season)]
    as_ = [home_surf.get((t, s_), np.nan) for t, s_ in zip(g.away_team, g.season)]
    g["home_surf"], g["away_surf"] = hs, as_
    ok = g.turf.notna()
    away_off = (ok & g.away_surf.notna() & (g.away_surf != g.turf)).astype(float)
    home_off = (ok & g.home_surf.notna() & (g.home_surf != g.turf)).astype(float)
    t2g, g2t = (g.turf == 0).astype(float), (g.turf == 1).astype(float)
    f = pd.DataFrame({"game_id": g.game_id, "d_surf": away_off - home_off,
                      "d_surf_t2g": (away_off - home_off) * t2g, "d_surf_g2t": (away_off - home_off) * g2t,
                      "s_turf": g.turf.fillna(g.turf.mean())})
    info = {"surface_unknown_after_fill": int(g.turf.isna().sum()),
            "away_off_surface_share": round(float(away_off[g.neutral == 0].mean()), 3),
            "turf_share": round(float(g.turf.mean()), 3)}
    return f, info


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    frame = model.build_frame(games, ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games))
    f, info = surface_features(games)
    frame = frame.merge(f, on="game_id", how="left")
    cols = [c for c in f.columns if c != "game_id"]
    frame[cols] = frame[cols].fillna(0.0)
    frame["act_margin"] = frame["home_score"] - frame["away_score"]
    frame["act_total"] = frame["home_score"] + frame["away_score"]
    MF, TF, hold = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS), C.HOLDOUT_FROM
    tune = list(range(int(frame.season.min()), hold))

    def out(mf, tf):
        o, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
        return o[o.played & o.model_margin.notna()]

    res = dict(info, holdout_from=hold)
    base = out(MF, TF)
    for name, mf, tf, pred, act in (
            ("margin + d_surf", MF + ["d_surf"], TF, "model_margin", "act_margin"),
            ("margin + turf-to-grass / grass-to-turf split", MF + ["d_surf_t2g", "d_surf_g2t"], TF, "model_margin", "act_margin"),
            ("total + s_turf", MF, TF + ["s_turf"], "model_total", "act_total")):
        res[name] = paired(base, out(mf, tf), pred, act, tune, hold)
        print(name, res[name], flush=True)
    print(info)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
