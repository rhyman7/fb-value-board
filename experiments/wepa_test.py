"""Does nfelo's WEPA (weighted EPA) add anything to the NFL game models?

    python experiments/wepa_test.py

WEPA (github.com/greerreNFL/wepa) reweights each play's EPA by its kind (turnovers counted in part,
special teams at a learned weight, defensive plays boosted), the weights fitted each season on
earlier seasons only. Its "point in time" file, data/wepa_by_game_pit.csv, gives every team's WEPA
in every game since 2000 from weights that never saw that season: no look-ahead.

Here each team-game's offensive WEPA (column wepa) becomes one more rated metric in the board's
engine (opponent-adjusted ridge, recency decay, last season carried at a discount, one observation
per game like the points rating), and enters the walk-forward models as:
  margin   + d_wepa (home minus away), and swapped in for the points rating d_pts
  total    + s_wepa

Rule, set before running (Nov 2, 2026): the ridge penalty is chosen from a three-value grid on the
tuning seasons (before 2019) with the metric added to the margin model; a change is kept only if it
lowers the raw model's average miss on the tuning seasons and on the holdout (2019 on), the holdout
by more than two paired standard errors. Writes experiments/out/wepa_test.json.
"""
from __future__ import annotations

import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

OUT = HERE / "out" / "wepa_test.json"
URL = "https://raw.githubusercontent.com/greerreNFL/wepa/main/data/wepa_by_game_pit.csv"
GRID = (0.2, 0.6, 1.8)


def load_wepa() -> pd.DataFrame:
    path = C.RAW / "wepa_by_game_pit.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "nfl-model"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    w = pd.read_csv(path, usecols=["game_id", "team", "wepa"])
    w["team"] = w["team"].replace({**C.TEAM_ALIASES, "LAR": "LA"})
    return w


def attach(eng, tg: pd.DataFrame, w: pd.DataFrame, ridge: float) -> None:
    t = tg.sort_values("order").reset_index(drop=True)
    j = t[["game_id", "team"]].merge(w, on=["game_id", "team"], how="left")
    v = j["wepa"].to_numpy(float)
    eng.y["wepa"] = np.nan_to_num(v, nan=np.nanmean(v))
    eng.n_obs["wepa"] = np.where(np.isnan(v), 0.0, 1.0)
    eng.ridge["wepa"] = ridge
    eng.metrics = tuple(eng.metrics) + ("wepa",)


def paired(base, test, pred, actual, hold):
    j = base[["game_id", "season", pred, actual]].merge(test[["game_id", pred]], on="game_id", suffixes=("", "_t")).dropna()
    j["d"] = (j[pred + "_t"] - j[actual]).abs() - (j[pred] - j[actual]).abs()
    res = {}
    for p, x in (("tune", j[j.season < hold]), ("hold", j[j.season >= hold])):
        res[p] = {"n": int(len(x)), "base_miss": round(float((x[pred] - x[actual]).abs().mean()), 4),
                  "change": round(float(x.d.mean()), 4), "se": round(float(x.d.std(ddof=1) / np.sqrt(len(x))), 4)}
    res["keep"] = bool(res["tune"]["change"] < 0 and res["hold"]["change"] < -2 * res["hold"]["se"])
    return res


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    w = load_wepa()
    cover = tg[["game_id", "team"]].merge(w, on=["game_id", "team"], how="left").wepa.notna().mean()
    MF, TF, hold = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS), C.HOLDOUT_FROM

    def frame(ridge):
        eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
        attach(eng, tg, w, ridge)
        f = model.build_frame(games, eng)
        f["act_margin"], f["act_total"] = f.home_score - f.away_score, f.home_score + f.away_score
        return f

    def out(f, mf, tf):
        o, _ = model.walk_forward(f, margin_feats=mf, total_feats=tf)
        return o[o.played & o.model_margin.notna()]

    grid = {}
    frames = {r: frame(r) for r in GRID}
    for r, f in frames.items():
        o = out(f, MF + ["d_wepa"], TF)
        t = o[o.season < hold]
        grid[str(r)] = round(float((t.model_margin - t.act_margin).abs().mean()), 4)
    r = float(min(grid, key=grid.get))
    print("ridge by tuning miss", grid, "->", r, flush=True)
    f = frames[r]
    base = out(f, MF, TF)
    res = {"team_games_with_wepa": round(float(cover), 3), "ridge_grid": grid, "ridge": r, "holdout_from": hold}
    for name, mf, tf, pred, act in (
            ("margin + d_wepa", MF + ["d_wepa"], TF, "model_margin", "act_margin"),
            ("margin, points rating swapped for WEPA", [x if x != "d_pts" else "d_wepa" for x in MF], TF, "model_margin", "act_margin"),
            ("total + s_wepa", MF, TF + ["s_wepa"], "model_total", "act_total")):
        res[name] = paired(base, out(f, mf, tf), pred, act, hold)
        print(name, res[name], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
