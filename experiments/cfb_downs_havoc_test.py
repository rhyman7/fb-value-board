"""Do down-split success rates, line yards and havoc add anything to the college game model?

    python experiments/cfb_downs_havoc_test.py

CollegeFootballData's adjusted metrics (and SP+) split success rate by standard downs and passing
downs, rate the offensive line by line yards per rush, and track havoc (tackles for loss, sacks,
interceptions, forced fumbles and pass breakups per play). The board rates overall success rate,
pass and rush EPA, and points. From cfbfastR's play-by-play (its own standard_down / passing_down,
line_yards and havoc columns), over the same non-garbage scrimmage plays the board uses:

  sdsr   success rate on standard downs        (weighted by standard-down plays)
  pdsr   success rate on passing downs         (weighted by passing-down plays)
  lyd    line yards per rush (each run clipped to -6..7)   (weighted by rushes)
  hav    havoc allowed per play                (weighted by plays)

Each is rated like the board's other metrics (opponent-adjusted ridge, recency decay, last season
at 0.55) and enters the walk-forward models as d_<m> / s_<m>.

Rule, set before running (Nov 2, 2026):
  * each ridge penalty is chosen from a three-value grid on 2017-2019 only, with the metric added
    to the margin model;
  * a metric is kept only if adding it lowers the raw model's average miss in both the tuning
    seasons and the holdout (2021 on), the holdout by at least 2 paired standard errors.

The same pass caches turnover-free pass and rush EPA for experiments/turnover_free_test.py.
Writes experiments/out/cfb_downs_havoc_test.json and data/cfb/cache/play_extra_<season>.csv.
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
from cfbmodel import config as C, data, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

OUT = HERE / "out" / "cfb_downs_havoc_test.json"
COLS = ["game_id", "pos_team_id", "period", "pos_score_diff_start", "rush", "pass", "kneel_down",
        "penalty_no_play", "EPA", "EPA_success", "standard_down", "passing_down", "line_yards", "havoc",
        "turnover_vec"]
# metric: (value column, weight column)
METRICS = {"sdsr": ("sd_succ", "sd_n"), "pdsr": ("pd_succ", "pd_n"), "lyd": ("lyd_sum", "n_rush"),
           "hav": ("hav_sum", "plays"), "pnt": ("pnt_epa", "pnt_n"), "rnt": ("rnt_epa", "rnt_n")}
GRID = (6.0, 18.0, 54.0)


def reduce_season(season: int, games: pd.DataFrame) -> pd.DataFrame:
    url = f"{C.BASE}/{C.FILES['pbp'].format(s=season)}"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pbp.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": "cfb-model"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(path).schema.names)
        p = pd.read_parquet(path, columns=[c for c in COLS if c in have])
    p = p[p["game_id"].astype(int).isin(set(games.game_id))].copy()
    p["game_id"] = p["game_id"].astype(int)
    flag = lambda c: p[c].fillna(False).astype(bool) if c in p else pd.Series(False, index=p.index)
    scrim = (flag("rush") | flag("pass")) & ~flag("kneel_down") & ~flag("penalty_no_play") & p["EPA"].notna()
    p = p[scrim & p["pos_team_id"].notna()].copy()
    period = pd.to_numeric(p["period"], errors="coerce").fillna(4).astype(int)
    lead = pd.to_numeric(p["pos_score_diff_start"], errors="coerce").abs().fillna(0)
    p = p[~(lead > period.map(C.GARBAGE_LEAD).fillna(999))]
    home = games.set_index("game_id")["home_id"]
    p["side"] = np.where(p["pos_team_id"].astype(int) == p["game_id"].map(home), "home", "away")
    p["epa"] = p["EPA"].clip(-C.EPA_CLIP, C.EPA_CLIP)
    p["succ"] = p["EPA_success"].fillna(False).astype(bool).astype(int)
    p["is_pass"] = flag("pass").loc[p.index].astype(int)
    p["is_rush"] = 1 - p["is_pass"]
    sd, pd_ = flag("standard_down").loc[p.index].astype(int), flag("passing_down").loc[p.index].astype(int)
    to = flag("turnover_vec").loc[p.index].astype(int)
    p = p.assign(sd_n=sd, sd_succ=sd * p.succ, pd_n=pd_, pd_succ=pd_ * p.succ, n_rush=p.is_rush,
                 lyd_sum=p.is_rush * pd.to_numeric(p["line_yards"], errors="coerce").fillna(0).clip(-6, 7),
                 hav_sum=flag("havoc").loc[p.index].astype(int), plays=1,
                 pnt_n=p.is_pass * (1 - to), pnt_epa=p.epa * p.is_pass * (1 - to),
                 rnt_n=p.is_rush * (1 - to), rnt_epa=p.epa * p.is_rush * (1 - to))
    cols = sorted({c for v in METRICS.values() for c in v})
    g = p.groupby(["game_id", "side"])[cols].sum().reset_index()
    g.insert(0, "season", season)
    return g


def load_extra(games: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in sorted(games.season.unique()):
        path = C.CACHE / f"play_extra_{s}.csv"
        if not path.exists():
            print(f"  downloading {s} plays ...", flush=True)
            reduce_season(int(s), games[games.season == s]).to_csv(path, index=False)
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True)


def add_metrics(engine, tg: pd.DataFrame, ex: pd.DataFrame, names, ridge: dict) -> None:
    """Attach extra rated metrics to an already-built rating engine (CFB or NFL)."""
    t = tg.sort_values("order").reset_index(drop=True)
    j = t[["game_id", "side"]].merge(ex, on=["game_id", "side"], how="left")
    for m in names:
        v, w = METRICS[m] if m in METRICS else m
        num, den = j[v].to_numpy(float), j[w].fillna(0).to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = num / den
        y[~np.isfinite(y)] = np.nan
        engine.y[m] = np.nan_to_num(y, nan=np.nanmean(y))
        engine.n_obs[m] = np.where(np.isnan(y), 0.0, den)
        engine.ridge[m] = ridge[m]
    engine.metrics = tuple(engine.metrics) + tuple(n for n in names if n not in engine.metrics)


def paired(base, test, pred, actual, tune, hold):
    j = base[["game_id", "season", pred, actual]].merge(test[["game_id", pred]], on="game_id", suffixes=("", "_t")).dropna()
    j["d"] = (j[pred + "_t"] - j[actual]).abs() - (j[pred] - j[actual]).abs()
    res = {}
    for p, x in (("tune", j[j.season.isin(tune)]), ("hold", j[j.season >= hold])):
        res[p] = {"n": int(len(x)), "base_miss": round(float((x[pred] - x[actual]).abs().mean()), 4),
                  "change": round(float(x.d.mean()), 4), "se": round(float(x.d.std(ddof=1) / np.sqrt(len(x))), 4)}
    res["keep"] = bool(res["tune"]["change"] < 0 and res["hold"]["change"] <= -2 * res["hold"]["se"])
    return res


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games = games[games.season <= 2025]
    ex = load_extra(games)
    names = ("sdsr", "pdsr", "lyd", "hav")
    MF, TF, hold, tune = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS), C.HOLDOUT_FROM, [2017, 2018, 2019]

    def frame(ridge):
        e = CFBRatingEngine(tg, qb, games=games)
        add_metrics(e, tg, ex, names, ridge)
        f = model.build_frame(games, e)
        f["act_margin"], f["act_total"] = f.home_score - f.away_score, f.home_score + f.away_score
        return f

    def out(f, mf, tf):
        o, _ = model.walk_forward(f, margin_feats=mf, total_feats=tf)
        return o[o.played & o.model_margin.notna()]

    frames = [frame({m: r for m in names}) for r in GRID]
    chosen, grid_log = {}, {}
    for m in names:
        sc = []
        for f in frames:
            o = out(f, MF + [f"d_{m}"], TF)
            t = o[o.season.isin(tune)]
            sc.append(float((t.model_margin - t.act_margin).abs().mean()))
        grid_log[m] = dict(zip(map(str, GRID), [round(s, 4) for s in sc]))
        chosen[m] = GRID[int(np.argmin(sc))]
        print(m, grid_log[m], "->", chosen[m], flush=True)
    f = frame(chosen)
    base = out(f, MF, TF)
    res = {"ridge_grid_tuning_margin_miss": grid_log, "ridge_chosen": chosen, "holdout_from": hold}
    tests = [(f"margin + d_{m}", MF + [f"d_{m}"], TF, "model_margin", "act_margin") for m in names]
    tests.append(("margin + all four", MF + [f"d_{m}" for m in names], TF, "model_margin", "act_margin"))
    tests += [(f"total + s_{m}", MF, TF + [f"s_{m}"], "model_total", "act_total") for m in names]
    tests.append(("total + all four", MF, TF + [f"s_{m}" for m in names], "model_total", "act_total"))
    for name, mf, tf, pred, act in tests:
        res[name] = paired(base, out(f, mf, tf), pred, act, tune, hold)
        print(name, res[name], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
