"""Do drive-level ratings (FEI / SP+ style) add anything to the college game model?

    python experiments/cfb_drive_test.py

The board rates each team per play (pass and rush EPA, success rate) and per game (points).
FEI rates each drive instead (points per drive against what the starting spot was worth), and
SP+ has a "finishing drives" factor (points per trip inside the opponent's 40). Neither is in
the model. Two new rated metrics, built from the same cfbfastR play-by-play:

  dve  drive value over expected: points on the drive minus the average points of drives that
       started at that spot (the average is taken from 2014-2016 only, before any scored season)
  fin  finishing: points per drive that reached the opponent's 40

Drive points: touchdown 7, field goal 3, safety -2, interception or fumble returned for a
touchdown -7, anything else 0. Dropped, as FEI does: drives starting in garbage time (the
board's lead-by-quarter cutoffs), drives starting with under a minute left in a half, and
drives that ended the half or game without a score.

Each metric is rated exactly like the board's others (opponent-adjusted weighted ridge, recency
decay, last season carried at 0.55), weighted by drives / trips, and enters the walk-forward
models as d_<m> (margin) and s_<m> (total).

Rule, set before running (Nov 2, 2026):
  * the ridge penalty for each metric is chosen from a small grid on the tuning seasons
    (2017-2019) only, with the metric added to the margin model;
  * a metric is kept only if adding it lowers the raw model's average miss in both the tuning
    seasons and the holdout (2021 on), the holdout by at least 2 paired standard errors.

Writes experiments/out/cfb_drive_test.json and caches data/cfb/cache/drive_games_<season>.csv.
"""
from __future__ import annotations

import io
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

OUT = HERE / "out" / "cfb_drive_test.json"
COLS = ["game_id", "pos_team_id", "drive.id", "drive.result", "start.yardsToEndzone", "start.TimeSecsRem",
        "period", "pos_score_diff_start", "rush", "pass", "drive_play_index"]
TD = {"TD", "END OF HALF TD", "END OF GAME TD"}
DEF_TD = {"INT TD", "FUMBLE RETURN TD", "FUMBLE TD"}
DROP = {"END OF HALF", "END OF GAME", "END OF QUARTER", "END OF 4TH QUARTER", "Not provided", "UNKNOWN",
        "KICKOFF", "2PT"}
BASE_SEASONS = (2014, 2015, 2016)
GRID = {"dve": (4.0, 12.0, 36.0), "fin": (2.0, 6.0, 18.0)}


def drives_raw(season: int, games: pd.DataFrame) -> pd.DataFrame:
    """One row per drive: game, side, start spot, points, whether it reached the 40."""
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
    scrim = p["rush"].fillna(False).astype(bool) | p["pass"].fillna(False).astype(bool)
    p = p[scrim & p["pos_team_id"].notna() & p["drive.id"].notna()]
    p = p.sort_values(["game_id", "drive.id", "drive_play_index"])
    grp = p.groupby(["game_id", "drive.id"], sort=False)
    d = grp.agg(team_id=("pos_team_id", "first"), result=("drive.result", "first"),
                ytg=("start.yardsToEndzone", "first"), secs=("start.TimeSecsRem", "first"),
                period=("period", "first"), lead=("pos_score_diff_start", "first"),
                min_ytg=("start.yardsToEndzone", "min")).reset_index()
    for c in ("ytg", "secs", "period", "lead", "min_ytg"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    home = games.set_index("game_id")["home_id"]
    d["side"] = np.where(d.team_id.astype(int) == d.game_id.map(home), "home", "away")
    d["pts"] = np.select([d.result.isin(TD), d.result == "FG", d.result == "SF", d.result.isin(DEF_TD)],
                         [7.0, 3.0, -2.0, -7.0], 0.0)
    garbage = d.lead.abs() > d.period.fillna(4).astype(int).map(C.GARBAGE_LEAD).fillna(999)
    keep = ~garbage & (d.secs >= 60) & ~d.result.isin(DROP) & d.ytg.between(1, 99)
    d = d[keep]
    d["trip"] = (d.min_ytg <= 40).astype(int)
    d.insert(0, "season", season)
    return d[["season", "game_id", "side", "ytg", "pts", "trip"]]


def load_drives(games: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in sorted(games.season.unique()):
        path = C.CACHE / f"drive_raw_{s}.csv"
        if not path.exists():
            print(f"  downloading {s} drives ...", flush=True)
            drives_raw(int(s), games[games.season == s]).to_csv(path, index=False)
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True)


def per_game(d: pd.DataFrame) -> pd.DataFrame:
    base = d[d.season.isin(BASE_SEASONS)].groupby("ytg")["pts"].mean()
    base = base.reindex(range(1, 100)).interpolate().bfill().ffill()
    base = base.rolling(5, center=True, min_periods=1).mean()           # smooth the 1-yard buckets
    d = d.assign(dve=d.pts - d.ytg.map(base))
    g = d.groupby(["game_id", "side"]).agg(drives=("dve", "size"), dve=("dve", "mean"))
    t = d[d.trip == 1].groupby(["game_id", "side"]).agg(trips=("pts", "size"), fin=("pts", "mean"))
    return g.join(t).fillna({"trips": 0}).reset_index()


class DriveEngine(CFBRatingEngine):
    def __init__(self, tg, qb, drv, extra, **kw):
        super().__init__(tg, qb, **kw)
        t = tg.sort_values("order").reset_index(drop=True)
        j = t[["game_id", "side"]].merge(drv, on=["game_id", "side"], how="left")
        for m, n in (("dve", "drives"), ("fin", "trips")):
            v = j[m].to_numpy(float)
            w = j[n].fillna(0).to_numpy(float)
            self.y[m] = np.nan_to_num(v, nan=np.nanmean(v))
            self.n_obs[m] = np.where(np.isnan(v), 0.0, w)
        self.metrics = self.metrics + tuple(extra)


def paired(base, test, pred, actual, hold):
    j = base[["game_id", "season", pred, actual]].merge(test[["game_id", pred]], on="game_id", suffixes=("", "_t")).dropna()
    j = j[j.season != 2020]
    j["d"] = (j[pred + "_t"] - j[actual]).abs() - (j[pred] - j[actual]).abs()
    res = {}
    for p, x in (("tune", j[j.season.between(2017, 2019)]), ("hold", j[j.season >= hold])):
        res[p] = {"n": int(len(x)), "base_miss": round(float((x[pred] - x[actual]).abs().mean()), 4),
                  "change": round(float(x.d.mean()), 4), "se": round(float(x.d.std(ddof=1) / np.sqrt(len(x))), 4)}
    res["keep"] = bool(res["tune"]["change"] < 0 and res["hold"]["change"] <= -2 * res["hold"]["se"])
    return res


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games = games[games.season <= 2025]
    drv = per_game(load_drives(games))
    print(f"drives per team-game: {drv.drives.mean():.1f}, trips: {drv.trips.mean():.1f}", flush=True)
    MF, TF, hold = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS), C.HOLDOUT_FROM

    def frame(ridge):
        r = dict(C.RIDGE, **ridge)
        f = model.build_frame(games, DriveEngine(tg, qb, drv, ("dve", "fin"), games=games, ridge=r))
        f["act_margin"], f["act_total"] = f.home_score - f.away_score, f.home_score + f.away_score
        return f

    def out(f, mf, tf):
        o, _ = model.walk_forward(f, margin_feats=mf, total_feats=tf)
        return o[o.played & o.model_margin.notna()]

    # 1. pick each ridge on the tuning seasons only (metric added to the margin model)
    chosen, grid_log = {}, {}
    frames = {}
    for i in range(3):
        rg = {"dve": GRID["dve"][i], "fin": GRID["fin"][i]}
        frames[i] = frame(rg)
    for m in ("dve", "fin"):
        scores = []
        for i in range(3):
            o = out(frames[i], MF + [f"d_{m}"], TF)
            t = o[o.season.between(2017, 2019)]
            scores.append(float((t.model_margin - t.act_margin).abs().mean()))
        grid_log[m] = dict(zip(map(str, GRID[m]), [round(s, 4) for s in scores]))
        chosen[m] = GRID[m][int(np.argmin(scores))]
        print(f"{m}: tuning miss by ridge {grid_log[m]} -> {chosen[m]}", flush=True)

    f = frame(chosen)
    base = out(f, MF, TF)
    res = {"ridge_grid_tuning_margin_miss": grid_log, "ridge_chosen": chosen, "holdout_from": hold,
           "drives_per_team_game": round(float(drv.drives.mean()), 2), "trips_per_team_game": round(float(drv.trips.mean()), 2)}
    tests = {
        "margin + d_dve": (MF + ["d_dve"], TF, "model_margin", "act_margin"),
        "margin + d_fin": (MF + ["d_fin"], TF, "model_margin", "act_margin"),
        "margin + both": (MF + ["d_dve", "d_fin"], TF, "model_margin", "act_margin"),
        "total + s_dve": (MF, TF + ["s_dve"], "model_total", "act_total"),
        "total + s_fin": (MF, TF + ["s_fin"], "model_total", "act_total"),
        "total + both": (MF, TF + ["s_dve", "s_fin"], "model_total", "act_total"),
        "margin, points rating swapped for d_dve": ([x if x != "d_pts" else "d_dve" for x in MF], TF, "model_margin", "act_margin"),
    }
    for name, (mf, tf, pred, act) in tests.items():
        res[name] = paired(base, out(f, mf, tf), pred, act, hold)
        print(name, res[name], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
