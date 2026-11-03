"""NFL rushing calibration per position group (the college fix, FEATURES.md Nov 3, 2026, tested
on the NFL).

    python experiments/nfl_rush_group_calib_test.py

The NFL chain differs from college: for running backs the board shows the boosted step's
number (players_ml.ADOPT), to which the linear calibration is one input; for quarterbacks'
rushing the linear calibration is what the board shows.

Rules, written before running (Nov 3, 2026):
1. Change: nflmodel.players.walk_forward fits rush_yds and carries per group (RB; QB; WR+TE),
   everything else as is.
2. Score A, quarterbacks (linear output live): RMSE of rushing yards for projected starters
   (att_share > 0.5, started) with chain or projection >= 5, tuning 2014-2018 and holdout
   2019 on, paired difference clustered by game.
3. Score B, running backs (boosted output live): players_ml.walk_forward rerun on the new
   table from 2019 as production runs it (the pre-2014 seasons are too few rows to train it),
   RMSE of RB rushing yards with chain or projection >= 20, holdout 2019 on.
4. Score C, lines: Brier of the over chance at the 2023-2025 DraftKings/FanDuel closing
   rushing-yards main lines (data/props/history, propbackfill.load(season, "close")), for QBs
   with the linear output and RBs with the boosted output, new vs current, clustered by game.
5. Keep only if QB RMSE improves on both periods, the holdout by 2+ SE, and the RB RMSE and
   both Briers are no worse (difference <= 0 or within 2 SE). Otherwise nothing changes.

Table: data/raw/players_proj.pkl (the full walk-forward of Nov 1, 2026; it carries every table
column, so walk_forward reruns on it). Writes experiments/out/nfl_rush_group_calib_test.json.
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
from nflmodel import config as C, data, odds as O, players as PL, players_ml as ML, propbackfill as B, propdist as D, propmarket as PM, props as P, roles  # noqa: E402

OUT = HERE / "out"
GROUP_STATS = ("rush_yds", "carries")
CALIB_GROUPS = (("RB",), ("QB",), ("WR", "TE"))
SEASONS = (2023, 2024, 2025)


def _calibrate(h, cur, stat):
    m = PL.Lin(1.0).fit(PL.calib_X(h, stat), h[stat].to_numpy(float))
    X = PL.calib_X(cur, stat)
    none = (X[:, 0] == 0) & (X[:, 1] == 0)
    return np.where(none, 0.0, np.clip(m.predict(X), 0, None))


def walk_forward_grouped(table: pd.DataFrame, first_test: int = 2013) -> pd.DataFrame:
    """nflmodel.players.walk_forward with rush_yds and carries calibrated per position group."""
    out = []
    seasons = sorted(table["season"].unique())
    tg = table.drop_duplicates(["game_id", "team"])
    for s in seasons:
        if s < first_test:
            continue
        tr_t = tg[(tg["season"] < s) & tg["played"].astype(bool) & tg["team_att"].notna()]
        tr_t = tr_t[tr_t["season"] >= s - 8]
        att = PL.Lin(5.0).fit(tr_t[PL.VOL_FEATS].fillna(0), tr_t["team_att"].to_numpy(float))
        car = PL.Lin(5.0).fit(tr_t[PL.VOL_FEATS].fillna(0), tr_t["team_car"].to_numpy(float))
        hist = PL.raw_projection(table[(table["season"] < s) & (table["season"] >= max(first_test - 1, s - 6))], att, car)
        hist = hist[hist["played"].astype(bool) & (hist["active"] == 1)]
        cur = PL.raw_projection(table[table["season"] == s], att, car)
        qb_fit = hist[(hist["att_share"] > 0.5) & (hist["started"] == 1)]
        for stat in PL.STATS:
            if stat in GROUP_STATS:
                cur[stat + "_proj"] = 0.0
                for grps in CALIB_GROUPS:
                    h = hist[hist["group"].isin(grps)]
                    sel = cur["group"].isin(grps).to_numpy()
                    if h.empty or not sel.any():
                        continue
                    cur.loc[sel, stat + "_proj"] = _calibrate(h, cur[sel], stat)
                continue
            h = qb_fit if stat in PL.PASS_STATS else hist
            cur[stat + "_proj"] = _calibrate(h, cur, stat)
        out.append(cur)
    return pd.concat(out, ignore_index=True)


def paired(err_new, err_old, game):
    d = np.asarray(err_new) - np.asarray(err_old)
    cl = pd.Series(d - d.mean()).groupby(np.asarray(game)).sum()
    return float(d.mean()), float(np.sqrt((cl ** 2).sum()) / len(d))


def rmse_block(x, new_col, old_col="rush_yds_proj"):
    out = {}
    for per, g in (("tuning 2014-18", x[x.season < C.HOLDOUT_FROM]), ("holdout 2019+", x[x.season >= C.HOLDOUT_FROM])):
        e_old = (g[old_col] - g.rush_yds).to_numpy() ** 2
        e_new = (g[new_col] - g.rush_yds).to_numpy() ** 2
        diff, se = paired(e_new, e_old, g.game_id)
        out[per] = {"n": int(len(g)), "rmse_current": float(np.sqrt(e_old.mean())), "rmse_new": float(np.sqrt(e_new.mean())),
                    "mse_diff": diff, "se": se, "z": diff / se if se else 0.0,
                    "bias_current": float((g[old_col] - g.rush_yds).mean()), "bias_new": float((g[new_col] - g.rush_yds).mean())}
    return out


def brier_block(rows: pd.DataFrame, group: str, new_col: str, model) -> dict:
    """Brier at the closing rushing-yards lines, new vs current center, one group."""
    parts = []
    for season in SEASONS:
        snaps = B.load(season, "close")
        if snaps.empty:
            continue
        snaps = snaps[snaps["market"] == "player_rush_yds"]
        info = pd.read_csv(C.CACHE / f"player_info_{season}.csv", dtype={"player_id": str})
        r = rows[(rows.season == season) & rows.played.astype(bool) & (rows.group == group)].merge(info[["player_id", "name"]], on="player_id", how="left")
        r["nname"] = r["name"].map(PM.norm_name)
        r = r[~r.duplicated(["game_id", "nname"], keep=False)]
        s = snaps.copy()
        s["nname"] = s["player"].map(PM.norm_name)
        s["side"] = s["side"].str.lower()
        w = s.pivot_table(index=["game_id", "nname", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
        w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
        w = w.groupby(["game_id", "nname", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
        w = w.merge(r, on=["game_id", "nname"], how="inner")
        if w.empty:
            continue
        w["stat"] = "rush_yds"
        parts.append(w)
    if not parts:
        return {"props": 0}
    base = pd.concat(parts, ignore_index=True)
    ps = {}
    for name, col in (("current", "rush_yds_proj"), ("new", new_col)):
        d = base.copy()
        d["rush_yds_proj"] = d[col]
        d = d[P.eligible(d, "rush_yds", group)].reset_index(drop=True)
        over, push, under = model.probs(d, "rush_yds", group, d["line"].to_numpy(float))
        d["p"] = over / (over + under)
        d = d[d["rush_yds"] != d["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
        ps[name] = d
    idx = ps["current"].index.intersection(ps["new"].index)
    a, b = ps["current"].loc[idx], ps["new"].loc[idx]
    hit = (a["rush_yds"] > a["line"]).astype(float).to_numpy()
    pa, pb = a["p"].to_numpy(), b["p"].to_numpy()
    d = (pb - hit) ** 2 - (pa - hit) ** 2
    cl = pd.Series(d - d.mean()).groupby(a["game_id"].to_numpy()).sum()
    return {"props": int(len(idx)), "brier_books": D.brier(a["fair"].to_numpy(), hit), "brier_current": D.brier(pa, hit),
            "brier_new": D.brier(pb, hit), "new_minus_current": float(d.mean()), "se": float(np.sqrt((cl ** 2).sum()) / len(d))}


def main():
    table = pd.read_pickle(C.RAW / "players_proj.pkl")
    print("current walk-forward ...")
    cur = PL.walk_forward(table)
    print("grouped walk-forward ...")
    new = walk_forward_grouped(table)
    key = ["game_id", "player_id"]
    # the linear outputs side by side
    m = cur.merge(new[key + ["rush_yds_proj", "carries_proj"]].rename(columns={"rush_yds_proj": "new_proj", "carries_proj": "new_car"}), on=key)
    res = {"rule": "keep only if QB RMSE improves on both periods, holdout by 2+ SE; RB boosted RMSE and both Briers no worse"}
    qb = m[(m.group == "QB") & (m.att_share > 0.5) & (m.started == 1) & m.played.astype(bool) & m.rush_yds.notna()]
    qb = qb[(qb.rush_yds_p >= 5) | (qb.rush_yds_proj >= 5)]
    res["qb_linear"] = rmse_block(qb, "new_proj")
    # the boosted step on each table (what the board shows for RBs)
    print("boosted step, current inputs ...")
    games, _, _ = data.load_all(refresh=False, verbose=False)
    R = roles.load_all(games, verbose=False)
    ml_cur = ML.walk_forward(ML.add_features(cur, R), first=C.HOLDOUT_FROM)
    print("boosted step, grouped inputs ...")
    ml_new = ML.walk_forward(ML.add_features(new, R), first=C.HOLDOUT_FROM)
    mm = ml_cur.merge(ml_new[key + ["rush_yds_proj", "carries_proj"]].rename(columns={"rush_yds_proj": "new_proj", "carries_proj": "new_car"}), on=key)
    rb = mm[(mm.group == "RB") & (mm.active == 1) & mm.played.astype(bool) & mm.rush_yds.notna()]
    rb = rb[((rb.rush_yds_p >= 20) | (rb.rush_yds_proj >= 20)) & (rb.season >= C.HOLDOUT_FROM)]
    res["rb_boosted"] = rmse_block(rb, "new_proj")
    model = P.PropModel.load()
    res["lines_qb"] = brier_block(m, "QB", "new_proj", model)
    res["lines_rb"] = brier_block(mm, "RB", "new_proj", model)
    q = res["qb_linear"]
    ok = lambda b: b.get("props", 1) == 0 or b["new_minus_current"] <= 0 or b["new_minus_current"] <= 2 * b["se"]
    rb_ok = all(v["mse_diff"] <= 0 or v["z"] <= 2 for v in res["rb_boosted"].values())
    res["kept"] = bool(all(v["mse_diff"] < 0 for v in q.values()) and q["holdout 2019+"]["z"] <= -2 and rb_ok and ok(res["lines_qb"]) and ok(res["lines_rb"]))
    OUT.mkdir(exist_ok=True)
    (OUT / "nfl_rush_group_calib_test.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
