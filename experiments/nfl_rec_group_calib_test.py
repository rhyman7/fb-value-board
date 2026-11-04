"""NFL receiving calibration per position group (the college fix of #175, tested on the NFL).

    python experiments/nfl_rec_group_calib_test.py

On the NFL board the linear calibration is live for tight ends' receiving; receivers' and backs'
receptions and receiving yards come from the boosted step (players_ml.ADOPT), to which the
linear projection is one input.

Rules, written before running (Nov 3, 2026):
1. Change: nflmodel.players.walk_forward fits receptions and rec_yds per group (WR; TE; RB),
   everything else as is (groups not listed keep the pooled fit).
2. Score A, tight ends (linear output live): RMSE of receptions and receiving yards for TEs with
   chain or projection >= 2 receptions / 20 yards, tuning 2014-2018 and holdout 2019 on, paired
   difference clustered by game. WR and RB through the boosted step (players_ml.walk_forward from
   2019 on both tables), holdout only, as the no-harm check.
3. Score B, the lines: Brier at the 2023-2025 DraftKings/FanDuel closing receptions and
   receiving-yards main lines, TEs with the linear output; WR and RB with the boosted output as
   the no-harm check. (The history holds no receptions lines, so receptions are judged on RMSE.)
4. Keep only if TE receiving-yards Brier is better by 2+ SE, or TE receiving-yards RMSE is better
   on both periods with the holdout by 2+ SE; and nothing (TE receptions, boosted WR/RB RMSE,
   any Brier) is worse by 2 SE. Otherwise nothing changes.

Writes experiments/out/nfl_rec_group_calib_test.json.
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
GROUP_STATS = ("receptions", "rec_yds")
CALIB_GROUPS = (("WR",), ("TE",), ("RB",))
THRESH = {"receptions": 2.0, "rec_yds": 20.0}
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
                cur[stat + "_proj"] = _calibrate(hist, cur, stat)
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


def rmse_block(x, new_col, stat, old_col=None):
    old_col = old_col or stat + "_proj"
    out = {}
    for per, g in (("tuning 2014-18", x[x.season < C.HOLDOUT_FROM]), ("holdout 2019+", x[x.season >= C.HOLDOUT_FROM])):
        if g.empty:
            continue
        e_old = (g[old_col] - g[stat]).to_numpy() ** 2
        e_new = (g[new_col] - g[stat]).to_numpy() ** 2
        diff, se = paired(e_new, e_old, g.game_id)
        out[per] = {"n": int(len(g)), "rmse_current": float(np.sqrt(e_old.mean())), "rmse_new": float(np.sqrt(e_new.mean())),
                    "mse_diff": diff, "se": se, "z": diff / se if se else 0.0,
                    "bias_current": float((g[old_col] - g[stat]).mean()), "bias_new": float((g[new_col] - g[stat]).mean())}
    return out


MARKET = {"receptions": "player_receptions", "rec_yds": "player_reception_yds"}


def brier_block(rows: pd.DataFrame, group: str, new_col: str, model, stat: str) -> dict:
    """Brier at the closing lines of one stat, new vs current center, one group."""
    parts = []
    for season in SEASONS:
        snaps = B.load(season, "close")
        if snaps.empty:
            continue
        snaps = snaps[snaps["market"] == MARKET[stat]]
        if snaps.empty:
            continue
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
        w["stat"] = stat
        parts.append(w)
    if not parts:
        return {"props": 0}
    base = pd.concat(parts, ignore_index=True)
    ps = {}
    for name, swap in (("current", False), ("new", True)):
        d = base.copy()
        if swap:
            d["rec_yds_proj"], d["receptions_proj"] = d["new_rec_yds_proj"], d["new_receptions_proj"]
        d = d[P.eligible(d, stat, group)].reset_index(drop=True)
        over, push, under = model.probs(d, stat, group, d["line"].to_numpy(float))
        d["p"] = over / (over + under)
        d = d[d[stat] != d["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
        ps[name] = d
    idx = ps["current"].index.intersection(ps["new"].index)
    a, b = ps["current"].loc[idx], ps["new"].loc[idx]
    hit = (a[stat] > a["line"]).astype(float).to_numpy()
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
    cols = ["receptions_proj", "rec_yds_proj"]
    ren = {c: "new_" + c for c in cols}
    m = cur.merge(new[key + cols].rename(columns=ren), on=key)
    res = {"rule": "keep only if TE rec-yards Brier better by 2+ SE or TE rec-yards RMSE better both periods (holdout 2+ SE); nothing worse by 2 SE", "backtest": {}}
    te = m[(m.group == "TE") & (m.active == 1) & m.played.astype(bool)]
    for stat in GROUP_STATS:
        x = te[te[stat].notna() & ((te[stat + "_p"] >= THRESH[stat]) | (te[stat + "_proj"] >= THRESH[stat]))]
        res["backtest"]["TE " + stat] = rmse_block(x, "new_" + stat + "_proj", stat)
    print("boosted step, current inputs ...")
    games, _, _ = data.load_all(refresh=False, verbose=False)
    R = roles.load_all(games, verbose=False)
    ml_cur = ML.walk_forward(ML.add_features(cur, R), first=C.HOLDOUT_FROM)
    print("boosted step, grouped inputs ...")
    ml_new = ML.walk_forward(ML.add_features(new, R), first=C.HOLDOUT_FROM)
    mm = ml_cur.merge(ml_new[key + cols].rename(columns=ren), on=key)
    for grp in ("WR", "RB"):
        g = mm[(mm.group == grp) & (mm.active == 1) & mm.played.astype(bool) & (mm.season >= C.HOLDOUT_FROM)]
        for stat in GROUP_STATS:
            x = g[g[stat].notna() & ((g[stat + "_p"] >= THRESH[stat]) | (g[stat + "_proj"] >= THRESH[stat]))]
            res["backtest"][f"{grp} {stat} (boosted)"] = rmse_block(x, "new_" + stat + "_proj", stat)
    model = P.PropModel.load()
    res["lines"] = {}
    for stat in GROUP_STATS:
        res["lines"][f"TE {stat}"] = brier_block(m, "TE", "new_" + stat + "_proj", model, stat)
        for grp in ("WR", "RB"):
            res["lines"][f"{grp} {stat} (boosted)"] = brier_block(mm, grp, "new_" + stat + "_proj", model, stat)
    bt, ln = res["backtest"], res["lines"]
    te_y = bt["TE rec_yds"]
    gain = (ln["TE rec_yds"].get("props", 0) > 0 and ln["TE rec_yds"]["new_minus_current"] <= -2 * ln["TE rec_yds"]["se"]) or \
        (all(v["mse_diff"] < 0 for v in te_y.values()) and te_y["holdout 2019+"]["z"] <= -2)
    no_harm = all(v["z"] < 2 for b in bt.values() for v in b.values()) and \
        all(v.get("props", 0) == 0 or v["new_minus_current"] < 2 * v["se"] for v in ln.values())
    res["kept"] = bool(gain and no_harm)
    OUT.mkdir(exist_ok=True)
    (OUT / "nfl_rec_group_calib_test.json").write_text(json.dumps(res, indent=1, default=float))
    for k, b in bt.items():
        for per, v in b.items():
            print(f"{k:26s} {per:15s} n={v['n']:6d} rmse {v['rmse_current']:.3f} -> {v['rmse_new']:.3f} z {v['z']:+.2f} bias {v['bias_current']:+.2f} -> {v['bias_new']:+.2f}")
    for k, v in ln.items():
        if v.get("props"):
            print(f"{k:26s} props={v['props']:5d} brier {v['brier_current']:.4f} -> {v['brier_new']:.4f} (books {v['brier_books']:.4f}) z {v['new_minus_current'] / v['se']:+.2f}")
    print("kept:", res["kept"])


if __name__ == "__main__":
    main()
