"""College receiving calibration per position group (the rushing fix of Nov 3, 2026, applied to
receptions and receiving yards).

    python experiments/cfb_rec_group_calib_test.py

Receptions and receiving yards are calibrated on one pool of receivers, tight ends and running
backs. Backs catch short passes at a different rate than wideouts, so the blend that fits a WR1
need not fit a back, as receivers were dragging the backs' rushing weights.

Rules, written before running (Nov 3, 2026):
1. Change: fit receptions and rec_yds per group (WR; TE; RB), everything else as is. One change,
   one rule: both stats must pass or neither is kept.
2. Score A, the backtest: RMSE per group (WR, TE, RB), players with chain or projection >= 2
   receptions / 20 yards, tuning 2017-2019 and holdout 2021 on, paired difference from the
   current projection, clustered by game.
3. Score B, the lines: Brier at the 2025 DraftKings/FanDuel receiving-yards main lines
   (cfb_props_real_lines.py's frame), new vs current, by group. The 2025 history has no college
   receptions lines, so receptions is judged on RMSE alone.
4. Keep only if WR RMSE improves on both periods for both stats, the holdout by 2+ SE; TE and RB
   are no worse (not worse by 2 SE on either period, either stat); and the receiving-yards Brier is
   no worse for every group with 100+ props. Otherwise nothing changes.

Writes experiments/out/cfb_rec_group_calib_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from cfbmodel import config as C, players as PL  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
GROUP_STATS = ("receptions", "rec_yds")
GROUPS = {"WR": ("WR",), "TE": ("TE",), "RB": ("RB",)}
THRESH = {"receptions": 2.0, "rec_yds": 20.0}


def walk_forward_grouped(table: pd.DataFrame, first_test: int = C.FIRST_TEST_SEASON, lookback: int = 6) -> pd.DataFrame:
    """cfbmodel.players.walk_forward with rush_yds and carries calibrated per position group."""
    out = []
    tg = table.drop_duplicates(["game_id", "team"])
    tg = tg[tg["proj_total"].notna()]
    for s in sorted(table["season"].unique()):
        if s < first_test:
            continue
        tr = tg[(tg["season"] < s) & (tg["season"] >= s - 8) & tg["played"].astype(bool) & tg["team_att"].notna()]
        att = PL.Lin(5.0).fit(tr[PL.VOL_FEATS].fillna(0), tr["team_att"].to_numpy(float))
        car = PL.Lin(5.0).fit(tr[PL.VOL_FEATS].fillna(0), tr["team_car"].to_numpy(float))
        past = table[(table["season"] < s) & (table["season"] >= max(first_test - 2, s - lookback)) & table["proj_total"].notna()]
        hist = PL.raw_projection(past, att, car)
        hist = hist[hist["played"].astype(bool) & (hist["appeared"] == 1)]
        cur = PL.raw_projection(table[(table["season"] == s) & table["proj_total"].notna()], att, car)
        qb_fit = hist[(hist["att_share"] > 0.5) & (hist["started"] == 1)]
        for stat in PL.STATS:
            if stat in ("rush_yds", "carries"):   # rushing, per group, as production now does
                cur[stat + "_proj"] = 0.0
                for grps in PL.CALIB_GROUPS[stat]:
                    h = hist[hist["group"].isin(grps)]
                    sel = cur["group"].isin(grps).to_numpy()
                    if h.empty or not sel.any():
                        continue
                    m = PL.Lin(1.0).fit(PL.calib_X(h, stat), h[stat].to_numpy(float))
                    X = PL.calib_X(cur[sel], stat)
                    none = (X[:, 0] == 0) & (X[:, 1] == 0)
                    cur.loc[sel, stat + "_proj"] = np.where(none, 0.0, np.clip(m.predict(X), 0, None))
                continue
            if stat in GROUP_STATS:
                cur[stat + "_proj"] = 0.0
                for grps in GROUPS.values():
                    h = hist[hist["group"].isin(grps)]
                    sel = cur["group"].isin(grps).to_numpy()
                    if h.empty or not sel.any():
                        continue
                    m = PL.Lin(1.0).fit(PL.calib_X(h, stat), h[stat].to_numpy(float))
                    X = PL.calib_X(cur[sel], stat)
                    none = (X[:, 0] == 0) & (X[:, 1] == 0)
                    cur.loc[sel, stat + "_proj"] = np.where(none, 0.0, np.clip(m.predict(X), 0, None))
                continue
            h = qb_fit if stat in PL.PASS_STATS else hist
            m = PL.Lin(1.0).fit(PL.calib_X(h, stat), h[stat].to_numpy(float))
            X = PL.calib_X(cur, stat)
            none = (X[:, 0] == 0) & (X[:, 1] == 0)
            cur[stat + "_proj"] = np.where(none, 0.0, np.clip(m.predict(X), 0, None))
        out.append(cur)
    return pd.concat(out, ignore_index=True)


def table():
    from cfbmodel import data, model, players, players_data
    from cfbmodel.ratings import CFBRatingEngine
    cache = Path("/tmp/cfb_ptable.pkl")
    if cache.exists():
        return pd.read_pickle(cache), pd.read_pickle("/tmp/cfb_pinfo_all.pkl"), pd.read_pickle("/tmp/cfb_games_all.pkl")
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, engine))
    pg, pinfo = players_data.load_player_games(verbose=False)
    t = players.build(games, frame, pg, pinfo, None, None)
    t.to_pickle(cache); pinfo.to_pickle("/tmp/cfb_pinfo_all.pkl"); games.to_pickle("/tmp/cfb_games_all.pkl")
    return t, pinfo, games


def paired(err_new: np.ndarray, err_old: np.ndarray, game: np.ndarray) -> tuple[float, float]:
    """Mean difference and its standard error, clustered by game."""
    d = err_new - err_old
    cl = pd.Series(d - d.mean()).groupby(game).sum()
    return float(d.mean()), float(np.sqrt((cl ** 2).sum()) / len(d))


def main():
    t, pinfo, games = table()
    cur = PL.walk_forward(t)                      # production (rushing already per group)
    new = walk_forward_grouped(t)
    key = ["game_id", "player_id"]
    cols = ["receptions_proj", "rec_yds_proj"]
    m = cur[key + ["season", "group", "appeared", "played", "receptions", "rec_yds", "receptions_p", "rec_yds_p"] + cols].merge(
        new[key + cols].rename(columns={c: "new_" + c for c in cols}), on=key)
    res = {"rule": "keep only if WR RMSE improves both periods for both stats, holdout 2+ SE; TE and RB not worse by 2 SE; rec_yds Brier no worse per group (100+ props)",
           "backtest": {}}
    for stat in GROUP_STATS:
        for grp in GROUPS:
            d = m[(m.group == grp) & (m.appeared == 1) & m.played.astype(bool) & m[stat].notna()]
            d = d[(d[stat + "_p"] >= THRESH[stat]) | (d[stat + "_proj"] >= THRESH[stat])]
            for per, x in (("tuning 2017-19", d[d.season <= 2019]), ("holdout 2021+", d[d.season >= 2021])):
                e_old = (x[stat + "_proj"] - x[stat]).to_numpy() ** 2
                e_new = (x["new_" + stat + "_proj"] - x[stat]).to_numpy() ** 2
                diff, se = paired(e_new, e_old, x.game_id.to_numpy())
                res["backtest"][f"{stat} {grp} {per}"] = {"n": int(len(x)), "rmse_current": float(np.sqrt(e_old.mean())), "rmse_new": float(np.sqrt(e_new.mean())),
                                                           "mse_diff": diff, "se": se, "z": diff / se if se else 0.0,
                                                           "bias_current": float((x[stat + "_proj"] - x[stat]).mean()), "bias_new": float((x["new_" + stat + "_proj"] - x[stat]).mean())}
    # Score B: 2025 receiving-yards lines, by group
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    rows = R.player_frame(cur, pinfo).merge(new[key + cols].rename(columns={c: "new_" + c for c in cols}), on=key, how="left")
    main_ = s[~s["alt"] & (s["stat"] == "rec_yds")]
    w = main_.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    base = w.merge(rows, on=["game_id", "nname"], how="inner")
    model = P.PropModel.load()
    outs = {}
    for name, swap in (("current", False), ("new", True)):
        dd = base.copy()
        if swap:
            dd["rec_yds_proj"], dd["receptions_proj"] = dd["new_rec_yds_proj"], dd["new_receptions_proj"]
        x = R.model_chances(dd, model)
        x = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
        outs[name] = x
    idx = outs["current"].index.intersection(outs["new"].index)
    a, b = outs["current"].loc[idx], outs["new"].loc[idx]
    res["lines_2025"] = {}
    for grp in list(GROUPS) + ["all"]:
        sel = (a["group"] == grp).to_numpy() if grp != "all" else np.ones(len(a), bool)
        if sel.sum() == 0:
            continue
        aa, bb = a[sel], b[sel]
        hit = (aa["y"] > aa["line"]).astype(float).to_numpy()
        pa = (aa["our_over"] / (aa["our_over"] + aa["our_under"])).to_numpy()
        pb = (bb["our_over"] / (bb["our_over"] + bb["our_under"])).to_numpy()
        diff, se = R.clustered_diff(pb, pa, hit, aa["game_id"].to_numpy())
        res["lines_2025"][grp] = {"props": int(sel.sum()), "brier_books": D.brier(aa["fair"].to_numpy(), hit), "brier_current": D.brier(pa, hit),
                                 "brier_new": D.brier(pb, hit), "new_minus_current": diff, "se": se}
    bt = res["backtest"]
    wr_ok = all(bt[f"{st} WR {per}"]["mse_diff"] < 0 for st in GROUP_STATS for per in ("tuning 2017-19", "holdout 2021+")) \
        and all(bt[f"{st} WR holdout 2021+"]["z"] <= -2 for st in GROUP_STATS)
    others_ok = all(bt[f"{st} {g} {per}"]["z"] < 2 for st in GROUP_STATS for g in ("TE", "RB") for per in ("tuning 2017-19", "holdout 2021+"))
    lines_ok = all(v["new_minus_current"] <= 0 for g, v in res["lines_2025"].items() if g != "all" and v["props"] >= 100)
    res["kept"] = bool(wr_ok and others_ok and lines_ok)
    OUT.mkdir(exist_ok=True)
    (OUT / "cfb_rec_group_calib_test.json").write_text(json.dumps(res, indent=1, default=float))
    rows_ = [{"test": k, **{kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}} for k, v in bt.items()]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows_)[["test", "n", "rmse_current", "rmse_new", "z", "bias_current", "bias_new"]].to_string(index=False))
    print(json.dumps(res["lines_2025"], indent=1, default=float))
    print("kept:", res["kept"])


if __name__ == "__main__":
    main()
