"""College rushing calibration per position group.

    python experiments/cfb_rush_group_calib_test.py

Nov 3, 2026. The calibration (cfbmodel.players.walk_forward) fits one regression per stat on
every player who appeared; for rushing yards that pool is mostly receivers and quarterbacks
with a few carries, and it settles on about 0.71 x chain + 0.43 x recent. Fitted on running
backs alone the weights are about 0.76 / 0.40, on prop-relevant backs 0.82 / 0.30, stable
2019-2025 (diagnostic, this session). Passing already fits on quarterbacks only.

Candidate: fit rush_yds and carries per group (RB; QB; WR+TE pooled), everything else as is.

Rules, written before running:
1. Score A, the backtest: RMSE of RB rushing yards (chain or projection >= 20) against actual,
   tuning 2017-2019 and holdout 2021 on, paired difference from the current projection with
   standard errors clustered by game. RMSE so a lower center cannot win on skew alone. Bias
   (mean projection minus actual) reported.
2. Score B, the lines: Brier of the over chance at the 2025 DraftKings/FanDuel rushing-yards
   main lines (cfb_rush_center_test.py's frame), candidate vs current, clustered by game.
3. Keep only if RMSE improves on both periods, on the holdout by 2+ standard errors, and the
   Brier at the lines is no worse (difference <= 0). Otherwise nothing changes.

Writes experiments/out/cfb_rush_group_calib_test.json.
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
GROUP_STATS = ("rush_yds", "carries")
GROUPS = {"RB": ("RB",), "QB": ("QB",), "WRTE": ("WR", "TE")}


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
    cur = pd.read_pickle("/tmp/cfb_pproj_all.pkl") if Path("/tmp/cfb_pproj_all.pkl").exists() else PL.walk_forward(t)
    new = walk_forward_grouped(t)
    key = ["game_id", "player_id"]
    m = cur[key + ["season", "group", "appeared", "played", "rush_yds", "rush_yds_p", "rush_yds_proj", "carries_proj"]].merge(
        new[key + ["rush_yds_proj", "carries_proj"]].rename(columns={"rush_yds_proj": "new_proj", "carries_proj": "new_car"}), on=key)
    d = m[(m.group == "RB") & (m.appeared == 1) & m.played.astype(bool) & m.rush_yds.notna()]
    d = d[(d.rush_yds_p >= 20) | (d.rush_yds_proj >= 20)]
    res = {"rule": "keep only if RMSE improves on both periods, holdout by 2+ SE, and Brier at 2025 lines is no worse", "backtest": {}}
    for per, x in (("tuning 2017-19", d[d.season <= 2019]), ("holdout 2021+", d[d.season >= 2021])):
        e_old = (x.rush_yds_proj - x.rush_yds).to_numpy() ** 2
        e_new = (x.new_proj - x.rush_yds).to_numpy() ** 2
        diff, se = paired(e_new, e_old, x.game_id.to_numpy())
        res["backtest"][per] = {"n": int(len(x)), "rmse_current": float(np.sqrt(e_old.mean())), "rmse_new": float(np.sqrt(e_new.mean())),
                                "mse_diff": diff, "se": se, "z": diff / se if se else 0.0,
                                "bias_current": float((x.rush_yds_proj - x.rush_yds).mean()), "bias_new": float((x.new_proj - x.rush_yds).mean()),
                                "mae_current": float(np.abs(x.rush_yds_proj - x.rush_yds).mean()), "mae_new": float(np.abs(x.new_proj - x.rush_yds).mean())}
    # Score B: the 2025 lines
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    rows = R.player_frame(cur, pinfo).merge(new[key + ["rush_yds_proj", "carries_proj"]].rename(columns={"rush_yds_proj": "new_proj", "carries_proj": "new_car"}), on=key, how="left")
    main_ = s[~s["alt"] & (s["stat"] == "rush_yds")]
    w = main_.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    base = w.merge(rows, on=["game_id", "nname"], how="inner")
    model = P.PropModel.load()
    outs = {}
    for name, cols in (("current", None), ("new", ("new_proj", "new_car"))):
        dd = base.copy()
        if cols:
            dd["rush_yds_proj"], dd["carries_proj"] = dd[cols[0]], dd[cols[1]]
        x = R.model_chances(dd, model)
        x = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
        outs[name] = x
    idx = outs["current"].index.intersection(outs["new"].index)
    a, b = outs["current"].loc[idx], outs["new"].loc[idx]
    hit = (a["y"] > a["line"]).astype(float).to_numpy()
    pa = (a["our_over"] / (a["our_over"] + a["our_under"])).to_numpy()
    pb = (b["our_over"] / (b["our_over"] + b["our_under"])).to_numpy()
    diff, se = R.clustered_diff(pb, pa, hit, a.index.get_level_values(0).to_numpy())
    res["lines_2025"] = {"props": int(len(idx)), "brier_books": D.brier(a["fair"].to_numpy(), hit), "brier_current": D.brier(pa, hit),
                         "brier_new": D.brier(pb, hit), "new_minus_current": diff, "se": se}
    bt = res["backtest"]
    res["kept"] = bool(all(v["mse_diff"] < 0 for v in bt.values()) and bt["holdout 2021+"]["z"] <= -2 and res["lines_2025"]["new_minus_current"] <= 0)
    OUT.mkdir(exist_ok=True)
    (OUT / "cfb_rush_group_calib_test.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
