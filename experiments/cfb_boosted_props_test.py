"""Pre-set test: a boosted-trees step for the college player projections (Nov 17, 2026).

    python experiments/cfb_boosted_props_test.py        (PFF data at ../pff-data)

Rule written before running (Project doc claude/plan-cfb-boosted-props-test-2026-11-17.md):

One candidate: a gradient-boosted last step after the college chain (cfbmodel.players.walk_forward),
built like nflmodel/players_ml.py (same estimator and settings, loss="poisson"), for RB carries and
rushing yards and WR/RB receptions and receiving yards. QBs and TEs stay on the chain (unchanged, so
not scored). Walk-forward by season on earlier seasons only (trained on played rows with
active == 1 of the same table kind). Inputs: the chain's projection and parts, shares, team volume,
game context, defense, PFF route/snap share and the kept PFF pair, defense vs position, and role-
change columns (teammates' raw shares, rank in the position group, last game's share minus the
decayed share, games this season, new team). The board's college prop widths, unchanged.

Stage 1: RMSE boosted minus current, paired by player-game, clustered by game, tuning 2017-19 and
  holdout 2021-25, game-day and bet-time projections, on the four stats (16 measures).
Stage 2: Brier at the DK/FD main lines, the stats it changes pooled (receiving yards WR/RB, rushing
  yards RB; college books post no receptions or carries main lines here), 2024 and 2025, game-day and
  bet-time (4 measures). Reported only: each stat, model minus books, floor rungs, the role-change
  slice.
Kept only if better on all 20, each holdout RMSE gain at least 1 SE. Writes scores only
(experiments/out/cfb_boosted_props_test.json), never PFF rows.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from cfbmodel import players as PL  # noqa: E402
from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_dvp_test as DV  # noqa: E402
import cfb_pff_role_test as T  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402
from nflmodel import players_ml as NML  # noqa: E402

OUT = HERE / "out" / "cfb_boosted_props_test.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
TUNE, HOLD = [2017, 2018, 2019], [2021, 2022, 2023, 2024, 2025]
ADOPT = {"carries": ("RB",), "rush_yds": ("RB",), "receptions": ("WR", "RB"), "rec_yds": ("WR", "RB")}
TRAIN_GROUPS = {"carries": ("RB", "QB"), "rush_yds": ("RB", "QB"), "receptions": ("WR", "TE", "RB"), "rec_yds": ("WR", "TE", "RB")}
SCORE = {"rec_yds": 15.0, "receptions": 1.5, "rush_yds": 20.0, "carries": 4.0}
VOL = {"rec_yds": "receptions", "rush_yds": "carries"}
LINE_GROUPS = {"rec_yds": ("WR", "RB"), "rush_yds": ("RB",)}

BASE_F = ["{s}_p", "{s}_proj", "{s}_recent", "dvpf_{s}", "car_share", "rec_share", "car_share_raw", "rec_share_raw", "proj_team_att",
          "proj_team_car", "team_comp_rate", "imp_pts", "margin", "proj_total", "wind", "home", "def_ypc", "def_ypa", "def_comp",
          "one_sum", "n_games", "ypr", "ypc", "route_share", "snap_share", "pff_yprr", "pff_opp_cov"]
ROLE_F = ["mate_car_tot", "mate_rec_tot", "mate_car_max", "mate_rec_max", "rank_car", "rank_rec", "last_car", "last_rec",
          "jump_car", "jump_rec", "games_season", "new_team"]
FEATS = BASE_F + ROLE_F


# ------------------------------------------------------------------ features

def history(t: pd.DataFrame) -> pd.DataFrame:
    """One row per played appearance: the player's actual shares that game (outcomes of earlier games)."""
    h = t[t["played"].astype(bool) & (t["appeared"] == 1)][["player_id", "team", "season", "t", "carries", "receptions", "team_car", "team_comp"]].copy()
    h["h_car"] = h["carries"] / h["team_car"].where(h["team_car"] > 0)
    h["h_rec"] = h["receptions"] / h["team_comp"].where(h["team_comp"] > 0)
    h = h.sort_values(["player_id", "t"])
    h["h_n_season"] = h.groupby(["player_id", "season"]).cumcount() + 1      # appearances this season, this one included
    return h[["player_id", "t", "team", "season", "h_car", "h_rec", "h_n_season"]].rename(columns={"team": "h_team", "season": "h_season", "t": "h_t"})


def add_features(p: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["_i"] = np.arange(len(p))
    for st in ADOPT:
        p[st + "_recent"] = p[st + "_sum"] / p["one_sum"].replace(0, np.nan)
        if "dvpf_" + st not in p:
            p["dvpf_" + st] = np.nan
    # teammates in the same position group who are expected to play (active), by raw share
    act = p["active"] == 1
    for c, raw in (("car", "car_share_raw"), ("rec", "rec_share_raw")):
        v = p[raw].where(act, 0.0).fillna(0.0)
        key = [p["game_id"], p["team"], p["group"]]
        tot = v.groupby(key).transform("sum")
        mx = v.groupby(key).transform("max")
        p[f"mate_{c}_tot"] = tot - v
        # the largest teammate share (the second largest when this player is the largest)
        rk = v.groupby(key).rank(ascending=False, method="first")
        sec = v.where(rk == 2).groupby(key).transform("max").fillna(0.0)
        p[f"mate_{c}_max"] = np.where(rk == 1, sec, mx)
        p[f"rank_{c}"] = v.groupby(key).rank(ascending=False, method="min")
    # last appearance before this game
    q = p[["_i", "player_id", "t", "team", "season"]].sort_values("t")
    hh = hist.sort_values("h_t")
    m = pd.merge_asof(q, hh, left_on="t", right_on="h_t", by="player_id", allow_exact_matches=False, direction="backward")
    m = m.set_index("_i").reindex(p["_i"])
    p["last_car"] = m["h_car"].to_numpy()
    p["last_rec"] = m["h_rec"].to_numpy()
    p["jump_car"] = p["last_car"] - p["car_share_sum"] / p["one_sum"].replace(0, np.nan)
    p["jump_rec"] = p["last_rec"] - p["rec_share_sum"] / p["one_sum"].replace(0, np.nan)
    p["games_season"] = np.where(m["h_season"].to_numpy() == p["season"].to_numpy(), m["h_n_season"].to_numpy(), 0)
    p["new_team"] = (m["h_team"].notna().to_numpy() & (m["h_team"].to_numpy() != p["team"].to_numpy())).astype(int)
    return p.drop(columns="_i")


def boosted(p: pd.DataFrame) -> pd.DataFrame:
    """p with <stat>_proj replaced, for the adopted stats and groups, by the boosted prediction from a
    model trained on earlier seasons (NFL estimator and settings). Rows with no earlier training season
    keep the chain's projection."""
    p = p.copy()
    # trained on players who appeared, as the chain's calibration is (players.walk_forward: hist
    # appeared == 1). The first run trained on active == 1, which at college also holds expected
    # players who never appeared (zeros), and pulled every projection down (process note, results doc).
    played = p["played"].astype(bool) & (p["appeared"] == 1)
    for stat, groups in ADOPT.items():
        cols = [f.format(s=stat) for f in FEATS]
        new = p[stat + "_proj"].copy()
        for s in sorted(p["season"].unique()):
            tr = p[played & (p["season"] < s) & p["group"].isin(TRAIN_GROUPS[stat])]
            te = p[(p["season"] == s) & p["group"].isin(groups)]
            if len(tr) < 1000 or te.empty:
                continue
            Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
            empty = np.isnan(Xtr).all(axis=0)          # a column with no history yet (e.g. defense vs position in 2017)
            Xtr[:, empty], Xte[:, empty] = 0.0, 0.0    # carries no information; sklearn cannot bin an all-missing column
            m = NML._model().fit(Xtr, tr[stat].clip(lower=0).to_numpy(float))
            new.loc[te.index] = m.predict(Xte)
        p[stat + "_proj"] = new
    return p


# ------------------------------------------------------------------ stage 1

def stage1(cur, new, kind) -> dict:
    out = {}
    for st, floor in SCORE.items():      # TEs and QBs are not changed by the step, so not scored
        out[f"{st} {kind} judged"] = T.rmse(cur, new, st, list(ADOPT[st]), floor, {"tuning 2017-19": TUNE, "holdout 2021-25": HOLD})
    return out


# ------------------------------------------------------------------ stage 2

def prop_rows(cur, new, pinfo, games, stat, season, alt=False):
    """Main-line props (or over-only alternate rungs) for stat in season, both projections priced."""
    R.set_season(season)
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    main = s[~s["market"].str.endswith("_alternate") & (s["stat"] == stat)]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    key = ["game_id", "player_id"]
    vol = VOL[stat]
    rows = R.player_frame(cur, pinfo).merge(new[key + [stat + "_proj", vol + "_proj"]].rename(columns={stat + "_proj": "n_s", vol + "_proj": "n_v"}), on=key, how="left")
    rows = rows[rows["group"].isin(LINE_GROUPS[stat])]
    base = w.merge(rows, on=["game_id", "nname"], how="inner")
    model = P.PropModel.load(PARAMS)
    outs = {}
    for name in ("current", "new"):
        dd = base.copy()
        if name == "new":
            dd[stat + "_proj"], dd[vol + "_proj"] = dd["n_s"], dd["n_v"]
        x = R.model_chances(dd, model)
        outs[name] = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
    idx = outs["current"].index.intersection(outs["new"].index)
    a, b = outs["current"].loc[idx], outs["new"].loc[idx]
    return pd.DataFrame({"game_id": a["game_id"].to_numpy(), "player_id": a["player_id"].to_numpy(), "stat": stat,
                         "hit": (a["y"] > a["line"]).astype(float).to_numpy(), "line": a["line"].to_numpy(),
                         "proj_cur": a[stat + "_proj"].to_numpy(), "p_cur": (a["our_over"] / (a["our_over"] + a["our_under"])).to_numpy(),
                         "p_new": (b["our_over"] / (b["our_over"] + b["our_under"])).to_numpy(), "books": a["fair"].to_numpy()})


def brier_block(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"props": 0}
    d, se = R.clustered_diff(x["p_new"].to_numpy(), x["p_cur"].to_numpy(), x["hit"].to_numpy(), x["game_id"].to_numpy())
    dc, sc = R.clustered_diff(x["p_cur"].to_numpy(), x["books"].to_numpy(), x["hit"].to_numpy(), x["game_id"].to_numpy())
    dn, sn = R.clustered_diff(x["p_new"].to_numpy(), x["books"].to_numpy(), x["hit"].to_numpy(), x["game_id"].to_numpy())
    return {"props": int(len(x)), "games": int(x["game_id"].nunique()),
            "brier_current": round(D.brier(x["p_cur"].to_numpy(), x["hit"].to_numpy()), 5),
            "brier_new": round(D.brier(x["p_new"].to_numpy(), x["hit"].to_numpy()), 5),
            "new_minus_current": round(d, 6), "se": round(se, 6), "z": round(d / se, 2) if se else None,
            "current_minus_books": round(dc, 5), "new_minus_books": round(dn, 5), "books_se": round(sn, 5)}


def teammate_out(t: pd.DataFrame) -> set:
    """(game_id, team, group) of played games where a regular (decayed share >= 0.2 of carries or
    0.12 of catches) who was expected to play did not appear (post hoc)."""
    x = t[t["played"].astype(bool) & (t["expected"] == 1) & (t["appeared"] == 0)]
    reg = (x["car_share_raw"] >= 0.2) | (x["rec_share_raw"] >= 0.12)
    return set(map(tuple, x.loc[reg, ["game_id", "team", "group"]].to_numpy()))


def stage2(cur_gd, new_gd, cur_bt, new_bt, pinfo, games, t) -> dict:
    out, judged = {}, {}
    out_set = teammate_out(t)
    tg = t.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])[["team", "group"]]
    for season in (2024, 2025):
        for kind, cur, new in (("game-day", cur_gd, new_gd), ("bet-time", cur_bt, new_bt)):
            parts = [prop_rows(cur, new, pinfo, games, st, season) for st in VOL]
            x = pd.concat(parts, ignore_index=True)
            judged[f"{season} {kind} pooled"] = brier_block(x)
            for st in VOL:
                out[f"{season} {kind} {st}"] = brier_block(x[x["stat"] == st])
            off = (np.abs(x["proj_cur"] - x["line"]) / x["line"].clip(lower=1)) >= 0.3
            out[f"{season} {kind} role slice: projection 30%+ off the line"] = brier_block(x[off])
            k = tg.reindex(pd.MultiIndex.from_frame(x[["game_id", "player_id"]]))
            mate = np.array([(g, a, b) in out_set for g, a, b in zip(x["game_id"], k["team"], k["group"])])
            out[f"{season} {kind} role slice: a regular teammate sat out"] = brier_block(x[mate])
        for st in VOL:
            R.set_season(season)
            fr = T.floor_rungs(cur_gd, new_gd, pinfo, games, st, VOL[st])
            out[f"{season} floor rungs {st} (game-day)"] = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in fr.items()}
    return {"judged": judged, "reported": out}


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    t, pinfo, games = G.table()
    if not (t.get("route_share", 0) > 0).any():
        sys.exit("no PFF data (check out rhyman7/pff-data at ../pff-data)")
    hist = history(t)
    tb = t.copy()
    m = tb["played"].astype(bool)
    tb.loc[m, "active"] = tb.loc[m, "expected"].astype(int)
    res = {"plan": "claude/plan-cfb-boosted-props-test-2026-11-17.md", "settings": "nflmodel.players_ml._model() as of Nov 17, 2026, unchanged",
           "features": FEATS, "stage1": {}, "stage2": {}}
    proj = {}
    for kind, table in (("game-day", t), ("bet-time", tb)):
        cache = Path(f"/tmp/cfb_boost_cur_{kind}.pkl")
        if cache.exists():
            cur = pd.read_pickle(cache)
        else:
            cur = PL.walk_forward(table)
            cur.to_pickle(cache)
        cur = add_features(cur, hist)
        new = boosted(cur)
        proj[kind] = (cur, new)
        res["stage1"].update(stage1(cur, new, kind))
        print(kind, "stage 1 done", round(time.time() - t0), "s", flush=True)
        for k, v in res["stage1"].items():
            if kind in k:
                print(" ", k, {p: (round(x["rmse_current"], 3), round(x["rmse_new"], 3), round(x["z"], 2)) for p, x in v.items()}, flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
    res["stage2"] = stage2(*proj["game-day"], *proj["bet-time"], pinfo, games, t)
    for k, v in res["stage2"]["judged"].items():
        print(k, v, flush=True)
    s1 = {k: v for k, v in res["stage1"].items() if "judged" in k}
    ok1 = all(v["tuning 2017-19"]["mse_diff"] < 0 and v["holdout 2021-25"]["z"] <= -1 for v in s1.values())
    ok2 = all(v["new_minus_current"] < 0 for v in res["stage2"]["judged"].values())
    res["measures"] = {"stage1": len(s1) * 2, "stage2": len(res["stage2"]["judged"])}
    res["kept"] = bool(ok1 and ok2)
    res["stage1_pass"], res["stage2_pass"] = bool(ok1), bool(ok2)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("stage 1", ok1, "stage 2", ok2, "kept", res["kept"], round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
