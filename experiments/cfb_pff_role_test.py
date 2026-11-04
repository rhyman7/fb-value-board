"""PFF role inputs in the college player chains: routes per team pass play and snap share.

    PFF_DATA_DIR=../pff-data python experiments/cfb_pff_role_test.py [--walk-forward]

Nov 4, 2026. Data: PFF Premium Stats from the private repo rhyman7/pff-data (cfbmodel.pff_data;
never committed here): receiving_summary (routes) and offense_summary (snap counts) per player
per week. PFF rows cover every division; they are filtered to the board's FBS teams (the teams in
the game-model frame each season: 136 in 2025). PFF franchises are matched to board teams by the
players they share in a season, players by normalised name within a team-season, and weeks to
the board's games by team and week (checked by the receiving yards agreeing).

Features, as of each game (earlier games only, the board's decay: 8-game half-life, last season
at 60%, nothing older):
  route_share = sum w x routes / sum w x team pass plays     (team pass plays: the most pass snaps
                                                              any player of the team logged that week)
  snap_share  = sum w x offensive snaps / sum w x team offensive snaps
over the games the player was on the field; 0 with no history. Each enters the chain the way
CAL_EXTRA columns do (cfbmodel.players.calib_X: chain x feature as an extra regression column in
the last calibration step), per position group:
  receiving chain (receptions, rec_yds): route_share and snap_share
  rushing chain   (carries, rush_yds):   snap_share

Rule, written before running (memo "PFF role test", Nov 3; this docstring Nov 4, before any run):
  Fit on 2024 (calibration window one season: players.walk_forward lookback=1, first_test=2025,
  the same for the current chain so the PFF columns are the only difference). Score on 2025.
  Each chain is judged on its own; it is kept only if all three hold:
   1. RMSE on 2025 player-games better by 2+ SE (clustered by game): receiving yards for WR/TE/RB
      (appeared, chain or projection >= 15 yards); rushing yards for RBs (>= 20 yards).
   2. Brier at the 2025 DraftKings/FanDuel main lines (experiments/cfb_props_real_lines.py
      snapshots) better than the current chain by 2+ SE: receiving yards for the receiving chain,
      rushing yards for the rushing chain.
   3. Floor rungs (claude/betting-strategies.md #1): the 2025 over-only alternate rungs whose
      books' reading (nflmodel.parlays.CFB_RUNG_CAL) is 65-95%: the model's Brier better than
      the current chain's by 2+ SE.
  If a chain passes, the 2017-2025 walk-forward follows (--walk-forward): RMSE better on tuning
  (2017-19) and holdout (2021+), holdout by 2+ SE, before anything goes near the board. Ryan
  decides whether anything is applied.

Writes experiments/out/cfb_pff_role_test.json (scores only, no PFF rows).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from cfbmodel import players as PL, pff_data  # noqa: E402
from nflmodel import odds as O, parlays as PA, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_dvp_test as DV  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_role_prior_test as RP  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "cfb_pff_role_test.json"
HL, CARRY = 8.0, 0.6
TUNE = [2017, 2018, 2019]
REC_FEATS, RUSH_FEATS = ["route_share", "snap_share"], ["snap_share"]
FLOOR_BAND = (0.65, 0.95)


# ---------------------------------------------------------------- PFF rows onto the board's games

def pff_player_weeks(seasons) -> pd.DataFrame:
    rec = pff_data.load("receiving_summary", seasons)[["season", "week", "player", "player_id", "franchise_id", "team_name", "routes", "yards"]]
    off = pff_data.load("offense_summary", seasons)
    off = off[["season", "week", "player", "player_id", "franchise_id", "team_name", "snap_counts_total", "snap_counts_total_pass"]]
    team = off.groupby(["season", "week", "franchise_id"]).agg(team_snaps=("snap_counts_total", "max"),
                                                              team_pass=("snap_counts_total_pass", "max")).reset_index()
    pw = off.merge(rec[["season", "week", "player_id", "franchise_id", "routes", "yards"]], on=["season", "week", "player_id", "franchise_id"], how="outer")
    pw["player"] = pw["player"].fillna(pw["player_id"].map(rec.drop_duplicates("player_id").set_index("player_id")["player"]))
    pw = pw.merge(team, on=["season", "week", "franchise_id"], how="left")
    pw[["routes", "snap_counts_total"]] = pw[["routes", "snap_counts_total"]].fillna(0)
    pw["nname"] = pw["player"].map(PM.norm_name)
    return pw


def map_onto_board(pw: pd.DataFrame, t: pd.DataFrame, pinfo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """PFF player-weeks -> board (game_id, player_id), FBS teams only."""
    b = t[t["played"].astype(bool) & (t["appeared"] == 1)][["game_id", "team", "season", "week", "player_id", "rec_yds"]].copy()
    b["nname"] = b["player_id"].map(pinfo["name"]).map(PM.norm_name)
    fbs = t[t["proj_total"].notna()][["season", "team"]].drop_duplicates()
    # franchise -> board team, per season, by shared player names
    bn = b[["season", "team", "nname"]].drop_duplicates()
    pn = pw[["season", "franchise_id", "nname"]].drop_duplicates()
    ov = pn.merge(bn, on=["season", "nname"]).groupby(["season", "franchise_id", "team"]).size().rename("n").reset_index()
    ov = ov.sort_values("n", ascending=False)
    best = ov.drop_duplicates(["season", "franchise_id"]).drop_duplicates(["season", "team"])
    best = best[best["n"] >= 5].merge(fbs, on=["season", "team"])
    m = pw.merge(best[["season", "franchise_id", "team"]], on=["season", "franchise_id"])
    # PFF weeks onto board games: the team's game in that week
    games = b[["game_id", "team", "season", "week"]].drop_duplicates(["team", "season", "week"])
    m = m.merge(games, on=["team", "season", "week"])
    # players by name within team-season (names unique on both sides)
    bp = b[["season", "team", "nname", "player_id"]].drop_duplicates()
    bp = bp[~bp.duplicated(["season", "team", "nname"], keep=False)]
    m = m[~m.duplicated(["game_id", "team", "nname"], keep=False)]
    m = m.merge(bp.rename(columns={"player_id": "board_id"}), on=["season", "team", "nname"], how="left")
    chk = m.dropna(subset=["board_id", "yards"]).merge(b[["game_id", "player_id", "rec_yds"]].rename(columns={"player_id": "board_id"}),
                                                      on=["game_id", "board_id"], how="left")
    agree = float((chk["yards"] == chk["rec_yds"]).mean()) if len(chk) else float("nan")
    stats = {"pff_rows": int(len(pw)), "franchise_seasons_mapped": int(len(best)),
             "fbs_team_seasons": int(len(fbs)), "rows_on_fbs_games": int(len(m)),
             "rows_with_board_player": int(m["board_id"].notna().sum()),
             "receiving_yards_agree": agree}
    return m, stats


def asof(t: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """route_share and snap_share for each board row, from the player's earlier games only."""
    h = m.dropna(subset=["board_id"]).copy()
    h["board_id"] = h["board_id"].astype(np.int64)
    h = h.merge(t[["game_id", "t"]].drop_duplicates("game_id"), on="game_id")
    h = h.sort_values(["board_id", "t"])
    keyed = {pid: (g["t"].to_numpy(), g["season"].to_numpy(), g["routes"].to_numpy(float), g["team_pass"].to_numpy(float),
                   g["snap_counts_total"].to_numpy(float), g["team_snaps"].to_numpy(float)) for pid, g in h.groupby("board_id")}
    n = len(t)
    rs, ss = np.zeros(n), np.zeros(n)
    tt, pid, sea = t["t"].to_numpy(), t["player_id"].to_numpy(), t["season"].to_numpy()
    for i in range(n):
        k = keyed.get(pid[i])
        if k is None:
            continue
        ht, hs, ro, tp, sn, ts = k
        sel = (ht < tt[i]) & (hs >= sea[i] - 1)
        if not sel.any():
            continue
        j = np.arange(sel.sum())[::-1]
        w = 0.5 ** (j / HL) * np.where(hs[sel] == sea[i], 1.0, CARRY)
        den = (w * np.nan_to_num(tp[sel])).sum()
        rs[i] = (w * ro[sel]).sum() / den if den > 0 else 0.0
        den = (w * np.nan_to_num(ts[sel])).sum()
        ss[i] = (w * sn[sel]).sum() / den if den > 0 else 0.0
    out = t.copy()
    out["route_share"], out["snap_share"] = np.clip(rs, 0, 1.2), np.clip(ss, 0, 1.2)
    return out


# ---------------------------------------------------------------- scoring

def rmse(cur, new, stat, groups, floor, seasons):
    key = ["game_id", "player_id"]
    m = cur[key + ["season", "group", "appeared", "played", stat, stat + "_p", stat + "_proj"]].merge(
        new[key + [stat + "_proj"]].rename(columns={stat + "_proj": "new"}), on=key)
    d = m[m.group.isin(groups) & (m.appeared == 1) & m.played.astype(bool) & m[stat].notna()
          & ((m[stat + "_p"] >= floor) | (m[stat + "_proj"] >= floor))]
    return {k: DV.paired_rmse(d[d.season.isin(v)], stat, "new") for k, v in seasons.items()}


def floor_rungs(cur, new, pinfo, games, stat, vol) -> dict:
    """2025 over-only alternate rungs with the books' reading (CFB_RUNG_CAL) in FLOOR_BAND: the
    model's Brier with the new projections minus with the current ones, clustered by game."""
    key = ["game_id", "player_id"]
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    a = s[s["market"].str.endswith("_alternate") & (s["side"] == "over") & (s["stat"] == stat)]
    a = a.groupby(["game_id", "nname", "stat", "line"], as_index=False)["price"].max()
    a["implied"] = O.implied_prob(a["price"])
    a["books"] = np.interp(a["implied"], *zip(*PA.CFB_RUNG_CAL))
    a = a[a["books"].between(*FLOOR_BAND)]
    rows = R.player_frame(cur, pinfo).merge(new[key + [stat + "_proj", vol + "_proj"]].rename(columns={stat + "_proj": "new_proj", vol + "_proj": "new_vol"}), on=key, how="left")
    base = a.merge(rows, on=["game_id", "nname"], how="inner")
    model = P.PropModel.load()
    outs = {}
    for name, cols in (("current", None), ("new", ("new_proj", "new_vol"))):
        dd = base.copy()
        if cols:
            dd[stat + "_proj"], dd[vol + "_proj"] = dd[cols[0]], dd[cols[1]]
        x = R.model_chances(dd, model)
        outs[name] = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
    idx = outs["current"].index.intersection(outs["new"].index)
    a_, b_ = outs["current"].loc[idx], outs["new"].loc[idx]
    hit = (a_["y"] > a_["line"]).astype(float).to_numpy()
    pa = (a_["our_over"] / (a_["our_over"] + a_["our_under"])).to_numpy()
    pb = (b_["our_over"] / (b_["our_over"] + b_["our_under"])).to_numpy()
    g = a_.index.get_level_values(0).to_numpy()
    diff, se = R.clustered_diff(pb, pa, hit, g)
    dbk, sebk = R.clustered_diff(pb, a_["books"].to_numpy(), hit, g)
    dck, seck = R.clustered_diff(pa, a_["books"].to_numpy(), hit, g)
    return {"rungs": int(len(idx)), "games": int(len(set(g))), "cleared": float(hit.mean()),
            "books_reading": float(a_["books"].mean()), "current_mean": float(pa.mean()), "new_mean": float(pb.mean()),
            "brier_books": D.brier(a_["books"].to_numpy(), hit), "brier_current": D.brier(pa, hit), "brier_new": D.brier(pb, hit),
            "new_minus_current": diff, "se": se, "z": diff / se if se else 0.0,
            "current_minus_books": dck, "current_books_se": seck, "new_minus_books": dbk, "new_books_se": sebk}


def run_chain(t, feats: dict, first_test: int, lookback: int):
    saved = dict(PL.CAL_EXTRA)
    PL.CAL_EXTRA = {**saved, **feats}
    try:
        return PL.walk_forward(t, first_test=first_test, lookback=lookback)
    finally:
        PL.CAL_EXTRA = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-forward", action="store_true", help="also run the 2017-2025 walk-forward")
    a = ap.parse_args()
    t, pinfo, games = G.table()
    seasons = list(range(2016, 2026))
    pw = pff_player_weeks([s for s in seasons if s >= 2017])
    m, mstats = map_onto_board(pw, t, pinfo)
    print("mapping", mstats, flush=True)
    tf = asof(t, m)
    cov = tf[(tf.season == 2025) & (tf.appeared == 1) & tf.group.isin(["WR", "TE", "RB"])]
    res = {"mapping": mstats, "coverage_2025": {"rows": int(len(cov)), "with_route_share": float((cov.route_share > 0).mean()),
                                                "with_snap_share": float((cov.snap_share > 0).mean())}}
    print("coverage", res["coverage_2025"], flush=True)
    cur = run_chain(tf, {}, 2025, 1)
    rec_new = run_chain(tf, {"receptions": REC_FEATS, "rec_yds": REC_FEATS}, 2025, 1)
    rush_new = run_chain(tf, {"carries": RUSH_FEATS, "rush_yds": RUSH_FEATS}, 2025, 1)
    s25 = {"2025": [2025]}
    rec = {"rmse": rmse(cur, rec_new, "rec_yds", ["WR", "TE", "RB"], 15.0, s25)["2025"],
           "rmse_receptions": rmse(cur, rec_new, "receptions", ["WR", "TE", "RB"], 1.5, s25)["2025"],
           "lines": RP.score_lines(cur, rec_new, pinfo, games, "rec_yds", "receptions"),
           "floor_rungs": floor_rungs(cur, rec_new, pinfo, games, "rec_yds", "receptions")}
    rec["kept"] = bool(rec["rmse"]["z"] <= -2 and rec["lines"]["z"] <= -2 and rec["floor_rungs"]["z"] <= -2)
    rush = {"rmse": rmse(cur, rush_new, "rush_yds", ["RB"], 20.0, s25)["2025"],
            "rmse_carries": rmse(cur, rush_new, "carries", ["RB"], 4.0, s25)["2025"],
            "lines": RP.score_lines(cur, rush_new, pinfo, games, "rush_yds", "carries"),
            "floor_rungs": floor_rungs(cur, rush_new, pinfo, games, "rush_yds", "carries")}
    rush["kept"] = bool(rush["rmse"]["z"] <= -2 and rush["lines"]["z"] <= -2 and rush["floor_rungs"]["z"] <= -2)
    res["receiving_chain"], res["rushing_chain"] = rec, rush
    for nm, r in (("receiving", rec), ("rushing", rush)):
        print(f"{nm}: rmse z {r['rmse']['z']:.2f}  lines z {r['lines']['z']:.2f} (vs books {r['lines']['new_minus_books']:+.4f})  "
              f"floor z {r['floor_rungs']['z']:.2f}  kept {r['kept']}", flush=True)
    if a.walk_forward or rec["kept"] or rush["kept"]:
        feats = {}
        if rec["kept"] or a.walk_forward:
            feats.update({"receptions": REC_FEATS, "rec_yds": REC_FEATS})
        if rush["kept"] or a.walk_forward:
            feats.update({"carries": RUSH_FEATS, "rush_yds": RUSH_FEATS})
        cur_all = PL.walk_forward(tf)
        new_all = run_chain(tf, feats, PL.C.FIRST_TEST_SEASON, 6)
        per = {"tuning 2017-19": TUNE, "holdout 2021+": [2021, 2022, 2023, 2024, 2025]}
        res["walk_forward"] = {"rec_yds": rmse(cur_all, new_all, "rec_yds", ["WR", "TE", "RB"], 15.0, per),
                               "rush_yds": rmse(cur_all, new_all, "rush_yds", ["RB"], 20.0, per)}
        print("walk-forward", json.dumps({k: {p: round(v["z"], 2) for p, v in d.items()} for k, d in res["walk_forward"].items()}))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
