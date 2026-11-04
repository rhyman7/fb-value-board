"""PFF role inputs for the college player projections: route share and snap share.

Kept Nov 4, 2026 (FEATURES.md, "College PFF role inputs"; experiments/cfb_pff_role_test.py): in the
receiving chain (receptions, receiving yards) they make the projections more accurate on every
measure (walk-forward receiving-yards RMSE 3.7 SE better on 2017-19, 10.1 SE on 2021+; better at
the 2025 lines and on the floor rungs). Not used for rushing (worse on the walk-forward).

As of each game, from the player's earlier games only (8-game half-life, last season at 60%):
  route_share = routes / team pass plays;  snap_share = offensive snaps / team offensive snaps
(team = the most any player of the team logged that week). PFF rows (all divisions) are matched to
the board's FBS games: franchises to teams by shared players per season, players by name within a
team-season, weeks by team and week.

The data is private (rhyman7/pff-data, read by cfbmodel.pff_data). These columns feed the
calibration only; never write them, or anything row by row from PFF, to docs/ or data/. Without the
data both columns are 0, which the calibration fits to no weight: the board is then what it was
before Nov 4, 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pff_data

HL, CARRY = 8.0, 0.6
COLS = ("route_share", "snap_share")


def _norm(s):
    from nflmodel.propmarket import norm_name
    return norm_name(s)


def player_weeks(seasons) -> pd.DataFrame:
    rec = pff_data.load("receiving_summary", seasons)
    off = pff_data.load("offense_summary", seasons)
    if rec.empty or off.empty:
        return pd.DataFrame()
    rec = rec[["season", "week", "player", "player_id", "franchise_id", "routes", "yards"]]
    off = off[["season", "week", "player", "player_id", "franchise_id", "snap_counts_total", "snap_counts_total_pass"]]
    team = off.groupby(["season", "week", "franchise_id"]).agg(team_snaps=("snap_counts_total", "max"),
                                                              team_pass=("snap_counts_total_pass", "max")).reset_index()
    pw = off.merge(rec[["season", "week", "player_id", "franchise_id", "routes", "yards"]],
                   on=["season", "week", "player_id", "franchise_id"], how="outer")
    pw["player"] = pw["player"].fillna(pw["player_id"].map(rec.drop_duplicates("player_id").set_index("player_id")["player"]))
    pw = pw.merge(team, on=["season", "week", "franchise_id"], how="left")
    pw[["routes", "snap_counts_total"]] = pw[["routes", "snap_counts_total"]].fillna(0)
    pw["nname"] = pw["player"].map(_norm)
    return pw


def on_board(pw: pd.DataFrame, t: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """PFF player-weeks on the board's FBS games, with the board's player id (board_id)."""
    b = t[t["played"].astype(bool) & (t["appeared"] == 1)][["game_id", "team", "season", "week", "player_id"]].copy()
    b["nname"] = b["player_id"].map(info["name"]).map(_norm)
    fbs = t[t["proj_total"].notna()][["season", "team"]].drop_duplicates()
    bn = b[["season", "team", "nname"]].drop_duplicates()
    pn = pw[["season", "franchise_id", "nname"]].drop_duplicates()
    ov = pn.merge(bn, on=["season", "nname"]).groupby(["season", "franchise_id", "team"]).size().rename("n").reset_index()
    best = ov.sort_values("n", ascending=False).drop_duplicates(["season", "franchise_id"]).drop_duplicates(["season", "team"])
    best = best[best["n"] >= 5].merge(fbs, on=["season", "team"])
    m = pw.merge(best[["season", "franchise_id", "team"]], on=["season", "franchise_id"])
    games = b[["game_id", "team", "season", "week"]].drop_duplicates(["team", "season", "week"])
    m = m.merge(games, on=["team", "season", "week"])
    bp = b[["season", "team", "nname", "player_id"]].drop_duplicates()
    bp = bp[~bp.duplicated(["season", "team", "nname"], keep=False)]
    m = m[~m.duplicated(["game_id", "team", "nname"], keep=False)]
    return m.merge(bp.rename(columns={"player_id": "board_id"}), on=["season", "team", "nname"], how="inner")


def asof(t: pd.DataFrame, m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    h = m.copy()
    h["board_id"] = h["board_id"].astype(np.int64)
    h = h.merge(t[["game_id", "t"]].drop_duplicates("game_id"), on="game_id").sort_values(["board_id", "t"])
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
    return np.clip(rs, 0, 1.2), np.clip(ss, 0, 1.2)


def add(t: pd.DataFrame, info: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """t with route_share and snap_share (0 where there is no PFF history, or no PFF data at all)."""
    t = t.copy()
    for c in COLS:
        t[c] = 0.0
    if not pff_data.available():
        if verbose:
            print("  PFF data not found: route/snap share left at 0 (projections as before Nov 4, 2026)")
        return t
    seasons = sorted(int(s) for s in t["season"].unique() if s >= 2017)
    pw = player_weeks(seasons)
    if pw.empty:
        if verbose:
            print("  PFF data empty: route/snap share left at 0")
        return t
    m = on_board(pw, t, info)
    # Stale data misleads (the calibration learned from current roles): if PFF is missing more
    # than the latest regular-season week the board has played (weeks 0-16; the board numbers
    # bowls and playoff rounds past PFF's 17-18), use none of it.
    done = t[t["played"].astype(bool) & t["proj_total"].notna() & (t["week"] <= 16)]
    if len(done):
        s_last = int(done["season"].max())
        played_weeks = set(done.loc[done["season"] == s_last, "week"].astype(int))
        have = m.loc[m["season"] == s_last, "week"]
        w_pff = int(have.max()) if len(have) else -1
        behind = sorted(w for w in played_weeks if w > w_pff)
        if len(behind) > 1:
            if verbose:
                print(f"  PFF data stale ({s_last}: has through week {w_pff}, board has played weeks {behind}): "
                      "route/snap share left at 0")
            return t
    t["route_share"], t["snap_share"] = asof(t, m)
    if verbose:
        cur = t[t["season"] == t["season"].max()]
        print(f"  PFF roles: {len(m):,} player-weeks matched; {(cur['route_share'] > 0).mean():.0%} of this season's rows have a route share")
    return t
