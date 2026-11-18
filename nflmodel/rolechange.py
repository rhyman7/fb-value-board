"""How much a player's role just moved, for the prop distributions' spread (Nov 17, 2026).

u = |the player's share in his last appearance - his decayed share coming in|, capped at 0.30:
  role_chg_car   share of the team's carries (carries and rushing-yards props)
  role_chg_rec   share of the team's targets (NFL) or catches (college: no targets in the box
                 scores), the share behind his receiving role (receptions and receiving yards)
The decayed share is an exponential average over his appearances (half-life four appearances)
including the last one. Known before kickoff: only earlier games count. 0 with fewer than two
earlier appearances, or when his last appearance is older than last season.

Read by props.role(): where a fit carries a role-change term (CountFit.d, YardsFit.e) the spread
widens after a change. Applied Nov 17, 2026, provisional (FEATURES.md "Prop spread that widens after
a role change", experiments/role_change_spread_test.py): NFL WR/TE/RB receptions and receiving yards
(data/prop_params.json) and college RB carries and rushing yards (data/cfb/prop_params.json). Every
other fit has the term at 0 and prices exactly as before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAP = 0.30
HALF_LIFE = 4.0
COLS = ("role_chg_car", "role_chg_rec")
# the fits that carry the term (the pieces that passed, Ryan's call Nov 17, 2026); a refit of either
# parameter file keeps them (props_calibration.py --production / --role-change, cfb_propdist_test.py --save)
APPLIED = {"nfl": [("receptions", "WR"), ("receptions", "TE"), ("receptions", "RB"),
                   ("rec_yds", "WR"), ("rec_yds", "TE"), ("rec_yds", "RB")],
           "cfb": [("carries", "RB"), ("rush_yds", "RB")]}
CATCH = {"nfl": "targets", "cfb": "receptions"}


def add(t: pd.DataFrame, catch_col: str) -> pd.DataFrame:
    """t with role_chg_car and role_chg_rec on every row. catch_col: "targets" (NFL) or
    "receptions" (college). Needs player_id, game_id, team, season, t, played, carries, catch_col."""
    t = t.copy()
    need = {"player_id", "game_id", "team", "season", "t", "played", "carries", catch_col}
    if not need <= set(t.columns) or t.empty:
        t["role_chg_car"], t["role_chg_rec"] = 0.0, 0.0
        return t
    t["_i"] = np.arange(len(t))
    pl = t["played"].astype(bool)
    touch = t.reindex(columns=["carries", "receptions", "targets", "pass_att"]).fillna(0).sum(axis=1) > 0
    h = t[pl & touch][["_i", "player_id", "game_id", "team", "season", "t", "carries", catch_col]].copy()
    tot = t[pl].groupby(["game_id", "team"])[["carries", catch_col]].sum()
    h = h.join(tot, on=["game_id", "team"], rsuffix="_team")
    h["s_car"] = (h["carries"] / h["carries_team"].where(h["carries_team"] > 0)).fillna(0.0)
    h["s_rec"] = (h[catch_col] / h[catch_col + "_team"].where(h[catch_col + "_team"] > 0)).fillna(0.0)
    h = h.sort_values(["player_id", "t"]).reset_index(drop=True)
    h["n_app"] = h.groupby("player_id").cumcount() + 1
    for c in ("car", "rec"):
        dec = h.groupby("player_id")["s_" + c].ewm(halflife=HALF_LIFE).mean().reset_index(level=0, drop=True).sort_index()
        h["after_" + c] = (h["s_" + c] - dec).abs().clip(upper=CAP)
    q = t[["_i", "player_id", "t", "season"]].sort_values("t")
    hh = h[["player_id", "t", "season", "n_app", "after_car", "after_rec"]].rename(columns={"t": "h_t", "season": "h_season"}).sort_values("h_t")
    m = pd.merge_asof(q, hh, left_on="t", right_on="h_t", by="player_id", allow_exact_matches=False, direction="backward")
    m = m.set_index("_i").reindex(t["_i"])
    ok = (m["n_app"].fillna(0).to_numpy() >= 2) & (m["h_season"].to_numpy() >= t["season"].to_numpy() - 1)
    t["role_chg_car"] = np.where(ok, m["after_car"].fillna(0).to_numpy(), 0.0)
    t["role_chg_rec"] = np.where(ok, m["after_rec"].fillna(0).to_numpy(), 0.0)
    return t.drop(columns="_i")
