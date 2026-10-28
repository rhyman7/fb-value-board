"""Two player-role signals for the game model's margin (Oct 2026, FEATURES.md):

  d_depth1_out   depth-chart starters at QB, RB, WR and TE who are out, home minus away
  d_snap_drop    how far the team's regular skill players' snap share fell in their last two
                 games against their average, summed per team, home minus away

"Out" uses only what is known by Friday, so nothing comes from after the board is built or
the opener is posted: roster status the week before (players.pregame_active; injured
reserve and the like are out) plus players listed Out or Doubtful on the week's injury
report. Game-day inactives are not used. Tested in experiments/game_roles_test.py and
experiments/game_roles_open_test.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import players as PL, roles

FEATS = ["d_depth1_out", "d_snap_drop"]


def friday_active(table: pd.DataFrame, R: dict, avail: pd.DataFrame) -> pd.Series:
    """1 if a player is expected to play as of Friday, else 0."""
    rows = table[["season", "week", "player_id"]].copy()
    rows["t"] = PL._t(rows["season"], rows["week"])
    a = avail[["season", "week", "player_id", "status"]].copy()
    a["t"] = PL._t(a["season"], a["week"])
    a = a.drop_duplicates(["player_id", "t"], keep="last")[["player_id", "t", "status"]]
    rows = rows.merge(a, on=["player_id", "t"], how="left")
    rows.index = table.index
    pre = PL.pregame_active(rows, avail)
    pr = R["practice"][["season", "week", "team", "player_id", "report"]].drop_duplicates(["season", "week", "team", "player_id"])
    rep = table[["season", "week", "team", "player_id"]].merge(pr, on=["season", "week", "team", "player_id"], how="left")["report"].to_numpy()
    out = (pre.to_numpy() == 1) & ~(np.nan_to_num(rep.astype(float), nan=-1) >= 2)
    if "played" in table and "active" in table:     # upcoming games: who the board already has out
        out &= ~((~table["played"].astype(bool)) & (table["active"] == 0)).to_numpy()
    return pd.Series(out.astype(int), index=table.index)


def team_features(table: pd.DataFrame, R: dict, avail: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: depth1_out and snap_drop."""
    f = roles.features(table, R)[["game_id", "player_id", "depth", "snap_avg", "snap_trend"]]
    x = table[["game_id", "team", "player_id"]].assign(active=friday_active(table, R, avail).to_numpy())
    x = x.merge(f, on=["game_id", "player_id"], how="left")
    out = x["active"] == 0
    regular = (x["snap_avg"].fillna(0) >= 0.4) & ~out
    key = [x["game_id"], x["team"]]
    return pd.DataFrame({
        "depth1_out": ((x["depth"] == 1) & out).groupby(key).sum(),
        "snap_drop": (-(x["snap_trend"].fillna(0)).clip(upper=0)).where(regular, 0).groupby(key).sum(),
    }).reset_index()


def attach(frame: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Add d_depth1_out and d_snap_drop (home minus away; 0 where unknown) to the game frame."""
    f = frame.drop(columns=[c for c in FEATS if c in frame])
    for side, team in (("h", "home_team"), ("a", "away_team")):
        s = t.rename(columns={"team": team, "depth1_out": side + "_d1", "snap_drop": side + "_sd"})
        f = f.merge(s, on=["game_id", team], how="left")
    f["d_depth1_out"] = f["h_d1"].fillna(0) - f["a_d1"].fillna(0)
    f["d_snap_drop"] = f["h_sd"].fillna(0) - f["a_sd"].fillna(0)
    return f.drop(columns=["h_d1", "a_d1", "h_sd", "a_sd"])
