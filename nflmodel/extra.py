"""Extra per-game stats for candidate features, cached like the main aggregates.

team_extra_<season>.csv, one row per team per game (offense view):
  snaps            pass + rush plays, garbage time included (pace)
  sec_per_play     seconds between snaps in neutral situations (pace)
  dropbacks, sacks pass-protection / pass-rush
  expl, plays_ng   explosive plays (pass >= 20 yds, rush >= 10) over non-garbage plays
  ed_pass_epa, ed_pass   early-down (1st/2nd) pass EPA and count, non-garbage
  fumbles, fumbles_lost  all offensive fumbles and those lost (recovery luck)
  st_epa           net special-teams EPA (kicks, punts, returns) for the team
qb_extra_<season>.csv, one row per passer per game:
  att_cpoe, cpoe_sum     attempts with a CPOE value and their summed CPOE
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import _download, current_season

COLS = ["game_id", "season", "posteam", "defteam", "home_team", "pass", "rush", "epa", "wp", "down",
        "yards_gained", "sack", "qb_dropback", "fumble", "fumble_lost", "play_type", "two_point_attempt",
        "qb_kneel", "qb_spike", "game_seconds_remaining", "score_differential", "qtr", "fixed_drive",
        "passer_player_id", "cpoe", "complete_pass", "incomplete_pass"]
ST = {"field_goal", "extra_point", "punt", "kickoff"}


def _reduce(season: int):
    raw = C.RAW / f"pbp_{season}.parquet"
    _download(C.PBP_URL.format(season=season), raw)
    p = pd.read_parquet(raw, columns=[c for c in COLS])
    raw.unlink()
    p["posteam"] = p["posteam"].replace(C.TEAM_ALIASES)
    p["defteam"] = p["defteam"].replace(C.TEAM_ALIASES)

    # ---- special teams: net EPA for each team in each game
    st = p[p["play_type"].isin(ST) & p["epa"].notna() & p["posteam"].notna()]
    st_long = pd.concat([
        st[["game_id", "posteam", "epa"]].rename(columns={"posteam": "team"}),
        st[["game_id", "defteam", "epa"]].rename(columns={"defteam": "team"}).assign(epa=lambda d: -d["epa"]),
    ])
    st_net = st_long.groupby(["game_id", "team"])["epa"].sum().rename("st_epa")

    s = p[((p["pass"] == 1) | (p["rush"] == 1)) & p["posteam"].notna() & (p["two_point_attempt"].fillna(0) == 0)
          & (p["qb_kneel"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0)].copy()
    s["ng"] = s["wp"].between(0.03, 0.97) | s["wp"].isna()
    s["is_pass"] = (s["pass"] == 1)
    s["expl"] = (s["is_pass"] & (s["yards_gained"] >= 20)) | (~s["is_pass"] & (s["yards_gained"] >= 10))
    s["ed"] = s["down"].isin([1, 2]) & s["is_pass"] & s["ng"]
    # pace: seconds to the next snap by the same offense on the same drive, neutral game state
    s = s.sort_values(["game_id", "fixed_drive", "game_seconds_remaining"], ascending=[True, True, False])
    nxt = s.groupby(["game_id", "fixed_drive"])["game_seconds_remaining"].shift(-1)
    s["gap"] = s["game_seconds_remaining"] - nxt
    neutral = (s["score_differential"].abs() <= 7) & (s["qtr"] <= 3) & s["gap"].between(5, 60)
    s["gap_n"] = np.where(neutral, s["gap"], np.nan)

    g = s.groupby(["game_id", "posteam"])
    out = pd.DataFrame({
        "snaps": g.size(),
        "sec_per_play": g["gap_n"].mean(),
        "dropbacks": g["qb_dropback"].sum(),
        "sacks": g["sack"].sum(),
        "expl": s[s.ng].groupby(["game_id", "posteam"])["expl"].sum(),
        "plays_ng": s[s.ng].groupby(["game_id", "posteam"]).size(),
        "ed_pass_epa": s[s.ed].groupby(["game_id", "posteam"])["epa"].sum(),
        "ed_pass": s[s.ed].groupby(["game_id", "posteam"]).size(),
        "fumbles": g["fumble"].sum(),
        "fumbles_lost": g["fumble_lost"].sum(),
    }).reset_index().rename(columns={"posteam": "team"})
    out = out.merge(st_net.reset_index(), on=["game_id", "team"], how="left")
    out = out.fillna({"st_epa": 0, "expl": 0, "plays_ng": 0, "ed_pass_epa": 0, "ed_pass": 0})
    out.insert(0, "season", season)

    q = p[p["cpoe"].notna() & p["passer_player_id"].notna()]
    qb = q.groupby(["game_id", "passer_player_id"]).agg(att_cpoe=("cpoe", "size"), cpoe_sum=("cpoe", "sum")).reset_index()
    qb = qb.rename(columns={"passer_player_id": "qb_id"})
    qb.insert(0, "season", season)
    return out, qb


def load_extra(refresh_current: bool = False, verbose: bool = True):
    now = current_season()
    ts, qs = [], []
    for season in range(C.FIRST_SEASON, now + 1):
        tp, qp = C.CACHE / f"team_extra_{season}.csv", C.CACHE / f"qb_extra_{season}.csv"
        if (season == now and refresh_current) or not (tp.exists() and qp.exists()):
            if verbose:
                print(f"  extra stats {season} ...", flush=True)
            t, q = _reduce(season)
            t.round(4).to_csv(tp, index=False)
            q.round(4).to_csv(qp, index=False)
        ts.append(pd.read_csv(tp))
        qs.append(pd.read_csv(qp))
    return pd.concat(ts, ignore_index=True), pd.concat(qs, ignore_index=True)
