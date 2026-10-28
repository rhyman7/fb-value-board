"""Role inputs for the player projections: playing time, situations, depth and background.

Cached per season in data/cache/ (small):

  snaps_<season>.csv      player-game offensive snaps and snap share (nflverse snap counts,
                          from Pro Football Reference; ids mapped to gsis through the players file)
  ctx_team_<season>.csv   team-game offensive plays and neutral pass rate (passes on early
                          downs with the win chance between 20% and 80%, first three quarters)
  ctx_player_<season>.csv player-game carries and targets in close games (win chance 20-80%)
  depth_<season>.csv      each skill player's depth-chart rank by team and week, as listed
                          before the game
  bio.csv                 birth date, draft round and pick, first season

Everything a game uses comes from earlier games, except depth charts, which teams publish
during the game week (before kickoff). Not used by the board until tested and adopted
(players.ROLE_FEATS; experiments/players_roles_test.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import _download, current_season

REL = "https://github.com/nflverse/nflverse-data/releases/download"
SKILL = ("QB", "RB", "FB", "WR", "TE")


def _get(url, path):
    if not path.exists():
        _download(url, path)
    return path


def bio(refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / "bio.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"player_id": str})
    raw = _get(f"{REL}/players/players.parquet", C.RAW / "players.parquet")
    p = pd.read_parquet(raw, columns=["gsis_id", "pfr_id", "birth_date", "draft_round", "draft_pick", "rookie_season", "position"])
    p = p[p["gsis_id"].notna()].rename(columns={"gsis_id": "player_id"})
    p = p[p["position"].isin(SKILL)]
    p.to_csv(path, index=False)
    return p


def snaps(season: int, ids: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"snaps_{season}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"player_id": str})
    raw = C.RAW / f"snap_counts_{season}.parquet"
    _download(f"{REL}/snap_counts/snap_counts_{season}.parquet", raw)
    s = pd.read_parquet(raw, columns=["game_id", "pfr_player_id", "position", "team", "offense_snaps", "offense_pct"])
    raw.unlink(missing_ok=True)
    s = s[s["position"].isin(SKILL)]
    m = ids.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["player_id"]
    s["player_id"] = s["pfr_player_id"].map(m)
    s = s[s["player_id"].notna()][["game_id", "player_id", "offense_snaps", "offense_pct"]]
    s = s.groupby(["game_id", "player_id"], as_index=False).max()
    s.to_csv(path, index=False)
    return s


def context(season: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    tp, pp = C.CACHE / f"ctx_team_{season}.csv", C.CACHE / f"ctx_player_{season}.csv"
    if tp.exists() and pp.exists() and not refresh:
        return pd.read_csv(tp), pd.read_csv(pp, dtype={"player_id": str})
    raw = C.RAW / f"pbp_ctx_{season}.parquet"
    _download(C.PBP_URL.format(season=season), raw)
    cols = ["game_id", "posteam", "play_type", "pass_attempt", "rush_attempt", "sack", "qb_kneel", "qb_spike", "qb_scramble",
            "two_point_attempt", "down", "qtr", "wp", "half_seconds_remaining", "rusher_player_id", "receiver_player_id", "yardline_100"]
    p = pd.read_parquet(raw, columns=cols)
    raw.unlink(missing_ok=True)
    p = p[p["posteam"].notna() & (p["two_point_attempt"].fillna(0) == 0) & p["play_type"].isin(["pass", "run"])]
    p["posteam"] = p["posteam"].replace(C.TEAM_ALIASES)
    p = p[(p["qb_kneel"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0)]
    p["dropback"] = (p["play_type"] == "pass").astype(int)
    close = p["wp"].between(0.2, 0.8)
    neutral = close & p["down"].isin([1, 2]) & (p["qtr"] <= 3)
    team = p.groupby(["game_id", "posteam"]).agg(plays=("play_type", "size")).reset_index()
    nt = p[neutral].groupby(["game_id", "posteam"]).agg(n_neutral=("dropback", "size"), pass_neutral=("dropback", "sum")).reset_index()
    cl = p[close].groupby(["game_id", "posteam"]).agg(team_car_close=("rush_attempt", "sum"), team_db_close=("dropback", "sum")).reset_index()
    team = team.merge(nt, on=["game_id", "posteam"], how="left").merge(cl, on=["game_id", "posteam"], how="left").fillna(0)
    team = team.rename(columns={"posteam": "team"})
    ru = p[close & (p["rush_attempt"] == 1) & (p["qb_scramble"].fillna(0) == 0) & p["rusher_player_id"].notna()]
    tg = p[close & (p["dropback"] == 1) & (p["sack"].fillna(0) == 0) & p["receiver_player_id"].notna()]
    a = ru.groupby(["game_id", "posteam", "rusher_player_id"]).size().rename("car_close")
    b = tg.groupby(["game_id", "posteam", "receiver_player_id"]).size().rename("tgt_close")
    a.index.names = b.index.names = ["game_id", "team", "player_id"]
    pl = pd.concat([a, b], axis=1).fillna(0).reset_index()
    pl["team"] = pl["team"].replace(C.TEAM_ALIASES)
    team.to_csv(tp, index=False)
    pl.to_csv(pp, index=False)
    return team, pl


def depth(season: int, games: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Depth-chart rank (1 = starter) for skill players by team and week."""
    path = C.CACHE / f"depth_{season}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"player_id": str})
    raw = C.RAW / f"depth_charts_{season}.parquet"
    try:
        _download(f"{REL}/depth_charts/depth_charts_{season}.parquet", raw)
    except Exception:
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "depth"])
    d = pd.read_parquet(raw)
    raw.unlink(missing_ok=True)
    if "depth_team" in d.columns:          # through 2024: one listing per team and week
        d = d[d["formation"] == "Offense"] if "formation" in d.columns else d
        d = d[d["position"].isin(SKILL)]
        d["depth"] = pd.to_numeric(d["depth_team"], errors="coerce")
        d = d.rename(columns={"club_code": "team", "gsis_id": "player_id"})
        d["week"] = pd.to_numeric(d["week"], errors="coerce")
        out = d.groupby(["week", "team", "player_id"], as_index=False)["depth"].min()
    else:                                  # 2025 on: dated snapshots; each game takes the last one before it
        d = d[d["pos_abb"].isin(["QB", "RB", "FB", "WR", "TE"])].rename(columns={"gsis_id": "player_id", "pos_rank": "depth"})
        d["dt"] = pd.to_datetime(d["dt"], utc=True)
        g = games[games["season"] == season]
        rows = []
        for side in ("home_team", "away_team"):
            x = g[["week", side, "gameday"]].rename(columns={side: "team"})
            rows.append(x)
        sched = pd.concat(rows)
        sched["team"] = sched["team"].replace(C.TEAM_ALIASES)
        sched["cut"] = pd.to_datetime(sched["gameday"]).dt.tz_localize("UTC")
        out = []
        for r in sched.itertuples():
            s = d[(d["team"] == r.team) & (d["dt"] < r.cut)]
            if s.empty:
                continue
            last = s[s["dt"] == s["dt"].max()]
            out.append(last.groupby("player_id", as_index=False)["depth"].min().assign(week=r.week, team=r.team))
        out = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["week", "team", "player_id", "depth"])
    out["team"] = out["team"].replace(C.TEAM_ALIASES)
    out.insert(0, "season", season)
    out = out.dropna(subset=["player_id"])
    out.to_csv(path, index=False)
    return out


def load_all(games: pd.DataFrame, refresh_current: bool = False, verbose: bool = True, first: int = 2012) -> dict:
    now = current_season()
    b = bio(refresh=refresh_current)
    sn, ct, cp, dp = [], [], [], []
    for s in range(first, now + 1):
        r = refresh_current and s == now
        if verbose:
            print(f"  roles {s} ...", flush=True)
        sn.append(snaps(s, b, r))
        t, p = context(s, r)
        ct.append(t); cp.append(p)
        dp.append(depth(s, games, r))
    return {"bio": b, "snaps": pd.concat(sn, ignore_index=True), "ctx_team": pd.concat(ct, ignore_index=True),
            "ctx_player": pd.concat(cp, ignore_index=True), "depth": pd.concat(dp, ignore_index=True)}


# --------------------------------------------------------------------------
# Features for each player-game, from earlier games only (depth: this week's listing)
# --------------------------------------------------------------------------

def features(table: pd.DataFrame, R: dict) -> pd.DataFrame:
    """Role features for every row of the player table (players.build output), keyed by
    game_id and player_id. Decayed averages use the player's earlier games he played."""
    from .players import PLAYER_HL, TEAM_HL, asof, _t

    t = table[["game_id", "player_id", "team", "opp", "season", "week", "t", "active", "played", "group"]].copy()
    t = t.merge(R["snaps"], on=["game_id", "player_id"], how="left")
    t = t.merge(R["ctx_player"].drop(columns="team"), on=["game_id", "player_id"], how="left")
    ct = R["ctx_team"]
    t = t.merge(ct[["game_id", "team", "team_car_close", "team_db_close"]], on=["game_id", "team"], how="left")
    use = t["played"].astype(bool) & (t["active"] == 1)
    t["snap"] = t["offense_pct"].where(use)
    t["car_close_sh"] = (t["car_close"].fillna(0) / t["team_car_close"].where(t["team_car_close"] > 0)).where(use)
    t["tgt_close_sh"] = (t["tgt_close"].fillna(0) / t["team_db_close"].where(t["team_db_close"] > 0)).where(use)
    t["one_r"] = np.where(use & t["snap"].notna(), 1.0, np.nan)
    f = asof(t, "player_id", ["snap", "car_close_sh", "tgt_close_sh"], ["one_r"], PLAYER_HL)
    n = f["one_r_sum"].replace(0, np.nan)
    out = f[["game_id", "player_id"]].copy()
    out["snap_avg"] = f["snap_sum"] / n
    out["car_close_avg"] = f["car_close_sh_sum"] / n
    out["tgt_close_avg"] = f["tgt_close_sh_sum"] / n
    # last game he played, and the one before: a role change shows here first
    f = f.sort_values(["player_id", "t"])
    played_snap = f["snap"]
    out = out.set_index(f.index)
    out["snap_last"] = played_snap.groupby(f["player_id"]).transform(lambda s: s.ffill().shift())
    out["snap_last2"] = played_snap.groupby(f["player_id"]).transform(lambda s: s.rolling(2, min_periods=1).mean().ffill().shift())
    out["snap_trend"] = out["snap_last2"] - out["snap_avg"]
    # a new team: his team differs from the last team he played for
    lastteam = f["team"].where(use.reindex(f.index)).groupby(f["player_id"]).transform(lambda s: s.ffill().shift())
    out["new_team"] = ((lastteam.notna()) & (lastteam != f["team"])).astype(float)
    out = out.reset_index(drop=True)
    # depth chart, this week
    d = R["depth"]
    out = out.merge(t[["game_id", "player_id", "season", "week", "team", "t"]], on=["game_id", "player_id"], how="left")
    out = out.merge(d, on=["season", "week", "team", "player_id"], how="left")
    out["depth"] = out["depth"].clip(upper=5)
    # background
    b = R["bio"].drop_duplicates("player_id").set_index("player_id")
    born = pd.to_datetime(out["player_id"].map(b["birth_date"]), errors="coerce")
    out["age"] = (pd.to_datetime(out["season"].astype(str) + "-10-01") - born).dt.days / 365.25   # age at mid-season
    out["draft_pick"] = out["player_id"].map(b["draft_pick"]).fillna(260).clip(upper=260)
    out["exp"] = (out["season"] - out["player_id"].map(b["rookie_season"])).clip(lower=0)
    # team pace and neutral pass rate, from the team's earlier games
    tt = table.drop_duplicates(["game_id", "team"])[["game_id", "team", "opp", "season", "t", "played"]].merge(ct[["game_id", "team", "plays", "n_neutral", "pass_neutral"]], on=["game_id", "team"], how="left")
    tt["plays_v"] = tt["plays"].where(tt["played"].astype(bool))
    tt["pn"] = tt["pass_neutral"].where(tt["played"].astype(bool))
    tt["nn"] = tt["n_neutral"].where(tt["played"].astype(bool))
    tt["one_t"] = np.where(tt["played"].astype(bool) & tt["plays"].notna(), 1.0, np.nan)
    a = asof(tt, "team", ["plays_v", "pn", "nn"], ["one_t"], TEAM_HL)
    a["pace"] = a["plays_v_sum"] / a["one_t_sum"].replace(0, np.nan)
    a["neutral_pass"] = (a["pn_sum"] + 20 * 0.55) / (a["nn_sum"] + 20)
    team_f = a[["game_id", "team", "pace", "neutral_pass"]]
    out = out.merge(team_f, on=["game_id", "team"], how="left")
    keep = ["game_id", "player_id", "snap_avg", "snap_last", "snap_last2", "snap_trend", "car_close_avg", "tgt_close_avg",
            "new_team", "depth", "age", "draft_pick", "exp", "pace", "neutral_pass"]
    return out[[c for c in keep if c in out.columns]]
