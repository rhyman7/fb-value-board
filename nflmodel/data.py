"""Download nflverse data and reduce it to small per-game tables.

Two sources, both public and free:
  * games.csv  - schedule, results, rest days, roof, starting QBs and betting lines
  * play-by-play parquet files - one row per play with EPA and success flags

The play-by-play files are large (about 20 MB a season), so each season is
reduced once to two small tables that are cached in data/cache/:
  team_games_<season>.csv  one row per team per game (offense side)
  qb_games_<season>.csv    one row per quarterback per game
  stat_games_<season>.csv  box-score style counts per team per game (last two
                           seasons only), for the matchup pages
Only seasons that are missing, or the season in progress, are downloaded again.
"""
from __future__ import annotations

import datetime as dt
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

PBP_COLS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "pass", "rush", "epa", "success", "wp",
    "id", "name", "two_point_attempt", "play_type",
    # box-score style stats for the matchup pages
    "yards_gained", "sack", "interception", "fumble_lost", "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed", "yardline_100", "fixed_drive", "fixed_drive_result",
]
STATS_SEASONS = 2   # box-score stats are only kept for the current and previous season


def _download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def current_season(today: dt.date | None = None) -> int:
    """NFL seasons start in September; before that, the 'current' season is last year's."""
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


# --------------------------------------------------------------------------
# Schedule, results and lines
# --------------------------------------------------------------------------

def load_games(refresh: bool = False) -> pd.DataFrame:
    path = C.RAW / "games.csv"
    if refresh or not path.exists():
        _download(C.GAMES_URL, path)
    g = pd.read_csv(path)
    g = g[g.season >= C.FIRST_SEASON].copy()
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(C.TEAM_ALIASES)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["played"] = g["result"].notna()
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    g["indoor"] = g["roof"].isin(["dome", "closed"]).astype(int)
    g["playoff"] = (g["game_type"] != "REG").astype(int)
    # A single running index of "game weeks" across seasons, used for ordering.
    g = g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    g["order"] = g["season"] * 100 + g["week"]
    return g


def load_openers(refresh: bool = False) -> pd.DataFrame:
    """Opening spreads by game, in the same sign convention as nflverse
    (positive = home team favoured). Empty if the file cannot be fetched."""
    path = C.RAW / "nfelo_games.csv"
    cols = ["game_id", "open_spread", "open_home_odds", "open_away_odds"]
    if refresh or not path.exists():
        try:
            _download(C.OPENERS_URL, path)
        except Exception as exc:
            print(f"  could not fetch opening lines ({exc})")
            if not path.exists():
                return pd.DataFrame(columns=cols)
    n = pd.read_csv(path, usecols=["game_id", "home_line_open", "home_line_open_price", "away_line_open_price"])
    n = n[n["home_line_open"].notna()]
    return pd.DataFrame({"game_id": n["game_id"], "open_spread": -n["home_line_open"],
                         "open_home_odds": n["home_line_open_price"], "open_away_odds": n["away_line_open_price"]})


# --------------------------------------------------------------------------
# Play-by-play -> per-game aggregates
# --------------------------------------------------------------------------

def _reduce_stats(p: pd.DataFrame, season: int) -> pd.DataFrame:
    """Conventional counting stats per offense per game. Nothing is filtered out:
    these are the numbers a box score would show."""
    p = p[p["posteam"].notna() & (p["two_point_attempt"].fillna(0) == 0)].copy()
    scrim = p["play_type"].isin(["pass", "run"])
    is_pass, is_run = p["play_type"] == "pass", p["play_type"] == "run"
    turnover = (p["interception"].fillna(0) == 1) | (p["fumble_lost"].fillna(0) == 1)
    f = pd.DataFrame({
        "game_id": p["game_id"], "team": p["posteam"], "opp": p["defteam"],
        "plays": scrim.astype(int), "yards": p["yards_gained"].fillna(0) * scrim,
        "pass_plays": is_pass.astype(int), "pass_yards": p["yards_gained"].fillna(0) * is_pass,
        "rush_plays": is_run.astype(int), "rush_yards": p["yards_gained"].fillna(0) * is_run,
        "epa": p["epa"].fillna(0) * scrim, "pass_epa": p["epa"].fillna(0) * is_pass, "rush_epa": p["epa"].fillna(0) * is_run,
        "succ": p["success"].fillna(0) * scrim,
        "sacks": p["sack"].fillna(0), "ints": p["interception"].fillna(0), "fumbles": p["fumble_lost"].fillna(0),
        "to_epa": p["epa"].fillna(0) * turnover,
        "third_conv": p["third_down_converted"].fillna(0),
        "third_att": p["third_down_converted"].fillna(0) + p["third_down_failed"].fillna(0),
        "fourth_conv": p["fourth_down_converted"].fillna(0),
        "fourth_att": p["fourth_down_converted"].fillna(0) + p["fourth_down_failed"].fillna(0),
    })
    out = f.groupby(["game_id", "team", "opp"], sort=False).sum().reset_index()
    # Red zone: drives that ran a play inside the 20, and how many ended in a touchdown.
    rz = p[scrim & (p["yardline_100"] <= 20)].groupby(["game_id", "posteam", "fixed_drive"])["fixed_drive_result"].first().reset_index()
    rz["td"] = (rz["fixed_drive_result"] == "Touchdown").astype(int)
    rz = rz.groupby(["game_id", "posteam"]).agg(rz_trips=("td", "size"), rz_td=("td", "sum")).reset_index().rename(columns={"posteam": "team"})
    out = out.merge(rz, on=["game_id", "team"], how="left").fillna({"rz_trips": 0, "rz_td": 0})
    out.insert(0, "season", season)
    for col in ("team", "opp"):
        out[col] = out[col].replace(C.TEAM_ALIASES)
    return out


def _reduce_season(season: int, with_stats: bool = False):
    raw = C.RAW / f"pbp_{season}.parquet"
    _download(C.PBP_URL.format(season=season), raw)
    p = pd.read_parquet(raw, columns=PBP_COLS)
    raw.unlink()  # keep the workspace small; the cache holds what we need
    stats = _reduce_stats(p, season) if with_stats else None

    p = p[((p["pass"] == 1) | (p["rush"] == 1)) & p["epa"].notna() & p["posteam"].notna()]
    p = p[p["two_point_attempt"].fillna(0) == 0].copy()
    # Drop garbage time: plays where the game is already decided tell us little.
    p = p[p["wp"].between(0.03, 0.97) | p["wp"].isna()]
    # One freak play (a pick-six, a 90-yard run) should not swing a game's rating.
    p["epa"] = p["epa"].clip(-4.5, 4.5)
    p["is_pass"] = (p["pass"] == 1).astype(int)
    p["is_rush"] = 1 - p["is_pass"]
    p["side"] = np.where(p["posteam"] == p["home_team"], "home", "away")

    grp = p.groupby(["game_id", "side"], sort=False)
    tg = grp.agg(
        team=("posteam", "first"), opp=("defteam", "first"),
        plays=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum"),
        n_pass=("is_pass", "sum"), n_rush=("is_rush", "sum"),
    ).reset_index()
    pe = p[p.is_pass == 1].groupby(["game_id", "side"])["epa"].sum().rename("pass_epa")
    re = p[p.is_rush == 1].groupby(["game_id", "side"])["epa"].sum().rename("rush_epa")
    tg = tg.join(pe, on=["game_id", "side"]).join(re, on=["game_id", "side"]).fillna({"pass_epa": 0, "rush_epa": 0})
    tg.insert(0, "season", season)

    d = p[(p.is_pass == 1) & p["id"].notna()]
    qb = d.groupby(["game_id", "side", "id"], sort=False).agg(
        name=("name", "first"), team=("posteam", "first"),
        dropbacks=("epa", "size"), epa=("epa", "sum"),
    ).reset_index().rename(columns={"id": "qb_id"})
    qb.insert(0, "season", season)

    for df in (tg, qb):
        df["team"] = df["team"].replace(C.TEAM_ALIASES)
    tg["opp"] = tg["opp"].replace(C.TEAM_ALIASES)
    return tg, qb, stats


def load_pbp_aggregates(refresh_current: bool = False, verbose: bool = True):
    """Return (team_games, qb_games, stat_games)."""
    C.CACHE.mkdir(parents=True, exist_ok=True)
    now = current_season()
    tgs, qbs, sts = [], [], []
    for season in range(C.FIRST_SEASON, now + 1):
        tpath, qpath = C.CACHE / f"team_games_{season}.csv", C.CACHE / f"qb_games_{season}.csv"
        spath = C.CACHE / f"stat_games_{season}.csv"
        want_stats = season > now - STATS_SEASONS
        have = tpath.exists() and qpath.exists() and (spath.exists() or not want_stats)
        if (season == now and refresh_current) or not have:
            if verbose:
                print(f"  downloading play-by-play {season} ...", flush=True)
            try:
                tg, qb, st = _reduce_season(season, with_stats=want_stats)
            except Exception as exc:  # season not published yet, or network trouble
                if have:
                    print(f"  could not refresh {season} ({exc}); using cached copy")
                else:
                    print(f"  no play-by-play for {season} ({exc}); skipping")
                    continue
            else:
                tg.round(4).to_csv(tpath, index=False)
                qb.round(4).to_csv(qpath, index=False)
                if st is not None:
                    st.round(4).to_csv(spath, index=False)
        tgs.append(pd.read_csv(tpath))
        qbs.append(pd.read_csv(qpath))
        if want_stats and spath.exists():
            sts.append(pd.read_csv(spath))
    stats = pd.concat(sts, ignore_index=True) if sts else pd.DataFrame()
    return pd.concat(tgs, ignore_index=True), pd.concat(qbs, ignore_index=True), stats


def load_all(refresh: bool = False, verbose: bool = True, with_stats: bool = False):
    """Return (games, team_games, qb_games), joined to the schedule ordering.
    With with_stats=True a fourth table of box-score stats is returned too."""
    games = load_games(refresh=refresh)
    tg, qb, stats = load_pbp_aggregates(refresh_current=refresh, verbose=verbose)
    meta = games[["game_id", "order", "week", "home_score", "away_score", "neutral"]]
    tg = tg.merge(meta, on="game_id", how="inner")
    tg["is_home"] = ((tg["side"] == "home") & (tg["neutral"] == 0)).astype(int)
    tg["pts"] = np.where(tg["side"] == "home", tg["home_score"], tg["away_score"])
    qb = qb.merge(meta[["game_id", "order"]], on="game_id", how="inner")
    tg = tg.drop(columns=["home_score", "away_score"])
    if with_stats:
        if len(stats):
            stats = stats.merge(meta[["game_id", "order"]], on="game_id", how="inner")
        return games, tg, qb, stats
    return games, tg, qb
