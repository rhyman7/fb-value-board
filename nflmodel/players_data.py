"""Per-player and per-team box scores from play-by-play, cached per season.

player_games_<season>.csv   one row per player per game with any passing,
                            rushing or receiving involvement (regular season
                            and playoffs; two-point tries excluded). carries and
                            rush_yds leave kneel-downs out, as the projections always
                            have; kneels and kneel_yds carry them separately, because
                            the books settle rushing props on the official line, which
                            counts a kneel as a rushing attempt (SETTLE below)
player_info_<season>.csv    position and name for every player id seen on a
                            roster that season (latest listing wins)

Every game situation counts, garbage time included: these are box-score
projections, not efficiency ratings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import _download, current_season

COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "pass_attempt", "rush_attempt",
        "sack", "qb_dropback", "qb_kneel", "qb_spike", "two_point_attempt", "complete_pass", "interception",
        "passing_yards", "receiving_yards", "rushing_yards", "pass_touchdown", "rush_touchdown", "td_player_id",
        "passer_player_id", "receiver_player_id", "rusher_player_id", "air_yards", "yardline_100", "fumble_lost"]
# What a sportsbook settles a rushing prop on: the official box score, kneel-downs included
# (2025: 0.79 kneels a team-game, about -1 yard each). Grading adds these columns to the stat;
# the projections and every model input stay on carries / rush_yds without kneels (Nov 9, 2026).
SETTLE = {"carries": "kneels", "rush_yds": "kneel_yds"}


def kneels(p: pd.DataFrame, key: list) -> pd.DataFrame:
    """Kneel-downs per rusher per game from play-by-play rows: kneels (count) and kneel_yds."""
    k = p[(p["qb_kneel"].fillna(0) == 1) & (p["rush_attempt"] == 1) & p["rusher_player_id"].notna()]
    return k.groupby(key + ["rusher_player_id"]).agg(kneels=("rush_attempt", "size"), kneel_yds=("rushing_yards", "sum")) \
            .reset_index().rename(columns={"rusher_player_id": "player_id"})


ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet"
FIRST = 2012


def _reduce(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = C.RAW / f"pbp_pl_{season}.parquet"
    _download(C.PBP_URL.format(season=season), raw)
    p = pd.read_parquet(raw, columns=COLS)
    raw.unlink()
    p = p[(p["two_point_attempt"].fillna(0) == 0) & p["posteam"].notna()].copy()
    for c in ("posteam", "defteam"):
        p[c] = p[c].replace(C.TEAM_ALIASES)
    p["rz"] = (p["yardline_100"] <= 20).astype(int)
    p["gl"] = (p["yardline_100"] <= 5).astype(int)
    key = ["game_id", "season", "week", "season_type", "posteam", "defteam"]

    att = p[(p["pass_attempt"] == 1) & (p["sack"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0)]
    passing = att.groupby(key + ["passer_player_id"]).agg(
        pass_att=("pass_attempt", "size"), completions=("complete_pass", "sum"), pass_yds=("passing_yards", "sum"),
        pass_td=("pass_touchdown", "sum"), ints=("interception", "sum")).reset_index().rename(columns={"passer_player_id": "player_id"})
    sk = p[p["sack"].fillna(0) == 1].groupby(key + ["passer_player_id"]).size().rename("sacks").reset_index().rename(columns={"passer_player_id": "player_id"})

    tg = att[att["receiver_player_id"].notna()].copy()
    tg["rec_td"] = ((tg["pass_touchdown"] == 1) & (tg["td_player_id"] == tg["receiver_player_id"])).astype(int)
    receiving = tg.groupby(key + ["receiver_player_id"]).agg(
        targets=("pass_attempt", "size"), receptions=("complete_pass", "sum"), rec_yds=("receiving_yards", "sum"),
        rec_td=("rec_td", "sum"), air_yards=("air_yards", "sum"), rz_targets=("rz", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"})

    ru = p[(p["rush_attempt"] == 1) & (p["qb_kneel"].fillna(0) == 0) & p["rusher_player_id"].notna()].copy()
    ru["rush_td1"] = ((ru["rush_touchdown"] == 1) & (ru["td_player_id"] == ru["rusher_player_id"])).astype(int)
    rushing = ru.groupby(key + ["rusher_player_id"]).agg(
        carries=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum"), rush_td=("rush_td1", "sum"),
        rz_carries=("rz", "sum"), gl_carries=("gl", "sum")).reset_index().rename(columns={"rusher_player_id": "player_id"})

    pg = passing
    for df in (sk, receiving, rushing):
        pg = pg.merge(df, on=key + ["player_id"], how="outer")
    num = [c for c in pg.columns if c not in key + ["player_id"]]
    pg[num] = pg[num].fillna(0)
    # kneel-downs, for settling props only: a left merge, so a player whose only plays were kneels
    # gets no row (he has no row today, and a new one would count him as having played)
    pg = pg.merge(kneels(p, key), on=key + ["player_id"], how="left")
    pg[["kneels", "kneel_yds"]] = pg[["kneels", "kneel_yds"]].fillna(0)

    # names and positions from the season's weekly rosters
    rp = C.RAW / f"roster_weekly_{season}.parquet"
    if not rp.exists():
        _download(ROSTER_URL.format(season=season), rp)
    r = pd.read_parquet(rp, columns=["gsis_id", "full_name", "position", "week"])
    rp.unlink(missing_ok=True)
    info = r[r["gsis_id"].notna()].sort_values("week").groupby("gsis_id").tail(1)[["gsis_id", "full_name", "position"]]
    info = info.rename(columns={"gsis_id": "player_id", "full_name": "name"})
    info.insert(0, "season", season)
    return pg, info


def load_player_games(refresh_current: bool = False, verbose: bool = True):
    now = current_season()
    pgs, infos = [], []
    for season in range(FIRST, now + 1):
        pp, ip = C.CACHE / f"player_games_{season}.csv", C.CACHE / f"player_info_{season}.csv"
        if (season == now and refresh_current) or not (pp.exists() and ip.exists()):
            if verbose:
                print(f"  player box scores {season} ...", flush=True)
            pg, info = _reduce(season)
            pg.round(2).to_csv(pp, index=False)
            info.to_csv(ip, index=False)
        one = pd.read_csv(pp)
        for c in SETTLE.values():          # a cache written before Nov 9, 2026
            if c not in one:
                one[c] = 0.0
        pgs.append(one)
        infos.append(pd.read_csv(ip))
    pg = pd.concat(pgs, ignore_index=True)
    info = pd.concat(infos, ignore_index=True).sort_values("season").groupby("player_id").tail(1)
    return pg, info.set_index("player_id")
