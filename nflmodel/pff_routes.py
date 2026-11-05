"""PFF route inputs for the NFL New projections (players_ml.py): route share and targets per route.

Applied Nov 5, 2026 (Ryan's call on a trade-off; PLAYERS.md "PFF route share in the New
projections", experiments/nfl_pff_routes_test.py): better on 11 of 12 measures, every Brier
measure against the 2023-2025 DraftKings/FanDuel lines included; the one miss, RB receiving
yards on the 2019-2025 holdout, is +0.0007 yards (0.04 SE).

As of each game, from the games he played before it (players.asof, the board's player decay):
  route_avg    decayed routes / decayed team pass plays (team pass plays: the most `pass_plays`
               any of the team's players logged that week in PFF's receiving_summary)
  route_last   route share in the last game he played with PFF data
  route_trend  route_last - route_avg
  tprr_avg     decayed targets / decayed routes (missing below 20 decayed routes)
PFF players are matched to the board's gsis ids by nflverse's players file (`pff_id`), and where
that file has no pff_id yet (new rookies, early in a season) by name among the board's players in
the same season and week; weeks by season and week, regular season only (PFF's NFL playoff weeks
are 28-32). A player who played but has no PFF row that week (his team has PFF data) ran 0
routes, exactly as in the test. That includes a game PFF hasn't charted for him yet (on Nov 5,
2026 week 8 lacked a receiver nflverse had at 78% of the snaps); the daily pull re-reads the
season, so a game PFF fills in later is picked up. Two guards were tried and dropped, each making
the backs' holdout worse (experiments/apply_check_nfl_pff_routes.py): leaving such games out when
nflverse had him at 20%+ of the snaps, and counting 0 only for players PFF had matched elsewhere.
Matching rookies by name is kept: without it a rookie the id file hasn't linked yet would be
counted at 0 routes every game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

COLS = ["route_avg", "route_last", "route_trend", "tprr_avg"]
FIRST = 2014            # PFF's NFL tables pulled from 2014
MIN_ROUTES = 20


def _players() -> pd.DataFrame:
    """nflverse's players file (gsis id, pff id, name), re-downloaded once a day."""
    from .roles import REL
    from .data import _download
    path = C.RAW / "players.parquet"
    if not path.exists() or (pd.Timestamp.now().timestamp() - path.stat().st_mtime) > 86400:
        _download(f"{REL}/players/players.parquet", path)
    return pd.read_parquet(path, columns=["gsis_id", "pff_id", "display_name"])


def pff_ids() -> pd.DataFrame:
    """nflverse gsis id -> PFF player id."""
    p = _players()[["gsis_id", "pff_id"]].dropna()
    p["pff_id"] = pd.to_numeric(p["pff_id"], errors="coerce")
    return p.dropna().drop_duplicates("pff_id").rename(columns={"gsis_id": "nfl_id"})


def _by_name(rec: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """nfl_id for PFF rows the id file doesn't link: a unique name match among the board's players
    of that season and week."""
    from .propmarket import norm_name
    miss = rec["nfl_id"].isna()
    if not miss.any():
        return rec
    names = _players().dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]
    linked = set(rec["nfl_id"].dropna())
    b = table[["player_id", "season", "week"]].drop_duplicates()
    b = b[~b["player_id"].isin(linked)].copy()
    b["nname"] = b["player_id"].map(names).map(norm_name)
    b = b.dropna(subset=["nname"])
    b = b[~b.duplicated(["season", "week", "nname"], keep=False)]
    u = rec.loc[miss, ["season", "week", "player"]].copy()
    u["nname"] = u["player"].map(norm_name)
    u = u[~u.duplicated(["season", "week", "nname"], keep=False)]
    m = u.reset_index().merge(b, on=["season", "week", "nname"], how="inner").set_index("index")["player_id"]
    rec = rec.copy()
    rec.loc[m.index, "nfl_id"] = m
    return rec


def player_weeks(last_season: int) -> pd.DataFrame:
    """PFF receiving_summary rows (regular season) with the board's id; empty without the data."""
    from cfbmodel import pff_data
    if not pff_data.available():
        return pd.DataFrame()
    rec = pff_data.load("receiving_summary", range(FIRST, last_season + 1), league="nfl")
    if rec.empty:
        return rec
    rec = rec[rec["week"].between(1, 18)][["season", "week", "player_id", "player", "team_name", "routes", "pass_plays", "targets"]].copy()
    team = rec.groupby(["season", "week", "team_name"])["pass_plays"].max().rename("team_pass").reset_index()
    rec = rec.merge(team, on=["season", "week", "team_name"], how="left")
    return rec.merge(pff_ids(), left_on="player_id", right_on="pff_id", how="left")


def features(table: pd.DataFrame, rec: pd.DataFrame | None = None) -> pd.DataFrame:
    """COLS for every row of the player table, keyed by game_id and player_id (empty frame
    without the PFF data)."""
    from .players import PLAYER_HL, asof
    if rec is None:
        try:
            rec = player_weeks(int(table["season"].max()))
        except Exception as exc:              # never a failed board over the private data
            print(f"  PFF routes skipped ({exc})")
            return pd.DataFrame()
    if rec is None or rec.empty:
        return pd.DataFrame()
    t = table[["game_id", "player_id", "team", "season", "week", "t", "active", "played"]].copy()
    if "player" in rec.columns:
        rec = _by_name(rec, t)
    r = rec.dropna(subset=["nfl_id"]).drop_duplicates(["nfl_id", "season", "week"])
    t = t.merge(r[["nfl_id", "season", "week", "routes", "team_pass", "targets"]].rename(columns={"nfl_id": "player_id", "targets": "pff_tgt"}),
                on=["player_id", "season", "week"], how="left")
    played = t["played"].fillna(False).astype(bool) & (t["active"] == 1)
    t["team_pass"] = t.groupby(["game_id", "team"])["team_pass"].transform("max")
    fill = played & t["routes"].isna() & t["team_pass"].gt(0)
    t.loc[fill, ["routes", "pff_tgt"]] = 0.0
    use = played & t["routes"].notna() & t["team_pass"].gt(0)
    t["rt"] = t["routes"].where(use)
    t["tp"] = t["team_pass"].where(use)
    t["tg"] = t["pff_tgt"].where(use)
    f = asof(t, "player_id", ["rt", "tp", "tg"], None, PLAYER_HL)       # sorted by player, t; index 0..n-1
    out = f[["game_id", "player_id"]].copy()
    out["route_avg"] = (f["rt_sum"] / f["tp_sum"].replace(0, np.nan)).clip(0, 1)
    out["tprr_avg"] = (f["tg_sum"] / f["rt_sum"].where(f["rt_sum"] >= MIN_ROUTES)).clip(0, 1)
    share = (f["rt"] / f["tp"]).clip(0, 1)
    out["route_last"] = share.groupby(f["player_id"]).transform(lambda s: s.ffill().shift())
    out["route_trend"] = out["route_last"] - out["route_avg"]
    return out[["game_id", "player_id"] + COLS]
