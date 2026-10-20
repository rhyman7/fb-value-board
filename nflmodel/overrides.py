"""Hand-entered corrections for upcoming games, read from data/overrides.csv.

The schedule's projected starting quarterbacks and the official injury report lag the
news by a few days (game-status tags only come out on Friday). One row fixes one player
for one game, so it lapses by itself once that game is played.

Columns:
  game_id    e.g. 2026_07_KC_SEA
  team       the player's team
  kind       starter  - this quarterback starts (the game model's quarterback feature)
             out      - the player does not play (the player projections)
  player_id  gsis id, e.g. 00-0034869
  name       display name
  status     for kind=out, the label shown (Out, Doubtful, Injured reserve); default Out
  source     where the news came from, with its date
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

PATH = C.DATA / "overrides.csv"
COLS = ["game_id", "team", "kind", "player_id", "name", "status", "source"]


def load(path=None) -> pd.DataFrame:
    path = PATH if path is None else path
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    o = pd.read_csv(path, dtype=str, comment="#").fillna("")
    bad = set(COLS) - set(o.columns)
    if bad:
        raise ValueError(f"{path.name} is missing columns: {sorted(bad)}")
    o = o[o["game_id"].str.strip() != ""].copy()
    for c in COLS:
        o[c] = o[c].str.strip()
    o["kind"] = o["kind"].str.lower()
    unknown = sorted(set(o["kind"]) - {"starter", "out"})
    if unknown:
        raise ValueError(f"{path.name}: unknown kind {unknown} (use starter or out)")
    return o[COLS].reset_index(drop=True)


def apply_starters(games: pd.DataFrame, ov: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Set the starting quarterback of upcoming games. Played games are never changed."""
    rows = ov[ov["kind"] == "starter"]
    if rows.empty:
        return games
    g = games.copy()
    for r in rows.itertuples():
        hit = g.index[(g["game_id"] == r.game_id) & ~g["played"]]
        if not len(hit):
            continue                                  # played already, or not on the schedule
        i = hit[0]
        side = "home" if g.at[i, "home_team"] == r.team else "away" if g.at[i, "away_team"] == r.team else None
        if side is None:
            print(f"  override skipped: {r.team} is not in {r.game_id}")
            continue
        if verbose and g.at[i, f"{side}_qb_id"] != r.player_id:
            print(f"  override: {r.team} starter {g.at[i, f'{side}_qb_name']} -> {r.name} ({r.game_id})")
        g.at[i, f"{side}_qb_id"] = r.player_id
        g.at[i, f"{side}_qb_name"] = r.name
    return g


def apply_out(rows: pd.DataFrame, ov: pd.DataFrame) -> pd.DataFrame:
    """Mark players out of upcoming games in the player table (game_id, player_id rows)."""
    o = ov[ov["kind"] == "out"]
    if o.empty or rows.empty:
        return rows
    rows = rows.copy()
    rows["report"] = rows["report"].astype(object)
    upcoming = ~rows["played"].astype(bool) if "played" in rows.columns else np.ones(len(rows), bool)
    for r in o.itertuples():
        m = upcoming & (rows["game_id"] == r.game_id) & (rows["player_id"] == r.player_id)
        rows.loc[m, "active"] = 0
        rows.loc[m, "report"] = r.status or "Out"
    return rows
