"""Injury designations for upcoming college games, from Covers.com.

College football has no official injury report. The daily college check task reads
Covers.com's NCAAF injury page and writes docs/cfb/designations.json (see CONVENTIONS.md):

    {"source": "Covers.com", "url": ..., "checked": "YYYY-MM-DD",
     "players": [{"team": "APP", "name": "Malachi Singleton", "pos": "QB", "d": "Q",
                  "injury": "Ankle", "updated": "YYYY-MM-DD", "season": false}]}

team is the board's abbreviation; name is the board's name for skill players, or Covers'
own ("J. Smith") for anyone else (linemen, defense). d is O, D, Q or P.

How the board uses it, for upcoming games only (played games and the backtests are untouched):
  - Out or Doubtful: the player is out of his team's next game ("season": true, every
    upcoming game). He gets no projection, his share of carries, catches and passes goes to
    the teammates who play, and he is listed under "Out" on the team's card. A quarterback
    out who started the team's last game is replaced in the game model by the team's
    next quarterback by dropbacks this season.
  - Questionable or Probable: projected as normal; the page shows the designation.
  - Linemen and defensive players are listed under "Out" but change no numbers: there is
    no college injury history to measure what they are worth.
A file not checked in the last MAX_AGE_DAYS days is ignored.
"""
from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from . import config as C

PATH = C.DOCS / "designations.json"
MAX_AGE_DAYS = 3
OUT_CODES = {"O": "Out", "D": "Doubtful"}
WORDS = {**OUT_CODES, "Q": "Questionable", "P": "Probable"}
COLS = ["team", "name", "pos", "d", "injury", "updated", "season", "key"]


def norm(name) -> str:
    """Comparable name: lower case, no suffixes or punctuation."""
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(name or "").lower())
    return re.sub(r"[^a-z]", "", s)


def load(path=None, today: dt.date | None = None) -> pd.DataFrame:
    """The file's rows (team is the board abbreviation), or an empty frame if it is missing or stale."""
    path = PATH if path is None else path
    empty = pd.DataFrame(columns=COLS)
    if not path.exists():
        return empty
    d = json.loads(path.read_text())
    today = today or dt.datetime.now(dt.timezone.utc).date()
    try:
        age = (today - dt.date.fromisoformat(d["checked"])).days
    except (KeyError, ValueError):
        return empty
    if age > MAX_AGE_DAYS:
        print(f"  designations ignored: checked {d['checked']}, more than {MAX_AGE_DAYS} days ago")
        return empty
    rows = [r for r in d.get("players", []) if r.get("team") and r.get("name") and r.get("d") in ("O", "D", "Q", "P")]
    if not rows:
        return empty
    o = pd.DataFrame(rows)
    for c in COLS:
        if c not in o.columns:
            o[c] = None
    o["season"] = o["season"].fillna(False).astype(bool)
    o["key"] = o["name"].map(norm)
    return o[COLS]


def listed(des: pd.DataFrame, abbr: dict) -> pd.DataFrame:
    """Every designation row (O, D, Q, P), with the data's team name in `team_name`."""
    o = des.copy()
    back = {v: k for k, v in abbr.items()}
    o["team_name"] = o["team"].map(back)
    missing = o[o["team_name"].isna()]
    if len(missing):
        print(f"  designations: unknown teams {sorted(missing['team'].unique())}")
    return o[o["team_name"].notna()].reset_index(drop=True)


def outs(des: pd.DataFrame, abbr: dict) -> pd.DataFrame:
    """Out and Doubtful rows, with the data's team name in `team_name`. Accepts the raw rows
    (load) or rows already given a team name (listed)."""
    o = des if "team_name" in des.columns else listed(des, abbr)
    return o[o["d"].isin(list(OUT_CODES))].reset_index(drop=True)


def applies(out_rows: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Which upcoming games each out row covers: the team's next game, or all of them for
    "season". upcoming has game_id, team (data name) and t (order) for unplayed games."""
    if out_rows.empty or upcoming.empty:
        return out_rows.assign(game_id=pd.Series(dtype=object)).iloc[0:0]
    up = upcoming.sort_values("t")
    nxt = up.groupby("team").head(1)[["team", "game_id"]]
    a = out_rows.merge(up[["team", "game_id"]], left_on="team_name", right_on="team", suffixes=("", "_g"))
    a = a.drop(columns=["team_g"])
    first = a.merge(nxt.rename(columns={"team": "team_name", "game_id": "next_gid"}), on="team_name", how="left")
    keep = first["season"] | (first["game_id"] == first["next_gid"])
    return first[keep].drop(columns=["next_gid"]).reset_index(drop=True)


def apply_qbs(games: pd.DataFrame, qb: pd.DataFrame, out_rows: pd.DataFrame, engine, verbose: bool = True) -> pd.DataFrame:
    """Upcoming games whose team's last starter is out: start the team's next quarterback
    (most dropbacks this season among the others) in the game model."""
    qbo = out_rows[out_rows["pos"].fillna("").str.upper().eq("QB")]
    if qbo.empty:
        return games
    g = games.copy()
    season = int(g.loc[g["played"], "season"].max())
    up = pd.concat([g.loc[~g["played"] & (g["season"] == season), ["game_id", "order", s + "_team"]]
                    .rename(columns={s + "_team": "team"}).assign(side=s) for s in ("home", "away")])
    cov = applies(qbo, up.rename(columns={"order": "t"}))
    if cov.empty:
        return g
    order_now = int(up["order"].min())
    last = engine.last_qb[engine.last_qb["order"] < order_now].groupby("team").tail(1).set_index("team")["qb_id"]
    cur = qb[qb["season"] == season]
    for r in cov.itertuples():
        side = up.loc[(up["game_id"] == r.game_id) & (up["team"] == r.team_name), "side"]
        if not len(side):
            continue
        side = side.iloc[0]
        col = f"{side}_qb_id"
        starter = g.loc[g["game_id"] == r.game_id, col].iloc[0]
        if not isinstance(starter, str):
            starter = last.get(r.team_name)
        if starter is None or norm(engine.q_name.get(starter)) != r.key:
            continue                                    # the out quarterback is not the starter
        team_out = set(cov.loc[cov["team_name"] == r.team_name, "key"])
        mine = cur[cur["team"] == r.team_name].groupby("qb_id").agg(db=("dropbacks", "sum"), name=("name", "first"))
        mine = mine[~mine["name"].map(norm).isin(team_out) & (mine.index != starter)]
        nxt = mine["db"].idxmax() if len(mine) else "replacement"
        g.loc[g["game_id"] == r.game_id, col] = nxt
        if verbose:
            print(f"  designations: {r.team_name} QB {r.name} {OUT_CODES[r.d].lower()}; "
                  f"{engine.q_name.get(nxt, 'a replacement-level quarterback')} starts ({r.game_id})")
    return g


def player_outs(rows: pd.DataFrame, out_rows: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team, player_id, status) for out rows matched to a player on that team's
    upcoming rows. rows: the player table's rows for unplayed games."""
    if out_rows.empty or rows.empty:
        return pd.DataFrame(columns=["game_id", "team", "player_id", "status"])
    up = rows[["game_id", "team", "t"]].drop_duplicates()
    cov = applies(out_rows, up)
    names = rows[["game_id", "team", "player_id"]].drop_duplicates()
    names["key"] = names["player_id"].map(info["name"]).map(norm)
    m = cov.merge(names, left_on=["game_id", "team_name", "key"], right_on=["game_id", "team", "key"], suffixes=("", "_r"))
    m["status"] = m["d"].map(OUT_CODES)
    return m[["game_id", "team_r", "player_id", "status"]].rename(columns={"team_r": "team"}).drop_duplicates(["game_id", "player_id"])
