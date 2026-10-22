"""This week's starting quarterbacks, from the CFB Matchup Dashboard's daily check.

The CFB dashboard's daily task (rhyman7/cfb, WEEKLY_PROMPT.md step 7e) checks on every run
who is expected to start at quarterback this week for every FBS team playing (news since the
team's last game: injuries, benchings, named starters) and publishes
overrides/qb_starters.json:

    {"season": 2026, "week": 8, "checked": "YYYY-MM-DD",
     "starters": {"Michigan": {"qb": "Tommy Carr", "status": "reported", "source": ..., "note": ...}}}

Team names are the schools' names, the same as the board's data names. status is confirmed or
reported (a change from the last game's starter), uncertain (the listed QB is the default)
or no change.

The board downloads it on each build (a copy is kept in docs/cfb/starters.json) and, when
a team's next game is in the file's week, for that game and the team's later games this season:
  - the game model starts that quarterback (the quarterback feature), and
  - the player projections give him the team's passes.
Played games and the backtests are untouched. Injury designations (designations.py) are
applied after this, so a listed starter who is Out or Doubtful is still replaced.
A quarterback who has never thrown a pass this season for the team gets the model's
replacement-level value when the file says the change is confirmed or reported, and is
skipped otherwise. A file for another season or week changes nothing.
"""
from __future__ import annotations

import json
import urllib.request

import pandas as pd

from . import config as C
from .designations import norm

URL = "https://raw.githubusercontent.com/rhyman7/cfb/main/overrides/qb_starters.json"
PATH = C.DOCS / "starters.json"
COLS = ["game_id", "team", "side", "qb_id", "name", "status"]
CHANGE = {"confirmed", "reported"}


def fetch(path=None, url: str = URL, timeout: int = 20, verbose: bool = True) -> bool:
    """Download the dashboard's list to path. On failure the previous copy stays."""
    path = PATH if path is None else path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        if not isinstance(d.get("starters"), dict):
            raise ValueError("no starters")
        path.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        return True
    except Exception as e:                    # network or format: keep the last copy
        if verbose:
            print(f"  starting QBs: could not download {url} ({e}); using the saved copy")
        return False


def load(path=None) -> dict:
    path = PATH if path is None else path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {}


def resolve(d: dict, games: pd.DataFrame, qb: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """(game_id, team, side, qb_id, name, status) for each listed team's upcoming games, when
    its next game is in the file's week."""
    empty = pd.DataFrame(columns=COLS)
    if not d or not d.get("starters") or not games["played"].any():
        return empty
    season = int(games.loc[games["played"], "season"].max())
    if d.get("season") != season:
        return empty
    up = games[~games["played"] & (games["season"] == season)]
    rows, missing = [], []
    for team, e in d["starters"].items():
        name, status = (e or {}).get("qb"), str((e or {}).get("status") or "").lower()
        if not name:
            continue
        mine = up[(up["home_team"] == team) | (up["away_team"] == team)].sort_values("order")
        if mine.empty:
            continue
        if int(mine.iloc[0]["week"]) != int(d.get("week", -1)):
            continue                              # the list is for another week
        cand = qb[(qb["season"] == season) & (qb["team"] == team)]
        hit = cand.loc[cand["name"].map(norm) == norm(name), "qb_id"]
        if len(hit):
            qid = str(hit.iloc[0])
        elif status in CHANGE:
            qid = "replacement"
            missing.append(f"{team} {name}")
        else:
            continue
        for g in mine.itertuples(index=False):    # this week's game and the rest of the season
            rows.append({"game_id": g.game_id, "team": team, "side": "home" if g.home_team == team else "away",
                         "qb_id": qid, "name": name, "status": status})
    if verbose and missing:
        print(f"  starting QBs with no pass this season (replacement level): {', '.join(missing)}")
    return pd.DataFrame(rows, columns=COLS)


def apply(games: pd.DataFrame, picks: pd.DataFrame, engine=None, verbose: bool = True) -> pd.DataFrame:
    """Set the starting quarterback of each pick's game. Played games are never changed."""
    if picks.empty:
        return games
    g = games.copy()
    last = {}
    season = int(g.loc[g["played"], "season"].max())
    up = g.loc[~g["played"] & (g["season"] == season)]
    if engine is not None and len(up):
        now = int(up["order"].min())
        last = engine.last_qb[engine.last_qb["order"] < now].groupby("team").tail(1).set_index("team")["qb_id"].to_dict()
    changed = set()
    for r in picks.itertuples(index=False):
        m = (g["game_id"] == r.game_id) & ~g["played"]
        if not m.any():
            continue
        g.loc[m, f"{r.side}_qb_id"] = r.qb_id
        if last and last.get(r.team) != r.qb_id:
            changed.add(f"{r.team} {r.name}")
    if verbose:
        print(f"  starting QBs: {picks['team'].nunique()} teams from the CFB dashboard's list"
              + (f"; changed from the last starter: {', '.join(sorted(changed))}" if changed else ""))
    return g
