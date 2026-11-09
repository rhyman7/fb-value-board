"""Injury designations for upcoming college games: the conferences' official availability
reports where a team has one, Covers.com for everyone else.

Official reports first. The four power conferences publish availability reports for conference
games (daily from three days out, a Game Day report 90 minutes to two hours before kickoff;
players not listed are available). The College availability Action writes each team's newest
live report to docs/cfb/availability.json (availability_log.current_reports). For every team in
it, the report's designations replace the team's Covers rows (a Covers row that says out for the
season stays unless the report lists him): Out -> O, Doubtful -> D, Questionable -> Q,
Probable -> P, Game Time Decision -> Q, Out (1st Half) -> P (a targeting suspension: he plays
the second half). Report rows carry `src` = the conference; Covers rows `src` = "Covers".

Covers.com for the rest. The daily college check task reads
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
A designation the player has played through (a game for his team after its `updated` date)
is left out: Covers keeps old entries. A newer one, posted after that game, stands.
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
AVAIL_PATH = C.DOCS / "availability.json"
MAX_AGE_DAYS = 3
MAX_AVAIL_AGE_DAYS = 5
OUT_CODES = {"O": "Out", "D": "Doubtful"}
WORDS = {**OUT_CODES, "Q": "Questionable", "P": "Probable"}
# the conferences' own statuses -> the board's code
REPORT_CODES = {"out": "O", "doubtful": "D", "questionable": "Q", "probable": "P", "game time decision": "Q", "out (1st half)": "P"}
COLS = ["team", "name", "pos", "d", "injury", "updated", "season", "key", "src", "raw"]
# Chance to play for the official reports' Questionable and Game Time Decision (Nov 9, 2026 test,
# Project doc claude/questionable-gtd-test-results-2026-11-09.md: fitted on 2026 weeks 0-6 from PFF
# snaps, kept on weeks 7+). Only official-report designations; Covers' Questionable is untested and
# stays a label. Refit at the January recheck, not mid-season.
PLAY_CHANCE = {"Questionable": 0.66, "Game Time Decision": 0.64}


def norm(name) -> str:
    """Comparable name: lower case, no suffixes or punctuation."""
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(name or "").lower())
    return re.sub(r"[^a-z]", "", s)


def load(path=None, today: dt.date | None = None, avail_path=None) -> pd.DataFrame:
    """Every current designation (team is the board abbreviation): the conference reports' rows
    for teams with a live report, Covers' rows for the rest. Empty if both files are missing or
    stale."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    cov = load_covers(PATH if path is None else path, today)
    rep, covered = _reports(AVAIL_PATH if avail_path is None else avail_path, today)
    if not covered:
        return cov
    if cov.empty:
        return rep
    listed = set(zip(rep["team"], rep["key"]))
    on_team = cov["team"].isin(covered)      # a live report with nobody listed still means everyone is available
    keep = ~on_team | (cov["season"] & ~pd.Series([(t, k) in listed for t, k in zip(cov["team"], cov["key"])], index=cov.index))
    dropped = int((~keep).sum())
    if dropped:
        print(f"  designations: {dropped} Covers rows replaced by the conference reports ({len(covered)} teams)")
    return pd.concat([cov[keep], rep], ignore_index=True)[COLS]


def load_covers(path, today: dt.date) -> pd.DataFrame:
    """docs/cfb/designations.json's rows, or an empty frame if it is missing or stale."""
    empty = pd.DataFrame(columns=COLS)
    if not path.exists():
        return empty
    d = json.loads(path.read_text())
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
    o["src"] = "Covers"
    o["raw"] = o["d"].map(WORDS)
    return o[COLS]


def load_reports(path, today: dt.date) -> pd.DataFrame:
    """docs/cfb/availability.json's live reports as designation rows (team is the board
    abbreviation; `updated` is the report date), or an empty frame if the file is missing, old,
    or every game in it has been played."""
    return _reports(path, today)[0]


def _reports(path, today: dt.date) -> tuple[pd.DataFrame, set]:
    """(designation rows, board abbreviations of every team with a live report, listed players or not)."""
    empty = pd.DataFrame(columns=COLS)
    if not path.exists():
        return empty, set()
    try:
        a = json.loads(path.read_text())
        fetched = dt.datetime.fromisoformat(a["fetched"].replace("Z", "+00:00")).date()
    except (KeyError, ValueError, TypeError):
        print(f"  availability ignored: {path.name} unreadable")
        return empty, set()
    if (today - fetched).days > MAX_AVAIL_AGE_DAYS:
        print(f"  availability ignored: fetched {fetched}, more than {MAX_AVAIL_AGE_DAYS} days ago")
        return empty, set()
    rows, covered = [], set()
    for team, rep in (a.get("teams") or {}).items():
        try:
            if dt.date.fromisoformat(rep.get("game_date", "")) < today:
                continue
        except ValueError:
            pass
        if not rep.get("abbr"):
            continue
        covered.add(rep["abbr"])
        for p in rep.get("players") or []:
            d = REPORT_CODES.get(str(p.get("status") or "").lower())
            if d is None or not p.get("name"):
                continue
            rows.append({"team": rep["abbr"], "name": p["name"], "pos": p.get("pos") or "", "d": d, "injury": "",
                         "updated": rep.get("published"), "season": bool(p.get("season_out")), "key": norm(p["name"]),
                         "src": rep.get("conf", "report"), "raw": str(p.get("status") or "")})
    return (pd.DataFrame(rows)[COLS] if rows else empty), covered


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


def drop_played(des: pd.DataFrame, played: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Leave out designations the player has played through: a row whose `updated` date is
    before a game he has since played in for that team (Covers keeps old entries). A newer
    designation, posted after his last game, stands. played has team_name, key (norm name)
    and date (game day) for each game a player played in; rows without `updated` stay."""
    if des.empty or played.empty or "team_name" not in des.columns:
        return des
    last = played.groupby(["team_name", "key"])["date"].max()
    upd = pd.to_datetime(des["updated"], errors="coerce")
    seen = pd.Series([last.get((t, k)) for t, k in zip(des["team_name"], des["key"])], index=des.index)
    seen = pd.to_datetime(seen, errors="coerce")
    stale = upd.notna() & seen.notna() & (seen.dt.normalize() > upd)
    if verbose and stale.any():
        names = [f"{t} {n} ({d})" for t, n, d in zip(des.loc[stale, "team_name"], des.loc[stale, "name"], des.loc[stale, "d"])]
        print(f"  designations dropped, played since they were posted: {', '.join(names)}")
    return des[~stale].reset_index(drop=True)


def played_from_qbs(qb: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """(team_name, key, date) for every game a quarterback has dropbacks in."""
    q = qb.merge(games[["game_id", "gameday"]], on="game_id")
    return pd.DataFrame({"team_name": q["team"], "key": q["name"].map(norm), "date": q["gameday"]})


def played_from_box(pg: pd.DataFrame, info: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """(team_name, key, date) for every game a player has a pass, carry or catch in."""
    off = pg[pg[["pass_att", "carries", "receptions"]].sum(axis=1) > 0]
    team = pd.concat([games[["home_id", "home_team"]].set_axis(["team_id", "team_name"], axis=1),
                      games[["away_id", "away_team"]].set_axis(["team_id", "team_name"], axis=1)]).drop_duplicates("team_id")
    o = off[["game_id", "team_id", "player_id"]].merge(games[["game_id", "gameday"]], on="game_id").merge(team, on="team_id")
    return pd.DataFrame({"team_name": o["team_name"], "key": o["player_id"].map(info["name"]).map(norm), "date": o["gameday"]})


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
