"""Download cfbfastR data and reduce it to small per-season tables.

Per season, cached in data/cfb/cache/:
  games_<s>.csv       every game with an FBS team: schedule, result, lines (closing and,
                      from 2021, opening), recorded wind, dome, talent, returning production
  team_games_<s>.csv  one row per offense per game: EPA, success, pass/rush splits, snaps
  qb_games_<s>.csv    one row per passer per game: dropbacks and EPA

Team keys are school names (e.g. "Ohio State"). Opponents outside FBS that season are
pooled into one key, C.POOL, so FBS teams' games against them still count for ratings.
Only the season in progress (or a missing one) is downloaded again.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

PBP_COLS = ["game_id", "season", "week", "pos_team_id", "def_pos_team_id", "homeTeamId", "awayTeamId",
            "period", "pos_score_diff_start", "rush", "pass", "kneel_down", "penalty_no_play",
            "EPA", "EPA_success", "passer_player_id", "passer_player_name", "seasonType"]


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _retry(fn, tries: int = 3):
    """Downloads through the proxy are occasionally cut off; try again before giving up."""
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise


def _get(name: str, season: int) -> pd.DataFrame:
    return _retry(lambda: _get_once(name, season))


def _get_pbp(season: int) -> pd.DataFrame:
    return _retry(lambda: _get_pbp_once(season))


def _get_once(name: str, season: int) -> pd.DataFrame:
    url = f"{C.BASE}/{C.FILES[name].format(s=season)}"
    req = urllib.request.Request(url, headers={"User-Agent": "cfb-model"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    return pd.read_parquet(io.BytesIO(raw))


def _get_pbp_once(season: int) -> pd.DataFrame:
    url = f"{C.BASE}/{C.FILES['pbp'].format(s=season)}"
    C.RAW.mkdir(parents=True, exist_ok=True)
    path = C.RAW / f"pbp_{season}.parquet"
    req = urllib.request.Request(url, headers={"User-Agent": "cfb-model"})
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema.names)
    p = pd.read_parquet(path, columns=[c for c in PBP_COLS if c in have])
    path.unlink()
    return p


# --------------------------------------------------------------------------
# Games
# --------------------------------------------------------------------------

def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _add_scheduled(s: pd.DataFrame, season: int) -> pd.DataFrame:
    """Add games not yet played from the full schedule file. Failure leaves the schedule as is."""
    try:
        def go():
            req = urllib.request.Request(C.FULL_SCHEDULE.format(s=season), headers={"User-Agent": "cfb-model"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return pd.read_csv(io.BytesIO(r.read()))
        f = _retry(go)
    except Exception as exc:
        print(f"  no full schedule for {season} ({exc})")
        return s
    f = f[~f["id"].isin(s["game_id"]) & ~f["status_type_completed"].fillna(False).astype(bool)]
    if not len(f):
        return s
    add = pd.DataFrame({
        "game_id": f["id"], "season": season, "week": f["week"], "season_type": f["season_type"],
        "game_date": f["start_date"], "neutral_site": f["neutral_site"], "conference_competition": f["conference_competition"],
        "home_id": f["home_id"], "away_id": f["away_id"], "home_team": f["home_display_name"], "away_team": f["away_display_name"],
        "home_score": np.nan, "away_score": np.nan, "venue": f["venue_full_name"], "status": f["status_type_name"],
        "venue_indoor": f.get("venue_indoor")})
    return pd.concat([s, add], ignore_index=True)


def reduce_games(season: int) -> pd.DataFrame:
    s = _get("schedule", season)
    t = _get("teams", season)
    s = _add_scheduled(s, season)
    fbs = t[t["classification"] == "fbs"]
    name = dict(zip(fbs["team_id"].astype(int), fbs["school"]))
    s = s[s["home_id"].isin(name) | s["away_id"].isin(name)].copy()
    s = s[s["status"].astype(str).str.contains("FINAL|SCHEDULED|STATUS", na=True)]
    key = lambda i: name.get(int(i), C.POOL)
    post = s["season_type"].astype(str) == "3"
    g = pd.DataFrame({
        "game_id": s["game_id"].astype(int), "season": season, "week": s["week"].astype(int),
        "postseason": post.astype(int).to_numpy(),
        "kickoff": pd.to_datetime(s["game_date"], utc=True, errors="coerce"),
        "home_id": s["home_id"].astype(int), "away_id": s["away_id"].astype(int),
        "home_team": s["home_id"].map(key), "away_team": s["away_id"].map(key),
        "neutral": s["neutral_site"].fillna(False).astype(bool).astype(int).to_numpy(),
        "conference_game": s["conference_competition"].fillna(False).astype(bool).astype(int).to_numpy(),
        "home_score": _num(s["home_score"]), "away_score": _num(s["away_score"]),
        "venue": s["venue"], "status": s["status"],
        "venue_indoor": s["venue_indoor"].astype(object) if "venue_indoor" in s else np.nan,
    })
    done = g["status"].astype(str).str.upper().str.contains("FINAL")
    g.loc[~done, ["home_score", "away_score"]] = np.nan
    g = g.drop(columns="status")

    try:
        ln = _get("lines", season)
    except Exception:
        ln = pd.DataFrame(columns=["game_id"])
    keep = {"spread": "cfb_spread", "spread_open": "cfb_spread_open", "over_under": "total_line",
            "over_under_open": "open_total", "wind_speed": "wind", "temperature": "temp",
            "home_talent": "home_talent", "away_talent": "away_talent",
            "home_ovr_rtprod": "home_rtprod", "away_ovr_rtprod": "away_rtprod",
            "home_off_rtprod": "home_off_rtprod", "away_off_rtprod": "away_off_rtprod",
            "home_def_rtprod": "home_def_rtprod", "away_def_rtprod": "away_def_rtprod",
            "home_returning_qb": "home_ret_qb", "away_returning_qb": "away_ret_qb",
            "home_dome": "dome", "home_latitude": "lat", "home_longitude": "lon",
            "home_elevation": "elevation", "home_timezone": "home_tz", "away_timezone": "away_tz",
            "away_latitude": "away_lat", "away_longitude": "away_lon", "venue_id": "venue_id"}
    ln = ln[[c for c in ["game_id", "home_team_id", *keep] if c in ln.columns]].rename(columns=keep)
    if len(ln):
        ln["game_id"] = ln["game_id"].astype(int)
        # The lines file and the schedule occasionally disagree on which team is home (neutral
        # sites such as Army-Navy). Put the lines file's numbers on the schedule's sides.
        sides = g.set_index("game_id")["away_id"]
        flip = ln["home_team_id"].notna() & (ln["home_team_id"].astype(float) == ln["game_id"].map(sides).astype(float))
        for c in ("cfb_spread", "cfb_spread_open"):
            ln.loc[flip, c] = -ln.loc[flip, c]
        pairs = [("home_talent", "away_talent"), ("home_rtprod", "away_rtprod"), ("home_off_rtprod", "away_off_rtprod"),
                 ("home_def_rtprod", "away_def_rtprod"), ("home_ret_qb", "away_ret_qb"), ("home_tz", "away_tz"),
                 ("lat", "away_lat"), ("lon", "away_lon")]
        for a, b in pairs:
            if a in ln and b in ln:
                ln.loc[flip, [a, b]] = ln.loc[flip, [b, a]].to_numpy()
        g = g.merge(ln.drop(columns=["home_team_id"]), on="game_id", how="left")
    for c in keep.values():
        if c not in g:
            g[c] = np.nan
    # nflverse convention used by the shared code: spread_line > 0 means the home team is favoured.
    g["spread_line"] = -_num(g["cfb_spread"])
    g["open_spread"] = -_num(g["cfb_spread_open"])
    g = g.drop(columns=["cfb_spread", "cfb_spread_open"])

    # Home stadium location for games without a line (used for wind forecasts and travel).
    home_site = fbs.set_index("school")[["latitude", "longitude", "dome", "timezone"]]
    for c in ("lat", "lon", "dome"):
        g[c] = g[c].astype(object)
    miss = g["lat"].isna() & g["home_team"].isin(home_site.index) & (g["neutral"] == 0)
    g.loc[miss, "lat"] = g.loc[miss, "home_team"].map(home_site["latitude"])
    g.loc[miss, "lon"] = g.loc[miss, "home_team"].map(home_site["longitude"])
    g.loc[g["dome"].isna() & miss, "dome"] = g.loc[g["dome"].isna() & miss, "home_team"].map(home_site["dome"])
    vi = g["venue_indoor"].map(lambda x: 1 if x is True or str(x).lower() == "true" else (0 if x is False or str(x).lower() == "false" else np.nan))
    g["dome"] = g["dome"].where(g["dome"].notna(), vi)
    g = g.drop(columns="venue_indoor")
    g["dome"] = g["dome"].map(lambda x: 1 if x is True or x == 1 or str(x).lower() == "true" else 0)
    return g


# --------------------------------------------------------------------------
# Play-by-play
# --------------------------------------------------------------------------

def reduce_pbp(season: int, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = _get_pbp(season)
    p = p[p["game_id"].astype(int).isin(set(games["game_id"]))].copy()
    p["game_id"] = p["game_id"].astype(int)
    ids = games.set_index("game_id")
    name_of = pd.concat([games[["home_id", "home_team"]].set_axis(["id", "team"], axis=1),
                         games[["away_id", "away_team"]].set_axis(["id", "team"], axis=1)]).drop_duplicates("id")
    name_of = dict(zip(name_of["id"].astype(int), name_of["team"]))
    flag = lambda c: p[c].fillna(False).astype(bool) if c in p else pd.Series(False, index=p.index)

    scrim = (flag("rush") | flag("pass")) & ~flag("kneel_down") & ~flag("penalty_no_play") & p["EPA"].notna()
    p = p[scrim & p["pos_team_id"].notna()].copy()
    p["team"] = p["pos_team_id"].astype(int).map(name_of)
    p["opp"] = p["def_pos_team_id"].astype(int).map(name_of)
    p = p[p["team"].notna() & p["opp"].notna() & (p["team"] != p["opp"])]
    p["side"] = np.where(p["pos_team_id"].astype(int) == p["game_id"].map(ids["home_id"]), "home", "away")
    p["is_pass"] = flag("pass").loc[p.index].astype(int)
    p["is_rush"] = 1 - p["is_pass"]
    period = _num(p["period"]).fillna(4).astype(int)
    lead = _num(p["pos_score_diff_start"]).abs().fillna(0)
    p["garbage"] = lead > period.map(C.GARBAGE_LEAD).fillna(999)
    p["epa"] = p["EPA"].clip(-C.EPA_CLIP, C.EPA_CLIP)
    p["succ"] = p["EPA_success"].fillna(False).astype(bool).astype(int)

    snaps = p.groupby(["game_id", "side"]).size().rename("snaps")
    c = p[~p["garbage"]]
    grp = c.groupby(["game_id", "side"], sort=False)
    tg = grp.agg(team=("team", "first"), opp=("opp", "first"), plays=("epa", "size"), epa=("epa", "sum"),
                 succ=("succ", "sum"), n_pass=("is_pass", "sum"), n_rush=("is_rush", "sum")).reset_index()
    pe = c[c.is_pass == 1].groupby(["game_id", "side"])["epa"].sum().rename("pass_epa")
    re = c[c.is_rush == 1].groupby(["game_id", "side"])["epa"].sum().rename("rush_epa")
    tg = (tg.join(pe, on=["game_id", "side"]).join(re, on=["game_id", "side"]).join(snaps, on=["game_id", "side"])
          .fillna({"pass_epa": 0, "rush_epa": 0}))
    tg.insert(0, "season", season)

    d = p[(p.is_pass == 1) & p["passer_player_id"].notna()].copy()
    d["qb_id"] = _num(d["passer_player_id"]).astype("Int64")
    qb = d.groupby(["game_id", "side", "qb_id"], sort=False).agg(
        name=("passer_player_name", "first"), team=("team", "first"),
        dropbacks=("epa", "size"), epa=("epa", "sum"), clean_db=("garbage", lambda x: int((~x).sum())),
    ).reset_index()
    qb.insert(0, "season", season)
    return tg, qb


# --------------------------------------------------------------------------
# Everything
# --------------------------------------------------------------------------

def load_season(season: int, refresh: bool = False, verbose: bool = True):
    C.CACHE.mkdir(parents=True, exist_ok=True)
    paths = [C.CACHE / f"{k}_{season}.csv" for k in ("games", "team_games", "qb_games")]
    if refresh or not all(p.exists() for p in paths):
        if verbose:
            print(f"  downloading {season} ...", flush=True)
        g = reduce_games(season)
        tg, qb = reduce_pbp(season, g)
        g.to_csv(paths[0], index=False)
        tg.round(4).to_csv(paths[1], index=False)
        qb.round(4).to_csv(paths[2], index=False)
    g = pd.read_csv(paths[0])
    tg = pd.read_csv(paths[1])
    qb = pd.read_csv(paths[2])
    return g, tg, qb


HALF_COLS = ["game_id", "season", "home_h1", "away_h1"]


def _half_from_linescores(s: str):
    """Points in periods 1 and 2 from the schedule file's linescores text
    ("[{'value': 7.0, ..., 'period': 1}\\n {'value': 14.0, ..., 'period': 2} ...]")."""
    pts = {int(m.group(2)): float(m.group(1)) for m in re.finditer(r"'value':\s*([\d.]+).*?'period':\s*(\d+)", str(s or ""))}
    if 1 not in pts or 2 not in pts:
        return None
    return int(round(pts[1] + pts[2]))


def half_scores(season: int) -> pd.DataFrame:
    """Half-time scores of the season's finished games from the full schedule file's
    quarter linescores (ESPN). Columns: HALF_COLS."""
    def go():
        req = urllib.request.Request(C.FULL_SCHEDULE.format(s=season), headers={"User-Agent": "cfb-model"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return pd.read_csv(io.BytesIO(r.read()), low_memory=False)
    f = _retry(go)
    f = f[f["status_type_completed"].fillna(False).astype(bool)]
    out = pd.DataFrame({"game_id": f["id"].astype(int), "season": season,
                        "home_h1": f["home_linescores"].map(_half_from_linescores), "away_h1": f["away_linescores"].map(_half_from_linescores)})
    return out.dropna().astype({"home_h1": int, "away_h1": int})[HALF_COLS]


def load_half_scores(refresh: bool = False, verbose: bool = True, last_season: int | None = None) -> pd.DataFrame:
    """Half-time scores for every season (half_scores_<season>.csv in the cache; the season in
    progress is refreshed with `refresh`). A download failure keeps the cached file or skips
    the season."""
    C.CACHE.mkdir(parents=True, exist_ok=True)
    now = last_season or current_season()
    out = []
    for s in range(C.FIRST_SEASON, now + 1):
        path = C.CACHE / f"half_scores_{s}.csv"
        if (refresh and s == now) or not path.exists():
            try:
                if verbose:
                    print(f"  half-time scores {s} ...", flush=True)
                half_scores(s).to_csv(path, index=False)
            except Exception as exc:
                print(f"  no half-time scores for {s} ({exc})" + ("; using cached copy" if path.exists() else ""))
        if path.exists():
            out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True)[HALF_COLS] if out else pd.DataFrame(columns=HALF_COLS)


def load_all(refresh: bool = False, verbose: bool = True, last_season: int | None = None):
    """Return (games, team_games, qb_games) for every season, joined to a running order."""
    now = last_season or current_season()
    gs, tgs, qbs = [], [], []
    for s in range(C.FIRST_SEASON, now + 1):
        try:
            g, tg, qb = load_season(s, refresh=refresh and s == now, verbose=verbose)
        except Exception as exc:
            print(f"  could not load {s} ({exc}); skipping")
            continue
        gs.append(g), tgs.append(tg), qbs.append(qb)
    games = pd.concat(gs, ignore_index=True)
    games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True)
    games["gameday"] = games["kickoff"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    games["order"] = games["season"] * 100 + np.where(games["postseason"] == 1, C.POSTSEASON_WEEK, games["week"])
    games["played"] = games["home_score"].notna() & games["away_score"].notna()
    games["result"] = games["home_score"] - games["away_score"]
    games["total"] = games["home_score"] + games["away_score"]
    games["indoor"] = games["dome"].fillna(0).astype(int)
    games["playoff"] = games["postseason"]
    games["fbs_game"] = ((games["home_team"] != C.POOL) & (games["away_team"] != C.POOL)).astype(int)
    games = games.sort_values(["order", "kickoff", "game_id"]).reset_index(drop=True)

    # Rest: days since each team's previous game (capped), 0 difference when unknown.
    long = pd.concat([games[["game_id", "season", "gameday"]].assign(team=games[s], side=s[:4]) for s in ("home_team", "away_team")])
    long = long[long.team != C.POOL].sort_values(["team", "gameday"])
    long["rest"] = long.groupby(["team", "season"])["gameday"].diff().dt.days.clip(upper=21)
    rest = long.pivot_table(index="game_id", columns="side", values="rest")
    games["home_rest"] = games["game_id"].map(rest.get("home", pd.Series(dtype=float)))
    games["away_rest"] = games["game_id"].map(rest.get("away", pd.Series(dtype=float)))

    tg = pd.concat(tgs, ignore_index=True)
    qb = pd.concat(qbs, ignore_index=True)
    meta = games[["game_id", "order", "week", "home_score", "away_score", "neutral"]]
    tg = tg.merge(meta, on="game_id", how="inner")
    tg["is_home"] = ((tg["side"] == "home") & (tg["neutral"] == 0)).astype(int)
    tg["pts"] = np.where(tg["side"] == "home", tg["home_score"], tg["away_score"])
    tg = tg[tg["pts"].notna()].drop(columns=["home_score", "away_score"])
    qb = qb.merge(meta[["game_id", "order"]], on="game_id", how="inner")
    qb["qb_id"] = qb["qb_id"].astype("int64").astype(str)

    # Each team's starting quarterback in each played game: the passer with the most dropbacks.
    top = qb.sort_values("dropbacks").groupby(["game_id", "side"]).tail(1).set_index(["game_id", "side"])["qb_id"]
    games["home_qb_id"] = [top.get((g, "home")) for g in games["game_id"]]
    games["away_qb_id"] = [top.get((g, "away")) for g in games["game_id"]]
    return games, tg, qb
