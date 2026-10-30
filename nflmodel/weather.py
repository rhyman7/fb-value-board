"""Wind at kickoff for upcoming outdoor games, from the free Open-Meteo forecast API.

Played games use the wind recorded in the nflverse games file. Upcoming games
have no recorded wind, so the board asks Open-Meteo for the forecast at each
stadium for the kickoff hour (10 m wind, mph), plus the gusts and the direction
the wind blows from, in the same call. Forecasts more than about a week out are
not available and are left blank; a blank becomes the league median in the
model, which is the same as having no wind information.

The model uses only the speed. Gusts and direction are logged (with the
along-field and crosswind parts, nflmodel/stadiums.py) so they can be tested
once enough games have been played, and the game header shows the gusts.

If the request fails (no network, rate limit) nothing breaks: the game simply
keeps a blank wind and the model treats it as a typical day.
"""
from __future__ import annotations

import json
import math
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C, stadiums
from .features_ext import SITES

LOG = C.DATA / "wind_log.csv"   # every forecast as fetched, so the wind under can be graded on forecasts
LOG_COLS = ["game_id", "fetched", "wind_mph", "gust_mph", "dir_deg", "along_mph", "cross_mph"]

HOURLY = ("wind_speed_10m", "wind_gusts_10m", "wind_direction_10m")
URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly=" + ",".join(HOURLY) + "&wind_speed_unit=mph&timezone=America%2FNew_York&start_date={d}&end_date={d}")
MAX_DAYS = 7


def _fetch(lat: float, lon: float, day: str) -> dict | None:
    """The day's hourly series: {"wind_speed_10m": [...24], "wind_gusts_10m": [...], "wind_direction_10m": [...]}."""
    req = urllib.request.Request(URL.format(lat=lat, lon=lon, d=day), headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())["hourly"]


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def at_hour(hourly, hour: int) -> dict:
    """{"mph", "gust", "dir"} from an Open-Meteo hourly block at one hour; gust and dir are
    None when the block lacks them (an old-style list of speeds is accepted too)."""
    h = min(max(int(hour), 0), 23)
    if isinstance(hourly, list):                                  # speeds only
        return {"mph": float(hourly[h]), "gust": None, "dir": None}
    speed = hourly["wind_speed_10m"]
    out = {"mph": float(speed[h]), "gust": None, "dir": None}
    for key, name in (("gust", "wind_gusts_10m"), ("dir", "wind_direction_10m")):
        series = hourly.get(name)
        if series is not None and len(series) > h:
            out[key] = _num(series[h])
    return out


def forecast_wind(games: pd.DataFrame, today=None) -> dict:
    """{game_id: {"mph", "gust", "dir"}} for unplayed outdoor, non-neutral games within a week.
    gust and dir are None when the forecast did not include them."""
    today = (today or datetime.now(timezone.utc)).date()
    out = {}
    g = games[~games["played"] & (games["indoor"] == 0) & (games["neutral"] == 0)]
    for r in g.itertuples(index=False):
        day = r.gameday.date()
        if not (today <= day <= today + timedelta(days=MAX_DAYS)) or r.home_team not in SITES:
            continue
        lat, lon, _ = SITES[r.home_team]
        try:
            hourly = _fetch(lat, lon, day.isoformat())
            hour = int(str(r.gametime or "13:00")[:2])            # kickoff hour, Eastern
            out[r.game_id] = at_hour(hourly, hour)
        except Exception as exc:                                  # network or format trouble
            print(f"  no wind forecast for {r.game_id} ({exc})")
    return out


def _rec(v) -> dict:
    """A forecast value as a record: a bare number is a speed with nothing else known."""
    if isinstance(v, dict):
        return {"mph": _num(v.get("mph")), "gust": _num(v.get("gust")), "dir": _num(v.get("dir"))}
    return {"mph": _num(v), "gust": None, "dir": None}


def _r1(v):
    return None if v is None else round(float(v), 1)


def log_rows(fc: dict, home: dict | None = None, now: str | None = None) -> pd.DataFrame:
    """The wind-log rows for one fetch: one per game, with the along-field and crosswind
    parts when `home` ({game_id: home team}) names a field whose bearing is known."""
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows = []
    for gid, v in fc.items():
        r = _rec(v)
        along = cross = None
        team = (home or {}).get(gid)
        if team is not None and r["mph"] is not None and r["dir"] is not None:
            a, c = stadiums.crosswind(team, r["mph"], r["dir"])
            along, cross = (None, None) if math.isnan(a) else (a, c)
        rows.append({"game_id": gid, "fetched": now, "wind_mph": _r1(r["mph"]), "gust_mph": _r1(r["gust"]),
                     "dir_deg": None if r["dir"] is None else int(round(r["dir"])),
                     "along_mph": _r1(along), "cross_mph": _r1(cross)})
    return pd.DataFrame(rows, columns=LOG_COLS)


def append_log(rows: pd.DataFrame, path, cols=LOG_COLS) -> None:
    """Append rows to a log. A log written before a column was added is rewritten once with
    the new header (old rows keep blanks in the new columns), so it stays readable."""
    path = Path(path)
    if not path.exists():
        rows.to_csv(path, index=False)
        return
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header == list(cols):
        rows.to_csv(path, mode="a", header=False, index=False)
        return
    old = read_log(path, cols)
    pd.concat([old, rows], ignore_index=True)[list(cols)].to_csv(path, index=False)


def read_log(path=LOG, cols=LOG_COLS) -> pd.DataFrame:
    """The wind log with every column in `cols` (blank where an older row lacks it)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(cols))
    log = pd.read_csv(path)
    for c in cols:
        if c not in log.columns:
            log[c] = np.nan
    return log[list(cols)]


def log_forecasts(fc: dict, path=LOG, home: dict | None = None) -> None:
    """Append forecasts to the wind log: one row per game per fetch."""
    if not fc:
        return
    append_log(log_rows(fc, home), path)


def fill_wind(games: pd.DataFrame, verbose: bool = True, log: bool = True) -> pd.DataFrame:
    """Return games with the wind column filled from forecasts where it is blank, and the
    forecast gusts and direction in `wind_gust` and `wind_dir` (blank when unknown)."""
    fc = forecast_wind(games)
    if verbose and fc:
        print(f"  wind forecasts for {len(fc)} upcoming outdoor games")
    games = games.copy()
    if log:
        home = dict(zip(games["game_id"], games["home_team"]))
        log_forecasts(fc, home=home)
    games["wind_source"] = games["wind"].notna().map({True: "recorded", False: ""})
    games["wind_gust"] = np.nan
    games["wind_dir"] = np.nan
    blank = games["wind"].isna() & games["game_id"].isin(fc)
    recs = {k: _rec(v) for k, v in fc.items()}
    games.loc[blank, "wind"] = games.loc[blank, "game_id"].map({k: v["mph"] for k, v in recs.items()})
    games.loc[blank, "wind_gust"] = games.loc[blank, "game_id"].map({k: v["gust"] for k, v in recs.items()}).astype(float)
    games.loc[blank, "wind_dir"] = games.loc[blank, "game_id"].map({k: v["dir"] for k, v in recs.items()}).astype(float)
    games.loc[blank, "wind_source"] = "forecast"
    return games
