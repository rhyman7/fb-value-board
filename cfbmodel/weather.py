"""Wind at kickoff for upcoming outdoor college games, from the free Open-Meteo forecast.

Played games use the wind recorded in the cfbfastR lines file. Upcoming outdoor games
within a week get the forecast for the kickoff hour at the home stadium (10 m wind, mph),
plus the gusts and the direction the wind blows from, in the same call. Neutral-site games
are skipped (the venue's location is not in the data). Every forecast is logged to
data/cfb/wind_log.csv (speed, gusts, direction) so the wind feature can later be graded on
forecasts. The model uses only the speed. A failed request leaves the wind blank, which the
model treats as a typical day.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from nflmodel.weather import _rec, append_log, at_hour, read_log  # noqa: F401  (same parsing and log format)

from . import config as C

LOG = C.DATA / "wind_log.csv"
LOG_COLS = ["game_id", "fetched", "wind_mph", "gust_mph", "dir_deg"]
HOURLY = ("wind_speed_10m", "wind_gusts_10m", "wind_direction_10m")
URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly=" + ",".join(HOURLY) + "&wind_speed_unit=mph&timezone=UTC&start_date={d}&end_date={d}")
MAX_DAYS = 7


def _fetch(lat: float, lon: float, day: str) -> dict:
    req = urllib.request.Request(URL.format(lat=lat, lon=lon, d=day), headers={"User-Agent": "cfb-model"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())["hourly"]


def forecast_wind(games: pd.DataFrame, today=None) -> dict:
    """{game_id: {"mph", "gust", "dir"}} for upcoming outdoor FBS games within a week."""
    today = (today or datetime.now(timezone.utc)).date()
    out = {}
    g = games[~games["played"] & (games["indoor"] == 0) & (games["neutral"] == 0) & (games["fbs_game"] == 1)]
    for r in g.itertuples(index=False):
        ko = pd.Timestamp(r.kickoff)
        lat, lon = pd.to_numeric(r.lat, errors="coerce"), pd.to_numeric(r.lon, errors="coerce")
        if pd.isna(ko) or not np.isfinite(lat) or not np.isfinite(lon):
            continue
        ko = ko.tz_convert("UTC")
        if not (today <= ko.date() <= today + timedelta(days=MAX_DAYS)):
            continue
        try:
            hourly = _fetch(round(lat, 3), round(lon, 3), ko.date().isoformat())
            out[r.game_id] = at_hour(hourly, ko.hour)
        except Exception as exc:
            print(f"  no wind forecast for {r.game_id} ({exc})")
    return out


def log_rows(fc: dict, now: str | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows = []
    for gid, v in fc.items():
        r = _rec(v)
        rows.append({"game_id": gid, "fetched": now,
                     "wind_mph": None if r["mph"] is None else round(r["mph"], 1),
                     "gust_mph": None if r["gust"] is None else round(r["gust"], 1),
                     "dir_deg": None if r["dir"] is None else int(round(r["dir"]))})
    return pd.DataFrame(rows, columns=LOG_COLS)


def log_forecasts(fc: dict, path=LOG) -> None:
    if not fc:
        return
    append_log(log_rows(fc), path, LOG_COLS)


def fill_wind(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Forecast wind for upcoming games; record where each game's wind came from, with the
    forecast gusts and direction in `wind_gust` and `wind_dir` (blank when unknown)."""
    g = games.copy()
    g["wind_source"] = np.where(g["played"] & pd.to_numeric(g["wind"], errors="coerce").notna(), "recorded", "")
    g["wind_gust"] = np.nan
    g["wind_dir"] = np.nan
    try:
        fc = forecast_wind(g)
    except Exception as exc:
        print(f"  wind forecasts skipped ({exc})")
        fc = {}
    if fc:
        recs = {k: _rec(v) for k, v in fc.items()}
        m = g["game_id"].isin(fc)
        g.loc[m, "wind"] = g.loc[m, "game_id"].map({k: v["mph"] for k, v in recs.items()})
        g.loc[m, "wind_gust"] = g.loc[m, "game_id"].map({k: v["gust"] for k, v in recs.items()}).astype(float)
        g.loc[m, "wind_dir"] = g.loc[m, "game_id"].map({k: v["dir"] for k, v in recs.items()}).astype(float)
        g.loc[m, "wind_source"] = "forecast"
        log_forecasts(fc)
    if verbose:
        print(f"  wind forecasts: {len(fc)} games")
    return g
