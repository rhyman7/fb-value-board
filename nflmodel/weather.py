"""Wind at kickoff for upcoming outdoor games, from the free Open-Meteo forecast API.

Played games use the wind recorded in the nflverse games file. Upcoming games
have no recorded wind, so the board asks Open-Meteo for the forecast at each
stadium for the kickoff hour (10 m wind, mph). Forecasts more than about a week
out are not available and are left blank; a blank becomes the league median in
the model, which is the same as having no wind information.

If the request fails (no network, rate limit) nothing breaks: the game simply
keeps a blank wind and the model treats it as a typical day.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config as C
from .features_ext import SITES

LOG = C.DATA / "wind_log.csv"   # every forecast as fetched, so the wind under can be graded on forecasts

URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly=wind_speed_10m&wind_speed_unit=mph&timezone=America%2FNew_York&start_date={d}&end_date={d}")
MAX_DAYS = 7


def _fetch(lat: float, lon: float, day: str) -> list | None:
    req = urllib.request.Request(URL.format(lat=lat, lon=lon, d=day), headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())["hourly"]["wind_speed_10m"]


def forecast_wind(games: pd.DataFrame, today=None) -> dict:
    """{game_id: forecast wind in mph} for unplayed outdoor, non-neutral games within a week."""
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
            out[r.game_id] = float(hourly[min(max(hour, 0), 23)])
        except Exception as exc:                                  # network or format trouble
            print(f"  no wind forecast for {r.game_id} ({exc})")
    return out


def log_forecasts(fc: dict, path=LOG) -> None:
    """Append forecasts to the wind log: one row per game per fetch."""
    if not fc:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows = pd.DataFrame({"game_id": list(fc), "fetched": now, "wind_mph": [round(v, 1) for v in fc.values()]})
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def fill_wind(games: pd.DataFrame, verbose: bool = True, log: bool = True) -> pd.DataFrame:
    """Return games with the wind column filled from forecasts where it is blank."""
    fc = forecast_wind(games)
    if verbose and fc:
        print(f"  wind forecasts for {len(fc)} upcoming outdoor games")
    if log:
        log_forecasts(fc)
    games = games.copy()
    games["wind_source"] = games["wind"].notna().map({True: "recorded", False: ""})
    blank = games["wind"].isna() & games["game_id"].isin(fc)
    games.loc[blank, "wind"] = games.loc[blank, "game_id"].map(fc)
    games.loc[blank, "wind_source"] = "forecast"
    return games
