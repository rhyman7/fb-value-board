"""Wind at kickoff for upcoming outdoor college games, from the free Open-Meteo forecast.

Played games use the wind recorded in the cfbfastR lines file. Upcoming outdoor games
within a week get the forecast for the kickoff hour at the home stadium (10 m wind, mph).
Neutral-site games are skipped (the venue's location is not in the data). Every forecast
is logged to data/cfb/wind_log.csv so the wind feature can later be graded on forecasts.
A failed request leaves the wind blank, which the model treats as a typical day.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config as C

LOG = C.DATA / "wind_log.csv"
URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly=wind_speed_10m&wind_speed_unit=mph&timezone=UTC&start_date={d}&end_date={d}")
MAX_DAYS = 7


def forecast_wind(games: pd.DataFrame, today=None) -> dict:
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
            req = urllib.request.Request(URL.format(lat=round(lat, 3), lon=round(lon, 3), d=ko.date().isoformat()), headers={"User-Agent": "cfb-model"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                hourly = json.loads(resp.read())["hourly"]["wind_speed_10m"]
            out[r.game_id] = float(hourly[min(max(ko.hour, 0), 23)])
        except Exception as exc:
            print(f"  no wind forecast for {r.game_id} ({exc})")
    return out


def log_forecasts(fc: dict, path=LOG) -> None:
    if not fc:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows = pd.DataFrame({"game_id": list(fc), "fetched": now, "wind_mph": [round(v, 1) for v in fc.values()]})
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def fill_wind(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Forecast wind for upcoming games; record where each game's wind came from."""
    g = games.copy()
    g["wind_source"] = np.where(g["played"] & pd.to_numeric(g["wind"], errors="coerce").notna(), "recorded", "")
    try:
        fc = forecast_wind(g)
    except Exception as exc:
        print(f"  wind forecasts skipped ({exc})")
        fc = {}
    if fc:
        m = g["game_id"].isin(fc)
        g.loc[m, "wind"] = g.loc[m, "game_id"].map(fc)
        g.loc[m, "wind_source"] = "forecast"
        log_forecasts(fc)
    if verbose:
        print(f"  wind forecasts: {len(fc)} games")
    return g
