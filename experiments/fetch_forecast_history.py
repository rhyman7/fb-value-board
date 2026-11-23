"""Fetch the rain and snow FORECAST for every played outdoor NFL home game since 2022 from
Open-Meteo's forecast archives, for experiments/rain_forecast_backtest.py.

    python experiments/fetch_forecast_history.py              # everything not yet cached
    python experiments/fetch_forecast_history.py --limit 4    # a few calls, to check it works

Two archives, one row per game in data/cache/forecast_history.csv:

- Historical Forecast API (2022 on): each hour from the newest model run at the time, so a
  forecast only a few hours old. Columns `rain_d0`, `snow_d0`: about what was known on game morning.
- Previous Runs API (January 2024 on): the forecast for the same hour as it stood 1, 2 and 3
  days earlier. Columns `rain_d1..d3`, `snow_d1..d3`. Blank before the 2024 season.

Totals are over the kickoff hour and the three after it (weather.GAME_HOURS), mm of rain and cm
of snow, the same window the board logs in data/wind_log.csv. One call per stadium per season to
each archive, 0.3 s apart, resumable: a stadium-season is written only when every call it needs
came back. Needs network to open-meteo.com; the "Forecast history" Action runs it from GitHub.
Free: no Odds API credits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import config as C, data, weather  # noqa: E402
from fetch_wind_history import PAUSE, wanted  # noqa: E402

CACHE = C.CACHE / "forecast_history.csv"
FIRST, FIRST_LEADS = 2022, 2024
LEADS = (1, 2, 3)
BASE = "?latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}&hourly={hourly}&timezone=America%2FNew_York"
URL_NOW = "https://historical-forecast-api.open-meteo.com/v1/forecast" + BASE
URL_LEAD = "https://previous-runs-api.open-meteo.com/v1/forecast" + BASE
HOURLY_NOW = "precipitation,snowfall"
HOURLY_LEAD = ",".join(f"{v}_previous_day{n}" for n in LEADS for v in ("precipitation", "snowfall"))
COLS = ["game_id", "season", "gameday", "hour", "home_team", "rain_d0", "snow_d0"] + [f"{k}_d{n}" for n in LEADS for k in ("rain", "snow")] + ["fetched"]


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["hourly"]


def fetch_now(lat, lon, start, end) -> dict:
    return _get(URL_NOW.format(lat=lat, lon=lon, d0=start, d1=end, hourly=HOURLY_NOW))


def fetch_lead(lat, lon, start, end) -> dict:
    return _get(URL_LEAD.format(lat=lat, lon=lon, d0=start, d1=end, hourly=HOURLY_LEAD))


def _window(hourly: dict, key: str, i: int):
    vals = [x for x in (hourly.get(key) or [])[i:i + weather.GAME_HOURS] if x is not None]
    return round(sum(vals), 2) if vals else None


def rows_for(day_games: pd.DataFrame, now: dict, lead: dict | None, fetched: str) -> list[dict]:
    times, ltimes = now.get("time") or [], (lead or {}).get("time") or []
    out = []
    for g in day_games.itertuples(index=False):
        key = f"{g.gameday}T{int(g.hour):02d}:00"
        if key not in times:
            continue
        i = times.index(key)
        row = {"game_id": g.game_id, "season": int(g.season), "gameday": g.gameday, "hour": int(g.hour), "home_team": g.home_team,
               "rain_d0": _window(now, "precipitation", i), "snow_d0": _window(now, "snowfall", i), "fetched": fetched}
        j = ltimes.index(key) if key in ltimes else None
        for n in LEADS:
            row[f"rain_d{n}"] = None if j is None else _window(lead, f"precipitation_previous_day{n}", j)
            row[f"snow_d{n}"] = None if j is None else _window(lead, f"snowfall_previous_day{n}", j)
        out.append(row)
    return out


def run(games, path=CACHE, now=fetch_now, lead=fetch_lead, limit=None, pause=PAUSE, verbose=True) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    todo = wanted(games)
    todo = todo[(todo.season >= FIRST) & ~todo.game_id.isin(set(cache.game_id))]
    groups = list(todo.groupby(["lat", "lon", "season"], sort=True))
    if verbose:
        print(f"  {len(cache)} games cached, {len(todo)} to fetch in {len(groups)} stadium-seasons")
    calls = ok = failed = 0
    for (lat, lon, season), sg in groups:
        if limit is not None and calls >= limit:
            break
        try:
            calls += 1
            h0 = now(lat, lon, sg.gameday.min(), sg.gameday.max())
            h1 = None
            if season >= FIRST_LEADS:
                time.sleep(pause)
                calls += 1
                h1 = lead(lat, lon, sg.gameday.min(), sg.gameday.max())
            rows = pd.DataFrame(rows_for(sg, h0, h1, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")), columns=COLS)
            rows.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
            ok += 1
        except Exception as exc:
            failed += 1
            if verbose:
                print(f"  {season} {sg.home_team.iloc[0]}: {exc}")
        time.sleep(pause)
    if verbose:
        print(f"  {calls} calls, {ok} stadium-seasons ok, {failed} failed")
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    run(data.load_games(refresh=False), limit=a.limit)


if __name__ == "__main__":
    main()
