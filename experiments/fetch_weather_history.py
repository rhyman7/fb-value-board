"""Fetch rain, snow and temperature during every played outdoor NFL home game since 2014 from
Open-Meteo's history archive, for experiments/precip_test.py.

    python experiments/fetch_weather_history.py              # everything not yet cached
    python experiments/fetch_weather_history.py --limit 20   # a few calls, to check it works

Same calls as fetch_wind_history.py (one per stadium per season, 0.3 s apart, resumable), with
the hourly precipitation (mm), snowfall (cm) and temperature (°F). One row per game in
data/cache/weather_history.csv: totals over the kickoff hour and the three after it (about a
game's length) and the temperature at kickoff. Needs network to archive-api.open-meteo.com;
the "Weather history" Action runs it from GitHub. Free: no Odds API credits.
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
from nflmodel import config as C, data  # noqa: E402
from fetch_wind_history import PAUSE, wanted  # noqa: E402

CACHE = C.CACHE / "weather_history.csv"
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}"
       "&hourly=precipitation,snowfall,temperature_2m&temperature_unit=fahrenheit&timezone=America%2FNew_York")
COLS = ["game_id", "season", "gameday", "hour", "home_team", "precip_mm", "snow_cm", "temp_f", "fetched"]
HOURS = 4


def fetch_range(lat, lon, start, end) -> dict:
    req = urllib.request.Request(URL.format(lat=lat, lon=lon, d0=start, d1=end), headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["hourly"]


def rows_for(day_games: pd.DataFrame, hourly: dict, fetched: str) -> list[dict]:
    times = hourly.get("time") or []
    out = []
    for g in day_games.itertuples(index=False):
        key = f"{g.gameday}T{int(g.hour):02d}:00"
        if key not in times:
            continue
        i = times.index(key)
        cut = lambda k: [x for x in (hourly.get(k) or [])[i:i + HOURS] if x is not None]
        p, s, t = cut("precipitation"), cut("snowfall"), (hourly.get("temperature_2m") or [None])[i] if hourly.get("temperature_2m") else None
        out.append({"game_id": g.game_id, "season": int(g.season), "gameday": g.gameday, "hour": int(g.hour), "home_team": g.home_team,
                    "precip_mm": round(sum(p), 2) if p else None, "snow_cm": round(sum(s), 2) if s else None,
                    "temp_f": None if t is None else round(float(t), 1), "fetched": fetched})
    return out


def run(games, path=CACHE, fetch=fetch_range, limit=None, pause=PAUSE, verbose=True) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    todo = wanted(games)
    todo = todo[~todo.game_id.isin(set(cache.game_id))]
    groups = list(todo.groupby(["lat", "lon", "season"], sort=True))
    if verbose:
        print(f"  {len(cache)} games cached, {len(todo)} to fetch in {len(groups)} stadium-seasons")
    calls = ok = failed = 0
    for (lat, lon, season), sg in groups:
        if limit is not None and calls >= limit:
            break
        calls += 1
        try:
            hourly = fetch(lat, lon, sg.gameday.min(), sg.gameday.max())
            rows = pd.DataFrame(rows_for(sg, hourly, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")), columns=COLS)
            rows.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
            ok += 1
        except Exception as exc:
            failed += 1
            if verbose:
                print(f"  {season} {sg.home_team.iloc[0]}: {exc}")
        time.sleep(pause)
    if verbose:
        print(f"  {calls} calls, {ok} ok, {failed} failed")
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    run(data.load_games(refresh=False), limit=a.limit)


if __name__ == "__main__":
    main()
