"""Fetch the wind at kickoff for every played outdoor NFL home game since 2014 from
Open-Meteo's history archive, for experiments/wind_direction_test.py.

    python experiments/fetch_wind_history.py              # everything not yet cached
    python experiments/fetch_wind_history.py --limit 50   # a few calls, to check it works

One request per stadium per game day (the archive returns the whole day, so a doubleheader
at one stadium is one call) with 0.3 s between calls; a call that fails is printed and
skipped, and the next run tries it again. Results go to data/cache/wind_history.csv, one
row per game: the archive's 10 m wind speed, gusts and the direction the wind blew from at
the kickoff hour (Eastern), the field's bearing (nflmodel/stadiums.py) and the along-field
and crosswind parts. The run is resumable: games already in the file are not fetched again.
Needs network to archive-api.open-meteo.com; the "Wind history" Action runs it from GitHub.
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
from nflmodel import config as C, data, stadiums  # noqa: E402
from nflmodel.weather import at_hour  # noqa: E402

CACHE = C.CACHE / "wind_history.csv"
FIRST = 2014
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d}&end_date={d}"
       "&hourly=wind_speed_10m,wind_gusts_10m,wind_direction_10m&wind_speed_unit=mph&timezone=America%2FNew_York")
COLS = ["game_id", "season", "gameday", "hour", "home_team", "stadium", "lat", "lon", "bearing", "bearing_conf",
        "wind_mph", "gust_mph", "dir_deg", "along_mph", "cross_mph", "fetched"]
PAUSE = 0.3


def fetch_day(lat: float, lon: float, day: str) -> dict:
    """The archive's hourly block for one place and day (24 hours, Eastern)."""
    req = urllib.request.Request(URL.format(lat=lat, lon=lon, d=day), headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["hourly"]


def wanted(games: pd.DataFrame) -> pd.DataFrame:
    """Played outdoor home games since FIRST with a known stadium location, with the kickoff hour."""
    g = games[(games.season >= FIRST) & games.played & (games.neutral == 0) & (games.indoor == 0)].copy()
    sites = [stadiums.site(t, s if isinstance(s, str) else None) for t, s in zip(g.home_team, g.stadium)]
    g["lat"] = [s[0] if s else None for s in sites]
    g["lon"] = [s[1] if s else None for s in sites]
    g["bearing"] = [s[2] if s else None for s in sites]
    g["bearing_conf"] = [s[3] if s else None for s in sites]
    g["hour"] = g["gametime"].fillna("13:00").astype(str).str[:2].astype(int).clip(0, 23)
    g["gameday"] = pd.to_datetime(g["gameday"]).dt.strftime("%Y-%m-%d")
    missing = g[g.lat.isna()]
    if len(missing):
        print(f"  no location for {len(missing)} games: {sorted(missing.home_team.unique())}")
    return g[g.lat.notna()]


def rows_for(day_games: pd.DataFrame, hourly: dict, fetched: str) -> list[dict]:
    """One cache row per game of a stadium-day from the archive's hourly block."""
    out = []
    for g in day_games.itertuples(index=False):
        w = at_hour(hourly, int(g.hour))
        along = cross = None
        if w["dir"] is not None and pd.notna(g.bearing):
            along, cross = stadiums.components(w["mph"], w["dir"], float(g.bearing))
        out.append({"game_id": g.game_id, "season": int(g.season), "gameday": g.gameday, "hour": int(g.hour),
                    "home_team": g.home_team, "stadium": g.stadium if isinstance(g.stadium, str) else "",
                    "lat": g.lat, "lon": g.lon, "bearing": None if pd.isna(g.bearing) else int(g.bearing), "bearing_conf": g.bearing_conf,
                    "wind_mph": round(w["mph"], 1), "gust_mph": None if w["gust"] is None else round(w["gust"], 1),
                    "dir_deg": None if w["dir"] is None else int(round(w["dir"])),
                    "along_mph": None if along is None else round(along, 1), "cross_mph": None if cross is None else round(cross, 1),
                    "fetched": fetched})
    return out


def load_cache(path=CACHE) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)


def run(games: pd.DataFrame, path=CACHE, fetch=fetch_day, limit: int | None = None, pause: float = PAUSE, verbose: bool = True) -> pd.DataFrame:
    """Fetch what the cache lacks, append as it goes, and return the whole cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(path)
    done = set(cache.game_id)
    todo = wanted(games)
    todo = todo[~todo.game_id.isin(done)]
    days = list(todo.groupby(["lat", "lon", "gameday"], sort=True))
    if verbose:
        print(f"  {len(cache)} games cached, {len(todo)} to fetch in {len(days)} stadium-days")
    calls = ok = failed = 0
    for (lat, lon, day), dg in days:
        if limit is not None and calls >= limit:
            break
        calls += 1
        try:
            hourly = fetch(lat, lon, day)
            rows = pd.DataFrame(rows_for(dg, hourly, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")), columns=COLS)
            rows.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
            cache = pd.concat([cache, rows], ignore_index=True)
            ok += 1
        except Exception as exc:
            failed += 1
            if verbose:
                print(f"  {day} {dg.home_team.iloc[0]}: {exc}")
        time.sleep(pause)
    if verbose:
        print(f"  {calls} calls, {ok} ok, {failed} failed; {len(cache)} games cached")
    return load_cache(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="most archive calls to make this run")
    ap.add_argument("--cache", default=str(CACHE))
    a = ap.parse_args(argv)
    games = data.load_games(refresh=False)
    run(games, a.cache, limit=a.limit)


if __name__ == "__main__":
    main()
