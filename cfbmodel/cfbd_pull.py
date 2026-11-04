"""Player-background data from CollegeFootballData.com (CFBD), saved under data/cfb/cfbd/.

    python -m cfbmodel.cfbd_pull --from 2014 --to 2026

Run by the **CFBD player data** Action (cfbd-pull.yml) with the `CFBD_API_KEY` secret; about five
calls a season. Four files a season, each a plain CSV of the endpoint's rows:

  recruits_{year}.csv   GET /recruiting/players?year=  for each classification (HighSchool, JUCO,
                        PrepSchool): id, athleteId (ESPN's id where known), ranking, name, school,
                        committedTo, position, stars, rating, recruitType, year
  portal_{year}.csv     GET /player/portal?year=  (2021 on): firstName, lastName, position, origin,
                        destination, transferDate, rating, stars, eligibility. No player id.
  returning_{year}.csv  GET /player/returning?year=  team-level returning usage and PPA shares
  roster_{year}.csv     GET /roster?year=  every roster: id (ESPN's), firstName, lastName, team,
                        position, year (class), jersey, height, weight, recruitIds

These are inputs to the early-season role prior tested in experiments/cfb_role_prior_test.py
(Nov 2026). A failed call prints one line and the season's other files are still written; an
existing file is rewritten only when `--refresh` is given or the season is the current one.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import config as C

ENV = "CFBD_API_KEY"
BASE = "https://api.collegefootballdata.com"
DIR = C.DATA / "cfbd"
RECRUIT_CLASSES = ("HighSchool", "JUCO", "PrepSchool")
PORTAL_FROM = 2021


def get(path: str, **params) -> list:
    key = os.environ.get(ENV)
    if not key:
        raise RuntimeError(f"{ENV} is not set")
    url = f"{BASE}{path}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "cfb-model", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(5)
                continue
            raise
    return []


def flat(rows: list) -> pd.DataFrame:
    """Nested dicts (hometownInfo) and lists (recruitIds) become JSON strings."""
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (dict, list))).any():
            df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    return df


def pull_season(year: int, refresh: bool, current: int) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    want = {"recruits": lambda: [r for cl in RECRUIT_CLASSES for r in get("/recruiting/players", year=year, classification=cl)],
            "returning": lambda: get("/player/returning", year=year),
            "roster": lambda: get("/roster", year=year)}
    if year >= PORTAL_FROM:
        want["portal"] = lambda: get("/player/portal", year=year)
    for name, fn in want.items():
        path = DIR / f"{name}_{year}.csv"
        if path.exists() and not (refresh or year == current):
            continue
        try:
            df = flat(fn())
        except Exception as exc:
            print(f"  {name} {year}: failed ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        df.to_csv(path, index=False)
        print(f"  {name} {year}: {len(df)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=C.FIRST_SEASON)
    ap.add_argument("--to", dest="end", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="rewrite files that already exist")
    a = ap.parse_args()
    from .data import current_season
    cur = current_season()
    end = a.end or cur
    for year in range(a.start, end + 1):
        print(f"{year}:")
        pull_season(year, a.refresh, cur)


if __name__ == "__main__":
    main()
