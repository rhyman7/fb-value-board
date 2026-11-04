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
(Nov 2026). With --targets it pulls GET /plays/stats by season and week instead, and keeps per
(game, athlete) counts of targets, receptions, incompletions, completions, pass attempts and
interceptions (targets_{year}.csv) with a count of every statType name seen (types_{year}.csv):
the coverage check for candidate 3 of the college-props memo (target share for receivers). A failed call prints one line and the season's other files are still written; an
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


TARGET_TYPES = {"target": "targets", "reception": "receptions", "incompletion": "incompletions", "completion": "completions",
                "pass attempt": "pass_att", "interception": "ints"}
CONFERENCES = ("SEC", "B1G", "ACC", "B12", "AAC", "MWC", "SBC", "MAC", "CUSA", "PAC", "Ind")


def pull_targets(year: int, budget: list, weeks=range(1, 17)) -> None:
    """targets_{year}.csv: per (game, athlete) counts of the play-stat types that make up a target
    share (GET /plays/stats by year and week; by conference where a whole week is refused),
    plus types_{year}.csv, how many rows each statType name had, so the coverage can be read."""
    path = DIR / f"targets_{year}.csv"
    rows, types = {}, {}
    def take(items):
        for it in items:
            st = str(it.get("statType") or "").strip()
            types[st] = types.get(st, 0) + 1
            col = next((c for k, c in TARGET_TYPES.items() if k in st.lower()), None)
            if col is None:
                continue
            key = (it.get("gameId"), it.get("athleteId"))
            r = rows.setdefault(key, {"gameId": it.get("gameId"), "week": it.get("week"), "team": it.get("team"), "opponent": it.get("opponent"),
                                      "athleteId": it.get("athleteId"), "athleteName": it.get("athleteName"),
                                      **{c: 0 for c in TARGET_TYPES.values()}})
            r[col] += 1
    for wk in weeks:
        if budget[0] <= 0:
            print(f"  targets {year}: call budget spent at week {wk}")
            break
        try:
            budget[0] -= 1
            take(get("/plays/stats", year=year, week=wk, seasonType="regular"))
        except Exception as exc:
            print(f"  targets {year} week {wk}: whole week failed ({type(exc).__name__}: {str(exc)[:60]}); by conference")
            for conf in CONFERENCES:
                if budget[0] <= 0:
                    break
                try:
                    budget[0] -= 1
                    take(get("/plays/stats", year=year, week=wk, seasonType="regular", conference=conf))
                except Exception as exc2:
                    print(f"    {conf}: {type(exc2).__name__}")
    pd.DataFrame(list(rows.values())).to_csv(path, index=False)
    pd.DataFrame([{"statType": k, "rows": v} for k, v in sorted(types.items())]).to_csv(DIR / f"types_{year}.csv", index=False)
    print(f"  targets {year}: {len(rows)} player-games; types: {dict(sorted(types.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=C.FIRST_SEASON)
    ap.add_argument("--to", dest="end", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="rewrite files that already exist")
    ap.add_argument("--targets", action="store_true", help="pull per-play player stats (targets etc.) instead of the background files")
    ap.add_argument("--max-calls", type=int, default=120, help="most CFBD calls a --targets run may make")
    a = ap.parse_args()
    from .data import current_season
    cur = current_season()
    end = a.end or cur
    budget = [a.max_calls]
    for year in range(a.start, end + 1):
        print(f"{year}:")
        if a.targets:
            DIR.mkdir(parents=True, exist_ok=True)
            if (DIR / f"targets_{year}.csv").exists() and not (a.refresh or year == cur):
                continue
            pull_targets(year, budget)
        else:
            pull_season(year, a.refresh, cur)


if __name__ == "__main__":
    main()
