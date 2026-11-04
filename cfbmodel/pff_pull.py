"""PFF Premium Stats Pro: per-week college receiving and rushing leaderboards, via the PFF
Developer API, saved under data/cfb/pff/<season>/w<week>_<table>.csv.

    python -m cfbmodel.pff_pull --from 2017 --to 2025 [--weeks 0-16] [--dry-run]

Run by the **PFF pull** Action (pff-pull.yml) with the `PFF_API_KEY` secret (a PFF Pro key,
`ak_live_...`, sent as `Authorization: Bearer`). Standard library only.

The API (developer.pff.com): base https://api.pff.com; the league-wide leaderboards are the
`facet-receiving-summary` and `facet-rushing-summary` operations, one row per player for a league,
season and week (`league=ncaa`, weeks 0-16 regular season, 17 conference championships, 18 bowls).
Their exact paths come from PFF's public OpenAPI file at run time (looked up by operationId), so a
rename on their side changes nothing here. Budget: 100 reads a minute per account; every response
carries x-ratelimit-remaining / x-ratelimit-reset and this script sleeps on them, and on a 429's
Retry-After. A full 2017-2025 pull is about 300 reads, four minutes.

Each file has the API's own fields (player, player_id, position, team_name, routes, targets,
receptions, yards, attempts, run_plays, ...) plus `season` and `week` columns. The same tables
exported by hand from the website (Nov 3, 2026) carry no season/week columns; scripts/pff_identify.py
stamps those by matching box scores, and the hand and API files should agree row for row.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config as C

ENV = "PFF_API_KEY"
BASE = "https://api.pff.com"
SPEC = "https://developer.pff.com/openapi.json"
DIR = C.DATA / "pff"
TABLES = {"receiving": "facet-receiving-summary", "rushing": "facet-rushing-summary"}
LEAGUE = "ncaa"


def _req(url: str, key: str | None = None, data: bytes | None = None):
    headers = {"Accept": "application/json", "User-Agent": "cfb-model pff-pull"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return urllib.request.Request(url, headers=headers, data=data)


def get(url: str, key: str | None = None, tries: int = 4):
    """GET with the rate-limit headers respected; returns (json, headers)."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(_req(url, key), timeout=120) as r:
                body = json.loads(r.read())
                rem = r.headers.get("x-ratelimit-remaining")
                reset = r.headers.get("x-ratelimit-reset")
                if rem is not None and reset is not None and int(rem) <= 2:
                    wait = max(0.0, float(reset) - time.time()) + 1.0
                    print(f"  budget low ({rem} left); sleeping {wait:.0f}s", flush=True)
                    time.sleep(min(wait, 90))
                return body, dict(r.headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < tries - 1:
                wait = float(exc.headers.get("Retry-After") or 15)
                print(f"  429; sleeping {wait:.0f}s", flush=True)
                time.sleep(min(wait, 120))
                continue
            detail = exc.read()[:300].decode(errors="replace") if hasattr(exc, "read") else ""
            raise RuntimeError(f"{exc.code} {url}: {detail}") from exc
    raise RuntimeError(f"gave up: {url}")


def resolve_paths() -> dict[str, tuple[str, list[str]]]:
    """operationId -> (path, query parameter names) from PFF's OpenAPI file."""
    spec, _ = get(SPEC)
    out = {}
    for path, ops in (spec.get("paths") or {}).items():
        for method, op in (ops or {}).items():
            if not isinstance(op, dict):
                continue
            oid = op.get("operationId")
            if oid in TABLES.values():
                params = [p.get("name") for p in op.get("parameters") or [] if p.get("in") == "query"]
                out[oid] = (path, params)
    missing = [v for v in TABLES.values() if v not in out]
    if missing:
        raise RuntimeError(f"operations not found in {SPEC}: {missing}")
    return out


def rows_of(body) -> list[dict]:
    """The leaderboard rows: the first list of dicts under the top level (e.g. receiving_summary)."""
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if isinstance(body, dict):
        for k, v in body.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        for k, v in body.items():
            if isinstance(v, dict):
                r = rows_of(v)
                if r:
                    return r
    return []


def flat(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in flat(v).items():
                out[f"{k}_{k2}"] = v2
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def pull(season: int, week: int, table: str, paths: dict, key: str) -> list[dict]:
    path, params = paths[TABLES[table]]
    q = {"league": LEAGUE, "season": season, "week": week}
    q = {k: v for k, v in q.items() if k in params or not params}
    body, _ = get(f"{BASE}{path}?{urllib.parse.urlencode(q)}", key)
    rows = [flat(r) for r in rows_of(body)]
    for r in rows:
        r["season"], r["week"] = season, week
    return rows


def write(rows: list[dict], season: int, week: int, table: str) -> Path:
    d = DIR / str(season)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"w{week:02d}_{table}.csv"
    cols = ["season", "week"] + sorted({k for r in rows for k in r} - {"season", "week"})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def parse_weeks(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2017)
    ap.add_argument("--to", dest="end", type=int, default=None)
    ap.add_argument("--weeks", default="0-16")
    ap.add_argument("--refresh", action="store_true", help="rewrite files that already exist")
    ap.add_argument("--dry-run", action="store_true", help="pull one week, print the row count and fields, write nothing")
    a = ap.parse_args()
    key = os.environ.get(ENV)
    if not key:
        sys.exit(f"{ENV} is not set")
    who, _ = get(f"{BASE}/v1/auth/whoami", key)
    print(f"whoami: credential={who.get('credential')} entitled={who.get('entitled')}")
    if not who.get("entitled"):
        sys.exit("key is not entitled (PFF Pro required)")
    paths = resolve_paths()
    for t, oid in TABLES.items():
        print(f"{t}: {oid} -> {paths[oid][0]} params {paths[oid][1]}")
    import datetime as dt
    today = dt.date.today()
    cur = today.year if today.month >= 8 else today.year - 1
    end = a.end or cur
    weeks = parse_weeks(a.weeks)
    if a.dry_run:
        s, w = end, weeks[0]
        for t in TABLES:
            rows = pull(s, w, t, paths, key)
            print(f"dry run {t} {s} week {w}: {len(rows)} rows; fields: {sorted(rows[0]) if rows else '-'}")
            for r in rows[:3]:
                print("  ", {k: r[k] for k in list(r)[:14]})
        return
    n_calls = 0
    for season in range(a.start, end + 1):
        for w in weeks:
            for t in TABLES:
                path = DIR / str(season) / f"w{w:02d}_{t}.csv"
                if path.exists() and not (a.refresh or season == cur):
                    continue
                try:
                    rows = pull(season, w, t, paths, key)
                    n_calls += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  {t} {season} w{w}: failed ({str(exc)[:120]})")
                    continue
                if rows:
                    write(rows, season, w, t)
                    print(f"  {t} {season} w{w}: {len(rows)} rows")
                else:
                    print(f"  {t} {season} w{w}: no rows")
    print(f"done: {n_calls} reads")


if __name__ == "__main__":
    main()
