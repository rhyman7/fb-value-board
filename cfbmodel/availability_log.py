"""College availability log: ESPN's injury designations for every FBS team, kept with the time
each change was first seen.

    python -m cfbmodel.availability_log [--season 2026] [--dry-run]

Run by the **College availability** Action (cfb-availability.yml); GitHub's runners can reach
ESPN, the Claude workspace can't. Since 2025 all four power conferences publish availability
reports (Big Ten 2023, SEC 2024, ACC and Big 12 2025; daily from three days out and a final one
90 minutes to two hours before kickoff), and ESPN's injury lists carry them. The board's Covers.com
designations (designations.py) are untouched by this module (and availability.py, the box-score version, is separate): this is the history that did not exist,
so that the kickoff-time availability can be tested against the 2026 live prop log before it is
allowed to change a projection.

Source: ESPN's core API, per team, the same feed rhyman7/nfl's Fetch injuries Action reads:
    sports.core.api.espn.com/v2/sports/football/leagues/college-football/teams/{id}/injuries
each item a $ref to an injury record (status, date, athlete $ref, details.type/detail/side/
returnDate, shortComment/longComment). Team ids are the board's (data/cfb/cache/teams_{season}.csv).

Files, under data/cfb/availability/{season}/:
    latest.json     every current entry: {"seen": ..., "entries": [{team_id, team, player_id, name,
                    pos, status, date, type, detail, comment, first_seen}]}
    log.csv         one row per change, appended: seen (UTC), team_id, team, player_id, name, pos,
                    status, date, type, detail, comment, change (new | status | details | cleared).
                    "cleared" means the player left the team's list. A run that finds no change
                    appends nothing, so the log stays small (a few hundred rows a week).
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config as C

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
DIR = C.DATA / "availability"
FIELDS = ["seen", "team_id", "team", "player_id", "name", "pos", "status", "date", "type", "detail", "comment", "change"]
MIN_TEAMS = 100          # fewer teams answering than this: ESPN is having a bad day; write nothing


def get(url: str, tries: int = 4):
    url = url.replace("http://", "https://", 1)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cfb-model availability)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            if i == tries - 1:
                raise RuntimeError(f"{url}: {exc}") from exc
            time.sleep(1.5 * (i + 1))


def teams(season: int) -> dict[int, str]:
    path = C.CACHE / f"teams_{season}.csv"
    with path.open() as f:
        return {int(r["team_id"]): r["school"] for r in csv.DictReader(f)}


def team_refs(tid: int) -> list[str] | None:
    refs, page = [], 1
    try:
        while True:
            d = get(f"{CORE}/teams/{tid}/injuries?limit=200&page={page}")
            refs += [it["$ref"] for it in d.get("items", []) if "$ref" in it]
            if page >= int(d.get("pageCount") or 1):
                return refs
            page += 1
    except RuntimeError as exc:
        print(f"  team {tid}: {exc}", file=sys.stderr)
        return None


def athlete_id(ref: str) -> int | None:
    try:
        return int(ref.split("?")[0].rstrip("/").rsplit("/", 1)[-1])
    except ValueError:
        return None


def fetch(season: int) -> tuple[list[dict], int]:
    """Current entries for every team; (entries, teams that answered)."""
    tm = teams(season)
    pool = ThreadPoolExecutor(max_workers=16)
    refs = dict(zip(tm, pool.map(team_refs, tm)))
    answered = sum(1 for v in refs.values() if v is not None)
    flat = [(tid, r) for tid, rs in refs.items() if rs for r in rs]
    items = list(pool.map(lambda x: get(x[1]), flat))
    latest: dict[tuple, dict] = {}
    for (tid, _), it in zip(flat, items):
        aref = (it.get("athlete") or {}).get("$ref")
        pid = athlete_id(aref) if aref else None
        if pid is None:
            continue
        k = (tid, pid)
        if k not in latest or str(it.get("date") or "") > str(latest[k].get("date") or ""):
            latest[k] = it
    ath = dict(zip(latest, pool.map(lambda k: get((latest[k]["athlete"]["$ref"])), latest)))
    out = []
    for (tid, pid), it in latest.items():
        a = ath.get((tid, pid)) or {}
        det = it.get("details") or {}
        status = str(it.get("status") or (it.get("type") or {}).get("description") or "").strip()
        out.append({"team_id": tid, "team": tm[tid], "player_id": pid, "name": a.get("displayName", ""),
                    "pos": ((a.get("position") or {}).get("abbreviation") or ""), "status": status,
                    "date": str(it.get("date") or "")[:10], "type": str(det.get("type") or ""),
                    "detail": str(det.get("detail") or ""), "comment": str(it.get("shortComment") or "")[:200]})
    out.sort(key=lambda e: (e["team"], e["name"]))
    return out, answered


def diff(old: list[dict], new: list[dict], seen: str) -> list[dict]:
    key = lambda e: (e["team_id"], e["player_id"])
    o = {key(e): e for e in old}
    n = {key(e): e for e in new}
    rows = []
    for k, e in n.items():
        if k not in o:
            rows.append({**e, "seen": seen, "change": "new"})
        elif e["status"] != o[k]["status"]:
            rows.append({**e, "seen": seen, "change": "status"})
        elif (e["date"], e["type"], e["detail"], e["comment"]) != (o[k]["date"], o[k]["type"], o[k]["detail"], o[k]["comment"]):
            rows.append({**e, "seen": seen, "change": "details"})
    for k, e in o.items():
        if k not in n:
            rows.append({**e, "seen": seen, "change": "cleared"})
    return [{f: r.get(f, "") for f in FIELDS} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    today = dt.date.today()        # data.current_season, without importing pandas (standard library only here)
    season = a.season or (today.year if today.month >= 8 else today.year - 1)
    d = DIR / str(season)
    seen = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    new, answered = fetch(season)
    if answered < MIN_TEAMS:
        sys.exit(f"only {answered} teams answered ESPN; nothing written")
    latest_path, log_path = d / "latest.json", d / "log.csv"
    old = json.loads(latest_path.read_text())["entries"] if latest_path.exists() else []
    first = {(e["team_id"], e["player_id"]): e.get("first_seen", "") for e in old}
    for e in new:
        e["first_seen"] = first.get((e["team_id"], e["player_id"])) or seen
    rows = diff(old, new, seen)
    print(f"{seen}: {answered} teams, {len(new)} entries, {len(rows)} changes")
    if a.dry_run:
        for r in rows[:20]:
            print("  ", r)
        return
    d.mkdir(parents=True, exist_ok=True)
    if rows or not latest_path.exists():
        write_header = not log_path.exists()
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(rows)
        latest_path.write_text(json.dumps({"seen": seen, "source": "ESPN injuries (conference availability reports)",
                                           "teams": answered, "entries": new}, indent=0))


if __name__ == "__main__":
    main()
