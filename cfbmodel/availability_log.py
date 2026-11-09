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

Source: ESPN's league-wide injuries feed (site.api.espn.com/.../college-football/injuries, the data
behind espn.com/college-football/injuries): one group per team, each entry with status, date,
athlete (id, displayName, position), details (type, detail, side, returnDate) and shortComment.
The per-team core API (.../teams/{id}/injuries) is the fallback; for college it held five stale
entries from 2020-2022 on the Nov 3, 2026 dry run. Team ids are the board's
(data/cfb/cache/teams_{season}.csv).

The conferences' own reports (`--conferences`, conf_log.csv and conf_archive.json; see CONFERENCES
below) are the primary source from Nov 3, 2026. ESPN's college feeds turned out empty on the Nov 3, 2026 dry runs (per-team: five 2020-2022
entries; league-wide: three teams, no entries), so until the conference pages themselves are read,
the one source with history is the Covers.com designations file the daily task writes
(`--covers`: covers_log.csv, one row per change in docs/cfb/designations.json, with the file's
"checked" date and the time the change was seen here).

`--conferences` also writes **docs/cfb/availability.json**, the current state the board reads
(designations.py): each power-conference team's newest report while its game is still ahead,
in the shape {"source", "fetched", "today", "teams": {school: {"conf", "abbr", "report" (Initial |
Update | Game Day), "published", "posted" (the conference's own time zone), "game_date",
"game_time", "opponent", "players": [{"name", "pos", "number", "status", "season_out"}]}}}.
"Exempt" rows (deep-roster players exempt from reporting) are left out of it.

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
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config as C

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
LEAGUE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries"   # the feed behind espn.com/college-football/injuries
DIR = C.DATA / "availability"
FIELDS = ["seen", "team_id", "team", "player_id", "name", "pos", "status", "date", "type", "detail", "comment", "change"]
MIN_TEAMS = 20           # the league feed lists only teams with an entry; fewer than this and ESPN is having a bad day: write nothing


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


def fetch_league(season: int) -> tuple[list[dict], int]:
    """ESPN's league-wide feed: one group per team with that team's current entries. Team ids are
    ESPN's (the board's); groups for teams outside the board's FBS list are dropped."""
    tm = teams(season)
    d = get(LEAGUE)
    out, answered = [], 0
    for grp in d.get("injuries") or []:
        try:
            tid = int(grp.get("id") or (grp.get("team") or {}).get("id"))
        except (TypeError, ValueError):
            continue
        if tid not in tm:
            continue
        answered += 1
        for it in grp.get("injuries") or []:
            a = it.get("athlete") or {}
            try:
                pid = int(a.get("id"))
            except (TypeError, ValueError):
                continue
            det = it.get("details") or {}
            status = str(it.get("status") or (it.get("type") or {}).get("description") or "").strip()
            out.append({"team_id": tid, "team": tm[tid], "player_id": pid, "name": a.get("displayName", ""),
                        "pos": ((a.get("position") or {}).get("abbreviation") or ""), "status": status,
                        "date": str(it.get("date") or "")[:10], "type": str(det.get("type") or ""),
                        "detail": str(det.get("detail") or ""), "comment": str(it.get("shortComment") or "")[:200]})
    out.sort(key=lambda e: (e["team"], e["name"]))
    return out, answered


def fetch(season: int) -> tuple[list[dict], int]:
    """Current entries for every team; (entries, teams that answered). The league feed first; the
    per-team core API (which for college carries only a few stale entries, Nov 2026 dry run) only
    if the league feed fails outright."""
    try:
        out, answered = fetch_league(season)
        print(f"  league feed: {answered} teams listed, {len(out)} entries")
        return out, answered
    except Exception as exc:  # noqa: BLE001
        print(f"  league feed failed ({exc}); falling back to the per-team feed", file=sys.stderr)
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


# ---- the conferences' own reports -------------------------------------------------------------------
# All four power conferences publish their football availability reports through one vendor app
# (app.hdintelligence.com, embedded on secsports.com/fbreports, theacc.com, big12sports.com and
# bigten.org; found by the Probe availability pages Action, Nov 3, 2026). Two public JSON calls:
#   POST /api/get-publish-public  {"sport":"Football","organization":X,"conference":X}
#        -> {report_id: {ReportType (Initial | Update | Game Day | Report Pending), publishDate,
#            postedTime, footer{date, time, location}, games: [{teamName, teamDisplayName,
#            rows: [{name "POS #N Player", status, exemptStatus}]}], statusReportOrder}}
#   POST /api/get-archive-public  same body -> {data: [{Team, Player, Number, Opponent, Week,
#            WeekNum, Initial, Update 1, Update 2, Game Day}]}  (the season to date)
HDI = "https://app.hdintelligence.com/api/"
CONFERENCES = {"SEC": "SEC", "ACC": "ACC", "B12": "Big 12", "B10": "Big Ten"}
CONF_FIELDS = ["seen", "conf", "report_id", "report_type", "publish_date", "posted_time", "game_date", "game_time", "location",
               "team", "pos", "number", "player", "status", "exempt"]
NAME_RE = re.compile(r"^\s*([A-Z/]{1,4})\s*#(\d+)\s+(.*?)\s*$")


def hdi(path: str, org: str):
    body = json.dumps({"sport": "Football", "organization": org, "conference": org}).encode()
    req = urllib.request.Request(HDI + path, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json",
                                          "Referer": f"https://app.hdintelligence.com/?source={org}&sport=Football&conf={org}&type=report",
                                          "User-Agent": "Mozilla/5.0 (cfb-model availability)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def conference_rows(org: str, payload: dict, seen: str) -> list[dict]:
    rows = []
    for rid, rep in (payload or {}).items():
        if not isinstance(rep, dict):
            continue
        foot = rep.get("footer") or {}
        for g in rep.get("games") or []:
            team = g.get("teamDisplayName") or g.get("teamName") or ""
            for r in g.get("rows") or []:
                m = NAME_RE.match(str(r.get("name") or ""))
                pos, num, player = (m.group(1), m.group(2), m.group(3)) if m else ("", "", str(r.get("name") or ""))
                rows.append({"seen": seen, "conf": CONFERENCES[org], "report_id": str(rid), "report_type": rep.get("ReportType", ""),
                             "publish_date": rep.get("publishDate", ""), "posted_time": rep.get("postedTime", ""),
                             "game_date": foot.get("date", ""), "game_time": foot.get("time", ""), "location": foot.get("location", ""),
                             "team": team, "pos": pos, "number": num, "player": player, "status": r.get("status", ""),
                             "exempt": r.get("exemptStatus", "")})
    return rows


CURRENT = C.DOCS / "availability.json"
REPORT_ORDER = {"Initial": 0, "Update": 1, "Game Day": 2}
LIVE_STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Game Time Decision", "Out (1st Half)"}


def eastern_today() -> dt.date:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001
        return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)).date()


def current_reports(payloads: dict, seen: str, today: dt.date | None = None, abbr: dict | None = None) -> dict:
    """docs/cfb/availability.json: each team's newest report whose game is on or after today."""
    today = today or eastern_today()
    if abbr is None:
        try:
            season = today.year if today.month >= 8 else today.year - 1
            with (C.CACHE / f"teams_{season}.csv").open() as f:
                abbr = {r["school"]: r["abbr"] for r in csv.DictReader(f)}
        except OSError:
            abbr = {}
    best: dict[str, tuple] = {}
    for org, pub in payloads.items():
        for rid, rep in (pub or {}).items():
            if not isinstance(rep, dict) or rep.get("ReportType") not in REPORT_ORDER:
                continue
            foot = rep.get("footer") or {}
            try:
                gd = dt.date.fromisoformat(str(foot.get("date") or "")[:10])
            except ValueError:
                continue
            if gd < today:
                continue
            games = rep.get("games") or []
            names = [g.get("teamDisplayName") or g.get("teamName") or "" for g in games]
            for g, team in zip(games, names):
                players = []
                for r in g.get("rows") or []:
                    status = str(r.get("status") or "").strip()
                    if status not in LIVE_STATUSES:
                        continue
                    m = NAME_RE.match(str(r.get("name") or ""))
                    pos, num, player = (m.group(1), m.group(2), m.group(3)) if m else ("", "", str(r.get("name") or "").strip())
                    players.append({"name": player, "pos": pos, "number": num, "status": status,
                                    "season_out": str(r.get("exemptStatus") or "") == "Out"})
                others = [n for n in names if n != team]
                key = (str(rep.get("publishDate") or ""), str(rep.get("postedTime") or ""), REPORT_ORDER[rep["ReportType"]])
                if team not in best or key > best[team][0]:
                    best[team] = (key, {"conf": CONFERENCES[org], "abbr": abbr.get(team, ""), "report": rep["ReportType"],
                                        "published": key[0][:10], "posted": key[1][:8], "game_date": gd.isoformat(),
                                        "game_time": str(foot.get("time") or "")[:8], "opponent": others[0] if len(others) == 1 else "",
                                        "players": players})
    missing = sorted(t for t, v in best.items() if not v[1]["abbr"])
    if missing:
        print(f"  current: no board abbreviation for {missing}", file=sys.stderr)
    return {"source": "Conference availability reports (SEC, ACC, Big 12, Big Ten) via app.hdintelligence.com",
            "fetched": seen, "today": today.isoformat(), "teams": {t: v[1] for t, v in sorted(best.items())}}


def conference_log(d: Path, seen: str, dry_run: bool) -> None:
    """conf_log.csv: every (report version, team, player, status) the first time it is seen, Available
    rows only for players who carried another status in an earlier version of the same report;
    conf_archive.json: the vendor's season-to-date archive per conference, rewritten when it grows."""
    state_path, log_path, arch_path = d / "conf_seen.json", d / "conf_log.csv", d / "conf_archive.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"keys": [], "listed": []}
    keys, listed = set(state["keys"]), set(state["listed"])
    new_rows = []
    archive = json.loads(arch_path.read_text()) if arch_path.exists() else {}
    payloads = {}
    for org in CONFERENCES:
        try:
            pub = hdi("get-publish-public", org)
        except Exception as exc:  # noqa: BLE001
            print(f"  {org}: reports failed ({type(exc).__name__}: {str(exc)[:80]})", file=sys.stderr)
            continue
        payloads[org] = pub
        rows = conference_rows(org, pub, seen)
        n_rep = len(pub) if isinstance(pub, dict) else 0
        kept = 0
        for r in rows:
            pk = f"{r['report_id']}|{r['team']}|{r['player']}"
            k = f"{pk}|{r['report_type']}|{r['publish_date']}|{r['posted_time']}|{r['status']}"
            if k in keys:
                continue
            if r["status"] == "Available" and pk not in listed:
                continue
            if r["status"] not in ("Available", "Exempt"):
                listed.add(pk)
            keys.add(k)
            new_rows.append(r)
            kept += 1
        try:
            arc = hdi("get-archive-public", org)
            data = arc.get("data") if isinstance(arc, dict) else None
            if isinstance(data, list) and len(data) >= len(archive.get(org, [])):
                archive[org] = data
                weeks = sorted({str(x.get("Week", ""))[:16] for x in data})
                print(f"  {org}: {n_rep} reports, {len(rows)} rows, {kept} new; archive {len(data)} rows, weeks {weeks[0] if weeks else '-'} .. {weeks[-1] if weeks else '-'}")
            else:
                print(f"  {org}: {n_rep} reports, {len(rows)} rows, {kept} new; archive unchanged")
        except Exception as exc:  # noqa: BLE001
            print(f"  {org}: {n_rep} reports, {len(rows)} rows, {kept} new; archive failed ({type(exc).__name__})", file=sys.stderr)
    current = current_reports(payloads, seen)
    print(f"  current: {len(current['teams'])} teams with a live report, "
          f"{sum(len(t['players']) for t in current['teams'].values())} designations")
    if dry_run:
        for r in new_rows[:15]:
            print("  ", {k: r[k] for k in ("conf", "report_type", "publish_date", "team", "pos", "player", "status")})
        return
    d.mkdir(parents=True, exist_ok=True)
    if payloads:
        CURRENT.parent.mkdir(parents=True, exist_ok=True)
        CURRENT.write_text(json.dumps(current, indent=1, ensure_ascii=False) + "\n")
    if new_rows:
        write_header = not log_path.exists()
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CONF_FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    state_path.write_text(json.dumps({"keys": sorted(keys), "listed": sorted(listed)}))
    if archive:
        arch_path.write_text(json.dumps(archive, separators=(",", ":")))


COVERS_FIELDS = ["seen", "checked", "team", "name", "pos", "d", "injury", "updated", "season_out", "change"]


def covers_log(d: Path, seen: str, dry_run: bool) -> None:
    """Append the changes in docs/cfb/designations.json (the Covers.com designations the daily task
    writes) to covers_log.csv, so the designations the board actually used each day are kept.
    The board file is rewritten in place and has no history of its own."""
    src = C.DOCS / "designations.json"
    if not src.exists():
        print("  no designations.json")
        return
    cur = json.loads(src.read_text())
    key = lambda e: (e.get("team"), e.get("name"))
    new = {key(e): e for e in cur.get("players", [])}
    state_path, log_path = d / "covers_latest.json", d / "covers_log.csv"
    old = {tuple(k.split("|", 1)): v for k, v in json.loads(state_path.read_text()).items()} if state_path.exists() else {}
    rows = []
    for k, e in new.items():
        o = old.get(k)
        if o is None:
            change = "new"
        elif e.get("d") != o.get("d"):
            change = "status"
        elif (e.get("injury"), e.get("updated"), e.get("season")) != (o.get("injury"), o.get("updated"), o.get("season")):
            change = "details"
        else:
            continue
        rows.append({"seen": seen, "checked": cur.get("checked", ""), "team": e.get("team"), "name": e.get("name"), "pos": e.get("pos"),
                     "d": e.get("d"), "injury": e.get("injury"), "updated": e.get("updated"), "season_out": e.get("season"), "change": change})
    for k, o in old.items():
        if k not in new:
            rows.append({"seen": seen, "checked": cur.get("checked", ""), "team": o.get("team"), "name": o.get("name"), "pos": o.get("pos"),
                         "d": o.get("d"), "injury": o.get("injury"), "updated": o.get("updated"), "season_out": o.get("season"), "change": "cleared"})
    print(f"  covers: checked {cur.get('checked')}, {len(new)} designations, {len(rows)} changes")
    if dry_run or not rows:
        return
    d.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COVERS_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    state_path.write_text(json.dumps({"|".join(str(x) for x in k): v for k, v in new.items()}, indent=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--covers", action="store_true", help="log the Covers.com designations file instead of ESPN")
    ap.add_argument("--conferences", action="store_true", help="log the SEC/ACC/Big 12/Big Ten reports from the vendor app instead of ESPN")
    a = ap.parse_args()
    today = dt.date.today()        # data.current_season, without importing pandas (standard library only here)
    season = a.season or (today.year if today.month >= 8 else today.year - 1)
    d = DIR / str(season)
    seen = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    if a.covers:
        covers_log(d, seen, a.dry_run)
        return
    if a.conferences:
        conference_log(d, seen, a.dry_run)
        return
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
