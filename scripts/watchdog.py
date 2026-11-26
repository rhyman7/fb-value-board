"""Watchdog: start a scheduled Action when GitHub's own schedule skipped it.

GitHub often drops scheduled runs. Two of this repo's Actions matter too much to miss:

  Market lines (market.yml)                 the three daily snapshots, 02:05, 14:05 and 20:05 UTC.
                                            A skipped one leaves the boards pricing bets off lines
                                            10+ hours old.
  College availability (cfb-availability.yml)  the conferences' Game Day injury reports, posted a
                                            few hours before kickoff on Saturdays and weeknights.

Until Nov 25, 2026 three Claude scheduled tasks did this ("Market lines backup", "College
availability backup (Saturdays)" and "(weeknights)"). It needs no judgement, so it is plain code
now, run by the Watchdog Action (.github/workflows/watchdog.yml), which keeps itself going by
starting its own next run instead of relying on the schedule.

Rules (the same ones the tasks followed):
  Market lines   From 40 minutes to 3 hours after a daily slot: if no run created since 20 minutes
                 before the slot is queued, running or succeeded, start one plain snapshot (no
                 inputs: no derivatives, no alternates; about 6 Odds API credits). At most two
                 failed runs per slot are retried, so a broken workflow can't burn credits.
  Availability   Saturdays 9:20 am to 10 pm ET and Tuesday to Friday 3:20 to 8 pm ET, August to
                 January: if no run is queued or running and none succeeded in the last 40 minutes,
                 start one. At most two failures in that 40 minutes are retried.

    python scripts/watchdog.py            # check, start what is due, print what happened
    python scripts/watchdog.py --dry-run  # check and print only

Needs GH_TOKEN (the Action's own token, with actions: write) and GITHUB_REPOSITORY. Prints
"sleep=<seconds>" on its last line: how long until the next check is worth making.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
MARKET_SLOTS = ((2, 5), (14, 5), (20, 5))       # UTC, the "three a day" cron in market.yml
MARKET_MONTHS = {8, 9, 10, 11, 12, 1, 2}
AVAIL_MONTHS = {8, 9, 10, 11, 12, 1}
ACTIVE = {"queued", "in_progress", "waiting", "pending", "requested"}
MIN_SLEEP, MAX_SLEEP = 300, 5 * 3600            # a job may run 6 hours; stay well under it


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def market_slot(now: datetime) -> datetime | None:
    """The daily slot this check is for: the latest one 40 minutes to 3 hours ago, else None."""
    if now.month not in MARKET_MONTHS:
        return None
    for back in (0, 1):
        day = (now - timedelta(days=back)).date()
        for h, m in reversed(MARKET_SLOTS):
            slot = datetime(day.year, day.month, day.day, h, m, tzinfo=UTC)
            if timedelta(minutes=40) <= now - slot < timedelta(hours=3):
                return slot
    return None


def availability_window(now: datetime) -> bool:
    """Is this a time the conferences post Game Day reports? (The old tasks' hours.)"""
    et = now.astimezone(ET)
    if et.month not in AVAIL_MONTHS:
        return False
    mins = et.hour * 60 + et.minute
    if et.weekday() == 5:                         # Saturday
        return 9 * 60 + 20 <= mins < 22 * 60
    if et.weekday() in (1, 2, 3, 4):              # Tuesday to Friday
        return 15 * 60 + 20 <= mins < 20 * 60
    return False


def decide(runs: list[dict], since: datetime) -> tuple[bool, str]:
    """Start a run? `runs` are the workflow's recent runs; only those created at or after `since`
    count. Returns (start, reason)."""
    recent = [r for r in runs if _t(r["created_at"]) >= since]
    active = [r for r in runs if r.get("status") in ACTIVE]
    if active:
        return False, f"run {active[0]['id']} is {active[0]['status']}"
    ok = [r for r in recent if r.get("conclusion") == "success"]
    if ok:
        return False, f"run {ok[0]['id']} succeeded at {ok[0]['created_at']}"
    failed = [r for r in recent if r.get("status") == "completed"]
    if len(failed) >= 2:
        return False, "gave up: " + ", ".join(f"run {r['id']} {r.get('conclusion')}" for r in failed[:3])
    why = "GitHub's schedule did not start one"
    if failed:
        why = f"run {failed[0]['id']} ended {failed[0].get('conclusion')}"
    return True, why


def next_sleep(now: datetime, acted: bool) -> int:
    """Seconds until the next check is worth making."""
    cands = []
    if acted or availability_window(now):
        cands.append(now + timedelta(minutes=20))
    for back in (-1, 0, 1):                       # the next market check: a slot + 40 minutes
        day = (now + timedelta(days=back)).date()
        for h, m in MARKET_SLOTS:
            t = datetime(day.year, day.month, day.day, h, m, tzinfo=UTC) + timedelta(minutes=40)
            if t > now and t.month in MARKET_MONTHS:
                cands.append(t)
    probe = now                                   # the next availability window, by 20-minute steps
    for _ in range(3 * 24 * 7):
        probe += timedelta(minutes=20)
        if availability_window(probe):
            cands.append(probe)
            break
    secs = int((min(cands) - now).total_seconds()) + 60 if cands else MAX_SLEEP
    return max(MIN_SLEEP, min(MAX_SLEEP, secs))


def _api(path: str, method: str = "GET", body: dict | None = None):
    repo = os.environ["GITHUB_REPOSITORY"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def check(name: str, workflow: str, since: datetime, dry: bool) -> bool:
    runs = _api(f"actions/workflows/{workflow}/runs?per_page=15")["workflow_runs"]
    start, why = decide(runs, since)
    if not start:
        print(f"{name}: nothing to do ({why}).")
        return False
    if dry:
        print(f"{name}: would start {workflow} ({why}). Dry run, not started.")
        return False
    _api(f"actions/workflows/{workflow}/dispatches", "POST", {"ref": "main"})
    print(f"{name}: started {workflow} ({why}).")
    return True


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    now = datetime.now(UTC)
    print(f"Watchdog check at {now:%Y-%m-%d %H:%M} UTC ({now.astimezone(ET):%a %-I:%M %p} ET)")
    acted = False
    slot = market_slot(now)
    if slot:
        acted |= check(f"Market lines, {slot:%H:%M} UTC snapshot", "market.yml", slot - timedelta(minutes=20), dry)
    else:
        print("Market lines: not due (checked from 40 minutes to 3 hours after 02:05, 14:05 and 20:05 UTC).")
    if availability_window(now):
        acted |= check("College availability", "cfb-availability.yml", now - timedelta(minutes=40), dry)
    else:
        print("College availability: outside the Game Day report hours.")
    print(f"sleep={next_sleep(now, acted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
