"""Fetch outside computer predictions for college football, for the pre-set test in the Project
doc claude/plan-cfb-outside-ratings-test-2026-11-20.md (step 0: the data).

Source: The Prediction Tracker (thepredictiontracker.com). Its season archive has one file per
season with every tracked system's predicted margin for every game, recorded before kickoff,
plus the opening and midweek lines. It is the only free point-in-time source found; season-final
ratings from the college data feed would leak the future. The site says its predictions are for
research and informational use: fine for a test, on the January data-rights list before any
product uses them.

What this writes, all under data/cfb/outside/ :
  ncaa<season>.csv        the season file exactly as served (2015 on by default)
  live/<YYYY-MM-DD>.csv   this week's predictions file as served today. Saved so the archive can
                          later be checked against it: were predictions revised after the games?
  manifest.json           per file: where it came from, when, size, sha256, rows, column names

Nothing here touches the model, the boards or any scheduled task. Run by hand through the
"Outside ratings" Action (it needs the open internet). Polite: robots.txt is honoured, one
request at a time with a pause, about 15 requests in all.

    python experiments/fetch_outside_ratings.py              # 2015 on, plus the live file
    python experiments/fetch_outside_ratings.py --from 2019  # fewer seasons
    python experiments/fetch_outside_ratings.py --no-live
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE = "https://www.thepredictiontracker.com/"
ARCHIVE = BASE + "ncaaarchive.html"
LIVE_PAGES = [BASE + "predncaa.html", BASE + "predncaa.php"]
OUT = Path(__file__).resolve().parents[1] / "data" / "cfb" / "outside"
UA = "fb-value-board research fetch (github.com/rhyman7/fb-value-board)"
PAUSE = 2.0
FIRST_SEASON = 2015

_HREF = re.compile(r"""href\s*=\s*["']?([^"'\s>]+\.csv)""", re.I)
_SEASON_LINK = re.compile(r"""(20\d\d|19\d\d)\s*Season\s*:?\s*(?:<[^>]*>\s*)*?<a\b[^>]*?href\s*=\s*["']?([^"'\s>]+)""", re.I)


def parse_archive(html: str, base: str = ARCHIVE) -> dict[int, str]:
    """{season: absolute file URL} from the archive page. The page lists '<year> Season: <link>';
    if that wording changes, fall back to any .csv link with a four-digit year in its name."""
    out: dict[int, str] = {}
    for year, href in _SEASON_LINK.findall(html):
        if href.lower().endswith(".csv"):
            out.setdefault(int(year), urljoin(base, href))
    if not out:
        for href in _HREF.findall(html):
            m = re.search(r"(19|20)\d\d", href.rsplit("/", 1)[-1])
            if m:
                out.setdefault(int(m.group(0)), urljoin(base, href))
    return out


def parse_live(html: str, base: str) -> str | None:
    """The 'comma separated file' link on this week's predictions page."""
    for href in _HREF.findall(html):
        return urljoin(base, href)
    return None


def looks_like_csv(text: str) -> bool:
    """A real predictions file: a comma-separated header naming the home team, and data rows.
    Guards against saving an error page."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5 or "<html" in text[:2000].lower():
        return False
    head = lines[0].lower()
    return head.count(",") >= 5 and "home" in head


def describe(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {"rows": max(len(lines) - 1, 0),
            "columns": [c.strip().strip('"') for c in lines[0].split(",")] if lines else []}


class Fetcher:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.robots = urllib.robotparser.RobotFileParser()
        self.robots_ok = False
        self._last = 0.0

    def load_robots(self) -> None:
        try:
            r = self.s.get(BASE + "robots.txt", timeout=30)
            if r.status_code == 200:
                self.robots.parse(r.text.splitlines())
                self.robots_ok = True
            else:                       # no robots.txt: nothing is disallowed
                self.robots.parse([])
                self.robots_ok = True
        except requests.RequestException as e:
            print(f"robots.txt could not be read ({e}); stopping", file=sys.stderr)

    def get(self, url: str) -> str | None:
        if not self.robots_ok or not self.robots.can_fetch(UA, url):
            print(f"  not allowed by robots.txt: {url}")
            return None
        wait = PAUSE - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        try:
            r = self.s.get(url, timeout=60)
        except requests.RequestException as e:
            print(f"  failed: {url} ({e})")
            return None
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {url}")
            return None
        return r.content.decode("latin-1")   # one byte per character, so save() writes the same bytes back


def save(path: Path, text: str, url: str, manifest: dict, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = text.encode("latin-1")
    path.write_bytes(raw)
    manifest[key] = {"url": url, "file": str(path.relative_to(OUT)), "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     **describe(text)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="first", type=int, default=FIRST_SEASON, help="first season to fetch")
    ap.add_argument("--no-live", action="store_true", help="skip this week's live file")
    a = ap.parse_args(argv)

    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    f = Fetcher()
    f.load_robots()
    if not f.robots_ok:
        return 2

    got, missed = [], []
    html = f.get(ARCHIVE)
    links = parse_archive(html) if html else {}
    if not links:
        print("No season files found on the archive page (page unreachable or its layout changed).")
    for season in sorted(s for s in links if s >= a.first):
        text = f.get(links[season])
        if text and looks_like_csv(text):
            save(OUT / f"ncaa{season}.csv", text, links[season], manifest, f"season_{season}")
            got.append(season)
            print(f"  {season}: {manifest[f'season_{season}']['rows']} rows, "
                  f"{len(manifest[f'season_{season}']['columns'])} columns")
        else:
            missed.append(season)
            print(f"  {season}: not saved (missing or not a predictions file)")

    live = None
    if not a.no_live:
        for page in LIVE_PAGES:
            html = f.get(page)
            url = parse_live(html, page) if html else None
            if not url:
                continue
            text = f.get(url)
            if text and looks_like_csv(text):
                today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
                save(OUT / "live" / f"{today}.csv", text, url, manifest, f"live_{today}")
                live = today
                print(f"  live file saved for {today}: {manifest[f'live_{today}']['rows']} rows")
                break
        if live is None:
            print("  live file: not saved")

    if got or live:
        OUT.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"Seasons saved: {got or 'none'}. Not saved: {missed or 'none'}. Live file: {live or 'no'}.")
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
