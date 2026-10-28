"""The Odds API client for NFL player props, careful with the free plan's credits.

  GET /v4/sports/americanfootball_nfl/events                 free
  GET /v4/sports/americanfootball_nfl/events/{id}/odds       costs (markets returned) x (regions)

Up to ten named bookmakers count as one region, and an empty response costs nothing.
Every call's cost and the credits left (response headers) go to data/props/api_usage.csv,
and a call is refused when it could take the account below PROPS_RESERVE credits.

The key comes from the ODDS_API_KEY environment variable (a repository secret in the
Actions run). It is never written to a file or a log.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import urllib.parse
import urllib.request

from . import config as C

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
USAGE_PATH = C.DATA / "props" / "api_usage.csv"

# Books asked for, sharpest first. Ten or fewer, so a call costs one region per market.
BOOKS = ["pinnacle", "betonlineag", "lowvig", "fanduel", "draftkings", "betmgm", "williamhill_us",
         "betrivers", "fanatics", "espnbet"]
# Weight of each book in the consensus no-vig price. An assumption, not a fit: books known
# for taking sharp action and low holds count more. Revisit once the log shows which books
# the closing price tends to move toward.
BOOK_WEIGHT = {"pinnacle": 3.0, "betonlineag": 1.5, "lowvig": 1.5, "fanduel": 1.25, "draftkings": 1.0,
               "betmgm": 0.75, "williamhill_us": 0.75, "betrivers": 0.75, "fanatics": 0.75, "espnbet": 0.75}

PROPS_RESERVE = int(os.environ.get("PROPS_RESERVE", "40"))   # never spend below this many credits


class BudgetError(RuntimeError):
    pass


class Client:
    def __init__(self, key: str | None = None, reserve: int = PROPS_RESERVE, opener=None):
        self.key = key or os.environ.get("ODDS_API_KEY", "")
        self.reserve = reserve
        self.remaining: int | None = None
        self.used_this_run = 0
        self._open = opener or urllib.request.urlopen

    @property
    def ready(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, params: dict, what: str):
        q = dict(params, apiKey=self.key)
        url = f"{BASE}{path}?{urllib.parse.urlencode(q)}"
        with self._open(urllib.request.Request(url, headers={"User-Agent": "fb-value-board"}), timeout=30) as r:
            body = json.loads(r.read().decode())
            h = {k.lower(): v for k, v in r.headers.items()}
        last = int(float(h.get("x-requests-last", 0) or 0))
        rem = h.get("x-requests-remaining")
        if rem is not None:
            self.remaining = int(float(rem))
        self.used_this_run += last
        _log_usage(what, last, self.remaining)
        return body

    def events(self) -> list[dict]:
        """Upcoming NFL events (free)."""
        return self._get(f"/sports/{SPORT}/events", {"dateFormat": "iso"}, "events")

    def event_odds(self, event_id: str, markets: list[str], books: list[str] | None = None) -> dict:
        """Player-prop odds for one event. Refused if the worst-case cost could take the
        account below the reserve (worst case: every market returned)."""
        worst = len(markets)
        if self.remaining is not None and self.remaining - worst < self.reserve:
            raise BudgetError(f"{self.remaining} credits left, reserve {self.reserve}: skipping {event_id}")
        params = {"markets": ",".join(markets), "bookmakers": ",".join(books or BOOKS),
                  "oddsFormat": "american", "dateFormat": "iso"}
        return self._get(f"/sports/{SPORT}/events/{event_id}/odds", params, f"odds {event_id}")


def _log_usage(what: str, cost: int, remaining):
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not USAGE_PATH.exists()
    with USAGE_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "call", "cost", "remaining"])
        w.writerow([dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), what, cost, "" if remaining is None else remaining])


def flatten(event: dict) -> list[dict]:
    """One row per book x market x player x side x line from an event-odds response."""
    rows = []
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                if o.get("price") is None:
                    continue
                rows.append({
                    "event_id": event.get("id"), "commence": event.get("commence_time"),
                    "home_name": event.get("home_team"), "away_name": event.get("away_team"),
                    "book": bk.get("key"), "market": mk.get("key"), "updated": mk.get("last_update"),
                    "player": o.get("description") or "", "side": str(o.get("name", "")).lower(),
                    "line": o.get("point"), "price": float(o["price"]),
                })
    return rows
