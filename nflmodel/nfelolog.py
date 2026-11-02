"""Save nfelo's published NFL predictions each week, before the games, for a clean forward test.

    python -m nflmodel.nfelolog

Why. nfelo's games file (data/raw/nfelo_games.csv) is rebuilt after the fact with its current
settings, and its quarterback adjustment uses the quarterback who actually started, so its past
numbers can't be graded fairly against the opening line (FEATURES.md, "Blending with nfelo's
model"). What nfelo publishes before a game can. Two files from its public repo
(github.com/greerreNFL/nfelo, output_data/):

  prediction_tracker.csv          this week's projected spreads, as of now. Appended here with
                                  the time it was saved, the nflverse game_id it belongs to and
                                  nflverse's current spread, so each week's
                                  Tuesday (early-week) and Saturday (late-week) numbers are kept.
  historic_projected_spreads.csv  nfelo's own archive of the spreads it published (2021 on).
                                  Copied as-is; git keeps every version.

Grading later joins on game_id to nfelo_games.csv, which carries each game's opening line once it
is played. Nothing on the board reads these files; no Odds API credits are used. Writes data/nfelo_log/.
"""
from __future__ import annotations

import datetime as dt
import io
import urllib.request

import pandas as pd

from . import config as C, data

BASE = "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data"
OUT = C.DATA / "nfelo_log"
TRACKER = OUT / "prediction_tracker_log.csv"
ARCHIVE = OUT / "historic_projected_spreads.csv"


def _get(name: str) -> bytes:
    req = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def match_games(t: pd.DataFrame, games: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Each tracker row's game: the nflverse game with the same home and away teams nearest to
    today, within a week either way (the tracker can still show a slate that has just been
    played; nfelo uses OAK and LAR, nflverse LV and LA)."""
    day = pd.to_datetime(games["gameday"]).dt.date
    g = games[(day >= today - dt.timedelta(days=7)) & (day <= today + dt.timedelta(days=10))].copy()
    g["away_days"] = (pd.to_datetime(g["gameday"]) - pd.Timestamp(today)).dt.days.abs()
    first = g.sort_values("away_days").drop_duplicates(["home_team", "away_team"])[["home_team", "away_team", "game_id", "gameday", "spread_line"]]
    alias = {**C.TEAM_ALIASES, "LAR": "LA"}
    t = t.assign(home_team=t["home_team"].replace(alias), away_team=t["away_team"].replace(alias))
    return t.merge(first, on=["home_team", "away_team"], how="left")


def run(now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(io.BytesIO(_get("prediction_tracker.csv")))
    games = data.load_games(refresh=True)
    t = match_games(t, games, now.date())
    t.insert(0, "saved_utc", now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    t = t.rename(columns={"spread_line": "nflverse_spread_now"})
    old = pd.read_csv(TRACKER) if TRACKER.exists() else pd.DataFrame()
    pd.concat([old, t], ignore_index=True).to_csv(TRACKER, index=False)
    ARCHIVE.write_bytes(_get("historic_projected_spreads.csv"))
    t["played"] = pd.to_datetime(t["gameday"]).dt.date < now.date()   # saved after kickoff: not a pre-game number
    res = {"saved": int(len(t)), "matched": int(t["game_id"].notna().sum()), "at": t["saved_utc"].iloc[0] if len(t) else None}
    print(f"nfelo: saved {res['saved']} projections ({res['matched']} matched to a game) and the archive")
    return res


if __name__ == "__main__":
    run()
