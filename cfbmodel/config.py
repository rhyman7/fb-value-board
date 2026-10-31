"""Paths and model settings for the college board."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cfb"
RAW = DATA / "raw"          # downloaded files, not committed
CACHE = DATA / "cache"      # small per-season aggregates, committed
DOCS = ROOT / "docs" / "cfb"
SITE = ROOT / "site"

# cfbfastR data, published by the sportsdataverse project on GitHub. Free, no key.
BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-data/main/cfb"
FILES = {
    "schedule": "schedules/parquet/cfb_schedule_{s}.parquet",
    "teams": "cfb_teams/parquet/cfb_teams_{s}.parquet",
    "lines": "cfb_matchup_line/parquet/cfb_matchup_line_{s}.parquet",
    "pbp": "pbp/parquet/play_by_play_{s}.parquet",
    "player_box": "player_box/parquet/player_box_{s}.parquet",
}

# The full schedule, including games not yet played (the parquet schedule has finished games only).
FULL_SCHEDULE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/schedules/csv/cfb_schedule_{s}.csv"

FIRST_SEASON = 2014         # first season of play-by-play pulled (burn-in for ratings)
FIRST_MODEL_SEASON = 2015   # first season with betting lines; first season the game model trains on
FIRST_TEST_SEASON = 2017    # first season scored walk-forward (two seasons of training before it)
HOLDOUT_FROM = 2021         # settings are chosen on 2017-2019; 2021 on is the check
SKIP_TUNING = (2020,)       # the shortened 2020 season is scored but is in neither set

POSTSEASON_WEEK = 20        # postseason games are ordered after the regular season
SEASON_LEN = 22             # weeks per season on the running clock used for decay

# Garbage time: a lead bigger than this in the quarter (the usual college cutoffs).
GARBAGE_LEAD = {1: 43, 2: 37, 3: 27, 4: 21}
EPA_CLIP = 5.0              # one freak play should not swing a game's rating

# ---- Rating settings (chosen on 2017-2019) ----------------------------------
HALF_LIFE_GAMES = 9.0
OFFSEASON_CARRY = 0.55
# NFL penalties times 0.05: college ratings need far less shrinkage (more teams, wider
# talent gaps; the game model and the market-implied ratings do the rest). Grid on 2017-2019.
RIDGE = {"pass": 12.0, "rush": 16.0, "sr": 18.0, "pts": 0.4, "snaps": 0.4}
MARKET_HALF_LIFE = 8.0     # weeks (NFL 4); 8 beat 2, 4 and 12 on 2017-2019
QB_PRIOR_DROPBACKS = 800.0  # college starters are judged on more dropbacks (NFL 200); QB-change games, 2017-2019
QB_REPLACEMENT = -0.30      # an unknown quarterback, EPA/dropback vs average (NFL -0.13): college backups are much worse
QB_HALF_LIFE_DROPBACKS = 1400.0

TOTAL_WEIGHT_CAP = 0.05
BLEND_LOOKBACK = 8
EARLY_DAYS = 4

KELLY_FRACTION = 0.25
MAX_STAKE = 0.03
MAX_GAME_STAKE = 0.05       # cap on one game's flagged stakes together (nflmodel/config.py)
MAX_WEEK_STAKE = 0.10       # cap on one week's flagged stakes together (nflmodel/config.py)
EDGE_HAIRCUT = True         # stakes from the model probability shrunk toward the market's (nflmodel/config.py)
POOL = "FCS"                # every non-FBS opponent is pooled into one team
