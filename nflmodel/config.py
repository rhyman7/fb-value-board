"""Paths and model settings in one place."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"          # downloaded files, not committed
CACHE = DATA / "cache"      # small per-season aggregates, committed
DOCS = ROOT / "docs"        # the published dashboard (GitHub Pages)
SITE = ROOT / "site"        # dashboard template

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

FIRST_SEASON = 2006         # first season of play-by-play pulled (burn-in for ratings)
FIRST_MODEL_SEASON = 2008   # first season whose games are used to train the game model
FIRST_TEST_SEASON = 2014    # first season scored in the walk-forward backtest
HOLDOUT_FROM = 2019         # settings were chosen on seasons before this one

# Franchise moves: map old abbreviations to the current one.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# ---- Rating settings (chosen on 2014-2018, see README) ----------------------
HALF_LIFE_GAMES = 9.0       # a game this many team-games ago counts half as much
OFFSEASON_CARRY = 0.55      # extra multiplier applied to everything from a prior season
RIDGE = {                   # shrinkage toward league average (plays; games for points)
    "pass": 240.0, "rush": 320.0, "sr": 360.0, "pts": 8.0,
}
MARKET_HALF_LIFE = 4.0      # weeks; the market re-prices teams quickly
QB_PRIOR_DROPBACKS = 200.0  # shrinkage for quarterback ratings
QB_REPLACEMENT = -0.07      # prior EPA/dropback (vs league avg) for an unknown QB
QB_HALF_LIFE_DROPBACKS = 700.0

BLEND_LOOKBACK = 8          # seasons of history used to measure how much to trust the model
                            # over the market (its edge has shrunk, so old seasons mislead)

# ---- Betting settings -------------------------------------------------------
KELLY_FRACTION = 0.25       # stake = this share of full Kelly
MAX_STAKE = 0.03            # cap on any single stake, as a share of bankroll

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
