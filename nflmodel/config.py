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
# Opening lines. nflverse only carries the closing number; the nfelo project
# publishes an opening and closing spread for most games since 2009.
OPENERS_URL = "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv"
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

EARLY_DAYS = 4              # a game this many days away (or more) is priced as an early line,
                            # using the weight the model has earned against opening spreads

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

# Primary and secondary team colours (from the nflverse teams file), used only as
# small identity swatches next to team names.
TEAM_COLORS = {
    "ARI": ("#97233F", "#000000"), "ATL": ("#A71930", "#000000"), "BAL": ("#241773", "#9E7C0C"),
    "BUF": ("#00338D", "#C60C30"), "CAR": ("#0085CA", "#000000"), "CHI": ("#0B162A", "#E64100"),
    "CIN": ("#FB4F14", "#000000"), "CLE": ("#FF3C00", "#311D00"), "DAL": ("#002244", "#B0B7BC"),
    "DEN": ("#002244", "#FB4F14"), "DET": ("#0076B6", "#B0B7BC"), "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"), "IND": ("#002C5F", "#A5ACAF"), "JAX": ("#006778", "#000000"),
    "KC": ("#E31837", "#FFB612"), "LA": ("#003594", "#FFD100"), "LAC": ("#007BC7", "#FFC20E"),
    "LV": ("#000000", "#A5ACAF"), "MIA": ("#008E97", "#F58220"), "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#C60C30"), "NO": ("#D3BC8D", "#000000"), "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#003F2D", "#000000"), "PHI": ("#004C54", "#A5ACAF"), "PIT": ("#000000", "#FFB612"),
    "SEA": ("#002244", "#69BE28"), "SF": ("#AA0000", "#B3995D"), "TB": ("#A71930", "#322F2B"),
    "TEN": ("#4495D2", "#D50A0A"), "WAS": ("#5A1414", "#FFB612"),
}
