import datetime as dt

import pandas as pd

from nflmodel import nfelolog


def test_match_games_nearest_and_aliases():
    games = pd.DataFrame({
        "game_id": ["2026_09_LV_LA", "2026_17_LV_LA", "2026_09_KC_DEN"],
        "home_team": ["LA", "LA", "DEN"], "away_team": ["LV", "LV", "KC"],
        "gameday": ["2026-11-08", "2026-12-27", "2026-11-08"], "spread_line": [3.0, 1.0, -2.5]})
    t = pd.DataFrame({"home_team": ["LAR", "DEN"], "away_team": ["OAK", "KC"], "nfelo_projected_home_spread": [-3.5, 2.0]})
    m = nfelolog.match_games(t, games, dt.date(2026, 11, 3))
    assert list(m["game_id"]) == ["2026_09_LV_LA", "2026_09_KC_DEN"]
