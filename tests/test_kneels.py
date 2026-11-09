"""Rushing props settle on the official line, kneel-downs included (players_data.SETTLE)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import parlaylog as PL, players_data as PD, proplog as PLOG  # noqa: E402


def test_kneels_are_counted_per_rusher_and_kept_out_of_carries():
    p = pd.DataFrame({"game_id": ["g"] * 4, "posteam": ["DEN"] * 4, "rusher_player_id": ["qb", "qb", "qb", "rb"],
                      "rush_attempt": [1, 1, 1, 1], "qb_kneel": [1, 1, 0, 0], "rushing_yards": [-1, -1, 6, 4]})
    k = PD.kneels(p, ["game_id", "posteam"])
    assert len(k) == 1 and k.iloc[0]["player_id"] == "qb"
    assert k.iloc[0]["kneels"] == 2 and k.iloc[0]["kneel_yds"] == -2


def test_prop_grading_columns_and_parlay_leg_use_the_official_line():
    assert PLOG.STAT_COLS["rush_yds"] == ["rush_yds", "kneel_yds"] and PLOG.STAT_COLS["carries"] == ["carries", "kneels"]
    arch = {"results": {"g": {"home_score": 20, "away_score": 17}}, "games": {"g": {"home": "DEN", "away": "KC"}},
            "actuals": {"g": {"qb": {"rush_yds": 5.0, "carries": 3.0, "kneels": 1.0, "kneel_yds": -1.0},
                              "old": {"rush_yds": 5.0, "carries": 3.0}}}}
    leg = {"g": "g", "m": "prop", "s": "rush_yds", "L": 4.5, "d": "over", "p": "qb"}
    assert PL.leg_result(leg, arch) == "L"                                      # 5 on the board's line, 4 officially
    assert PL.leg_result({**leg, "p": "old"}, arch) == "W"                      # a week frozen before the kneel fields
    assert PL.leg_result({**leg, "s": "carries", "L": 3.5}, arch) == "W"
