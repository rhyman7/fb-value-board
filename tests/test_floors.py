import numpy as np
import pandas as pd

from nflmodel import floors as F


def _log(rows):
    base = {"pass_att": 0, "carries": 0, "targets": 0, "receptions": 0, "pass_yds": 0, "rush_yds": 0, "rec_yds": 0,
            "share": np.nan, "qb_start": False, "team": "A", "opp": "B", "player_id": "p"}
    out = []
    for i, r in enumerate(rows):
        out.append({**base, "game_id": f"g{i}", "t": i, "week": i + 1, **r})
    return pd.DataFrame(out)


def test_position_split_and_min_games():
    log = _log([{"carries": 10, "rush_yds": 50, "targets": 2, "rec_yds": 10}] * 3)
    assert set(F.player_floor(log, "RB", "A", "nfl")) == {"rush", "rec", "rr"}
    assert set(F.player_floor(log, "WR", "A", "nfl")) == {"rec"}
    assert F.player_floor(log.iloc[:2], "RB", "A", "nfl") == {}


def test_adjusted_drops_early_exit_other_team_and_non_starts():
    log = _log([
        {"carries": 15, "rush_yds": 80, "share": 0.7},
        {"carries": 14, "rush_yds": 60, "share": 0.65},
        {"carries": 2, "rush_yds": 3, "share": 0.1},            # left early
        {"carries": 16, "rush_yds": 70, "share": 0.7},
        {"carries": 12, "rush_yds": 5, "share": 0.6, "team": "Z"},   # old team
    ])
    adj = F.player_floor(log, "RB", "A", "nfl")["rush"]
    raw = F.player_floor(log, "RB", "A", "nfl", adjusted=False)["rush"]
    assert adj["floor"] == 60 and adj["g"] == 3
    assert raw["floor"] == 3 and raw["g"] == 5
    qb = _log([{"pass_att": 30, "pass_yds": y, "qb_start": s} for y, s in ((250, True), (40, False), (200, True), (220, True))])
    assert F.player_floor(qb, "QB", "A", "nfl")["pass"]["floor"] == 200


def test_played_game_without_a_catch_counts_as_zero_in_adjusted():
    log = _log([{"targets": 6, "rec_yds": 60, "share": 0.8}, {"targets": 0, "rec_yds": 0, "share": 0.75},
                {"targets": 5, "rec_yds": 40, "share": 0.8}])
    assert F.player_floor(log, "WR", "A", "nfl")["rec"]["floor"] == 0


def test_flags():
    used = pd.DataFrame({"game_id": ["g1", "g2"], "opp": ["B", "C"]})
    ctx = {("g1", "A"): {"spread": 3, "wind": 5, "qb": "q1"}, ("g2", "A"): {"spread": -2, "wind": np.nan, "qb": "q1"}}
    allow = {("B", "WR", "rec"): 100, ("C", "WR", "rec"): 90, ("D", "WR", "rec"): 80}
    now = {"team": "A", "opp": "D", "spread": -6, "wind": 18, "qb": "q2"}
    f = F.flags("rec", "WR", used, ctx, now, allow, "m", {"m"})
    assert set(f) == {"tough_d", "new_qb", "wind", "mate_out"}
    assert "dog" in F.flags("rush", "RB", used, ctx, now, {}, None, set())
