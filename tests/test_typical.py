"""Typical: the model's 50/50 yardage sits at or below the projection and moves with it."""
from nflmodel import typical as T


def _wr(i, yds):
    return {"id": f"p{i}", "name": f"Player {i}", "pos": "WR", "team": "AAA", "opp": "BBB", "game": "g1", "week": 1,
            "proj": {"targets": yds / 8.0, "receptions": yds / 12.5, "rec_yds": yds},
            "why": {"tgt_share": min(0.3, yds / 300), "car_share": 0.0, "games": 8}}


def test_typical_below_projection_and_rising():
    board = {"players": [_wr(1, 30.0), _wr(2, 60.0), _wr(3, 90.0)]}
    assert T.attach(board, "nfl") == 3
    t = [p["typical"]["rec_yds"] for p in board["players"]]
    assert t[0] < t[1] < t[2]
    for p, v in zip(board["players"], t):
        assert 0 < v <= p["proj"]["rec_yds"]
    assert "pass_yds" not in board["players"][0]["typical"]      # a receiver has no passing number
