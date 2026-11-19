"""The Friday quarterback rule (nflmodel/bettime.py, Nov 19, 2026)."""
import pandas as pd

from nflmodel import bettime


def test_friday_qb_rule(monkeypatch):
    # team HOM: A starts wk1-2; wk3 B starts with A listed Out Friday (known); wk4 C starts, A not listed (a surprise:
    # unknown, the board assumes B); wk5 A is back (an earlier starter returning: known)
    games = pd.DataFrame({"season": 2025, "week": [1, 2, 3, 4, 5], "game_id": [f"g{w}" for w in range(1, 6)],
                          "home_team": "HOM", "away_team": "AWY", "home_qb_id": ["A", "A", "B", "C", "A"],
                          "away_qb_id": ["Z"] * 5})
    rep = pd.DataFrame({"season": 2025, "week": [3], "team": ["HOM"], "gsis_id": ["A"], "report_status": ["Out"]})
    ros = pd.DataFrame({"season": 2025, "week": [1, 2, 3, 4, 5] * 3, "team": "HOM",
                        "gsis_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5, "status": "ACT"})
    monkeypatch.setattr(bettime, "_parquet", lambda kind, name, season, cols: rep if kind == "injuries" else ros)
    q = bettime._qb_season(games, 2025).set_index("game_id")
    assert [None if pd.isna(x) else x for x in q.home_qb_id] == ["A", "A", "B", None, "A"]
    assert list(q.away_qb_id) == ["Z"] * 5
