"""The Parlays page's record (nflmodel/parlaylog.py): logging once, grading from the archive."""
import datetime as dt
import json

import pandas as pd

from nflmodel import parlaylog as PL

ARCH = {"games": [{"id": "G1", "home": "DAL", "away": "ARI"}, {"id": "G2", "home": "KC", "away": "DEN"}],
        "results": {"G1": {"home_score": 27, "away_score": 20}, "G2": {"home_score": 17, "away_score": 20}},
        "actuals": {"G1": {"p1": {"rec_yds": 64, "rec_td": 1, "rush_td": 0}}}}


def test_leg_results():
    r = lambda **l: PL.leg_result(l, ARCH)
    assert r(g="G1", m="spread", tm="DAL", L=-6.5) == "W"
    assert r(g="G1", m="spread", tm="DAL", L=-7.0) == "P"
    assert r(g="G1", m="spread", tm="ARI", L=6.5) == "L"
    assert r(g="G2", m="ml", tm="DEN", L=None) == "W"
    assert r(g="G1", m="total", L=46.5, d=None, k="over") == "W"
    assert r(g="G2", m="total", L=37.0, d=None, k="under") == "P"
    assert r(g="G1", m="prop", p="p1", s="rec_yds", L=59.5, d="over") == "W"
    assert r(g="G1", m="prop", p="p1", s="anytime_td", L=0.5, d="over") == "W"
    assert r(g="G1", m="prop", p="p2", s="rec_yds", L=20.5, d="over") == "P"      # did not play: void


def test_snapshot_logs_once_and_grades(tmp_path, monkeypatch):
    now = dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)
    legs = [{"g": "G1", "m": "spread", "lab": "DAL -6.5", "L": -6.5, "w": 0.55, "h": 0.0, "bk": {"dk": -105}, "c": {"k": "cover", "tm": "DAL"}},
            {"g": "G2", "m": "ml", "lab": "DEN ML", "L": None, "w": 0.6, "h": 0.0, "bk": {"dk": -130}, "c": {"k": "cover", "tm": "DEN"}}]
    data = {"season": 2026, "games": {"G1": {"ko": "2099-01-02T18:00Z", "week": 9}, "G2": {"ko": "2099-01-02T21:00Z", "week": 9}}, "legs": legs}
    n = PL.snapshot(data, "nfl", now=now, root=tmp_path)
    assert n == 3                                            # two legs and the 2-leg parlay at DraftKings
    assert PL.snapshot(data, "nfl", now=now, root=tmp_path) == 0
    monkeypatch.setattr(PL, "_archive", lambda league, season, week: ARCH)
    assert PL.grade("nfl", now=now, root=tmp_path) == 3
    res = pd.read_csv(tmp_path / "nfl_results.csv").set_index("key")
    p2 = res[res.index.str.startswith("p2")].iloc[0]
    assert p2["result"] == "W" and abs(p2["units"] - ((1 + 100 / 105) * (1 + 100 / 130) - 1)) < 1e-3
    s = PL.summary("nfl", root=tmp_path)
    assert s["kinds"]["p2"]["w"] == 1 and s["pending"] == 0
