"""The over-lean shade on the books' vig-free chance (nflmodel/propshade.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import propshade as S  # noqa: E402


def test_groups_cover_only_the_tested_markets():
    assert S.group("receptions", "WR") == S.group("rec_yds", "RB") == S.group("rush_yds", "RB") == S.group("carries", "RB") == "rec_rush"
    assert S.group("pass_yds", "QB") == "pass_yds"
    for stat, pos in (("rush_yds", "QB"), ("carries", "QB"), ("pass_td", "QB"), ("completions", "QB"), ("pass_att", "QB"),
                      ("ints", "QB"), ("anytime_td", "WR")):
        assert S.group(stat, pos) is None and S.shade_for(stat, pos) == 0.0


def test_shade_moves_the_right_way_and_is_bounded(monkeypatch):
    monkeypatch.setattr(S, "_cache", {"rec_rush": -0.0266, "pass_yds": 0.0123})
    assert abs(S.apply(0.5, "receptions", "WR") - 0.4734) < 1e-9
    assert abs(S.apply(0.5, "pass_yds", "QB") - 0.5123) < 1e-9
    assert S.apply(0.5, "pass_td", "QB") == 0.5
    assert S.apply(0.015, "rec_yds", "TE") == 0.01


def test_saved_shade_matches_the_fit():
    S.reset()
    sh = S.load()
    assert set(sh) >= {"rec_rush", "pass_yds"} and -0.04 < sh["rec_rush"] < -0.01 and 0.005 < sh["pass_yds"] < 0.03
