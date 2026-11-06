import pandas as pd

from cfbmodel import pff_data, pff_returning as P


def test_names_normalise():
    assert P.nn("Travis Hunter Jr.") == P.nn("travis hunter") == "travishunter"
    assert P.nn("D'Andre Swift III") == "dandreswift"


def test_no_pff_data_means_zero(monkeypatch):
    monkeypatch.setattr(pff_data, "available", lambda: False)
    f = pd.DataFrame({"season": [2024], "home_team": ["A"], "away_team": ["B"]})
    out = P.add(f)
    assert out["d_pffret"].tolist() == [0.0]


def test_home_minus_away_with_median_fill(monkeypatch):
    monkeypatch.setattr(pff_data, "available", lambda: True)
    tv = pd.DataFrame({"season": [2024, 2024, 2024], "team": ["A", "B", "C"], "ret": [50.0, 30.0, 40.0],
                       "ret_off": [0.0] * 3, "ret_def": [0.0] * 3})
    monkeypatch.setattr(P, "team_values", lambda seasons: (tv, {"matched": 3, "transfers_in": 0}))
    f = pd.DataFrame({"season": [2024, 2024, 2016], "home_team": ["A", "Z", "A"], "away_team": ["B", "B", "B"]})
    out = P.add(f)
    assert out["d_pffret"].tolist()[0] == 20.0          # 50 - 30
    assert out["d_pffret"].tolist()[1] == 0.0           # unknown home takes the frame's median (50, 30, 30: 30) - 30
    assert out["d_pffret"].tolist()[2] == 0.0           # no PFF season before: both filled, 0
