"""experiments/nfl_questionable_test.py: the pure pieces of the Nov 9, 2026 test."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import nfl_questionable_test as T  # noqa: E402


def test_listed_active_ignores_inactive_list():
    st = pd.Series(["ACT", "INA", "RES", None, "ACT"])
    rep = pd.Series([None, "Questionable", None, None, "Doubtful"])
    assert T.listed_active(st, rep).tolist() == [1, 1, 0, 1, 0]


def test_play_rates_shrink_toward_overall():
    h = pd.DataFrame({"group": ["WR"] * 4 + ["RB"] * 4, "practice_lvl": [0] * 4 + [2] * 4,
                      "played_snap": [1, 1, 1, 1, 0, 0, 0, 0]})
    v1, v2 = T.play_rates(h, shrink=4)
    assert v1 == 0.5
    assert np.isclose(v2[("WR", 0)], 0.5 + 0.5 * 0.5)
    assert np.isclose(v2[("RB", 2)], 0.5 - 0.5 * 0.5)


def test_shade_moves_share_to_unlisted_teammates_only():
    p = pd.DataFrame({"game_id": ["g"] * 4, "team": ["A", "A", "A", "B"],
                      "report": ["Questionable", None, None, None], "listed": [1, 1, 0, 1],
                      **{s + "_proj": [10.0, 30.0, 5.0, 20.0] for s in T.SHADE_STATS}})
    c = pd.Series([0.6, np.nan, np.nan, np.nan])
    out = T.shade(p, c)
    assert np.isclose(out.loc[0, "rec_yds_proj"], 6.0)
    assert np.isclose(out.loc[1, "rec_yds_proj"], 34.0)     # gets all 4: the other teammate isn't expected to play
    assert out.loc[2, "rec_yds_proj"] == 5.0 and out.loc[3, "rec_yds_proj"] == 20.0
    assert np.isclose(out.loc[[0, 1, 2], "rec_yds_proj"].sum(), 45.0)


def test_verdict():
    good = {s: {"n": 5, "mse_diff": -1.0} for s, _ in T.MEASURES}
    bad = {s: {"n": 5, "mse_diff": 1.0} for s, _ in T.MEASURES}
    assert T.verdict({"a": {"all": good, "listed": good}, "b": {"all": good, "listed": good}}) == "kept"
    assert T.verdict({"a": {"all": good, "listed": bad}, "b": {"all": good, "listed": good}}) == "mixed"
    assert T.verdict({"a": {"all": bad, "listed": bad}, "b": {"all": bad, "listed": bad}}) == "not kept"
    assert T.verdict({"a": {"all": good, "listed": good}, "b": {"all": bad, "listed": bad}}) == "mixed"


def test_questionable_fit_skips_players_who_sat():
    """The live multiplier (players_ml.questionable with snaps) ignores earlier Questionable
    players who never got on the field; without snaps it is the old fit."""
    from nflmodel import players_ml as ML
    n = 40
    hist = pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "player_id": [f"p{i}" for i in range(n)],
                         "season": 2020, "group": "WR", "report_lvl": 1, "practice_lvl": 1, "played": True, "active": 1,
                         "rec_yds_proj": 50.0, "rec_yds": [50.0] * (n // 2) + [0.0] * (n // 2),
                         "targets": [6.0] * (n // 2) + [0.0] * (n // 2), "carries": 0.0, "pass_att": 0.0, "receptions": 0.0})
    cur = hist.iloc[:1].assign(game_id="new", player_id="x", season=2021, rec_yds=np.nan)
    p = pd.concat([hist, cur], ignore_index=True)
    snaps = pd.DataFrame({"game_id": hist["game_id"][: n // 2], "player_id": hist["player_id"][: n // 2], "offense_snaps": 30})
    old = ML.questionable(p, first=2021, shrink=0)
    new = ML.questionable(p, first=2021, shrink=0, snaps=snaps)
    assert np.isclose(old.loc[n, "rec_yds_proj"], 25.0)      # the sitters halve it
    assert np.isclose(new.loc[n, "rec_yds_proj"], 50.0)      # only those who played
