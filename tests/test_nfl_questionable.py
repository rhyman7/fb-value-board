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
