"""College model tests. Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import bets as B  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402
from nflmodel.ratings import RatingEngine  # noqa: E402


def _toy():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(12)]
    rows, qbs = [], []
    gid = 0
    for week in range(1, 9):
        order = 201500 + week
        perm = rng.permutation(teams)
        for h, a in zip(perm[::2], perm[1::2]):
            gid += 1
            for side, t, o in (("home", h, a), ("away", a, h)):
                n_pass, n_rush = int(rng.integers(20, 40)), int(rng.integers(20, 40))
                rows.append({"game_id": gid, "season": 2015, "side": side, "team": t, "opp": o, "order": order,
                             "week": week, "neutral": 0, "is_home": int(side == "home"),
                             "plays": n_pass + n_rush, "epa": rng.normal(0, 8), "succ": rng.integers(20, 40),
                             "n_pass": n_pass, "n_rush": n_rush, "pass_epa": rng.normal(0, 5), "rush_epa": rng.normal(0, 5),
                             "pts": rng.integers(0, 50), "snaps": rng.integers(55, 90)})
                qbs.append({"game_id": gid, "season": 2015, "side": side, "qb_id": f"{t}qb", "name": t, "team": t,
                            "dropbacks": n_pass, "epa": rng.normal(0, 5), "order": order})
    return pd.DataFrame(rows), pd.DataFrame(qbs)


def test_fast_solver_matches_nfl_solver():
    tg, qb = _toy()
    e = CFBRatingEngine(tg, qb)
    idx = np.flatnonzero(e.order < 201508)
    w = e._weights(201508, e.order[idx])
    for m in e.metrics:
        a, b = RatingEngine._solve(e, m, idx, w), e._solve(m, idx, w)
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and np.isclose(a[2], b[2]) and np.isclose(a[3], b[3])


def test_ratings_ignore_the_current_week():
    tg, qb = _toy()
    e = CFBRatingEngine(tg, qb)
    before = e.ratings_at(201505)
    tg2 = tg.copy()
    tg2.loc[tg2.order >= 201505, "pts"] = 99           # change this week's and later results
    after = CFBRatingEngine(tg2, qb).ratings_at(201505)
    assert np.allclose(before["pts"]["off"], after["pts"]["off"])


def test_half_point_rounding_and_grading():
    assert list(B.half([-3.17, 6.8, 14.25, 0.2])) == [-3.0, 7.0, 14.5, 0.0]
    games = pd.DataFrame({"game_id": [1], "season": [2025], "week": [3], "home_team": ["A"], "away_team": ["B"],
                          "spread_line": [6.83], "total_line": [52.2], "proj_margin": [6.83], "proj_total": [52.2],
                          "result": [7.0], "total": [55.0]})
    dist = {"margin": np.ones(len(B.MARGINS)), "total": np.ones(len(B.TOTALS)), "sd_margin": 16.0, "sd_total": 15.0, "sd_win": 16.0}
    b = B.price_games(games, dist, "proj").set_index(["market", "side"])
    assert b.loc[("Spread", "home"), "outcome"] == 0        # A -7 pushes on a 7-point win
    assert b.loc[("Total", "over"), "outcome"] == 1
    assert abs(b.loc[("Spread", "home"), "model_prob"] - 0.5) < 0.01   # no disagreement, no edge
