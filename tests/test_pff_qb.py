import numpy as np
import pandas as pd

from nflmodel import config as C, pff_qb


def _games():
    rows = []
    for s in (2014, 2015, 2016):
        for wk in range(1, 18):
            for i, g in enumerate((90.0, 60.0, 75.0) + tuple(60.0 + 3 * k for k in range(10))):
                rows.append({"nfl_id": f"q{i}", "season": s, "order": s * 100 + wk, "dropbacks": 35.0,
                             "grades_pass": g, "epa_db": (g - 75.0) * 0.015})
    return pd.DataFrame(rows)


def test_map_is_walk_forward_and_grade_raises_value():
    g = _games()
    co = pff_qb.fit_map(g, [2014, 2015, 2016, 2017])
    assert 2014 not in co and 2015 in co          # 2015 is fitted on 2014 only
    assert co[2016][1] > 0.01                     # higher grade, higher EPA
    pr = pff_qb.Prior(g, co)
    v = pr.values(201605, ["q0", "q1", "q2", "nobody"], C.QB_HALF_LIFE_DROPBACKS)
    assert v[0] > v[2] > v[1]
    assert v[3] == C.QB_REPLACEMENT               # no PFF games: the flat prior


def test_only_earlier_games_count():
    g = _games()
    pr = pff_qb.Prior(g, pff_qb.fit_map(g, [2016]))
    later = g.copy()
    later.loc[later.order >= 201605, "grades_pass"] = 0.0     # games from the rated week on
    pr2 = pff_qb.Prior(later, pff_qb.fit_map(g, [2016]))
    a = pr.values(201605, ["q0"], C.QB_HALF_LIFE_DROPBACKS)
    b = pr2.values(201605, ["q0"], C.QB_HALF_LIFE_DROPBACKS)
    assert np.allclose(a, b)


def test_no_season_map_or_no_data_gives_replacement():
    g = _games()
    pr = pff_qb.Prior(g, {})
    assert (pr.values(201605, ["q0"], 2800.0) == C.QB_REPLACEMENT).all()
    empty = pff_qb.Prior(g.iloc[:0], {2016: np.array([0.0, 0.01])})
    assert (empty.values(201605, ["q0"], 2800.0) == C.QB_REPLACEMENT).all()
