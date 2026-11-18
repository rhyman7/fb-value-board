"""The role-change term in the prop distributions (nflmodel/rolechange.py, props.role, propdist)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import propdist as D, props as P, rolechange as RC  # noqa: E402


def test_a_zero_term_prices_exactly_as_before():
    mu, share = np.array([3.0, 6.0]), np.array([0.1, 0.25])
    two = np.column_stack([share, [0.3, 0.0]])
    f = D.CountFit(a=-1.0, b=0.2, c=-2.0, q=0.05, f=0.3)
    assert np.allclose(f.pmf(mu, share, 20), f.pmf(mu, two, 20))
    y = D.YardsFit(a=0.5, b=0.1, c=-1.0)
    assert np.allclose(y.k1(mu, share), y.k1(mu, two))
    wide = D.CountFit(a=-1.0, b=0.2, c=-2.0, q=0.05, f=0.3, d=3.0)
    assert wide.alpha(mu, two)[0] > f.alpha(mu, two)[0]          # wider after a change
    assert np.isclose(wide.alpha(mu, two)[1], f.alpha(mu, two)[1])   # no change, no effect


def test_role_carries_u_only_where_it_applies():
    df = pd.DataFrame({"tgt_share": [0.2, 0.1], "car_share": [0.5, 0.0], "role_chg_rec": [0.1, np.nan], "role_chg_car": [0.2, 0.0]})
    r = P.role(df, "rec_yds", "WR")
    assert r.shape == (2, 2) and np.allclose(r[:, 1], [0.1, 0.0])
    assert P.role(df, "carries", "QB").ndim == 1
    assert P.role(df, "anytime_td", "WR").ndim == 1
    assert P.role(df.drop(columns=["role_chg_rec", "role_chg_car"]), "rec_yds", "WR").ndim == 1


def test_role_change_uses_earlier_games_only():
    rows = []
    for w, car in enumerate([10, 10, 10, 20]):           # his team's carries are 20 each game
        rows.append({"player_id": "a", "game_id": f"g{w}", "team": "X", "season": 2025, "t": w, "played": True,
                     "carries": car, "targets": 0})
        rows.append({"player_id": "b", "game_id": f"g{w}", "team": "X", "season": 2025, "t": w, "played": True,
                     "carries": 20 - car, "targets": 1})
    rows.append({"player_id": "a", "game_id": "g4", "team": "X", "season": 2025, "t": 4, "played": False, "carries": 0, "targets": 0})
    out = RC.add(pd.DataFrame(rows), "targets").set_index(["player_id", "game_id"])
    assert out.loc[("a", "g0"), "role_chg_car"] == 0.0           # no earlier game
    assert out.loc[("a", "g2"), "role_chg_car"] == 0.0           # steady half share
    assert out.loc[("a", "g3"), "role_chg_car"] == 0.0           # his jump to all the carries happens in g3 itself
    assert 0.2 < out.loc[("a", "g4"), "role_chg_car"] <= RC.CAP  # the game after it sees the jump


def test_a_save_keeps_keys_other_modules_own(tmp_path):
    m = P.PropModel.from_dict({"meta": {}, "family": {}, "counts": {}, "yards": {}, "isotonic": {}, "shade": {"x": 1}})
    p = tmp_path / "p.json"
    m.save(p)
    assert json.loads(p.read_text())["shade"] == {"x": 1}
