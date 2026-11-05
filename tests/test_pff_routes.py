"""PFF route inputs for the NFL New projections (nflmodel/pff_routes.py), on made-up rows: no PFF
data is needed (or allowed) in this public repo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import players_ml, pff_routes  # noqa: E402


def _table():
    rows = []
    for w in range(1, 5):
        for pid in ("00-1", "00-2"):
            rows.append({"game_id": f"2025_{w:02d}_AAA_BBB", "player_id": pid, "team": "AAA", "opp": "BBB", "season": 2025,
                         "week": w, "t": 2025 * 100 + w, "active": 1, "played": w <= 3, "group": "WR"})
    return pd.DataFrame(rows)


def _rec(routes_wk3=30):
    rows = []
    for w in range(1, 4):
        rows.append({"season": 2025, "week": w, "player_id": 11, "player": "Veteran One", "team_name": "AAA",
                     "routes": 30 if w < 3 else routes_wk3, "pass_plays": 40, "targets": 8, "team_pass": 40, "nfl_id": "00-1"})
        rows.append({"season": 2025, "week": w, "player_id": 22, "player": "Rookie Two Jr.", "team_name": "AAA",
                     "routes": 20, "pass_plays": 25, "targets": 4, "team_pass": 40, "nfl_id": np.nan})
    return pd.DataFrame(rows)


def _players(monkeypatch):
    p = pd.DataFrame({"gsis_id": ["00-1", "00-2"], "pff_id": [11, np.nan], "display_name": ["Veteran One", "Rookie Two"]})
    monkeypatch.setattr(pff_routes, "_players", lambda: p)


def test_no_pff_data_means_todays_features():
    t = _table()
    assert pff_routes.features(t, rec=pd.DataFrame()).empty
    assert players_ml.feats_for(t) == players_ml.FEATS
    filled = t.assign(**{c: 0.5 for c in players_ml.ROUTE_F})
    assert players_ml.feats_for(filled) == players_ml.FEATS + players_ml.ROUTE_F


def test_route_inputs_use_earlier_games_only(monkeypatch):
    _players(monkeypatch)
    a = pff_routes.features(_table(), rec=_rec(30)).set_index(["game_id", "player_id"])
    b = pff_routes.features(_table(), rec=_rec(5)).set_index(["game_id", "player_id"])
    k3, k4 = ("2025_03_AAA_BBB", "00-1"), ("2025_04_AAA_BBB", "00-1")
    assert np.isclose(a.loc[k3, "route_avg"], b.loc[k3, "route_avg"])          # week 3's own routes not used for week 3
    assert np.isclose(a.loc[k3, "route_last"], 30 / 40)
    assert np.isclose(b.loc[k4, "route_last"], 5 / 40)                          # but used for week 4
    assert np.isnan(a.loc[("2025_01_AAA_BBB", "00-1"), "route_avg"])            # no history


def test_unlinked_rookie_is_matched_by_name_not_counted_at_zero(monkeypatch):
    _players(monkeypatch)
    f = pff_routes.features(_table(), rec=_rec()).set_index(["game_id", "player_id"])
    assert np.isclose(f.loc[("2025_04_AAA_BBB", "00-2"), "route_last"], 20 / 40)
