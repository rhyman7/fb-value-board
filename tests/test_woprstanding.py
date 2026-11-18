import json

import numpy as np
import pandas as pd

from nflmodel import woprstanding as WS

K = {"min_proj": 10.0, "clip": 2.5, "b": 0.05,
     "line": {"WR": {"c0": 0.06, "c1": 0.0067, "sd": 0.075}, "TE": {"c0": 0.02, "c1": 0.0071, "sd": 0.035}}}


def test_factor_moves_only_receivers_above_the_cutoff():
    grp = np.array(["WR", "WR", "TE", "RB", "QB", "WR"])
    proj = np.array([50.0, 9.0, 40.0, 30.0, 5.0, 50.0])
    usual = 0.06 + 0.0067 * 50
    w = np.array([usual + 0.075, 0.5, 0.02 + 0.0071 * 40 - 0.035, 0.4, 0.0, usual + 10])
    f = WS.factor(grp, proj, w, K)
    assert np.allclose(f, [1.05, 1.0, 0.95, 1.0, 1.0, 1 + 0.05 * 2.5])


def test_constants_file_matches_the_tested_version():
    k = json.loads(WS.PATH.read_text())
    assert k["version"] == "A" and k["min_proj"] == WS.MIN_PROJ and k["clip"] == WS.CLIP
    assert 0 < k["b"] <= 0.2 and set(k["line"]) == set(WS.GROUPS)


def _table():
    rows = []
    for t, wk in enumerate(range(1, 5)):
        played = wk < 4
        for pid, grp, air, sh in (("a", "WR", 120.0, 0.30), ("b", "WR", 40.0, 0.15), ("c", "TE", 40.0, 0.12), ("r", "RB", 0.0, 0.10)):
            rows.append({"game_id": f"g{wk}", "player_id": pid, "team": "X", "season": 2026, "t": 2026 * 100 + wk, "played": played,
                         "active": 1, "group": grp, "air_yards": air if played else 0.0, "tgt_share_raw": sh})
    return pd.DataFrame(rows)


def test_apply_changes_this_seasons_receiving_yards_only():
    t = _table()
    proj = t[["game_id", "player_id", "season", "group"]].copy()
    proj["rec_yds_proj"] = proj["player_id"].map({"a": 60.0, "b": 30.0, "c": 35.0, "r": 20.0})
    proj["receptions_proj"] = 4.0
    out, n = WS.apply(proj, t, 2026, k=K)
    assert n > 0 and (out["receptions_proj"] == 4.0).all()
    assert (out.loc[out.group == "RB", "rec_yds_proj"] == 20.0).all()
    a = out[(out.player_id == "a") & (out.game_id == "g4")]["rec_yds_proj"].iloc[0]
    assert a > 60.0                                   # 60% of the team's air yards at a 60-yard projection
    same, n0 = WS.apply(proj, t, 2025, k=K)
    assert n0 == 0 and same["rec_yds_proj"].equals(proj["rec_yds_proj"])
    only_te, _ = WS.apply(proj, t, 2026, groups=("TE",), k=K)
    assert (only_te.loc[only_te.group == "WR", "rec_yds_proj"] == proj.loc[proj.group == "WR", "rec_yds_proj"]).all()


def test_off_switch(monkeypatch):
    t = _table()
    proj = t[["game_id", "player_id", "season", "group"]].assign(rec_yds_proj=50.0)
    monkeypatch.setattr(WS, "APPLIED", False)
    out, n = WS.apply(proj, t, 2026, k=K)
    assert n == 0 and out is proj
