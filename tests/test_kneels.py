"""Rushing props settle on the official line, kneel-downs included (players_data.SETTLE)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import parlaylog as PL, players_data as PD, proplog as PLOG  # noqa: E402


def test_kneels_are_counted_per_rusher_and_kept_out_of_carries():
    p = pd.DataFrame({"game_id": ["g"] * 4, "posteam": ["DEN"] * 4, "rusher_player_id": ["qb", "qb", "qb", "rb"],
                      "rush_attempt": [1, 1, 1, 1], "qb_kneel": [1, 1, 0, 0], "rushing_yards": [-1, -1, 6, 4]})
    k = PD.kneels(p, ["game_id", "posteam"])
    assert len(k) == 1 and k.iloc[0]["player_id"] == "qb"
    assert k.iloc[0]["kneels"] == 2 and k.iloc[0]["kneel_yds"] == -2


def test_prop_grading_columns_and_parlay_leg_use_the_official_line():
    assert PLOG.STAT_COLS["rush_yds"] == ["rush_yds", "kneel_yds"] and PLOG.STAT_COLS["carries"] == ["carries", "kneels"]
    arch = {"results": {"g": {"home_score": 20, "away_score": 17}}, "games": {"g": {"home": "DEN", "away": "KC"}},
            "actuals": {"g": {"qb": {"rush_yds": 5.0, "carries": 3.0, "kneels": 1.0, "kneel_yds": -1.0},
                              "old": {"rush_yds": 5.0, "carries": 3.0}}}}
    leg = {"g": "g", "m": "prop", "s": "rush_yds", "L": 4.5, "d": "over", "p": "qb"}
    assert PL.leg_result(leg, arch) == "L"                                      # 5 on the board's line, 4 officially
    assert PL.leg_result({**leg, "p": "old"}, arch) == "W"                      # a week frozen before the kneel fields
    assert PL.leg_result({**leg, "s": "carries", "L": 3.5}, arch) == "W"


# ---- expected kneel-downs in the quarterback's rush-attempts prop (nflmodel/kneels.py) --------

import numpy as np  # noqa: E402

from nflmodel import kneels as KN, propboard as PB, props as P  # noqa: E402
from nflmodel.propdist import fit_count  # noqa: E402


def _starts(n=300, seasons=(2020, 2021)):
    rng = np.random.default_rng(0)
    rows = []
    for s in seasons:
        margin = rng.normal(0, 6, n)
        wp = 0.5 + np.clip(margin / 27, -0.45, 0.45)
        rows.append(pd.DataFrame({"season": s, "game_id": [f"{s}_{i}" for i in range(n)], "player_id": "qb", "group": "QB", "att_share": 1.0,
                                  "started": 1, "played": True, "margin": margin, "kneels": rng.poisson(0.2 + 0.9 * wp)}))
    return pd.concat(rows, ignore_index=True)


def test_expected_kneels_are_walk_forward_and_for_the_starter_only():
    t = _starts()
    t = pd.concat([t, t.iloc[:5].assign(season=2021, group="RB", player_id="rb"),
                   t.iloc[:5].assign(season=2021, att_share=0.0, player_id="qb2")], ignore_index=True)
    out = KN.project(t)
    assert (out.loc[out.season == 2020, "kneels_proj"] == 0).all()          # no earlier season to fit on
    s21 = out[(out.season == 2021) & (out.player_id == "qb")]
    assert s21["kneels_proj"].between(0.0, 1.5).all() and 0.5 < s21["kneels_proj"].mean() < 0.8
    assert np.corrcoef(s21["kneels_proj"], s21["margin"])[0, 1] > 0.95       # more when his team is favoured
    assert (out.loc[out.player_id.isin(["rb", "qb2"]), "kneels_proj"] == 0).all()
    changed = t.copy()
    changed.loc[changed.season == 2021, "kneels"] = 9                        # this season's results cannot move this season's number
    assert np.allclose(KN.project(changed).loc[lambda d: d.season == 2021, "kneels_proj"], out.loc[out.season == 2021, "kneels_proj"])


def test_only_the_quarterback_rush_attempts_prop_reads_the_kneels():
    rng = np.random.default_rng(1)
    n = 400
    mu = rng.uniform(2, 7, n)
    d = pd.DataFrame({"carries_proj": mu, "rush_yds_proj": mu * 4.5, "car_share": 0.1, "kneels_proj": 0.7, "group": "QB"})
    m = P.PropModel({})
    m.counts[("carries", "QB")] = fit_count(rng.poisson(mu), mu, np.zeros(n))
    lines = np.full(n, 4.5)
    before = m.probs(d, "carries", "QB", lines)
    m.counts[P.OFFICIAL_KEY] = fit_count(rng.poisson(mu + 0.7), mu + 0.7, np.zeros(n))
    after = m.probs(d, "carries", "QB", lines)
    assert after[0].mean() > before[0].mean() + 0.03                         # overs more likely with the kneels in
    assert np.allclose(m.probs(d.drop(columns="kneels_proj"), "carries", "QB", lines)[0], before[0])   # no column, no change
    assert np.allclose(P.mean(d, "carries"), mu) and np.allclose(P.line_mean(d, "carries", "QB", m), mu + 0.7)
    assert np.allclose(P.line_mean(d.assign(group="RB"), "carries", "RB", m), mu)                    # backs never
    # the rushing-yards volume still comes from the plain carries fit and mean
    assert np.allclose(m._vol_pmf(d, "rush_yds", "QB"), m.counts[("carries", "QB")].pmf(np.maximum(mu, 0.05), np.zeros(n), P.SPECS["carries"].kmax))


def test_props_rows_are_labelled_from_the_cutover_only():
    base = {"game_id": "g", "player_id": "p", "player": "A QB", "team": "DEN", "pos": "QB", "stat": "carries", "market": "player_rush_attempts",
            "line": 5.5, "proj": 5.0, "p_over": 0.4, "p_under": 0.6, "p_push": 0.0, "fair_over": 0.5, "edge_over": 0, "edge_under": 0,
            "ev_over": 0, "ev_under": 0, "books": "{}", "flag_over": False, "flag_under": False, "why_not": "", "snapshot": "flag", "commence": ""}
    assert PB._rec({**base, "time": "2026-11-12 02:00:00"})["sl"] == PB.KNEELS_LABEL
    assert "sl" not in PB._rec({**base, "time": "2026-10-28 02:00:00"})
    assert "sl" not in PB._rec({**base, "time": "2026-11-12 02:00:00", "pos": "RB"})


# ---- a starting quarterback's rushing from his starts (players.QB_RUSH) ----------------------

from nflmodel import players as PLY  # noqa: E402


def test_calib_X_per_start_only_changes_the_starter_and_only_when_asked():
    p = pd.DataFrame({"carries_p": [1.0, 10.0], "carries_qp": [3.0, np.nan], "carries_sum": [2.0, 40.0], "one_sum": [8.0, 8.0],
                      "carries_st_sum": [6.0, np.nan], "start_sum": [2.0, 0.0], "att_share": [1.0, 0.0]})
    plain = PLY.calib_X(p, "carries")
    assert np.allclose(plain[:, 0], [1.0, 10.0]) and np.allclose(plain[:, 1], [0.25, 5.0])
    ps = PLY.calib_X(p, "carries", per_start=True)
    assert np.allclose(ps[0], [3.0, 3.0])                 # the starter: per-start chain and per-start form
    assert np.allclose(ps[1], plain[1])                   # the back: unchanged
    few = p.assign(start_sum=[0.2, 0.0])
    assert np.allclose(PLY.calib_X(few, "carries", per_start=True)[0], [3.0, 3.0])   # under 0.3 starts: the chain as form
