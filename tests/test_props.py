"""Player props: distributions, line settlement, vig removal, pricing, the log."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import odds as O, propdist as D, propmarket as PM, proplog as PLOG, props as P  # noqa: E402
from nflmodel.propdist import CountFit, YardsFit  # noqa: E402


# ---- distributions ---------------------------------------------------------------

def test_count_pmf_sums_to_one_and_keeps_the_mean():
    mu = np.array([0.4, 3.0, 9.0, 33.0])
    pmf = D.count_pmf(mu, alpha=np.array([0.5, 0.2, 0.1, 0.04]), q=0.05, f=0.3, kmax=120)
    assert np.allclose(pmf.sum(1), 1)
    assert np.allclose((pmf * np.arange(121)).sum(1), mu, rtol=1e-3)


def test_poisson_limit():
    pmf = D.count_pmf(np.array([1.3]), 0.0, 0.0, 0.3, 30)[0]
    from scipy import stats
    assert np.allclose(pmf[:10], stats.poisson.pmf(np.arange(10), 1.3), atol=1e-9)


def test_compound_yards_mean_and_zero_mass():
    mu_v, ypp, k1 = np.array([4.0]), np.array([11.0]), np.array([0.8])
    vol = D.count_pmf(mu_v, 0.15, 0.03, 0.2, 30)
    x = np.arange(0, 400)[None, :] + 0.5
    F = D.compound_cdf(vol, ypp, k1, x)[0]
    assert np.all(np.diff(F) >= -1e-12)
    pmf = np.diff(np.r_[0.0, F])
    assert abs((pmf * np.arange(400)).sum() - 44.0) < 1.0           # mean = volume x yards per play
    assert F[0] >= vol[0, 0]                                         # zero plays is zero yards


def test_half_point_lines_never_push_and_whole_lines_do():
    fit = CountFit(a=np.log(0.2), b=0.0, c=0.0, q=0.0, f=0.3)
    m = P.PropModel()
    m.counts[("receptions", "WR")] = fit
    d = pd.DataFrame({"receptions_proj": [4.2, 4.2], "tgt_share": [0.2, 0.2]})
    over, push, under = m.probs(d, "receptions", "WR", np.array([4.5, 4.0]), calibrate=False)
    pmf = fit.pmf(np.array([4.2]), np.array([0.2]), 25)[0]
    assert push[0] == 0 and abs(push[1] - pmf[4]) < 1e-9
    assert abs(over[0] - pmf[5:].sum()) < 1e-9 and abs(over[1] - pmf[5:].sum()) < 1e-9
    assert np.allclose(over + push + under, 1)


def test_isotonic_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    hit = (rng.uniform(size=5000) < 0.5 + 0.6 * (p - 0.5)).astype(float)   # overconfident predictions
    iso = D.Isotonic.fit(p, hit)
    q = iso(np.linspace(0, 1, 101))
    assert np.all(np.diff(q) >= -1e-12) and q.min() >= 0 and q.max() <= 1
    assert iso(0.9) < 0.9 and iso(0.1) > 0.1


# ---- vig removal ----------------------------------------------------------------

def test_power_method():
    a, b = O.no_vig_power(-110, -110)
    assert abs(a - 0.5) < 1e-6 and abs(a + b - 1) < 1e-9
    pa, pb = O.no_vig(-400, 300)
    wa, wb = O.no_vig_power(-400, 300)
    assert abs(wa + wb - 1) < 1e-9
    assert wb < pb                       # the power method takes more of the margin from the longshot


def test_one_sided_fair():
    half = 1 / ((1 + 0.08) / 2)          # decimal price of a fair 50% side with an 8% overround
    american = -100 / (half - 1) if half < 2 else (half - 1) * 100
    assert abs(float(O.one_sided_fair(american, 0.08, "power")) - 0.5) < 1e-3
    assert abs(float(O.one_sided_fair(200, 0.08, "proportional")) - (1 / 3) / 1.08) < 1e-9


# ---- pricing ----------------------------------------------------------------------

def _toy_model():
    m = P.PropModel()
    m.counts[("receptions", "WR")] = CountFit(a=np.log(0.1), b=0.0, c=0.0, q=0.02, f=0.2)
    m.yards[("rec_yds", "WR")] = YardsFit(a=0.0, b=0.0, c=0.0)
    return m


def _toy_board():
    return pd.DataFrame([{
        "player_id": "00-1", "name": "A.J. Brown Jr.", "pos": "WR", "group": "WR", "team": "PHI", "opp": "DAL",
        "game_id": "2026_09_DAL_PHI", "week": 9, "status": None, "active": 1, "played": False, "started": np.nan,
        "receptions_proj": 6.0, "rec_yds_proj": 80.0, "tgt_share": 0.27, "car_share": 0.0, "att_share": 0.0,
        "tgt_share_hist": 0.26, "car_share_hist": 0.0, "games": 7.0}])


def _toy_odds(over_price=-110, under_price=-110, line=60.5, books=("draftkings", "fanduel")):
    rows = []
    for b in books:
        for side, pr in (("Over", over_price), ("Under", under_price)):
            rows.append({"event_id": "e1", "commence": "2026-11-01T18:00:00Z", "home_name": "Philadelphia Eagles",
                         "away_name": "Dallas Cowboys", "book": b, "market": "player_reception_yds", "updated": "",
                         "player": "AJ Brown", "side": side.lower(), "line": line, "price": float(pr)})
    return pd.DataFrame(rows)


def test_names_match_across_punctuation_and_suffixes():
    assert PM.norm_name("A.J. Brown Jr.") == PM.norm_name("AJ Brown")
    o = PM.match(_toy_odds(), _toy_board(), [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])
    assert (o["player_id"] == "00-1").all()


def test_price_flags_a_clear_edge_and_sizes_the_stake():
    board = _toy_board()
    o = PM.match(_toy_odds(line=60.5), board, [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])
    cons = PM.consensus(o)
    assert abs(cons.loc[0, "fair_power"] - 0.5) < 1e-6 and cons.loc[0, "n_books"] == 2
    b = PM.price(cons, board, _toy_model())
    over = b[b.side == "over"].iloc[0]
    assert over["our_p"] > 0.55 and over["flagged"]                # projection 80 against a 60.5 line
    assert 0 < over["stake"] <= PM.MAX_STAKE_BET
    under = b[b.side == "under"].iloc[0]
    assert not under["flagged"] and under["stake"] == 0
    ev = over["our_p"] * (O.american_to_decimal(-110) - 1) - (1 - over["our_p"])
    assert abs(over["ev"] - ev) < 1e-9


def test_unclear_role_and_thin_markets_are_not_flagged():
    board = _toy_board().assign(status="Questionable")
    o = PM.match(_toy_odds(), board, [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])
    b = PM.price(PM.consensus(o), board, _toy_model())
    assert not b["flagged"].any() and "role" in b.loc[b.side == "over", "why_not"].iloc[0]
    board = _toy_board()
    o = PM.match(_toy_odds(books=("betmgm",)), board, [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])
    b = PM.price(PM.consensus(o), board, _toy_model())
    assert "liquidity" in b.loc[b.side == "over", "why_not"].iloc[0]


def test_game_cap():
    b = pd.DataFrame({"edge": [0.1] * 6, "ev": [0.1] * 6, "n_books": [3] * 6, "sharp": [True] * 6, "hold": [0.05] * 6,
                      "price": [-110] * 6, "role_clear": [True] * 6, "kelly": [0.2] * 6, "game_id": ["g"] * 6})
    f = PM.flag(b)
    assert f["stake"].sum() <= PM.MAX_STAKE_GAME + 1e-9


# ---- the log ------------------------------------------------------------------------

def test_log_is_append_only_and_grades(tmp_path, monkeypatch):
    for name in ("LOG", "RESULTS", "SNAPS", "WEEKLY", "SUMMARY"):
        monkeypatch.setattr(PLOG, name, tmp_path / Path(getattr(PLOG, name)).name)
    board = _toy_board()
    o = PM.match(_toy_odds(), board, [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])
    b = PM.price(PM.consensus(o), board, _toy_model())
    b["event_id"] = "e1"
    PLOG.append_priced(b, "flag", 2026, 9)
    first = PLOG.LOG.read_text()
    close = b.assign(line=62.5, price=-120, fair=0.55)
    PLOG.append_priced(close, "close", 2026, 9)
    assert PLOG.LOG.read_text().startswith(first)                  # earlier rows untouched
    pg = pd.DataFrame({"game_id": ["2026_09_DAL_PHI"], "player_id": ["00-1"], "rec_yds": [77.0], "receptions": [6.0],
                       "rec_td": [0], "rush_td": [0]})
    av = pd.DataFrame({"season": [2026], "week": [9], "team": ["PHI"], "player_id": ["00-1"], "active": [1]})
    monkeypatch.setattr(PLOG, "outcomes", lambda seasons: (pg, av))
    assert PLOG.grade() == 2
    assert PLOG.grade() == 0                                       # nothing graded twice
    res = PLOG.read_results()
    log = PLOG.read_log()
    over = res.merge(log, on="row_id")
    over = over[over.side == "over"].iloc[0]
    assert over["result"] == "win" and over["clv_note"].startswith("line moved")
    s = PLOG.summarize()
    assert s["all"]["bets"] == 1 and s["all"]["record"] == "1-0-0"


# ---- the fitted parameters ------------------------------------------------------

PARAMS = Path(__file__).resolve().parent.parent / "data" / "prop_params.json"


@pytest.mark.skipif(not PARAMS.exists(), reason="no fitted prop parameters yet")
def test_saved_parameters_load_and_give_sane_probabilities():
    m = P.PropModel.load(PARAMS)
    for stat, s in P.SPECS.items():
        for g in s.groups:
            if (stat, g) not in m.counts and (stat, g) not in m.yards:
                continue
            mu = {"receptions": 4, "rec_yds": 55, "carries": 14, "rush_yds": 60, "pass_att": 34, "completions": 22,
                  "pass_yds": 240, "pass_td": 1.6, "ints": 0.8, "anytime_td": 0.4}[stat]
            d = pd.DataFrame({c: [mu / len(s.proj)] for c in s.proj})
            for c in ("tgt_share", "car_share", "att_share"):
                d[c] = 0.2
            if s.vol:
                d[SPECS_PROJ(s.vol)] = {"receptions": 4.5, "carries": 14, "completions": 22}[s.vol]
            over, push, under = m.probs(d, stat, g, np.array([np.floor(mu) + 0.5]))
            assert 0.02 < over[0] < 0.98 and push[0] == 0


def SPECS_PROJ(stat):
    return P.SPECS[stat].proj[0]


@pytest.mark.skipif(not PARAMS.exists(), reason="no fitted prop parameters yet")
def test_dry_run_on_this_weeks_board(tmp_path, monkeypatch, capsys):
    """run_props.py flag against saved odds for a real upcoming game on the board."""
    import run_props
    _, players, games, season, week = run_props.load_board()
    wr = players[(players["group"] == "WR") & (players["rec_yds_proj"] > 40)] if len(players) else players
    if wr.empty:
        pytest.skip("no upcoming receivers on the board (offseason)")
    r = wr.iloc[0]
    g = next(x for x in games if x["id"] == r["game_id"])
    kick = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev = {"id": "evt1", "commence_time": kick, "home_team": g["home_name"], "away_team": g["away_name"]}
    line = float(np.floor(r["rec_yds_proj"]) + 0.5)
    outcomes = [{"name": n, "description": r["name"], "price": -110, "point": line} for n in ("Over", "Under")]
    body = dict(ev, bookmakers=[{"key": k, "markets": [{"key": "player_reception_yds", "last_update": kick, "outcomes": outcomes}]}
                                for k in ("draftkings", "fanduel")])
    (tmp_path / "events.json").write_text(json.dumps([ev]))
    (tmp_path / "event_evt1.json").write_text(json.dumps(body))
    for name in ("LOG", "RESULTS", "SNAPS"):
        monkeypatch.setattr(PLOG, name, tmp_path / Path(getattr(PLOG, name)).name)
    run_props.cmd_flag(run_props.Fixture(tmp_path), log_it=True)
    log = PLOG.read_log()
    assert len(log) == 2 and set(log["side"]) == {"over", "under"}
    assert abs(log["our_p"].sum() + log["our_push"].iloc[0] - 1) < 1e-6
    assert abs(log["fair"].sum() - 1) < 1e-6
