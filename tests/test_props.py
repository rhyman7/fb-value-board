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


def test_price_flags_a_clear_edge_and_sizes_the_stake(monkeypatch):
    monkeypatch.setattr(PM, "FLAGS_ON", True)
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


def test_paused_flags_price_everything_and_flag_nothing():
    board = _toy_board()
    o = PM.match(_toy_odds(line=60.5), board, [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])
    b = PM.price(PM.consensus(o), board, _toy_model())
    assert len(b) == 2 and not b["flagged"].any() and (b["stake"] == 0).all()
    assert b["why_not"].str.contains("paused").all()


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
    monkeypatch.setattr(PM, "FLAGS_ON", True)
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


# ---- no look-ahead -----------------------------------------------------------------

def test_no_look_ahead_in_probabilities_or_history():
    """Nothing used for a game may come from after its kickoff: the probabilities depend only
    on pre-game projection columns, and the recent-games baseline only on earlier games."""
    m = _toy_model()
    d = _toy_board().assign(rec_yds=12.0, receptions=1.0)
    a = m.probs(d, "rec_yds", "WR", np.array([60.5]), calibrate=False)
    b = m.probs(d.assign(rec_yds=150.0, receptions=11.0), "rec_yds", "WR", np.array([60.5]), calibrate=False)
    assert np.allclose(np.r_[a], np.r_[b])                         # the outcome changes nothing
    h = pd.DataFrame({"player_id": ["p"] * 4, "game_id": list("abcd"), "t": [1, 2, 3, 4], "played": True,
                      "active": 1, "started": 1, "rec_yds": [10.0, 20.0, 30.0, 40.0]})
    lags = P.recent_history(h, "rec_yds").set_index("game_id")
    assert np.isnan(lags.loc["a", "lag1"])                         # first game: no history
    assert lags.loc["c", "lag1"] == 20.0 and lags.loc["c", "lag2"] == 10.0 and np.isnan(lags.loc["c", "lag3"])


def test_games_already_started_are_never_priced(tmp_path, monkeypatch):
    import run_props
    past = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "events.json").write_text(json.dumps([{"id": "old", "commence_time": past, "home_team": "x", "away_team": "y"}]))
    called = []

    class F(run_props.Fixture):
        def event_odds(self, *a, **k):
            called.append(a)
            return {}
    for name in ("LOG", "RESULTS", "SNAPS"):
        monkeypatch.setattr(PLOG, name, tmp_path / Path(getattr(PLOG, name)).name)
    if P.PropModel.load() is None:
        pytest.skip("no fitted prop parameters yet")
    run_props.cmd_flag(F(tmp_path), log_it=False)
    run_props.cmd_close(F(tmp_path), log_it=False)
    assert not called


# ---- same-game combinations --------------------------------------------------------

def test_bivariate_normal_matches_scipy():
    from scipy import stats
    from nflmodel import propsim as S
    for a, b, r in [(0.3, -0.2, 0.4), (-1.0, 0.5, -0.3), (1.2, 1.5, 0.8)]:
        ref = stats.multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([-a, -b])
        assert abs(float(S.bvn_upper(np.array([a]), np.array([b]), r)[0]) - ref) < 1e-6


def test_combo_keeps_each_leg_and_raises_the_joint_chance():
    from nflmodel import propsim as S
    m = _toy_model()
    m.meta["correlation"] = {"pairs": {"self|rec_yds|receptions": [0.7, 1000]}}
    pl = _toy_board()
    pl["role"] = "R1"
    out = S.combo_prob(pl, m, [(0, "rec_yds", 70.5, "over"), (0, "receptions", 5.5, "over")], n=60000)
    single = [float(m.probs(pl, s, "WR", np.array([l]))[0][0]) for s, l in (("rec_yds", 70.5), ("receptions", 5.5))]
    assert np.allclose(out["legs"], single)
    assert out["joint"] > out["independent"] + 0.05
    assert out["joint"] <= min(single) + 0.01
    opposite = S.combo_prob(pl, m, [(0, "rec_yds", 70.5, "over"), (0, "receptions", 5.5, "under")], n=60000)
    assert opposite["joint"] < opposite["independent"]


# ---- historical backfill -----------------------------------------------------------

def test_backfill_snapshot_times_budget_and_restart(tmp_path, monkeypatch):
    from nflmodel import propbackfill as B
    kick = pd.Timestamp("2025-09-07T17:00:00Z")
    assert B.flag_time(kick) == pd.Timestamp("2025-09-05T12:47:00Z")          # Friday: the first daily run under 3 days out
    sched = pd.DataFrame({"game_id": ["2025_01_DAL_PHI", "2025_01_KC_LAC"], "season": 2025, "week": 1,
                          "home_team": ["PHI", "LAC"], "away_team": ["DAL", "KC"], "kickoff": [kick, kick + pd.Timedelta(hours=3)]})
    monkeypatch.setattr(B, "schedule", lambda s: sched)
    monkeypatch.setattr(B, "HIST", tmp_path)

    class Fake:
        remaining, last_cost, used_this_run = 10000, 0, 0
        calls = []

        def historical_events(self, date, a, b):
            self.calls.append(("events", date))
            return [{"id": "e1", "home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys", "commence_time": "2025-09-07T17:00:00Z"},
                    {"id": "e2", "home_team": "Los Angeles Chargers", "away_team": "Kansas City Chiefs", "commence_time": "2025-09-07T20:00:00Z"}]

        def historical_event_odds(self, eid, markets, date):
            self.calls.append((eid, date))
            self.last_cost = 30
            return {"id": eid, "_snapshot": date, "bookmakers": [{"key": "draftkings", "markets": [{"key": "player_pass_yds",
                    "outcomes": [{"name": "Over", "description": "X", "price": -110, "point": 230.5}]}]}]}

    f = Fake()
    r = B.run(f, 2025, B.YARDAGE, max_credits=1 + 30 * 3, verbose=False)
    assert not r["complete"] and r["snapshots"] == 3 and r["spent"] == 91        # stops at the cap
    assert ("e1", "2025-09-07T16:50:00Z") in f.calls                              # close: ten minutes before kickoff
    r = B.run(f, 2025, B.YARDAGE, max_credits=100, verbose=False)
    assert r["complete"] and r["snapshots"] == 1                                  # restart fetches only what's missing
    assert len(B.load(2025, "close")) == 2 and set(B.load(2025, "flag")["game_id"]) == set(sched["game_id"])


def test_pregame_availability_ignores_the_weeks_news():
    """A pregame projection may not know this week's injury report or inactives: only the
    roster status the week before."""
    from nflmodel import players as PL
    avail = pd.DataFrame({"season": 2025, "week": [1, 2, 1, 2, 1, 2], "player_id": ["a", "a", "b", "b", "c", "c"],
                          "status": ["ACT", "INA", "ACT", "ACT", "RES", "ACT"]})
    rows = pd.DataFrame({"season": 2025, "week": 2, "player_id": ["a", "b", "c", "d"], "t": PL._t(2025, 2),
                         "status": ["INA", "ACT", "ACT", np.nan], "report": [None, "Out", None, None],
                         "active": [0, 0, 1, 1]})
    pre = PL.pregame_active(rows, avail)
    assert list(pre) == [1, 1, 0, 1]     # a: scratched on game day; b: ruled out late; c: on IR the week before; d: no listing


def test_alternate_ladder_rungs_are_priced_from_the_same_distribution():
    """A rung posted over-only (e.g. 40+ receiving yards at -400) gets our chance from the
    same distribution as the main line, and a fair price under the assumed ladder margin."""
    board = _toy_board()
    rows = []
    for b in ("draftkings", "fanduel"):
        for line, price in ((39.5, -400), (59.5, -150), (79.5, 120)):
            rows.append({"event_id": "e1", "commence": "", "home_name": "Philadelphia Eagles", "away_name": "Dallas Cowboys",
                         "book": b, "market": "player_reception_yds_alternate", "updated": "", "player": "AJ Brown",
                         "side": "over", "line": line, "price": float(price)})
    o = PM.match(pd.DataFrame(rows), board, [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])
    b = PM.price(PM.consensus(o), board, _toy_model())
    assert set(b["line"]) == {39.5, 59.5, 79.5} and (b["side"] == "over").all() and b["one_sided"].all()
    main = PM.price(PM.consensus(PM.match(_toy_odds(line=59.5), board, [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}])), board, _toy_model())
    assert abs(b.loc[b.line == 59.5, "our_p"].iloc[0] - main.loc[main.side == "over", "our_p"].iloc[0]) < 1e-12
    p = b.sort_values("line")["our_p"].to_numpy()
    assert p[0] > p[1] > p[2]                        # lower rungs are likelier
    assert np.allclose(b["fair"], O.one_sided_fair(b["price"], PM.ALT_MARGIN, "power"))


# ---- the board's Props section ------------------------------------------------------

def test_board_week_files_from_the_log(tmp_path, monkeypatch):
    from nflmodel import propboard as PB
    monkeypatch.setattr(PM, "FLAGS_ON", True)
    for name in ("LOG", "RESULTS", "SNAPS", "WEEKLY", "SUMMARY"):
        monkeypatch.setattr(PLOG, name, tmp_path / Path(getattr(PLOG, name)).name)
    board = _toy_board()
    games = [{"id": "2026_09_DAL_PHI", "home": "PHI", "away": "DAL"}]
    b = PM.price(PM.consensus(PM.match(_toy_odds(line=60.5), board, games)), board, _toy_model())
    b["event_id"], b["commence"] = "e1", "2026-11-01T18:00:00Z"
    PLOG.append_priced(b.assign(flagged=False, stake=0.0), "open", 2026, 9)
    PLOG.append_priced(b, "flag", 2026, 9)
    PLOG.append_priced(b.assign(line=62.5), "close", 2026, 9)
    pg = pd.DataFrame({"game_id": ["2026_09_DAL_PHI"], "player_id": ["00-1"], "rec_yds": [77.0], "receptions": [6.0], "rec_td": [0], "rush_td": [0]})
    av = pd.DataFrame({"season": [2026], "week": [9], "team": ["PHI"], "player_id": ["00-1"], "active": [1]})
    monkeypatch.setattr(PLOG, "outcomes", lambda seasons: (pg, av))
    PLOG.grade()
    ix = PB.build(tmp_path / "props")
    assert [w["file"] for w in ix["weeks"]] == ["2026_w09.json"]
    wk = json.loads((tmp_path / "props" / "2026_w09.json").read_text())
    (p,) = [x for x in wk["props"] if x["s"] == "rec_yds"]
    assert p["snap"] == "flag" and p["L"] == 60.5 and p["cL"] == 62.5       # the pricing snapshot, with the close
    assert p["res"] == "win" and p["y"] == 77.0 and p["fl"] == "over"
    assert abs(p["po"] + p["pu"] + (p["pp"] or 0) - 1) < 1e-6
    assert wk["summary"]["moved_toward"] == 1.0 and wk["summary"]["flag_record"] == "1-0-0"


def test_board_files_are_strict_json(tmp_path):
    """The page's JSON.parse rejects NaN, so every missing value must be null."""
    from nflmodel import propboard as PB
    PB._dump(tmp_path / "x.json", {"a": float("nan"), "b": [np.nan, 1.5, pd.NaT], "c": {"d": np.float64("nan")}})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": [None, 1.5, None], "c": {"d": None}}
