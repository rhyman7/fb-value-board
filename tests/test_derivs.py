"""Derivative markets (nflmodel/derivs.py): team totals and first-half lines off the full-game
curves, their snapshots (market.snapshot_derivatives), tiers (sheet.annotate) and ledger rows."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nflmodel import bets as B, derivs as DV, distributions as D, ledger as LG, market as M, sheet as SH
from tests.test_model import DIST, _one_game

HDIST = {"margin": np.ones(len(D.MARGINS)), "total": np.ones(len(D.TOTALS)), "sd_margin": 9.0, "sd_total": 8.5,
         "ratio_margin": 0.55, "total_fit": (0.0, 0.5), "plain_sd": (9.0, 8.5), "n": 1000}


def test_team_pmf_combines_the_curves():
    g = _one_game(spread_line=3.0, total_line=44.5)
    cv = B.curves(g, DIST, "line")
    pts = DV.team_support(D.MARGINS, D.TOTALS)
    home, away = DV.team_pmf(cv["pm"][0], cv["pt"][0], D.MARGINS, D.TOTALS, "home"), DV.team_pmf(cv["pm"][0], cv["pt"][0], D.MARGINS, D.TOTALS, "away")
    assert home.sum() == pytest.approx(1.0) and away.sum() == pytest.approx(1.0)
    # home points centre on (total + margin) / 2, away on (total - margin) / 2
    assert (home * pts).sum() == pytest.approx((44.5 + 3.0) / 2, abs=0.6)
    assert (away * pts).sum() == pytest.approx((44.5 - 3.0) / 2, abs=0.6)
    normal = DV.team_pmf(cv["pm"][0], cv["pt"][0], D.MARGINS, D.TOTALS, "home", normal=True)
    assert abs((normal * pts).sum() - (home * pts).sum()) < 0.3
    # shifting and anchoring: a pmf moved up makes the over likelier; the anchor hits its target
    up = DV.shift(home, pts, 2.0)
    assert (up * pts).sum() == pytest.approx((home * pts).sum() + 2.0, abs=0.05)
    d = DV.anchor_pmf(home, pts, 26.5, 0.6)
    over, push, _ = D.over_push_under(DV.shift(home, pts, d)[None, :], pts, 26.5)
    assert over[0] / (1 - push[0]) == pytest.approx(0.6, abs=1e-3)


def test_fit_half_is_walk_forward_and_centred():
    rng = np.random.default_rng(3)
    n = 1200
    seasons = np.repeat(np.arange(2016, 2026), n // 10)
    spread = rng.normal(0, 6, n).round() / 2
    total = rng.normal(45, 4, n).round(1)
    games = pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "season": seasons, "played": True, "spread_line": spread, "total_line": total})
    h1m = np.round(0.5 * spread + rng.normal(0, 8, n))
    h1t = np.round(0.48 * total + rng.normal(0, 7, n))
    half = pd.DataFrame({"game_id": games.game_id, "home_h1": ((h1t + h1m) / 2).round(), "away_h1": ((h1t - h1m) / 2).round()})
    hd = DV.fit_half(games, half, 2026, lookback=10)
    assert hd is not None and 0.4 < hd["ratio_margin"] < 0.6 and 0.4 < hd["total_fit"][1] < 0.6
    assert 6.0 < hd["sd_margin"] < 12.0 and hd["n"] == n
    assert DV.fit_half(games, half, 2019, lookback=10)["n"] == 3 * n // 10     # only seasons before the one priced
    assert DV.fit_half(games, half, 2017) is None                               # one season (120 games) is too few


def _quotes(gid):
    return {gid: {
        "pinnacle": {"spread_line": 3.0, "home_spread_odds": -110, "away_spread_odds": -110, "total_line": 44.5, "over_odds": -110, "under_odds": -110,
                     "tt_home_line": 23.5, "tt_home_over_odds": -115, "tt_home_under_odds": -105,
                     "h1_spread_line": 1.5, "h1_home_spread_odds": -110, "h1_away_spread_odds": -110, "h1_total_line": 21.5, "h1_over_odds": -110, "h1_under_odds": -110},
        "draftkings": {"tt_home_line": 23.5, "tt_home_over_odds": 100, "tt_home_under_odds": -120, "tt_away_line": 20.5, "tt_away_over_odds": -110, "tt_away_under_odds": -110,
                       "h1_spread_line": 1.5, "h1_home_spread_odds": -112, "h1_away_spread_odds": -108, "h1_total_line": 22.5, "h1_over_odds": -105, "h1_under_odds": -115,
                       "h1_home_moneyline": -130, "h1_away_moneyline": 110},
        "kalshi": {"tt_home_line": 23.5, "tt_home_over_odds": -105, "tt_home_under_odds": -105}}}


def test_shop_prices_each_book_at_its_own_derivative_line():
    g = _one_game(proj_margin=6.0, proj_total=44.5)        # the model likes the home side by 3 more points
    gid = g["game_id"].iloc[0]
    s = DV.shop(g, DIST, HDIST, _quotes(gid), ["draftkings", "kalshi", "pinnacle"], "proj", haircut=0.0)
    tt = {q["book"]: q for q in s[(gid, "Team total", "home_over")]}
    # anchored to Pinnacle's own team-total quote: at Pinnacle's line the market chance is its no-vig number
    assert tt["pinnacle"]["prob_line"] == pytest.approx(B.O.no_vig(-115, -105)[0], abs=1e-3)
    assert tt["draftkings"]["prob_line"] == tt["pinnacle"]["prob_line"]                      # same line, same market chance
    assert tt["draftkings"]["ev_line"] > 0 and tt["draftkings"]["ev"] > tt["kalshi"]["ev"]   # +100 beats -105 net of a fee
    assert tt["draftkings"]["prob"] > tt["draftkings"]["prob_line"]                          # the model's extra points show up in home points
    assert s[(gid, "Team total", "away_over")][0]["ev_line"] is None                       # Pinnacle quotes no away total: no "beats"
    # first half: the half curve sits on Pinnacle's first-half line; DK's higher total makes its under likelier
    h1 = {q["book"]: q for q in s[(gid, "1H Total", "under")]}
    assert h1["pinnacle"]["prob_line"] == pytest.approx(0.5, abs=1e-3) and h1["draftkings"]["prob_line"] > 0.5
    assert h1["draftkings"]["line"] == 22.5 and s[(gid, "1H Spread", "home")][0]["line"] == -1.5
    ml = s[(gid, "1H Moneyline", "home")][0]
    assert ml["book"] == "draftkings" and ml["p_push"] > 0.02 and ml["ev_line"] is None     # a tie at the half is a push; Pinnacle quotes no 1H moneyline
    # no half curve: team totals only
    assert all(k[1] == "Team total" for k in DV.shop(g, DIST, None, _quotes(gid), ["draftkings"], "proj"))
    # the sharp price is never a place to bet, and a price beating it is the price tier
    ents = DV.entries(g, s)[gid]
    by = {e["pick"]: e for e in ents}
    e = by["HOM over 23.5"]
    assert e["deriv"] and e["odds"] == -115 and e["sharp"]["line"] == 23.5 and e["best"] == "draftkings"
    assert all(q["book"] != "pinnacle" for q in e["books"]) and e["beats_sharp"] == "draftkings" and e["sharp_ev"] > 0.01
    assert by["AWY over 20.5"]["sharp"] is None and by["AWY over 20.5"]["beats_sharp"] is None
    game = {"bets": [], "derivs": ents, "market": {"spread": 3.0, "total": 44.5}, "model": {"margin": 6.0, "total": 44.5}}
    SH.annotate(game, "nfl")
    tiers = {e["pick"]: e["tier"] for e in ents}
    assert tiers["HOM over 23.5"] == "price" and tiers["AWY over 20.5"] is None and tiers["1H HOM ML"] is None
    assert all(e["why"] == [] and e["gap"] is None for e in ents)
    # the ledger records the price bet at DraftKings and grades it on the team's points; a first-half row waits for the half-time score
    import tempfile
    from pathlib import Path
    ents2 = [dict(x) for x in ents]
    for x in ents2:
        if x["pick"] == "1H Under 21.5":
            x["tier"] = "price"
    game2 = {"id": "g", "week": 1, "anchor": "pinnacle", "bets": [], "derivs": ents2, "market": {"spread": 3.0, "total": 44.5}, "model": {"margin": 6.0, "total": 44.5}}
    with tempfile.TemporaryDirectory() as tmp:
        path, res = Path(tmp) / "ledger.csv", Path(tmp) / "res.csv"
        now = dt.datetime(2026, 10, 30, 9, tzinfo=dt.timezone.utc)
        assert LG.record([game2], 2026, {"g": pd.Timestamp("2026-11-01T18:00Z")}, path, now=now) == 2
        led = pd.read_csv(path).set_index("row_id")
        assert led.loc["g|Team total|home_over", "book"] == "draftkings" and led.loc["g|Team total|home_over", "line"] == 23.5
        assert LG.grade(path, res, {"g": (27.0, 20.0)}, {}, None) == 1                       # 27 home points: over 23.5 wins; the half waits
        assert LG.grade(path, res, {"g": (27.0, 20.0)}, {}, None, halves={"g": (10, 13)}) == 1   # 23 at the half: over 21.5, the under loses
        r = pd.read_csv(res).set_index("row_id")
        assert r.loc["g|Team total|home_over", "result"] == "W" and r.loc["g|1H Total|under", "result"] == "L" and pd.isna(r.loc["g|1H Total|under", "clv"])


def _deriv_event(eid, home, away, commence):
    return {"id": eid, "commence_time": commence, "home_team": home, "away_team": away, "bookmakers": [
        {"key": "pinnacle", "markets": [
            {"key": "team_totals", "last_update": "2026-10-30T12:00:00Z", "outcomes": [
                {"name": "Over", "description": home, "price": -115, "point": 23.5}, {"name": "Under", "description": home, "price": -105, "point": 23.5},
                {"name": "Over", "description": away, "price": -110, "point": 20.5}, {"name": "Under", "description": away, "price": -110, "point": 20.5}]},
            {"key": "spreads_h1", "last_update": "2026-10-30T12:00:00Z", "outcomes": [{"name": home, "price": -110, "point": -1.5}, {"name": away, "price": -110, "point": 1.5}]},
            {"key": "totals_h1", "last_update": "2026-10-30T12:00:00Z", "outcomes": [{"name": "Over", "price": -108, "point": 21.5}, {"name": "Under", "price": -112, "point": 21.5}]},
            {"key": "h2h_h1", "last_update": "2026-10-30T12:00:00Z", "outcomes": [{"name": home, "price": -135}, {"name": away, "price": 115}]}]}]}


def test_derivative_snapshot_rows_and_budget(tmp_path):
    evs = [_deriv_event("e1", "Kansas City Chiefs", "Denver Broncos", "2026-11-01T17:00:00Z"),
           _deriv_event("e2", "Buffalo Bills", "Baltimore Ravens", "2026-11-01T20:00:00Z"),
           _deriv_event("e3", "Dallas Cowboys", "New York Giants", "2026-11-20T20:00:00Z")]      # beyond the horizon

    class Fx:
        ready, remaining, used_this_run, last_cost = True, None, 0, 0
        calls = []
        def events(self, sport):
            return [{k: e[k] for k in ("id", "commence_time", "home_team", "away_team")} for e in evs]
        def event_odds(self, event_id, markets, books=None, sport=None):
            self.calls.append(event_id)
            self.last_cost = len(markets)
            return next(e for e in evs if e["id"] == event_id)
    now = dt.datetime(2026, 10, 30, 14, tzinfo=dt.timezone.utc)
    n = M.snapshot_derivatives(Fx(), "nfl", now=now, root=tmp_path, budget=4)        # one game's worth of credits
    rows = M.load("nfl", tmp_path)
    assert n == 10 and set(rows["event_id"]) == {"e1"} and Fx.calls == ["e1"]
    assert set(rows["market"]) == {"team_totals", "spreads_h1", "totals_h1", "h2h_h1"}
    assert set(rows.loc[rows["market"] == "team_totals", "side"]) == set(DV.TEAM_SIDES)
    Fx.calls.clear()
    assert M.snapshot_derivatives(Fx(), "nfl", now=now, root=tmp_path, budget=100) == 10 and Fx.calls == ["e1", "e2"]   # e3 is 3 weeks out; e1 unchanged
    games = pd.DataFrame({"game_id": ["2026_09_DEN_KC", "2026_09_BAL_BUF"], "home_team": ["KC", "BUF"], "away_team": ["DEN", "BAL"], "gameday": ["2026-11-01", "2026-11-01"]})
    q = M.latest(M.match_games(rows, games, "nfl"))["2026_09_DEN_KC"]["pinnacle"]
    assert q["tt_home_line"] == 23.5 and q["tt_home_over_odds"] == -115 and q["tt_away_under_odds"] == -110
    assert q["h1_spread_line"] == 1.5 and q["h1_home_spread_odds"] == -110 and q["h1_total_line"] == 21.5 and q["h1_home_moneyline"] == -135
    # a game the book lists the other way round swaps every side, team totals included
    flipped = pd.DataFrame({"game_id": ["x"], "home_team": ["DEN"], "away_team": ["KC"], "gameday": ["2026-11-01"]})
    qf = M.latest(M.match_games(rows[rows["event_id"] == "e1"], flipped, "nfl"))["x"]["pinnacle"]
    assert qf["tt_away_line"] == 23.5 and qf["tt_home_line"] == 20.5 and qf["h1_spread_line"] == -1.5
