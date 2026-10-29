"""Market lines (nflmodel/market.py) and Kalshi fee math (nflmodel/odds.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflmodel import market as M, odds as O


def _event(eid, home, away, commence, books):
    return {"id": eid, "commence_time": commence, "home_team": home, "away_team": away,
            "bookmakers": [{"key": bk, "markets": [
                {"key": "spreads", "last_update": "2026-10-29T12:00:00Z", "outcomes": [
                    {"name": home, "price": sp[1], "point": sp[0]}, {"name": away, "price": sp[2], "point": -sp[0]}]},
                {"key": "totals", "last_update": "2026-10-29T12:00:00Z", "outcomes": [
                    {"name": "Over", "price": to[1], "point": to[0]}, {"name": "Under", "price": to[2], "point": to[0]}]},
                {"key": "h2h", "last_update": "2026-10-29T12:00:00Z", "outcomes": [
                    {"name": home, "price": ml[0]}, {"name": away, "price": ml[1]}]},
            ]} for bk, (sp, to, ml) in books.items()]}


EVENTS = [
    _event("e1", "Kansas City Chiefs", "Denver Broncos", "2026-11-01T17:00:00Z",
           {"pinnacle": ((-3.0, -108, -102), (41.5, -105, -105), (-165, 150)),
            "draftkings": ((-2.5, -115, -105), (41.5, -110, -110), (-160, 135)),
            "kalshi": ((-3.0, -110, -110), (42.5, -115, -105), (-150, 130))}),
    _event("e2", "Ohio State Buckeyes", "Penn State Nittany Lions", "2026-11-01T19:30:00Z",
           {"pinnacle": ((-7.5, -110, -110), (49.5, -110, -110), (-300, 250))}),
]


def test_flatten_and_latest_follow_the_boards_conventions():
    rows = M.flatten(EVENTS[:1], "nfl", "2026-10-29 12:00")
    assert len(rows) == 18 and set(rows["side"]) == {"home", "away", "over", "under"}
    games = pd.DataFrame({"game_id": ["2026_09_DEN_KC"], "home_team": ["KC"], "away_team": ["DEN"], "gameday": ["2026-11-01"]})
    m = M.match_games(rows, games, "nfl")
    assert (m["game_id"] == "2026_09_DEN_KC").all()
    q = M.latest(m)["2026_09_DEN_KC"]
    assert q["pinnacle"]["spread_line"] == 3.0 and q["pinnacle"]["total_line"] == 41.5      # positive = home favoured
    assert q["draftkings"]["spread_line"] == 2.5 and q["kalshi"]["total_line"] == 42.5
    assert q["pinnacle"]["home_moneyline"] == -165 and q["pinnacle"]["under_odds"] == -105


def test_match_handles_college_names_and_swapped_sides():
    rows = M.flatten(EVENTS[1:], "cfb", "2026-10-29 12:00")
    games = pd.DataFrame({"game_id": [401], "home_team": ["Penn State"], "away_team": ["Ohio State"],
                          "kickoff": pd.to_datetime(["2026-11-01T19:30:00Z"], utc=True)})
    names = {"Ohio State Buckeyes": "Ohio State", "Penn State Nittany Lions": "Penn State"}
    m = M.match_games(rows, games, "cfb", names)
    assert (m["game_id"] == 401).all()
    q = M.latest(m)[401]["pinnacle"]
    assert q["spread_line"] == -7.5           # the book's home (Ohio State) is the board's away, so the board's home is +7.5
    assert q["home_moneyline"] == 250 and q["away_moneyline"] == -300
    assert M.unmatched(M.match_games(rows, games, "cfb", {})) == ["Ohio State Buckeyes", "Penn State Nittany Lions"]


def test_store_keeps_only_changes(tmp_path):
    rows = M.flatten(EVENTS[:1], "nfl", "2026-10-29 12:00")
    assert M.store(rows, "nfl", tmp_path) == 18
    assert M.store(rows.assign(snapshot="2026-10-29 13:00"), "nfl", tmp_path) == 0
    moved = rows.copy()
    moved["snapshot"] = "2026-10-29 14:00"
    moved.loc[(moved["book"] == "draftkings") & (moved["market"] == "totals"), "line"] = 42.0
    assert M.store(moved, "nfl", tmp_path) == 2
    stored = M.load("nfl", tmp_path)
    assert len(stored) == 20 and stored["snapshot"].max() == "2026-10-29 14:00"
    games = pd.DataFrame({"game_id": ["2026_09_DEN_KC"], "home_team": ["KC"], "away_team": ["DEN"], "gameday": ["2026-11-01"]})
    q = M.latest(M.match_games(stored, games, "nfl"))["2026_09_DEN_KC"]
    assert q["draftkings"]["total_line"] == 42.0 and q["pinnacle"]["total_line"] == 41.5
    first = M.first_seen(M.match_games(stored, games, "nfl"), "draftkings")["2026_09_DEN_KC"]
    assert first["total"] == 41.5 and first["spread"] == 2.5


def test_anchor_uses_pinnacle_when_complete():
    df = pd.DataFrame({"game_id": ["g1", "g2"], "spread_line": [2.5, 1.0], "total_line": [44.5, 40.0],
                       "home_spread_odds": [-110.0, -110.0], "away_spread_odds": [-110.0, -110.0],
                       "over_odds": [-110.0, -110.0], "under_odds": [-110.0, -110.0],
                       "home_moneyline": [-130.0, np.nan], "away_moneyline": [110.0, np.nan]})
    quotes = {"g1": {"pinnacle": {"spread_line": 3.0, "home_spread_odds": -104, "away_spread_odds": -106,
                                  "total_line": 45.0, "over_odds": -108, "under_odds": -102, "home_moneyline": -150, "away_moneyline": 135}},
              "g2": {"pinnacle": {"spread_line": 1.5, "home_spread_odds": -110, "away_spread_odds": -110}}}   # no total: not used
    out, src = M.anchor(df, quotes)
    assert list(src) == ["pinnacle", "consensus"]
    assert out.loc[0, "spread_line"] == 3.0 and out.loc[0, "total_line"] == 45.0 and out.loc[0, "home_moneyline"] == -150
    assert out.loc[1, "spread_line"] == 1.0


def test_kalshi_fee_lowers_the_price():
    assert abs(O.kalshi_fee(0.5) - 0.0175) < 1e-9 and abs(O.kalshi_fee(0.5, maker=True) - 0.004375) < 1e-9
    eff = float(O.kalshi_effective_odds(-150))
    assert -165 < eff < -155                      # 60c + 1.68c fee is about 61.7c, -161
    assert abs(float(O.kalshi_effective_odds(-150, maker=True)) - (-150)) < 3.5
    assert abs(float(O.prob_to_american(0.6)) - (-150)) < 1e-9 and abs(float(O.prob_to_american(0.4)) - 150) < 1e-9
    # a fair 50% side at +100 is -EV as a taker, barely so as a maker
    assert O.expected_value(0.5, O.kalshi_effective_odds(100)) < 0
    assert O.expected_value(0.5, O.kalshi_effective_odds(100, maker=True)) < 0
    assert O.expected_value(0.5, O.kalshi_effective_odds(100, maker=True)) > O.expected_value(0.5, O.kalshi_effective_odds(100))


def test_shop_scores_each_book_at_its_own_line():
    from nflmodel import bets, distributions as D
    from tests.test_model import DIST, _one_game
    g = _one_game(proj_margin=5.0, proj_total=44.0)          # line: home -3, total 44 (see test_model)
    quotes = {g["game_id"].iloc[0]: {
        "draftkings": {"spread_line": 3.0, "home_spread_odds": -110, "away_spread_odds": -110, "total_line": 44.0, "over_odds": -110, "under_odds": -110},
        "fanduel": {"spread_line": 2.5, "home_spread_odds": -115, "away_spread_odds": -105, "total_line": 44.5, "over_odds": -108, "under_odds": -112},
        "kalshi": {"spread_line": 3.0, "home_spread_odds": -110, "away_spread_odds": -110, "home_moneyline": -150, "away_moneyline": 130}}}
    s = bets.shop(g, DIST, quotes, ["draftkings", "fanduel", "kalshi"], "proj")
    gid = g["game_id"].iloc[0]
    home = {q["book"]: q for q in s[(gid, "Spread", "home")]}
    assert home["fanduel"]["prob"] > home["draftkings"]["prob"]            # -2.5 covers more often than -3
    assert home["fanduel"]["line"] == -2.5 and home["draftkings"]["line"] == -3.0
    assert home["kalshi"]["odds_net"] < -110 and home["kalshi"]["ev"] < home["draftkings"]["ev"]   # the fee
    assert s[(gid, "Spread", "home")][0]["ev"] >= s[(gid, "Spread", "home")][-1]["ev"]
    # the anchor's own price has no edge against itself; a better line at the same price does
    assert abs(home["draftkings"]["ev_line"] - bets.O.expected_value(0.5, -110)) < 0.02
    assert home["fanduel"]["ev_line"] > home["draftkings"]["ev_line"]
    assert "ev_line" in s[(gid, "Moneyline", "home")][0] and s[(gid, "Moneyline", "home")][0]["book"] == "kalshi"


def test_slate_with_market_anchors_and_reblends():
    from nflmodel import slate
    from tests.test_model import _one_game
    g = _one_game(proj_margin=4.0, proj_total=45.0)
    g["model_margin"], g["model_total"], g["w_margin"], g["w_total"] = 8.0, 50.0, 0.2, 0.05
    gid = g["game_id"].iloc[0]
    quotes = {gid: {"pinnacle": {"spread_line": 2.5, "home_spread_odds": -105, "away_spread_odds": -105,
                                 "total_line": 43.5, "over_odds": -110, "under_odds": -110}}}
    out = slate.with_market(g, quotes)
    assert out["anchor"].iloc[0] == "pinnacle" and out["spread_line"].iloc[0] == 2.5
    assert abs(out["proj_margin"].iloc[0] - (2.5 + 0.2 * (8.0 - 2.5))) < 1e-9
    assert abs(out["proj_total"].iloc[0] - (43.5 + 0.05 * (50.0 - 43.5))) < 1e-9
    none = slate.with_market(g, {})
    assert none["anchor"].iloc[0] == "consensus" and none["spread_line"].iloc[0] == g["spread_line"].iloc[0]


def test_closing_value_measures_against_the_sharp_close():
    from nflmodel import archive, bets
    from tests.test_model import DIST
    g = {"id": "g", "market": {"spread": 3.0, "total": 44.5}}
    curve = lambda df: bets.curves(df, DIST, "line")
    close = {"spread_line": 4.0, "home_spread_odds": -110, "away_spread_odds": -110, "total_line": 44.5, "over_odds": -110, "under_odds": -110,
             "home_moneyline": -180, "away_moneyline": 160}
    # bet home -3 at -110 when the close is -4: positive closing line value
    home = archive.closing_value({"market": "Spread", "side": "home", "odds": -110}, g, close, curve)
    away = archive.closing_value({"market": "Spread", "side": "away", "odds": -110}, g, close, curve)
    assert home["clv"] > 0 > away["clv"] and home["close_line"] == 4.0
    # the best book's line and price are what count when they are there
    better = archive.closing_value({"market": "Spread", "side": "home", "odds": -110, "best_line": -2.5, "best_odds": -105, "best_odds_net": -105}, g, close, curve)
    assert better["clv"] > home["clv"]
    assert archive.closing_value({"market": "Total", "side": "over", "odds": -110}, g, close, curve)["clv"] < 0   # the vig, no move
    assert archive.closing_value({"market": "Moneyline", "side": "home", "odds": -160}, g, close, curve)["clv"] > 0
    assert archive.closing_value({"market": "Spread", "side": "home", "odds": -110}, g, {"total_line": 44.5}, curve) == {}
    assert archive.closing_value({"market": "Spread", "side": "home", "odds": -110}, g, None, curve) == {}
