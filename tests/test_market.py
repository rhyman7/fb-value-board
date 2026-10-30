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
