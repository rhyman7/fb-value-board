"""Market lines (nflmodel/market.py) and Kalshi fee math (nflmodel/odds.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_kalshi_maker_post_price():
    """Where to rest a maker order: the highest whole cent still +EV after the 1.75% maker fee."""
    r = O.kalshi_post_at(0.55)
    assert r["post_at"] == 54 and r["post_match"] is None
    assert O.expected_value(0.55, O.kalshi_net_odds_at(54, maker=True)) >= 0 > O.expected_value(0.55, O.kalshi_net_odds_at(55, maker=True))
    assert O.kalshi_post_at(0.55, match_ev=0.03)["post_match"] == 52           # a cheaper fill to match a retail +3%
    assert O.kalshi_post_at(0.55, match_ev=-0.5)["post_match"] == 54            # never above the break-even cent
    assert O.kalshi_post_at(0.004)["post_at"] is None                            # no cent is +EV
    assert O.kalshi_post_at(0.52, 0.04)["post_at"] == 53                         # pushes set aside, as the quotes carry them
    # the maker fee is a quarter of the taker fee on the game series
    assert O.kalshi_fee(0.5, maker=True) == pytest.approx(O.kalshi_fee(0.5) / 4)


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
    # every Kalshi quote says where a resting order is worth posting, against this side's chance and the best retail price
    k = home["kalshi"]
    assert k["post_at"] is not None and k["post_at"] <= 100 * k["prob"] and k["post_match"] is not None
    assert k["post_match"] <= k["post_at"] and "post_at" not in home["draftkings"]
    ml = {q["book"]: q for q in s[(gid, "Moneyline", "home")]}["kalshi"]
    assert ml["post_match"] is None                                              # no retail price on the moneyline here


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


def test_snapshot_and_quotes_skip_games_that_kicked_off(tmp_path):
    import datetime as dt

    class Fx:
        ready, remaining, used_this_run = True, None, 0
        def sport_odds(self, sport, params):
            return EVENTS
    # e1 kicks off 2026-11-01T17:00Z: a snapshot after that keeps only e2
    n = M.snapshot(Fx(), "nfl", now=dt.datetime(2026, 11, 1, 18, 0, tzinfo=dt.timezone.utc), root=tmp_path)
    assert n == 6 and set(M.load("nfl", tmp_path)["event_id"]) == {"e2"}
    # stored quotes for a game past kickoff are not used at build time either
    M.store(M.flatten(EVENTS[:1], "nfl", "2026-10-29 12:00"), "nfl", tmp_path)
    games = pd.DataFrame({"game_id": ["2026_09_DEN_KC"], "home_team": ["KC"], "away_team": ["DEN"], "gameday": ["2026-11-01"]})
    assert M.quotes_for(games, "nfl", root=tmp_path, before=pd.Timestamp("2026-11-01T16:00Z"))["latest"]
    assert not M.quotes_for(games, "nfl", root=tmp_path, before=pd.Timestamp("2026-11-01T18:00Z"))["latest"]


# --------------------------------------------------------------------------
# Historical backfill (nflmodel/marketbackfill.py): planning and the fetch loop, no API calls
# --------------------------------------------------------------------------

def test_backfill_plan_open_and_close_times():
    from nflmodel import marketbackfill as MB
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    assert MB.open_time(ts("2025-09-05 00:20")) == ts("2025-09-01 12:00")      # Thursday night: that Monday
    assert MB.open_time(ts("2025-09-09 00:15")) == ts("2025-09-08 12:00")      # Monday night: that Monday noon
    assert MB.open_time(ts("2025-09-08 14:00")) == ts("2025-09-01 12:00")      # too close to Monday noon: the one before
    k = [ts("2025-09-07 17:00")] * 3 + [ts("2025-09-07 20:05"), ts("2025-09-07 20:25"), ts("2025-09-08 00:20")]
    assert MB.close_times(k, 30) == [ts("2025-09-07 16:50"), ts("2025-09-07 19:55"), ts("2025-09-08 00:10")]
    g = pd.DataFrame({"game_id": list("abcdef"), "kickoff": k})
    p = MB.plan(g, "nfl")
    assert list(p.kind) == ["open", "close", "close", "close"]


class _HistClient:
    """Stands in for oddsapi.Client: serves EVENTS at any date, 10 credits a market."""
    def __init__(self, fail_at=None, cost=None):
        self.remaining, self.last_cost, self.calls, self.fail_at, self.cost = 30000, 0, [], fail_at, cost

    def historical_sport_odds(self, sport, markets, date, books):
        from nflmodel import oddsapi
        self.calls.append(date)
        if date == self.fail_at:
            raise oddsapi.ApiError(f"historical game odds {sport} {date}: HTTP 404 Not Found")
        self.last_cost = self.cost or 10 * len(markets)
        self.remaining -= self.last_cost
        return {"timestamp": date, "data": EVENTS}


def test_backfill_run_saves_resumes_and_skips_404(tmp_path, monkeypatch):
    from nflmodel import marketbackfill as MB
    monkeypatch.setattr(MB, "HIST", tmp_path)
    g = pd.DataFrame({"game_id": ["a", "b"], "home_team": ["KC", "BUF"], "away_team": ["DEN", "MIA"],
                      "kickoff": pd.to_datetime(["2026-11-01T17:00:00Z", "2026-11-01T20:25:00Z"], utc=True)})
    monkeypatch.setattr(MB, "schedule", lambda league, season: g)
    plan = MB.plan(g, "nfl")
    assert len(plan) == 3                                                    # one open, two closes
    c = _HistClient(fail_at=MB._iso(plan["at"].iloc[1]))
    r = MB.run(c, "nfl", 2025, 1000, verbose=False)
    assert r["complete"] and r["spent"] == 60 and r["skipped"] == 1
    files = sorted(p.name for p in (tmp_path / "nfl" / "2025").iterdir())
    assert len(files) == 3
    empty = tmp_path / "nfl" / "2025" / MB.file_name(plan["kind"].iloc[1], plan["at"].iloc[1])
    assert empty.read_text().strip() == ",".join(M.COLS)                      # the 404 is saved empty
    c2 = _HistClient()
    assert MB.run(c2, "nfl", 2025, 1000, verbose=False)["spent"] == 0 and not c2.calls     # nothing fetched twice
    assert MB.estimate("nfl", 2025)["to_fetch"] == 0


def test_backfill_stops_at_the_cap_and_on_a_surprise_cost(tmp_path, monkeypatch):
    from nflmodel import marketbackfill as MB
    monkeypatch.setattr(MB, "HIST", tmp_path)
    g = pd.DataFrame({"game_id": ["a", "b"], "home_team": ["KC", "BUF"], "away_team": ["DEN", "MIA"],
                      "kickoff": pd.to_datetime(["2026-11-01T17:00:00Z", "2026-11-01T20:25:00Z"], utc=True)})
    monkeypatch.setattr(MB, "schedule", lambda league, season: g)
    r = MB.run(_HistClient(), "nfl", 2025, 45, verbose=False)
    assert not r["complete"] and r["spent"] == 30                            # a second 30 would pass the cap
    r = MB.run(_HistClient(cost=90), "nfl", 2026, 1000, verbose=False)
    assert not r["complete"] and r["probe_cost"] == 90                      # more than 30: stop and check


def test_backfill_load_drops_games_already_under_way(tmp_path, monkeypatch):
    from nflmodel import marketbackfill as MB
    monkeypatch.setattr(MB, "HIST", tmp_path)
    d = tmp_path / "nfl" / "2026"
    d.mkdir(parents=True)
    M.flatten(EVENTS, "nfl", "2026-11-01T18:00:00Z").to_csv(d / "close_20261101T1800.csv", index=False)
    x = MB.load("nfl", 2026)
    assert set(x["event_id"]) == {"e2"} and set(x["kind"]) == {"close"}       # e1 kicked off at 17:00


def test_backfill_adds_only_missing_markets(tmp_path):
    """A moment saved with totals only fetches spreads alone, into a second file load() merges."""
    import pandas as pd
    from nflmodel import market as M, marketbackfill as MB
    at = pd.Timestamp("2025-09-06T15:50Z")
    row = dict(zip(M.COLS, ["2025-09-06T15:45:00Z", "cfb", "e1", "2025-09-06T16:00:00Z", "A", "B", "pinnacle", "totals", "over", 50.5, -110, "x"]))
    pd.DataFrame([row])[M.COLS].to_csv(tmp_path / MB.file_name("close", at), index=False)
    assert MB.missing_markets(tmp_path, "close", at, ["spreads", "totals"]) == ["spreads"]
    t = MB.target(tmp_path, "close", at, ["spreads"])
    assert t.name == "close_20250906T1550_spreads.csv"
    pd.DataFrame([{**row, "market": "spreads", "side": "home", "line": -3.5}])[M.COLS].to_csv(t, index=False)
    assert MB.missing_markets(tmp_path, "close", at, ["spreads", "totals"]) == []
    (tmp_path / MB.file_name("open", at)).write_text(",".join(M.COLS) + "\n")      # the API had nothing
    assert MB.missing_markets(tmp_path, "open", at, ["spreads", "totals"]) == []


def test_closing_quotes_fall_back_when_pinnacle_has_none(tmp_path):
    import pandas as pd
    from nflmodel import market as M
    rows = pd.DataFrame([
        {"snapshot": "2099-01-01 12:00", "league": "nfl", "event_id": "e1", "commence": "2099-01-01T18:00:00Z", "home_name": "Dallas Cowboys",
         "away_name": "Arizona Cardinals", "book": "draftkings", "market": "spreads", "side": "home", "line": -3.5, "price": -110.0, "updated": "x"},
        {"snapshot": "2099-01-01 12:00", "league": "nfl", "event_id": "e1", "commence": "2099-01-01T18:00:00Z", "home_name": "Dallas Cowboys",
         "away_name": "Arizona Cardinals", "book": "draftkings", "market": "spreads", "side": "away", "line": 3.5, "price": -110.0, "updated": "x"}])
    M.store(rows, "nfl", tmp_path)
    games = pd.DataFrame([{"game_id": "G", "home_team": "DAL", "away_team": "ARI", "kickoff": pd.Timestamp("2099-01-01T18:00Z")}])
    q = M.closing_quotes(games, "nfl", {"G": pd.Timestamp("2099-01-01T18:00Z")}, root=tmp_path)
    assert q["G"]["close_book"] == "draftkings" and q["G"]["spread_line"] == 3.5
    assert M.closing_quotes(games, "nfl", {"G": pd.Timestamp("2099-01-01T18:00Z")}, root=tmp_path, fallback=()) == {}
