"""The Parlays page's legs (nflmodel/parlays.py): push math, the haircut, prop guards."""
import numpy as np

from nflmodel import parlays as X


def _game():
    # home favoured by 3: P(home margin > x) on the projection and the market's curve
    lines = np.arange(-3.5, 10.0, 1.0)
    proj = {x: float(np.clip(0.62 - 0.05 * (x - 0.5), 0.02, 0.98)) for x in lines}
    sp = [{"line": float(x), "home": proj[x] - 0.02, "away": 1 - proj[x] + 0.02, "home_proj": proj[x], "away_proj": 1 - proj[x]} for x in lines]
    to = [{"line": float(x), "over": 0.5, "under": 0.5, "over_proj": 0.5, "under_proj": 0.5} for x in np.arange(40.5, 50.0, 1.0)]
    return {"id": "2026_09_AAA_BBB", "kickoff": "2099-01-01T18:00Z", "home": "BBB", "away": "AAA",
            "market": {"spread": 3.0, "total": 45.5}, "win": {"adj": 0.62, "market": 0.60},
            "books": {"draftkings": {"spread_line": 3.0, "home_spread_odds": -110, "away_spread_odds": -110,
                                     "total_line": 45.5, "over_odds": -110, "under_odds": -110,
                                     "home_moneyline": -150, "away_moneyline": 130}},
            "alts": {"spread": sp, "total": to}}


def test_whole_number_spread_has_push():
    legs = X.game_legs(_game(), "nfl", haircut=1.0)
    home3 = [l for l in legs if l["m"] == "spread" and l["lab"] == "BBB -3"][0]
    # wins if margin > 3 (= the 3.5 line), pushes on exactly 3 (the 2.5 line minus the 3.5 line)
    g = {r["line"]: r["home_proj"] for r in _game()["alts"]["spread"]}
    assert abs(home3["w"] - g[3.5]) < 1e-4
    assert abs(home3["h"] - (g[2.5] - g[3.5])) < 1e-4
    assert home3["bk"] == {"dk": -110}
    away3 = [l for l in legs if l["m"] == "spread" and l["lab"] == "AAA +3"][0]
    assert abs(home3["w"] + home3["h"] + away3["w"] - 1) < 1e-3


def test_haircut_zero_uses_the_market():
    legs = X.game_legs(_game(), "nfl", haircut=0.0)
    ml = [l for l in legs if l["m"] == "ml" and l["c"]["tm"] == "BBB"][0]
    assert abs(ml["w"] - 0.60) < 1e-9 and abs(ml["pm"] - 0.62) < 1e-9


def test_prop_pull_is_capped():
    assert X._pull(0.40, 0.2) == X.PULL_MAX
    assert X._pull(-0.40, 0.2) == -X.PULL_MAX
    assert abs(X._pull(0.05, 0.2) - 0.01) < 1e-12


def test_rung_calibration_is_monotone_and_harsh_at_the_ends():
    p_long = X._fair_rung(1500)        # implied 6.25%
    p_short = X._fair_rung(-1000)      # implied 90.9%
    assert p_long < 1 / 16 and p_short < 10 / 11
    assert X._fair_rung(-200) < X._fair_rung(-400) < X._fair_rung(-1000)


def test_american():
    assert X.american(0.5) == -100 and X.american(0.75) == -300 and X.american(0.2) == 400


def test_alternate_ladders_reach_the_legs(tmp_path):
    """A fetched ladder is stored, matched to the board's game and priced on its legs."""
    import datetime as dt
    import pandas as pd
    from nflmodel import market as M

    class Client:
        last_cost = 0
        def events(self, sport):
            return [{"id": "ev1", "commence_time": "2099-01-01T18:00:00Z", "home_team": "Dallas Cowboys", "away_team": "Arizona Cardinals"}]
        def event_odds(self, event_id, markets, books=None, sport=None):
            self.last_cost = 2
            return {"id": "ev1", "commence_time": "2099-01-01T18:00:00Z", "home_team": "Dallas Cowboys", "away_team": "Arizona Cardinals",
                    "bookmakers": [{"key": "draftkings", "markets": [
                        {"key": "alternate_spreads", "outcomes": [{"name": "Dallas Cowboys", "price": -150, "point": -1.5},
                                                                  {"name": "Arizona Cardinals", "price": 125, "point": 1.5}]},
                        {"key": "alternate_totals", "outcomes": [{"name": "Over", "price": -200, "point": 42.5},
                                                                 {"name": "Under", "price": 160, "point": 42.5}]}]}]}

    now = dt.datetime(2098, 12, 30, tzinfo=dt.timezone.utc)
    assert M.snapshot_alternates(Client(), "nfl", now=now, root=tmp_path, days=5) == 4
    games = pd.DataFrame([{"game_id": "2026_09_ARI_DAL", "home_team": "DAL", "away_team": "ARI", "kickoff": pd.Timestamp("2099-01-01T18:00Z")}])
    q = M.alt_quotes(games, "nfl", root=tmp_path)
    assert q["2026_09_ARI_DAL"]["dk"]["spread"][1.5] == {"home": -150.0, "away": 125.0}
    assert q["2026_09_ARI_DAL"]["dk"]["total"][42.5] == {"over": -200.0, "under": 160.0}
    g = {**_game(), "home": "DAL", "away": "ARI"}
    legs = X.game_legs(g, "nfl", 1.0, q["2026_09_ARI_DAL"])
    dal = [l for l in legs if l["lab"] == "DAL -1.5"][0]
    ari = [l for l in legs if l["lab"] == "ARI +1.5"][0]
    under = [l for l in legs if l["lab"] == "Under 42.5"][0]
    assert dal["bk"] == {"dk": -150} and ari["bk"] == {"dk": 125} and under["bk"] == {"dk": 160}
    assert "moved line" not in dal["t"]
