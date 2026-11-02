"""College prop backfill (nflmodel/cfbpropbackfill.py) against a stand-in API."""
import pandas as pd

from nflmodel import cfbpropbackfill as B


class Fake:
    def __init__(self):
        self.remaining, self.last_cost, self.odds_calls, self.list_calls = 10_000, 0, [], 0

    def historical_events(self, date, frm, to, sport=None):
        self.list_calls += 1
        assert sport == B.SPORT
        return [{"id": f"ev{i}", "commence_time": "2025-11-01T20:00:00Z", "home_team": f"H{i}", "away_team": f"A{i}"} for i in range(3)]

    def historical_event_odds(self, eid, markets, date, sport=None):
        self.odds_calls.append((eid, date))
        has = eid != "ev2"
        self.last_cost = 10 * 2 if has else 0
        if not has:
            return {}
        return {"id": eid, "home_team": "H", "away_team": "A", "commence_time": "2025-11-01T20:00:00Z", "_snapshot": date,
                "bookmakers": [{"key": "draftkings", "markets": [{"key": "player_pass_yds", "last_update": date,
                    "outcomes": [{"name": "Over", "description": "QB One", "point": 230.5, "price": -110},
                                 {"name": "Under", "description": "QB One", "point": 230.5, "price": -110}]}]}]}


def test_one_week_fetches_each_event_once_and_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "HIST", tmp_path)
    monkeypatch.setattr(B, "weeks", lambda s: [(10, pd.Timestamp("2025-10-30T23:00Z"), pd.Timestamp("2025-11-02T03:00Z"),
                                                 pd.Timestamp("2025-11-01T12:00Z"))])
    c = Fake()
    res = B.run(c, 2025, 1000, verbose=False)
    assert res["complete"] and res["fetched"] == 3 and res["empty"] == 1
    assert c.list_calls == 2 and res["spent"] == 2 + 40
    assert all(d == "2025-11-01T19:50:00Z" for _, d in c.odds_calls)        # ten minutes before kickoff
    d = tmp_path / "2025"
    assert len(pd.read_csv(d / "ev0.csv")) == 2 and pd.read_csv(d / "ev2.csv").empty
    c2 = Fake()                                                              # a second run spends nothing
    assert B.run(c2, 2025, 1000, verbose=False)["spent"] == 0 and not c2.odds_calls and c2.list_calls == 0


def test_stops_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "HIST", tmp_path)
    monkeypatch.setattr(B, "weeks", lambda s: [(10, pd.Timestamp("2025-10-30T23:00Z"), pd.Timestamp("2025-11-02T03:00Z"),
                                                 pd.Timestamp("2025-11-01T12:00Z"))])
    res = B.run(Fake(), 2025, 2 + 80 + 5, verbose=False)                   # room for one event at the 80-credit worst case
    assert not res["complete"] and res["fetched"] == 1
