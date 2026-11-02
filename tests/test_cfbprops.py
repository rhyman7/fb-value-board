"""Live college props: the fetch (nflmodel/cfbprops.py) and the legs priced from it (parlays.cfb_book_legs)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nflmodel import cfbprops as CP, odds as O, parlays as PA

NOW = dt.datetime(2026, 11, 7, 12, 13, tzinfo=dt.timezone.utc)


class Fake:
    def __init__(self, evs, posted=("a",)):
        self.evs, self.posted, self.last_cost, self.calls = evs, posted, 0, []

    def events(self, sport):
        return self.evs

    def event_odds(self, eid, markets, sport=None):
        self.calls.append(eid)
        if eid not in self.posted:
            self.last_cost = 0
            return {"id": eid, "bookmakers": []}
        self.last_cost = 3
        return {"id": eid, "home_team": "Ohio Bobcats", "away_team": "Akron Zips", "commence_time": "2026-11-07T17:00:00Z",
                "bookmakers": [{"key": "draftkings", "markets": [{"key": "player_pass_yds", "outcomes": [
                    {"name": "Over", "description": "Nick Poulos", "point": 210.5, "price": -115},
                    {"name": "Under", "description": "Nick Poulos", "point": 210.5, "price": -105}]}]}]}


def ev(i, hours):
    return {"id": i, "commence_time": (NOW + dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": "H", "away_team": "A"}


def test_day_run_takes_games_within_40_hours_and_skips_fresh_files(tmp_path, monkeypatch):
    monkeypatch.setattr(CP, "LIVE", tmp_path)
    c = Fake([ev("a", 5), ev("b", 30), ev("c", 50), ev("d", -1)])
    r = CP.run(c, "day", now=NOW, verbose=False)
    assert c.calls == ["a", "b"] and r["with_props"] == 1 and r["spent"] == 3
    assert (tmp_path / "2026" / "a.csv").exists() and (tmp_path / "2026" / "b.csv").exists()
    c2 = Fake([ev("a", 4), ev("b", 29)])
    CP.run(c2, "day", now=NOW + dt.timedelta(hours=1), verbose=False)
    assert c2.calls == []                                   # fetched an hour ago: still fresh


def test_an_empty_answer_keeps_earlier_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(CP, "LIVE", tmp_path)
    CP.run(Fake([ev("a", 5)]), "day", now=NOW, verbose=False)
    CP.run(Fake([ev("a", 3)], posted=()), "kick", now=NOW + dt.timedelta(hours=4), verbose=False)
    assert len(pd.read_csv(tmp_path / "2026" / "a.csv")) == 2


def payload():
    return {"meta": {"season": 2026}, "slate": {"games": [{"id": "2026_10_AKR_OHIO", "kickoff": "2026-11-07T17:00Z",
                                                           "home_name": "Ohio Bobcats", "away_name": "Akron Zips", "home": "OHIO", "away": "AKR"}]},
            "players": {"players": [{"game": "2026_10_AKR_OHIO", "id": "11", "name": "Nick Poulos", "team": "OHIO", "pos": "QB"},
                                    {"game": "2026_10_AKR_OHIO", "id": "12", "name": "Jo Back", "team": "OHIO", "pos": "RB"}]}}


def snaps():
    base = {"snapshot": "2026-11-07T12:13:00Z", "event_id": "e1", "commence": "2026-11-07T17:00:00Z", "home_name": "Ohio Bobcats",
            "away_name": "Akron Zips", "updated": ""}
    rows = [("draftkings", "player_pass_yds", "Nick Poulos", "over", 210.5, -115), ("draftkings", "player_pass_yds", "Nick Poulos", "under", 210.5, -105),
            ("fanduel", "player_pass_yds", "Nick Poulos", "over", 210.5, -110), ("fanduel", "player_pass_yds", "Nick Poulos", "under", 210.5, -110),
            ("draftkings", "player_rush_yds_alternate", "Jo Back", "over", 99.5, 250), ("fanduel", "player_rush_yds_alternate", "Jo Back", "over", 99.5, 230),
            ("fanduel", "player_anytime_td", "Jo Back", "yes", None, 120), ("fanduel", "player_pass_yds", "Some Stranger", "over", 150.5, -110)]
    return pd.DataFrame([dict(base, book=b, market=m, player=p, side=s, line=l, price=pr) for b, m, p, s, l, pr in rows])


def test_book_legs_price_as_the_test_found():
    legs = PA.cfb_book_legs(payload(), {}, NOW, snaps())
    by = {(l["s"], l["d"], l["L"]): l for l in legs}
    o = by[("pass_yds", "over", 210.5)]
    nv = np.mean([O.no_vig_power(-115, -105)[0], O.no_vig_power(-110, -110)[0]])
    assert o["w"] == pytest.approx(nv, abs=1e-4) and o["bk"] == {"dk": -115, "fd": -110} and o["src"] == "books"
    assert by[("pass_yds", "under", 210.5)]["w"] == pytest.approx(1 - nv, abs=1e-4)
    r = by[("rush_yds", "over", 99.5)]
    want = np.mean([np.interp(O.implied_prob(p), *zip(*PA.CFB_RUNG_CAL)) for p in (250, 230)])
    assert r["w"] == pytest.approx(want, abs=1e-4) and "alt rung" in r["t"] and r["lab"].endswith("100+ rush yds")
    td = by[("anytime_td", "over", 0.5)]
    assert td["w"] == pytest.approx(float(O.one_sided_fair(120, PA.CFB_TD_MARGIN, "power")), abs=1e-4)
    assert not any(l["pl"] == "Some Stranger" for l in legs)          # not on the board: left out
    assert all("not market-tested" not in l["t"] for l in legs)
