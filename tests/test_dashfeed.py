"""The matchup dashboards' feed (nflmodel/dashfeed.py)."""
import json

from nflmodel import dashfeed


def _data(league="nfl"):
    bet = {"market": "Spread", "pick": "BUF -3", "odds": -115, "ev": 0.009, "tier": "opener", "missed": False}
    return {
        "meta": {"season": 2026, "generated": "2026-11-01 12:00"},
        "slate": {"games": [{
            "id": "2026_08_BAL_BUF", "game_id": 401, "away": "BAL", "home": "BUF",
            "proj": {"margin": 3.81, "total": 54.36}, "market": {"spread": 3.0, "total": 54.5},
            "bets": [bet, {**bet, "pick": "Over 54.5", "tier": None},            # not on the bet sheet
                     {**bet, "pick": "BAL +3", "missed": True}],                   # missed: off the sheet
        }]},
        "players": {"players": [{"id": "1", "name": "Kenneth Walker III", "pos": "RB", "team": "KC", "game": "g",
                                 "week": 8, "status": None,
                                 "proj": {"rush_yds": 116.84, "rec_yds": 0.0, "carries": 21.2}}]},
    }


def test_norm_drops_suffixes_and_punctuation():
    assert dashfeed.norm("Kenneth Walker III") == "kenneth walker"
    assert dashfeed.norm("Amon-Ra St. Brown") == "amon ra st brown"
    assert dashfeed.norm("D.J. Moore Jr.") == "dj moore"


def test_games_keep_projection_and_only_bet_sheet_bets():
    f = dashfeed.build("nfl", _data())
    g = f["games"]["2026_08_BAL_BUF"]
    assert (g["margin"], g["total"], g["mkt_spread"]) == (3.8, 54.4, 3.0)
    assert [b["pick"] for b in g["bets"]] == ["BUF -3"]
    assert f["week"] == 8 and f["season"] == 2026


def test_college_games_keyed_by_espn_id():
    f = dashfeed.build("cfb", _data())
    assert list(f["games"]) == ["401"]


def test_players_keyed_by_team_and_name_without_zero_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(dashfeed, "DOCS", tmp_path)   # no props file for the week
    f = dashfeed.build("nfl", _data())
    p = f["players"]["KC|kenneth walker"]
    assert p["p"] == {"rush_yds": 116.8, "carries": 21.2}
    assert "L" not in p                      # no props file for the week


def test_prop_lines_attach_main_lines_only(tmp_path, monkeypatch):
    props = tmp_path / "props"
    props.mkdir()
    rows = [{"t": "KC", "n": "Kenneth Walker III", "s": "rush_yds", "alt": False, "L": 89.5, "fo": 0.5012},
            {"t": "KC", "n": "Kenneth Walker III", "s": "rush_yds", "alt": True, "L": 120.5, "fo": 0.2}]
    (props / "2026_w08.json").write_text(json.dumps({"props": rows}))
    monkeypatch.setattr(dashfeed, "DOCS", tmp_path)
    p = dashfeed.build("nfl", _data())["players"]["KC|kenneth walker"]
    assert p["L"] == {"rush_yds": {"L": 89.5, "fo": 0.501}}
