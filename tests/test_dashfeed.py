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
            "open": {"seen": "2026-10-20", "spread": 3.5, "total": 53.5, "moves": 4},
            "bets": [bet, {**bet, "pick": "Over 54.5", "tier": None},            # not on the bet sheet
                     {**bet, "pick": "BAL +3", "missed": True}],                   # missed: off the sheet
        }]},
        "players": {"players": [{"id": "1", "name": "Kenneth Walker III", "pos": "RB", "team": "KC", "game": "g",
                                 "week": 8, "status": None,
                                 "proj": {"rush_yds": 116.84, "rec_yds": 0.0, "carries": 21.2}, "td_any": 0.5123,
                                 "why": {"def_ypc": 1.0412, "def_ypa": 1.0, "def_ypt": 0.97}}]},
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
    assert g["open"] == {"spread": 3.5, "total": 53.5, "at": "2026-10-20"}
    assert f["week"] == 8 and f["season"] == 2026


def test_college_games_keyed_by_espn_id():
    f = dashfeed.build("cfb", _data())
    assert list(f["games"]) == ["401"]


def test_players_keyed_by_team_and_name_without_zero_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(dashfeed, "DOCS", tmp_path)   # no props file for the week
    f = dashfeed.build("nfl", _data())
    p = f["players"]["KC|kenneth walker"]
    assert p["p"] == {"rush_yds": 116.8, "carries": 21.2}
    assert p["td"] == 0.512
    assert p["f"] == {"ypc": 1.041, "ypt": 0.97}          # 1.0 (no adjustment) is left out
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


def test_write_keeps_last_feed_of_the_week_just_left(tmp_path, monkeypatch):
    monkeypatch.setattr(dashfeed, "OUT", tmp_path)
    d8 = _data()
    dashfeed.write("nfl", d8)
    assert not (tmp_path / "nfl_prev.json").exists()
    dashfeed.write("nfl", d8)                                  # same week again: nothing kept
    assert not (tmp_path / "nfl_prev.json").exists()
    d9 = _data(); d9["players"]["players"][0]["week"] = 9; d9["slate"]["games"][0]["id"] = "2026_09_BAL_BUF"
    dashfeed.write("nfl", d9)
    assert json.loads((tmp_path / "nfl.json").read_text())["week"] == 9
    assert json.loads((tmp_path / "nfl_prev.json").read_text())["week"] == 8
    dashfeed.write("nfl", d8)                                  # going back a week never overwrites it
    assert json.loads((tmp_path / "nfl_prev.json").read_text())["week"] == 8


def test_ladders_main_line_then_three_rungs_below(tmp_path):
    cols = ["season", "week", "team", "player", "stat", "market", "line", "books", "time"]
    def row(t, market, line, books): return dict(zip(cols, [2026, 8, "KC", "Kenneth Walker III", "rush_yds", market, line, json.dumps(books), t]))
    rows = [row("2026-10-25 12:00", "player_rush_yds", 70.5, {"draftkings": [-110, -110]}),                 # an older pull: ignored
            row("2026-10-25 15:00", "player_rush_yds", 80.5, {"draftkings": [-112, -112]}),
            row("2026-10-25 15:00", "player_rush_yds", 79.5, {"fanduel": [-114, -114]})]
    for line, dk, fd in ((89.5, 130, 140), (69.5, -167, -162), (59.5, -262, -250), (49.5, -439, None), (39.5, -810, -800), (29.5, -1700, None)):
        rows.append(row("2026-10-25 15:00", "player_rush_yds_alternate", line, {"draftkings": [dk, None], **({"fanduel": [fd, None]} if fd else {})}))
    import csv
    with (tmp_path / "prop_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
    lad = dashfeed._ladders("nfl", 8, 2026, tmp_path)[("KC", "kenneth walker")]["rush_yds"]
    assert lad["dk"] == [[80.5, -112], [69.5, -167], [59.5, -262], [49.5, -439]]      # nothing above the main line, three below
    assert lad["fd"] == [[79.5, -114], [69.5, -162], [59.5, -250], [39.5, -800]]
    assert dashfeed._ladders("nfl", 9, 2026, tmp_path) == {}

