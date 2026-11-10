"""nflmodel/qshade.py: the display shade for NFL Questionable players (Nov 9, 2026)."""
import json

from nflmodel import qshade

C = {"chance": {"overall": 0.64, "WR": {"0": 0.76, "1": 0.69, "2": 0.53}, "RB": {"1": 0.6}},
     "lambda": {"rec_yds": 0.7, "rush_yds": 0.7, "receptions": 0.7, "carries": 0.7}}


def _p(**kw):
    p = {"id": "x", "game": "g", "pos": "WR", "status": None,
         "proj": {"targets": 9.0, "receptions": 7.0, "rec_yds": 100.0, "carries": 0.0, "rush_yds": 0.0, "rec_td": 0.5}}
    p.update(kw)
    return p


def test_shades_this_weeks_questionable_and_leaves_proj():
    p = _p()
    before = json.dumps(p["proj"], sort_keys=True)
    assert qshade.apply([p], {("g", "x"): (1, 1)}, C) == 1
    assert json.dumps(p["proj"], sort_keys=True) == before          # props price from proj: untouched
    k = 1 - 0.7 * (1 - 0.69)
    assert p["proj_ev"]["rec_yds"] == round(100 * k, 1) and p["proj_ev"]["receptions"] == round(7 * k, 1)
    assert "targets" not in p["proj_ev"] and "rec_td" not in p["proj_ev"]
    assert p["play_chance"] == 0.69 and p["practice"] == "limited in practice"


def test_uses_this_weeks_report_not_board_status():
    stale = _p(status="Questionable")                 # last week's tag, no listing this week
    assert qshade.apply([stale], {}, C) == 0 and "proj_ev" not in stale
    out = _p()
    assert qshade.apply([out], {("g", "x"): (3, 2)}, C) == 0       # Out is handled elsewhere


def test_no_practice_uses_overall_and_qbs_skipped():
    p = _p()
    qshade.apply([p], {("g", "x"): (1, -1)}, C)
    assert p["play_chance"] == 0.64 and "practice" not in p
    qb = _p(pos="QB")
    assert qshade.apply([qb], {("g", "x"): (1, 1)}, C) == 0


def test_constants_file_loads():
    c = qshade.load()
    assert c and set(c["lambda"]) == set(qshade.STATS)
    for g in ("RB", "WR", "TE"):
        assert set(c["chance"][g]) == {"0", "1", "2"} and all(0 < v < 1 for v in c["chance"][g].values())


def test_board_questionable_tag_needs_this_weeks_report():
    from nflmodel.players import this_week
    rep = {("g", "a"): 1, ("g", "b"): -1, ("g", "c"): 0}
    assert this_week("Questionable", rep, ("g", "a")) == "Questionable"
    assert this_week("Questionable", rep, ("g", "b")) is None          # last week's tag, not on this week's report
    assert this_week("Questionable", rep, ("g", "z")) is None
    assert this_week("Probable", rep, ("g", "c")) == "Probable"
    assert this_week("Out", rep, ("g", "b")) == "Out"                   # Out / Doubtful / roster labels untouched
    assert this_week("Injured reserve", rep, ("g", "b")) == "Injured reserve"
    assert this_week("Questionable", None, ("g", "b")) == "Questionable"  # no report data: as before


def test_qb_flag_only_after_limited_or_no_practice():
    games = [{"id": "g1", "away": "CLE", "home": "NO", "away_qb": {"name": "Deshaun Watson"}, "home_qb": {"name": "Tyler Shough"}},
             {"id": "g2", "away": "A", "home": "B", "away_qb": {"name": "Full Guy"}, "home_qb": {"name": "No Report"}, "qb_flag": [{"stale": 1}]}]
    names = {"w": "Deshaun Watson", "s": "Tyler Shough", "f": "Full Guy", "x": "Backup Guy"}
    rep = {("g1", "w"): (1.0, 1.0), ("g1", "s"): (-1.0, -1.0), ("g1", "x"): (1, 2), ("g2", "f"): (1.0, 0.0)}
    assert qshade.qb_flags(games, rep, names) == 1
    assert games[0]["qb_flag"] == [{"team": "CLE", "name": "Deshaun Watson", "practice": "limited Friday", "started": 0.54}]
    assert "qb_flag" not in games[1]          # full practice: no flag; an old flag is cleared
