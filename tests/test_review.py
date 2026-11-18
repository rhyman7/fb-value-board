"""Weekly Review: the chance ledger and the lean verdicts."""
from nflmodel import review as R
from nflmodel import reviewdraft as D

PTS = {"fumble": 3.0, "nonoff_td": 7.0, "fg_miss": 1.0}


def side(**k):
    base = dict(fumbles=0, fum_lost=0, nonoff_td=0, fg_miss_short=0, qb_exit=None, success=0.4, epa_play=0.0)
    base.update(k)
    return base


def test_chance_signs_favour_the_lucky_side():
    h = side(fumbles=2, fum_lost=0, nonoff_td=1)
    a = side(fumbles=1, fum_lost=1, fg_miss_short=1)
    c = R.chance(h, a, PTS)
    assert c["fumble_bounces"] == 3.0 * 1.5          # away lost 1 of 1, home 0 of 2: 1.5 bounces to home
    assert c["return_def_tds"] == 7.0
    assert c["short_fg_misses"] == 1.0
    assert c["total_pts_nonoff"] == 7
    assert c["total"] == 4.5 + 7.0 + 1.0


def game(margin, total, spread=-3.0, mtotal=45.0, model=(-6.0, 50.0), h=None, a=None):
    h, a = h or side(), a or side()
    g = {"home": "HOM", "away": "AWY", "market": {"spread": spread, "total": mtotal},
         "model": {"margin": model[0], "total": model[1]}, "margin": margin, "total": total,
         "box": {"HOM": h, "AWY": a}}
    g["chance"] = R.chance(h, a, PTS)
    g["neutral_margin"] = margin - g["chance"]["total"]
    g["leans"] = {}
    for k, mv, lv, act, hi in (("spread", model[0], spread, margin, "home"), ("total", model[1], mtotal, total, "over")):
        s = hi if mv > lv else ("away" if hi == "home" else "under")
        d = act - lv
        g["leans"][k] = {"side": s, "gap": mv - lv, "result": "W" if (d > 0) == (s == hi) else "L"}
    return g


def test_verdict_is_chance_only_when_luck_flips_it():
    # model likes the away side (−6 vs −3); away wins by 5 → W on merit
    assert D.verdicts(game(-5, 40))["spread"] == "W"
    # home won by 1 only thanks to a pick-six: without it away covers → chance
    g = game(1, 40, h=side(nonoff_td=1))
    assert g["leans"]["spread"]["result"] == "L"
    assert D.verdicts(g)["spread"] == "chance"
    # the over (model 50 vs 45) cashed at 48 only because of 7 defensive points → chance
    assert D.verdicts(game(-5, 48, a=side(nonoff_td=1)))["total"] == "chance"


def test_qb_exit_marks_both_leans_chance():
    a = side(qb_exit={"qb": "A.Starter", "share": 0.2, "qtr": 1, "time": "10:00", "why": "injured"})
    v = D.verdicts(game(-5, 40, a=a))
    assert v == {"spread": "chance", "total": "chance"}


def test_drive_kinds_and_short_reason():
    from nflmodel import reviewctx as C
    assert C._drive_kind("Touchdown") == "TD"
    assert C._drive_kind("Missed field goal") == "Missed FG"
    assert C._drive_kind("Turnover on downs") == "Downs"
    assert C._drive_kind("INT") == "Turnover"
    m = {"name": "A Back", "pos": "RB", "team": "AWY", "stat": "rush_yds", "proj": 65.0, "act": 15.0,
         "volume": {"kind": "carries", "proj": 13.0, "act": 7.0}, "script": {"trailing9": 0.95, "leading9": 0.0},
         "prev": {"week": 9, "val": 143}, "season_avg": 88.0, "season_low": 15.0}
    s = D.why_short(m)
    assert "usage fell: 7.0 carries" in s or "usage fell: 7 carries" in s
    assert "trailed by 9+" in s and "season low" in s
