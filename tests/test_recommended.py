"""The Recommended list (nflmodel/recommended.py): the rules as approved on Nov 20, 2026."""
import datetime as dt
import json

import pandas as pd

from nflmodel import recommended as R

NOW = dt.datetime(2026, 11, 23, 12, 0, tzinfo=dt.timezone.utc)
KO = pd.Timestamp("2026-11-29T18:00Z")


def quote(book, line, odds, ev_line=None):
    return {"book": book, "line": line, "odds": odds, "odds_net": odds, "ev_line": ev_line}


def bet(market, side, books, gap=0.0, opener_gap=None):
    return {"market": market, "side": side, "pick": "x", "books": books, "gap": gap, "opener_gap": opener_gap}


def game(gid, margin, total, bets, week=12, mtotal=44.5, **kw):
    away, home = gid.split("_")[2:4]
    return {"id": gid, "week": week, "home": home, "away": away, "market": {"spread": 3.0, "total": mtotal},
            "model": {"margin": margin, "total": total}, "opening_spread": 3.0, "opening_total": mtotal, "bets": bets, **kw}


def test_opener_pick_takes_the_better_book_and_needs_two_points_there():
    # home favoured by 3 at the open, the model says 6: home -3 at FanDuel is 3 points off, DraftKings' -4.5 only 1.5
    g = game("2026_12_AAA_BBB", 6.0, 44.0, [
        bet("Spread", "home", [quote("draftkings", -4.5, -110), quote("fanduel", -3.0, -112), quote("betmgm", -2.5, -110)], opener_gap=3.0),
        bet("Spread", "away", [quote("draftkings", 4.5, -110)], opener_gap=-3.0)])
    got = R.opener_picks(g, "nfl")
    assert [(t, b["side"], q["book"]) for t, b, q, _ in got] == [("nfl_spread_open", "home", "fanduel")]
    # every book has moved to within 2 points of the model: no pick, though it was 3 off the opener
    g2 = game("2026_12_AAA_BBB", 6.0, 44.0, [bet("Spread", "home", [quote("draftkings", -4.5, -110)], opener_gap=3.0)])
    assert R.opener_picks(g2, "nfl") == []
    # another book alone never makes a pick
    g3 = game("2026_12_AAA_BBB", 6.0, 44.0, [bet("Spread", "home", [quote("betmgm", -3.0, -110)], opener_gap=3.0)])
    assert R.opener_picks(g3, "nfl") == []


def test_college_rules_week_five_and_totals_at_three():
    books = [quote("draftkings", -3.0, -110)]
    early = game("2026_05_AAA_BBB", 6.0, 50.0, [bet("Spread", "home", books, opener_gap=3.0)], week=4)
    later = game("2026_05_AAA_BBB", 6.0, 50.0, [bet("Spread", "home", books, opener_gap=3.0)], week=5)
    assert R.opener_picks(early, "cfb") == [] and len(R.opener_picks(later, "cfb")) == 1
    tot = game("2026_02_CCC_DDD", 0.0, 51.5, [bet("Total", "over", [quote("fanduel", 48.5, -110)], opener_gap=3.0),
                                    bet("Total", "under", [quote("fanduel", 48.5, -110)], opener_gap=-3.0)], week=2, mtotal=48.5)
    assert [(t, b["side"]) for t, b, _, _ in R.opener_picks(tot, "cfb")] == [("cfb_total_open", "over")]
    assert R.opener_picks(tot, "nfl") == []                    # NFL totals are not a track
    final = game("2026_18_AAA_BBB", 6.0, 44.0, [bet("Spread", "home", books, opener_gap=3.0)], final_week=True)
    assert R.opener_picks(final, "nfl") == []


def test_price_pick_needs_one_of_the_two_books_and_the_model_on_its_side():
    b_ok = bet("Spread", "home", [quote("draftkings", -3.0, -105, ev_line=0.02), quote("betmgm", -3.0, 100, ev_line=0.05)], gap=0.5)
    b_no = bet("Total", "over", [quote("fanduel", 44.5, -105, ev_line=0.02)], gap=-0.5)       # the model is on the other side
    b_thin = bet("Moneyline", "home", [quote("fanduel", None, -150, ev_line=0.005)], gap=0.5)
    got = R.price_picks(game("2026_12_AAA_BBB", 3.5, 44.0, [b_ok, b_no, b_thin]))
    assert [(b["market"], q["book"]) for _, b, q, _ in got] == [("Spread", "draftkings")]


def test_record_is_written_once_and_never_changes(tmp_path):
    g = game("2026_12_AAA_BBB", 6.0, 44.0, [bet("Spread", "home", [quote("fanduel", -3.0, -112)], opener_gap=3.0)])
    assert R.record_games([g], "nfl", 2026, {g["id"]: KO}, "b1", NOW, tmp_path) == 1
    g["bets"][0]["books"] = [quote("draftkings", -2.5, -105)]                       # a better number later: the pick stays as published
    assert R.record_games([g], "nfl", 2026, {g["id"]: KO}, "b2", NOW + dt.timedelta(hours=5), tmp_path) == 0
    assert R.record_games([g], "nfl", 2026, {g["id"]: KO}, "b3", KO.to_pydatetime() + dt.timedelta(hours=1), tmp_path) == 0
    row = pd.read_csv(R.picks_path("nfl", tmp_path)).iloc[0]
    assert (row["book"], row["line"], row["odds"], row["pick"], row["built"]) == ("fanduel", -3.0, -112.0, "BBB -3", "b1")
    late = game("2026_12_CCC_DDD", 6.0, 44.0, [bet("Spread", "home", [quote("fanduel", -3.0, -112)], opener_gap=3.0)])
    assert R.record_games([late], "nfl", 2026, {late["id"]: KO}, "b4", KO.to_pydatetime(), tmp_path) == 0     # kicked off: nothing


def test_teasers_pair_two_games_at_one_book_and_grade_at_minus_120(tmp_path):
    def dog(gid, mtotal=44.5):
        return game(gid, 0.0, 44.0, [bet("Spread", "away", [quote("fanduel", 2.5, -110), quote("draftkings", 3.0, -115)]),
                                     bet("Spread", "home", [quote("fanduel", -2.5, -110)])], mtotal=mtotal)
    gs = [dog("2026_12_AAA_BBB"), dog("2026_12_CCC_DDD"), dog("2026_12_EEE_FFF"), dog("2026_12_GGG_HHH", mtotal=50.5)]
    kick = {g["id"]: KO for g in gs}
    assert R.record_games(gs, "nfl", 2026, kick, "b1", NOW, tmp_path) == 1          # three legs: one pair, one left over; 50.5 is out
    row = pd.read_csv(R.picks_path("nfl", tmp_path)).iloc[0]
    legs = json.loads(row["legs"])
    assert row["book"] == "fanduel" and row["odds"] == -120 and [l["teased"] for l in legs] == [8.5, 8.5]
    assert row["pick"] == "AAA +8.5 and CCC +8.5"
    assert R.record_games(gs, "nfl", 2026, kick, "b2", NOW, tmp_path) == 0          # the used games are not paired again
    scores = {"2026_12_AAA_BBB": (27, 20), "2026_12_CCC_DDD": (24, 20)}             # AAA lost by 7, CCC by 4: both inside +8.5
    assert R.grade_games("nfl", scores, root=tmp_path) == 1
    res = pd.read_csv(R.results_path("nfl", tmp_path)).iloc[0]
    assert res["result"] == "W" and abs(res["units"] - 0.8333) < 1e-4


def test_a_held_track_publishes_nothing_new_but_keeps_and_grades_its_picks(tmp_path, monkeypatch):
    props = tmp_path / "props"
    props.mkdir()
    row = {"g": "2026_12_AAA_BBB", "t": "AAA", "alt": False, "snap": "flag", "ko": "2026-11-29T18:00:00Z", "cL": None, "cfo": None, "res": None,
           "id": "p1", "n": "One", "pos": "WR", "s": "rec_yds", "L": 54.5, "po": 0.30, "pu": 0.70, "bk": {"fanduel": [-110, -110]}}
    f = props / "2026_w12.json"
    f.write_text(json.dumps({"props": [row]}))
    monkeypatch.setattr(R, "HELD", set())
    assert R.record_props(2026, "b1", NOW, tmp_path, props) == 1                     # published before the track was stopped
    monkeypatch.setattr(R, "HELD", {"nfl_prop_under"})
    f.write_text(json.dumps({"props": [{**row, "res": "loss"}, {**row, "id": "p2", "n": "Two"}]}))
    assert R.STATUS["nfl_prop_under"] == "held" and R.record_props(2026, "b2", NOW, tmp_path, props) == 0
    assert R.grade_props(2026, NOW, tmp_path, props) == 1
    tr = {t["key"]: t for t in R.summary("nfl", tmp_path, NOW)["tracks"]}["nfl_prop_under"]
    assert (tr["status"], tr["n"], tr["w"]) == ("held", 1, 1)


def test_prop_unders_rule_grade_and_rolling_total(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "HELD", set())                                            # the rule itself, as it will run once the track starts
    props = tmp_path / "props"
    props.mkdir()
    base = {"g": "2026_12_AAA_BBB", "t": "AAA", "alt": False, "snap": "flag", "ko": "2026-11-29T18:00:00Z", "cL": None, "cfo": None, "res": None}
    rows = [
        {**base, "id": "p1", "n": "One", "pos": "WR", "s": "rec_yds", "L": 54.5, "po": 0.30, "pu": 0.70, "bk": {"draftkings": [-110, -114], "fanduel": [-108, -118], "betmgm": [-110, 105]}},
        {**base, "id": "p2", "n": "Two", "pos": "QB", "s": "rush_yds", "L": 20.5, "po": 0.20, "pu": 0.80, "bk": {"fanduel": [-110, -110]}},      # a quarterback's rushing: not in the rule
        {**base, "id": "p3", "n": "Three", "pos": "RB", "s": "carries", "L": 15.5, "po": 0.40, "pu": 0.60, "bk": {"fanduel": [-110, -110]}},     # over chance 40%: not below 35%
        {**base, "id": "p4", "n": "Four", "pos": "RB", "s": "rush_yds", "L": 61.5, "po": 0.30, "pu": 0.70, "bk": {"fanduel": [-115, 100]}},
        {**base, "id": "p5", "n": "Five", "pos": "WR", "s": "receptions", "L": 4.5, "po": 0.30, "pu": 0.70, "alt": True, "bk": {"fanduel": [-110, -110]}},
    ]
    f = props / "2026_w12.json"
    f.write_text(json.dumps({"season": 2026, "week": 12, "props": rows}))
    assert R.record_props(2026, "b1", NOW, tmp_path, props) == 2
    picks = pd.read_csv(R.picks_path("nfl", tmp_path))
    assert list(picks["label"]) == ["One (AAA), receiving yards", "Four (AAA), rushing yards"]
    assert list(picks["book"]) == ["draftkings", "fanduel"] and list(picks["odds"]) == [-114.0, 100.0]
    assert R.record_props(2026, "b2", NOW, tmp_path, props) == 0
    rows[0]["res"], rows[3]["res"] = "loss", "win"                                   # the over lost: the under won; and the reverse
    f.write_text(json.dumps({"season": 2026, "week": 12, "props": rows}))
    assert R.grade_props(2026, NOW, tmp_path, props) == 2 and R.grade_props(2026, NOW, tmp_path, props) == 0
    s = R.summary("nfl", tmp_path, NOW)
    tr = {t["key"]: t for t in s["tracks"]}["nfl_prop_under"]
    assert (tr["w"], tr["l"], tr["to_go"]) == (1, 1, 98)
    assert abs(s["all"]["dollars"] - (8.77 - 10.0)) < 0.01                            # $10 at -114 wins $8.77; the other loses $10
    assert s["weeks"][0]["running"] == s["all"]["dollars"]


def test_a_bet_on_two_tracks_counts_once_in_the_totals(tmp_path):
    g = game("2026_12_AAA_BBB", 6.0, 44.0, [bet("Spread", "home", [quote("fanduel", -3.0, -110, ev_line=0.03)], gap=3.0, opener_gap=3.0)])
    assert R.record_games([g], "nfl", 2026, {g["id"]: KO}, "b1", NOW, tmp_path) == 2    # the opener track and the price track
    assert R.grade_games("nfl", {g["id"]: (30, 20)}, root=tmp_path) == 2
    s = R.summary("nfl", tmp_path, NOW)
    assert s["all"]["graded"] == 1 and abs(s["all"]["dollars"] - 9.09) < 0.01
    assert sorted(t["w"] for t in s["tracks"]) == [0, 0, 1, 1] and sum(p["dup"] for p in s["picks"]) == 1


def test_page_file_is_a_script_the_page_can_read_from_disk(tmp_path):
    path = R.write("cfb", tmp_path, tmp_path / "out", NOW)
    body = path.read_text()
    assert body.startswith("window.RECOMMENDED=window.RECOMMENDED||{};window.RECOMMENDED.cfb=")
    d = json.loads(body.split("window.RECOMMENDED.cfb=", 1)[1].rstrip().rstrip(";"))
    assert [t["key"] for t in d["tracks"]] == ["cfb_spread_open", "cfb_total_open"] and d["picks"] == [] and d["unit"] == 10


def test_every_track_starts_on_trial_and_the_rules_are_the_approved_ones():
    assert set(R.STATUS.values()) == {"trial", "held"} and R.HELD == {"nfl_prop_under"} and len(R.TRACKS) == 6
    assert (R.OPEN_GAP, R.TOTAL_GAP, R.CFB_FROM_WEEK, R.DOG, R.MAX_TOTAL, R.TEASER_ODDS, R.NEED, R.UNIT) == (2.0, 3.0, 5, (1.5, 2.5), 49.5, -120.0, 100, 10.0)
    assert R.MY_BOOKS == ("draftkings", "fanduel")
