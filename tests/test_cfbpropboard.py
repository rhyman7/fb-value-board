"""The college Props section's builder (nflmodel/cfbpropboard.py)."""
import pandas as pd

from nflmodel import cfbpropboard as CB


def test_target_share_from_reception_share():
    pay = {"players": {"players": [{"id": "1", "why": {"rec_share": 0.24, "rec_share_hist": 0.2}},
                                   {"id": "2", "why": {"tgt_share": 0.1, "rec_share": 0.3}},
                                   {"id": "3", "why": None}]}}
    out = CB.with_target_share(pay)
    w = [p["why"] for p in out["players"]["players"]]
    assert w[0]["tgt_share"] == 0.24 and w[0]["tgt_share_hist"] == 0.2
    assert w[1]["tgt_share"] == 0.1                      # a real target share is kept
    assert "tgt_share" not in (pay["players"]["players"][0]["why"])   # the original is untouched


def test_grade_over_push_void():
    acts = {"2026_10_A_B": {"7": {"rec_yds": 60.0, "rush_td": 0.0, "rec_td": 1.0}}}
    full = {"2026_10_A_B"}
    r = {"game_id": "2026_10_A_B", "player_id": "7", "stat": "rec_yds", "line": 59.5}
    assert CB._grade(r, acts, full) == (60.0, "win")
    assert CB._grade({**r, "line": 60.0}, acts, full) == (60.0, "push")
    assert CB._grade({**r, "stat": "anytime_td", "line": 0.5}, acts, full) == (1.0, "win")
    assert CB._grade({**r, "player_id": "8"}, acts, full) == (None, "void")          # didn't play
    assert CB._grade({**r, "player_id": "8"}, acts, set()) == (None, None)            # not fully graded yet
    assert CB._grade({**r, "game_id": "2026_10_C_D"}, acts, full) == (None, None)     # not played


def test_lines_fair_chances():
    rows = []
    for b, o, u in (("draftkings", -110, -110), ("fanduel", -120, 100)):
        rows += [dict(snapshot="t", event_id="e", commence="2026-11-07T20:00:00Z", book=b, market="player_reception_yds",
                      player="A B", side="over", line=50.5, price=o, stat="rec_yds", alt=False, nn="a b"),
                 dict(snapshot="t", event_id="e", commence="2026-11-07T20:00:00Z", book=b, market="player_reception_yds",
                      player="A B", side="under", line=50.5, price=u, stat="rec_yds", alt=False, nn="a b")]
    rows.append(dict(snapshot="t", event_id="e", commence="2026-11-07T20:00:00Z", book="fanduel", market="player_reception_yds_alternate",
                     player="A B", side="over", line=29.5, price=-400, stat="rec_yds", alt=True, nn="a b"))
    x = CB._lines(pd.DataFrame(rows), {"e": "2026_10_A_B"}).set_index("line")
    assert 0.5 < x.loc[50.5, "fair_over"] < 0.53
    assert x.loc[50.5, "game_id"] == "2026_10_A_B"
    assert 0.5 < x.loc[29.5, "fair_over"] < 0.8 and '"fanduel":[-400,null]' in x.loc[29.5, "books"]


def test_parlays_college_legs_use_reception_share(monkeypatch):
    """parlays.cfb_prop_legs reads players through with_target_share."""
    from nflmodel import parlays as PL, propmarket as PM
    seen = {}

    def fake(payload):
        seen["why"] = payload["players"]["players"][0]["why"]
        return pd.DataFrame()
    monkeypatch.setattr(PM, "board_players", fake)
    pay = {"players": {"players": [{"id": "1", "why": {"rec_share": 0.3}}]}, "slate": {"games": []}}
    assert PL.cfb_prop_legs(pay, None, {}, None) == []
    assert seen["why"]["tgt_share"] == 0.3
