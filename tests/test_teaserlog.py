import datetime as dt
import json

import pandas as pd

from nflmodel import teaserlog as T

NOW = dt.datetime(2026, 11, 8, 12, tzinfo=dt.timezone.utc)
KO = pd.Timestamp("2026-11-08T18:00Z")


def game(gid, spread, total, dk=None, fd=None):
    books = {"draftkings": {"spread_line": spread if dk is None else dk}, "fanduel": {"spread_line": spread if fd is None else fd}}
    return {"id": gid, "week": 10, "home": "H" + gid[-1], "away": "A" + gid[-1], "market": {"spread": spread, "total": total}, "books": books}


def test_kinds():
    assert T.kind_of(2.5) == "dog" and T.kind_of(1.5) == "dog" and T.kind_of(3.0) is None
    assert T.kind_of(-7.5) == "fav" and T.kind_of(-9.0) is None


def test_record_grade_summary(tmp_path):
    # game 1: home favoured by 2.5 -> away is a +2.5 dog (qualifies, total 44); home -2.5 is not Wong
    # game 2: home +1.5 dog (spread -1.5 = away favoured), total 51 -> dog, not qualifying
    # game 3: away +2 dog, total 41
    gs = [game("g1", 2.5, 44.0), game("g2", -1.5, 51.0), game("g3", 2.0, 41.0), game("g4", 6.0, 40.0)]
    ko = {g["id"]: KO for g in gs}
    assert T.record(gs, 2026, ko, tmp_path, NOW) == 3
    assert T.record(gs, 2026, ko, tmp_path, NOW) == 0              # nothing changed
    gs[0]["books"]["draftkings"]["spread_line"] = 3.0              # DK moves to +3: still tracked (anchor in range)
    assert T.record(gs, 2026, ko, tmp_path, NOW) == 1
    log = pd.read_csv(tmp_path / "nfl_log.csv")
    assert set(log["side"]) == {"away", "home"} and log["qualifies"].tolist() == [1, 0, 1, 1]
    # g1 away loses by 7 (+2.5+6 = +8.5 wins, DK +3+6 = +9 wins); g2 home loses by 9 (+7.5 loses); g3 away loses by 8 (+8 pushes)
    scores = {"g1": (24, 17), "g2": (10, 19), "g3": (20, 12)}
    assert T.grade(scores, tmp_path, NOW) == 3
    r = pd.read_csv(tmp_path / "nfl_results.csv").set_index("game_id")
    assert r.loc["g1", "result"] == "W" and r.loc["g1", "result_dk"] == "W"
    assert r.loc["g2", "result"] == "L" and r.loc["g3", "result"] == "P"
    s = T.summary(tmp_path)
    assert s["legs"]["dog, total 49.5 or less (the strategy)"] == {"w": 1, "l": 0, "p": 1, "hit": 1.0}
    assert s["pairs"]["-120"]["p"] == 1 and s["pairs"]["-120"]["units"] == 0
    json.dumps(s)


def test_kicked_off_games_skipped(tmp_path):
    gs = [game("g1", 2.5, 44.0)]
    assert T.record(gs, 2026, {"g1": pd.Timestamp("2026-11-08T11:00Z")}, tmp_path, NOW) == 0
