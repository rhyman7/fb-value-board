import datetime as dt
import json

import pandas as pd

from nflmodel import olwatch as W

NOW = dt.datetime(2026, 11, 15, 12, tzinfo=dt.timezone.utc)
LATER = dt.datetime(2026, 11, 15, 17, tzinfo=dt.timezone.utc)
KO = pd.Timestamp("2026-11-15T18:00Z")


def game(gid, home, away, spread, total):
    return {"id": gid, "week": 11, "home": home, "away": away, "market": {"spread": spread, "total": total}}


def core(rows):
    """rows: (game_id, team, name, out, reason)"""
    return pd.DataFrame([{"game_id": g, "team": t, "name": n, "player_id": "", "share": 0.2, "out": o, "reason": r}
                         for g, t, n, o, r in rows])


def five(gid, team, outs):
    """Five linemen; outs: {index: reason}."""
    return [(gid, team, f"{team} L{i}", int(i in outs), outs.get(i, "")) for i in range(5)]


def test_record_grade_summary(tmp_path):
    gs = [game("g1", "AAA", "BBB", 3.0, 44.0), game("g2", "CCC", "DDD", -2.5, 47.0)]
    ko = {g["id"]: KO for g in gs}
    c = core(five("g1", "AAA", {0: "report: Out", 1: "roster: RES"}) + five("g1", "BBB", {0: "report: Out"})
             + five("g2", "CCC", {}) + five("g2", "DDD", {2: "report: Out"}))
    assert W.record(gs, 2026, ko, c, tmp_path, NOW) == 1           # only AAA has two out
    assert W.record(gs, 2026, ko, c, tmp_path, NOW) == 0           # nothing changed
    log = pd.read_csv(tmp_path / "nfl_log.csv")
    r = log.iloc[0]
    assert r["team"] == "AAA" and r["n_out"] == 2 and r["late"] == 0 and r["spread"] == -3.0
    assert r["team_total"] == 23.5                                  # (44 + 3) / 2
    # inactives: DDD loses a second lineman on late news; AAA's line moves
    c2 = core(five("g1", "AAA", {0: "report: Out", 1: "roster: RES"}) + five("g1", "BBB", {0: "report: Out"})
              + five("g2", "CCC", {}) + five("g2", "DDD", {2: "report: Out", 3: "override"}))
    gs[0]["market"]["spread"] = 2.0
    assert W.record(gs, 2026, ko, c2, tmp_path, LATER) == 2
    log = pd.read_csv(tmp_path / "nfl_log.csv")
    d = log[log["team"] == "DDD"].iloc[0]
    assert d["late"] == 1 and d["spread"] == -2.5 and d["team_total"] == 24.75   # away favoured by 2.5
    assert W.record(gs, 2026, ko, c2, tmp_path, pd.Timestamp("2026-11-15T19:00Z").to_pydatetime()) == 0   # kicked off
    # g1: AAA 20, BBB 17 -> AAA wins by 3: covers -2 (last), pushes -3 (flag); 20 vs 23.5 and 23.0
    # g2: CCC 24, DDD 21 -> DDD loses by 3 as a 2.5 favourite
    assert W.grade({"g1": (20, 17), "g2": (24, 21)}, tmp_path, LATER) == 2
    assert W.grade({"g1": (20, 17), "g2": (24, 21)}, tmp_path, LATER) == 0      # graded once
    res = pd.read_csv(tmp_path / "nfl_results.csv").set_index("team")
    a = res.loc["AAA"]
    assert a["cover_flag"] == "P" and a["cover_last"] == "W" and a["vs_flag_total"] == -3.5 and a["vs_last_total"] == -3.0
    assert a["flag_at"] == "2026-11-15 12:00"
    assert res.loc["DDD", "cover_flag"] == "L" and res.loc["DDD", "vs_flag_total"] == -3.75
    s = W.summary(tmp_path)
    assert s["all"]["n"] == 2 and s["all"]["vs_flag_total"]["mean"] == -3.62
    assert s["late news among the linemen out"]["n"] == 1 and s["all"]["cover_last"] == {"w": 1, "l": 1, "p": 0, "rate": 0.5}


def test_drops_out_before_kickoff(tmp_path):
    gs = [game("g1", "AAA", "BBB", 3.0, 44.0)]
    ko = {"g1": KO}
    W.record(gs, 2026, ko, core(five("g1", "AAA", {0: "report: Out", 1: "override"}) + five("g1", "BBB", {})), tmp_path, NOW)
    # the late row is withdrawn: AAA is back to one out, and that is logged
    assert W.record(gs, 2026, ko, core(five("g1", "AAA", {0: "report: Out"}) + five("g1", "BBB", {})), tmp_path, LATER) == 1
    assert pd.read_csv(tmp_path / "nfl_log.csv")["n_out"].tolist() == [2, 1]
    assert W.grade({"g1": (20, 17)}, tmp_path, LATER) == 0 and not (tmp_path / "nfl_results.csv").exists()


def test_update_without_core_file(tmp_path, capsys):
    s = W.update([], 2026, {}, {}, tmp_path, core=pd.DataFrame(columns=["game_id", "team", "name", "player_id", "share", "out", "reason"]))
    assert "all" not in s and not (tmp_path / "nfl_record.json").exists()
    W.record([game("g1", "AAA", "BBB", 3.0, 44.0)], 2026, {"g1": KO},
             core(five("g1", "AAA", {0: "report: Out", 1: "roster: RES"})), tmp_path, NOW)
    W.update([], 2026, {}, {"g1": (10, 13)}, tmp_path, core=pd.DataFrame(columns=["game_id", "team", "name", "player_id", "share", "out", "reason"]))
    assert json.loads((tmp_path / "nfl_record.json").read_text())["all"]["n"] == 1
