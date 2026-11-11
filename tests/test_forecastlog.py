import pandas as pd

from nflmodel import forecastlog as F


def game(gid, margin=1.0, spread=2.5, proj=2.0, qb="A Passer", ko="2026-11-15T18:00Z", **kw):
    return {"id": gid, "week": 11, "away": "AAA", "home": "HHH", "kickoff": ko, "weight": 0.3, "early": False,
            "model": {"margin": margin, "total": 44.0}, "market": {"spread": spread, "total": 43.5},
            "proj": {"margin": proj, "total": 43.6}, "away_qb": {"name": qb}, "home_qb": {"name": "H Passer"}, **kw}


def test_appends_only_on_change(tmp_path):
    gs = [game("g1"), game("g2")]
    assert F.record(gs, "nfl", 2026, "2026-11-11 12:00 UTC", root=tmp_path) == 2
    assert F.record(gs, "nfl", 2026, "2026-11-11 15:00 UTC", root=tmp_path) == 0          # nothing changed
    before = F.path("nfl", 2026, tmp_path).read_text()
    gs[0]["market"]["spread"] = 3.0                                                        # the line moves
    gs[1]["away_qb"]["name"] = "B Backup"                                                  # a starter changes
    assert F.record(gs, "nfl", 2026, "2026-11-12 12:00 UTC", root=tmp_path) == 2
    after = F.path("nfl", 2026, tmp_path).read_text()
    assert after.startswith(before)                                                        # earlier rows are never rewritten
    log = F.read("nfl", 2026, tmp_path)
    assert list(log.columns) == F.COLS and len(log) == 4
    assert log["market_spread"].tolist() == [2.5, 2.5, 3.0, 2.5] and log["away_qb"].iloc[3] == "B Backup"
    assert set(log["source"]) == {"live"} and log["early"].tolist() == [0, 0, 0, 0]


def test_refit_wobble_is_not_a_row_but_drift_is(tmp_path):
    g = game("g1", margin=1.00, proj=2.00)
    assert F.record([g], "nfl", 2026, "2026-11-11 12:00 UTC", root=tmp_path) == 1
    for i, m in enumerate((1.01, 1.02, 1.04)):                                             # under 0.05 from the logged 1.00
        g["model"]["margin"] = m
        assert F.record([g], "nfl", 2026, f"2026-11-11 1{3 + i}:00 UTC", root=tmp_path) == 0
    g["model"]["margin"] = 1.05                                                            # drift adds up against the last logged row
    assert F.record([g], "nfl", 2026, "2026-11-11 18:00 UTC", root=tmp_path) == 1
    g["weight"] = 0.21                                                                     # the early weight switching is a row
    assert F.record([g], "nfl", 2026, "2026-11-11 19:00 UTC", root=tmp_path) == 1


def test_stops_at_kickoff_and_keeps_leagues_apart(tmp_path):
    gs = [game("g1", ko="2026-11-11T10:00Z"), game("g2")]
    assert F.record(gs, "cfb", 2026, "2026-11-11 12:00 UTC", root=tmp_path) == 1           # g1 has kicked off
    assert F.read("cfb", 2026, tmp_path)["game_id"].tolist() == ["g2"]
    assert not F.path("nfl", 2026, tmp_path).exists()


def test_quarterback_status(tmp_path):
    gs = [game("g1"), game("g2")]
    names = {"p1": "A. Passer", "p2": "H Passer"}
    st = F.qb_status(gs, {("g1", "p1"): (1.0, 2.0), ("g1", "p2"): (-1.0, -1.0)}, names)
    assert st == {("g1", "away"): (1, 2), ("g1", "home"): (-1, -1)}
    assert F.record(gs, "nfl", 2026, "2026-11-13 22:00 UTC", st, root=tmp_path) == 2
    log = F.read("nfl", 2026, tmp_path).set_index("game_id")
    assert log.loc["g1", "away_qb_report"] == 1 and log.loc["g1", "away_qb_practice"] == 2
    assert log.loc["g1", "home_qb_report"] == -1 and pd.isna(log.loc["g2", "away_qb_report"])
    st[("g1", "away")] = (1, 1)                                                            # a status change alone is a new row
    assert F.record(gs, "nfl", 2026, "2026-11-14 12:00 UTC", st, root=tmp_path) == 1
    # the backfill reads status from the board's qb_flag
    g = game("g3", qb_flag=[{"team": "HHH", "name": "H Passer", "practice": "limited Friday", "started": 0.54}])
    assert F.status_from_flags([g]) == {("g3", "home"): (1, 1)}
