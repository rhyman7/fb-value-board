"""College model tests. Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import bets as B  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402
from nflmodel.ratings import RatingEngine  # noqa: E402


def _toy():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(12)]
    rows, qbs = [], []
    gid = 0
    for week in range(1, 9):
        order = 201500 + week
        perm = rng.permutation(teams)
        for h, a in zip(perm[::2], perm[1::2]):
            gid += 1
            for side, t, o in (("home", h, a), ("away", a, h)):
                n_pass, n_rush = int(rng.integers(20, 40)), int(rng.integers(20, 40))
                rows.append({"game_id": gid, "season": 2015, "side": side, "team": t, "opp": o, "order": order,
                             "week": week, "neutral": 0, "is_home": int(side == "home"),
                             "plays": n_pass + n_rush, "epa": rng.normal(0, 8), "succ": rng.integers(20, 40),
                             "n_pass": n_pass, "n_rush": n_rush, "pass_epa": rng.normal(0, 5), "rush_epa": rng.normal(0, 5),
                             "pts": rng.integers(0, 50), "snaps": rng.integers(55, 90)})
                qbs.append({"game_id": gid, "season": 2015, "side": side, "qb_id": f"{t}qb", "name": t, "team": t,
                            "dropbacks": n_pass, "epa": rng.normal(0, 5), "order": order})
    return pd.DataFrame(rows), pd.DataFrame(qbs)


def test_fast_solver_matches_nfl_solver():
    tg, qb = _toy()
    e = CFBRatingEngine(tg, qb)
    idx = np.flatnonzero(e.order < 201508)
    w = e._weights(201508, e.order[idx])
    for m in e.metrics:
        a, b = RatingEngine._solve(e, m, idx, w), e._solve(m, idx, w)
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and np.isclose(a[2], b[2]) and np.isclose(a[3], b[3])


def test_ratings_ignore_the_current_week():
    tg, qb = _toy()
    e = CFBRatingEngine(tg, qb)
    before = e.ratings_at(201505)
    tg2 = tg.copy()
    tg2.loc[tg2.order >= 201505, "pts"] = 99           # change this week's and later results
    after = CFBRatingEngine(tg2, qb).ratings_at(201505)
    assert np.allclose(before["pts"]["off"], after["pts"]["off"])


def test_half_point_rounding_and_grading():
    assert list(B.half([-3.17, 6.8, 14.25, 0.2])) == [-3.0, 7.0, 14.5, 0.0]
    games = pd.DataFrame({"game_id": [1], "season": [2025], "week": [3], "home_team": ["A"], "away_team": ["B"],
                          "spread_line": [6.83], "total_line": [52.2], "proj_margin": [6.83], "proj_total": [52.2],
                          "result": [7.0], "total": [55.0]})
    dist = {"margin": np.ones(len(B.MARGINS)), "total": np.ones(len(B.TOTALS)), "sd_margin": 16.0, "sd_total": 15.0, "sd_win": 16.0}
    b = B.price_games(games, dist, "proj").set_index(["market", "side"])
    assert b.loc[("Spread", "home"), "outcome"] == 0        # A -7 pushes on a 7-point win
    assert b.loc[("Total", "over"), "outcome"] == 1
    assert abs(b.loc[("Spread", "home"), "model_prob"] - 0.5) < 0.01   # no disagreement, no edge


def test_espn_odds_both_shapes():
    from cfbmodel import espn
    comp = lambda odds: {"id": "401", "competitions": [{"odds": [odds], "competitors": [
        {"homeAway": "home", "team": {"abbreviation": "OSU"}}, {"homeAway": "away", "team": {"abbreviation": "MICH"}}]}]}
    old = espn.parse_event(comp({"details": "OSU -7.5", "overUnder": 48.5, "provider": {"name": "ESPN BET"},
                                 "homeTeamOdds": {"moneyLine": -300, "spreadOdds": -110}, "awayTeamOdds": {"moneyLine": 240, "spreadOdds": -110}}))
    assert old["spread_line"] == 7.5 and old["total_line"] == 48.5 and old["home_moneyline"] == -300 and old["away_moneyline"] == 240
    away_fav = espn.parse_event(comp({"details": "MICH -3", "overUnder": 44}))
    assert away_fav["spread_line"] == -3
    new = espn.parse_event(comp({"pointSpread": {"home": {"close": {"line": "-6.5", "odds": "-105"}}, "away": {"close": {"line": "+6.5", "odds": "-115"}}},
                                 "total": {"over": {"close": {"line": "o51.5", "odds": "-110"}}, "under": {"close": {"line": "u51.5", "odds": "-110"}}},
                                 "moneyline": {"home": {"close": {"odds": "-250"}}, "away": {"close": {"odds": "+205"}}}}))
    assert new["spread_line"] == 6.5 and new["total_line"] == 51.5 and new["home_spread_odds"] == -105 and new["away_moneyline"] == 205


def test_cfbd_lines_parse_and_fall_back_in_order(monkeypatch):
    """CFBD's lines: the home-view spread is flipped, consensus beats DraftKings beats the first
    provider, ESPN-priced games are left alone, and nothing happens without the key."""
    from cfbmodel import cfbd
    ln = lambda prov, spread, total, hml=None, aml=None: {"provider": prov, "spread": spread, "spreadOpen": spread,
                                                          "overUnder": total, "overUnderOpen": total,
                                                          "homeMoneyline": hml, "awayMoneyline": aml, "formattedSpread": ""}
    payload = [
        {"id": 1, "homeTeam": "A", "awayTeam": "B", "startDate": "2026-11-07T20:00:00.000Z",
         "lines": [ln("Bovada", -7.5, 50.5), ln("consensus", -6.5, 51.0), ln("DraftKings", -7.0, 51.5, -280, 230)]},
        {"id": 2, "homeTeam": "C", "awayTeam": "D", "startDate": "2026-11-07T20:00:00.000Z",
         "lines": [ln("Bovada", 3.0, 44.0), ln("DraftKings", 3.5, 44.5, 150, -175)]},
        {"id": 3, "homeTeam": "E", "awayTeam": "F", "startDate": "2026-11-07T20:00:00.000Z",
         "lines": [ln("ESPN Bet", None, 55.0), ln("Bovada", -1.5, 56.0)]},
        {"id": 4, "homeTeam": "G", "awayTeam": "H", "startDate": "2026-11-07T20:00:00.000Z", "lines": []},
        {"id": "bad", "lines": [ln("DraftKings", -3, 40)]},
    ]
    L = cfbd.parse(payload).set_index("game_id")
    assert list(L.index) == [1, 2, 3]
    assert L.loc[1, "book"] == "CFBD consensus" and L.loc[1, "spread_line"] == 6.5 and L.loc[1, "total_line"] == 51.0
    assert L.loc[2, "book"] == "CFBD DraftKings" and L.loc[2, "spread_line"] == -3.5 and L.loc[2, "home_moneyline"] == 150
    assert L.loc[3, "book"] == "CFBD ESPN Bet" and np.isnan(L.loc[3, "spread_line"]) and L.loc[3, "total_line"] == 55.0
    assert cfbd.parse({"message": "unauthorized"}).empty and cfbd.parse([]).empty

    games = pd.DataFrame({"game_id": [1, 2, 3, 5], "played": [False, False, False, True],
                          "spread_line": [4.0, np.nan, np.nan, 10.0], "total_line": [49.0, np.nan, np.nan, 50.0],
                          "line_source": ["DraftKings", np.nan, np.nan, np.nan]})
    out, n = cfbd.apply(games, L.reset_index(), have={1})
    assert n == 1
    assert out.loc[0, "spread_line"] == 4.0 and out.loc[0, "line_source"] == "DraftKings"     # ESPN's price stands
    assert out.loc[1, "spread_line"] == -3.5 and out.loc[1, "total_line"] == 44.5 and out.loc[1, "line_source"] == "CFBD DraftKings"
    assert np.isnan(out.loc[2, "spread_line"]) and out.loc[2, "total_line"] == 55.0 and out.loc[2, "line_source"] == "CFBD ESPN Bet"
    assert out.loc[3, "spread_line"] == 10.0 and pd.isna(out.loc[3, "line_source"])          # played games untouched
    same, n0 = cfbd.apply(games, pd.DataFrame())
    assert n0 == 0 and same.equals(games)

    monkeypatch.delenv(cfbd.ENV, raising=False)
    assert not cfbd.enabled() and cfbd.current_lines(2026, [10]).empty       # no key, no request
    from cfbmodel import pipeline
    assert pipeline.lines_from(0) == "cfbfastR consensus (an average across books)"
    assert pipeline.lines_from(3, 2) == "ESPN for 3 games, CFBD for 2 games, cfbfastR consensus otherwise"


def test_board_breakdowns_add_up():
    import json
    path = Path(__file__).resolve().parent.parent / "docs" / "cfb" / "data.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    for g in d["slate"]["games"]:
        bd = g["detail"]["breakdown"]
        assert abs(bd["margin_sum"] - g["model"]["margin"]) < 0.06, g["id"]
        assert abs(bd["total_sum"] - g["model"]["total"]) < 0.06, g["id"]
        assert 0 <= g["win"]["adj"] <= 1


def test_archive_grades_at_the_half_point():
    from cfbmodel import archive
    g = {"market": {"spread": 3.0, "total": 52.5}, "proj": {"margin": 4.0, "total": 50.0}}
    assert archive._bet_result({"market": "Spread", "side": "home", "odds": -110}, g, 3, 50)[0] == "P"
    assert archive._bet_result({"market": "Total", "side": "under", "odds": -110}, g, 3, 50)[0] == "W"
    d = {"games": {"x": {**g, "bets": [{"market": "Spread", "side": "home", "prob": 0.55, "stake": 0.01, "odds": -110}]}},
         "results": {"x": {"home_score": 30, "away_score": 20, "bets": [{"market": "Spread", "side": "home", "result": "W", "units": 0.909}]}}}
    s = archive.summarize(d)
    assert s["winners"] == [1, 0] and s["miss"]["margin"] == 6.0 and s["miss"]["line_total"] == 2.5


def test_designations_file_is_valid():
    """docs/cfb/designations.json (injury designations shown next to names; display only)."""
    import datetime as dt
    import json
    docs = Path(__file__).resolve().parent.parent / "docs" / "cfb"
    path = docs / "designations.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    dt.date.fromisoformat(d["checked"])
    assert d.get("source") and isinstance(d["players"], list)
    data = docs / "data.json"
    teams = set(json.loads(data.read_text())["meta"]["colors"]) if data.exists() else set()   # the board's team abbreviations
    seen = set()
    for r in d["players"]:
        assert r["d"] in {"O", "D", "Q", "P"}, r
        assert r["name"].strip() and r["team"].strip(), r
        if teams:
            assert r["team"] in teams, f"{r['team']} is not a board team abbreviation"
        assert isinstance(r.get("season", False), bool), r
        if r.get("updated"):
            dt.date.fromisoformat(r["updated"])
        assert (r["team"], r["name"]) not in seen, f"duplicate {r['team']} {r['name']}"
        seen.add((r["team"], r["name"]))


def test_designations_rules(tmp_path):
    """Out/Doubtful take a player out of his team's next game only ("season": every upcoming
    game); Questionable doesn't; a stale file is ignored."""
    import datetime as dt
    import json
    from cfbmodel import designations as D
    f = tmp_path / "designations.json"
    f.write_text(json.dumps({"source": "Covers.com", "checked": "2026-10-21", "players": [
        {"team": "AAA", "name": "Joe Smith Jr.", "pos": "WR", "d": "O"},
        {"team": "AAA", "name": "Al Q", "pos": "RB", "d": "Q"},
        {"team": "BBB", "name": "Big Lineman", "pos": "OT", "d": "D", "season": True}]}))
    none = tmp_path / "no-availability.json"
    assert D.load(f, today=dt.date(2026, 10, 25), avail_path=none).empty    # more than 3 days old
    des = D.load(f, today=dt.date(2026, 10, 22), avail_path=none)
    out = D.outs(des, {"Team A": "AAA", "Team B": "BBB"})
    assert sorted(out["name"]) == ["Big Lineman", "Joe Smith Jr."]          # Q is not out
    up = pd.DataFrame({"game_id": ["g1", "g2", "g1", "g3"], "team": ["Team A", "Team A", "Team B", "Team B"], "t": [1, 2, 1, 2]})
    cov = D.applies(out, up)
    assert set(zip(cov["name"], cov["game_id"])) == {("Joe Smith Jr.", "g1"), ("Big Lineman", "g1"), ("Big Lineman", "g3")}
    rows = pd.DataFrame({"game_id": ["g1", "g1", "g2"], "team": ["Team A"] * 3, "t": [1, 1, 2], "player_id": [1, 2, 1]})
    info = pd.DataFrame({"name": ["Joe Smith", "Other Guy"]}, index=[1, 2])
    po = D.player_outs(rows, out, info)
    assert list(zip(po["game_id"], po["player_id"], po["status"])) == [("g1", 1, "Out")]


def test_board_out_players_have_no_projection():
    """A designation covers one game: Questionable shows only there, and a player listed Out
    (here one the player table still has as active) never gets a projection for that game."""
    import datetime as dt
    import json
    import tempfile
    from cfbmodel import designations as D, players as P
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "designations.json"
        f.write_text(json.dumps({"source": "Covers.com", "checked": "2026-10-21", "players": [
            {"team": "AAA", "name": "Out Guy", "pos": "WR", "d": "O"},
            {"team": "AAA", "name": "Q Guy", "pos": "RB", "d": "Q"}]}))
        des = D.listed(D.load(f, today=dt.date(2026, 10, 21), avail_path=Path(tmp) / "none.json"), {"Team A": "AAA", "Team B": "BBB"})
    base = {"team": "Team A", "opp": "Team B", "season": 2026, "active": 1, "played": False, "appeared": 0,
            "proj_team_att": 30, "proj_team_car": 35, "imp_pts": 28, "margin": 3, "wind": 5, "def_ypa": 1, "def_ypc": 1,
            "group": "WR", "position": "WR", "att_share": 0, "rec_share": 0.2, "car_share": 0.1, "rec_share_raw": 0.2,
            "car_share_raw": 0.1, "ypr": 12, "ypc": 4, "ypa": 7, "comp": 0.6, "one_sum": 5, "receptions_proj": 4,
            "carries_proj": 3, "rec_td_proj": 0.3, "rush_td_proj": 0.1, "carries": 0, "receptions": 0,
            "team_car": 0, "team_comp": 0, "pass_att": 0, "team_att": 0}
    base.update({s + "_proj": 1.0 for s in P.STATS})
    rows = [dict(base, game_id=g, t=t, week=w, player_id=pid) for g, t, w in (("g1", 1, 8), ("g2", 2, 9)) for pid in (1, 2, 3)]
    proj = pd.DataFrame(rows)
    info = pd.DataFrame({"name": ["Out Guy", "Q Guy", "Fine Guy"], "position": ["WR", "RB", "WR"]}, index=[1, 2, 3])
    b = P.board(proj, proj, info, {"g1": "K1", "g2": "K2"}, {"Team A": "AAA", "Team B": "BBB"}, des)
    got = {(p["game"], p["name"]): p["status"] for p in b["players"]}
    assert ("K1", "Out Guy") not in got                                   # out: no projection that week
    assert got[("K2", "Out Guy")] is None                                 # the next week he is projected, no tag
    assert got[("K1", "Q Guy")] == "Questionable" and got[("K2", "Q Guy")] is None
    assert got[("K1", "Fine Guy")] is None
    assert [o["name"] for t in b["teams"] if t["game"] == "K1" for o in t["out"]] == ["Out Guy"]


def test_board_shows_calibration_inputs():
    """Each player card carries the chain value and the recent average that the calibration
    blends, per shown stat, and the recent average is the per-game one calib_X uses."""
    from cfbmodel import players as P
    base = {"team": "Team A", "opp": "Team B", "season": 2026, "active": 1, "played": False, "appeared": 0,
            "proj_team_att": 30, "proj_team_car": 35, "imp_pts": 28, "margin": 3, "wind": 5, "def_ypa": 1, "def_ypc": 1,
            "group": "RB", "position": "RB", "att_share": 0, "rec_share": 0.1, "car_share": 0.4, "rec_share_raw": 0.1,
            "car_share_raw": 0.35, "ypr": 8, "ypc": 5, "ypa": 7, "comp": 0.6, "one_sum": 5, "receptions_proj": 2,
            "carries_proj": 14, "rec_td_proj": 0.1, "rush_td_proj": 0.5, "carries": 0, "receptions": 0,
            "team_car": 0, "team_comp": 0, "pass_att": 0, "team_att": 0, "start_sum": 0,
            "game_id": "g1", "t": 1, "week": 8, "player_id": 1}
    base.update({s + "_proj": 1.0 for s in P.STATS})
    base.update({s + "_p": 2.0 for s in P.STATS})
    base.update({s + "_sum": 15.0 for s in P.STATS})
    base.update({s + "_st_sum": 0.0 for s in P.STATS})
    base.update({"rush_yds_p": 62.8, "rush_yds_sum": 305.0, "rush_yds_proj": 70.0})
    proj = pd.DataFrame([base])
    info = pd.DataFrame({"name": ["Back"], "position": ["RB"]}, index=[1])
    w = P.board(proj, proj, info, {"g1": "K1"}, {"Team A": "AAA", "Team B": "BBB"})["players"][0]["why"]
    assert w["chain"]["rush_yds"] == 62.8 and w["recent"]["rush_yds"] == 61.0
    assert set(w["chain"]) == set(w["recent"]) == {"pass_yds", "carries", "rush_yds", "receptions", "rec_yds"}


def test_board_files_have_no_projections_for_out_players():
    """On both boards' committed data, nobody out, doubtful or on IR for a game is projected in it."""
    root = Path(__file__).resolve().parent.parent / "docs"
    for f in (root / "data.json", root / "cfb" / "data.json"):
        if not f.exists():
            continue
        import json
        pl = (json.loads(f.read_text()).get("players") or {}).get("players") or []
        bad = [(p["name"], p["game"], p["status"]) for p in pl
               if str(p.get("status") or "").lower() in ("out", "doubtful", "ir") or str(p.get("status") or "").lower().startswith("injured reserve")]
        assert not bad, f"{f.parent.name}: projected while out {bad}"


def test_starters_rules(tmp_path):
    """The CFB dashboard's starting-QB list sets the starter for a team's upcoming games when
    its next game is in the list's week; another season or week, or a played game, is untouched;
    an unknown name counts only for a reported change."""
    import json
    from cfbmodel import starters as S
    games = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "season": [2026] * 4, "week": [7, 8, 9, 8], "order": [202607, 202608, 202609, 202608],
        "home_team": ["Team A", "Team A", "Team B", "Team C"], "away_team": ["Team B", "Team B", "Team A", "Team D"],
        "played": [True, False, False, False], "home_qb_id": ["10", None, None, None], "away_qb_id": ["20", None, None, None]})
    qb = pd.DataFrame({"season": [2026] * 3, "team": ["Team A", "Team A", "Team B"], "name": ["Old Starter", "New Guy Jr.", "B Starter"],
                       "qb_id": ["10", "11", "20"]})
    lst = {"season": 2026, "week": 8, "checked": "2026-10-22", "starters": {
        "Team A": {"qb": "New Guy", "status": "reported"},
        "Team B": {"qb": "B Starter", "status": "no change"},
        "Team C": {"qb": "Never Threw", "status": "uncertain"},
        "Team D": {"qb": "Freshman", "status": "confirmed"}}}
    f = tmp_path / "starters.json"
    f.write_text(json.dumps(lst))
    picks = S.resolve(S.load(f), games, qb, verbose=False)
    got = set(zip(picks["game_id"], picks["team"], picks["qb_id"]))
    assert got == {(2, "Team A", "11"), (3, "Team A", "11"), (2, "Team B", "20"), (3, "Team B", "20"), (4, "Team D", "replacement")}
    g = S.apply(games, picks, verbose=False)
    assert g.loc[0, "home_qb_id"] == "10"                       # played game untouched
    assert g.loc[1, "home_qb_id"] == "11" and g.loc[2, "away_qb_id"] == "11"
    assert S.resolve({**lst, "week": 9}, games, qb, verbose=False).empty      # list for another week
    assert S.resolve({**lst, "season": 2025}, games, qb, verbose=False).empty


def test_starters_file_is_valid():
    """docs/cfb/starters.json, the saved copy of the CFB dashboard's list (if present)."""
    import datetime as dt
    import json
    path = Path(__file__).resolve().parent.parent / "docs" / "cfb" / "starters.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    dt.date.fromisoformat(d["checked"])
    assert isinstance(d["season"], int) and isinstance(d["week"], int)
    for team, e in d["starters"].items():
        assert team.strip() and e["qb"].strip(), team
        assert e.get("status") in {"confirmed", "reported", "uncertain", "no change"}, (team, e)


def test_designations_played_through():
    """A designation the player has played through is dropped; one posted after his last game,
    or one with no date, stands."""
    from cfbmodel import designations as D
    des = pd.DataFrame({"team": ["AAA"] * 4, "team_name": ["Team A"] * 4, "d": ["O", "O", "D", "Q"],
                        "name": ["Old Out", "New Out", "No Date", "Never Played"],
                        "updated": ["2026-10-04", "2026-10-19", None, "2026-10-04"]})
    des["key"] = des["name"].map(D.norm)
    played = pd.DataFrame({"team_name": ["Team A"] * 3, "key": [D.norm("Old Out"), D.norm("New Out"), D.norm("No Date")],
                           "date": pd.to_datetime(["2026-10-17", "2026-10-17", "2026-10-17"])})
    kept = D.drop_played(des, played, verbose=False)
    assert sorted(kept["name"]) == ["Never Played", "New Out", "No Date"]
    other = played.assign(team_name="Team B")                        # played for another team: stands
    assert len(D.drop_played(des, other, verbose=False)) == 4


def test_spread_card_covers_every_model_feature():
    from cfbmodel import board, model
    from nflmodel.ratings import METRICS
    from cfbmodel.extra import RATED
    assert board.MatchupBuilder.uncovered(model.MARGIN_FEATS, model.TOTAL_FEATS, METRICS + ("snaps",) + RATED) == []


def test_archive_fills_the_whole_box_score(tmp_path):
    """Graded with only the projected players' lines, a later run with player names keeps every
    player who had a touch, naming the ones not projected (team rows are left out)."""
    import json
    from cfbmodel import archive as A
    d = {"season": 2026, "week": 3, "games": {"2026_03_X_Y": {"id": "2026_03_X_Y", "game_id": 401, "home": "Y", "away": "X",
                                                               "market": {"spread": 3.0, "total": 50.5}, "proj": {"margin": 2.0, "total": 48.0}, "bets": []}},
         "players": {"2026_03_X_Y": [{"id": "11", "team": "Y"}]}, "teams": {}, "results": {}, "actuals": {}}
    (tmp_path / "2026_w03.json").write_text(json.dumps(d))
    games = pd.DataFrame([{"game_id": 401, "home_id": 1, "away_id": 2, "home_score": 24.0, "away_score": 20.0}])
    pg = pd.DataFrame([{"game_id": 401, "team_id": t, "player_id": i, **{k: v for k in A.BOX}}
                       for t, i, v in ((1, 11, 1.0), (2, 22, 2.0), (1, -5866, 5.0), (2, 33, 0.0))])
    assert A.grade(games, pg, tmp_path) == 1
    a = json.loads((tmp_path / "2026_w03.json").read_text())
    assert set(a["actuals"]["2026_03_X_Y"]) == {"11"} and "full" not in a
    info = pd.DataFrame({"name": ["Eleven", "Twenty Two", " Team", "Nobody"], "position": ["QB", "WR", np.nan, "RB"]}, index=[11, 22, -5866, 33])
    assert A.grade(games, pg, tmp_path, info=info) == 0
    a = json.loads((tmp_path / "2026_w03.json").read_text())
    act = a["actuals"]["2026_03_X_Y"]
    assert a["full"] == ["2026_03_X_Y"] and set(act) == {"11", "22"}
    assert act["22"]["name"] == "Twenty Two" and act["22"]["pos"] == "WR" and act["22"]["team"] == "X" and "name" not in act["11"]


def test_rushing_is_calibrated_per_position_group():
    """Rushing yards and carries get a calibration fit per position group (RB; QB; WR+TE); a
    group's fit is untouched by the other groups' games. Passing stays fitted on quarterbacks."""
    import numpy as np
    from cfbmodel import players as P
    assert set(P.GROUP_STATS) == {"rush_yds", "carries", "receptions", "rec_yds"}
    assert P.CALIB_GROUPS["rush_yds"] == (("RB",), ("QB",), ("WR", "TE"))
    assert P.CALIB_GROUPS["rec_yds"] == (("WR",), ("TE",), ("RB",))
    rng = np.random.default_rng(0)
    rows = []
    # team carries 35; shares 0.5 and 0.1 (normalised to 0.588 / 0.118); yards per carry 5u, so the
    # chain is about 103u for the back and 21u for the receiver; the back runs for 1x his chain,
    # the receiver for 3x his
    for season in (2015, 2016, 2017):
        for g in range(80):
            u = rng.uniform(0.6, 1.4)
            for grp, share, mult in (("RB", 0.5, 1.0), ("WR", 0.1, 3.0)):
                r = {s: 0.0 for s in P.STATS}
                r.update({s + "_p": 0.0 for s in P.STATS}); r.update({s + "_sum": 0.0 for s in P.STATS}); r.update({s + "_st_sum": 0.0 for s in P.STATS})
                chain = share / 0.85 * 35 * 5 * u
                r.update({"season": season, "game_id": f"{season}_{g}", "team": "A", "player_id": f"{grp}{g}", "group": grp, "active": 1,
                          "played": season < 2017, "appeared": 1, "started": 0, "att_share_raw": 0.0, "car_share_raw": share, "rec_share_raw": 0.2,
                          "one_sum": 4.0, "start_sum": 0.0, "proj_total": 50.0, "team_att": 30.0, "team_car": 35.0, "team_car_avg": 35.0, "team_att_avg": 30.0,
                          "opp_att_allowed": 30.0, "opp_car_allowed": 35.0, "margin": 0.0, "home": 1, "wind": 5.0,
                          "ypc": 5 * u, "ypr": 8.0, "ypa": 7.0, "comp": 0.6, "def_ypc": 1.0, "def_ypa": 1.0, "def_comp": 1.0, "team_comp_rate": 0.6,
                          "imp_pts": 28.0, "ptd_share": 0.5, "ctd_rate": 0.02, "rtd_rate": 0.05, "int_rate": 0.02,
                          "rush_yds_sum": 4 * chain * mult, "carries_sum": 4 * chain / 5,
                          "rush_yds": chain * mult + rng.normal(0, 2), "carries": chain / 5})
                rows.append(r)
    t = pd.DataFrame(rows)
    out = P.walk_forward(t, first_test=2017)
    cur = out[out.season == 2017]
    rb, wr = cur[cur.group == "RB"], cur[cur.group == "WR"]
    assert np.allclose(rb.rush_yds_proj / rb.rush_yds_p, 1.0, atol=0.1)       # backs calibrated on backs: about 1x the chain
    assert np.allclose(wr.rush_yds_proj / wr.rush_yds_p, 3.0, atol=0.3)       # receivers on receivers: about 3x, not dragged toward 1x


def test_coach_change_flag_is_mid_season_only(tmp_path):
    """extra.coach_flags: 1 for a team whose head coach changed after Aug 31 of the season and before
    the game; an offseason change is not a mid-season one; home minus away for the margin."""
    from cfbmodel import extra
    f = tmp_path / "coach.csv"
    f.write_text("team,date,incoming,interim\nA,2026-10-21,X,yes\nB,2026-02-18,Y,yes\n")
    games = pd.DataFrame({"game_id": [1, 2, 3], "season": [2026] * 3, "gameday": pd.to_datetime(["2026-10-17", "2026-10-31", "2026-11-07"]),
                          "home_team": ["A", "A", "C"], "away_team": ["B", "C", "A"]})
    c = extra.coach_flags(games, path=f).set_index("game_id")
    assert list(c["d_mid"]) == [0.0, 1.0, -1.0] and list(c["s_mid"]) == [0.0, 1.0, 1.0]
    frame = extra.attach_game_features(games.copy(), games, None, c.reset_index())
    assert list(frame["d_mid"]) == [0.0, 1.0, -1.0]
    assert {"d_miss_defcov", "d_dve", "d_mid"} <= set(__import__("cfbmodel.model", fromlist=["model"]).MARGIN_FEATS)


def test_college_legs_read_the_college_prop_widths():
    """parlays.prop_params_path: the college legs read data/cfb/prop_params.json (college-fitted
    widths, Nov 2026) when it exists; the NFL legs always read the NFL file."""
    import json
    from nflmodel import config as C, parlays, props as P
    assert parlays.prop_params_path("nfl") == P.PARAMS_PATH
    cfb = C.DATA / "cfb" / "prop_params.json"
    assert parlays.prop_params_path("cfb") == cfb and cfb.exists()
    a, b = json.loads(P.PARAMS_PATH.read_text()), json.loads(cfb.read_text())
    assert b["isotonic"] == a["isotonic"] and b["family"] == a["family"]      # widths only: the correction is the NFL's
    assert b["yards"] != a["yards"] and set(b["yards"]) == set(a["yards"])


def test_designations_conference_reports_beat_covers(tmp_path):
    """docs/cfb/availability.json (the conferences' official reports) replaces a team's Covers rows,
    keeps Covers rows for other teams and season-long Covers outs the report doesn't list, maps the
    conferences' statuses to the board's codes, and is ignored once the game has been played or the
    file is old."""
    import datetime as dt
    import json
    from cfbmodel import designations as D
    cov = tmp_path / "designations.json"
    cov.write_text(json.dumps({"source": "Covers.com", "checked": "2026-11-07", "players": [
        {"team": "AAA", "name": "Stale Guy", "pos": "WR", "d": "Q", "updated": "2026-10-17"},
        {"team": "AAA", "name": "Gone Guy", "pos": "LB", "d": "O", "season": True},
        {"team": "AAA", "name": "Listed Guy", "pos": "RB", "d": "O", "season": True},
        {"team": "CCC", "name": "Covers Guy", "pos": "QB", "d": "D"},
        {"team": "DDD", "name": "Fine Now", "pos": "WR", "d": "Q"}]}))
    av = tmp_path / "availability.json"
    av.write_text(json.dumps({"fetched": "2026-11-07T01:20Z", "teams": {
        "Team A": {"conf": "SEC", "abbr": "AAA", "report": "Update", "published": "2026-11-06", "game_date": "2026-11-07", "players": [
            {"name": "Listed Guy", "pos": "RB", "status": "Probable", "season_out": False},
            {"name": "Hurt Guy", "pos": "WR", "status": "Out", "season_out": True},
            {"name": "Maybe Guy", "pos": "TE", "status": "Game Time Decision", "season_out": False},
            {"name": "Half Guy", "pos": "DB", "status": "Out (1st Half)", "season_out": False}]},
        "Team B": {"conf": "ACC", "abbr": "BBB", "report": "Game Day", "published": "2026-11-07", "game_date": "2026-11-07", "players": [
            {"name": "Doubt Guy", "pos": "QB", "status": "Doubtful", "season_out": False}]},
        "Team D": {"conf": "Big 12", "abbr": "DDD", "report": "Game Day", "published": "2026-11-07", "game_date": "2026-11-07", "players": []}}}))
    des = D.load(cov, today=dt.date(2026, 11, 7), avail_path=av)
    got = {(r.team, r.name): (r.d, r.src, bool(r.season)) for r in des.itertuples()}
    assert ("AAA", "Stale Guy") not in got                      # Covers row replaced by the report
    assert got[("AAA", "Gone Guy")] == ("O", "Covers", True)   # season-long Covers out, not on the report: kept
    assert got[("AAA", "Listed Guy")] == ("P", "SEC", False)   # on the report: the report's status wins
    assert got[("AAA", "Hurt Guy")] == ("O", "SEC", True)
    assert got[("AAA", "Maybe Guy")] == ("Q", "SEC", False)    # Game Time Decision
    assert got[("AAA", "Half Guy")] == ("P", "SEC", False)     # Out (1st Half): plays the second half
    assert got[("BBB", "Doubt Guy")] == ("D", "ACC", False)
    assert got[("CCC", "Covers Guy")] == ("D", "Covers", False)  # no report for this team: Covers stands
    assert ("DDD", "Fine Now") not in got                       # a live report listing nobody: everyone available
    assert list(des.columns) == D.COLS
    # the game has been played: the report no longer applies, Covers (still fresh) stands alone
    later = D.load(cov, today=dt.date(2026, 11, 8), avail_path=av)
    assert set(later["src"]) == {"Covers"} and ("AAA", "Stale Guy") in set(zip(later["team"], later["name"]))
    # an old availability file is ignored even if its game date were ahead
    av.write_text(json.dumps({"fetched": "2026-10-20T01:20Z", "teams": {"Team B": {"conf": "ACC", "abbr": "BBB", "report": "Initial",
                  "published": "2026-10-19", "game_date": "2026-11-14", "players": [{"name": "X", "pos": "QB", "status": "Out"}]}}}))
    assert "BBB" not in set(D.load(cov, today=dt.date(2026, 11, 7), avail_path=av)["team"])
    # reports alone, no Covers file
    assert D.load(tmp_path / "missing.json", today=dt.date(2026, 11, 7), avail_path=av).empty


def test_current_reports_from_vendor_payload():
    """availability_log.current_reports: the newest live report per team from the vendor app's
    publish payload, in the shape designations.load_reports reads; Exempt rows and Report Pending
    dropped, played games dropped."""
    import datetime as dt
    from cfbmodel import availability_log as A
    rep = lambda rid, rtype, pub, posted, gdate, games: {rid: {"ReportType": rtype, "publishDate": pub, "postedTime": posted,
                                                              "footer": {"date": gdate, "time": "19:00:00", "location": "X"}, "games": games}}
    teamA = lambda rows: {"teamName": "Team A", "teamDisplayName": "Team A", "rows": rows}
    teamB = lambda rows: {"teamName": "Team B", "teamDisplayName": "Team B", "rows": rows}
    sec = {}
    sec.update(rep("1", "Initial", "2026-11-04", "19:10:00", "2026-11-07", [teamA([{"name": "WR #1 Hurt Guy", "status": "Questionable", "exemptStatus": "NonExempt"}]), teamB([])]))
    sec.update(rep("2", "Update", "2026-11-05", "19:10:00", "2026-11-07", [teamA([{"name": "WR #1 Hurt Guy", "status": "Out", "exemptStatus": "Out"},
                                                                                   {"name": "OL #70 Walk On", "status": "Exempt", "exemptStatus": "Exempt"}]), teamB([])]))
    sec.update(rep("3", "Report Pending", "2026-11-06", "00:00:00", "2026-11-07", [teamA([]), teamB([])]))
    sec.update(rep("4", "Game Day", "2026-10-31", "10:00:00", "2026-10-31", [teamA([{"name": "QB #7 Old Guy", "status": "Out", "exemptStatus": "NonExempt"}]), teamB([])]))
    cur = A.current_reports({"SEC": sec}, "2026-11-06T01:20Z", today=dt.date(2026, 11, 6), abbr={"Team A": "AAA", "Team B": "BBB"})
    a = cur["teams"]["Team A"]
    assert (a["report"], a["published"], a["conf"], a["abbr"], a["opponent"]) == ("Update", "2026-11-05", "SEC", "AAA", "Team B")
    assert a["players"] == [{"name": "Hurt Guy", "pos": "WR", "number": "1", "status": "Out", "season_out": True}]
    assert cur["teams"]["Team B"]["players"] == []          # a live report with nobody listed: everyone available
    assert A.current_reports({"SEC": sec}, "x", today=dt.date(2026, 11, 8), abbr={})["teams"] == {}   # played


def test_availability_shade_and_props_use_played():
    """An official Questionable / GTD player keeps PLAY_CHANCE of his receptions, receiving yards,
    carries and rushing yards; undesignated teammates pick up the rest in proportion; passing is left
    alone; Covers' Questionable is untouched; the college props read his full projection."""
    from cfbmodel import designations as D, players as P
    from nflmodel.cfbpropboard import with_target_share
    rows = [  # game g1, Team A: Q WR (official), free WR, free RB; Team B: Covers-Q WR
        {"game_id": "g1", "team": "Team A", "name": "Q Guy", "receptions_proj": 6.0, "rec_yds_proj": 80.0, "carries_proj": 0.0, "rush_yds_proj": 0.0, "pass_yds_proj": 0.0},
        {"game_id": "g1", "team": "Team A", "name": "Free Wr", "receptions_proj": 4.0, "rec_yds_proj": 40.0, "carries_proj": 0.0, "rush_yds_proj": 0.0, "pass_yds_proj": 0.0},
        {"game_id": "g1", "team": "Team A", "name": "Free Rb", "receptions_proj": 2.0, "rec_yds_proj": 10.0, "carries_proj": 15.0, "rush_yds_proj": 70.0, "pass_yds_proj": 0.0},
        {"game_id": "g1", "team": "Team B", "name": "Covers Q", "receptions_proj": 5.0, "rec_yds_proj": 60.0, "carries_proj": 0.0, "rush_yds_proj": 0.0, "pass_yds_proj": 0.0},
    ]
    p = pd.DataFrame(rows)
    p["play_chance"] = [D.PLAY_CHANCE["Questionable"], None, None, None]
    tag = {("g1", "Team A", D.norm("Q Guy")): "Q", ("g1", "Team B", D.norm("Covers Q")): "Q"}
    out, played, gained = P.shade_for_availability(p, tag)
    c = D.PLAY_CHANCE["Questionable"]
    assert abs(out.at[0, "receptions_proj"] - 6.0 * c) < 1e-9
    lost = 6.0 * (1 - c)
    assert abs(out.at[1, "receptions_proj"] - (4.0 + lost * 4 / 6)) < 1e-9      # in proportion to 4 : 2
    assert abs(out.at[2, "receptions_proj"] - (2.0 + lost * 2 / 6)) < 1e-9
    assert abs(out["receptions_proj"][:3].sum() - 12.0) < 1e-9                    # team total unchanged
    assert out.at[2, "carries_proj"] == 15.0                                       # nothing lost on the ground
    assert out.at[3, "receptions_proj"] == 5.0                                     # Covers' Q untouched
    assert played[0]["receptions"] == 6.0 and set(played) == {0}
    assert set(gained) == {1, 2} and abs(gained[1]["receptions"] - lost * 4 / 6) < 1e-9
    payload = {"players": {"players": [{"id": "1", "proj": {"receptions": 4.0}, "proj_played": {"receptions": 6.0},
                                        "play_chance": c, "why": {"rec_share": 0.3}},
                                       {"id": "2", "proj": {"receptions": 3.0}, "why": {}}]}}
    w = with_target_share(payload)
    assert w["players"]["players"][0]["proj"]["receptions"] == 6.0                 # props price given he plays
    assert w["players"]["players"][1]["proj"]["receptions"] == 3.0
    assert payload["players"]["players"][0]["proj"]["receptions"] == 4.0           # the board's copy unchanged
    assert D.PLAY_CHANCE == {"Questionable": 0.66, "Game Time Decision": 0.64}
