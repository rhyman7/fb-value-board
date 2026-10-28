"""Run with:  python -m pytest -q   (or just: python tests/test_model.py)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import bets, distributions as D, odds as O  # noqa: E402

close = lambda a, b, tol=1e-4: abs(float(a) - float(b)) < tol


def test_odds_conversions():
    assert close(O.implied_prob(-110), 110 / 210)
    assert close(O.implied_prob(150), 0.4)
    assert close(O.american_to_decimal(-110), 1 + 100 / 110)
    a, b = O.no_vig(-110, -110)
    assert close(a, 0.5) and close(b, 0.5)
    a, b = O.no_vig(-180, 150)
    assert close(a + b, 1)
    assert close(O.hold(-110, -110), 2 * 110 / 210 - 1)


def test_expected_value_and_kelly():
    assert close(O.expected_value(0.55, -110), 0.55 * 100 / 110 - 0.45)
    assert close(O.expected_value(0.5, -110), -0.04545)
    assert close(O.expected_value(0.5, -110, p_push=0.1), 0.5 * 100 / 110 - 0.4)
    assert close(O.kelly(0.5, -110), 0.0)                       # no edge, no bet
    assert close(O.kelly(0.55, 100), 0.10)                      # classic: p - q at even money


def test_distribution_sums_and_anchor():
    mult = np.ones(len(D.MARGINS))
    mult[D.MARGINS == 3] = mult[D.MARGINS == -3] = 2.5          # a key number
    p = D.pmf([3.0, -6.5], 11.5, D.MARGINS, mult)
    assert np.allclose(p.sum(axis=1), 1)
    over, push, under = D.over_push_under(p, D.MARGINS, [3.0, -6.5])
    assert push[0] > 0.05 and push[1] == 0                      # whole numbers push, half points cannot
    # anchoring: with no disagreement, each side matches the target probability
    for target in (0.5, 0.52):
        loc = D.anchor([3.0, 7.0, -2.5], target, 11.5, D.MARGINS, mult)
        o, pu, u = D.over_push_under(D.pmf(loc, 11.5, D.MARGINS, mult), D.MARGINS, [3.0, 7.0, -2.5])
        assert np.allclose(o / (1 - pu), target, atol=2e-3)


def _one_game(**kw):
    base = dict(game_id="g", season=2026, week=1, home_team="HOM", away_team="AWY", spread_line=3.0, total_line=44.5,
                home_spread_odds=-110.0, away_spread_odds=-110.0, over_odds=-110.0, under_odds=-110.0,
                home_moneyline=-160.0, away_moneyline=135.0, model_margin=3.0, model_total=44.5,
                proj_margin=3.0, proj_total=44.5)
    base.update(kw)
    return pd.DataFrame([base])


DIST = {"margin": np.ones(len(D.MARGINS)), "total": np.ones(len(D.TOTALS)), "sd_margin": 11.5, "sd_total": 13.0, "sd_win": 11.4}


def test_no_disagreement_means_no_edge():
    b = bets.price_games(_one_game(), DIST, "proj")
    assert len(b) == 6
    assert np.allclose(b["edge"], 0, atol=3e-3)                 # model agrees with the no-vig market
    assert (b["ev"] < 0).all()                                  # so every bet just pays the vig
    assert (b["stake"] == 0).all()


def test_disagreement_moves_the_right_side():
    b = bets.price_games(_one_game(proj_margin=6.0, proj_total=48.0), DIST, "proj").set_index("pick")
    assert b.loc["HOM -3", "edge"] > 0 > b.loc["AWY +3", "edge"]
    assert b.loc["HOM ML", "edge"] > 0 > b.loc["AWY ML", "edge"]
    assert b.loc["Over 44.5", "edge"] > 0 > b.loc["Under 44.5", "edge"]
    for a, c in (("HOM -3", "AWY +3"), ("HOM ML", "AWY ML"), ("Over 44.5", "Under 44.5")):
        assert close(b.loc[a, "model_prob"] + b.loc[c, "model_prob"], 1, 1e-6)


def test_grading():
    g = _one_game(result=3.0, total=50.0, proj_margin=6.0)
    b = bets.price_games(g, DIST, "proj").set_index("pick")
    assert b.loc["HOM -3", "profit"] == 0 and b.loc["AWY +3", "profit"] == 0     # landed on the number: push
    assert close(b.loc["Over 44.5", "profit"], 100 / 110) and b.loc["Under 44.5", "profit"] == -1
    assert close(b.loc["HOM ML", "profit"], 100 / 160) and b.loc["AWY ML", "profit"] == -1


def test_ratings_do_not_see_the_future():
    """Ratings for a week must be identical whether or not later games exist."""
    from nflmodel import config as C, data
    from nflmodel.ratings import RatingEngine
    if not (C.CACHE / "team_games_2024.csv").exists():
        return                                                   # no data downloaded yet
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    cut = 202410
    full = RatingEngine(tg, qb, games=games).ratings_at(cut)
    past = RatingEngine(tg[tg.order < cut], qb[qb.order < cut], games=games[games.order < cut]).ratings_at(cut)
    for m in ("pass", "rush", "sr", "pts"):
        assert np.allclose(full[m]["off"], past[m]["off"]) and np.allclose(full[m]["def"], past[m]["def"])
    assert np.allclose(full["mkt"]["R"], past["mkt"]["R"]) and np.allclose(full["qb_base"], past["qb_base"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def _ctx_games():
    return pd.DataFrame({
        "game_id": ["a", "b", "c"], "home_team": ["NYG", "BUF", "JAX"], "away_team": ["SEA", "MIA", "HOU"],
        "neutral": [0, 0, 1], "indoor": [0, 0, 0], "played": [True, True, True], "gametime": ["13:00", "20:20", "09:30"],
        "wind": [12.0, np.nan, 5.0], "temp": [50.0, 30.0, 60.0], "hfa": [1.5, 1.5, 0.0], "div_game": [0, 1, 1],
    })


def test_context_features():
    from nflmodel.features_ext import add_context
    d = add_context(_ctx_games())
    assert d.loc[2, "travel"] == 0 and d.loc[2, "tz_east"] == 0          # neutral site: nobody is the traveller
    assert d.loc[0, "travel"] > 3.5 and d.loc[0, "tz_east"] == 3          # Seattle flying to New Jersey
    assert d.loc[0, "west_early"] == 1                                    # Pacific team, 1 p.m. Eastern
    assert d.loc[1, "wind_known"] == 0 and d.loc[1, "wind"] > 0           # blank wind becomes the median, flagged unknown
    assert d.loc[1, "cold"] == 15 and d.loc[1, "hfa_div"] == 1.5


def test_wind_forecast_failure_is_harmless(monkeypatch):
    from nflmodel import weather
    g = _ctx_games().assign(played=False, gameday=pd.Timestamp.now().normalize(), wind=np.nan)
    def boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(weather, "_fetch", boom)
    out = weather.fill_wind(g, verbose=False, log=False)
    assert out["wind"].isna().all() and (out["wind_source"] == "").all()
    monkeypatch.setattr(weather, "_fetch", lambda *a, **k: [float(h) for h in range(24)])
    out = weather.fill_wind(g, verbose=False, log=False)
    assert out.loc[0, "wind"] == 13 and out.loc[0, "wind_source"] == "forecast"   # kickoff hour 13
    assert np.isnan(out.loc[2, "wind"])                                            # neutral site skipped


def test_wind_log_appends(tmp_path):
    from nflmodel import weather
    path = tmp_path / "wind_log.csv"
    weather.log_forecasts({"g1": 12.34}, path)
    weather.log_forecasts({"g1": 14.0, "g2": 3.0}, path)
    log = pd.read_csv(path)
    assert len(log) == 3 and list(log.columns) == ["game_id", "fetched", "wind_mph"] and log.wind_mph.iloc[0] == 12.3


def test_injury_features():
    from nflmodel.injuries import GROUPS, add_injury_features
    g = pd.DataFrame({"game_id": ["x", "y"], "home_team": ["PHI", "NYG"], "away_team": ["CAR", "DAL"]})
    z = add_injury_features(g, None)                       # no data: every feature is zero
    assert all((z[f"d_chg_{k}"] == 0).all() and (z[f"d_miss_{k}"] == 0).all() for k in GROUPS)
    inj = pd.DataFrame([{"season": 2026, "game_id": "x", "team": "PHI", **{f"miss_{k}": 0.0 for k in GROUPS}, **{f"mrec_{k}": 0.0 for k in GROUPS}},
                        {"season": 2026, "game_id": "x", "team": "CAR", **{f"miss_{k}": 0.0 for k in GROUPS}, **{f"mrec_{k}": 0.0 for k in GROUPS}}])
    inj.loc[0, "miss_ol"], inj.loc[0, "mrec_ol"] = 0.4, 0.1   # home line: 40% out now, 10% in recent games
    f = add_injury_features(g, inj)
    assert abs(f.loc[0, "d_miss_ol"] - 0.4) < 1e-9 and abs(f.loc[0, "d_chg_ol"] - 0.3) < 1e-9
    assert f.loc[1, "d_chg_ol"] == 0                      # a game with no injury rows gets zeros


def test_blank_roof_filled_from_home_stadium(tmp_path, monkeypatch):
    from nflmodel import config as C, data
    games = pd.DataFrame({
        "season": [2025, 2025, 2026, 2026], "week": [1, 2, 7, 7], "game_type": "REG",
        "game_id": ["a", "b", "c", "d"], "gameday": ["2025-09-07", "2025-09-14", "2026-10-25", "2026-10-25"],
        "home_team": ["ATL", "CHI", "ATL", "CHI"], "away_team": ["TB", "GB", "SF", "NE"],
        "location": "Home", "roof": ["closed", "outdoors", np.nan, np.nan], "result": [3.0, -7.0, np.nan, np.nan],
    })
    raw = tmp_path / "raw"; raw.mkdir()
    games.to_csv(raw / "games.csv", index=False)
    monkeypatch.setattr(C, "RAW", raw)
    monkeypatch.setattr(C, "FIRST_SEASON", 2025)
    g = data.load_games(refresh=False).set_index("game_id")
    assert g.loc["c", "indoor"] == 1 and g.loc["d", "indoor"] == 0


def test_player_asof_uses_only_earlier_games():
    from nflmodel.players import asof
    d = pd.DataFrame({"pid": ["a"] * 3 + ["b"], "t": [1, 2, 3, 1], "season": [2025] * 4, "x": [10.0, 20.0, np.nan, 5.0]})
    r = asof(d, "pid", ["x"], None, hl=1.0).set_index(["pid", "t"])
    assert r.loc[("a", 1), "x_sum"] == 0 and r.loc[("b", 1), "x_sum"] == 0      # nothing before the first game
    assert abs(r.loc[("a", 2), "x_sum"] - 10 * 0.5) < 1e-9                      # one game ago, half-life 1
    assert abs(r.loc[("a", 3), "x_sum"] - (20 * 0.5 + 10 * 0.25)) < 1e-9        # a game with no value adds nothing


def test_player_status_label():
    from nflmodel.players import status_label
    assert status_label("RES", "Out") == "Injured reserve"
    assert status_label("ACT", "Questionable") == "Questionable"
    assert status_label("INA", None) == "Inactive"
    assert status_label("INA", None, upcoming=True) is None          # last week's inactive list says nothing about this week
    assert status_label("ACT", np.nan) is None


def test_player_shares_and_starter():
    from nflmodel import players as PL
    base = dict(game_id="g", team="PHI", opp="DAL", season=2026, week=7, t=1, played=False, imp_pts=21.0, ptd_share=0.6,
                catch=0.65, ypt=8.0, ypc=4.2, ypa=7.0, comp=0.64, int_rate=0.02, def_ypt=1, def_catch=1, def_ypc=1, def_ypa=1, def_comp=1,
                rz_tgt_rate=0.1, rz_car_rate=0.1, gl_car_rate=0.03, car_share_raw=0.0, att_share_raw=0.0, tgt_share_raw=0.0,
                prev_qb="hurts")
    rows = [dict(base, player_id="hurts", group="QB", active=0, att_share_raw=0.85),
            dict(base, player_id="mckee", group="QB", active=1, att_share_raw=0.26),
            dict(base, player_id="third", group="QB", active=1, att_share_raw=0.10),
            dict(base, player_id="wr1", group="WR", active=1, tgt_share_raw=0.30),
            dict(base, player_id="wr2", group="WR", active=1, tgt_share_raw=0.20),
            dict(base, player_id="wr3", group="WR", active=0, tgt_share_raw=0.25)]
    p = pd.DataFrame(rows)
    for c in PL.VOL_FEATS:
        p[c] = 1.0
    const = type("M", (), {"predict": lambda self, X: np.full(len(X), 35.0)})()
    q = PL.raw_projection(p, const, const).set_index("player_id")
    assert q.loc["mckee", "att_share"] == 1 and q.loc["third", "att_share"] == 0 and q.loc["hurts", "att_share"] == 0
    assert q.loc["wr3", "tgt_share"] == 0                                   # out: no targets
    assert abs(q.loc["wr1", "tgt_share"] / q.loc["wr2", "tgt_share"] - 1.5) < 1e-9   # the rest keep their order
    assert q.loc["wr1", "tgt_share"] > 0.30                                 # and pick up the missing receiver's targets

    # starter choice: last game's starter keeps the job unless the clear regular is back
    def starter(qbs, prev):
        d = pd.DataFrame([dict(base, player_id=n, group="QB", active=1, att_share_raw=s, prev_qb=prev) for n, s in qbs])
        for c in PL.VOL_FEATS:
            d[c] = 1.0
        r = PL.raw_projection(d, const, const)
        return r.loc[r["att_share"] == 1, "player_id"].tolist()
    assert starter([("vet", 0.6), ("rookie", 0.4)], "rookie") == ["rookie"]      # benching sticks
    assert starter([("hurts", 0.85), ("mckee", 0.3)], "mckee") == ["hurts"]      # regular returns from injury
    assert starter([("a", 0.6), ("b", 0.3)], None) == ["a"]                      # no previous starter: largest share


def test_archive_freezes_at_kickoff_and_grades(tmp_path):
    from nflmodel import archive as A
    games = pd.DataFrame([{"game_id": "2026_07_A_B", "gameday": pd.Timestamp("2026-10-25"), "gametime": "13:00",
                           "home_score": np.nan, "away_score": np.nan, "home_qb_id": None, "away_qb_id": None}])
    bets = [{"market": "Spread", "side": "home", "pick": "B -3", "odds": -110, "prob": 0.55, "stake": 0.01},
            {"market": "Spread", "side": "away", "pick": "A +3", "odds": -110, "prob": 0.45, "stake": 0.0},
            {"market": "Total", "side": "over", "pick": "Over 44.5", "odds": -110, "prob": 0.52, "stake": 0.0},
            {"market": "Moneyline", "side": "away", "pick": "A ML", "odds": 140, "prob": 0.4, "stake": 0.0}]
    def out(margin, gen):
        g = {"id": "2026_07_A_B", "week": 7, "away": "A", "home": "B", "market": {"spread": 3.0, "total": 44.5},
             "proj": {"margin": margin, "total": 45.0}, "bets": bets}
        return {"meta": {"season": 2026, "generated": gen}, "slate": {"games": [g]},
                "players": {"players": [{"id": "p1", "game": "2026_07_A_B", "team": "B"}], "teams": []}}
    before = pd.Timestamp("2026-10-25 16:00", tz="UTC")       # 12:00 ET, an hour before kickoff
    after = pd.Timestamp("2026-10-25 18:00", tz="UTC")
    assert A.snapshot(out(3.5, "first"), games, before, tmp_path) == 1
    assert A.snapshot(out(9.9, "late"), games, after, tmp_path) == 0      # kicked off: frozen
    d = json.loads((tmp_path / "2026_w07.json").read_text())
    assert d["games"]["2026_07_A_B"]["proj"]["margin"] == 3.5 and d["games"]["2026_07_A_B"]["frozen_at"] == "first"
    games.loc[0, ["home_score", "away_score"]] = [24, 20]                  # B by 4: covers -3, under 44.5
    pg = pd.DataFrame([{"game_id": "2026_07_A_B", "player_id": "p1", **{k: 1.0 for k in A.BOX}}])
    assert A.grade(games, pg, tmp_path) == 1
    d = json.loads((tmp_path / "2026_w07.json").read_text())
    res = {(b["market"], b["side"]): (b["result"], b["units"]) for b in d["results"]["2026_07_A_B"]["bets"]}
    assert res[("Spread", "home")] == ("W", round(100 / 110, 4)) and res[("Total", "over")][0] == "L" and res[("Moneyline", "away")] == ("L", -1.0)
    assert d["actuals"]["2026_07_A_B"]["p1"]["pass_yds"] == 1.0
    w = A.write_index(tmp_path)[0]
    assert w["spread"] == [1, 0, 0] and w["total"] == [0, 1, 0] and w["bet_wlp"] == [1, 0, 0]


def test_overrides_starter_and_out(tmp_path):
    from nflmodel import overrides as OV
    p = tmp_path / "overrides.csv"
    p.write_text("game_id,team,kind,player_id,name,status,source\n"
                 "G2,SEA,starter,qb-new,New QB,,test\n"
                 "G1,SEA,starter,qb-x,Played QB,,test\n"
                 "G2,KC,out,wr-1,Some WR,,test\n")
    ov = OV.load(p)
    games = pd.DataFrame({"game_id": ["G1", "G2"], "home_team": ["SEA", "SEA"], "away_team": ["KC", "KC"],
                          "home_qb_id": ["qb-old", "qb-old"], "home_qb_name": ["Old", "Old"],
                          "away_qb_id": ["kc", "kc"], "away_qb_name": ["KC", "KC"], "played": [True, False]})
    g = OV.apply_starters(games, ov, verbose=False)
    assert list(g.home_qb_id) == ["qb-old", "qb-new"]          # a played game is never changed
    assert g.home_qb_name.iloc[1] == "New QB" and list(g.away_qb_id) == ["kc", "kc"]
    rows = pd.DataFrame({"game_id": ["G1", "G2", "G2"], "player_id": ["wr-1", "wr-1", "wr-2"],
                         "played": [True, False, False], "active": [1, 1, 1], "report": [np.nan] * 3})
    r = OV.apply_out(rows, ov)
    assert list(r.active) == [1, 0, 1] and r.report.iloc[1] == "Out"


def test_overrides_file_is_valid():
    from nflmodel import overrides as OV
    ov = OV.load()
    assert ov["player_id"].str.match(r"^00-\d{7}$").all()
    assert ov["game_id"].str.match(r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$").all()
    for r in ov.itertuples():
        assert r.team in r.game_id.split("_")[2:], f"{r.team} not in {r.game_id}"


def test_questionable_adjustment_uses_only_earlier_seasons():
    """The Questionable multiplier for a season comes from earlier seasons only: changing
    a later season's results leaves an earlier season's projections unchanged."""
    import numpy as np
    import pandas as pd
    from nflmodel import players_ml

    rng = np.random.default_rng(0)
    rows = []
    for s in (2014, 2015, 2016):
        for i in range(300):
            rows.append({"season": s, "group": "WR", "report_lvl": 1, "practice_lvl": i % 3, "played": True, "active": 1,
                         "rec_yds_proj": 50.0, "rec_yds": float(rng.poisson(35)), "receptions_proj": 4.0, "receptions": 3.0})
    p = pd.DataFrame(rows)
    a = players_ml.questionable(p, first=2015)
    p2 = p.copy()
    p2.loc[p2.season == 2016, "rec_yds"] = 200.0
    b = players_ml.questionable(p2, first=2015)
    early = p.season <= 2016
    assert np.allclose(a.loc[early, "rec_yds_proj"], b.loc[early, "rec_yds_proj"])   # 2016 is predicted from 2014-15
    assert (a.loc[p.season == 2014, "rec_yds_proj"] == 50.0).all()                    # before `first`: untouched
    assert (a.loc[p.season == 2015, "rec_yds_proj"] < 50.0).all()                     # under-used, so scaled down
    # not listed: untouched
    c = players_ml.questionable(p.assign(report_lvl=-1), first=2015)
    assert (c["rec_yds_proj"] == 50.0).all()
