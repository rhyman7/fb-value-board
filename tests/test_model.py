"""Run with:  python -m pytest -q   (or just: python tests/test_model.py)"""
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
