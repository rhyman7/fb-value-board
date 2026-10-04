"""Price every bet on a set of games: model probability vs market, EV, stake.

Used by both the backtest (played games, with results graded) and the live
slate (upcoming games). One row per side of each market, six rows per game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr, ndtri

from . import config as C
from . import distributions as D
from . import odds as O

DEFAULT_PRICE = -110.0  # used when a spread or total has no recorded price


def fit_distribution(games: pd.DataFrame, before_season: int, lookback: int = 10) -> dict:
    """Key-number multipliers and curve widths from the seasons before a season.

    Measured against closing lines, the best available estimate of where each
    game was centred.
    """
    h = games[games.played & games.spread_line.notna() & games.total_line.notna()
              & (games.season < before_season) & (games.season >= before_season - lookback)]
    # Multipliers and width depend on each other a little: fit the multipliers at
    # a typical width, fit the width, then refresh the multipliers at that width.
    mm = D.fit_multipliers(h.spread_line, h.result, 11.8, D.MARGINS, fold=True, shrink=8)
    tm = D.fit_multipliers(h.total_line, h.total, 13.1, D.TOTALS, fold=False, shrink=40)
    sd_m = D.fit_scale(h.spread_line, h.result, D.MARGINS, mm)
    sd_t = D.fit_scale(h.total_line, h.total, D.TOTALS, tm, bounds=(10.0, 17.0))
    mm = D.fit_multipliers(h.spread_line, h.result, sd_m, D.MARGINS, fold=True, shrink=8)
    tm = D.fit_multipliers(h.total_line, h.total, sd_t, D.TOTALS, fold=False, shrink=40)

    # Winning outright is its own question: fit the margin-to-win-probability
    # curve directly on wins and losses, using every season available.
    a = games[games.played & games.spread_line.notna() & (games.season < before_season) & (games.result != 0)]
    won, mu = (a.result > 0).to_numpy(), a.spread_line.to_numpy(float)
    nll = lambda sd: -np.sum(np.where(won, log_ndtr(mu / sd), log_ndtr(-mu / sd)))
    sd_w = float(minimize_scalar(nll, bounds=(8.0, 16.0), method="bounded").x)
    return {"margin": mm, "total": tm, "sd_margin": sd_m, "sd_total": sd_t, "sd_win": sd_w, "n": int(len(h))}


def _fmt(x: float, signed: bool = True) -> str:
    if signed and x == 0:
        return "PK"
    return f"{x:+g}" if signed else f"{x:g}"


def price_games(df: pd.DataFrame, dist: dict, source: str = "proj") -> pd.DataFrame:
    """Return the long table of bets for the games in `df`.

    source = "proj"   market-adjusted projection (the one to bet from)
    source = "model"  raw model, before it is pulled toward the market
    """
    df = df.reset_index(drop=True)

    def col(name):
        return df[name].to_numpy(float) if name in df else np.full(len(df), np.nan)

    def price_of(name, default=None):
        x = col(name)
        return x if default is None else np.where(np.isnan(x), default, x)

    spread = col("spread_line")                            # positive = home favoured
    total = col("total_line")
    d_margin = col(f"{source}_margin") - spread            # how far we disagree with the line
    d_total = col(f"{source}_total") - total

    # Anchor each curve to the market's own no-vig price, then slide it by the disagreement.
    fair_home, _ = O.no_vig(price_of("home_spread_odds", DEFAULT_PRICE), price_of("away_spread_odds", DEFAULT_PRICE))
    fair_over, _ = O.no_vig(price_of("over_odds", DEFAULT_PRICE), price_of("under_odds", DEFAULT_PRICE))
    loc_m = D.anchor(spread, fair_home, dist["sd_margin"], D.MARGINS, dist["margin"]) + d_margin
    loc_t = D.anchor(total, fair_over, dist["sd_total"], D.TOTALS, dist["total"]) + d_total
    pm = D.pmf(loc_m, dist["sd_margin"], D.MARGINS, dist["margin"])
    pt = D.pmf(loc_t, dist["sd_total"], D.TOTALS, dist["total"])
    h_cov, s_push, a_cov = D.over_push_under(pm, D.MARGINS, spread)
    over, t_push, under = D.over_push_under(pt, D.TOTALS, total)

    # Moneyline: start from the market's no-vig win probability, expressed as a
    # margin on the win-probability curve, and slide by the same disagreement.
    tie = pm[:, D.MARGINS == 0].ravel()
    fair_ml, _ = O.no_vig(col("home_moneyline"), col("away_moneyline"))
    implied = np.where(np.isnan(fair_ml), spread, dist["sd_win"] * ndtri(np.clip(fair_ml, 1e-4, 1 - 1e-4)))
    h_win = ndtr((implied + d_margin) / dist["sd_win"]) * (1 - tie)
    a_win = 1 - tie - h_win

    res = col("result")
    tot = col("total")
    markets = [
        # market, side, label, price column, p_win, p_push, opposite price col, outcome (+1 win, 0 push, -1 loss)
        ("Spread", "home", [f"{t} {_fmt(-s)}" for t, s in zip(df.home_team, spread)], "home_spread_odds", h_cov, s_push, "away_spread_odds", np.sign(res - spread)),
        ("Spread", "away", [f"{t} {_fmt(s)}" for t, s in zip(df.away_team, spread)], "away_spread_odds", a_cov, s_push, "home_spread_odds", np.sign(spread - res)),
        ("Moneyline", "home", [f"{t} ML" for t in df.home_team], "home_moneyline", h_win, tie, "away_moneyline", np.sign(res)),
        ("Moneyline", "away", [f"{t} ML" for t in df.away_team], "away_moneyline", a_win, tie, "home_moneyline", np.sign(-res)),
        ("Total", "over", [f"Over {_fmt(t, False)}" for t in total], "over_odds", over, t_push, "under_odds", np.sign(tot - total)),
        ("Total", "under", [f"Under {_fmt(t, False)}" for t in total], "under_odds", under, t_push, "over_odds", np.sign(total - tot)),
    ]
    out = []
    for market, side, label, pc, p_win, p_push, oc, outcome in markets:
        default = None if market == "Moneyline" else DEFAULT_PRICE
        price, other = price_of(pc, default), price_of(oc, default)
        dec = O.american_to_decimal(price)
        fair, _ = O.no_vig(price, other)
        model_p = p_win / (1 - p_push)                      # win probability with pushes set aside
        ev = O.expected_value(p_win, price, p_push)
        profit = np.where(outcome > 0, dec - 1, np.where(outcome < 0, -1.0, 0.0))
        profit = np.where(np.isnan(outcome), np.nan, profit)
        stake = np.minimum(O.kelly(p_win, price, p_push) * C.KELLY_FRACTION, C.MAX_STAKE)
        line = {"Spread": np.where(side == "home", -spread, spread), "Total": total}.get(market, np.full(len(df), np.nan))
        out.append(pd.DataFrame({
            "game_id": df["game_id"], "season": df["season"], "week": df["week"],
            "market": market, "side": side, "pick": label, "line": line, "odds": price,
            "model_prob": model_p, "market_prob": fair, "breakeven": 1 / dec,
            "edge": model_p - fair, "ev": ev, "p_push": p_push, "stake": stake,
            "outcome": outcome, "profit": profit,
        }))
    b = pd.concat(out, ignore_index=True)
    return b[b["odds"].notna() & b["ev"].notna()].reset_index(drop=True)
