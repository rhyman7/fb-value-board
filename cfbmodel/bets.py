"""Price every bet on a set of college games. Same method as nflmodel/bets.py.

Differences from the NFL version:
  * a wider range of margins and totals (college blowouts and shootouts)
  * consensus lines in the data are averages across books (e.g. -3.17), so a game is
    priced and graded at that line rounded to the nearest half point, as a book would post it
  * no historical prices: spreads and totals are priced at -110 when no price is given,
    and moneylines only when the slate supplies a price
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr, ndtri

from nflmodel import bets as NB, distributions as D
from nflmodel import odds as O

from . import config as C

DEFAULT_PRICE = -110.0
MARGINS = np.arange(-100, 101)
TOTALS = np.arange(0, 171)


def half(x):
    """Round to the nearest half point (a posted line); quarter points round up."""
    return np.floor(np.asarray(x, float) * 2 + 0.5) / 2


def fit_distribution(games: pd.DataFrame, before_season: int, lookback: int = 8) -> dict:
    h = games[games.played & games.spread_line.notna() & games.total_line.notna()
              & (games.season < before_season) & (games.season >= before_season - lookback)]
    sp, tl = half(h.spread_line), half(h.total_line)
    mm = D.fit_multipliers(sp, h.result, 16.0, MARGINS, fold=True, shrink=8)
    tm = D.fit_multipliers(tl, h.total, 16.0, TOTALS, fold=False, shrink=40)
    sd_m = D.fit_scale(sp, h.result, MARGINS, mm, bounds=(10.0, 24.0))
    sd_t = D.fit_scale(tl, h.total, TOTALS, tm, bounds=(10.0, 24.0))
    mm = D.fit_multipliers(sp, h.result, sd_m, MARGINS, fold=True, shrink=8)
    tm = D.fit_multipliers(tl, h.total, sd_t, TOTALS, fold=False, shrink=40)
    a = games[games.played & games.spread_line.notna() & (games.season < before_season) & (games.result != 0)]
    won, mu = (a.result > 0).to_numpy(), a.spread_line.to_numpy(float)
    nll = lambda sd: -np.sum(np.where(won, log_ndtr(mu / sd), log_ndtr(-mu / sd)))
    sd_w = float(minimize_scalar(nll, bounds=(8.0, 24.0), method="bounded").x)
    return {"margin": mm, "total": tm, "sd_margin": sd_m, "sd_total": sd_t, "sd_win": sd_w, "n": int(len(h))}


def _fmt(x: float, signed: bool = True) -> str:
    if signed and x == 0:
        return "PK"
    return f"{x:+g}" if signed else f"{x:g}"


def price_games(df: pd.DataFrame, dist: dict, source: str = "proj", haircut: float = 1.0) -> pd.DataFrame:
    """One row per side of each market. source = "proj" (market-adjusted) or "model" (raw).
    haircut: stakes sized from the model probability shrunk toward the market's (nflmodel.bets.stake_prob)."""
    df = df.reset_index(drop=True)

    def col(name):
        return df[name].to_numpy(float) if name in df else np.full(len(df), np.nan)

    def price_of(name, default=None):
        x = col(name)
        return x if default is None else np.where(np.isnan(x), default, x)

    raw_spread, raw_total = col("spread_line"), col("total_line")
    spread, total = half(raw_spread), half(raw_total)
    d_margin = col(f"{source}_margin") - raw_spread
    d_total = col(f"{source}_total") - raw_total

    fair_home, _ = O.no_vig(price_of("home_spread_odds", DEFAULT_PRICE), price_of("away_spread_odds", DEFAULT_PRICE))
    fair_over, _ = O.no_vig(price_of("over_odds", DEFAULT_PRICE), price_of("under_odds", DEFAULT_PRICE))
    loc_m = D.anchor(spread, fair_home, dist["sd_margin"], MARGINS, dist["margin"]) + d_margin
    loc_t = D.anchor(total, fair_over, dist["sd_total"], TOTALS, dist["total"]) + d_total
    pm = D.pmf(loc_m, dist["sd_margin"], MARGINS, dist["margin"])
    pt = D.pmf(loc_t, dist["sd_total"], TOTALS, dist["total"])
    h_cov, s_push, a_cov = D.over_push_under(pm, MARGINS, spread)
    over, t_push, under = D.over_push_under(pt, TOTALS, total)

    tie = pm[:, MARGINS == 0].ravel()
    fair_ml, _ = O.no_vig_power(col("home_moneyline"), col("away_moneyline"))      # power method: see nflmodel.bets.curves
    implied = np.where(np.isnan(fair_ml), raw_spread, dist["sd_win"] * ndtri(np.clip(fair_ml, 1e-4, 1 - 1e-4)))
    h_win = ndtr((implied + d_margin) / dist["sd_win"]) * (1 - tie)
    a_win = 1 - tie - h_win

    res, tot = col("result"), col("total")
    markets = [
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
        fair, _ = O.no_vig_power(price, other)
        model_p = p_win / (1 - p_push)
        ev = O.expected_value(p_win, price, p_push)
        profit = np.where(outcome > 0, dec - 1, np.where(outcome < 0, -1.0, 0.0))
        profit = np.where(np.isnan(outcome), np.nan, profit)
        stake = np.minimum(O.kelly(NB.stake_prob(p_win, p_push, fair, haircut), price, p_push) * C.KELLY_FRACTION, C.MAX_STAKE)
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


def shop(df: pd.DataFrame, dist: dict, quotes: dict, books: list[str], source: str = "proj", kalshi_maker: bool = False,
         haircut: float = 1.0) -> dict:
    """Line shopping on the college supports and half-point lines (nflmodel.bets.shop)."""
    return NB.shop(df, dist, quotes, books, source, kalshi_maker, margins=MARGINS, totals=TOTALS, snap=half, haircut=haircut)


def curves(df: pd.DataFrame, dist: dict, source: str = "proj", margins=None, totals=None, snap=None) -> dict:
    """The outcome curves on the college supports (nflmodel.bets.curves)."""
    return NB.curves(df, dist, source, MARGINS if margins is None else margins, TOTALS if totals is None else totals, half if snap is None else snap)
