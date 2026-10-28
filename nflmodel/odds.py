"""Odds math: implied probability, vig removal, expected value, Kelly stakes."""
from __future__ import annotations

import numpy as np


def american_to_decimal(odds):
    """-110 -> 1.909, +150 -> 2.5. Decimal odds include the returned stake."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 + np.where(odds > 0, odds / 100.0, 100.0 / -odds)


def implied_prob(odds):
    """Raw implied probability, vig included. This is the break-even win rate."""
    return 1 / american_to_decimal(odds)


def no_vig(odds_a, odds_b):
    """Strip the book's margin from a two-way market (proportional method)."""
    a, b = implied_prob(odds_a), implied_prob(odds_b)
    return a / (a + b), b / (a + b)


def power_exponent(implied_a, implied_b, iters: int = 60):
    """k with a^k + b^k = 1 (the power method's exponent; k > 1 when there is vig)."""
    a, b = np.asarray(implied_a, float), np.asarray(implied_b, float)
    lo, hi = np.full(np.broadcast(a, b).shape, 0.2), np.full(np.broadcast(a, b).shape, 5.0)
    for _ in range(iters):
        k = (lo + hi) / 2
        too_big = a ** k + b ** k > 1          # sum falls as k rises
        lo, hi = np.where(too_big, k, lo), np.where(too_big, hi, k)
    return (lo + hi) / 2


def no_vig_power(odds_a, odds_b):
    """Power method: fair p = implied ** k, with k making the two sides sum to one. Takes
    more of the margin from the longshot than proportional scaling does, which suits
    lopsided markets (the favourite-longshot bias)."""
    a, b = implied_prob(odds_a), implied_prob(odds_b)
    k = power_exponent(a, b)
    return a ** k, b ** k


def one_sided_fair(odds, margin: float, method: str = "power"):
    """Fair probability for a market with only one side posted (anytime touchdown), given an
    assumed two-way overround `margin` (0.08 means the two sides would sum to 1.08).
    proportional: implied / (1 + margin). power: implied ** k, with k set so that a price
    whose fair chance is 50% carries that overround."""
    p = implied_prob(odds)
    if method == "proportional":
        return p / (1 + margin)
    half = (1 + margin) / 2                    # implied price of a fair 50% side
    k = np.log(0.5) / np.log(half)
    return p ** k


def hold(odds_a, odds_b):
    """The book's margin on a two-way market, e.g. 0.0476 at -110/-110."""
    return implied_prob(odds_a) + implied_prob(odds_b) - 1


def expected_value(p_win, odds, p_push=0.0):
    """Expected profit per 1 unit staked. A push returns the stake."""
    p_lose = 1 - p_win - p_push
    return p_win * (american_to_decimal(odds) - 1) - p_lose


def kelly(p_win, odds, p_push=0.0):
    """Full-Kelly share of bankroll for a bet with a possible push; 0 if not +EV."""
    b = american_to_decimal(odds) - 1
    p_lose = 1 - p_win - p_push
    f = (p_win * b - p_lose) / (b * (p_win + p_lose))
    return np.maximum(f, 0.0)
