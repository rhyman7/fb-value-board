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
