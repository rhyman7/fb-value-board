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


# ---- Kalshi -----------------------------------------------------------------
# Kalshi contracts pay $1 and cost the price P in dollars. The exchange charges a fee on top
# of the price, per contract, of rate x P x (1 - P), rounded up to the cent on the order:
#   taker (an order that fills at once)   fees = round up(M x 0.07   x C x P x (1 - P))
#   maker (a resting order that fills)    fees = round up(M x 0.0175 x C x P x (1 - P))
# with C contracts and M the series multiplier, which is 1 for both rates on the game series
# the boards use, KXNFLGAME (Professional Football Game) and KXNCAAFGAME (College Football
# Game), so makers do pay on them. Source: Kalshi's fee schedule, "last updated and effective
# July 7, 2026", https://kalshi.com/docs/kalshi-fee-schedule.pdf (also kalshi.com/fee-schedule;
# the rounding rule at docs.kalshi.com/getting_started/fee_rounding), read Oct 30, 2026.
# Cancelling a resting order is free. The rounding (under a cent a contract) is ignored here.
# A fee is charged on the stake whatever the result, so it lowers the odds the bettor really
# gets; set KALSHI_TAKER / KALSHI_MAKER if the schedule changes.
KALSHI_TAKER = 0.07
KALSHI_MAKER = 0.0175


def kalshi_fee(price_prob, maker: bool = False):
    """Fee per $1 contract at contract price `price_prob` (0-1)."""
    p = np.clip(np.asarray(price_prob, dtype=float), 0.0, 1.0)
    return (KALSHI_MAKER if maker else KALSHI_TAKER) * p * (1 - p)


def kalshi_effective_odds(odds, maker: bool = False):
    """American odds net of Kalshi's fee: the contract price plus the fee is what a $1
    payout really costs, so -150 (60c) becomes about -159 as a taker."""
    p = implied_prob(odds)
    cost = p + kalshi_fee(p, maker)
    cost = np.clip(cost, 1e-6, 1 - 1e-6)
    return prob_to_american(cost)


def kalshi_net_odds_at(cents, maker: bool = True):
    """American odds net of the fee for a contract bought at `cents` (1-99), as a maker by default."""
    p = np.asarray(cents, dtype=float) / 100.0
    return prob_to_american(np.clip(p + kalshi_fee(p, maker), 1e-6, 1 - 1e-6))


def kalshi_post_at(p_win, p_push: float = 0.0, match_ev: float | None = None) -> dict:
    """Where to rest a maker order on Kalshi for a side the board gives `p_win` (with `p_push`
    set aside as on the main markets): the highest whole-cent contract price at which the EV
    net of the maker fee is still >= 0 (`post_at`), and the highest at which the EV equals or
    beats `match_ev`, the best retail book's EV on the same side (`post_match`; None when no
    retail price is on file or no price gets there). Both None when no cent is +EV. A resting
    order may never fill, and the market can move before it does."""
    cents = np.arange(1, 100)
    ev = expected_value(float(p_win), kalshi_net_odds_at(cents, maker=True), float(p_push))
    ok = cents[ev >= 0]
    out = {"post_at": int(ok.max()) if len(ok) else None, "post_match": None}
    if match_ev is not None and out["post_at"] is not None:
        hit = cents[ev >= max(float(match_ev), 0.0)]          # never above the break-even cent
        out["post_match"] = int(hit.max()) if len(hit) else None
    return out


def prob_to_american(p):
    """The American price whose implied probability is p."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)


def kelly(p_win, odds, p_push=0.0):
    """Full-Kelly share of bankroll for a bet with a possible push; 0 if not +EV."""
    b = american_to_decimal(odds) - 1
    p_lose = 1 - p_win - p_push
    f = (p_win * b - p_lose) / (b * (p_win + p_lose))
    return np.maximum(f, 0.0)
