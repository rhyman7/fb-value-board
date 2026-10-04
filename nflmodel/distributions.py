"""Turn a projected margin or total into probabilities, respecting key numbers.

NFL scores are not smooth. Games land on a margin of exactly 3 about one time in
seven and on 7 about one time in twelve, so a bell curve badly understates how
often a spread of 3 pushes and how much the half-point from 2.5 to 3.5 is worth.

Three pieces:

Key numbers   Start from a bell curve cut into whole-number buckets, then
              multiply each bucket by how much more (or less) often that number
              has actually come up than the curve expected. The multipliers are
              measured from history (observed / expected, shrunk toward 1).

Width         Fitted, and not to the overall standard deviation of results.
              Margins have a tight centre and long tails, and every bet priced
              here lives near the centre, so the width is chosen to match how
              often results have cleared lines a few points either side of the
              closing number.

Anchor        The curve is positioned so that, when the projection equals the
              market line, each side is exactly as likely as the market's own
              no-vig price says. A projection that differs from the line slides
              the curve by that difference. Edges therefore come only from
              disagreeing with the market, never from quirks of the curve.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr as _cdf

MARGINS = np.arange(-60, 61)
TOTALS = np.arange(0, 121)
# Alternate lines used to measure how tightly results cluster around the line.
SHIFTS = (-6.5, -4.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 4.5, 6.5)


def pmf(loc, sd, support: np.ndarray, mult: np.ndarray | None = None) -> np.ndarray:
    """Probability of every whole-number outcome; one row per game."""
    loc = np.atleast_1d(np.asarray(loc, dtype=float))[:, None]
    p = _cdf((support[None, :] + 0.5 - loc) / sd) - _cdf((support[None, :] - 0.5 - loc) / sd)
    if mult is not None:
        p = p * mult[None, :]
    return p / p.sum(axis=1, keepdims=True)


def over_push_under(p: np.ndarray, support: np.ndarray, line) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(outcome > line), P(outcome == line), P(outcome < line) for each game."""
    line = np.atleast_1d(np.asarray(line, dtype=float))[:, None]
    over = (p * (support[None, :] > line + 1e-9)).sum(axis=1)
    push = (p * (np.abs(support[None, :] - line) < 1e-9)).sum(axis=1)
    return over, push, 1 - over - push


def anchor(line, target, sd, support: np.ndarray, mult: np.ndarray | None = None) -> np.ndarray:
    """Where to centre the curve so P(over the line | no push) equals `target`."""
    line = np.atleast_1d(np.asarray(line, dtype=float))
    target = np.broadcast_to(np.asarray(target, dtype=float), line.shape)
    lo, hi = line - 8.0, line + 8.0
    for _ in range(14):                      # bisection; over-probability rises with loc
        mid = (lo + hi) / 2
        over, push, _ = over_push_under(pmf(mid, sd, support, mult), support, line)
        too_low = over / (1 - push) < target
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    return (lo + hi) / 2


def fit_multipliers(mu, actual, sd, support: np.ndarray, *, fold: bool, shrink: float) -> np.ndarray:
    """Observed / expected frequency of each whole number, shrunk toward 1.

    fold=True treats +k and -k as the same number (margins of victory).
    `shrink` is a pseudo-count: numbers seen rarely stay close to 1.
    """
    exp = pmf(mu, sd, support).sum(axis=0)
    obs = np.zeros(len(support))
    idx = np.clip(np.round(np.asarray(actual, dtype=float)).astype(int) - support[0], 0, len(support) - 1)
    np.add.at(obs, idx, 1.0)
    if fold:
        exp, obs = exp + exp[::-1], obs + obs[::-1]
    return (obs + shrink) / (exp + shrink)


def fit_scale(mu, actual, support: np.ndarray, mult: np.ndarray, bounds=(9.0, 16.0)) -> float:
    """Curve width that best explains results against nearby alternate lines."""
    mu = np.asarray(mu, dtype=float)
    actual = np.asarray(actual, dtype=float)

    def nll(sd: float) -> float:
        p = pmf(anchor(mu, 0.5, sd, support, mult), sd, support, mult)
        total = 0.0
        for k in SHIFTS:
            over, _, _ = over_push_under(p, support, mu + k)
            over = np.clip(over, 1e-6, 1 - 1e-6)
            hit = actual > mu + k
            total -= np.sum(np.where(hit, np.log(over), np.log(1 - over)))
        return total

    return float(minimize_scalar(nll, bounds=bounds, method="bounded", options={"xatol": 0.05}).x)
