"""From a projected mean to the whole range of outcomes, for player props.

The player projections (players.py) give a mean. A prop needs P(over the line), so
each stat gets a distribution around that mean, fitted on past misses:

  counts   receptions, targets, carries, attempts, completions, touchdowns,
           interceptions: negative binomial (Poisson as the limit), with the
           overdispersion depending on the projected mean and the player's role
  yards    receiving, rushing, passing: a compound distribution, volume times
           yards per play. The volume (receptions, carries, completions) is the
           count distribution above; given n plays the yards are a gamma with
           n times a per-play shape. Zero plays gives zero yards, so the chance
           of a dud comes out of the volume, not a separate guess.
  exit     every stat carries a small chance the player leaves early or loses his
           role (injury, blowout, benching), as a second component with a fraction
           of the usual volume. It mostly fattens the low end.

Everything is evaluated as integer outcomes, so lines are handled the way books
settle them: on a half-point line there is no push; on a whole-number line
P(push) = P(X = line) and is reported separately.

The shared math here is also used by the college board (cfbmodel).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

EPS = 1e-12


# --------------------------------------------------------------------------
# Count distributions
# --------------------------------------------------------------------------

def nb_logpmf(k, mu, alpha):
    """Negative binomial log P(X = k) with mean mu and Var = mu + alpha * mu^2
    (alpha -> 0 is Poisson). Broadcasts."""
    k = np.asarray(k, float)
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    alpha = np.asarray(alpha, float)
    small = alpha < 1e-7
    a = np.where(small, 1.0, alpha)
    r = 1.0 / a
    nb = (special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
          + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))
    po = k * np.log(mu) - mu - special.gammaln(k + 1)
    return np.where(small, po, nb)


def count_pmf(mu, alpha, q, f, kmax: int) -> np.ndarray:
    """P(X = k), k = 0..kmax, per row: (1 - q) NB(m) + q NB(f m), with m chosen so the
    mixture's mean is mu. Rows of mu, alpha, q, f broadcast; returns (rows, kmax + 1)."""
    mu = np.atleast_1d(np.asarray(mu, float))
    alpha, q, f = (np.broadcast_to(np.asarray(x, float), mu.shape) for x in (alpha, q, f))
    m = mu / (1 - q + q * f)
    k = np.arange(kmax + 1)[None, :]
    full = np.exp(nb_logpmf(k, m[:, None], alpha[:, None]))
    part = np.exp(nb_logpmf(k, (f * m)[:, None], alpha[:, None]))
    pmf = (1 - q)[:, None] * full + q[:, None] * part
    # whatever lies beyond kmax goes in the last cell, so rows sum to one
    pmf[:, -1] += np.clip(1 - pmf.sum(1), 0, None)
    return pmf


def count_loglik(y, mu, alpha, q, f):
    m = mu / (1 - q + q * f)
    a = np.log(1 - q + EPS) + nb_logpmf(y, m, alpha)
    b = np.log(q + EPS) + nb_logpmf(y, f * m, alpha)
    return np.logaddexp(a, b)


# --------------------------------------------------------------------------
# Yards: volume x yards per play
# --------------------------------------------------------------------------

def compound_cdf(vol_pmf: np.ndarray, ypp, k1, x) -> np.ndarray:
    """P(yards <= x) for continuous yards = sum of n plays, each gamma with mean ypp and
    shape k1 (so n plays: shape n k1, scale ypp / k1). vol_pmf is (rows, nmax + 1);
    x is (rows, m). Zero plays is zero yards."""
    rows, nn = vol_pmf.shape
    x = np.asarray(x, float).reshape(rows, -1)
    ypp = np.maximum(np.asarray(ypp, float).reshape(rows), 1e-6)
    k1 = np.asarray(k1, float).reshape(rows)
    n = np.arange(1, nn)[None, :, None]
    shape = n * k1[:, None, None]
    scale = (ypp / k1)[:, None, None]
    g = special.gammainc(shape, np.maximum(x[:, None, :], 0) / scale)      # (rows, n, m)
    out = vol_pmf[:, :1] * (x >= 0) + np.einsum("rn,rnm->rm", vol_pmf[:, 1:], g)
    return np.clip(out, 0, 1)


def compound_loglik(y, vol_pmf, ypp, k1):
    """Log P(integer yards = y), outcomes at or below zero pooled as zero."""
    y = np.maximum(np.asarray(y, float), 0)
    hi = compound_cdf(vol_pmf, ypp, k1, (y + 0.5)[:, None])[:, 0]
    lo = np.where(y > 0, compound_cdf(vol_pmf, ypp, k1, (y - 0.5)[:, None])[:, 0], 0.0)
    return np.log(np.maximum(hi - lo, 1e-10))


# --------------------------------------------------------------------------
# Hurdle gamma (the yards alternative tested against the compound)
# --------------------------------------------------------------------------

def hurdle_cdf(mu, p0, k, x):
    mu = np.asarray(mu, float)[:, None]
    p0 = np.asarray(p0, float)[:, None]
    k = np.asarray(k, float)[:, None]
    mpos = np.maximum(mu / np.maximum(1 - p0, 1e-6), 1e-6)
    g = special.gammainc(k, np.maximum(np.asarray(x, float), 0) / (mpos / k))
    return np.clip(p0 * (np.asarray(x) >= 0) + (1 - p0) * g, 0, 1)


def hurdle_loglik(y, mu, p0, k):
    y = np.maximum(np.asarray(y, float), 0)
    hi = hurdle_cdf(mu, p0, k, (y + 0.5)[:, None])[:, 0]
    lo = np.where(y > 0, hurdle_cdf(mu, p0, k, (y - 0.5)[:, None])[:, 0], 0.0)
    return np.log(np.maximum(hi - lo, 1e-10))


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------

def is_whole(line) -> np.ndarray:
    line = np.asarray(line, float)
    return np.isclose(line, np.round(line))


def settle_probs(cdf_at, line):
    """(P(over), P(push), P(under)) from a function giving P(X <= k) at integer k.
    Half-point line L: over = 1 - F(floor L). Whole line L: push = F(L) - F(L - 1)."""
    line = np.asarray(line, float)
    whole = is_whole(line)
    k = np.where(whole, np.round(line), np.floor(line))
    f_k = cdf_at(k)
    f_km1 = np.where(whole, cdf_at(k - 1), f_k)
    over = 1 - f_k
    push = np.where(whole, f_k - f_km1, 0.0)
    return over, push, 1 - over - push


# --------------------------------------------------------------------------
# Fitted parameters
# --------------------------------------------------------------------------

def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


@dataclass
class CountFit:
    """alpha = exp(a + b log(mu) + c role); exit chance q at fraction f of the usual volume."""
    a: float = -20.0
    b: float = 0.0
    c: float = 0.0
    q: float = 0.0
    f: float = 0.3
    family: str = "nb"
    n: int = 0

    def alpha(self, mu, role):
        if self.family == "poisson":
            return np.zeros_like(np.asarray(mu, float))
        return np.exp(np.clip(self.a + self.b * np.log(np.maximum(mu, 0.05)) + self.c * np.asarray(role, float), -20, 5))

    def pmf(self, mu, role, kmax):
        return count_pmf(mu, self.alpha(mu, role), self.q, self.f, kmax)

    def loglik(self, y, mu, role):
        return count_loglik(y, mu, self.alpha(mu, role), self.q, self.f)


def fit_count(y, mu, role, family: str = "nb", exit_mix: bool = True) -> CountFit:
    y, mu, role = (np.asarray(v, float) for v in (y, mu, role))
    mu = np.maximum(mu, 0.02)

    def unpack(th):
        a, b, c = (th[0], th[1], th[2]) if family == "nb" else (-20.0, 0.0, 0.0)
        q, f = (_sig(th[-2]) * 0.3, _sig(th[-1])) if exit_mix else (0.0, 0.3)
        return CountFit(a, b, c, q, f, family, len(y))

    def nll(th):
        return -unpack(th).loglik(y, mu, role).mean()

    th0 = ([np.log(0.1), -0.5, 0.0] if family == "nb" else []) + ([-2.0, -1.0] if exit_mix else [])
    if not th0:
        return CountFit(family="poisson", n=len(y))
    res = optimize.minimize(nll, np.array(th0, float), method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
    return unpack(res.x)


@dataclass
class YardsFit:
    """Compound: per-play gamma shape k1 = exp(a + b log(volume) + c role)."""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    n: int = 0

    def k1(self, vol_mu, role):
        return np.exp(np.clip(self.a + self.b * np.log(np.maximum(vol_mu, 0.05)) + self.c * np.asarray(role, float), -6, 6))


def fit_yards(y, vol_pmf, ypp, vol_mu, role) -> YardsFit:
    y, ypp, vol_mu, role = (np.asarray(v, float) for v in (y, ypp, vol_mu, role))

    def nll(th):
        fit = YardsFit(*th)
        return -compound_loglik(y, vol_pmf, ypp, fit.k1(vol_mu, role)).mean()

    res = optimize.minimize(nll, np.array([0.0, 0.0, 0.0]), method="Nelder-Mead",
                            options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-6})
    return YardsFit(*res.x, n=len(y))


@dataclass
class HurdleFit:
    """p0 = sigmoid(a + b log mu + c role); positive-part gamma shape exp(d + e log mu)."""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    n: int = 0

    def params(self, mu, role):
        lm = np.log(np.maximum(mu, 0.05))
        return (_sig(self.a + self.b * lm + self.c * np.asarray(role, float)),
                np.exp(np.clip(self.d + self.e * lm, -6, 6)))


def fit_hurdle(y, mu, role) -> HurdleFit:
    y, mu, role = (np.asarray(v, float) for v in (y, mu, role))

    def nll(th):
        p0, k = HurdleFit(*th).params(mu, role)
        return -hurdle_loglik(y, mu, p0, k).mean()

    res = optimize.minimize(nll, np.array([0.0, -1.0, 0.0, 0.0, 0.0]), method="Nelder-Mead",
                            options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-7})
    return HurdleFit(*res.x, n=len(y))


# --------------------------------------------------------------------------
# Isotonic correction
# --------------------------------------------------------------------------

@dataclass
class Isotonic:
    """A monotone map p -> calibrated p (pool-adjacent-violators), interpolated."""
    x: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    y: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    @classmethod
    def fit(cls, p, hit, bins: int = 40):
        p, hit = np.asarray(p, float), np.asarray(hit, float)
        # pre-bin to keep the steps from chasing noise
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        w = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        keep = w > 0
        xs = (np.bincount(idx, p, len(edges) - 1)[keep] / w[keep])
        ys = (np.bincount(idx, hit, len(edges) - 1)[keep] / w[keep])
        w = w[keep]
        # pool adjacent violators
        vals, wts, cnt = [], [], []
        for v, ww in zip(ys, w):
            vals.append(v); wts.append(ww); cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
                wts[-1] += w2; cnt[-1] += c2
        yfit = np.repeat(vals, cnt)
        return cls(np.r_[0.0, xs, 1.0], np.r_[min(0.0, yfit[0]), yfit, max(1.0, yfit[-1])])

    def __call__(self, p):
        return np.interp(np.asarray(p, float), self.x, self.y)

    def to_dict(self):
        return {"x": [round(float(v), 5) for v in self.x], "y": [round(float(v), 5) for v in self.y]}


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def brier(p, hit):
    return float(np.mean((np.asarray(p) - np.asarray(hit)) ** 2))


def log_loss(p, hit):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    h = np.asarray(hit, float)
    return float(-np.mean(h * np.log(p) + (1 - h) * np.log(1 - p)))


def calibration_table(p, hit, edges=None):
    """Predicted bands against how often it happened, with counts."""
    p, hit = np.asarray(p, float), np.asarray(hit, float)
    if edges is None:
        edges = np.r_[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append({"band": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()), "pred": float(p[m].mean()), "actual": float(hit[m].mean())})
    return rows
