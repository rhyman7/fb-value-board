"""College team and quarterback ratings: the NFL rating engine with college settings.

Same method as nflmodel/ratings.py (opponent-adjusted ridge per metric, decayed by
recency, prior seasons carried at a discount, market-implied ratings from earlier
closing lines, quarterback EPA per dropback shrunk toward a replacement level).
Differences:
  * every non-FBS opponent is one pooled team, C.POOL
  * one more rated metric, snaps per game (college tempo varies far more than the NFL's)
  * college defaults for an unknown quarterback and the market's starting point
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflmodel.ratings import METRICS, RatingEngine

from . import config as C


class CFBRatingEngine(RatingEngine):
    QB_REPLACEMENT = C.QB_REPLACEMENT
    MARKET_DEFAULT = {"hfa": 2.5, "avg_total": 55.0}

    def __init__(self, team_games: pd.DataFrame, qb_games: pd.DataFrame, *, ridge: dict | None = None, **kw):
        kw.setdefault("half_life", C.HALF_LIFE_GAMES)
        kw.setdefault("carry", C.OFFSEASON_CARRY)
        kw.setdefault("market_half_life", C.MARKET_HALF_LIFE)
        kw.setdefault("qb_half_life", C.QB_HALF_LIFE_DROPBACKS)
        kw.setdefault("qb_prior", C.QB_PRIOR_DROPBACKS)
        super().__init__(team_games, qb_games, ridge=dict(C.RIDGE if ridge is None else ridge), **kw)
        t = team_games.sort_values("order").reset_index(drop=True)
        snaps = t["snaps"].to_numpy(float)
        self.y["snaps"] = np.nan_to_num(snaps, nan=np.nanmean(snaps))
        self.n_obs["snaps"] = np.ones(len(t))
        self.metrics = METRICS + ("snaps",)

    def _solve(self, metric: str, idx: np.ndarray, decay: np.ndarray):
        """Same weighted ridge as the NFL engine, built from sparse sums (every row has only
        four non-zero columns), which is much faster with 270 rating columns."""
        n = self.n
        k = 2 * n + 2
        w = decay * self.n_obs[metric][idx]
        y = self.y[metric][idx]
        h = (self.home[idx] - self.away[idx]) / 2.0
        cols = [self.off[idx], n + self.dfn[idx], np.full(len(idx), 2 * n), np.full(len(idx), 2 * n + 1)]
        vals = [np.ones(len(idx)), np.ones(len(idx)), np.ones(len(idx)), h]
        A = np.zeros(k * k)
        rhs = np.zeros(k)
        for ci, vi in zip(cols, vals):
            np.add.at(rhs, ci, w * vi * y)
            for cj, vj in zip(cols, vals):
                A += np.bincount(ci * k + cj, weights=w * vi * vj, minlength=k * k)
        A = A.reshape(k, k)
        pen = np.zeros(k)
        pen[: 2 * n] = self.ridge[metric]
        b = np.linalg.solve(A + np.diag(pen) + 1e-9 * np.eye(k), rhs)
        return b[:n], b[n: 2 * n], b[2 * n], b[2 * n + 1]
