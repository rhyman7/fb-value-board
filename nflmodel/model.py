"""The game model: projected margin and total, then a market-aware blend.

Three layers, each fitted walk-forward (season S only ever sees seasons < S):

1. Raw model      margin and total from EPA-based team ratings, market-implied
                  ratings from earlier weeks' lines, the quarterback adjustment,
                  home field, rest and roof. Ridge regression.
2. Market blend   projection = market + w * (raw model - market).
                  w is how much the model has historically added on top of the
                  closing line. It is the honest number to bet from: the raw
                  model alone overstates every disagreement with the market.

Turning a projection into probabilities is a separate step (distributions.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ratings import RatingEngine

MARGIN_FEATS = ["d_pass", "d_rush", "d_sr", "d_pts", "d_mkt", "d_qb", "hfa", "rest_diff"]
TOTAL_FEATS = ["s_pass", "s_rush", "s_sr", "s_pts", "s_mkt", "s_qb", "lg_pts", "indoor"]
FIRST_OOS = C.FIRST_MODEL_SEASON + 3   # first season with out-of-sample projections


def build_frame(games: pd.DataFrame, engine: RatingEngine, verbose: bool = False) -> pd.DataFrame:
    """Every game from FIRST_MODEL_SEASON on, with pre-game features attached."""
    g = games[games.season >= C.FIRST_MODEL_SEASON].copy()
    feats = engine.features(g, verbose=verbose)
    df = g.drop(columns=["home_qb_id", "away_qb_id"]).merge(feats, on="game_id", how="inner")

    # Home-field edge has shrunk over the years, so measure it from recent seasons
    # instead of assuming a constant: average home margin over the prior five.
    home_margin = games[(games.neutral == 0) & games.played].groupby("season")["result"].mean()
    trail = {s: home_margin[(home_margin.index < s) & (home_margin.index >= s - 5)].mean()
             for s in df.season.unique()}
    df["hfa"] = df["season"].map(trail).fillna(home_margin.mean()) * (1 - df["neutral"])
    df["rest_diff"] = (df["home_rest"] - df["away_rest"]).clip(-7, 7).fillna(0) * (1 - df["playoff"])
    return df


class Linear:
    """Ridge regression on standardised features (so one penalty fits all)."""

    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
        w = np.ones(len(y)) if w is None else w
        self.mu = np.average(X, axis=0, weights=w)
        self.sd = np.sqrt(np.average((X - self.mu) ** 2, axis=0, weights=w))
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        self.y0 = np.average(y, weights=w)
        Zw = Z * w[:, None]
        self.beta = np.linalg.solve(Z.T @ Zw + self.alpha * np.eye(Z.shape[1]), Zw.T @ (y - self.y0))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.y0 + ((X - self.mu) / self.sd) @ self.beta

    def raw_coefs(self, names: list[str]) -> dict:
        return dict(zip(names, (self.beta / self.sd).round(4)))


def _blend_weight(model: np.ndarray, market: np.ndarray, actual: np.ndarray) -> float:
    """Least-squares w in  actual - market = w * (model - market), kept in [0, 1]."""
    d = model - market
    ok = ~np.isnan(d) & ~np.isnan(actual)
    if ok.sum() < 200:
        return 0.0
    w = float(d[ok] @ (actual[ok] - market[ok]) / (d[ok] @ d[ok]))
    return min(max(w, 0.0), 1.0)


def walk_forward(df: pd.DataFrame, alpha: float = 20.0) -> tuple[pd.DataFrame, dict]:
    """Add model, blend and spread columns to every game from FIRST_OOS on."""
    df = df.copy()
    for col in ("model_margin", "model_total", "proj_margin", "proj_total", "w_margin", "w_total"):
        df[col] = np.nan
    info = {}
    seasons = sorted(s for s in df.season.unique() if s >= FIRST_OOS)
    for s in seasons:
        train = df[(df.season < s) & df.played]
        test = df.season == s
        m = Linear(alpha).fit(train[MARGIN_FEATS].to_numpy(), train["result"].to_numpy())
        t = Linear(alpha).fit(train[TOTAL_FEATS].to_numpy(), train["total"].to_numpy())
        df.loc[test, "model_margin"] = m.predict(df.loc[test, MARGIN_FEATS].to_numpy())
        df.loc[test, "model_total"] = t.predict(df.loc[test, TOTAL_FEATS].to_numpy())

        # Blend weights and residual spread come from earlier out-of-sample seasons.
        past = df[(df.season < s) & (df.season >= max(FIRST_OOS, s - C.BLEND_LOOKBACK)) & df.played]
        wm = _blend_weight(*(past[c].to_numpy() for c in ("model_margin", "spread_line", "result")))
        wt = _blend_weight(*(past[c].to_numpy() for c in ("model_total", "total_line", "total")))
        df.loc[test, "w_margin"], df.loc[test, "w_total"] = wm, wt
        has_line = test & df["spread_line"].notna()
        df.loc[test, "proj_margin"] = df.loc[test, "model_margin"]
        df.loc[test, "proj_total"] = df.loc[test, "model_total"]
        df.loc[has_line, "proj_margin"] = df.loc[has_line, "spread_line"] + wm * (df.loc[has_line, "model_margin"] - df.loc[has_line, "spread_line"])
        has_tot = test & df["total_line"].notna()
        df.loc[has_tot, "proj_total"] = df.loc[has_tot, "total_line"] + wt * (df.loc[has_tot, "model_total"] - df.loc[has_tot, "total_line"])

        info[int(s)] = {
            "margin_coefs": m.raw_coefs(MARGIN_FEATS), "total_coefs": t.raw_coefs(TOTAL_FEATS),
            "margin_intercept": round(float(m.y0 - (m.mu / m.sd) @ m.beta), 3),
            "total_intercept": round(float(t.y0 - (t.mu / t.sd) @ t.beta), 3),
            "w_margin": round(wm, 3), "w_total": round(wt, 3), "n_train": int(len(train)),
        }
    return df[df.season >= FIRST_OOS], info
