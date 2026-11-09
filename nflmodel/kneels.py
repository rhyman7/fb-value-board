"""Expected kneel-downs for the starting quarterback, for pricing his rush-attempts prop.

The books settle rush attempts on the official box score, which counts a kneel-down; the
projections are for his runs without them (players_data.SETTLE). Tested Nov 9, 2026
(experiments/qb_kneel_proj_test.py; PROPS.md "Quarterback rushing projections on the official
line"): adding the expected kneels made the rush-attempts projection better on every measure
(RMSE 2.25 -> 2.10 tuning, 2.64 -> 2.49 holdout; Brier at the 2025 lines -0.008 / -0.012), and
taking their yards off rushing yards made that worse on every measure. So only attempts move, and
only where the number meets a book's line (props.py); `carries_proj`, the Players tab and every
model input are untouched.

    k = a + b * wp, floored at 0      wp = Phi(projected margin / 13.5), his team's chance of winning

a and b are least squares on projected starters' starts in every earlier season (walk-forward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

SD = 13.5            # points: turns the game model's margin into a chance of winning
MIN_STARTS = 200     # fewer earlier starts than this: no adjustment


def attach(table: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """The player table with each quarterback's kneel-downs in the game (0 for everyone else, and
    for box scores cached without the column)."""
    if "kneels" not in pg:
        return table.assign(kneels=0.0)
    k = pg[["game_id", "player_id", "kneels"]].drop_duplicates(["game_id", "player_id"])
    t = table.drop(columns=["kneels"], errors="ignore").merge(k, on=["game_id", "player_id"], how="left")
    t["kneels"] = t["kneels"].fillna(0.0).where(t["group"] == "QB", 0.0)
    return t


def fit(starts: pd.DataFrame) -> tuple[float, float]:
    """(a, b) from starts with columns wp and kneels."""
    X = np.c_[np.ones(len(starts)), starts["wp"].to_numpy(float)]
    a, b = np.linalg.lstsq(X, starts["kneels"].to_numpy(float), rcond=None)[0]
    return float(a), float(b)


def project(table: pd.DataFrame) -> pd.DataFrame:
    """Adds kneels_proj: the projected starter's expected kneel-downs, each season from the seasons
    before it; 0 for everyone else. Needs kneels (attach), margin, att_share, started, played."""
    t = table.copy()
    t["kneels_proj"] = 0.0
    wp = pd.Series(norm.cdf(t["margin"].fillna(0.0).to_numpy(float) / SD), index=t.index)
    qb1 = (t["group"] == "QB") & (t["att_share"] > 0.5)
    starts = qb1 & (t["started"] == 1) & t["played"].astype(bool)
    for s in sorted(t["season"].unique()):
        tr = starts & (t["season"] < s)
        if tr.sum() < MIN_STARTS:
            continue
        a, b = fit(pd.DataFrame({"wp": wp[tr], "kneels": t.loc[tr, "kneels"]}))
        m = qb1 & (t["season"] == s)
        t.loc[m, "kneels_proj"] = (a + b * wp[m]).clip(lower=0.0)
    return t
