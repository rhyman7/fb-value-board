"""Alternate lines and teaser legs priced off the key-number curve, for both boards.

The curve (distributions.py, bets.curves) already says how likely every margin and total
is, anchored so that at the market line each side equals the sharp book's no-vig price.
That makes two things cheap that books price crudely:

Alternate lines   P(cover) at every half point within ALT_RANGE of the market line, on the
                  market's own curve (what a half point through 3 or 7 is really worth) and on
                  the market-adjusted projection. The fair price at each line is what a book's
                  alternate ladder has to beat; the boards do not fetch the ladders themselves
                  (The Odds API charges per market for them).

Teaser legs       P(cover) with the line moved TEASE points the bettor's way, pushes set
                  aside (a push drops the leg at most books). A two-leg teaser at price q
                  needs each leg above sqrt(1 / decimal(q)): about 73.9% at -120. The classic
                  NFL legs (Wong teasers) are favourites of 7.5 to 8.5 teased through 7 and 3
                  to -1.5 .. -2.5, and underdogs of +1.5 to +2.5 teased to +7.5 .. +8.5; the
                  page marks them. Only the sides themselves are priced here; the page combines
                  legs at the price the bettor enters.

Shape, per game: {"alts": {"spread": [{line, home, away, home_proj, away_proj}, ...],
                           "total":  [{line, over, under, over_proj, under_proj}, ...]},
                  "teasers": {"6": {home, away, over, under, home_proj, ...}, "7": {...}}}
with spread lines home-positive as on the rest of the board.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bets as B
from . import distributions as D

ALT_RANGE = 7.0           # half points either side of the market line
TEASE = (6.0, 7.0)        # teaser sizes priced
WONG_FAV = (7.5, 8.5)     # favourite's line (points laid) that a 6-point tease carries through 7 and 3
WONG_DOG = (1.5, 2.5)     # underdog's line that a 6-point tease carries through 3 and 7


def _cover(cv: dict, i: int, market: str, side: str, line: float) -> float:
    """P(this side wins) with pushes set aside, at a line, from a curves() result."""
    p, push = B._side_prob(cv, i, market, side, line)
    return float(p / (1 - push)) if push < 1 else 0.0


def _lines(center: float, snap=None) -> list[float]:
    lo, hi = center - ALT_RANGE, center + ALT_RANGE
    xs = np.round(np.arange(lo, hi + 0.25, 0.5) * 2) / 2
    if snap is not None:
        xs = np.unique([float(snap(x)) for x in xs])
    return [float(x) for x in xs]


def price(df: pd.DataFrame, dist: dict, margins=None, totals=None, snap=None, r: int = 4) -> dict:
    """{game_id: {"alts": ..., "teasers": ...}} for the games in df (unplayed, with lines)."""
    df = df.reset_index(drop=True)
    if df.empty:
        return {}
    line = B.curves(df, dist, "line", margins, totals, snap)
    proj = B.curves(df, dist, "proj", margins, totals, snap)
    out = {}
    for i, g in enumerate(df.itertuples(index=False)):
        sp, to = float(line["spread"][i]), float(line["total"][i])
        if not (np.isfinite(sp) and np.isfinite(to)):
            continue
        alts = {"spread": [], "total": []}
        for L in _lines(sp, snap):
            alts["spread"].append({"line": L, "home": round(_cover(line, i, "Spread", "home", L), r), "away": round(_cover(line, i, "Spread", "away", L), r),
                                   "home_proj": round(_cover(proj, i, "Spread", "home", L), r), "away_proj": round(_cover(proj, i, "Spread", "away", L), r)})
        for L in _lines(to, snap):
            if L <= 0:
                continue
            alts["total"].append({"line": L, "over": round(_cover(line, i, "Total", "over", L), r), "under": round(_cover(line, i, "Total", "under", L), r),
                                  "over_proj": round(_cover(proj, i, "Total", "over", L), r), "under_proj": round(_cover(proj, i, "Total", "under", L), r)})
        teasers = {}
        for t in TEASE:
            # home gets t points: its line moves down; away gets t: the line moves up
            teasers[str(int(t)) if t == int(t) else str(t)] = {
                "home": round(_cover(line, i, "Spread", "home", sp - t), r), "away": round(_cover(line, i, "Spread", "away", sp + t), r),
                "over": round(_cover(line, i, "Total", "over", to - t), r), "under": round(_cover(line, i, "Total", "under", to + t), r),
                "home_proj": round(_cover(proj, i, "Spread", "home", sp - t), r), "away_proj": round(_cover(proj, i, "Spread", "away", sp + t), r),
                "over_proj": round(_cover(proj, i, "Total", "over", to - t), r), "under_proj": round(_cover(proj, i, "Total", "under", to + t), r),
            }
        out[g.game_id] = {"alts": alts, "teasers": teasers}
    return out


def wong(spread: float, side: str) -> bool:
    """Whether a 6-point tease of this side crosses both 3 and 7 (spread home-positive)."""
    laid = spread if side == "home" else -spread
    return WONG_FAV[0] <= laid <= WONG_FAV[1] or WONG_DOG[0] <= -laid <= WONG_DOG[1]
