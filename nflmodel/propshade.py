"""The over lean: a shade on the books' vig-free chance of a player prop.

On the 2023-2025 DraftKings/FanDuel main lines, overs on receptions, receiving yards and
running-back rushing (yards and attempts) went in 47.2% of the time against the 49.8% the
vig-free prices said; overs lost 10.7% at the posted prices and unders 0.6%. Passing yards lean
the other way (51.2% over). PROPS.md "The over lean and model-agreed unders"; applied Nov 9, 2026
(Ryan's call on a three-of-four trade-off; Project doc claude/plan-over-lean-shade-2026-11-09.md).

The shade is the average of (went over - vig-free over) on those lines, one number for the
receiving and running-back rushing markets and one for passing yards, fitted by
experiments/props_shade_fit.py from the committed over-lean test output and stored in
data/prop_params.json under "shade". Nothing else is shaded: passing TDs, completions, attempts,
interceptions, anytime TD, quarterback rushing, ladder rungs (RUNG_CAL is already fitted on how
rungs settled) and college keep the plain number.
"""
from __future__ import annotations

import json

from . import config as C

PARAMS_PATH = C.DATA / "prop_params.json"
_cache: dict | None = None


def load(path=PARAMS_PATH) -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(path.read_text()).get("shade", {}) or {}
        except (OSError, ValueError):
            _cache = {}
    return _cache


def reset():
    global _cache
    _cache = None


def group(stat: str, pos) -> str | None:
    """The shade group a prop belongs to, or None for a market that is not shaded."""
    qb = pos == "QB"
    if stat in ("receptions", "rec_yds"):
        return "rec_rush"
    if stat in ("rush_yds", "carries") and not qb:
        return "rec_rush"
    if stat == "pass_yds" and qb:
        return "pass_yds"
    return None


def shade_for(stat: str, pos) -> float:
    """Points added to the books' vig-free over chance (negative = toward the under)."""
    g = group(stat, pos)
    return float(load().get(g, 0.0)) if g else 0.0


def apply(fair_over: float, stat: str, pos) -> float:
    return min(max(fair_over + shade_for(stat, pos), 0.01), 0.99)
