"""Player outlook for the Props section: each player's whole range, not just one line.

A book's line asks one question ("over or under 68.5 receiving yards?"). The prop model
(props.PropModel) holds the whole distribution behind the projection, so it can also say how
likely every other number is: "68 is about a coin flip, but he gets 45 or more 80% of the
time." This writes that range for every player on the board, from the board's projections
(docs/data.json) and the fitted distributions (data/prop_params.json). No prices, no API calls.

    docs/props/outlook.json
      {generated, season, week, built, stats: {stat: {step, unit}},
       players: {"<game_id>|<player_id>|<stat>": {mu, lo, mid, hi, s: [P(at least step), P(at least 2 x step), ...]}}}

  mu    the projection (the mean)
  mid   the median: the most he reaches at least half the time
  lo    reached at least 90% of the time;  hi  reached at least 10% of the time
        (so lo to hi holds about 80% of outcomes)
  s     the chance of at least k x step, k = 1, 2, ... until it falls below 0.5%

The chances are the model's, with the same tuning-season correction as the prop prices, so
a rung on the ladder table and the outlook agree. They are projections, not bets: against
DraftKings and FanDuel the books have been the better forecast (PROPS.md).
Written by run_props.py after every props run and by the Update board build; a failure here
never stops either.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config as C, propmarket as PM, props as P

OUT = C.DOCS / "props" / "outlook.json"
STEP = {"rec_yds": 5, "rush_yds": 5, "pass_yds": 10}       # yards in steps; counts in ones
UNIT = {"rec_yds": "yds", "rush_yds": "yds", "pass_yds": "yds", "receptions": "rec", "carries": "car",
        "pass_att": "att", "completions": "cmp", "pass_td": "TD", "ints": "INT", "anytime_td": "TD"}
FLOOR = 0.005                                           # stop the curve once the chance is below this


def _survival(model, d: pd.DataFrame, stat: str, grp: str) -> tuple[int, np.ndarray]:
    """(step, matrix) with matrix[i, k-1] = P(player i reaches at least k x step)."""
    step = STEP.get(stat, 1)
    mu = P.mean(d, stat)
    top = int(np.ceil(max(float(np.nanmax(mu)) * 3 + 6 * step, 4 * step) / step))
    ks = np.arange(1, top + 1) * step
    lines = np.tile(ks - 0.5, (len(d), 1))                  # at least k = over k - 0.5
    over, _, _ = model.probs(d.reset_index(drop=True), stat, grp, lines)
    s = np.minimum.accumulate(np.clip(over, 0, 1), axis=1)  # never rising (the correction is per line)
    return step, s


def _at(s: np.ndarray, step: int, p: float) -> int:
    """The most a player reaches at least p of the time (0 if not even one step)."""
    ok = np.nonzero(s >= p)[0]
    return int((ok[-1] + 1) * step) if len(ok) else 0


def build(payload: dict, model) -> dict:
    players = PM.board_players(payload)
    meta = payload.get("meta") or {}
    out = {"generated": meta.get("generated"), "season": meta.get("season"), "week": meta.get("week"),
           "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "stats": {k: {"step": STEP.get(k, 1), "unit": UNIT[k]} for k in P.SPECS}, "players": {}}
    if players.empty or model is None:
        return out
    out_ok = ~players["status"].fillna("").str.lower().str.match(r"^(out|doubtful|injured reserve)")
    players = players[out_ok]
    for stat in P.SPECS:
        for grp in players["group"].unique():
            if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
                continue
            for c in P.SPECS[stat].proj:
                if c not in players:
                    players[c] = 0.0
            # anyone in the group with a projection (not the model's minimum for test lines: the
            # books post props on small roles too); quarterbacks only as the projected starter
            ok = (players["group"] == grp) & (P.mean(players, stat) > 0)
            if grp == "QB":
                ok &= players["att_share"] > 0.5
            d = players[ok].reset_index(drop=True)
            if d.empty:
                continue
            step, s = _survival(model, d, stat, grp)
            mu = P.mean(d, stat)
            for i, r in d.iterrows():
                row = s[i]
                keep = row[row >= FLOOR]
                out["players"][f"{r['game_id']}|{r['player_id']}|{stat}"] = {
                    "mu": round(float(mu[i]), 1), "lo": _at(row, step, 0.9), "mid": _at(row, step, 0.5), "hi": _at(row, step, 0.1),
                    "s": [round(float(x), 3) for x in keep]}
    return out


def write(payload: dict | None = None, model=None, path=OUT) -> int:
    """Build and save the outlook; returns the number of player-stats written."""
    if payload is None:
        payload = json.loads((C.DOCS / "data.json").read_text())
    if model is None:
        model = P.PropModel.load()
    o = build(payload, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o, separators=(",", ":")))
    return len(o["players"])
