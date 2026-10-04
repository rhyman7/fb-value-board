"""Walk-forward backtest against closing lines.

Every number here is out of sample: a season's projections, blend weight,
key-number multipliers and curve widths are all fitted on earlier seasons only.
Bets are graded at the closing price recorded in the nflverse games file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import bets as B
from . import odds as O

EV_STEPS = (0.0, 0.02, 0.05)


def grade_all(games: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Price and grade every bet on every played game from the first test season."""
    parts, dists = [], {}
    for s in sorted(frame.season.unique()):
        if s < C.FIRST_TEST_SEASON:
            continue
        dists[int(s)] = B.fit_distribution(games, int(s))
        played = frame[(frame.season == s) & frame.played & frame.spread_line.notna()]
        if played.empty:
            continue
        for src in ("model", "proj"):
            b = B.price_games(played, dists[int(s)], src)
            b["source"] = src
            parts.append(b)
    return pd.concat(parts, ignore_index=True), dists


def _record(b: pd.DataFrame) -> dict:
    w, l, p = int((b.outcome > 0).sum()), int((b.outcome < 0).sum()), int((b.outcome == 0).sum())
    n = len(b)
    return {
        "n": n, "w": w, "l": l, "p": p,
        "win_pct": round(w / (w + l), 4) if w + l else None,
        "roi": round(float(b.profit.mean()), 4) if n else None,
        "units": round(float(b.profit.sum()), 1),
        "claimed_ev": round(float(b.ev.mean()), 4) if n else None,
        # one standard error of the ROI, so small samples are not over-read
        "roi_se": round(float(b.profit.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else None,
    }


def leans(b: pd.DataFrame) -> pd.DataFrame:
    """The better side of each market in each game: one bet per game per market."""
    return b.loc[b.groupby(["game_id", "market"]).ev.idxmax()]


def summarize(frame: pd.DataFrame, graded: pd.DataFrame) -> dict:
    f = frame[frame.played & frame.spread_line.notna() & (frame.season >= C.FIRST_TEST_SEASON)]
    mae = lambda a, b: round(float(np.mean(np.abs(a - b))), 3)

    def accuracy(x: pd.DataFrame) -> dict:
        decided = x[x.result != 0]
        return {
            "n": int(len(x)),
            "margin_mae": {k: mae(x[c], x.result) for k, c in (("model", "model_margin"), ("blend", "proj_margin"), ("market", "spread_line"))},
            "total_mae": {k: mae(x[c], x.total) for k, c in (("model", "model_total"), ("blend", "proj_total"), ("market", "total_line"))},
            "winner_pct": {k: round(float(np.mean(np.sign(decided[c]) == np.sign(decided.result))), 4)
                           for k, c in (("model", "model_margin"), ("market", "spread_line"))},
        }

    out = {
        "first_season": int(f.season.min()), "last_season": int(f.season.max()),
        "holdout_from": C.HOLDOUT_FROM, "games": int(len(f)),
        "accuracy": accuracy(f),
        "accuracy_holdout": accuracy(f[f.season >= C.HOLDOUT_FROM]),
        "by_season": [{"season": int(s), **accuracy(x),
                       "w_margin": round(float(x.w_margin.iloc[0]), 3), "w_total": round(float(x.w_total.iloc[0]), 3)}
                      for s, x in f.groupby("season")],
    }

    # ---- betting records ---------------------------------------------------
    records = {}
    for src in ("model", "proj"):
        g = graded[graded.source == src]
        records[src] = {}
        for market in ("Spread", "Total", "Moneyline"):
            m = g[g.market == market]
            ho = m[m.season >= C.HOLDOUT_FROM]
            rows = [{"rule": "Every game", **_record(leans(m)), "holdout": _record(leans(ho))}]
            rows += [{"rule": f"EV above {int(t * 100)}%", **_record(m[m.ev > t]), "holdout": _record(ho[ho.ev > t])} for t in EV_STEPS]
            records[src][market] = {"rows": rows,
                                    "by_season": [{"season": int(s), **_record(leans(x))} for s, x in m.groupby("season")]}
    out["records"] = records

    # ---- does claimed EV show up? ------------------------------------------
    ev_cal = {}
    edges = [-1, -0.04, -0.02, 0, 0.02, 0.04, 0.07, 0.11, 1]
    for src in ("model", "proj"):
        g = graded[(graded.source == src) & (graded.market != "Moneyline")].copy()
        g["bucket"] = pd.cut(g.ev, edges)
        rows = []
        for _, x in g.groupby("bucket", observed=True):
            if len(x) >= 40:
                rows.append({"claimed": round(float(x.ev.mean()), 4), "realized": round(float(x.profit.mean()), 4),
                             "se": round(float(x.profit.std(ddof=1) / np.sqrt(len(x))), 4), "n": int(len(x))})
        ev_cal[src] = rows
    out["ev_calibration"] = ev_cal

    # ---- win-probability calibration (home team, moneyline) -----------------
    ml = graded[(graded.source == "proj") & (graded.market == "Moneyline") & (graded.outcome != 0)].copy()
    cal = []
    ml["bin"] = pd.cut(ml.model_prob, np.linspace(0, 1, 11))
    for _, x in ml.groupby("bin", observed=True):
        if len(x) >= 30:
            cal.append({"predicted": round(float(x.model_prob.mean()), 4), "market": round(float(x.market_prob.mean()), 4),
                        "actual": round(float((x.outcome > 0).mean()), 4), "n": int(len(x))})
    out["calibration"] = cal

    # ---- cumulative units over time ------------------------------------------
    series = {}
    picks = {
        "Raw model, every spread": leans(graded[(graded.source == "model") & (graded.market == "Spread")]),
        "Market-adjusted, spreads with EV above 0": graded[(graded.source == "proj") & (graded.market == "Spread") & (graded.ev > 0)],
        "Raw model, every total": leans(graded[(graded.source == "model") & (graded.market == "Total")]),
        "Raw model, every moneyline": leans(graded[(graded.source == "model") & (graded.market == "Moneyline")]),
    }
    weeks = sorted(set(zip(f.season, f.week)))
    for name, b in picks.items():
        wk = b.groupby(["season", "week"]).profit.sum()
        cum, pts = 0.0, []
        for s, w in weeks:
            cum += float(wk.get((s, w), 0.0))
            pts.append(round(cum, 2))
        series[name] = pts
    out["cumulative"] = {"weeks": [f"{s} W{w}" for s, w in weeks], "seasons": [int(s) for s, _ in weeks], "series": series}
    return out


def key_number_report(games: pd.DataFrame, dist: dict, before_season: int, lookback: int = 10) -> dict:
    """How often each margin comes up, and how the push model compares with reality."""
    from . import distributions as D
    h = games[games.played & games.spread_line.notna() & (games.season < before_season) & (games.season >= before_season - lookback)]
    absm = h.result.abs().astype(int)
    smooth = D.pmf(h.spread_line, dist["sd_margin"], D.MARGINS).sum(axis=0)   # bell curve only
    smooth = (smooth + smooth[::-1])[60:]
    smooth[0] /= 2
    freq = [{"margin": k, "actual": round(float((absm == k).mean()), 4), "smooth": round(float(smooth[k] / len(h)), 4),
             "mult": round(float(dist["margin"][60 + k]), 2)} for k in range(0, 22)]
    pushes = []
    for line in (3, 7, 6, 10, 4):
        x = h[h.spread_line.abs() == line]
        if len(x) < 40:
            continue
        loc = D.anchor(x.spread_line, 0.5, dist["sd_margin"], D.MARGINS, dist["margin"])
        key = D.over_push_under(D.pmf(loc, dist["sd_margin"], D.MARGINS, dist["margin"]), D.MARGINS, x.spread_line)[1].mean()
        bell = D.over_push_under(D.pmf(x.spread_line, dist["sd_margin"], D.MARGINS), D.MARGINS, x.spread_line)[1].mean()
        pushes.append({"line": line, "n": int(len(x)), "actual": round(float((x.result == x.spread_line).mean()), 4),
                       "model": round(float(key), 4), "bell": round(float(bell), 4)})
    return {"seasons": f"{before_season - lookback}-{before_season - 1}", "games": int(len(h)), "frequency": freq, "pushes": pushes,
            "sd_margin": round(dist["sd_margin"], 2), "sd_total": round(dist["sd_total"], 2), "sd_win": round(dist["sd_win"], 2)}
