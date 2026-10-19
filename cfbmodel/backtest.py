"""Walk-forward backtest of the college model against closing and opening lines.

Every number is out of sample: a season's projections, blend weight and curve widths
are fitted on earlier seasons only. Tuning seasons are 2017-2019 (settings and features
were chosen on them); 2021 on is the holdout; the shortened 2020 season is in neither.
Spreads and totals are graded at -110 (the data has no historical prices) at the
consensus line rounded to the nearest half point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflmodel import odds as O
from nflmodel.model import _blend_weight

from . import bets as B
from . import config as C

EV_STEPS = (0.0, 0.02, 0.05)


def period(season):
    s = np.asarray(season)
    return np.where(s >= C.HOLDOUT_FROM, "holdout", np.where(np.isin(s, C.SKIP_TUNING), "2020", "tuning"))


def grade_all(games: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts, dists = [], {}
    for s in sorted(frame.season.unique()):
        dists[int(s)] = B.fit_distribution(games[games.fbs_game == 1], int(s))
        played = frame[(frame.season == s) & frame.played & frame.spread_line.notna() & frame.total_line.notna()]
        for src in ("model", "proj"):
            b = B.price_games(played, dists[int(s)], src)
            b["source"] = src
            parts.append(b)
    return pd.concat(parts, ignore_index=True), dists


def _record(b: pd.DataFrame) -> dict:
    w, l, p = int((b.outcome > 0).sum()), int((b.outcome < 0).sum()), int((b.outcome == 0).sum())
    n = len(b)
    return {"n": n, "w": w, "l": l, "p": p,
            "win_pct": round(w / (w + l), 4) if w + l else None,
            "roi": round(float(b.profit.mean()), 4) if n else None,
            "units": round(float(b.profit.sum()), 1),
            "claimed_ev": round(float(b.ev.mean()), 4) if n else None,
            "roi_se": round(float(b.profit.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else None}


def leans(b: pd.DataFrame) -> pd.DataFrame:
    return b.loc[b.groupby(["game_id", "market"]).ev.idxmax()]


def accuracy(x: pd.DataFrame) -> dict:
    mae = lambda a, b: round(float(np.mean(np.abs(a - b))), 3)
    decided = x[x.result != 0]
    return {"n": int(len(x)),
            "margin_mae": {k: mae(x[c], x.result) for k, c in (("model", "model_margin"), ("blend", "proj_margin"), ("market", "spread_line"))},
            "total_mae": {k: mae(x[c], x.total) for k, c in (("model", "model_total"), ("blend", "proj_total"), ("market", "total_line"))},
            "winner_pct": {k: round(float(np.mean(np.sign(decided[c]) == np.sign(decided.result))), 4)
                           for k, c in (("model", "model_margin"), ("blend", "proj_margin"), ("market", "spread_line"))}}


def summarize(frame: pd.DataFrame, graded: pd.DataFrame) -> dict:
    f = frame[frame.played & frame.spread_line.notna() & frame.total_line.notna()].copy()
    f["period"] = period(f.season)
    wk = np.where(f.postseason == 1, 99, f.week)
    out = {"first_season": int(f.season.min()), "last_season": int(f.season.max()), "holdout_from": C.HOLDOUT_FROM,
           "games": int(len(f)),
           "accuracy": {p: accuracy(f[f.period == p]) for p in ("tuning", "2020", "holdout")},
           "by_season": [{"season": int(s), **accuracy(x), "w_margin": round(float(x.w_margin.iloc[0]), 3),
                          "w_total": round(float(x.w_total.iloc[0]), 3)} for s, x in f.groupby("season")],
           "by_week": [{"weeks": lab, **accuracy(f[m & (f.season >= C.HOLDOUT_FROM)])}
                       for lab, m in (("0-2", wk <= 2), ("3-5", (wk >= 3) & (wk <= 5)), ("6-9", (wk >= 6) & (wk <= 9)),
                                      ("10+", (wk >= 10) & (wk < 99)), ("Bowls", wk == 99))]}
    g = graded.copy()
    g["period"] = period(g.season)
    rec = {}
    for src in ("model", "proj"):
        rec[src] = {}
        for market in ("Spread", "Total"):
            m = g[(g.source == src) & (g.market == market)]
            rows = []
            rows.append({"rule": "Every game", **{p: _record(leans(m[m.period == p])) for p in ("tuning", "holdout")},
                         "all": _record(leans(m))})
            for t in EV_STEPS:
                x = m[m.ev > t]
                rows.append({"rule": f"EV above {int(t * 100)}%", **{p: _record(x[x.period == p]) for p in ("tuning", "holdout")},
                             "all": _record(x)})
            rec[src][market] = {"rows": rows, "by_season": [{"season": int(s), **_record(leans(x))} for s, x in m.groupby("season")]}
    out["records"] = rec
    return out


# --------------------------------------------------------------------------
# Opening lines (2021 on)
# --------------------------------------------------------------------------

def _short_week(games: pd.DataFrame) -> pd.Series:
    """True when either team played in the five days before the game: its opener was
    likely posted before that earlier game, so the model would know a result the opener did not."""
    p = games[games.played].sort_values("gameday")
    long = pd.concat([p[["game_id", "season", "gameday"]].assign(team=p[s]) for s in ("home_team", "away_team")])
    long = long[long.team != C.POOL].sort_values(["team", "gameday"])
    long["gap"] = long.groupby(["team", "season"])["gameday"].diff().dt.days
    return (long["gap"] <= 5).groupby(long["game_id"]).any()


def opening_report(games: pd.DataFrame, frame: pd.DataFrame, dists: dict | None = None) -> dict | None:
    """The model against the opening spread and total. `frame` should come from a run that
    assumes each team starts last week's quarterback (openers precede late QB news)."""
    f = frame[frame.played & frame.open_spread.notna() & frame.spread_line.notna()].copy()
    if not len(f):
        return None
    short = _short_week(games)
    f["short"] = f["game_id"].map(short).fillna(False).to_numpy(bool)
    every = f[~f.short]
    out = {"games": int(len(every)), "left_out_short_week": int(f.short.sum()),
           "first_season": int(every.season.min()), "last_season": int(every.season.max())}
    for kind, line, close, actual, mcol in (("spread", "open_spread", "spread_line", "result", "model_margin"),
                                            ("total", "open_total", "total_line", "total", "model_total")):
        x = every[every[line].notna()].copy()
        x["openh"] = B.half(x[line])
        x["d"] = x[mcol] - x[line]
        pick = np.sign(x["d"])
        x["cover"] = np.sign(x[actual] - x["openh"]) * pick
        x["cover_close"] = np.sign(x[actual] - B.half(x[close])) * pick
        x["clv"] = (x[close] - x[line]) * pick
        dec = O.american_to_decimal(B.DEFAULT_PRICE)
        x["profit"] = np.where(x.cover > 0, dec - 1, np.where(x.cover < 0, -1.0, 0.0))
        x = x[x.d != 0]

        def rec(y):
            w, l, p = int((y.cover > 0).sum()), int((y.cover < 0).sum()), int((y.cover == 0).sum())
            wc, lc = int((y.cover_close > 0).sum()), int((y.cover_close < 0).sum())
            n = len(y)
            return {"n": n, "w": w, "l": l, "p": p, "win_pct": round(w / (w + l), 4) if w + l else None,
                    "roi": round(float(y.profit.mean()), 4) if n else None,
                    "roi_se": round(float(y.profit.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else None,
                    "toward": round(float((y.clv > 0).mean()), 4) if n else None,
                    "away": round(float((y.clv < 0).mean()), 4) if n else None,
                    "clv": round(float(y.clv.mean()), 3) if n else None,
                    "close_win_pct": round(wc / (wc + lc), 4) if wc + lc else None}
        steps = (0, 1, 2, 3, 5) if kind == "spread" else (0, 1, 2, 3, 5)
        mae = lambda a, b: round(float(np.mean(np.abs(a - b))), 3)
        out[kind] = {"rows": [{"rule": "Every game" if t == 0 else f"Disagrees by {t}+ points", "threshold": t, **rec(x[x.d.abs() >= t])}
                              for t in steps],
                     "mae": {"model": mae(x[mcol], x[actual]), "open": mae(x[line], x[actual]), "close": mae(x[close], x[actual])},
                     "weight": round(_blend_weight(x[mcol].to_numpy(float), x[line].to_numpy(float), x[actual].to_numpy(float)), 3)}
    return out


# --------------------------------------------------------------------------
# Wind and totals
# --------------------------------------------------------------------------

def wind_report(frame: pd.DataFrame) -> dict:
    f = frame[frame.played & frame.total_line.notna() & (frame.indoor == 0) & (frame.wind_known == 1)].copy()
    f["under"] = np.sign(B.half(f.total_line) - f.total)
    dec = O.american_to_decimal(B.DEFAULT_PRICE)
    f["profit"] = np.where(f.under > 0, dec - 1, np.where(f.under < 0, -1.0, 0.0))

    def rec(x):
        w, l, p = int((x.under > 0).sum()), int((x.under < 0).sum()), int((x.under == 0).sum())
        n = len(x)
        return {"n": n, "w": w, "l": l, "p": p, "win_pct": round(w / (w + l), 4) if w + l else None,
                "roi": round(float(x.profit.mean()), 4) if n else None,
                "roi_se": round(float(x.profit.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else None,
                "close_minus_actual": round(float((x.total_line - x.total).mean()), 2) if n else None}
    tiers = [(f"{a}+ mph" if b >= 99 else f"{a}-{b - 1} mph", (f.wind >= a) & (f.wind < b)) for a, b in ((0, 10), (10, 15), (15, 20), (20, 99))]
    out = {"tiers": []}
    for lab, m in tiers:
        out["tiers"].append({"label": lab, "all": rec(f[m]), "tuning": rec(f[m & (f.season < 2020)]), "holdout": rec(f[m & (f.season >= C.HOLDOUT_FROM)])})
    return out
