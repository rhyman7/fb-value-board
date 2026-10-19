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


def summarize(frame: pd.DataFrame, graded: pd.DataFrame, dists: dict) -> dict:
    """The record in the NFL board's shape (so the same page reads it), plus the college splits:
    tuning seasons, 2020 and holdout, and the holdout by part of the season."""
    from .board import win_probs
    f = frame[frame.played & frame.spread_line.notna() & frame.total_line.notna()].copy()
    f["period"] = period(f.season)
    wk = np.where(f.postseason == 1, 99, f.week)
    out = {"first_season": int(f.season.min()), "last_season": int(f.season.max()), "holdout_from": C.HOLDOUT_FROM,
           "tuning": [2017, 2019], "games": int(len(f)),
           "accuracy": accuracy(f), "accuracy_holdout": accuracy(f[f.period == "holdout"]),
           "periods": {p: accuracy(f[f.period == p]) for p in ("tuning", "2020", "holdout")},
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
        for market in ("Spread", "Total", "Moneyline"):
            m = g[(g.source == src) & (g.market == market)]
            ho, tu = m[m.period == "holdout"], m[m.period == "tuning"]
            if not len(m):
                rec[src][market] = {"rows": [], "by_season": []}
                continue
            rows = [{"rule": "Every game", **_record(leans(m)), "holdout": _record(leans(ho)), "tuning": _record(leans(tu))}]
            rows += [{"rule": f"EV above {int(t * 100)}%", **_record(m[m.ev > t]), "holdout": _record(ho[ho.ev > t]), "tuning": _record(tu[tu.ev > t])}
                     for t in EV_STEPS]
            rec[src][market] = {"rows": rows, "by_season": [{"season": int(s), **_record(leans(x))} for s, x in m.groupby("season")]}
    out["records"] = rec

    ev_cal, edges = {}, [-1, -0.04, -0.02, 0, 0.02, 0.04, 0.07, 0.11, 1]
    for src in ("model", "proj"):
        x = g[(g.source == src) & (g.market != "Moneyline")].copy()
        x["bucket"] = pd.cut(x.ev, edges)
        ev_cal[src] = [{"claimed": round(float(y.ev.mean()), 4), "realized": round(float(y.profit.mean()), 4),
                        "se": round(float(y.profit.std(ddof=1) / np.sqrt(len(y))), 4), "n": int(len(y))}
                       for _, y in x.groupby("bucket", observed=True) if len(y) >= 40]
    out["ev_calibration"] = ev_cal

    # win chances: the board's adjusted chance against how often the home team won
    parts = [win_probs(x, dists[int(s)]).assign(result=x.result.to_numpy()) for s, x in f.groupby("season") if int(s) in dists]
    w = pd.concat(parts)
    w = w[w.result != 0]
    w["bin"] = pd.cut(w["adj"], np.linspace(0, 1, 11))
    out["calibration"] = [{"predicted": round(float(y.adj.mean()), 4), "market": round(float(y.market.mean()), 4),
                           "actual": round(float((y.result > 0).mean()), 4), "n": int(len(y))}
                          for _, y in w.groupby("bin", observed=True) if len(y) >= 30]

    picks = {
        "Raw model, every spread": leans(g[(g.source == "model") & (g.market == "Spread")]),
        "Market-adjusted, spreads with EV above 0": g[(g.source == "proj") & (g.market == "Spread") & (g.ev > 0)],
        "Raw model, every total": leans(g[(g.source == "model") & (g.market == "Total")]),
    }
    weeks = sorted(set(zip(f.season, np.where(f.postseason == 1, C.POSTSEASON_WEEK, f.week))))
    gw = dict(zip(f.game_id, np.where(f.postseason == 1, C.POSTSEASON_WEEK, f.week)))
    series = {}
    for name, b in picks.items():
        b = b.assign(wk=b.game_id.map(gw))
        per = b.groupby(["season", "wk"]).profit.sum()
        cum, pts = 0.0, []
        for s, wv in weeks:
            cum += float(per.get((s, wv), 0.0))
            pts.append(round(cum, 2))
        series[name] = pts
    out["cumulative"] = {"weeks": [f"{s} W{wv}" if wv < C.POSTSEASON_WEEK else f"{s} Bowls" for s, wv in weeks],
                         "seasons": [int(s) for s, _ in weeks], "series": series}
    return out


def key_number_report(games: pd.DataFrame, dist: dict, before_season: int, lookback: int = 8) -> dict:
    from nflmodel import distributions as D
    h = games[games.played & games.spread_line.notna() & (games.fbs_game == 1) & (games.season < before_season) & (games.season >= before_season - lookback)]
    sp = pd.Series(B.half(h.spread_line), index=h.index)
    absm = h.result.abs().astype(int)
    smooth = D.pmf(sp, dist["sd_margin"], B.MARGINS).sum(axis=0)
    mid = len(B.MARGINS) // 2
    smooth = (smooth + smooth[::-1])[mid:]
    smooth[0] /= 2
    freq = [{"margin": k, "actual": round(float((absm == k).mean()), 4), "smooth": round(float(smooth[k] / len(h)), 4),
             "mult": round(float(dist["margin"][mid + k]), 2)} for k in range(0, 22)]
    pushes = []
    for ln in (3, 7, 6, 10, 4, 14):
        x = h[sp.abs() == ln]
        if len(x) < 40:
            continue
        xs = B.half(x.spread_line)
        loc = D.anchor(xs, 0.5, dist["sd_margin"], B.MARGINS, dist["margin"])
        key = D.over_push_under(D.pmf(loc, dist["sd_margin"], B.MARGINS, dist["margin"]), B.MARGINS, xs)[1].mean()
        bell = D.over_push_under(D.pmf(xs, dist["sd_margin"], B.MARGINS), B.MARGINS, xs)[1].mean()
        pushes.append({"line": ln, "n": int(len(x)), "actual": round(float((x.result == xs).mean()), 4),
                       "model": round(float(key), 4), "bell": round(float(bell), 4)})
    return {"seasons": f"{before_season - lookback}-{before_season - 1}", "games": int(len(h)), "frequency": freq, "pushes": pushes,
            "sd_margin": round(dist["sd_margin"], 2), "sd_total": round(dist["sd_total"], 2), "sd_win": round(dist["sd_win"], 2)}


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
    """The model against the opening spread (top level, the NFL page's shape) and the opening
    total (under "total"). `frame` should come from a run that assumes last week's starting
    quarterbacks and a typical wind: openers are posted before that news."""
    f = frame[frame.played & frame.open_spread.notna() & frame.spread_line.notna()].copy()
    if not len(f):
        return None
    short = _short_week(games)
    f["short"] = f["game_id"].map(short).fillna(False).to_numpy(bool)
    every = f[~f.short]
    dec = O.american_to_decimal(B.DEFAULT_PRICE)

    def prep(x, line, close, actual, mcol):
        x = x[x[line].notna()].copy()
        x["openh"] = B.half(x[line])
        x["d"] = x[mcol] - x[line]
        pick = np.sign(x["d"])
        x["cover"] = np.sign(x[actual] - x["openh"]) * pick
        x["cover_close"] = np.sign(x[actual] - B.half(x[close])) * pick
        x["move"] = x[close] - x[line]
        x["clv"] = x["move"] * pick
        x["profit"] = np.where(x.cover > 0, dec - 1, np.where(x.cover < 0, -1.0, 0.0))
        return x[x.d != 0]

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

    mae = lambda a, b: round(float(np.mean(np.abs(a - b))), 3)

    def report(x, mcol, line, close, actual):
        rows = []
        for t in (0, 1, 2, 3, 5):
            y = x[x.d.abs() >= t]
            r = rec(y)
            rows.append({"rule": "Every game" if t == 0 else f"Disagrees by {t}+ points", "threshold": t, **r,
                         "holdout": rec(y[y.season >= C.HOLDOUT_FROM])})
        bins = []
        for _, y in x.groupby(pd.cut(x.d, [-99, -5, -3, -2, -1, 0, 1, 2, 3, 5, 99]), observed=True):
            if len(y) >= 30:
                bins.append({"d": round(float(y.d.mean()), 2), "move": round(float(y.move.mean()), 3),
                             "se": round(float(y.move.std(ddof=1) / np.sqrt(len(y))), 3), "n": int(len(y))})
        d, mv = x.d.to_numpy(), x.move.to_numpy()
        return {"rows": rows, "bins": bins, "slope": round(float(d @ mv / (d @ d)), 3), "corr": round(float(np.corrcoef(d, mv)[0, 1]), 3),
                "weight": round(_blend_weight(x[mcol].to_numpy(float), x[line].to_numpy(float), x[actual].to_numpy(float)), 3),
                "mae": {"model": mae(x[mcol], x[actual]), "open": mae(x[line], x[actual]), "close": mae(x[close], x[actual])},
                "mean_move": round(float(np.abs(mv).mean()), 2)}

    xs = prep(every, "open_spread", "spread_line", "result", "model_margin")
    out = {"games": int(len(every)), "left_out_monday": int(f.short.sum()), "left_out_short_week": int(f.short.sum()),
           "first_season": int(every.season.min()), "last_season": int(every.season.max()),
           **report(xs, "model_margin", "open_spread", "spread_line", "result")}
    # the early-line pricing, replayed: each season priced at the opener with the weight known before it
    if dists:
        parts = []
        for s in sorted(xs.season.unique()):
            past = xs[xs.season < s]
            if int(s) not in dists or len(past) < 200:
                continue
            w = _blend_weight(past.model_margin.to_numpy(float), past.open_spread.to_numpy(float), past.result.to_numpy(float))
            y = xs[xs.season == s].copy()
            y["spread_line"] = y["open_spread"]
            for c in ("home_spread_odds", "away_spread_odds"):
                y[c] = np.nan
            y["proj_margin"] = y["open_spread"] + w * (y["model_margin"] - y["open_spread"])
            b = B.price_games(y, dists[int(s)], "proj")
            parts.append(b[(b.market == "Spread") & (b.ev > 0)])
        if parts:
            fb = pd.concat(parts, ignore_index=True)
            out["flagged"] = {**_record(fb), "holdout": _record(fb[fb.season >= C.HOLDOUT_FROM])}
    xt = prep(every, "open_total", "total_line", "total", "model_total")
    out["total"] = report(xt, "model_total", "open_total", "total_line", "total")
    return out


# --------------------------------------------------------------------------
# Wind and totals
# --------------------------------------------------------------------------

def wind_report(frame: pd.DataFrame, lo: float = 10.0, hi: float = 20.0) -> dict:
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
    band = f[(f.wind >= lo) & (f.wind < hi)]
    eras = [("2017-2019", band[band.season.between(2017, 2019)]), ("2020", band[band.season == 2020]),
            (f"{C.HOLDOUT_FROM}-", band[band.season >= C.HOLDOUT_FROM])]
    tiers = [(f"{a}+ mph" if b >= 99 else f"{a}-{b - 1} mph", f[(f.wind >= a) & (f.wind < b)]) for a, b in ((0, 10), (10, 15), (15, 20), (20, 99))]
    return {"band": [lo, hi], "all": rec(band), "eras": [{"label": k, **rec(x)} for k, x in eras],
            "tiers": [{"label": k, **rec(x)} for k, x in tiers]}
