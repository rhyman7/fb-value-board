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
from . import model

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


# --------------------------------------------------------------------------
# Against opening lines
# --------------------------------------------------------------------------

def _came_off_monday(games: pd.DataFrame) -> pd.Series:
    """True for games where either team played on a Monday the week before.
    Their opening line was posted before that game, so the model would be
    using a result the opener could not have known."""
    p = games[games.played].sort_values("gameday")
    long = pd.concat([p[["season", "week", "gameday", "weekday", "game_id"]].assign(team=p[side]) for side in ("home_team", "away_team")])
    long = long.sort_values(["team", "gameday"])
    long["prev_day"] = long.groupby(["team", "season"])["weekday"].shift(1)
    long["prev_week"] = long.groupby(["team", "season"])["week"].shift(1)
    flag = (long["prev_day"] == "Monday") & (long["prev_week"] == long["week"] - 1)
    return flag.groupby(long["game_id"]).any()


def early_weight(clean: pd.DataFrame, before_season: int) -> float:
    """Share of the model's disagreement with an opening spread that has shown up
    in results, measured on the seasons before `before_season`."""
    past = clean[(clean.season < before_season) & (clean.season >= before_season - C.BLEND_LOOKBACK)]
    return model._blend_weight(past.model_margin.to_numpy(), past.open_spread.to_numpy(), past.result.to_numpy())


def opening_report(games: pd.DataFrame, frame: pd.DataFrame, openers: pd.DataFrame, dists: dict | None = None) -> dict | None:
    """How the model does when the bet is placed at the opening spread.

    `frame` must come from a model run that assumes each team starts whoever
    started the week before, so no late quarterback news leaks in. Games where
    a team played the previous Monday night are left out for the same reason.
    """
    if openers is None or not len(openers):
        return None
    f = frame.merge(openers, on="game_id", how="inner")
    monday = _came_off_monday(games)
    f["monday"] = f["game_id"].map(monday).fillna(False).to_numpy(bool)
    every = f[f.played & f.spread_line.notna() & ~f["monday"]]          # all seasons, for walk-forward weights
    f = f[f.played & f.spread_line.notna() & (f.season >= C.FIRST_TEST_SEASON)].copy()
    f["d"] = f["model_margin"] - f["open_spread"]            # model's disagreement with the opener
    f["move"] = f["spread_line"] - f["open_spread"]          # where the line went by the close
    pick = np.sign(f["d"])
    f["cover"] = np.sign(f["result"] - f["open_spread"]) * pick
    f["cover_close"] = np.sign(f["result"] - f["spread_line"]) * pick
    f["clv"] = f["move"] * pick                              # points of closing line value
    price = np.where(pick > 0, f["open_home_odds"], f["open_away_odds"])
    price = np.where(np.isnan(price), B.DEFAULT_PRICE, price)
    dec = O.american_to_decimal(price)
    f["profit"] = np.where(f["cover"] > 0, dec - 1, np.where(f["cover"] < 0, -1.0, 0.0))
    clean = f[~f["monday"] & (f["d"] != 0)]

    def rec(x: pd.DataFrame) -> dict:
        w, l, p = int((x.cover > 0).sum()), int((x.cover < 0).sum()), int((x.cover == 0).sum())
        wc, lc = int((x.cover_close > 0).sum()), int((x.cover_close < 0).sum())
        n = len(x)
        return {"n": n, "w": w, "l": l, "p": p, "win_pct": round(w / (w + l), 4) if w + l else None,
                "roi": round(float(x.profit.mean()), 4) if n else None,
                "roi_se": round(float(x.profit.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else None,
                "toward": round(float((x.clv > 0).mean()), 4) if n else None,
                "away": round(float((x.clv < 0).mean()), 4) if n else None,
                "clv": round(float(x.clv.mean()), 3) if n else None,
                "close_win_pct": round(wc / (wc + lc), 4) if wc + lc else None}

    rows = []
    for th in (0, 1, 2, 3):
        x = clean[clean.d.abs() >= th]
        rows.append({"rule": "Every game" if th == 0 else f"Disagrees by {th}+ points", "threshold": th,
                     **rec(x), "holdout": rec(x[x.season >= C.HOLDOUT_FROM])})

    # How far the line moves, by how far the model disagreed with the opener.
    edges = [-99, -4, -3, -2, -1, 0, 1, 2, 3, 4, 99]
    bins = []
    for _, x in clean.groupby(pd.cut(clean.d, edges), observed=True):
        if len(x) >= 30:
            bins.append({"d": round(float(x.d.mean()), 2), "move": round(float(x.move.mean()), 3),
                         "se": round(float(x.move.std(ddof=1) / np.sqrt(len(x))), 3), "n": int(len(x))})
    d, mv = clean.d.to_numpy(), clean.move.to_numpy()
    mae = lambda a, b: round(float(np.mean(np.abs(a - b))), 3)
    with_monday = f[f.d.abs() >= 2]

    # Early-week pricing, replayed: each season priced at the opener with the weight
    # known at the time. Does the EV it claimed match what the bets returned?
    flagged = None
    if dists:
        parts = []
        for s in sorted(clean.season.unique()):
            if int(s) not in dists:
                continue
            x = clean[clean.season == s].copy()
            w = early_weight(every, int(s))
            x["spread_line"], x["home_spread_odds"], x["away_spread_odds"] = x["open_spread"], x["open_home_odds"], x["open_away_odds"]
            x["proj_margin"] = x["open_spread"] + w * (x["model_margin"] - x["open_spread"])
            b = B.price_games(x, dists[int(s)], "proj")
            parts.append(b[(b.market == "Spread") & (b.ev > 0)])
        if parts:
            fb = pd.concat(parts, ignore_index=True)
            flagged = {**_record(fb), "holdout": _record(fb[fb.season >= C.HOLDOUT_FROM])}
    return {
        "first_season": int(clean.season.min()), "last_season": int(clean.season.max()), "games": int(len(clean)),
        "left_out_monday": int(f["monday"].sum()),
        "rows": rows, "bins": bins,
        "slope": round(float(d @ mv / (d @ d)), 3), "corr": round(float(np.corrcoef(d, mv)[0, 1]), 3),
        "weight": round(float(early_weight(every, int(every.season.max()) + 1)), 3),
        "flagged": flagged,
        "mae": {"model": mae(clean.model_margin, clean.result), "open": mae(clean.open_spread, clean.result), "close": mae(clean.spread_line, clean.result)},
        "mean_move": round(float(np.abs(mv).mean()), 2),
        "including_monday_2plus": rec(with_monday),
    }
