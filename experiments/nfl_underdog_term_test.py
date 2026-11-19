"""Pre-set test: an underdog term in the NFL yardage projections (Nov 19, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/nfl_underdog_term_test.py

Rule written before running (Project doc claude/plan-nfl-underdog-term-test-2026-11-19.md). Leads: the
college big-underdog quarterback term (experiments/chatgpt_player_changes_test.py, 7 of 8) and the NFL
floor-rung diagnosis's cell (rungs on teams projected to lose by 7+ cleared 73% against the books' 78%).

The change, after the board's final projection (boosted and Questionable steps included):
    new = proj - b * max(0, -margin - T)          margin: the board's projected margin for his team
  b: one number per stat, least squares through the origin on earlier seasons only (from 2014), >= 0.
Versions: A T = 7, B T = 3, C T = 0.
Stats, each judged on its own: pass_yds (QB), rush_yds (RB), rec_yds (WR/TE/RB); rows as in
experiments/nfl_lift_shrink_test.py.
Measures per stat (6): RMSE, game-day and bet-time, tuning 2014-18 and holdout 2019-25; Brier at the
2023-25 DK/FD main lines, closing lines with game-day projections and the pricing snapshot with bet-time.
Rule: the version is chosen on tuning alone (better on both tuning RMSE measures, then the lowest game-day
tuning RMSE). Kept only if better on all six and each holdout RMSE gain >= 1 paired SE. Mixed, or under
1 SE: a trade-off for Ryan. No version better on both tuning measures: nothing goes to the holdout.
Reported, not used: bias on affected rows and by size of margin, the term for favourites, 2026 so far, MAE.

Writes experiments/out/nfl_underdog_term_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import nfl_lift_shrink_test as L  # noqa: E402

OUT = HERE / "out" / "nfl_underdog_term_test.json"
VERSIONS = {"A": 7.0, "B": 3.0, "C": 0.0}
STATS = L.STATS
TUNE, HOLD = L.TUNE, L.HOLD


def hinge(p: pd.DataFrame, T: float, side: str = "dog") -> np.ndarray:
    m = p["margin"].to_numpy(float)
    return np.nan_to_num(np.maximum(0.0, (-m if side == "dog" else m) - T))


def adjusted(p: pd.DataFrame, st: str, T: float, side: str = "dog") -> tuple[np.ndarray, dict]:
    h, m = hinge(p, T, side), L.mask(p, st).to_numpy()
    proj, y, season = p[st + "_proj"].to_numpy(float), p[st].to_numpy(float), p["season"].to_numpy()
    new, bs = proj.copy(), {}
    for s in sorted(set(season[season >= 2014])):
        k = m & (season < s) & (season >= 2013) & (h > 0)
        b = float(max(0.0, (h[k] @ (proj[k] - y[k])) / max(1e-9, h[k] @ h[k]))) if k.sum() >= 100 else 0.0
        new[season == s] = np.maximum(0.0, proj[season == s] - b * h[season == s])
        bs[int(s)] = round(b, 3)
    return new, bs


def detail(p, new, st, lo, hi, h) -> dict:
    k = (L.mask(p, st) & p["season"].between(lo, hi)).to_numpy() & (h > 0)
    if not k.any():
        return {"n": 0}
    y, old, nw = p[st].to_numpy(float)[k], p[st + "_proj"].to_numpy(float)[k], new[k]
    se = float(y.std(ddof=1) / np.sqrt(k.sum()))
    return {"n": int(k.sum()), "bias_now": round(float((y - old).mean()), 2), "bias_new": round(float((y - nw).mean()), 2), "se": round(se, 2),
            "rmse_now": round(float(np.sqrt(((old - y) ** 2).mean())), 3), "rmse_new": round(float(np.sqrt(((nw - y) ** 2).mean())), 3),
            "mae_now": round(float(np.abs(old - y).mean()), 3), "mae_new": round(float(np.abs(nw - y).mean()), 3)}


def bands(p, new, st, lo, hi) -> dict:
    m = -p["margin"].to_numpy(float)
    out = {}
    for a, b in ((0, 3), (3, 7), (7, 10), (10, 30)):
        out[f"losing by {a}-{b}"] = detail(p, new, st, lo, hi, ((m > a) & (m <= b)).astype(float))
    out["favoured"] = detail(p, new, st, lo, hi, (m <= 0).astype(float))
    return out


def main():
    T = L.tables()
    res = {"plan": "claude/plan-nfl-underdog-term-test-2026-11-19.md"}
    for st in STATS:
        r = {"tuning": {}}
        cand = {v: {w: adjusted(T[w], st, tau) for w in ("gameday", "bettime")} for v, tau in VERSIONS.items()}
        for v in VERSIONS:
            r["tuning"][v] = {w: L.rmse(T[w], cand[v][w][0], st, *TUNE) for w in ("gameday", "bettime")}
        r["slopes_gameday"] = {v: cand[v]["gameday"][1] for v in VERSIONS}
        r["by_margin_tuning_now"] = bands(T["gameday"], T["gameday"][st + "_proj"].to_numpy(float), st, *TUNE)
        ok = [v for v in VERSIONS if all(r["tuning"][v][w]["diff"] < 0 for w in ("gameday", "bettime"))]
        r["better_on_both_tuning"] = ok
        if not ok:
            r["verdict"] = "No version better on both tuning measures: nothing carried to the holdout, nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["gameday"]["candidate"])
            tau = VERSIONS[pick]
            r["chosen"] = pick
            r["holdout"] = {w: L.rmse(T[w], cand[pick][w][0], st, *HOLD) for w in ("gameday", "bettime")}
            import nfl_pff_pairs_test as T0
            frames = {}
            for w in ("gameday", "bettime"):
                c = T[w].copy()
                c[st + "_proj"] = cand[pick][w][0]
                frames[w] = c
            r["brier"] = {"closing lines, game-day": T0.brier(T["gameday"], frames["gameday"], "close", (st,)),
                          "pricing snapshot, bet-time": T0.brier(T["bettime"], frames["bettime"], "flag", (st,))}
            g = T["gameday"]
            r["affected_holdout"] = detail(g, cand[pick]["gameday"][0], st, *HOLD, hinge(g, tau))
            r["affected_tuning"] = detail(g, cand[pick]["gameday"][0], st, *TUNE, hinge(g, tau))
            r["by_margin_holdout"] = bands(g, cand[pick]["gameday"][0], st, *HOLD)
            r["season_2026_affected"] = detail(g, cand[pick]["gameday"][0], st, 2026, 2026, hinge(g, tau))
            r["season_2026_all"] = L.rmse(g, cand[pick]["gameday"][0], st, 2026, 2026)
            fav, fb = adjusted(g, st, tau, "fav")
            r["favourites_reported"] = {"slopes": fb, "tuning": L.rmse(g, fav, st, *TUNE), "holdout": L.rmse(g, fav, st, *HOLD)}
            six = list(r["tuning"][pick].values()) + list(r["holdout"].values()) + list(r["brier"].values())
            n = sum(x["diff"] < 0 for x in six)
            better = n == 6
            one_se = all(-x["diff"] >= x["se"] for x in r["holdout"].values())
            r["better_on"] = f"{n} of 6"
            r["verdict"] = "KEPT (provisional)" if better and one_se else ("Trade-off: better on all six, a holdout gain under 1 SE" if better else f"Trade-off: better on {n} of 6")
        res[st] = r
        print(st, r["verdict"], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
