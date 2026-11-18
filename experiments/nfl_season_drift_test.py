"""Pre-set test: an in-season drift correction for the NFL player projections.

    PFF_DATA_DIR=<pff-data clone> python experiments/nfl_season_drift_test.py

Rule written before running (Project doc claude/plan-nfl-season-drift-test-2026-11-18.md). Lead: the
Weekly Review's Players page (2026): NFL players reach Typical 55% of the time. Across 2019-25 the
projections are centred, but single seasons drift.

For each stat and position group (passing QB; rushing RB; receiving WR, TE, RB each alone), before week w:
    r = (sum actual - sum projection) / sum projection
  over the group's earlier player-games that season (played, the measure's minimum; quarterbacks:
  starts only); 0 until 100 such player-games (quarterbacks 60).
    new = proj * (1 + k * r)
A: k per stat by least squares on earlier seasons (from 2014), held in [0, 1].  B: k = 0.5.
C: as A with r from the last 4 weeks only (minimum counts halved).
Measures and rule: as experiments/nfl_lift_shrink_test.py (six measures per stat; version chosen on
tuning; kept only if better on all six and each holdout RMSE gain >= 1 paired SE).

Writes experiments/out/nfl_season_drift_test.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import nfl_lift_shrink_test as L  # noqa: E402

OUT = HERE / "out" / "nfl_season_drift_test.json"
VERSIONS = ("A", "B", "C")
GROUPS = {"pass_yds": ("QB",), "rush_yds": ("RB",), "rec_yds": ("WR", "TE", "RB")}


def drift(p: pd.DataFrame, st: str, last4: bool) -> np.ndarray:
    """r for every row: the group's running miss this season, from earlier weeks only."""
    m = L.mask(p, st).to_numpy()
    if st == "pass_yds":
        m = m & (p["started"] == 1).to_numpy()
    need = (60 if st == "pass_yds" else 100) / (2 if last4 else 1)
    r = np.zeros(len(p))
    d = p.loc[m, ["season", "week", "group", st, st + "_proj"]]
    wk = d.groupby(["season", "group", "week"]).agg(y=(st, "sum"), pj=(st + "_proj", "sum"), n=(st, "size")).reset_index()
    look = {}
    for (s, g), x in wk.groupby(["season", "group"]):
        x = x.sort_values("week")
        for w in range(1, 23):
            e = x[(x["week"] < w) & ((x["week"] >= w - 4) if last4 else True)]
            if e["n"].sum() >= need and e["pj"].sum() > 0:
                look[(s, g, w)] = float((e["y"].sum() - e["pj"].sum()) / e["pj"].sum())
    key = list(zip(p["season"], p["group"], p["week"]))
    ok = p["group"].isin(GROUPS[st]).to_numpy()
    return np.array([look.get(k, 0.0) if o else 0.0 for k, o in zip(key, ok)])


def adjusted(p: pd.DataFrame, st: str, v: str):
    r = drift(p, st, v == "C")
    m = L.mask(p, st).to_numpy()
    proj, y, season = p[st + "_proj"].to_numpy(float), p[st].to_numpy(float), p["season"].to_numpy()
    x = proj * r
    new, ks = proj.copy(), {}
    for s in sorted(set(season[season >= 2014])):
        if v == "B":
            k = 0.5
        else:
            e = m & (season < s) & (season >= 2014)
            k = float(np.clip((x[e] @ (y[e] - proj[e])) / max(1e-9, x[e] @ x[e]), 0, 1)) if e.any() else 0.0
        new[season == s] = proj[season == s] + k * x[season == s]
        ks[int(s)] = round(k, 3)
    return new, ks, r


def main():
    T = L.tables()
    res = {}
    for st in L.STATS:
        r = {"tuning": {}}
        cand = {v: {w: adjusted(T[w], st, v) for w in ("gameday", "bettime")} for v in VERSIONS}
        for v in VERSIONS:
            r["tuning"][v] = {w: L.rmse(T[w], cand[v][w][0], st, *L.TUNE) for w in ("gameday", "bettime")}
        g = T["gameday"]
        now = g[(g["season"] == 2026)].assign(r=cand["A"]["gameday"][2][(g["season"] == 2026).to_numpy()])
        r["r_2026_latest_week"] = {grp: round(float(x.sort_values("week")["r"].iloc[-1]), 4) for grp, x in now[now["group"].isin(GROUPS[st])].groupby("group")}
        ok = [v for v in VERSIONS if all(r["tuning"][v][w]["diff"] < 0 for w in ("gameday", "bettime"))]
        r["better_on_both_tuning"] = ok
        r["k_by_season"] = {v: cand[v]["gameday"][1] for v in ("A", "C")}
        if not ok:
            r["verdict"] = "No version better on both tuning measures: nothing carried to the holdout, nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["gameday"]["candidate"])
            r["chosen"] = pick
            r["holdout"] = {w: L.rmse(T[w], cand[pick][w][0], st, *L.HOLD) for w in ("gameday", "bettime")}
            import nfl_pff_pairs_test as T0
            frames = {}
            for w in ("gameday", "bettime"):
                c = T[w].copy()
                c[st + "_proj"] = cand[pick][w][0]
                frames[w] = c
            r["brier"] = {"closing lines, game-day": T0.brier(T["gameday"], frames["gameday"], "close", (st,)),
                          "pricing snapshot, bet-time": T0.brier(T["bettime"], frames["bettime"], "flag", (st,))}
            six = list(r["tuning"][pick].values()) + list(r["holdout"].values()) + list(r["brier"].values())
            better = all(x["diff"] < 0 for x in six)
            one_se = all(-x["diff"] >= x["se"] for x in r["holdout"].values())
            r["verdict"] = "KEPT (provisional)" if better and one_se else ("Trade-off: better on all six, a holdout gain under 1 SE" if better else "Trade-off: mixed across the six measures")
        res[st] = r
        print(st, json.dumps(r, default=str), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
