"""Pre-set test: does the NFL board over-project players it puts well above their season average?

    PFF_DATA_DIR=<pff-data clone> python experiments/nfl_lift_shrink_test.py

Rule written before running (Project doc claude/plan-nfl-lift-shrink-test-2026-11-18.md). Lead: the
Weekly Review's Best Matchups page (Nov 18, 2026): NFL players picked for a big projected lift over
their season average reached their typical game less often than similar players (2026 weeks 7-10).

The change, after the board's final projection (boosted step included):
    new = proj - b * max(0, proj - (1 + T) * season average)
  season average: players.season_average (earlier games this season; a quarterback's starts); only
  players with 3+ earlier games this season and an average above zero are touched.
  b: one number per stat, least squares on earlier seasons only (from 2014), held in [0, 1].
Versions: A T = 0, B T = 0.25, C T = 0.50.
Stats, each judged on its own: pass_yds (QB), rush_yds (RB), rec_yds (WR/TE/RB).
Measures per stat: RMSE, game-day and bet-time tables, tuning 2014-18 and holdout 2019-25 (players
  who played: QBs with 10+ projected attempts; backs with a 10+ yard rushing projection; WR/TE/RB with
  1+ projected reception). Brier at the 2023-25 DK/FD main lines for the stat: closing lines with
  game-day projections, pricing snapshot with bet-time (the props check).
Rule: the version is chosen on tuning alone (better on both tuning RMSE measures, then the lowest
  game-day tuning RMSE). Kept only if better on all six measures and each holdout RMSE gain >= 1
  paired SE. Mixed, or under 1 SE: a trade-off for Ryan. No version better on both tuning measures:
  nothing is carried to the holdout for that stat.

Writes experiments/out/nfl_lift_shrink_test.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import config as C, players as PL, players_ml as ML  # noqa: E402

OUT = HERE / "out" / "nfl_lift_shrink_test.json"
CACHE = C.RAW / "nfl_lift_shrink_tables.pkl"
VERSIONS = {"A": 0.0, "B": 0.25, "C": 0.50}
STATS = ("pass_yds", "rush_yds", "rec_yds")
TUNE, HOLD = (2014, 2018), (2019, 2025)
MIN_GAMES = 3


def tables() -> dict:
    """The board's full pipeline, game-day and bet-time, projected walk-forward (cached)."""
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    from nflmodel import data, extra, game_roles, injuries, model, pff_qb, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False)
    prior = pff_qb.load(qb, sorted(games.season.unique()))
    print("quarterback prior:", "PFF grades" if prior is not None else "flat (NOT the production model)", flush=True)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = PL.load_availability(refresh_current=False, verbose=False)
    rush = injuries.load_pass_rush(games, refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = PL.player_rows(PL.team_games(f0, pg), pg, avail, pinfo)
    frame, _ = model.walk_forward(game_roles.attach(base, game_roles.team_features(rows, R, avail)))
    out = {}
    for when, pre in (("gameday", False), ("bettime", True)):
        t = PL.build(frame, pg, pinfo, avail, inj, rush, games=games, overrides=None, pregame=pre)
        pall = PL.walk_forward(t, first_test=C.FIRST_TEST_SEASON)
        x = ML.add_features(pall, R)
        ml = ML.questionable(ML.walk_forward(x, first=C.FIRST_TEST_SEASON), first=C.FIRST_TEST_SEASON, snaps=R.get("snaps"))
        p = ml.merge(PL.season_average(t), on=["game_id", "player_id"], how="left")
        # earlier games this season behind the average (a quarterback's starts for passing)
        a = t[t["active"] == 1].sort_values(["player_id", "t"])
        pl = a["played"].astype(bool)
        cnt = a[["game_id", "player_id"]].copy()
        cnt["n_prev"] = pl.astype(float).groupby([a["player_id"], a["season"]]).transform(lambda s: s.shift().expanding().sum()).fillna(0).to_numpy()
        cnt["n_prev_start"] = (pl & (a["started"] == 1)).astype(float).groupby([a["player_id"], a["season"]]).transform(lambda s: s.shift().expanding().sum()).fillna(0).to_numpy()
        out[when] = p.merge(cnt, on=["game_id", "player_id"], how="left").reset_index(drop=True)
        print(when, len(out[when]), flush=True)
    C.RAW.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def mask(p: pd.DataFrame, st: str) -> pd.Series:
    live = p["played"].astype(bool) & (p["active"] == 1) & (p["season"] >= 2014) & p[st].notna()
    if st == "pass_yds":
        return live & (p["group"] == "QB") & (p["pass_att_proj"] >= 10)
    if st == "rush_yds":
        return live & (p["group"] == "RB") & (p["rush_yds_proj"] >= 10)
    return live & p["group"].isin(["WR", "TE", "RB"]) & (p["receptions_proj"] >= 1)


def hinge(p: pd.DataFrame, st: str, T: float) -> np.ndarray:
    avg = p[st + "_savg"].to_numpy(float)
    n = p["n_prev_start" if st == "pass_yds" else "n_prev"].to_numpy(float)
    h = np.maximum(0.0, p[st + "_proj"].to_numpy(float) - (1 + T) * avg)
    return np.where((n >= MIN_GAMES) & (avg > 0), h, 0.0)


def adjusted(p: pd.DataFrame, st: str, T: float) -> tuple[np.ndarray, dict]:
    """The candidate projection, b for each season fitted on the seasons before it."""
    h, m = hinge(p, st, T), mask(p, st).to_numpy()
    proj, y, season = p[st + "_proj"].to_numpy(float), p[st].to_numpy(float), p["season"].to_numpy()
    new, bs = proj.copy(), {}
    for s in sorted(set(season[season >= 2014])):
        k = m & (season < s) & (season >= 2014 - 1)
        b = float(np.clip((h[k] @ (proj[k] - y[k])) / max(1e-9, h[k] @ h[k]), 0, 1)) if k.any() else 0.0
        new[season == s] = proj[season == s] - b * h[season == s]
        bs[int(s)] = round(b, 3)
    return new, bs


def rmse(p, new, st, lo, hi) -> dict:
    k = (mask(p, st) & p["season"].between(lo, hi)).to_numpy()
    y, old = p[st].to_numpy(float)[k], p[st + "_proj"].to_numpy(float)[k]
    e0, e1 = (old - y) ** 2, (new[k] - y) ** 2
    r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
    cl = pd.Series(e1 - e0).groupby(p["game_id"].to_numpy()[k]).sum()
    return {"n": int(k.sum()), "now": round(r0, 4), "candidate": round(r1, 4), "diff": round(r1 - r0, 5),
            "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / k.sum() / (2 * r0)), 5)}


def big_lift(p, new, st, lo, hi) -> dict:
    """Players projected 25%+ above their season average (the Best Overall kind of pick)."""
    k = (mask(p, st) & p["season"].between(lo, hi)).to_numpy() & (hinge(p, st, 0.25) > 0)
    y, old = p[st].to_numpy(float)[k], p[st + "_proj"].to_numpy(float)[k]
    se = float((y - old).std(ddof=1) / np.sqrt(max(1, k.sum())))
    return {"n": int(k.sum()), "avg_actual_minus_proj_now": round(float((y - old).mean()), 2), "se": round(se, 2),
            "after": round(float((y - new[k]).mean()), 2), "met_now": round(float((y >= old).mean()), 3)}


def main():
    T = tables()
    res = {}
    for st in STATS:
        r = {"tuning": {}}
        cand = {}
        for v, tau in VERSIONS.items():
            cand[v] = {w: adjusted(T[w], st, tau) for w in ("gameday", "bettime")}
            r["tuning"][v] = {w: rmse(T[w], cand[v][w][0], st, *TUNE) for w in ("gameday", "bettime")}
        ok = [v for v in VERSIONS if all(r["tuning"][v][w]["diff"] < 0 for w in ("gameday", "bettime"))]
        r["better_on_both_tuning"] = ok
        r["touched_share_tuning"] = {v: round(float((hinge(T["gameday"], st, tau)[(mask(T["gameday"], st) & T["gameday"]["season"].between(*TUNE)).to_numpy()] > 0).mean()), 3) for v, tau in VERSIONS.items()}
        r["big_lift_tuning"] = big_lift(T["gameday"], T["gameday"][st + "_proj"].to_numpy(float), st, *TUNE)
        if not ok:
            r["verdict"] = "No version better on both tuning measures: nothing carried to the holdout, nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["gameday"]["candidate"])
            r["chosen"], r["b_by_season"] = pick, cand[pick]["gameday"][1]
            r["holdout"] = {w: rmse(T[w], cand[pick][w][0], st, *HOLD) for w in ("gameday", "bettime")}
            r["big_lift_holdout"] = big_lift(T["gameday"], cand[pick]["gameday"][0], st, *HOLD)
            import nfl_pff_pairs_test as T0
            frames = {}
            for w in ("gameday", "bettime"):
                c = T[w].copy()
                c[st + "_proj"] = cand[pick][w][0]
                frames[w] = c
            r["brier"] = {"closing lines, game-day": T0.brier(T["gameday"], frames["gameday"], "close", (st,)),
                          "pricing snapshot, bet-time": T0.brier(T["bettime"], frames["bettime"], "flag", (st,))}
            better = all(x["diff"] < 0 for x in list(r["tuning"][pick].values()) + list(r["holdout"].values()) + list(r["brier"].values()))
            one_se = all(-x["diff"] >= x["se"] for x in r["holdout"].values())
            r["verdict"] = "KEPT (provisional)" if better and one_se else ("Trade-off: better on all six, a holdout gain under 1 SE" if better else "Trade-off: mixed across the six measures")
        res[st] = r
        print(st, json.dumps(r, default=str), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
