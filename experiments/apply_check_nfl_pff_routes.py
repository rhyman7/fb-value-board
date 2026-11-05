"""Check before applying PFF route share to the NFL New projections (nflmodel/pff_routes.py).

    PFF_DATA_DIR=../pff-data python experiments/apply_check_nfl_pff_routes.py

Nov 5, 2026. Ryan applied the route inputs as-is (PLAYERS.md "PFF route share in the New
projections"). On the board's own path (players_ml.add_features with the private data, then
walk_forward for the current season, then the Questionable step), against the same path with the
route inputs off: how far the upcoming games' WR and RB receptions and receiving yards move,
and that with no PFF data the projections are exactly today's. The board's version differs a
little from the tested one (rookies matched by name when nflverse has no pff_id yet), so
the test's 8 walk-forward RMSE measures are re-run on the board's code (--rmse). Writes
experiments/out/apply_check_nfl_pff_routes.json (summary numbers only, no PFF rows).
"""
from __future__ import annotations

import argparse
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
from nflmodel import config as C, data, players_ml as ML, pff_routes, roles  # noqa: E402
from props_calibration import load_proj  # noqa: E402

OUT = HERE / "out" / "apply_check_nfl_pff_routes.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmse", action="store_true")
    args = ap.parse_args()
    games = data.load_games()
    R = roles.load_all(games, verbose=False)
    pall = load_proj(None)
    season = int(pall["season"].max())
    on = ML.add_features(pall, R)                       # the board's path: routes when the data is there
    off = ML.add_features(pall, R, routes=False)
    assert ML.feats_for(on) == ML.FEATS + ML.ROUTE_F, "route inputs missing: is the PFF data there?"
    assert ML.feats_for(off) == ML.FEATS
    a = ML.questionable(ML.walk_forward(on, first=season), first=C.FIRST_TEST_SEASON)
    b = ML.questionable(ML.walk_forward(off, first=season), first=C.FIRST_TEST_SEASON)
    # no PFF data at all: an empty frame from pff_routes, today's features, identical projections
    empty = pff_routes.features(pall, rec=pd.DataFrame())
    assert empty.empty
    up = (a["season"] == season) & ~a["played"].fillna(False).astype(bool)
    res = {"season": season, "upcoming_rows": int(up.sum()), "weeks": sorted(int(w) for w in a.loc[up, "week"].unique())}
    nxt = int(a.loc[up, "week"].min())
    res["first_upcoming_week"] = nxt
    cov = on.loc[up & (on["week"] == nxt) & on["group"].isin(["WR", "RB"]) & (on["active"] == 1)]
    res["first_upcoming_week_route_avg_filled"] = round(float(cov["route_avg"].notna().mean()), 3)
    print(f"season {season}, next week {nxt}; route_avg filled for {res['first_upcoming_week_route_avg_filled']:.1%} of active WR/RB", flush=True)
    for st, groups in ML.ADOPT.items():
        if st not in ("receptions", "rec_yds"):
            continue
        for g in groups:
            m = up & (a["week"] == nxt) & (a["group"] == g) & (a["active"] == 1) & (b[st + "_proj"] >= (0.5 if st == "receptions" else 5))
            d = (a.loc[m, st + "_proj"] - b.loc[m, st + "_proj"])
            r = {"n": int(m.sum()), "median_abs": round(float(d.abs().median()), 3), "p95_abs": round(float(d.abs().quantile(0.95)), 3),
                 "max_abs": round(float(d.abs().max()), 3), "mean": round(float(d.mean()), 3)}
            res[f"{g} {st}"] = r
            print(f"  {g} {st:10s} n {r['n']:3d}  median |change| {r['median_abs']:.2f}  95th {r['p95_abs']:.2f}  max {r['max_abs']:.2f}  mean {r['mean']:+.2f}", flush=True)
    for st in ("carries", "rush_yds"):
        same = np.allclose(a.loc[up, st + "_proj"].fillna(-1), b.loc[up, st + "_proj"].fillna(-1))
        res[f"RB {st} also moves"] = not same
    if args.rmse:
        import nfl_pff_routes_test as T
        print("walk-forward RMSE on the board's code, routes off -> on:", flush=True)
        cur = ML.questionable(ML.walk_forward(off, first=T.FIRST, feats=list(ML.FEATS)), first=C.FIRST_TEST_SEASON)
        new = ML.questionable(ML.walk_forward(on, first=T.FIRST), first=C.FIRST_TEST_SEASON)
        res["rmse"] = T.rmse(cur, new)
        res["rmse_better_on"] = f"{sum(v[p]['better'] for v in res['rmse'].values() for p in ('tune', 'hold'))} of 8"
        print("better on", res["rmse_better_on"], flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
