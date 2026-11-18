"""Pre-set test: a shut-out chance in the college receiving prices and Typical (Nov 18, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_shutout_pricing_test.py

Rule written before running (Project doc claude/plan-cfb-shutout-pricing-test-2026-11-18.md). Follow-up
to experiments/cfb_shutout_test.py: the projection stays "yards if he records a stat"; the odds become
    over = (1 - p0) x the model's over,  under = the rest
p0 = chance a player who takes the field (any PFF offensive snap, or a box-score line) records no
attempt, carry or catch. Per position (TE, WR), walk-forward on earlier seasons from 2017:
  P1: logistic on log(projected receptions).
  P2: P1 plus PFF route share, snap share and his own earlier shut-out rate (shrunk, weight 5 games).
Measures per position (8): Brier at his own Typical line (current model 50% by construction), every
projected player who took the field at the Players page minimums, game-day and bet-time, tuning
2018-19 and holdout 2021-25; Brier at the DK/FD main receiving-yards lines, 2024 and 2025, game-day
and bet-time. Version chosen on tuning; kept only if better on all 8, each holdout gain >= 1 SE.
Writes scores only, never PFF rows.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402
import cfb_shutout_test as X  # noqa: E402

OUT = HERE / "out" / "cfb_shutout_pricing_test.json"
TUNE, HOLD = [2018, 2019], [2021, 2022, 2023, 2024, 2025]
KEY = X.KEY
MIN = {"TE": 15.0, "WR": 20.0, "RB": 10.0}


def prepare(cur: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    cur = cur.merge(S[KEY].assign(anysnap=1), on=KEY, how="left")
    cur["anysnap"] = cur["anysnap"].fillna(0).astype(int)
    cur["took"] = (cur["played"].astype(bool) & ((cur["appeared"] == 1) | (cur["anysnap"] == 1))).astype(int)
    cur["shut"] = ((cur["took"] == 1) & (cur["appeared"] == 0)).astype(int)
    cur = cur.sort_values(["player_id", "t"]).reset_index(drop=True)
    g = cur.groupby("player_id")
    cur["n_prev"] = g["took"].cumsum() - cur["took"]
    cur["s_prev"] = g["shut"].cumsum() - cur["shut"]
    for c in ("route_share", "snap_share"):
        cur[c] = cur[c].fillna(0.0) if c in cur else 0.0
    return cur


def fit_p0(cur: pd.DataFrame, version: str) -> tuple[np.ndarray, dict]:
    p0, info = np.zeros(len(cur)), {}
    lx = np.log(cur["receptions_proj"].clip(lower=0.05).to_numpy(float))
    ok = ((cur["took"] == 1) & (cur["receptions_proj"] > 0)).to_numpy()
    y, sea = cur["shut"].to_numpy(), cur["season"].to_numpy()
    for g in ("TE", "WR", "RB"):
        gm = (cur["group"] == g).to_numpy()
        for s in sorted(set(sea)):
            tr = ok & gm & (sea < s) & (sea >= 2017)
            if tr.sum() < 500 or y[tr].sum() < 20:
                continue
            m = float(y[tr].mean())
            own = (cur["s_prev"].to_numpy() + 5 * m) / (cur["n_prev"].to_numpy() + 5)
            Xall = lx[:, None] if version == "P1" else np.column_stack([lx, cur["route_share"].to_numpy(float), cur["snap_share"].to_numpy(float), np.log(own / (1 - own))])
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=500)).fit(Xall[tr], y[tr])
            te = gm & (sea == s)
            p0[te] = lr.predict_proba(Xall[te])[:, 1]
            info[f"{g} {s}"] = {"train": int(tr.sum()), "train_rate": round(m, 4)}
    return p0, info


def typical(cur: pd.DataFrame, g: str) -> pd.Series:
    """The current model's 50/50 receiving-yards line for each row of group g over the minimum."""
    model = P.PropModel.load(X.PARAMS)
    idx = cur.index[(cur["group"] == g) & (cur["rec_yds_proj"] > MIN[g]) & (cur["took"] == 1)]
    out = pd.Series(np.nan, index=cur.index)
    for i in range(0, len(idx), 5000):
        ix = idx[i:i + 5000]
        d = cur.loc[ix].reset_index(drop=True)
        d["tgt_share"] = d.get("rec_share", 0.0)
        mu = P.mean(d, "rec_yds")
        L = np.floor(mu[:, None] * np.linspace(0.05, 1.7, 67)[None, :]) + 0.5
        over, _, under = model.probs(d, "rec_yds", g, L)
        dec = over / np.maximum(over + under, 1e-9)
        out.loc[ix] = [float(np.interp(0.5, dec[k][::-1], L[k][::-1])) for k in range(len(d))]
    return out


def at_typical(cur, ty, p0, g, seasons) -> dict:
    k = (ty.notna() & cur["season"].isin(seasons)).to_numpy()
    hit = (cur["rec_yds"].to_numpy(float)[k] >= ty.to_numpy()[k]).astype(float)
    pn = 0.5 * (1 - p0[k])
    d, se = R.clustered_diff(pn, np.full(k.sum(), 0.5), hit, cur["game_id"].to_numpy()[k])
    return {"n": int(k.sum()), "reach_typical": round(float(hit.mean()), 4), "candidate_says": round(float(pn.mean()), 4),
            "shut_out_rate": round(float(cur["shut"].to_numpy()[k].mean()), 4), "diff": round(d, 6), "se": round(se, 6), "z": round(d / se, 2) if se else 0.0}


def at_lines(cur, p0, pinfo, games, g, season) -> dict:
    R.set_season(season)
    s = R.snapshots()
    s = s.merge(R.match_events(s, games), left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    main = s[~s["market"].str.endswith("_alternate") & (s["stat"] == "rec_yds")]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    fr = R.player_frame(cur.assign(p0=p0)[cur["took"] == 1], pinfo)
    x = R.model_chances(w.merge(fr[fr["group"] == g], on=["game_id", "nname"], how="inner"), P.PropModel.load(X.PARAMS))
    if x.empty:
        return {"props": 0}
    x = x[x["y"] != x["line"]]
    hit = (x["y"] > x["line"]).astype(float).to_numpy()
    pc = (x["our_over"] / (x["our_over"] + x["our_under"])).to_numpy()
    pn = pc * (1 - x["p0"].to_numpy())
    d, se = R.clustered_diff(pn, pc, hit, x["game_id"].to_numpy())
    return {"props": int(len(x)), "shut_outs_with_a_line": int((x["appeared"] == 0).sum()), "avg_p0": round(float(x["p0"].mean()), 4),
            "now": round(D.brier(pc, hit), 5), "candidate": round(D.brier(pn, hit), 5), "diff": round(d, 6), "se": round(se, 6),
            "books": round(D.brier(x["fair"].to_numpy(), hit), 5), "thin": bool(len(x) < 200)}


def main():
    t0 = time.time()
    t, pinfo, games, _, _ = X.base()
    S, _ = X.snaps(t, pinfo)
    F = {}
    for bt in (False, True):
        cur = prepare(X.walk("cur", t, bt), S)
        F["bt" if bt else "gd"] = {"cur": cur, "P1": fit_p0(cur, "P1")[0], "P2": fit_p0(cur, "P2")[0], "ty": {g: typical(cur, g) for g in ("TE", "WR")}}
        print("prepared", "bt" if bt else "gd", round(time.time() - t0), flush=True)
    res = {"plan": "claude/plan-cfb-shutout-pricing-test-2026-11-18.md", "positions": {}}
    for g in ("TE", "WR"):
        r = {"tuning": {v: {k: at_typical(F[k]["cur"], F[k]["ty"][g], F[k][v], g, TUNE) for k in ("gd", "bt")} for v in ("P1", "P2")}}
        ok = [v for v in ("P1", "P2") if all(r["tuning"][v][k]["diff"] < 0 for k in ("gd", "bt"))]
        r["better_on_both_tuning"] = ok
        if not ok:
            r["verdict"] = "Neither version better on both tuning measures: nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["gd"]["diff"])
            r["chosen"] = pick
            r["holdout"] = {k: at_typical(F[k]["cur"], F[k]["ty"][g], F[k][pick], g, HOLD) for k in ("gd", "bt")}
            r["lines"] = {f"{s} {k}": at_lines(F[k]["cur"], F[k][pick], pinfo, games, g, s) for s in (2024, 2025) for k in ("gd", "bt")}
            other = "P2" if pick == "P1" else "P1"
            r["other_version_reported"] = {"version": other, "holdout": {k: at_typical(F[k]["cur"], F[k]["ty"][g], F[k][other], g, HOLD) for k in ("gd", "bt")},
                                           "lines": {f"{s} {k}": at_lines(F[k]["cur"], F[k][other], pinfo, games, g, s) for s in (2024, 2025) for k in ("gd", "bt")}}
            ln = [x for x in r["lines"].values() if x.get("props")]
            better = all(x["diff"] < 0 for x in list(r["tuning"][pick].values()) + list(r["holdout"].values()) + ln) and len(ln) == 4
            one_se = all(x["z"] <= -1 for x in r["holdout"].values())
            r["verdict"] = "KEPT (provisional)" if better and one_se else ("Trade-off: better on all 8, a holdout gain under 1 SE" if better else "Trade-off: mixed across the 8 measures")
            cur, p0 = F["gd"]["cur"], F["gd"][pick]
            k = (F["gd"]["ty"][g].notna() & cur["season"].isin(HOLD)).to_numpy()
            d = cur[k].assign(p0=p0[k])
            r["by_projected_receptions_holdout"] = {str(b): {"n": int(len(x)), "actual": round(float(x["shut"].mean()), 4), "predicted": round(float(x["p0"].mean()), 4)}
                                                    for b, x in d.groupby(pd.cut(d["receptions_proj"], [0, 1.5, 2, 3, 4, 99]), observed=True)}
        res["positions"][g] = r
        print(g, json.dumps(r, default=float), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
