"""Pre-set test: the shut-out chance, judged only on players the college board lists (Nov 18, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_shutout_board_test.py

Rule written before running (Project doc claude/plan-cfb-shutout-board-test-2026-11-18.md). The
yardstick was chosen after seeing experiments/cfb_shutout_pricing_test.py's results; anything kept is
provisional and judged again on 2026 weeks 12+ and in January.

over = (1 - p0) x the model's over. p0 (P1, P2 as in cfb_shutout_pricing_test.py) fitted on earlier
seasons on expected players only. Scored: bet-time projections, active == 1, took the field, TE over
15 projected yards, WR over 20.
Measures per position (4): Brier at his own Typical line, tuning 2018-19 and holdout 2021-25; Brier at
the DK/FD main receiving-yards lines, 2024 and 2025. Version chosen on tuning (must be better); kept
only if better on all 4, the holdout by >= 1 SE. Scores only, never PFF rows.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_pff_role_test as T  # noqa: E402
import cfb_shutout_pricing_test as Q  # noqa: E402
import cfb_shutout_test as X  # noqa: E402

OUT = HERE / "out" / "cfb_shutout_board_test.json"
TUNE, HOLD = [2018, 2019], [2021, 2022, 2023, 2024, 2025]
GRID = np.linspace(0.05, 1.7, 67)


def curves(cur: pd.DataFrame, g: str) -> pd.DataFrame:
    """Per scored row: the current Typical line, and the model's chance of the over at 60%, 100% and
    150% of it; plus the whole decision curve (kept in memory only) to move Typical."""
    model = P.PropModel.load(X.PARAMS)
    idx = cur.index[(cur["group"] == g) & (cur["rec_yds_proj"] > Q.MIN[g]) & (cur["took"] == 1) & (cur["active"] == 1)]
    rows = []
    for i in range(0, len(idx), 5000):
        ix = idx[i:i + 5000]
        d = cur.loc[ix].reset_index(drop=True)
        d["tgt_share"] = d.get("rec_share", 0.0)
        mu = P.mean(d, "rec_yds")
        L = np.floor(mu[:, None] * GRID[None, :]) + 0.5
        over, _, under = model.probs(d, "rec_yds", g, L)
        dec = over / np.maximum(over + under, 1e-9)
        for k in range(len(d)):
            ty = float(np.interp(0.5, dec[k][::-1], L[k][::-1]))
            rows.append((ix[k], ty, float(np.interp(0.6 * ty, L[k], dec[k])), float(np.interp(1.5 * ty, L[k], dec[k])), L[k], dec[k]))
    return pd.DataFrame(rows, columns=["i", "ty", "p60", "p150", "L", "dec"]).set_index("i")


def score(cur, C, p0, seasons, at="ty") -> dict:
    k = cur.loc[C.index, "season"].isin(seasons).to_numpy()
    c = C[k]
    y = cur.loc[c.index, "rec_yds"].to_numpy(float)
    line = c["ty"].to_numpy() * {"ty": 1.0, "p60": 0.6, "p150": 1.5}[at]
    pc = np.full(len(c), 0.5) if at == "ty" else c[at].to_numpy()
    q = p0[cur.index.get_indexer(c.index)]
    pn = pc * (1 - q)
    hit = (y >= line).astype(float)
    d, se = R.clustered_diff(pn, pc, hit, cur.loc[c.index, "game_id"].to_numpy())
    out = {"n": int(len(c)), "happened": round(float(hit.mean()), 4), "model_now": round(float(pc.mean()), 4), "candidate": round(float(pn.mean()), 4),
           "diff": round(d, 6), "se": round(se, 6), "z": round(d / se, 2) if se else 0.0}
    if at == "ty":
        new_ty = np.array([float(np.interp(0.5, (dd * (1 - qq))[::-1], LL[::-1])) if (dd[0] * (1 - qq)) >= 0.5 else float(LL[0]) for LL, dd, qq in zip(c["L"], c["dec"], q)])
        out["shut_out_rate"] = round(float(cur.loc[c.index, "shut"].mean()), 4)
        out["typical_now_avg"], out["typical_new_avg"] = round(float(c["ty"].mean()), 1), round(float(new_ty.mean()), 1)
        out["reach_new_typical"] = round(float((y >= new_ty).mean()), 4)
    return out


def main():
    t0 = time.time()
    t, pinfo, games, _, _ = X.base()
    pw = T.pff_player_weeks(list(range(2017, 2027)))
    m, _ = T.map_onto_board(pw, t, pinfo)
    m = m.dropna(subset=["board_id"])
    m = m[m["snap_counts_total"] > 0].assign(player_id=lambda x: x["board_id"].astype(np.int64))
    S = m.drop_duplicates(X.KEY)[X.KEY]
    cur = Q.prepare(X.walk("cur", t, True), S)
    cur = cur[cur["active"] == 1].reset_index(drop=True)          # the board's players
    p0 = {v: Q.fit_p0(cur, v)[0] for v in ("P1", "P2")}
    print("prepared", round(time.time() - t0), flush=True)
    res = {"plan": "claude/plan-cfb-shutout-board-test-2026-11-18.md", "positions": {}}
    for g in ("TE", "WR"):
        C = curves(cur, g)
        r = {"tuning": {v: score(cur, C, p0[v], TUNE) for v in ("P1", "P2")}}
        ok = [v for v in ("P1", "P2") if r["tuning"][v]["diff"] < 0]
        if not ok:
            r["verdict"] = "Neither version better on tuning: nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["diff"])
            other = "P2" if pick == "P1" else "P1"
            r["chosen"] = pick
            r["holdout"] = score(cur, C, p0[pick], HOLD)
            r["lines"] = {str(s): Q.at_lines(cur, p0[pick], pinfo, games, g, s) for s in (2024, 2025)}
            four = [r["tuning"][pick], r["holdout"]] + list(r["lines"].values())
            better = all(x.get("diff", 1) < 0 for x in four)
            r["verdict"] = "KEPT (provisional)" if better and r["holdout"]["z"] <= -1 else ("Trade-off: better on all 4, the holdout gain under 1 SE" if better else "Trade-off: mixed across the 4 measures")
            r["reported"] = {"other_version": {"version": other, "holdout": score(cur, C, p0[other], HOLD), "lines": {str(s): Q.at_lines(cur, p0[other], pinfo, games, g, s) for s in (2024, 2025)}},
                             "holdout at 60% of Typical": score(cur, C, p0[pick], HOLD, "p60"), "holdout at 150% of Typical": score(cur, C, p0[pick], HOLD, "p150"),
                             "2026 weeks 1-11 (seen)": score(cur, C, p0[pick], [2026])}
            k = cur.loc[C.index, "season"].isin(HOLD).to_numpy()
            d = cur.loc[C.index[k]].assign(p0=p0[pick][cur.index.get_indexer(C.index[k])])
            r["reported"]["by_projected_receptions_holdout"] = {str(b): {"n": int(len(x)), "actual": round(float(x["shut"].mean()), 4), "predicted": round(float(x["p0"].mean()), 4)}
                                                                for b, x in d.groupby(pd.cut(d["receptions_proj"], [0, 1.5, 2, 3, 4, 99]), observed=True)}
        res["positions"][g] = r
        print(g, json.dumps(r, default=float), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
