"""College: a separate early-season weight in the margin model (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_early_weight_test.py

T2 of the Project doc claude/plan-batch-tests-2026-11-20.md (written before the run). Committed
before its first run.

Today's market-residual fit (cfbmodel.model.market_residual) uses one set of weights all season.
  A  two fits, one on weeks 1-4 of earlier seasons and one on weeks 5+, each used for its part
  B  today's fit, its move away from the line multiplied in weeks 1-4 by k. For season S, k is the
     least-squares slope (through the origin) of the actual residual on the predicted residual in
     the weeks 1-4 games of earlier walk-forward seasons; k = 1 for the first season; clipped 0-1.5.

Measures and rule: the twelve of experiments/cfb_outside_ratings_test.py (average miss, RMS miss,
side of the closing line; board frame and bet-time frame; tuning 2017-19 and holdout 2021-25), same
games. Chosen on tuning alone (better on all six, then the lowest tuning average miss). Kept only
if better on all twelve and the holdout average miss by at least 1 paired SE. Otherwise a trade-off
or nothing.

Writes experiments/out/cfb_early_weight_test.json.
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

import cfb_outside_ratings_test as T  # noqa: E402  (frames, measures, the rule's helpers)

OUT = HERE / "out" / "cfb_early_weight_test.json"
EARLY = 4
TUNE, HOLD = T.TUNE, T.HOLD


def version_a(df: pd.DataFrame) -> pd.DataFrame:
    from cfbmodel import model
    wf, _ = model.walk_forward(df)
    full = df.copy()
    MF = list(model.MARGIN_FEATS)
    for col in MF:
        if col not in full:
            full[col] = 0.0
    proj = wf["proj_margin"].copy()
    for s in sorted(wf.season.unique()):
        train = full[(full.season < s) & full.played & full["spread_line"].notna()]
        for part, sel_tr, sel_te in (("early", train.week <= EARLY, wf.week <= EARLY), ("late", train.week > EARLY, wf.week > EARLY)):
            st = model.market_residual(train[sel_tr], MF, s)
            hl = (wf.season == s) & wf["spread_line"].notna() & sel_te
            if st is not None and hl.any():
                proj.loc[hl] = wf.loc[hl, "spread_line"] + st.predict(full.loc[wf.index[hl], MF + ["spread_line"]].to_numpy(float))
    return wf.assign(cand=proj)


def version_b(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from cfbmodel import model
    wf, _ = model.walk_forward(df)
    pred = wf["proj_margin"] - wf["spread_line"]
    act = wf["result"] - wf["spread_line"]
    proj = wf["proj_margin"].copy()
    ks = {}
    for s in sorted(wf.season.unique()):
        past = (wf.season < s) & wf.played & (wf.week <= EARLY) & pred.notna() & act.notna()
        k = 1.0
        if past.sum() >= 100 and float(pred[past] @ pred[past]) > 0:
            k = float(np.clip((pred[past] @ act[past]) / (pred[past] @ pred[past]), 0.0, 1.5))
        ks[int(s)] = round(k, 3)
        hl = (wf.season == s) & (wf.week <= EARLY) & wf["spread_line"].notna()
        proj.loc[hl] = wf.loc[hl, "spread_line"] + k * pred[hl]
    return wf.assign(cand=proj), ks


def main() -> None:
    close, early = T.frames()
    res = {"versions": {}}
    frames = {}
    wa_c, wa_e = version_a(close), version_a(early)
    (wb_c, ks), (wb_e, _) = version_b(close), version_b(early)
    for v, (wc, we) in {"A": (wa_c, wa_e), "B": (wb_c, wb_e)}.items():
        gc, ge = T.scored(wc), T.scored(we)
        mt, mte = T.measures(T.period(gc, *TUNE)), T.measures(T.period(ge, *TUNE))
        res["versions"][v] = {"tune": mt, "tune_bet": mte, "tune_better": T.better(mt, mte)}
        frames[v] = (gc, ge)
    res["versions"]["B"]["k_by_season"] = ks
    passing = [v for v in res["versions"] if all(res["versions"][v]["tune_better"].values())]
    res["tune_passing"] = passing
    if not passing:
        res["verdict"] = "No version is better than the board on all six tuning measures. Nothing is carried to the holdout; nothing is kept."
    else:
        chosen = min(passing, key=lambda v: res["versions"][v]["tune"]["mae_cand"])
        gc, ge = frames[chosen]
        mh, mhe = T.measures(T.period(gc, *HOLD)), T.measures(T.period(ge, *HOLD))
        bh = T.better(mh, mhe)
        se_ok = mh["d_mae"] < 0 and abs(mh["d_mae"]) >= mh["se_d_mae"]
        n_better = sum(res["versions"][chosen]["tune_better"].values()) + sum(bh.values())
        res.update({"chosen": chosen, "holdout": {"close": mh, "bet": mhe, "better": bh, "one_se": bool(se_ok)},
                    "2026": {"close": T.measures(T.period(gc, 2026, 2026)), "bet": T.measures(T.period(ge, 2026, 2026))}})
        res["verdict"] = (f"Version {chosen}: better on all 12 and the holdout average miss by at least 1 SE. Passes the keep rule (provisional)."
                          if all(bh.values()) and se_ok else f"Version {chosen}: a trade-off ({n_better} of 12 better; holdout 1 SE: {bool(se_ok)}). Ryan's call.")
    # reported either way: the early weeks alone, each version, each period (the board's frame)
    rep = {}
    for v, (gc, _) in frames.items():
        e = gc[gc.week <= EARLY]
        rep[v] = {"tune": T.measures(T.period(e, *TUNE)), "holdout": T.measures(T.period(e, *HOLD)), "2026": T.measures(T.period(e, 2026, 2026))}
    res["early_weeks_only_reported"] = rep
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for v, r in res["versions"].items():
        t, tb = r["tune"], r["tune_bet"]
        print(f"{v} tuning: miss {t['mae_board']:.3f} -> {t['mae_cand']:.3f}, rms {t['rmse_board']:.3f} -> {t['rmse_cand']:.3f}, "
              f"side {100 * t['win_board']:.1f}% -> {100 * t['win_cand']:.1f}%; bet-time miss {tb['mae_board']:.3f} -> {tb['mae_cand']:.3f}; "
              f"better on {sum(r['tune_better'].values())} of 6")
    print("k by season (B):", ks)
    print(res["verdict"])


if __name__ == "__main__":
    main()
