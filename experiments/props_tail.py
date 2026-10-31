"""The prop correction fitted on the whole ladder of lines, not just lines near the projection.

    python experiments/props_tail.py

Why. The ladder test (PROPS.md, "Alternate ladders against real prices") found the model's
low end too fat for running backs: on 2025 rushing-yard rungs priced -400 or shorter, 81.6%
cleared and the model said 73.2%. The same shows on 2019-2024 at synthetic low lines (RB
rushing yards 81% predicted, 85.5% happened; 88% and 93.6%). The final correction
(props.PropModel.iso, an isotonic map of P(over) per stat) was fitted only at lines within two
steps of the projection (props.test_lines), so it never saw a low line.

What. The same correction, fitted the same way (tuning seasons 2013-2018, each scored with
widths fitted on the other five) but at a ladder of lines from 20% to 170% of the projection
(LADDER_FRACS), per stat and position group. Compared with the current correction on:
  1. holdout 2019-2024 (widths fitted on 2013-2018, so nothing from the holdout), at the ladder lines and
     at the usual lines near the projection (it must not get worse there);
  2. the real 2025 DraftKings/FanDuel ladder rungs (props_ladders.priced_rungs), with the
     books' margin-free chance as the benchmark.
Nothing here is fitted on 2019 or later. Run by hand; writes experiments/out/props_tail.json.
If it holds, props_calibration.py fits the correction this way (WIDE_CORRECTION).
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
from nflmodel import propbackfill as B, propdist as D, props as P  # noqa: E402
import props_real_lines as R  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out"
TUNE = list(range(2013, 2019))
HOLD = list(range(2019, 2025))
STATS = ["rush_yds", "rec_yds", "pass_yds", "carries", "receptions", "pass_att", "completions"]


def preds(df: pd.DataFrame, model: P.PropModel, stat: str, lines: str) -> pd.DataFrame:
    out = []
    for g in P.SPECS[stat].groups:
        if (stat, g) not in model.counts and (stat, g) not in model.yards:
            continue
        d = df[P.eligible(df, stat, g)].reset_index(drop=True)
        if d.empty:
            continue
        mu = P.mean(d, stat)
        L = P.ladder_lines(mu, stat) if lines == "ladder" else P.test_lines(mu, stat)
        raw = np.concatenate([model.probs(d.iloc[i:i + 3000].reset_index(drop=True), stat, g, L[i:i + 3000], calibrate=False)[0]
                              for i in range(0, len(d), 3000)])          # in chunks: the yards cdf is (rows x plays x lines)
        y = P.actual(d, stat)
        k = L.shape[1]
        out.append(pd.DataFrame({"stat": stat, "group": g, "season": np.repeat(d["season"].to_numpy(), k),
                                 "game_id": np.repeat(d["game_id"].to_numpy(), k), "frac": (L / np.maximum(mu, 0.5)[:, None]).ravel(),
                                 "line": L.ravel(), "p_raw": raw.ravel(), "hit": (y[:, None] > L).astype(float).ravel()}))
    r = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return r[r["line"].notna() & r["p_raw"].notna()] if len(r) else r


def apply_iso(r: pd.DataFrame, iso: dict) -> np.ndarray:
    p = r["p_raw"].to_numpy(float).copy()
    for (st, g), x in r.groupby(["stat", "group"]):
        f = iso.get(f"{st}|{g}") or iso.get(st)
        if f is not None:
            p[x.index] = f(x["p_raw"].to_numpy(float))
    return p


def table(r: pd.DataFrame, p: np.ndarray) -> list[dict]:
    r = r.assign(p=p)
    rows = []
    for (st, g), x in r.groupby(["stat", "group"]):
        for lo, hi in ((0.6, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.0)):
            k = x[(x.p >= lo) & (x.p < hi)]
            if len(k) >= 200:
                rows.append({"stat": st, "group": g, "band": f"{lo:.2f}-{hi:.2f}", "n": int(len(k)),
                             "pred": round(float(k.p.mean()), 4), "actual": round(float(k.hit.mean()), 4)})
    return rows


def brier(r, p) -> dict:
    out = {}
    for st, x in r.assign(p=p).groupby("stat"):
        out[st] = round(float(((x.p - x.hit) ** 2).mean()), 5)
    out["all"] = round(float(((p - r["hit"].to_numpy()) ** 2).mean()), 5)
    return out


def main():
    from props_calibration import FAMILY, load_proj
    proj = load_proj(None)
    saved = P.PropModel.load()
    cur = {k: v for k, v in saved.iso.items()}

    print("tuning, leave one season out ...", flush=True)
    tune = proj[proj.season.isin(TUNE)]
    cache = R.C.RAW / "props_tail_tuning.pkl"
    if cache.exists():
        tp = pd.read_pickle(cache)
    else:
        parts = []
        for hold in TUNE:
            m = P.PropModel(FAMILY).fit(tune[tune.season != hold])
            te = tune[tune.season == hold]
            parts += [preds(te, m, st, "ladder") for st in STATS]
            print(" ", hold, flush=True)
        tp = pd.concat(parts, ignore_index=True)
        tp.to_pickle(cache)
    wide = {}
    for (st, g), x in tp.groupby(["stat", "group"]):
        if len(x) >= 2000:
            wide[f"{st}|{g}"] = D.Isotonic.fit(x["p_raw"], x["hit"])

    print("holdout 2019-2024, widths fitted on 2013-2018 ...", flush=True)
    hcache = R.C.RAW / "props_tail_holdout.pkl"
    if hcache.exists():
        hl, hn = pd.read_pickle(hcache)
    else:
        hl, hn = [], []
        m = P.PropModel(FAMILY).fit(tune)
        for s in HOLD:
            te = proj[proj.season == s]
            hl += [preds(te, m, st, "ladder") for st in STATS]
            hn += [preds(te, m, st, "test") for st in STATS]
            print(" ", s, flush=True)
        hl, hn = pd.concat(hl, ignore_index=True), pd.concat(hn, ignore_index=True)
        pd.to_pickle((hl, hn), hcache)
    rep = {"ladder_fracs": list(P.LADDER_FRACS),
           "holdout_ladder": {"brier_current": brier(hl, apply_iso(hl, cur)), "brier_wide": brier(hl, apply_iso(hl, wide)),
                              "table_current": table(hl, apply_iso(hl, cur)), "table_wide": table(hl, apply_iso(hl, wide))},
           "holdout_near_line": {"brier_current": brier(hn, apply_iso(hn, cur)), "brier_wide": brier(hn, apply_iso(hn, wide))}}

    print("2025 real ladder rungs ...", flush=True)
    import props_ladders as L
    pre = R.load_pregame()
    m25 = P.PropModel(FAMILY).fit(pre[(pre.season < 2025) & (pre.season >= 2013)])
    res = {}
    for name, iso in (("current", cur), ("wide", wide)):
        m25.iso = dict(iso)                                   # PropModel.probs reads stat|group first
        x = L.priced_rungs(pre, m25, B.load(2025, "flag"), 2025)
        sq = lambda p: (p - x["hit"]) ** 2
        d = sq(x["ours"]) - sq(x["fair"])
        hi = x[x.price <= -400]
        res[name] = {"n": int(len(x)), "brier_model": round(float(sq(x["ours"]).mean()), 5), "brier_books": round(float(sq(x["fair"]).mean()), 5),
                     "model_minus_books": round(float(d.mean()), 5), "se": round(L._se(d, x["game_id"]), 5),
                     "by_stat": {st: {"brier_model": round(float(((g.ours - g.hit) ** 2).mean()), 5),
                                      "brier_books": round(float(((g.fair - g.hit) ** 2).mean()), 5)} for st, g in x.groupby("stat")},
                     "short_rungs": {st: {"n": int(len(g)), "hit": round(float(g.hit.mean()), 4), "model": round(float(g.ours.mean()), 4),
                                          "implied": round(float(g.raw.mean()), 4)} for st, g in hi.groupby("stat")},
                     "bands": L.calibration(x)["bands"]["model"]}
    rep["real_2025"] = res
    OUT.mkdir(exist_ok=True)
    (OUT / "props_tail.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("holdout_near_line",)}, indent=1))
    print("holdout ladder brier", rep["holdout_ladder"]["brier_current"], rep["holdout_ladder"]["brier_wide"])
    for k in ("table_current", "table_wide"):
        t = pd.DataFrame(rep["holdout_ladder"][k]); t["gap"] = t.actual - t.pred
        print(k); print(t[t.stat.isin(["rush_yds", "carries", "rec_yds", "pass_yds"])].to_string(index=False))
    for name in res:
        r = res[name]
        print(name, {k: r[k] for k in ("brier_model", "brier_books", "model_minus_books", "se")}, r["short_rungs"])


if __name__ == "__main__":
    main()
