"""The ladder-wide correction, fitted per position on recent seasons.

    python experiments/props_recent_iso.py

Why. After fitting the widths on 2019 on (props_window.py) the 2025 running-back rushing rungs
at -400 or shorter still cleared 81.6% against the model's 76.1%. The projections are not
biased (backs' actual yards land within a yard of the projection on average every season), so
the spread around the projection is too wide for backs. props_tail.py fitted a ladder-wide
correction, but on 2013-2018, the seasons before the 2019 change, where the model was already
calibrated; it could not learn the new shape.

What. Widths fitted on recent seasons; the correction (isotonic, per stat and position group,
at the ladder of lines props.ladder_lines) fitted on 2019-2023 with each season scored by
widths fitted on the other recent seasons. Scored where nothing was fitted:
  1. 2024 (widths fitted on 2019-2023): Brier at the usual lines and on the ladder, and the
     75%+ band, against the saved correction;
  2. the real 2025 DraftKings/FanDuel ladder rungs (widths fitted on 2019-2024, pregame
     projections): Brier against the books, the short rungs by stat, and the bets.
Kept only if it is no worse at the usual lines on any stat and better on the ladder and on
the 2025 rungs. Writes experiments/out/props_recent_iso.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import propbackfill as B, propdist as D, props as P  # noqa: E402
import props_real_lines as R  # noqa: E402
import props_tail as T  # noqa: E402
import props_window as W  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out"
FIT = list(range(2019, 2024))
STATS = W.STATS


def fit_iso(proj: pd.DataFrame, family) -> dict:
    cache = R.C.RAW / "props_recent_iso_preds.pkl"
    if cache.exists():
        tp = pd.read_pickle(cache)
    else:
        parts = []
        rec = proj[proj.season.isin(FIT)]
        for hold in FIT:
            m = P.PropModel(family).fit(rec[rec.season != hold])
            te = rec[rec.season == hold]
            parts += [T.preds(te, m, st, "ladder") for st in STATS]
            print(" ", hold, flush=True)
        tp = pd.concat(parts, ignore_index=True)
        tp.to_pickle(cache)
    iso = {}
    for (st, g), x in tp.groupby(["stat", "group"]):
        if len(x) >= 2000:
            iso[f"{st}|{g}"] = D.Isotonic.fit(x["p_raw"], x["hit"])
    return iso


def main():
    from props_calibration import FAMILY, load_proj
    import props_ladders as L
    proj = load_proj(None)
    saved = P.PropModel.load()
    print("correction on 2019-2023, leave one season out ...", flush=True)
    iso_new = fit_iso(proj, FAMILY)
    rep = {"fit_seasons": FIT}

    print("2024 ...", flush=True)
    m = P.PropModel(FAMILY).fit(proj[proj.season.isin(FIT)])
    te = proj[proj.season == 2024]
    near = pd.concat([T.preds(te, m, st, "test") for st in STATS], ignore_index=True)
    lad = pd.concat([T.preds(te, m, st, "ladder") for st in STATS], ignore_index=True)
    rep["season_2024"] = {}
    for name, iso in (("saved", dict(saved.iso)), ("recent", iso_new)):
        rep["season_2024"][name] = {"near": W.by_group(near.assign(p=T.apply_iso(near, iso))),
                                    "ladder": W.by_group(lad.assign(p=T.apply_iso(lad, iso)))}

    print("2025 real rungs ...", flush=True)
    pre = R.load_pregame()
    m25 = P.PropModel(FAMILY).fit(pre[(pre.season >= 2019) & (pre.season < 2025)])
    flag, close = B.load(2025, "flag"), B.load(2025, "close")
    rep["real_2025"] = {}
    for name, iso in (("saved", dict(saved.iso)), ("recent", iso_new)):
        m25.iso = dict(iso)
        x = L.priced_rungs(pre, m25, flag, 2025)
        d = (x.ours - x.hit) ** 2 - (x.fair - x.hit) ** 2
        hi = x[x.price <= -400]
        cl = L.priced_rungs(pre, m25, close, 2025)
        rep["real_2025"][name] = {
            "brier_model": round(float(((x.ours - x.hit) ** 2).mean()), 5), "brier_books": round(float(((x.fair - x.hit) ** 2).mean()), 5),
            "model_minus_books": round(float(d.mean()), 5), "se": round(L._se(d, x["game_id"]), 5),
            "short_rungs": {st: {"n": int(len(g)), "hit": round(float(g.hit.mean()), 4), "model": round(float(g.ours.mean()), 4)} for st, g in hi.groupby("stat")},
            "bands": L.calibration(x)["bands"]["model"],
            "bets_ev2_in_band": L.grade(L.with_close(L.model_bets(x, 0.02, True), cl)),
            "bets_ev2_any": L.grade(L.with_close(L.model_bets(x, 0.02, False), cl))}
        r = rep["real_2025"][name]
        print(" ", name, {k: r[k] for k in ("brier_model", "brier_books", "model_minus_books", "se")}, r["short_rungs"], flush=True)
        print("    bets in band:", r["bets_ev2_in_band"])
    OUT.mkdir(exist_ok=True)
    (OUT / "props_recent_iso.json").write_text(json.dumps(rep, indent=1, default=float))
    a, b = rep["season_2024"]["saved"], rep["season_2024"]["recent"]
    print("\n2024, saved -> recent correction: near-line Brier | ladder Brier | ladder 75%+ pred/hit")
    for k in a["near"]:
        if k in b["near"]:
            print(f"  {k:18s} {a['near'][k]['brier']:.5f} -> {b['near'][k]['brier']:.5f} | {a['ladder'][k]['brier']:.5f} -> {b['ladder'][k]['brier']:.5f} | "
                  f"{a['ladder'][k]['high_pred']}/{a['ladder'][k]['high_hit']} -> {b['ladder'][k]['high_pred']}/{b['ladder'][k]['high_hit']}")


if __name__ == "__main__":
    main()
