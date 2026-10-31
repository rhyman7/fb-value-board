"""Which seasons the prop distributions should be fitted on: all since 2013, or recent ones.

    python experiments/props_window.py

Why. experiments/props_tail.py found a break in 2019: for running backs, lines the model put
at 75-95% cleared 82-85% of the time in every season 2013-2018 (calibrated) and 87-88% in every
season 2019-2024. The distributions are fitted on every season since 2013, so they carry the
older, more volatile backfield into today's prices; the 2025 ladder rungs show it (RB rushing
rungs at -400 or shorter cleared 81.6%, the model said 73.2%).

What. Two fits of the same model, the same correction (the saved one), scored where neither
has seen the data:
  1. 2024 (game-day projections): fitted on 2013-2023 against 2019-2023;
     Brier at the usual lines near the projection and at a ladder of lines, by stat and group,
     and calibration at the high bands;
  2. the real 2025 DraftKings/FanDuel ladder rungs (pregame projections, as props_ladders.py):
     fitted on 2013-2024 against 2019-2024, Brier against the books and the short rungs by stat.
The window is kept only if it is no worse on every stat at the usual lines and better on the
ladder. Writes experiments/out/props_window.json.
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
from nflmodel import propbackfill as B, props as P  # noqa: E402
import props_real_lines as R  # noqa: E402
import props_tail as T  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out"
RECENT = 2019
STATS = ["rush_yds", "rec_yds", "pass_yds", "carries", "receptions", "pass_att", "completions", "pass_td", "anytime_td"]


def scored(r: pd.DataFrame, model) -> pd.DataFrame:
    return r.assign(p=T.apply_iso(r, model.iso))


def by_group(r: pd.DataFrame) -> dict:
    out = {}
    for (st, g), x in r.groupby(["stat", "group"]):
        d = (x.p - x.hit) ** 2
        hi = x[x.p >= 0.75]
        out[f"{st}|{g}"] = {"n": int(len(x)), "brier": round(float(d.mean()), 5),
                            "high_pred": round(float(hi.p.mean()), 4) if len(hi) else None,
                            "high_hit": round(float(hi.hit.mean()), 4) if len(hi) else None}
    return out


def main():
    from props_calibration import FAMILY, load_proj
    proj = load_proj(None)
    saved = P.PropModel.load()
    rep = {"recent_from": RECENT}

    print("2024: fitted on 2013-2023 vs 2019-2023 ...", flush=True)
    te = proj[proj.season == 2024]
    fits = {"all": P.PropModel(FAMILY).fit(proj[(proj.season >= 2013) & (proj.season < 2024)]),
            "recent": P.PropModel(FAMILY).fit(proj[(proj.season >= RECENT) & (proj.season < 2024)])}
    for k in fits:
        fits[k].iso = dict(saved.iso)
    res = {}
    for name, m in fits.items():
        near = scored(pd.concat([T.preds(te, m, st, "test") for st in STATS], ignore_index=True), m)
        lad = scored(pd.concat([T.preds(te, m, st, "ladder") for st in STATS], ignore_index=True), m)
        res[name] = {"near": by_group(near), "ladder": by_group(lad),
                     "near_all": round(float(((near.p - near.hit) ** 2).mean()), 5),
                     "ladder_all": round(float(((lad.p - lad.hit) ** 2).mean()), 5)}
        print(" ", name, res[name]["near_all"], res[name]["ladder_all"], flush=True)
    rep["season_2024"] = res

    print("2025 real rungs: fitted on 2013-2024 vs 2019-2024 ...", flush=True)
    import props_ladders as L
    pre = R.load_pregame()
    snaps = B.load(2025, "flag")
    real = {}
    for name, lo in (("all", 2013), ("recent", RECENT)):
        m = P.PropModel(FAMILY).fit(pre[(pre.season >= lo) & (pre.season < 2025)])
        m.iso = dict(saved.iso)
        x = L.priced_rungs(pre, m, snaps, 2025)
        d = (x.ours - x.hit) ** 2 - (x.fair - x.hit) ** 2
        hi = x[x.price <= -400]
        real[name] = {"n": int(len(x)), "brier_model": round(float(((x.ours - x.hit) ** 2).mean()), 5),
                      "brier_books": round(float(((x.fair - x.hit) ** 2).mean()), 5),
                      "model_minus_books": round(float(d.mean()), 5), "se": round(L._se(d, x["game_id"]), 5),
                      "by_stat": {st: round(float(((g.ours - g.hit) ** 2).mean()), 5) for st, g in x.groupby("stat")},
                      "short_rungs": {st: {"n": int(len(g)), "hit": round(float(g.hit.mean()), 4), "model": round(float(g.ours.mean()), 4)}
                                      for st, g in hi.groupby("stat")},
                      "bands": L.calibration(x)["bands"]["model"],
                      "bets_ev2_in_band": L.grade(L.with_close(L.model_bets(x, 0.02, True), L.priced_rungs(R.load_pregame(), m, B.load(2025, "close"), 2025)))}
        print(" ", name, {k: real[name][k] for k in ("brier_model", "brier_books", "model_minus_books", "se")}, real[name]["short_rungs"], flush=True)
    rep["real_2025"] = real
    OUT.mkdir(exist_ok=True)
    (OUT / "props_window.json").write_text(json.dumps(rep, indent=1, default=float))
    a, r = res["all"]["near"], res["recent"]["near"]
    print("\n2024 near-line Brier by stat|group (all -> recent):")
    for k in a:
        if k in r:
            print(f"  {k:18s} {a[k]['brier']:.5f} -> {r[k]['brier']:.5f}  ({r[k]['brier'] - a[k]['brier']:+.5f})")
    a, r = res["all"]["ladder"], res["recent"]["ladder"]
    print("2024 ladder Brier and the 75%+ band (pred/hit), all -> recent:")
    for k in a:
        if k in r:
            print(f"  {k:18s} {a[k]['brier']:.5f} -> {r[k]['brier']:.5f}   {a[k]['high_pred']}/{a[k]['high_hit']} -> {r[k]['high_pred']}/{r[k]['high_hit']}")


if __name__ == "__main__":
    main()
