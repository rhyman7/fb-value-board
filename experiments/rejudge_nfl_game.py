"""Re-judge the NFL game-model feature candidates under the keep rule, on today's model.

    python experiments/rejudge_nfl_game.py

Nov 4, 2026. The keep rule (CONVENTIONS.md, "Model tests"): kept if better on every measure the
test reports and worse on none, however small. The original feature_test.py added total-model
candidates to the model from before Oct 2026 (no wind, no s_snaps) and used a frame without
injuries. Here every candidate is added to the board's current model: the frame the Update board
Action builds (PRODUCTION_METRICS, injuries), MARGIN_FEATS / TOTAL_FEATS as they are now.
Measures: average miss on the tuning seasons (before HOLDOUT_FROM) and the holdout. Standard
errors are clustered by game (paired differences); reported, not used to decide.

Writes experiments/out/rejudge_nfl_game.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model  # noqa: E402
from nflmodel.features_ext import EXTRA_METRICS, PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "rejudge_nfl_game.json"
MARGIN = ["d_sack", "d_expl", "d_ed", "d_st", "d_fum", "d_cpoe", "d_pace", "d_snaps", "travel", "tz_east",
          "west_early", "hfa_div"]
TOTAL = ["s_pace", "s_expl", "s_sack", "s_ed", "s_fum", "wind_hi", "cold"]


def run(frame, mf, tf):
    out, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    o = out[out.played & out.spread_line.notna()].set_index("game_id")
    return o


def diff(base, new, col, actual):
    e0 = (base[col] - base[actual]).abs()
    e1 = (new.loc[base.index, col] - base[actual]).abs()
    res = {}
    for name, m in (("tune", base.season < C.HOLDOUT_FROM), ("hold", base.season >= C.HOLDOUT_FROM)):
        d = (e1 - e0)[m]
        res[name] = {"n": int(m.sum()), "change": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    res["verdict"] = ("FLIPS" if all(res[k]["change"] < 0 for k in ("tune", "hold"))
                      else "STAYS NO" if all(res[k]["change"] >= 0 for k in ("tune", "hold")) else "TRADE-OFF")
    return res


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    # PRODUCTION_METRICS is the board's; the candidates built from the extra metrics (d_sack, d_st, ...)
    # need EXTRA_METRICS, a superset (checked below: the base model's misses are unchanged by it).
    frame = model.build_frame(games, ExtRatingEngine(tg, qb, te, qe, extra_metrics=EXTRA_METRICS, games=games), injuries=inj)
    prod = model.build_frame(games, ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games), injuries=inj)
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    base = run(frame, mf, tf)
    pb = run(prod, mf, tf)
    same = bool(np.allclose(pb["model_margin"], base.loc[pb.index, "model_margin"]) and np.allclose(pb["model_total"], base.loc[pb.index, "model_total"]))
    print(f"base identical with the board's metrics: {same}", flush=True)
    print(f"base margin feats {mf}\nbase total feats {tf}", flush=True)
    res = {"margin": {}, "total": {}}
    for c in MARGIN:
        if c not in frame or c in mf:
            print(f"  margin {c}: {'missing' if c not in frame else 'already in'}")
            continue
        r = diff(base, run(frame, mf + [c], tf), "model_margin", "result")
        res["margin"][c] = r
        print(f"  margin {c:11s} tune {r['tune']['change']:+.4f} ± {r['tune']['se']:.4f}  hold {r['hold']['change']:+.4f} ± {r['hold']['se']:.4f}  {r['verdict']}", flush=True)
    for c in TOTAL:
        if c not in frame or c in tf:
            print(f"  total {c}: {'missing' if c not in frame else 'already in'}")
            continue
        r = diff(base, run(frame, mf, tf + [c]), "model_total", "total")
        res["total"][c] = r
        print(f"  total  {c:11s} tune {r['tune']['change']:+.4f} ± {r['tune']['se']:.4f}  hold {r['hold']['change']:+.4f} ± {r['hold']['se']:.4f}  {r['verdict']}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
