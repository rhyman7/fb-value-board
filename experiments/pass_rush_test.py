"""Pass-rush availability, weighted by production (QB hits and sacks), tested walk-forward.

    python experiments/pass_rush_test.py

The base is the current model (it already includes the offensive line).
Tuning seasons 2013-2018; 2019 on is the check.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")


def score(frame, mf, tf):
    out, info = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    o = out[out.played & out.spread_line.notna() & (out.season >= 2013)]
    res = {}
    for name, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        hv = x[x.d_chg_pr.abs() >= 0.25]                     # one side lost a quarter of its rush beyond recent games
        res[name] = dict(m=np.mean(np.abs(x.model_margin - x.result)), t=np.mean(np.abs(x.model_total - x.total)),
                         heavy=np.mean(np.abs(hv.model_margin - hv.result)), n=len(hv))
    return res, info


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame = injuries.add_pass_rush_features(model.build_frame(games, eng, injuries=inj), pr)
    frame["s_miss_pr"] = frame["h_miss_pr"] + frame["a_miss_pr"]
    frame["s_chg_pr"] = (frame["h_miss_pr"] - frame["h_mrec_pr"]) + (frame["a_miss_pr"] - frame["a_mrec_pr"])
    # the old position-based group, for comparison
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    b, _ = score(frame, mf, tf)
    print(f"base           margin tune {b['tune']['m']:.3f} hold {b['hold']['m']:.3f} | heavy games tune {b['tune']['heavy']:.3f} (n {b['tune']['n']}) "
          f"hold {b['hold']['heavy']:.3f} (n {b['hold']['n']}) | total tune {b['tune']['t']:.3f} hold {b['hold']['t']:.3f}")
    for name, add_m, add_t in (("d_miss_pr", ["d_miss_pr"], []), ("d_chg_pr", ["d_chg_pr"], []),
                               ("d_miss_dl (old)", ["d_miss_dl"], []), ("d_chg_dl (old)", ["d_chg_dl"], []),
                               ("total s_miss_pr", [], ["s_miss_pr"]), ("total s_chg_pr", [], ["s_chg_pr"])):
        r, info = score(frame, mf + add_m, tf + add_t)
        c = info[2026]["margin_coefs" if add_m else "total_coefs"][(add_m or add_t)[0]]
        print(f"{name:15s} margin tune {r['tune']['m'] - b['tune']['m']:+.4f} hold {r['hold']['m'] - b['hold']['m']:+.4f}"
              f" | heavy tune {r['tune']['heavy'] - b['tune']['heavy']:+.4f} hold {r['hold']['heavy'] - b['hold']['heavy']:+.4f}"
              f" | total tune {r['tune']['t'] - b['tune']['t']:+.4f} hold {r['hold']['t'] - b['hold']['t']:+.4f} | coef {c:+.2f}", flush=True)


if __name__ == "__main__":
    main()
