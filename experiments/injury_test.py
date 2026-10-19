"""Missing starters by position group, tested walk-forward.

    python experiments/injury_test.py

Tuning seasons are 2013-2018 here (injury data starts 2012, which has no usable
snap counts); 2019 on is the check. 'Heavy' games are those where the two teams'
change in missing starters differs by at least one full starter's worth.
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
GROUPS = list(injuries.GROUPS)


def score(frame, mf):
    out, info = model.walk_forward(frame, margin_feats=mf)
    o = out[out.played & out.spread_line.notna() & (out.season >= 2013)]
    res = {}
    for name, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        hv = x[x.heavy]
        res[name] = dict(all=np.mean(np.abs(x.model_margin - x.result)), heavy=np.mean(np.abs(hv.model_margin - hv.result)),
                         heavy_mkt=np.mean(np.abs(hv.spread_line - hv.result)), n=len(hv),
                         gap=np.mean(np.abs(hv.model_margin - hv.spread_line)))
    return res, info


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame = injuries.add_injury_features(model.build_frame(games, eng), inj)
    w = {"ol": 5, "pc": 3, "rb": 1, "dl": 4, "lb": 2, "db": 4}
    frame["heavy"] = sum(frame[f"d_chg_{g}"].abs() * w[g] for g in GROUPS) >= 1.0
    base = list(model.MARGIN_FEATS)
    b, _ = score(frame, base)
    fmt = lambda r: f"all {r['all']:.3f}  heavy {r['heavy']:.3f} (mkt {r['heavy_mkt']:.3f}, n {r['n']})"
    print(f"{'base':16s} tune {fmt(b['tune'])} | hold {fmt(b['hold'])}")
    tests = [[f"d_miss_{g}"] for g in GROUPS] + [[f"d_chg_{g}"] for g in GROUPS]
    tests += [[f"d_miss_{g}" for g in GROUPS], [f"d_chg_{g}" for g in GROUPS]]
    for t in tests:
        r, info = score(frame, base + t)
        name = t[0] if len(t) == 1 else ("all miss" if t[0].startswith("d_miss") else "all chg")
        coef = info[2026]["margin_coefs"]
        cs = " ".join(f"{k[2:]}={coef[k]:+.1f}" for k in t)
        print(f"{name:16s} tune {r['tune']['all'] - b['tune']['all']:+.4f} heavy {r['tune']['heavy'] - b['tune']['heavy']:+.4f}"
              f" | hold {r['hold']['all'] - b['hold']['all']:+.4f} heavy {r['hold']['heavy'] - b['hold']['heavy']:+.4f} | {cs}", flush=True)


if __name__ == "__main__":
    main()
