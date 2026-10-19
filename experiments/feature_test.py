"""Test candidate features one at a time, walk-forward.

    python experiments/feature_test.py            # all candidates
    python experiments/feature_test.py d_cpoe wind

Each candidate is added to the base margin or total model on its own. The
change in the raw model's average miss is reported for the seasons the
settings are chosen on (2011-2018, out of sample) and for the holdout (2019 on).
A feature earns a place only if it helps on the tuning seasons; the holdout is
the check, not the selector. Results as of Oct 2026 are in FEATURES.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.features_ext import EXTRA_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
MARGIN_CANDIDATES = ["d_sack", "d_expl", "d_ed", "d_st", "d_fum", "d_cpoe", "d_pace", "d_snaps",
                     "travel", "tz_east", "west_early", "hfa_div"]
TOTAL_CANDIDATES = ["s_pace", "s_snaps", "s_expl", "s_sack", "s_ed", "s_fum", "wind_hi", "cold", "wind"]
BASE_TOTAL = [f for f in model.TOTAL_FEATS if f not in ("wind", "s_snaps")]   # the model before Oct 2026


def score(frame, mf, tf):
    out, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    o = out[out.played & out.spread_line.notna()]
    res = {}
    for name, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        res[name] = {"margin": np.mean(np.abs(x.model_margin - x.result)), "total": np.mean(np.abs(x.model_total - x.total))}
    return res


def main(only: list[str]) -> None:
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=EXTRA_METRICS, games=games)
    frame = model.build_frame(games, engine)
    mf, tf = list(model.MARGIN_FEATS), BASE_TOTAL
    base = score(frame, mf, tf)
    print(f"base  margin tune {base['tune']['margin']:.3f} hold {base['hold']['margin']:.3f} | "
          f"total tune {base['tune']['total']:.3f} hold {base['hold']['total']:.3f}")
    print("change in average miss when added (negative = better)")
    for c in MARGIN_CANDIDATES:
        if only and c not in only:
            continue
        r = score(frame, mf + [c], tf)
        print(f"  margin {c:11s} tune {r['tune']['margin'] - base['tune']['margin']:+.4f}  hold {r['hold']['margin'] - base['hold']['margin']:+.4f}")
    for c in TOTAL_CANDIDATES:
        if only and c not in only:
            continue
        r = score(frame, mf, tf + [c])
        print(f"  total  {c:11s} tune {r['tune']['total'] - base['tune']['total']:+.4f}  hold {r['hold']['total'] - base['hold']['total']:+.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
