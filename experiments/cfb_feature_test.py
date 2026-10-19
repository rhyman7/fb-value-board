"""Candidate features for the college game model, scored walk-forward.

A feature is kept only if it lowers the raw model's average miss on the tuning seasons
(2017-2019). The holdout (2021 on) is the check, not the selector. Kept features are
shown as the change when each is removed; the others as the change when each is added.

    python experiments/cfb_feature_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import data, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

MARGIN_CANDIDATES = ["d_talent", "d_talent_early", "d_rtprod", "d_rtprod_early", "d_off_rtprod", "d_def_rtprod",
                     "d_ret_qb", "d_qb_early", "d_mkt_early", "d_pts_early", "d_sr_early", "d_snaps", "travel",
                     "elev", "hfa_conf"]
TOTAL_CANDIDATES = ["s_snaps", "s_def_rtprod", "s_off_rtprod", "s_rtprod", "s_talent", "s_ret_qb", "wind", "wind_hi",
                    "cold", "elev"]
mae = lambda a, b: float(np.mean(np.abs(a - b)))


def score(fr, MF=None, TF=None):
    out, _ = model.walk_forward(fr, margin_feats=MF, total_feats=TF)
    f = out[out.played & out.spread_line.notna() & out.total_line.notna()]
    T, H = f[f.season.between(2017, 2019)], f[f.season >= 2021]
    return (mae(T.model_margin, T.result), mae(H.model_margin, H.result),
            mae(T.model_total, T.total), mae(H.model_total, H.total),
            mae(T.spread_line, T.result), mae(H.spread_line, H.result), mae(T.total_line, T.total), mae(H.total_line, H.total))


def main():
    games, tg, qb = data.load_all(verbose=False)
    fr = model.build_frame(games, CFBRatingEngine(tg, qb, games=games))
    base = score(fr)
    print(f"Current model, margin: tuning {base[0]:.3f}, holdout {base[1]:.3f} (closing line {base[4]:.3f} / {base[5]:.3f})")
    print(f"Current model, total:  tuning {base[2]:.3f}, holdout {base[3]:.3f} (closing line {base[6]:.3f} / {base[7]:.3f})")
    nfl_base = score(fr, MF=model.BASE_MARGIN, TF=model.BASE_TOTAL)
    print(f"NFL feature set only:  margin {nfl_base[0]:.3f} / {nfl_base[1]:.3f}, total {nfl_base[2]:.3f} / {nfl_base[3]:.3f}")
    print("\nMargin                tuning  holdout  kept")
    for c in MARGIN_CANDIDATES:
        kept = c in model.MARGIN_FEATS
        mf = [x for x in model.MARGIN_FEATS if x != c] if kept else model.MARGIN_FEATS + [c]
        s = score(fr, MF=mf)
        sign = -1 if kept else 1     # for kept features, report the gain from having them
        print(f"{c:20s} {sign * (s[0] - base[0]):+.3f}  {sign * (s[1] - base[1]):+.3f}   {'yes' if kept else 'no'}", flush=True)
    print("\nTotal                 tuning  holdout  kept")
    for c in TOTAL_CANDIDATES:
        kept = c in model.TOTAL_FEATS
        tf = [x for x in model.TOTAL_FEATS if x != c] if kept else model.TOTAL_FEATS + [c]
        s = score(fr, TF=tf)
        sign = -1 if kept else 1
        print(f"{c:20s} {sign * (s[2] - base[2]):+.3f}  {sign * (s[3] - base[3]):+.3f}   {'yes' if kept else 'no'}", flush=True)


if __name__ == "__main__":
    main()
