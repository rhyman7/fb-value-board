"""Paired standard errors for the defensive candidates that passed the October rule
(def_test.py): per game, |candidate miss| - |base miss|, mean and standard error, on the
tuning and holdout windows. A change smaller than about one standard error is noise.

    python experiments/def_se.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_common as K  # noqa: E402
import def_ratings as DR  # noqa: E402
import def_test as T  # noqa: E402
from nflmodel import model  # noqa: E402


def errors(frame, mf, tf):
    out, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    o = out[out.played & out.spread_line.notna()].set_index("game_id")
    return o.season, (o.model_margin - o.result).abs(), (o.model_total - o.total).abs()


def main():
    res = pd.read_csv(K.OUT / "def_test.csv")
    frame = T.attach(K.production_frame(), pd.read_csv(DR.RAW / "team_def.csv"))
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    season, bm, bt = errors(frame, mf, tf)
    rows = []
    for r in res[res.passes].itertuples():
        feats = r.features.split()
        am = [f for f in feats if f.startswith("d_")]
        at = [f for f in feats if f.startswith("s_")]
        _, cm, ct = errors(frame, mf + am, tf + at)
        d = (cm - bm) if am else (ct - bt)
        tu, ho = T.WINDOWS[r.window]
        out = {"candidate": r.candidate, "window": r.window}
        for name, (a, b) in (("tune", tu), ("hold", ho)):
            x = d[(season >= a) & (season <= b)]
            out[name] = x.mean()
            out[name + "_se"] = x.std(ddof=1) / np.sqrt(len(x))
            out[name + "_n"] = len(x)
        rows.append(out)
        print(f"{r.candidate:52s} tune {out['tune']:+.4f} ± {out['tune_se']:.4f} (n {out['tune_n']})  hold {out['hold']:+.4f} ± {out['hold_se']:.4f} (n {out['hold_n']})", flush=True)
    pd.DataFrame(rows).round(4).to_csv(K.OUT / "def_se.csv", index=False)


if __name__ == "__main__":
    main()
