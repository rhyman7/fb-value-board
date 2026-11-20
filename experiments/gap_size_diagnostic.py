"""Does a bigger disagreement with the line win more? Both boards (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/gap_size_diagnostic.py

T5 of the Project doc claude/plan-batch-tests-2026-11-20.md (written before the run). A diagnostic:
nothing changes. Committed before its first run.

Each board's walk-forward number against the closing spread, the side graded at -110 at the line
rounded to the half point, by the size of the gap between the board number and the line:
under 0.5, 0.5 to 1, 1 to 2, 2 or more. Tuning, holdout and 2026 (college 2017-19 / 2021-25;
NFL 2014-18 / 2019-25).

Reading, fixed before the run: a win rate that rises with the gap in both tuning and holdout means
the gap ranks bets and a threshold is meaningful; flat or reversed means it does not.

Writes experiments/out/gap_size_diagnostic.json.
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

import recent_accuracy_weight_test as R  # noqa: E402  (the two boards' frames)

OUT = HERE / "out" / "gap_size_diagnostic.json"
BUCKETS = (("under 0.5", 0.0, 0.5), ("0.5 to 1", 0.5, 1.0), ("1 to 2", 1.0, 2.0), ("2 or more", 2.0, 99.0))
DEC = 1 + 100 / 110


def half(x):
    return np.floor(np.asarray(x, float) * 2 + 0.5) / 2


def graded(f: pd.DataFrame) -> pd.DataFrame:
    x = f[f.played & f.spread_line.notna() & f.proj_margin.notna()].copy()
    x["d"] = x.proj_margin - x.spread_line
    pick = np.sign(x.d)
    x["cover"] = np.sign(x.result - half(x.spread_line)) * pick
    x["profit"] = np.where(x.cover > 0, DEC - 1, np.where(x.cover < 0, -1.0, 0.0))
    return x[x.d != 0]


def stats(y: pd.DataFrame) -> dict:
    n, w, l = int(len(y)), int((y.cover > 0).sum()), int((y.cover < 0).sum())
    return {"n": n, "w": w, "l": l, "win": w / (w + l) if w + l else None,
            "ret": float(y.profit.mean()) if n else None, "se": float(y.profit.std(ddof=1) / np.sqrt(n)) if n > 1 else None}


def main() -> None:
    out = {}
    for league, build in (("cfb", R.cfb_frame), ("nfl", R.nfl_frame)):
        g = graded(build())
        (tlo, thi), (hlo, hhi) = R.PERIODS[league]
        res = {}
        for name, lo, hi in (("tuning", tlo, thi), ("holdout", hlo, hhi), ("2026", 2026, 2026)):
            p = g[g.season.between(lo, hi)]
            res[name] = {b: stats(p[(p.d.abs() >= a) & (p.d.abs() < c)]) for b, a, c in BUCKETS}
            res[name]["all"] = stats(p)
        wins = {name: [res[name][b]["win"] for b, _, _ in BUCKETS] for name in ("tuning", "holdout")}
        rising = all(all(w[i] is not None and w[i + 1] is not None and w[i + 1] > w[i] for i in range(len(w) - 1)) for w in wins.values())
        top = all(w[-1] is not None and w[0] is not None and w[-1] > w[0] for w in wins.values())
        res["reading"] = ("The win rate rises with the gap at every step in both tuning and holdout: the gap ranks bets." if rising
                          else "The biggest gaps win more than the smallest in both periods, but not at every step." if top
                          else "Flat or reversed: the size of the gap does not rank bets.")
        out[league] = res
        for name in ("tuning", "holdout", "2026"):
            row = " | ".join(f"{b}: {s['w']}-{s['l']} {100 * s['win']:.1f}% {100 * s['ret']:+.1f}±{100 * s['se']:.1f}" if s["n"] > 1 and s["win"] is not None else f"{b}: -"
                             for b, s in res[name].items())
            print(f"{league} {name}: {row}")
        print(league, res["reading"], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
