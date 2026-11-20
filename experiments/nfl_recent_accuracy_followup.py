"""T3b: the NFL recent-accuracy weight under a corrected rule (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/nfl_recent_accuracy_followup.py

Addendum, written before the NFL holdout was looked at: Project doc
claude/plan-batch-tests-addendum-t3b-2026-11-20.md. Committed before its first run.

experiments/recent_accuracy_weight_test.py (T3) stopped on NFL tuning because its rule needed a
strictly better side win rate, and this change never moves a number across the line (0 sides
changed of 1,244). The term and the three versions are unchanged; see that script.

Corrected rule, NFL only:
  - Chosen on tuning 2014-18 alone: better on average miss and RMS miss, side win rate not worse;
    then the lowest tuning average miss.
  - Kept only if on the holdout 2019-25 the average miss is better by at least 1 paired SE, the
    RMS miss is better, and the side win rate is not worse. Provisional, rechecked in January.
  - Better on one of miss and RMS only, or under 1 SE: a trade-off for Ryan. Worse on both: not kept.
  - Reported, not used: 2026 so far; flagged bets (1+ and 2+ points off the closing spread, -110)
    under the board's number and under the candidate's.

Writes experiments/out/nfl_recent_accuracy_followup.json.
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

import recent_accuracy_weight_test as R  # noqa: E402

OUT = HERE / "out" / "nfl_recent_accuracy_followup.json"
TUNE, HOLD = R.PERIODS["nfl"]
DEC = 1 + 100 / 110


def flagged(g: pd.DataFrame, col: str, t: float) -> dict:
    d = g[col] - g.spread_line
    x = g[d.abs() >= t]
    pick = np.sign(d[x.index])
    cover = np.sign(x.result - np.floor(x.spread_line * 2 + 0.5) / 2) * pick
    profit = np.where(cover > 0, DEC - 1, np.where(cover < 0, -1.0, 0.0))
    n, w, l = len(x), int((cover > 0).sum()), int((cover < 0).sum())
    return {"n": int(n), "w": w, "l": l, "win": w / (w + l) if w + l else None,
            "ret": float(profit.mean()) if n else None, "se": float(profit.std(ddof=1) / np.sqrt(n)) if n > 1 else None}


def main() -> None:
    f = R.nfl_frame()
    res = {"versions": {}}
    frames = {}
    for v, (n, sign) in R.VERSIONS.items():
        cand, cs = R.candidate(f, n, sign)
        g = f.assign(cand=cand)
        g = g[g.played & g.spread_line.notna() & g.proj_margin.notna() & g.cand.notna()]
        mt = R.measures(g[g.season.between(*TUNE)])
        ok = mt["mae_cand"] < mt["mae_board"] and mt["rmse_cand"] < mt["rmse_board"] and mt["win_cand"] >= mt["win_board"]
        res["versions"][v] = {"tune": mt, "tune_ok": bool(ok), "c_by_season": cs}
        frames[v] = g
    passing = [v for v in R.VERSIONS if res["versions"][v]["tune_ok"]]
    if not passing:
        res["verdict"] = "No version clears tuning under the corrected rule. Nothing is kept."
    else:
        chosen = min(passing, key=lambda v: res["versions"][v]["tune"]["mae_cand"])
        g = frames[chosen]
        mh = R.measures(g[g.season.between(*HOLD)])
        mae_ok, rmse_ok, win_ok = mh["mae_cand"] < mh["mae_board"], mh["rmse_cand"] < mh["rmse_board"], mh["win_cand"] >= mh["win_board"]
        se_ok = mae_ok and abs(mh["d_mae"]) >= mh["se_d_mae"]
        res.update({"chosen": chosen, "holdout": mh, "checks": {"miss": bool(mae_ok), "miss_1se": bool(se_ok), "rms": bool(rmse_ok), "side_not_worse": bool(win_ok)},
                    "2026": R.measures(g[g.season == 2026]) if (g.season == 2026).any() else None,
                    "by_season": {int(s): {"d_mae": float(((d.cand - d.result).abs() - (d.proj_margin - d.result).abs()).mean()), "n": int(len(d))}
                                  for s, d in g[g.season >= TUNE[0]].groupby("season")},
                    "flagged": {per: {f"{t}+": {"board": flagged(p, "proj_margin", t), "candidate": flagged(p, "cand", t)} for t in (1, 2)}
                                for per, p in (("tuning", g[g.season.between(*TUNE)]), ("holdout", g[g.season.between(*HOLD)]), ("2026", g[g.season == 2026]))},
                    "gap_size": {per: {"board": float((p.proj_margin - p.spread_line).abs().mean()), "candidate": float((p.cand - p.spread_line).abs().mean())}
                                 for per, p in (("tuning", g[g.season.between(*TUNE)]), ("holdout", g[g.season.between(*HOLD)]))}})
        if se_ok and rmse_ok and win_ok:
            res["verdict"] = f"Version {chosen}: holdout average miss better by at least 1 SE, RMS better, side not worse. Passes the corrected rule (provisional)."
        elif not mae_ok and not rmse_ok:
            res["verdict"] = f"Version {chosen}: worse on both holdout miss measures. Not kept."
        else:
            res["verdict"] = f"Version {chosen}: a trade-off (holdout miss better: {bool(mae_ok)}, by 1 SE: {bool(se_ok)}; RMS better: {bool(rmse_ok)}; side not worse: {bool(win_ok)}). Ryan's call."
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")
    show = lambda m: f"n {m['n']} miss {m['mae_board']:.4f} -> {m['mae_cand']:.4f} ({m['d_mae']:+.4f} ± {m['se_d_mae']:.4f}), rms {m['rmse_board']:.4f} -> {m['rmse_cand']:.4f}, side {100 * m['win_board']:.2f}% -> {100 * m['win_cand']:.2f}% ({m['sides_changed']} changed)"
    for v, r in res["versions"].items():
        print(v, "tuning", show(r["tune"]), "ok" if r["tune_ok"] else "no")
    if "holdout" in res:
        print("chosen", res["chosen"], "| holdout", show(res["holdout"]))
        if res["2026"]:
            print("2026", show(res["2026"]))
        print("by season d_mae", {s: round(x["d_mae"], 4) for s, x in res["by_season"].items()})
        for per, r in res["flagged"].items():
            for t, x in r.items():
                s = lambda y: "-" if not y["n"] or y["win"] is None else f"{y['w']}-{y['l']} {100 * y['win']:.1f}% {100 * y['ret']:+.1f}±{100 * (y['se'] or 0):.1f}"
                print(f"flagged {per} {t}: board {s(x['board'])} | candidate {s(x['candidate'])}")
        print("average gap from the line", res["gap_size"])
    print(res["verdict"])


if __name__ == "__main__":
    main()
