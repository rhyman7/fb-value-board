"""Defense against each position on the projections the board actually shows.

    python experiments/dvp_ml_test.py

experiments/def_vs_position.py tested the position split on the chain projections
(players.walk_forward). Since Oct 31, 2026 the board shows the New projections
(players_ml.ADOPT: backs' carries, rushing yards, receptions and receiving yards; wide receivers'
receptions and receiving yards), which already take the team-level defense as an input. This
checks nflmodel/dvp.py on them, and re-checks the chain with the production code:

  chain   b fitted 2013-2018, scored 2019 on (as def_vs_position.py)
  New     the New projections exist from 2019 (players_ml.walk_forward): b fitted 2019-2021,
          scored 2022 on

Rule, set before running: a pair changes the board if its scored seasons' average miss is lower
by more than two standard errors (clustered by game) on the projections the board shows for it.

Writes experiments/out/dvp_ml_test.json.
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
import props_blend as PB  # noqa: E402
from nflmodel import config as C, data, dvp, players_ml, roles  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "dvp_ml_test.json"


def score(proj, rt, b, fit_seasons, test_seasons, label):
    played = proj["played"].fillna(False).astype(bool) & (proj["active"] == 1)
    rows = []
    for (grp, stat), bb in b.items():
        col = stat + "_proj"
        d = proj[played & dvp._eligible(proj, grp) & proj[col].notna() & (proj[col] > 0)]
        d = d[d["season"].isin(test_seasons)]
        r = dvp.attach(d, rt, grp, stat)
        adj = d[col] * np.exp(bb * r)
        diff = (adj - d[stat]).abs() - (d[col] - d[stat]).abs()
        se = PB._se_by_game(diff, d["game_id"])
        rows.append({"projections": label, "group": grp, "stat": stat, "b": round(bb, 3), "n": int(len(d)),
                     "mae": round(float((d[col] - d[stat]).abs().mean()), 3), "gain": round(float(diff.mean()), 4),
                     "se": round(se, 4), "passes": bool(diff.mean() < -2 * se)})
    return rows


def main():
    from props_calibration import load_proj
    pall = load_proj(None)
    print("chain ratings ...", flush=True)
    rt = dvp.ratings(pall)
    chain = score(pall, rt, dvp.fit(pall, rt, range(2013, C.HOLDOUT_FROM)), None,
                  [s for s in pall.season.unique() if s >= C.HOLDOUT_FROM], "chain")
    print(pd.DataFrame(chain).to_string(index=False), flush=True)

    print("New projections (walk-forward from 2019) ...", flush=True)
    games = data.load_games()
    R = roles.load_all(games, verbose=False)
    ml = players_ml.questionable(players_ml.walk_forward(players_ml.add_features(pall, R), first=C.HOLDOUT_FROM),
                                 first=C.FIRST_TEST_SEASON)
    ml = ml[ml["season"] >= C.HOLDOUT_FROM]
    rows = []
    for st, groups in players_ml.ADOPT.items():
        for grp in groups:
            if (grp, st) not in dvp.APPLY:
                continue
            x = ml[ml["group"] == grp]
            # rate defenses on the New projections themselves (what the board would do)
            rt_ml = dvp.ratings(x)
            b = dvp.fit(x, rt_ml, range(C.HOLDOUT_FROM, 2022))
            if (grp, st) in b:
                rows += score(x, rt_ml, {(grp, st): b[(grp, st)]}, None, list(range(2022, 2027)), "New")
    print(pd.DataFrame(rows).to_string(index=False))
    OUT.write_text(json.dumps({"chain": chain, "new": rows}, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
