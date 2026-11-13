"""College props: the 2025-only results re-scored on the 2024 lines (Nov 13, 2026).

    python experiments/cfb_props_2024_recheck.py

The 2024 college prop backfill (data/props/history_cfb/2024/, pulled Nov 13, 2026: weeks 1-16,
855 games with DraftKings or FanDuel props) gives the college prop tests a second season. Every
college prop test before this date was judged on the 2025 lines alone.

What this runs (the board-level numbers, the passing-yards nudge #2, the anytime TD #3 and the
college ladder reading come from `cfb_props_real_lines.py --season 2024 --params
data/cfb/prop_params.json`; this script adds the one near-miss that needs a model refit):

  #7 on the near-miss list: college backs' usage by down type x projected margin in the rushing
  chain (cfb_usage_ppa_test.py, candidate a). Its 2025 result was -0.0012 +/- 0.0007 at the
  rushing-yards main lines (1.87 SE). Scored here at the 2024 and 2025 lines, the same way
  (cfb_role_prior_test.score_lines: Brier of the new projections minus the current ones at the
  same props, NFL prop widths as in the original test, standard errors clustered by game).

No rule decides anything here: this is a re-score of results already on the near-miss list, for
the January pass. Nothing on the board changes.

Writes experiments/out/cfb_props_2024_recheck.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from cfbmodel import players as PL  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_role_prior_test as RP  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402
import cfb_usage_ppa_test as U  # noqa: E402

OUT = HERE / "out" / "cfb_props_2024_recheck.json"


def main():
    t, pinfo, games = G.table()
    seasons = sorted(int(s) for s in t["season"].unique())
    cur = PL.walk_forward(t)
    up = U.usage_prev(seasons)
    ta = t.merge(up[["season", "player_id", "pd_diff"]], on=["season", "player_id"], how="left")
    ta["pd_diff"] = ta["pd_diff"].fillna(0.0)
    ta["usage_script"] = ta["pd_diff"] * ta["margin"].fillna(0.0) / 10.0
    saved = dict(PL.CAL_EXTRA)
    PL.CAL_EXTRA = {**saved, "carries": ["usage_script"], "rush_yds": ["usage_script"]}
    new_a = PL.walk_forward(ta)
    PL.CAL_EXTRA = saved
    res = {"usage_by_down_type": {}}
    for s in (2024, 2025):
        R.set_season(s)
        r = RP.score_lines(cur, new_a, pinfo, games, "rush_yds", "carries")
        res["usage_by_down_type"][f"lines_{s}"] = {k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in r.items()}
        print(s, res["usage_by_down_type"][f"lines_{s}"], flush=True)
    res["usage_by_down_type"]["rmse"] = {"carries": U.score(cur, new_a, "carries", 4.0), "rush_yds": U.score(cur, new_a, "rush_yds", 20.0)}
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
