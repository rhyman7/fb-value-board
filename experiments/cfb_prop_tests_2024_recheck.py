"""Three college player-projection tests judged on the 2025 lines alone, re-scored on 2024 (Nov 13, 2026).

    python experiments/cfb_prop_tests_2024_recheck.py      (PFF data at ../pff-data, as the board)

The 2024 college prop backfill (data/props/history_cfb/2024/, weeks 1-16) gives each a second season.
For each, the board's walk-forward projections with the piece against without it, Brier at the
DK/FD main lines of the stats it touches (new minus current, clustered by game; NFL prop widths, as
the original tests scored), 2024 and 2025:

  1. Rushing calibrated per position group (cfb_rush_group_calib_test.py, Nov 3; on the board).
     Without: CALIB_GROUPS for rushing yards and carries emptied (the pooled fit alone).
  2. Defense against each position (cfb_dvp_test.py, Nov 3; five pairs on the board). Without:
     DVP_PAIRS emptied. Lines: receiving yards, rushing yards, passing yards.
  3. The early-season role prior from CFBD recruiting/portal data (cfb_role_prior_test.py, Nov 3; not
     kept, 0.7 SE on 2025 weeks 3-8). With: the prior at the weight chosen then (w = 1). Lines:
     rushing yards in weeks 1-8; 2024's lines start in week 1, which 2025's (week 3) could not test.

Reported for the January pass; nothing is decided here.

Writes experiments/out/cfb_prop_tests_2024_recheck.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from cfbmodel import config as C, players as PL  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_role_prior_test as RP  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = HERE / "out" / "cfb_prop_tests_2024_recheck.json"
VOL = {"rush_yds": "carries", "rec_yds": "receptions", "pass_yds": "pass_att"}


def lines(cur, new, pinfo, games, stats, weeks=None) -> dict:
    out = {}
    for s in (2024, 2025):
        R.set_season(s)
        for st in stats:
            r = RP.score_lines(cur, new, pinfo, games, st, VOL[st], weeks=weeks)
            out[f"{s} {st}"] = {k: round(float(v), 6) for k, v in r.items()}
            print(f"  {s} {st}: {r['props']} props, new - current {r['new_minus_current']:+.5f} ± {r['se']:.5f} (z {r['z']:+.2f})", flush=True)
    return out


def main():
    t, pinfo, games = G.table()
    cur = PL.walk_forward(t)
    res = {}

    print("1. rushing calibration by position group", flush=True)
    saved = dict(PL.CALIB_GROUPS)
    PL.CALIB_GROUPS = {**saved, "rush_yds": (), "carries": ()}
    pooled = PL.walk_forward(t)
    PL.CALIB_GROUPS = saved
    res["rush_group_calibration"] = lines(pooled, cur, pinfo, games, ["rush_yds"])       # new = the board

    print("2. defense against each position", flush=True)
    saved_pairs = list(PL.DVP_PAIRS)
    PL.DVP_PAIRS = []
    nodvp = PL.walk_forward(t)
    PL.DVP_PAIRS = saved_pairs
    res["defense_vs_position"] = lines(nodvp, cur, pinfo, games, ["rec_yds", "rush_yds", "pass_yds"])

    print("3. early-season role prior (w = 1), weeks 1-8", flush=True)
    seasons = sorted(int(s) for s in t["season"].unique())
    bg = RP.background([s for s in seasons if (RP.CFBD / f"roster_{s}.csv").exists()])
    ret = RP.returning(seasons)
    f = RP.features(t, bg, ret, "car_share", ("RB",))
    pri = RP.fit_priors(f, C.FIRST_TEST_SEASON)
    prior = PL.walk_forward(RP.apply_prior(t, pri, "car_share", ("RB",), 1))
    res["role_prior_weeks_1_8"] = lines(cur, prior, pinfo, games, ["rush_yds"], weeks=(1, 8))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
