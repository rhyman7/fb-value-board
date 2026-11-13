"""The PFF receiving inputs (route and snap share) re-scored on the 2024 college lines (Nov 13, 2026).

    python experiments/cfb_pff_role_2024_recheck.py      (needs the private PFF data, ../pff-data)

The Nov 4 test (cfb_pff_role_test.py) fitted on 2024 and scored on 2025, with one season of lines.
Its receiving floor-rung score was 1.6 SE (worth a recheck, FEATURES.md "PFF role inputs"); the
chain was later kept under the Nov 4 rule and is on the board. Here the same three scores are run
the same way one season earlier (calibration fitted on 2023, lookback 1, scored on 2024) against the
2024 DK/FD lines, and again for 2025 on today's model for comparison:
  1. receiving-yards RMSE (WR/TE/RB, appeared, chain or projection >= 15 yards)
  2. Brier at the receiving-yards main lines, with vs without the PFF columns
  3. floor rungs: over-only alternates whose books' reading (CFB_RUNG_CAL) is 65-95%
"Without" is the chain with CAL_EXTRA's receiving columns emptied; "with" is the board's chain.
Reported for the January pass; nothing is decided here.

Writes experiments/out/cfb_pff_role_2024_recheck.json (scores only, no PFF rows).
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

from cfbmodel import players as PL  # noqa: E402
import cfb_pff_role_test as T  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_role_prior_test as RP  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = HERE / "out" / "cfb_pff_role_2024_recheck.json"


def main():
    t, pinfo, games = G.table()
    if not (t.get("route_share", 0) > 0).any():
        sys.exit("no PFF route share in the table: check out rhyman7/pff-data (../pff-data or PFF_DATA_DIR)")
    off = {"receptions": [], "rec_yds": []}
    on = {"receptions": T.REC_FEATS, "rec_yds": T.REC_FEATS}
    res = {}
    for season in (2024, 2025):
        cur = T.run_chain(t, off, season, 1)
        new = T.run_chain(t, on, season, 1)
        R.set_season(season)
        r = {"rmse": T.rmse(cur, new, "rec_yds", ["WR", "TE", "RB"], 15.0, {str(season): [season]})[str(season)],
             "lines": RP.score_lines(cur, new, pinfo, games, "rec_yds", "receptions"),
             "floor_rungs": T.floor_rungs(cur, new, pinfo, games, "rec_yds", "receptions")}
        res[str(season)] = r
        print(season, "rmse z", round(r["rmse"]["z"], 2), "lines", round(r["lines"]["new_minus_current"], 5), "z", round(r["lines"]["z"], 2),
              "floor", round(r["floor_rungs"]["new_minus_current"], 5), "z", round(r["floor_rungs"]["z"], 2), flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
