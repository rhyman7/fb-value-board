"""The player-projection candidates that passed (def_players_test.py), together:
team missed-tackle rate and yards per target allowed in receiving yards, team pressure
rate in interceptions. Change in average miss (tuning 2020-2022, holdout 2023 on, clustered
standard errors) and how much the projections move.

    python experiments/def_players_combined.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_common as K  # noqa: E402
import def_players_se as S  # noqa: E402
import def_players_test as P  # noqa: E402
import def_ratings as DR  # noqa: E402
from nflmodel import players as PL  # noqa: E402

ADD = {"rec_yds": ["opp_agg_mt", "opp_agg_ypt"], "ints": ["opp_agg_press"]}


def main():
    t = P.add_opp(P.table(), pd.read_csv(DR.RAW / "team_def.csv"))
    base_extra = {k: list(v) for k, v in PL.CAL_EXTRA.items()}
    base = PL.walk_forward(t, first_test=2014)
    PL.CAL_EXTRA.update({k: v + ADD.get(k, []) for k, v in base_extra.items()})
    cand = PL.walk_forward(t, first_test=2014)
    PL.CAL_EXTRA.clear()
    PL.CAL_EXTRA.update(base_extra)
    rows = []
    for st in ADD:
        for name, (a, b) in (("tune", (2020, 2022)), ("hold", (2023, 2100))):
            d = (S.errs(cand, st, a, b) - S.errs(base, st, a, b)).dropna()
            g = d.groupby(level=0).agg(["sum", "size"])
            m = g["sum"].sum() / g["size"].sum()
            se = np.sqrt(((g["sum"] - m * g["size"]) ** 2).sum()) / g["size"].sum()
            rows.append({"stat": st, "period": name, "change": m, "se": se, "n": len(d)})
            print(f"{st:8s} {name}: {m:+.4f} ± {se:.4f} (n {len(d)})")
        k = ["game_id", "player_id"]
        mv = cand[cand.season == 2026].set_index(k)[st + "_proj"] - base[base.season == 2026].set_index(k)[st + "_proj"]
        x = cand[cand.season == 2026].set_index(k)[st + "_proj"]
        sel = x > (0.5 if st == "ints" else 20)
        print(f"{st:8s} 2026 projections move: median |{mv[sel].abs().median():.3f}|, 95th pct |{mv[sel].abs().quantile(0.95):.3f}|, max |{mv[sel].abs().max():.3f}|")
    pd.DataFrame(rows).round(5).to_csv(K.OUT / "def_players_combined.csv", index=False)


if __name__ == "__main__":
    main()
