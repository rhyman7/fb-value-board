"""Check of the production version (nflmodel/defense.py) of the opponent's team rates in the
player projections, against the projections without them: the same test as
def_players_combined.py, on the rates exactly as the board computes them (pressures per
opponent dropback, weekly league average as the reference).

    python experiments/def_players_production.py
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
from nflmodel import data, defense, players as PL  # noqa: E402

NEW = {"rec_yds": ["opp_def_mt", "opp_def_ypt"], "ints": ["opp_def_press"]}


def main():
    games, tg, _ = data.load_all(refresh=False, verbose=False)
    rates = defense.team_rates(games, tg, defense.load(verbose=False))
    T0 = P.table()
    t = defense.attach(T0.drop(columns=[c for c in T0.columns if c.startswith(("opp_def_", "opp_lg_", "opp_ref_"))]), rates)
    without = {k: list(v) for k, v in PL.CAL_EXTRA.items()}
    full = {k: v + [c for c in NEW.get(k, []) if c not in v] for k, v in without.items()}
    PL.CAL_EXTRA.clear(); PL.CAL_EXTRA.update(without)
    base = PL.walk_forward(t, first_test=2014)
    PL.CAL_EXTRA.clear(); PL.CAL_EXTRA.update(full)
    cand = PL.walk_forward(t, first_test=2014)
    rows = []
    for st in NEW:
        for name, (a, b) in (("tune", (2020, 2022)), ("hold", (2023, 2100))):
            d = (S.errs(cand, st, a, b) - S.errs(base, st, a, b)).dropna()
            g = d.groupby(level=0).agg(["sum", "size"])
            m = g["sum"].sum() / g["size"].sum()
            se = np.sqrt(((g["sum"] - m * g["size"]) ** 2).sum()) / g["size"].sum()
            rows.append({"stat": st, "period": name, "change": m, "se": se, "n": len(d)})
            print(f"{st:8s} {name}: {m:+.4f} ± {se:.4f} (n {len(d)})")
    pd.DataFrame(rows).round(5).to_csv(K.OUT / "def_players_production.csv", index=False)


if __name__ == "__main__":
    main()
