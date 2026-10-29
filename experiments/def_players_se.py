"""Paired standard errors (clustered by game) for the player-projection candidates that passed
(def_players_test.py): per player-game |candidate miss| - |base miss|.

    python experiments/def_players_se.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_common as K  # noqa: E402
import def_players_test as P  # noqa: E402
import def_ratings as DR  # noqa: E402
from nflmodel import players as PL  # noqa: E402


def errs(p, stat, a, b):
    x = p[p["played"].astype(bool) & (p["active"] == 1) & (p["season"] >= a) & (p["season"] <= b)]
    if stat in PL.PASS_STATS:
        x = x[(x["att_share"] > 0.5) & (x["started"] == 1)]
    elif stat.startswith("rec"):
        x = x[x["group"] != "QB"]
    return (x.set_index(["game_id", "player_id"])[stat + "_proj"] - x.set_index(["game_id", "player_id"])[stat]).abs()


def main():
    res = pd.read_csv(K.OUT / "def_players_test.csv")
    res = res[res.passes]
    t = P.add_opp(P.table(), pd.read_csv(DR.RAW / "team_def.csv"))
    base_extra = {k: list(v) for k, v in PL.CAL_EXTRA.items()}
    base = PL.walk_forward(t, first_test=2014)
    rows = []
    for label in res["candidate"].unique():
        _, feat, stats, (tu, ho) = next(c for c in P.CANDS if c[0] == label)
        PL.CAL_EXTRA.clear()
        PL.CAL_EXTRA.update({k: v + (["opp_" + feat] if k in stats else []) for k, v in base_extra.items()})
        cand = PL.walk_forward(t, first_test=2014)
        for st in res.loc[res.candidate == label, "stat"]:
            out = {"candidate": label, "stat": st}
            for name, (a, b) in (("tune", tu), ("hold", ho)):
                d = (errs(cand, st, a, b) - errs(base, st, a, b)).dropna()
                # players in the same game share the opponent, so the standard error is
                # clustered by game: the mean difference per game, over games, weighted by players
                g = d.groupby(level=0).agg(["sum", "size"])
                m = g["sum"].sum() / g["size"].sum()
                resid = g["sum"] - m * g["size"]
                se = np.sqrt((resid ** 2).sum()) / g["size"].sum()
                out[name], out[name + "_se"], out[name + "_n"] = d.mean(), se, len(d)
            rows.append(out)
            print(f"{label:40s} {st:10s} tune {out['tune']:+.4f} ± {out['tune_se']:.4f} (n {out['tune_n']})  hold {out['hold']:+.4f} ± {out['hold_se']:.4f} (n {out['hold_n']})", flush=True)
    PL.CAL_EXTRA.clear()
    PL.CAL_EXTRA.update(base_extra)
    pd.DataFrame(rows).round(5).to_csv(K.OUT / "def_players_se.csv", index=False)


if __name__ == "__main__":
    main()
