"""Player projections, walk-forward, against simple baselines.

    python experiments/players_test.py

Every projection for season S is fitted on seasons before S. Baselines:
  season avg   the player's average in games he played this season so far
               (last season's average in week 1); per start for passing
  recent avg   the same stat averaged over his games with recent ones weighted
               more (the decayed history the model starts from)
Scored on active players in played games: receiving for WR/TE/RB, rushing for
RB and QB, passing for the projected starter in games he started (with an earlier start, so
every baseline exists). Tuning 2013-2018, check 2019 on.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model, players as PL, players_data as PD  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
SETS = {
    "receiving": (["targets", "receptions", "rec_yds", "rec_td"], lambda d: d.group.isin(["WR", "TE", "RB"])),
    "rushing": (["carries", "rush_yds", "rush_td"], lambda d: d.group.isin(["RB", "QB"])),
    "passing": (["pass_att", "completions", "pass_yds", "pass_td", "ints"], lambda d: (d.group == "QB") & (d.att_share > 0.5) & (d.started == 1) & (d.start_sum > 0.3) & (d.pass_att_savg > 0)),
}


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    av = PL.load_availability(verbose=False)
    table = PL.build(frame, pg, info, av, inj, pr, games=games)
    proj = PL.walk_forward(table)
    proj = proj.merge(PL.season_average(table), on=["game_id", "player_id"], how="left")
    for st in PL.STATS:
        proj[f"{st}_recent"] = (proj[f"{st}_st_sum"] / proj["start_sum"].where(proj["start_sum"] > 0.3) if st in PL.PASS_STATS
                                else proj[f"{st}_sum"] / proj["one_sum"].replace(0, np.nan))
    ev = proj[proj.played.astype(bool) & (proj.active == 1)]
    rows = []
    for name, (stats, sel) in SETS.items():
        d = ev[sel(ev)]
        for st in stats:
            for period, x in (("2013-18", d[d.season < C.HOLDOUT_FROM]), ("2019-26", d[d.season >= C.HOLDOUT_FROM])):
                x = x[(x[f"{st}_savg"] > 0) | (x[st] > 0)]
                mae = lambda c: np.mean(np.abs(x[c].fillna(0) - x[st]))
                rows.append({"stat": st, "period": period, "n": len(x), "model": mae(f"{st}_proj"),
                             "season avg": mae(f"{st}_savg"), "recent avg": mae(f"{st}_recent"),
                             "bias": float((x[f"{st}_proj"] - x[st]).mean())})
    # touchdowns as probabilities: chance of at least one, from a Poisson rate
    for st, sel in (("rec_td", SETS["receiving"][1]), ("rush_td", SETS["rushing"][1])):
        d = ev[sel(ev)]
        for period, x in (("2013-18", d[d.season < C.HOLDOUT_FROM]), ("2019-26", d[d.season >= C.HOLDOUT_FROM])):
            x = x[(x[f"{st}_savg"] > 0) | (x[st] > 0)]
            hit = (x[st] > 0).astype(float)
            brier = lambda lam: np.mean((1 - np.exp(-lam.fillna(0).clip(lower=0)) - hit) ** 2)
            rows.append({"stat": st + " (anytime, Brier)", "period": period, "n": len(x), "model": brier(x[f"{st}_proj"]),
                         "season avg": brier(x[f"{st}_savg"]), "recent avg": brier(x[f"{st}_recent"]), "bias": float(np.nan)})
    r = pd.DataFrame(rows)
    r["vs season avg"] = (r["model"] / r["season avg"] - 1) * 100
    print(r.round(3).to_string(index=False))
    proj.to_pickle("/tmp/players_proj.pkl") if len(sys.argv) > 1 else None


if __name__ == "__main__":
    main()
