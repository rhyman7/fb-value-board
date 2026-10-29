"""Part B: the opponent's defensive ratings in the player projections.

    python experiments/def_players_test.py

The player projections already multiply each player's efficiency by what the opposing
defense has allowed (yards per target, carry and attempt), and the receiving calibration has
the opponent's missing defensive backs and pass rush by head count (opp_miss_db, opp_miss_pr;
PLAYERS.md). Each candidate here adds one more term to the last calibration step of the stats
it should touch (raw projection x the opponent's value, as CAL_EXTRA does), walk-forward.
Scored on players who played: average miss per player-game, by stat.

Windows as def_test.py: play-by-play pass rush tuning 2014-2018, holdout 2019 on; PFR inputs
tuning 2020-2022, holdout 2023 on (short). A candidate is kept for a stat only if it helps
that stat on both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_common as K  # noqa: E402
import def_ratings as DR  # noqa: E402
from nflmodel import injuries, model, players as PL, players_data as PD  # noqa: E402

LONG, PFR = ((2014, 2018), (2019, 2100)), ((2020, 2022), (2023, 2100))
CANDS = [  # label, opponent feature, stats, window
    ("opp pass rush lost (pbp, vs backup)", "miss_pr_pbp_bk_all", ["pass_yds", "completions", "ints", "rec_yds"], LONG),
    ("opp pressure lost, change (PFR, vs backup)", "chg_pr_pfr_bk_all", ["pass_yds", "completions", "ints", "rec_yds"], PFR),
    ("opp coverage lost, change (PFR, DBs, vs backup)", "chg_cov_db_bk_db", ["pass_yds", "completions", "rec_yds", "receptions"], PFR),
    ("opp coverage lost, missing now (PFR, DBs, vs backup)", "miss_cov_db_bk_db", ["pass_yds", "completions", "rec_yds", "receptions"], PFR),
    ("opp tackling lost (PFR, vs backup)", "miss_mt_bk_all", ["rush_yds", "rec_yds"], PFR),
    ("opp team pressure rate", "agg_press", ["pass_yds", "completions", "ints"], PFR),
    ("opp team yards per target allowed", "agg_ypt", ["pass_yds", "rec_yds", "receptions"], PFR),
    ("opp team missed-tackle rate", "agg_mt", ["rush_yds", "rec_yds"], PFR),
]


def table():
    path = K.CACHE / "ptable.pkl"
    if path.exists():
        return pd.read_pickle(path)
    frame = K.production_frame()
    f0, _ = model.walk_forward(frame)
    g = K.games()
    pg, pinfo = PD.load_player_games(verbose=False)
    avail = PL.load_availability(verbose=False)
    inj = injuries.load_injuries(g, verbose=False)
    t = PL.build(f0, pg, pinfo, avail, inj, injuries.load_pass_rush(g, verbose=False), games=g)
    t.to_pickle(path)
    return t


def add_opp(t: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    T = T.copy()
    for c in [c for c in T.columns if c.startswith("miss_")]:
        T["chg_" + c[5:]] = T[c] - T["mrec_" + c[5:]]
    cols = sorted({c for _, c, _, _ in CANDS})
    x = T[["game_id", "team"] + cols].rename(columns={"team": "opp", **{c: "opp_" + c for c in cols}})
    t = t.merge(x, on=["game_id", "opp"], how="left")
    for c in cols:
        v = t["opp_" + c]
        t["opp_" + c] = (v - v.mean()).fillna(0.0) if c.startswith("agg_") else v.fillna(0.0)
        t["opp_" + c] = t["opp_" + c] / (t["opp_" + c].std() or 1.0)     # one unit = one standard deviation
    return t


def miss(p: pd.DataFrame, stat: str, a: int, b: int) -> float:
    x = p[p["played"].astype(bool) & (p["active"] == 1) & (p["season"] >= a) & (p["season"] <= b)]
    if stat in PL.PASS_STATS:      # the projected starter in games he started, as backtest_summary
        x = x[(x["att_share"] > 0.5) & (x["started"] == 1)]
    else:
        x = x[x["group"] != "QB"] if stat.startswith("rec") else x
    col = stat + "_proj" if stat + "_proj" in x else stat + "_p"
    return float(np.mean(np.abs(x[col] - x[stat])))


def main():
    t = add_opp(table(), pd.read_csv(DR.RAW / "team_def.csv"))
    base_extra = {k: list(v) for k, v in PL.CAL_EXTRA.items()}
    base = PL.walk_forward(t, first_test=2014)
    rows = []
    for label, feat, stats, (tu, ho) in CANDS:
        PL.CAL_EXTRA.clear()
        PL.CAL_EXTRA.update({k: v + (["opp_" + feat] if k in stats else []) for k, v in base_extra.items()})
        cand = PL.walk_forward(t, first_test=2014)
        for st in stats:
            r = {"candidate": label, "stat": st,
                 "tune": miss(cand, st, *tu) - miss(base, st, *tu), "hold": miss(cand, st, *ho) - miss(base, st, *ho),
                 "base_tune": miss(base, st, *tu), "base_hold": miss(base, st, *ho)}
            r["passes"] = bool(r["tune"] < 0 and r["hold"] < 0)
            rows.append(r)
            print(f"{label:52s} {st:12s} tune {r['tune']:+.4f} (base {r['base_tune']:.3f})  hold {r['hold']:+.4f} (base {r['base_hold']:.3f})"
                  + ("  <-- passes" if r["passes"] else ""), flush=True)
    PL.CAL_EXTRA.clear()
    PL.CAL_EXTRA.update(base_extra)
    K.OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(K.OUT / "def_players_test.csv", index=False)


if __name__ == "__main__":
    main()
