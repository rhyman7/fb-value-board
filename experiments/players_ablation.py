"""Which pieces of the player model earn their place? Remove one at a time.

    python experiments/players_ablation.py
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
KEY = {"rec_yds": ["WR", "TE", "RB"], "receptions": ["WR", "TE", "RB"], "rush_yds": ["RB", "QB"], "pass_yds": ["QB"], "rec_td": ["WR", "TE", "RB"], "rush_td": ["RB", "QB"]}


def score(proj):
    ev = proj[proj.played.astype(bool) & (proj.active == 1)]
    res = {}
    for st, grp in KEY.items():
        x = ev[ev.group.isin(grp)]
        if st == "pass_yds":
            x = x[x.att_share_raw > 0.5]
        x = x[(x[f"{st}_sum"] > 0) | (x[st] > 0)]
        for period, y in (("tune", x[x.season < C.HOLDOUT_FROM]), ("hold", x[x.season >= C.HOLDOUT_FROM])):
            if st.endswith("_td"):
                hit = (y[st] > 0).astype(float)
                res[(st, period)] = np.mean((1 - np.exp(-y[f"{st}_proj"].clip(lower=0)) - hit) ** 2)
            else:
                res[(st, period)] = np.mean(np.abs(y[f"{st}_proj"] - y[st]))
    return res


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    av = PL.load_availability(verbose=False)
    table = PL.build(frame, pg, info, av, inj, pr)

    variants = {}
    variants["full model"] = table
    t = table.copy()
    for c in ("def_ypt", "def_catch", "def_ypc", "def_ypa", "def_comp"):
        t[c] = 1.0
    variants["no defense matchup"] = t
    t = table.copy()
    t[["opp_miss_db", "opp_miss_pr", "opp_miss_lb"]] = 0.0
    variants["no defensive injuries"] = t
    t = table.copy()
    t["margin"] = 0.0
    t["proj_total"] = t["proj_total"].mean()
    t["imp_pts"] = t["imp_pts"].mean()
    variants["no game script (score)"] = t
    t = table.copy()
    t["wind"] = t["wind"].mean()
    variants["no wind"] = t

    base = None
    rows = []
    for name, tab in variants.items():
        if name == "no teammate redistribution":
            continue
        r = score(PL.walk_forward(tab))
        base = base or r
        rows.append({"variant": name, **{f"{st} {p}": (r[(st, p)] - base[(st, p)]) if name != "full model" else r[(st, p)] for st, p in r}})
        print(name, flush=True)
    # teammates out: compare with shares that are not renormalised among those available
    saved = PL.raw_projection
    def no_redistribution(p, att, car):
        q = saved(p, att, car)
        for c in ("tgt_share", "car_share", "att_share"):
            q[c] = q[f"{c}_raw"].where(q["active"] == 1, 0.0)
        q["pass_att_p"] = q["att_share"] * q["proj_team_att"]
        q["targets_p"] = q["tgt_share"] * q["proj_team_att"]
        q["carries_p"] = q["car_share"] * q["proj_team_car"]
        q["receptions_p"] = q["targets_p"] * (q["catch"] * q["def_catch"]).clip(0.2, 0.95)
        q["rec_yds_p"] = q["targets_p"] * q["ypt"] * q["def_ypt"]
        q["rush_yds_p"] = q["carries_p"] * q["ypc"] * q["def_ypc"]
        q["completions_p"] = q["pass_att_p"] * (q["comp"] * q["def_comp"]).clip(0.4, 0.8)
        q["pass_yds_p"] = q["pass_att_p"] * q["ypa"] * q["def_ypa"]
        return q
    PL.raw_projection = no_redistribution
    r = score(PL.walk_forward(table))
    PL.raw_projection = saved
    rows.append({"variant": "no teammate redistribution", **{f"{st} {p}": r[(st, p)] - base[(st, p)] for st, p in r}})
    out = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 250)
    print("first row: full model (average miss; Brier for TDs). Other rows: change when that piece is removed (positive = the piece helps)")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
