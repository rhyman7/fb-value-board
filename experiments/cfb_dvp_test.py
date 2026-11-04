"""College defense against each position: does it improve the player projections, and the chances
at the 2025 prop lines?

    python experiments/cfb_dvp_test.py

Nov 3, 2026, candidate 2 of the college-props memo. The college player model judges a defense by
team-level yards per carry and per attempt allowed, nothing for receivers. The NFL board's
defense-against-position rating (nflmodel/dvp.py: the log of what a defense allowed to a position
over what the projections expected those players to get, 6-game half-life, last season at half
weight, shrunk by 4 games, applied as projection x exp(b x rating)) cleared 2 SE on every pair but
passing yards for the NFL projections, and tied the books. This is the same rating on the college
walk-forward projections (cfb_rush_group_calib_test.table + players.walk_forward, 2017 on).

Rules, written before running:
1. Accuracy: b per (group, stat) fitted on 2017-2019; RMSE on the holdout (2021+) against the
   projection alone, paired difference clustered by game; also the tuning seasons themselves
   (fitted there). Rows: players who appeared, chain or projection at least 20 yards (yards) or
   4 (carries) / 2 (receptions). A pair goes on the board if the holdout improves by 2+ SE and the
   tuning seasons are not worse.
2. Lines: b refitted on 2017-2024, applied to the 2025 projections; Brier at the 2025
   DraftKings/FanDuel main lines (rushing and receiving yards; passing yards reported) against the
   current chances, clustered by game. The pair counts toward "past the books" only if this
   improves by 2+ SE; it is applied to the board on rule 1 alone if the Brier for its group is no
   worse (<= 0). Pairs with no 2025 lines (receptions, carries) are applied on rule 1 alone.

Result of the rules above (first run): every pair but RB receiving yards is WORSE, on the tuning
seasons too. The NFL rating counts a player who was expected but did not appear as 0 yards allowed
(played & active), and in college, with no injury report, that is common: the rating measured the
opponents' absences. Second stage, rule written before it ran: the same rating from players who
APPEARED (active := appeared). Decision per pair as above (rule 1 on the holdout and tuning, rule 2
at the 2025 lines, Brier for the group no worse). The first stage is kept in the results as
"stage1_pre_set" for the record; the decision is the second stage's.

Writes experiments/out/cfb_dvp_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from cfbmodel import players as PL  # noqa: E402
from nflmodel import dvp as D, odds as O, propdist as PD, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "cfb_dvp_test.json"
TUNE = [2017, 2018, 2019]
FLOORS = {"rec_yds": 20.0, "rush_yds": 20.0, "pass_yds": 100.0, "carries": 4.0, "receptions": 2.0}
LINE_STATS = {"rush_yds": "carries", "rec_yds": "receptions", "pass_yds": "completions"}


def paired_rmse(d: pd.DataFrame, stat: str, new: str) -> dict:
    e_old = (d[stat + "_proj"] - d[stat]).to_numpy() ** 2
    e_new = (d[new] - d[stat]).to_numpy() ** 2
    diff, se = G.paired(e_new, e_old, d.game_id.to_numpy())
    return {"n": int(len(d)), "rmse_current": float(np.sqrt(e_old.mean())), "rmse_new": float(np.sqrt(e_new.mean())),
            "mse_diff": diff, "se": se, "z": diff / se if se else 0.0}


def main():
    t, pinfo, games = G.table()
    cur_path = Path("/tmp/cfb_pproj_all.pkl")
    cur = pd.read_pickle(cur_path) if cur_path.exists() else PL.walk_forward(t)
    cur.to_pickle(cur_path)
    full = {}
    for stage, active in (("stage1_pre_set", "active"), ("stage2_appeared", "appeared")):
        print(f"== {stage}", flush=True)
        c = cur.copy()
        c["active"] = c[active].fillna(0).astype(int)
        full[stage] = run(c, pinfo, games)
    out = dict(full["stage2_appeared"])
    out["stage1_pre_set"] = {k: v for k, v in full["stage1_pre_set"].items() if k in ("pairs", "lines_2025", "applied_pairs")}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("apply (stage 2):", out["applied_pairs"])


def run(cur: pd.DataFrame, pinfo, games) -> dict:
    print("ratings ...", flush=True)
    rt = D.ratings(cur)
    res = {"settings": {"half_life": D.HALF_LIFE, "last_season": D.LAST_SEASON, "shrink": D.SHRINK}, "pairs": {}}

    # rule 1: b on tuning, scored on holdout and tuning
    b_tune = D.fit(cur, rt, TUNE)
    new_t = D.apply(cur, rt, b_tune, pairs=list(b_tune))
    played = cur["played"].astype(bool) & (cur["appeared"] == 1)
    for (grp, stat), b in b_tune.items():
        m = played & D._eligible(cur, grp) & cur[stat].notna()
        m &= (cur[stat + "_p"] >= FLOORS[stat]) | (cur[stat + "_proj"] >= FLOORS[stat])
        d = cur[m][["game_id", "season", stat, stat + "_proj"]].assign(new=new_t.loc[m, stat + "_proj"].to_numpy())
        r = {"b_tuning": b, "tuning 2017-19": paired_rmse(d[d.season.isin(TUNE)], stat, "new"),
             "holdout 2021+": paired_rmse(d[d.season >= 2021], stat, "new")}
        r["rule1_pass"] = bool(r["holdout 2021+"]["z"] <= -2 and r["tuning 2017-19"]["mse_diff"] <= 0)
        res["pairs"][f"{grp}|{stat}"] = r
        print(f"{grp:2s} {stat:10s} b={b:+.3f} tuning {r['tuning 2017-19']['rmse_current']:.3f}->{r['tuning 2017-19']['rmse_new']:.3f} "
              f"holdout {r['holdout 2021+']['rmse_current']:.3f}->{r['holdout 2021+']['rmse_new']:.3f} z {r['holdout 2021+']['z']:+.2f}", flush=True)

    # rule 2: b on 2017-2024, the 2025 lines
    b_all = D.fit(cur, rt, range(2017, 2025))
    new_l = D.apply(cur, rt, b_all, pairs=list(b_all))
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    key = ["game_id", "player_id"]
    rows = R.player_frame(cur, pinfo)
    model = P.PropModel.load()
    for stat, vol in LINE_STATS.items():
        nl = new_l[key + [stat + "_proj", vol + "_proj"]].rename(columns={stat + "_proj": "new_proj", vol + "_proj": "new_vol"})
        rr = rows.merge(nl, on=key, how="left")
        main_ = s[~s["alt"] & (s["stat"] == stat)]
        w = main_.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
        w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
        w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
        base = w.merge(rr, on=["game_id", "nname"], how="inner")
        outs = {}
        for name, cols in (("current", None), ("new", ("new_proj", "new_vol"))):
            dd = base.copy()
            if cols:
                dd[stat + "_proj"], dd[vol + "_proj"] = dd[cols[0]], dd[cols[1]]
            x = R.model_chances(dd, model)
            x = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
            outs[name] = x
        idx = outs["current"].index.intersection(outs["new"].index)
        a, bb = outs["current"].loc[idx], outs["new"].loc[idx]
        hit = (a["y"] > a["line"]).astype(float).to_numpy()
        pa = (a["our_over"] / (a["our_over"] + a["our_under"])).to_numpy()
        pb = (bb["our_over"] / (bb["our_over"] + bb["our_under"])).to_numpy()
        diff, se = R.clustered_diff(pb, pa, hit, a.index.get_level_values(0).to_numpy())
        dbk, sebk = R.clustered_diff(pb, a["fair"].to_numpy(), hit, a.index.get_level_values(0).to_numpy())
        lines = {"props": int(len(idx)), "brier_books": PD.brier(a["fair"].to_numpy(), hit), "brier_current": PD.brier(pa, hit), "brier_new": PD.brier(pb, hit),
                 "new_minus_current": diff, "se": se, "z": diff / se if se else 0.0, "new_minus_books": dbk, "books_se": sebk}
        # per group
        lines["by_group"] = {}
        for grp, g in a.groupby("group"):
            ii = g.index
            h = (g["y"] > g["line"]).astype(float).to_numpy()
            p1 = (g["our_over"] / (g["our_over"] + g["our_under"])).to_numpy()
            p2 = (bb.loc[ii, "our_over"] / (bb.loc[ii, "our_over"] + bb.loc[ii, "our_under"])).to_numpy()
            dg, sg = R.clustered_diff(p2, p1, h, ii.get_level_values(0).to_numpy())
            lines["by_group"][grp] = {"props": int(len(g)), "new_minus_current": dg, "se": sg, "z": dg / sg if sg else 0.0}
        res.setdefault("lines_2025", {})[stat] = lines
        print(f"lines {stat}: {len(idx)} props, current {lines['brier_current']:.5f} new {lines['brier_new']:.5f} ({diff:+.5f} ± {se:.5f}, z {lines['z']:+.2f})", flush=True)
    res["b_2017_2024"] = {f"{k[0]}|{k[1]}": v for k, v in b_all.items()}
    # decision per pair
    applied = []
    for k, r in res["pairs"].items():
        grp, stat = k.split("|")
        ln = res["lines_2025"].get(stat, {}).get("by_group", {}).get(grp)
        r["lines_z"] = ln["z"] if ln else None
        r["apply"] = bool(r["rule1_pass"] and (ln is None or ln["new_minus_current"] <= 0))
        r["past_books"] = bool(ln and ln["z"] <= -2)
        if r["apply"]:
            applied.append(k)
    res["applied_pairs"] = applied
    print("apply:", applied)
    return res


if __name__ == "__main__":
    main()
