"""Pre-set test 2 of Nov 22, 2026: NFL tracking stats (Next Gen Stats) in the player projections.

    PFF_DATA_DIR=<pff-data clone> python experiments/tracking_stats_test.py

Rule written before running (Project doc claude/plan-three-untracked-stats-tests-2026-11-22.md).
Source: nflverse nextgen_stats (weekly rows; receiving only for a player with 5+ targets that week,
rushing only with 10+ carries; rush yards over expected from 2018, the rest from 2016).
Inputs, as of each game from his earlier games with a tracking row (players.asof, the board's player
decay), zero with no history. Each is entered in the stat's calibration (players.CAL_EXTRA, raw
projection x column) and as a feature of the boosted step; only the tested stat's projection is taken
from the candidate run (as experiments/air_yards_wopr_test.py).

2a rushing yards, RB. Tuning 2019-21, holdout 2022-25; league constants 2018-21.
  A  ryoe          rush yards over expected per carry, shrunk to zero by 50 carries
  B  ryoe + box8   plus his share of carries against 8+ in the box minus the league share (50 carries)
  Measures (7): rushing-yards RMSE, RBs who played with 3+ projected carries, game-day and bet-time,
  tuning and holdout (4); Brier at the 2023-25 DK/FD rushing-yards main lines, closing with game-day
  and pricing snapshot with bet-time (2); Brier on the 2025 alternate rushing-yards rungs the books
  read at 65-95%, pricing snapshot (1).
2b receiving yards, WR and TE. Tuning 2017-18, holdout 2019-25; league constants 2016-18.
  A  sep           average separation / his position's league average - 1, shrunk by 30 targets
  B  sep + yacoe   plus yards after catch over expected per reception, shrunk to zero by 20 receptions
  Measures (7): receiving-yards RMSE, WR/TE who played with 1+ projected reception, game-day and
  bet-time, tuning and holdout (4); Brier at the 2023-25 receiving-yards main lines, closing and
  pricing (2); Brier on the 2025 alternate receiving-yards rungs at 65-95% (1).
Rule (each on its own): the version is chosen on tuning alone (better than the current model on both
  tuning RMSE measures, then the lowest game-day tuning RMSE). Kept only if better on every measure
  and each holdout RMSE gain >= 1 paired SE (clustered by game). Mixed, or under 1 SE: a trade-off for
  Ryan. No version better on both tuning measures: nothing is carried to the holdout.

Writes experiments/out/tracking_stats_test.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import config as C, players as PL, propbackfill as B  # noqa: E402
import air_yards_wopr_test as AY  # noqa: E402

OUT = HERE / "out" / "tracking_stats_test.json"      # each part merges into this file; run the parts one after the other
TABLES = C.RAW / "air_yards_wopr_tables.pkl"
NGS = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_{kind}.parquet"
TESTS = {
    "2a rushing yards (RB)": {"stat": "rush_yds", "groups": ("RB",), "filter": ("carries_proj", 3), "tune": (2019, 2021), "hold": (2022, 2025),
                              "versions": {"A": ["ryoe"], "B": ["ryoe", "box8"]}, "alt": "player_rush_yds_alternate"},
    "2b receiving yards (WR, TE)": {"stat": "rec_yds", "groups": ("WR", "TE"), "filter": ("receptions_proj", 1), "tune": (2017, 2018), "hold": (2019, 2025),
                                    "versions": {"A": ["sep"], "B": ["sep", "yacoe"]}, "alt": "player_reception_yds_alternate"},
}


def ngs(kind: str) -> pd.DataFrame:
    from nflmodel.data import _download
    f = C.RAW / f"ngs_{kind}.parquet"
    if not f.exists():
        _download(NGS.format(kind=kind), f)
    x = pd.read_parquet(f)
    x = x[(x["week"] > 0) & (x["season_type"] == "REG")].rename(columns={"player_gsis_id": "player_id"})
    x["player_id"] = x["player_id"].astype(str)
    return x.drop_duplicates(["season", "week", "player_id"])


def inputs(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a = t[["game_id", "player_id", "season", "week", "t", "played", "active", "group"]].copy()
    a["player_id"] = a["player_id"].astype(str)
    r, c = ngs("rushing"), ngs("receiving")
    r["n_att"] = r["rush_attempts"].where(r["rush_yards_over_expected"].notna())
    r["ryoe_tot"] = r["rush_yards_over_expected"]
    r["box_n"] = r["percent_attempts_gte_eight_defenders"] / 100 * r["rush_attempts"]
    r["box_att"] = r["rush_attempts"]
    c["sep_tot"] = c["avg_separation"] * c["targets"]
    c["sep_n"] = c["targets"]
    c["yac_tot"] = c["avg_yac_above_expectation"] * c["receptions"]
    c["yac_n"] = c["receptions"].where(c["avg_yac_above_expectation"].notna())
    a = a.merge(r[["season", "week", "player_id", "n_att", "ryoe_tot", "box_n", "box_att"]], on=["season", "week", "player_id"], how="left") \
         .merge(c[["season", "week", "player_id", "sep_tot", "sep_n", "yac_tot", "yac_n"]], on=["season", "week", "player_id"], how="left")
    use = (a["active"] == 1) & a["played"].astype(bool)
    num = ["n_att", "ryoe_tot", "box_n", "box_att", "sep_tot", "sep_n", "yac_tot", "yac_n"]
    for col in num:
        a[col] = a[col].where(use)
    lg_r = a[use & a["season"].between(2018, 2021) & (a["group"] == "RB")]
    box_lg = float(lg_r["box_n"].sum() / lg_r["box_att"].sum())
    lg_c = a[use & a["season"].between(2016, 2018) & a["group"].isin(["WR", "TE"])]
    sep_lg = (lg_c.groupby("group")["sep_tot"].sum() / lg_c.groupby("group")["sep_n"].sum())
    f = PL.asof(a, "player_id", num, None, PL.PLAYER_HL)
    g = lambda col: f[col + "_sum"].to_numpy(float)
    rb, wt = (f["group"] == "RB").to_numpy(), f["group"].isin(["WR", "TE"]).to_numpy()
    s = f["group"].map(sep_lg).to_numpy(float)
    out = f[["game_id", "player_id"]].copy()
    out["ryoe"] = np.where(rb, g("ryoe_tot") / (g("n_att") + 50), 0.0)
    out["box8"] = np.where(rb, (g("box_n") + 50 * box_lg) / (g("box_att") + 50) - box_lg, 0.0)
    out["sep"] = np.where(wt, ((g("sep_tot") + 30 * s) / (g("sep_n") + 30)) / s - 1, 0.0)
    out["yacoe"] = np.where(wt, g("yac_tot") / (g("yac_n") + 20), 0.0)
    out[["ryoe", "box8", "sep", "yacoe"]] = out[["ryoe", "box8", "sep", "yacoe"]].fillna(0.0)
    info = {"league_8_in_box_share_2018_21": round(box_lg, 4), "league_separation_2016_18": sep_lg.round(3).to_dict(),
            "tracking_rows_matched": {"rushing": int(a["box_att"].notna().sum()), "of": int(len(r)), "receiving": int(a["sep_n"].notna().sum()), "of_": int(len(c))}}
    return out, info


def rmse(base, cand, st, groups, filt, per) -> dict:
    k = (base["played"].astype(bool) & (base["active"] == 1) & base["group"].isin(groups) & (base[filt[0]] >= filt[1])
         & base["season"].between(*per) & base[st].notna()).to_numpy()
    y = base[st].to_numpy(float)[k]
    e0, e1 = (base[st + "_proj"].to_numpy(float)[k] - y) ** 2, (cand[st + "_proj"].to_numpy(float)[k] - y) ** 2
    r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
    cl = pd.Series(e1 - e0).groupby(base["game_id"].to_numpy()[k]).sum()
    return {"n": int(k.sum()), "now": round(r0, 4), "candidate": round(r1, 4), "diff": round(r1 - r0, 5),
            "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / k.sum() / (2 * r0)), 5)}


def floor_rungs(base, cand, market, groups) -> dict:
    from nfl_pff_prop_inputs_test import rungs
    key = ["game_id", "player_id", "stat", "line", "book"]
    got = {"now": [], "cand": []}
    snaps = B.load(2025, "flag")
    for k, t, m in zip(("now", "cand"), (base, cand), AY._models(base, cand, 2025)):
        r = rungs(t, m, snaps, 2025, market)
        got[k].append(r[r["group"].isin(groups)])
    a, b = pd.concat(got["now"], ignore_index=True), pd.concat(got["cand"], ignore_index=True)
    a, b = a[a["y"] != a["line"]].drop_duplicates(key), b[b["y"] != b["line"]].drop_duplicates(key)
    j = a[key + ["y", "our_over", "books"]].merge(b[key + ["our_over"]], on=key, suffixes=("_c", "_n"))
    hit = (j["y"] > j["line"]).astype(float)
    d = (j["our_over_n"] - hit) ** 2 - (j["our_over_c"] - hit) ** 2
    cl = d.groupby(j["game_id"]).sum()
    return {"n": int(len(j)), "now": round(float(((j["our_over_c"] - hit) ** 2).mean()), 5),
            "candidate": round(float(((j["our_over_n"] - hit) ** 2).mean()), 5), "books": round(float(((j["books"] - hit) ** 2).mean()), 5),
            "diff": round(float(d.mean()), 6), "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(j)), 6)}


def main():
    T = pickle.load(open(TABLES, "rb"))
    R = T["R"]
    feats, info = inputs(T["gameday"])
    cols = ["ryoe", "box8", "sep", "yacoe"]
    for k in ("gameday", "bettime"):
        T[k] = T[k].assign(player_id=T[k]["player_id"].astype(str)).merge(feats, on=["game_id", "player_id"], how="left")
        T[k][cols] = T[k][cols].fillna(0.0)
    g = T["gameday"]
    w = g[g.played.astype(bool) & (g.active == 1) & g.season.between(2019, 2025)]
    info["share_with_input_2019_25"] = {"ryoe (RB)": round(float((w[w.group == "RB"]["ryoe"] != 0).mean()), 3),
                                        "sep (WR/TE)": round(float((w[w.group.isin(["WR", "TE"])]["sep"] != 0).mean()), 3)}
    info["spread_2019_25"] = {c: round(float(w[w.group.isin(["RB"] if c in ("ryoe", "box8") else ["WR", "TE"])][c].std()), 4) for c in cols}
    res = {"inputs": info}
    if OUT.exists():
        res.update({k: v for k, v in json.loads(OUT.read_text()).items() if k != "inputs"})
    print(json.dumps(info), flush=True)
    which = sys.argv[1:] or ["2a", "2b"]
    now = {k: AY.project(T[k], R, {}, []) for k in ("gameday", "bettime")}
    print("current projections done", flush=True)
    for name, cfg in TESTS.items():
        if name[:2] not in which:
            continue
        st, grp = cfg["stat"], cfg["groups"]
        r, cands = {"tuning": {}}, {}
        for v, vc in cfg["versions"].items():
            cands[v] = {k: AY.candidate(now[k], AY.project(T[k], R, {st: vc}, vc), (st,)) for k in ("gameday", "bettime")}
            r["tuning"][v] = {k: rmse(now[k], cands[v][k], st, grp, cfg["filter"], cfg["tune"]) for k in ("gameday", "bettime")}
            print(name, v, json.dumps(r["tuning"][v]), flush=True)
        ok = [v for v in cfg["versions"] if all(m["diff"] < 0 for m in r["tuning"][v].values())]
        r["better_on_both_tuning_measures"] = ok
        if not ok:
            r["verdict"] = "No version better on both tuning measures: nothing carried to the holdout, nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["gameday"]["candidate"])
            c = cands[pick]
            r["chosen"] = f"{pick} ({' + '.join(cfg['versions'][pick])})"
            r["holdout"] = {k: rmse(now[k], c[k], st, grp, cfg["filter"], cfg["hold"]) for k in ("gameday", "bettime")}
            r["brier"] = {"closing lines, game-day": AY.brier(now["gameday"], c["gameday"], "close", (st,), grp),
                          "pricing snapshot, bet-time": AY.brier(now["bettime"], c["bettime"], "flag", (st,), grp),
                          "floor rungs 65-95%, 2025, pricing snapshot": floor_rungs(now["bettime"], c["bettime"], cfg["alt"], grp)}
            ms = list(r["tuning"][pick].values()) + list(r["holdout"].values()) + list(r["brier"].values())
            better = [m["diff"] < 0 for m in ms]
            se_ok = all(-m["diff"] >= m["se"] for m in r["holdout"].values())
            r["better_on"] = f"{sum(better)} of {len(better)}"
            r["kept"] = bool(all(better) and se_ok)
            r["verdict"] = "Kept (provisional)." if r["kept"] else ("Better on every measure but a holdout gain is under 1 SE: a trade-off for Ryan."
                                                                     if all(better) else "Mixed: a trade-off for Ryan, not kept by the rule.")
        res[name] = r
        print(name, json.dumps(r, default=str), flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
