"""Pre-set test: the untested PFF college tables in the player projections (Nov 13, 2026).

    python experiments/cfb_pff_tables_test.py [--stage2-only R1,T6]   (PFF data at ../pff-data)

Rule written before running (Project doc claude/plan-cfb-pff-tables-test-2026-11-13.md):

14 candidates, each alone, as of each game from earlier games only (8-game half-life, last season at
60%, nothing older; 0 with no history), entering the chain's last calibration step as CAL_EXTRA does
(raw projection x feature). League constants from 2017-2019 only.
  Player rates, shrunk to the league rate by a pseudo-count, as rate / league - 1:
    R1 yards per route run (100 routes)           -> rec_yds
    R2 targets per route (100 routes)             -> receptions, rec_yds
    R3 average depth of target (20 targets)       -> rec_yds
    R4 slot rate (100 snaps; slot rate - league)  -> receptions, rec_yds
    U1 yards after contact per carry (40)         -> rush_yds
    U2 avoided tackles per carry (40)             -> rush_yds
    P1 PFF passing grade (50 dropbacks; (g - league) / spread) -> pass_yds
    P2 QB average depth of target (40 attempts)   -> pass_yds
  Team grades (team_overview, per game; 2 games prior; (g - league) / spread):
    T1 own run block -> carries, rush_yds      T2 opponent run defense -> carries, rush_yds
    T3 own pass block -> pass_yds              T4 opponent pass rush -> pass_yds
    T5 opponent coverage -> pass_yds           T6 opponent coverage -> receptions, rec_yds
Stage 1 (walk-forward): RMSE new minus current, paired, clustered by game, tuning 2017-19 and holdout
  2021-25, on every stat the candidate enters (receiving WR/TE/RB chain or projection >= 15 yards /
  1.5 receptions; rushing RBs >= 20 yards / 4 carries; passing QBs >= 100 yards, standing in for a
  projected start). Passes if better on every stat in both periods, holdout by 2+ SE on each.
Stage 2 (passers): Brier at the DK/FD main lines (receiving, rushing, passing yards), new minus
  current, college widths, clustered by game; 2024 and 2025 lines, game-day and bet-time projections
  (active = expected to play). Floor rungs and model-minus-books reported.
Kept only if better on every measure (holdout 2+ SE). Two or more passers in one chain: their
combination is run too and replaces the best single only if better on every measure. Mixed results
go to Ryan. Writes scores only (experiments/out/cfb_pff_tables_test.json), never PFF rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from cfbmodel import players as PL, pff_data, pff_roles  # noqa: E402
from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_dvp_test as DV  # noqa: E402
import cfb_pff_role_test as T  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = HERE / "out" / "cfb_pff_tables_test.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
HL, CARRY = 8.0, 0.6
LEAGUE = [2017, 2018, 2019]
TUNE, HOLD = [2017, 2018, 2019], [2021, 2022, 2023, 2024, 2025]
ENTERS = {"R1": ["rec_yds"], "R2": ["receptions", "rec_yds"], "R3": ["rec_yds"], "R4": ["receptions", "rec_yds"],
          "U1": ["rush_yds"], "U2": ["rush_yds"], "P1": ["pass_yds"], "P2": ["pass_yds"],
          "T1": ["carries", "rush_yds"], "T2": ["carries", "rush_yds"], "T3": ["pass_yds"], "T4": ["pass_yds"],
          "T5": ["pass_yds"], "T6": ["receptions", "rec_yds"]}
SCORE = {"rec_yds": (["WR", "TE", "RB"], 15.0), "receptions": (["WR", "TE", "RB"], 1.5),
         "rush_yds": (["RB"], 20.0), "carries": (["RB"], 4.0), "pass_yds": (["QB"], 100.0)}
VOL = {"rec_yds": "receptions", "rush_yds": "carries", "pass_yds": "pass_att"}
LINE_STATS = ("rec_yds", "rush_yds", "pass_yds")


# ------------------------------------------------------------------ as-of decayed sums

def decayed(t: pd.DataFrame, key: str, hist: pd.DataFrame, hkey: str, cols: list[str]) -> np.ndarray:
    """For each row of t, the decayed sums of hist[cols] over the key's earlier games (this season
    and last, last at CARRY). Returns an array (rows, len(cols))."""
    h = hist.sort_values([hkey, "t"])
    keyed = {k: (g["t"].to_numpy(), g["season"].to_numpy(), g[cols].to_numpy(float)) for k, g in h.groupby(hkey)}
    out = np.zeros((len(t), len(cols)))
    tt, kk, ss = t["t"].to_numpy(), t[key].to_numpy(), t["season"].to_numpy()
    for i in range(len(t)):
        k = keyed.get(kk[i])
        if k is None:
            continue
        ht, hs, v = k
        sel = (ht < tt[i]) & (hs >= ss[i] - 1)
        if not sel.any():
            continue
        j = np.arange(sel.sum())[::-1]
        w = 0.5 ** (j / HL) * np.where(hs[sel] == ss[i], 1.0, CARRY)
        out[i] = (w[:, None] * np.nan_to_num(v[sel])).sum(axis=0)
    return out


def on_board(table: str, t: pd.DataFrame, info: pd.DataFrame, cols: list[str], seasons) -> pd.DataFrame:
    x = pff_data.load(table, seasons)
    x = x[["season", "week", "player", "player_id", "franchise_id"] + [c for c in cols if c in x]].copy()
    x["nname"] = x["player"].map(pff_roles._norm)
    m = pff_roles.on_board(x, t, info)
    m = m.merge(t[["game_id", "t"]].drop_duplicates("game_id"), on="game_id")
    m["board_id"] = m["board_id"].astype(np.int64)
    return m


def features(t: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(int(s) for s in t["season"].unique() if s >= 2017)
    t = t.copy()
    meta = {}
    # receiving
    rc = on_board("receiving_summary", t, info, ["yards", "routes", "targets", "avg_depth_of_target", "slot_snaps", "wide_snaps", "inline_snaps"], seasons)
    rc["adot_t"] = rc["avg_depth_of_target"].fillna(0) * rc["targets"].fillna(0)
    rc["snaps3"] = rc[["slot_snaps", "wide_snaps", "inline_snaps"]].fillna(0).sum(axis=1)
    lg = rc[rc.season.isin(LEAGUE)]
    lg_yprr = lg.yards.sum() / lg.routes.sum(); lg_tprr = lg.targets.sum() / lg.routes.sum()
    lg_adot = lg.adot_t.sum() / lg.targets.sum(); lg_slot = lg.slot_snaps.sum() / lg.snaps3.sum()
    s = decayed(t, "player_id", rc, "board_id", ["yards", "routes", "targets", "adot_t", "slot_snaps", "snaps3"])
    t["R1"] = ((s[:, 0] + 100 * lg_yprr) / (s[:, 1] + 100)) / lg_yprr - 1
    t["R2"] = ((s[:, 2] + 100 * lg_tprr) / (s[:, 1] + 100)) / lg_tprr - 1
    t["R3"] = ((s[:, 3] + 20 * lg_adot) / (s[:, 2] + 20)) / lg_adot - 1
    t["R4"] = (s[:, 4] + 100 * lg_slot) / (s[:, 5] + 100) - lg_slot
    for c, has in (("R1", s[:, 1]), ("R2", s[:, 1]), ("R3", s[:, 2]), ("R4", s[:, 5])):
        t.loc[has <= 0, c] = 0.0
    meta["receiving"] = {"yprr": lg_yprr, "tprr": lg_tprr, "adot": lg_adot, "slot": lg_slot}
    # rushing
    ru = on_board("rushing_summary", t, info, ["attempts", "yards_after_contact", "avoided_tackles"], seasons)
    lg = ru[ru.season.isin(LEAGUE)]
    lg_yco = lg.yards_after_contact.sum() / lg.attempts.sum(); lg_av = lg.avoided_tackles.sum() / lg.attempts.sum()
    s = decayed(t, "player_id", ru, "board_id", ["yards_after_contact", "avoided_tackles", "attempts"])
    t["U1"] = np.where(s[:, 2] > 0, ((s[:, 0] + 40 * lg_yco) / (s[:, 2] + 40)) / lg_yco - 1, 0.0)
    t["U2"] = np.where(s[:, 2] > 0, ((s[:, 1] + 40 * lg_av) / (s[:, 2] + 40)) / lg_av - 1, 0.0)
    meta["rushing"] = {"yco": lg_yco, "avoided": lg_av}
    # passing
    pa = on_board("passing_summary", t, info, ["grades_pass", "dropbacks", "avg_depth_of_target", "attempts"], seasons)
    pa["g_db"] = pa["grades_pass"].fillna(0) * pa["dropbacks"].fillna(0)
    pa["adot_a"] = pa["avg_depth_of_target"].fillna(0) * pa["attempts"].fillna(0)
    lg = pa[pa.season.isin(LEAGUE)]
    q = lg[lg.dropbacks >= 15]
    lg_g, sd_g = float(np.average(q.grades_pass, weights=q.dropbacks)), float(q.grades_pass.std())
    lg_ad = lg.adot_a.sum() / lg.attempts.sum()
    s = decayed(t, "player_id", pa, "board_id", ["g_db", "dropbacks", "adot_a", "attempts"])
    t["P1"] = np.where(s[:, 1] > 0, (((s[:, 0] + 50 * lg_g) / (s[:, 1] + 50)) - lg_g) / sd_g, 0.0)
    t["P2"] = np.where(s[:, 3] > 0, ((s[:, 2] + 40 * lg_ad) / (s[:, 3] + 40)) / lg_ad - 1, 0.0)
    meta["passing"] = {"grade": lg_g, "grade_sd": sd_g, "adot": lg_ad}
    # team grades: franchise -> board team per season (the receiving mapping), week -> game
    fm = rc[["season", "franchise_id", "team"]].drop_duplicates(["season", "franchise_id"])
    to = pff_data.load("team_overview", seasons)
    gcols = ["grades_run_block", "grades_run_defense", "grades_pass_block", "grades_pass_rush_defense", "grades_coverage_defense"]
    to = to[["season", "week", "franchise_id"] + gcols].merge(fm, on=["season", "franchise_id"])
    gm = t[t["played"].astype(bool)][["game_id", "team", "season", "week", "t"]].drop_duplicates(["team", "season", "week"])
    to = to.merge(gm, on=["team", "season", "week"])
    to["one"] = 1.0
    lg = to[to.season.isin(LEAGUE)]
    tg = t[["game_id", "team", "opp", "season", "t"]].drop_duplicates(["game_id", "team"])
    st = decayed(tg, "team", to, "team", gcols + ["one"])
    so = decayed(tg.assign(team=tg["opp"]), "team", to, "team", gcols + ["one"])
    def z(sums, i, c):
        m, sd = float(lg[c].mean()), float(lg[c].std())
        return np.where(sums[:, -1] > 0, (((sums[:, i] + 2 * m) / (sums[:, -1] + 2)) - m) / sd, 0.0)
    tg = tg.assign(T1=z(st, 0, gcols[0]), T2=z(so, 1, gcols[1]), T3=z(st, 2, gcols[2]), T4=z(so, 3, gcols[3]), T5=z(so, 4, gcols[4]))
    tg["T6"] = tg["T5"]
    t = t.merge(tg[["game_id", "team", "T1", "T2", "T3", "T4", "T5", "T6"]], on=["game_id", "team"], how="left")
    for c in ("T1", "T2", "T3", "T4", "T5", "T6"):
        t[c] = t[c].fillna(0.0)
    meta["coverage_2025"] = {c: float((t.loc[(t.season == 2025) & (t.appeared == 1), c] != 0).mean()) for c in ENTERS}
    return t, meta


# ------------------------------------------------------------------ chains and scores

def chain(t, cands: list[str], first_test=None):
    extra = dict(PL.CAL_EXTRA)
    for c in cands:
        for st in ENTERS[c]:
            extra[st] = list(extra.get(st, [])) + [c]
    saved = PL.CAL_EXTRA
    PL.CAL_EXTRA = extra
    try:
        return PL.walk_forward(t) if first_test is None else PL.walk_forward(t, first_test=first_test)
    finally:
        PL.CAL_EXTRA = saved


def stage1(cur, new, stats) -> dict:
    out = {}
    for st in stats:
        grp, floor = SCORE[st]
        out[st] = T.rmse(cur, new, st, grp, floor, {"tuning 2017-19": TUNE, "holdout 2021-25": HOLD})
    ok = all(v["tuning 2017-19"]["mse_diff"] < 0 and v["holdout 2021-25"]["z"] <= -2 for v in out.values())
    return {"stats": out, "pass": bool(ok)}


def lines(cur, new, pinfo, games, stat, season) -> dict:
    R.set_season(season)
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    main = s[~s["market"].str.endswith("_alternate") & (s["stat"] == stat)]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    key = ["game_id", "player_id"]
    vol = VOL[stat]
    rows = R.player_frame(cur, pinfo).merge(new[key + [stat + "_proj", vol + "_proj"]].rename(columns={stat + "_proj": "n_s", vol + "_proj": "n_v"}), on=key, how="left")
    base = w.merge(rows, on=["game_id", "nname"], how="inner")
    model = P.PropModel.load(PARAMS)
    outs = {}
    for name in ("current", "new"):
        dd = base.copy()
        if name == "new":
            dd[stat + "_proj"], dd[vol + "_proj"] = dd["n_s"], dd["n_v"]
        x = R.model_chances(dd, model)
        outs[name] = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
    idx = outs["current"].index.intersection(outs["new"].index)
    a, b = outs["current"].loc[idx], outs["new"].loc[idx]
    hit = (a["y"] > a["line"]).astype(float).to_numpy()
    pa = (a["our_over"] / (a["our_over"] + a["our_under"])).to_numpy()
    pb = (b["our_over"] / (b["our_over"] + b["our_under"])).to_numpy()
    g = a.index.get_level_values(0).to_numpy()
    d, se = R.clustered_diff(pb, pa, hit, g)
    dbk, sbk = R.clustered_diff(pb, a["fair"].to_numpy(), hit, g)
    return {"props": int(len(idx)), "new_minus_current": round(d, 6), "se": round(se, 6), "z": round(d / se, 2) if se else None,
            "new_minus_books": round(dbk, 5), "books_se": round(sbk, 5)}


def stage2(t, tb, cur_gd, cur_bt, cands, pinfo, games) -> dict:
    new_gd = chain(t, cands)
    new_bt = chain(tb, cands)
    stats = sorted({st for c in cands for st in ENTERS[c]} & set(LINE_STATS))
    out = {}
    for st in stats:
        for season in (2024, 2025):
            out[f"{st} {season} game-day"] = lines(cur_gd, new_gd, pinfo, games, st, season)
            out[f"{st} {season} bet-time"] = lines(cur_bt, new_bt, pinfo, games, st, season)
            out[f"{st} {season} floor rungs"] = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                                                 for k, v in T.floor_rungs(cur_gd, new_gd, pinfo, games, st, VOL[st]).items()}
    ok = all(v["new_minus_current"] < 0 for k, v in out.items() if "floor" not in k)
    return {"lines": out, "pass": bool(ok), "_new_gd": new_gd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-only", default="", help="comma-separated candidates to send straight to stage 2 (re-runs)")
    a = ap.parse_args()
    t0, pinfo, games = G.table()
    if not (t0.get("route_share", 0) > 0).any():
        sys.exit("no PFF data (check out rhyman7/pff-data at ../pff-data)")
    t, meta = features(t0, pinfo)
    print("coverage 2025", {k: round(v, 2) for k, v in meta["coverage_2025"].items()}, flush=True)
    res = {"meta": {k: v for k, v in meta.items()}, "stage1": {}, "stage2": {}}
    cur = PL.walk_forward(t)
    passers = [c for c in a.stage2_only.split(",") if c]
    if not passers:
        for c in ENTERS:
            new = chain(t, [c])
            r = stage1(cur, new, ENTERS[c])
            res["stage1"][c] = r
            zs = {st: (round(v["tuning 2017-19"]["z"], 2), round(v["holdout 2021-25"]["z"], 2)) for st, v in r["stats"].items()}
            print(c, "pass" if r["pass"] else "-", zs, flush=True)
        passers = [c for c, r in res["stage1"].items() if r["pass"]]
        OUT.write_text(json.dumps(res, indent=1, default=float))
    if passers:
        tb = t.copy()
        m = tb["played"].astype(bool)
        tb.loc[m, "active"] = tb.loc[m, "expected"].astype(int)
        cur_bt = PL.walk_forward(tb)
        for c in passers:
            r = stage2(t, tb, cur, cur_bt, [c], pinfo, games)
            r.pop("_new_gd")
            res["stage2"][c] = r
            print(c, "stage 2", "pass" if r["pass"] else "-", {k: v.get("new_minus_current") for k, v in r["lines"].items()}, flush=True)
        kept = [c for c in passers if res["stage2"][c]["pass"]]
        chains = {}
        for c in kept:
            chains.setdefault(tuple(sorted(ENTERS[c])), []).append(c)
        for k, cs in chains.items():
            if len(cs) > 1:
                comb = stage1(cur, chain(t, cs), sorted({st for c in cs for st in ENTERS[c]}))
                r2 = stage2(t, tb, cur, cur_bt, cs, pinfo, games)
                r2.pop("_new_gd")
                res.setdefault("combinations", {})["+".join(cs)] = {"stage1": comb, "stage2": r2}
        res["kept"] = kept
    else:
        res["kept"] = []
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("kept:", res["kept"])


if __name__ == "__main__":
    main()
