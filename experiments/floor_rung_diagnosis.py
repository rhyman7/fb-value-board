"""Where the model loses to the books on the NFL floor rungs: a diagnosis, no model change.

    PFF_DATA_DIR=<pff-data clone> python experiments/floor_rung_diagnosis.py

Plan written before running (Project doc claude/plan-floor-rung-diagnosis-2026-11-18.md).

Rungs: DK/FD alternate passing, rushing and receiving yards, over only, the books' reading
(parlays.RUNG_CAL on the raw implied chance) 65-95%.
  2025 (where the cuts are read): pricing snapshot, the model's chance from bet-time projections with
    the prop model fitted on earlier seasons (nfl_pff_prop_inputs_test.rungs, the Nov 6 definition).
  2026 (the check): the live prop log's pricing-snapshot rungs with the chance the board logged,
    graded rows only. The log starts in week 8, so the check is weeks 8-10, not 1-10.
Step 1: each side's Brier split into calibration and sorting (Murphy, 5-point bands of the stated
  chance), by stat; and the model re-read by an isotonic fit on 2025 weeks 1-9, scored on 2025 weeks
  10-18 and on 2026.
Step 2: ten cuts, and no others: stat; position; projection thirds within the stat; rung / main line
  (under 50%, 50-70%, 70-90%, 90%+); the books' reading (65-75, 75-85, 85-95); disagreement (model
  5+ points above the books, within 5, 5+ below); injury report (listed Questionable or limited in
  practice, or not); role change (|rolechange| >= 0.10, or steady; quarterbacks are steady); game
  script (the team's projected margin +7 or more, within 7, -7 or worse; projected total above or
  below 44.5: the model's game numbers, the table carries no market spread); weeks 1-6, 7-12, 13-18.
  A cell is a lead only with 300+ rungs from 30+ games on 2025, the model's Brier no worse than the
  books', the rungs clearing at least as often as the books read, and the same two signs on 2026.
Step 3 (reported): rungs the model has 5+ points above the books' reading, at the better DK/FD
  price: return per $1 and price closing line value; a three-leg, 25%-boost parlay of such rungs
  worked out from the average leg (legs treated as independent).
Standard errors clustered by game.

Writes experiments/out/floor_rung_diagnosis.json.
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

from nflmodel import config as C, odds as O, parlays, propbackfill as B, propmarket as PM, props as P, rolechange  # noqa: E402

OUT = HERE / "out" / "floor_rung_diagnosis.json"
ALTS = {"pass_yds": "player_pass_yds_alternate", "rush_yds": "player_rush_yds_alternate", "rec_yds": "player_reception_yds_alternate"}
MAIN = {v.removesuffix("_alternate"): k for k, v in ALTS.items()}
ROLE = {"rush_yds": "role_chg_car", "rec_yds": "role_chg_rec"}
XS, YS = zip(*parlays.RUNG_CAL)


def reading(price) -> np.ndarray:
    return np.interp(O.implied_prob(np.asarray(price, float)), XS, YS)


def dec(price) -> np.ndarray:
    p = np.asarray(price, float)
    return np.where(p > 0, 1 + p / 100, 1 + 100 / np.abs(p))


def projections() -> pd.DataFrame:
    import air_yards_wopr_test as AW
    import wopr_standing_test as W
    p = W.current()["bettime"]
    rc = rolechange.add(AW.tables()["bettime"], "targets").drop_duplicates(["game_id", "player_id"])
    return p.merge(rc[["game_id", "player_id", *rolechange.COLS]], on=["game_id", "player_id"], how="left").fillna({c: 0.0 for c in rolechange.COLS})


def rungs_2025(p: pd.DataFrame) -> pd.DataFrame:
    from nfl_pff_prop_inputs_test import rungs
    from props_calibration import FAMILY
    season = 2025
    m = P.PropModel(FAMILY).fit(p[(p.season < season) & (p.season >= 2013)])
    saved = P.PropModel.load()
    if saved is not None:
        m.iso = saved.iso
    flag, close = B.load(season, "flag"), B.load(season, "close")
    out = []
    for stat, mk in ALTS.items():
        r = rungs(p, m, flag, season, mk)
        if r.empty:
            continue
        r["proj"] = P.mean(r, stat)
        ml = flag[flag["market"] == mk.removesuffix("_alternate")].copy()
        ml["nname"] = ml["player"].map(PM.norm_name)
        r = r.merge(ml.groupby(["game_id", "nname"])["line"].median().rename("main_line").reset_index(), on=["game_id", "nname"], how="left")
        c = close[(close["market"] == mk) & (close["side"].str.lower() == "over")].copy()
        c["nname"] = c["player"].map(PM.norm_name)
        c = c.groupby(["game_id", "nname", "line", "book"], as_index=False)["price"].last().rename(columns={"price": "close_price"})
        out.append(r.merge(c, on=["game_id", "nname", "line", "book"], how="left"))
    x = pd.concat(out, ignore_index=True)
    x = x[x["y"] != x["line"]].drop_duplicates(["game_id", "player_id", "stat", "line", "book"]).copy()
    x["hit"], x["model"] = (x["y"] > x["line"]).astype(float), x["our_over"]
    return x


def rungs_2026(p: pd.DataFrame) -> pd.DataFrame:
    log = pd.read_csv(C.DATA / "props" / "prop_log.csv", low_memory=False)
    res = pd.read_csv(C.DATA / "props" / "prop_results.csv", low_memory=False).drop_duplicates("row_id", keep="last")
    main = log[(log["snapshot"] == "flag") & log["market"].isin(MAIN)].groupby(["game_id", "player_id", "stat"])["line"].median().rename("main_line").reset_index()
    a = log[(log["snapshot"] == "flag") & log["market"].isin(ALTS.values()) & (log["side"] == "over")].merge(res, on="row_id", how="inner")
    a = a[a["result"].isin(["win", "loss"]) & a["actual"].notna() & a["our_p"].notna()].copy()
    a["books"], a["model"], a["hit"] = reading(a["price"]), a["our_p"].astype(float), (a["actual"] > a["line"]).astype(float)
    a = a[a["books"].between(0.65, 0.95)].merge(main, on=["game_id", "player_id", "stat"], how="left")
    cols = ["game_id", "player_id", "group", "margin", "proj_total", "report_lvl", "practice_lvl", *rolechange.COLS]
    a = a.drop(columns=[c for c in cols[2:] if c in a.columns]).merge(p[p.season == 2026][cols].drop_duplicates(["game_id", "player_id"]), on=["game_id", "player_id"], how="left")
    a["group"] = a["group"].fillna(a["pos"])
    return a.rename(columns={"close_price": "close_price"})


def cuts(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["c_stat"] = x["stat"]
    x["c_position"] = x["group"]
    x["c_projection"] = x.groupby("stat")["proj"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low third", "middle third", "high third"])).astype(str)
    r = x["line"] / x["main_line"]
    x["c_rung_vs_main"] = np.select([r.isna(), r < 0.5, r < 0.7, r < 0.9], ["no main line", "under 50%", "50-70%", "70-90%"], "90%+")
    x["c_books_reading"] = np.select([x["books"] < 0.75, x["books"] < 0.85], ["65-75%", "75-85%"], "85-95%")
    d = x["model"] - x["books"]
    x["c_disagreement"] = np.select([d >= 0.05, d <= -0.05], ["model 5+ above", "model 5+ below"], "within 5")
    listed = (x["report_lvl"].fillna(0) > 0) | (x["practice_lvl"].fillna(0) > 0)
    x["c_injury_report"] = np.where(listed, "listed", "not listed")
    rc = np.where(x["stat"] == "rush_yds", x["role_chg_car"].abs(), np.where(x["stat"] == "rec_yds", x["role_chg_rec"].abs(), 0.0))
    x["c_role_change"] = np.where(rc >= 0.10, "changed", "steady")
    x["c_margin"] = np.select([x["margin"] >= 7, x["margin"] <= -7], ["favourite by 7+", "underdog by 7+"], "within 7")
    x["c_total"] = np.where(x["proj_total"] > 44.5, "total above 44.5", "total 44.5 or less")
    x["c_weeks"] = np.select([x["week"] <= 6, x["week"] <= 12], ["weeks 1-6", "weeks 7-12"], "weeks 13-18")
    return x


def cell(g: pd.DataFrame) -> dict:
    bm, bb = (g["model"] - g["hit"]) ** 2, (g["books"] - g["hit"]) ** 2
    d = (bm - bb).groupby(g["game_id"]).sum()
    se = float(d.std(ddof=1) * np.sqrt(len(d)) / len(g)) if len(d) > 1 else None
    return {"n": int(len(g)), "games": int(g["game_id"].nunique()), "cleared": round(float(g["hit"].mean()), 4),
            "model_says": round(float(g["model"].mean()), 4), "books_say": round(float(g["books"].mean()), 4),
            "brier_model": round(float(bm.mean()), 5), "brier_books": round(float(bb.mean()), 5),
            "model_minus_books": round(float((bm - bb).mean()), 5), "se": None if se is None else round(se, 5)}


def murphy(p, hit) -> dict:
    p, hit = np.asarray(p, float), np.asarray(hit, float)
    b = np.clip(np.floor(p * 20), 0, 19)
    base, rel, res = hit.mean(), 0.0, 0.0
    for k in np.unique(b):
        m = b == k
        rel += m.sum() * (p[m].mean() - hit[m].mean()) ** 2
        res += m.sum() * (hit[m].mean() - base) ** 2
    return {"calibration_miss": round(rel / len(p), 5), "sorting": round(res / len(p), 5), "brier": round(float(((p - hit) ** 2).mean()), 5)}


def bands(x) -> list:
    out = []
    for lo in np.arange(0.4, 1.0, 0.1):
        m = (x["model"] >= lo) & (x["model"] < lo + 0.1)
        if m.sum() >= 50:
            out.append({"model_says": f"{int(round(lo * 100))}-{int(round(lo * 100 + 10))}%", "n": int(m.sum()), "avg_model": round(float(x.loc[m, "model"].mean()), 3),
                        "avg_books": round(float(x.loc[m, "books"].mean()), 3), "cleared": round(float(x.loc[m, "hit"].mean()), 3)})
    return out


def money(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"bets": 0}
    b = g.sort_values("price").drop_duplicates(["game_id", "player_id", "stat", "line"], keep="last")   # the better of the two prices
    d = dec(b["price"])
    prof = np.where(b["hit"] == 1, d - 1, -1.0)
    cl = pd.Series(prof).groupby(b["game_id"].to_numpy()).sum()
    out = {"bets": int(len(b)), "games": int(b["game_id"].nunique()), "cleared": round(float(b["hit"].mean()), 4), "avg_decimal_price": round(float(d.mean()), 4),
           "return_per_dollar": round(float(prof.mean()), 4), "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(b)), 4) if len(cl) > 1 else None}
    if "close_price" in b and b["close_price"].notna().any():
        k = b["close_price"].notna().to_numpy()
        clv = d[k] / dec(b.loc[k, "close_price"]) - 1
        cc = pd.Series(clv).groupby(b["game_id"].to_numpy()[k]).sum()
        out["price_clv"] = {"n": int(k.sum()), "mean": round(float(clv.mean()), 4), "se": round(float(cc.std(ddof=1) * np.sqrt(len(cc)) / k.sum()), 4) if len(cc) > 1 else None}
    p, dm = float(b["hit"].mean()), float(d.mean())
    out["three_leg_parlay_from_average_leg"] = {"no_boost": round(p ** 3 * dm ** 3 - 1, 4), "boost_25pct": round(p ** 3 * (1 + 1.25 * (dm ** 3 - 1)) - 1, 4)}
    return out


def main():
    from sklearn.isotonic import IsotonicRegression
    p = projections()
    a = cuts(rungs_2025(p))
    b2 = rungs_2026(p)
    b2["proj"] = pd.to_numeric(b2["proj"], errors="coerce")
    b = cuts(b2)
    res = {"rungs": {"2025": cell(a), "2026 weeks 8-10": cell(b)}, "weeks_2026": sorted(int(w) for w in b["week"].unique())}
    # step 1
    s1 = {}
    for name, x in (("2025", a), ("2026 weeks 8-10", b)):
        s1[name] = {st: {"n": int(len(g)), "model": murphy(g["model"], g["hit"]), "books": murphy(g["books"], g["hit"])} for st, g in [("all", x), *x.groupby("stat")]}
        s1[name]["cleared_by_model_band"] = bands(x)
    early, late = a[a["week"] <= 9], a[a["week"] >= 10]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(early["model"], early["hit"])
    for name, x in (("2025 weeks 10-18", late), ("2026 weeks 8-10", b)):
        y = x.assign(model=iso.predict(x["model"]))
        s1[f"re-read model, {name}"] = {"as is": cell(x), "re-read": cell(y)}
    res["step1"] = s1
    # step 2
    s2, leads = {}, []
    for c in [k for k in a.columns if k.startswith("c_")]:
        s2[c[2:]] = {}
        for lvl, g in a.groupby(c):
            r = {"2025": cell(g)}
            h = b[b[c] == lvl]
            r["2026 weeks 8-10"] = cell(h) if len(h) else None
            ok25 = r["2025"]["n"] >= 300 and r["2025"]["games"] >= 30 and r["2025"]["model_minus_books"] <= 0 and r["2025"]["cleared"] >= r["2025"]["books_say"]
            ok26 = bool(len(h)) and r["2026 weeks 8-10"]["model_minus_books"] <= 0 and r["2026 weeks 8-10"]["cleared"] >= r["2026 weeks 8-10"]["books_say"]
            r["met_2025_bar"], r["lead"] = bool(ok25), bool(ok25 and ok26)
            if r["lead"]:
                leads.append(f"{c[2:]}: {lvl}")
            s2[c[2:]][str(lvl)] = r
    res["step2"], res["leads"] = s2, leads
    # step 3
    s3 = {}
    for name, x in (("2025", a), ("2026 weeks 8-10", b)):
        s3[name] = {"model 5+ above the books": money(x[x["c_disagreement"] == "model 5+ above"]), "every floor rung": money(x)}
        for ld in leads:
            c, lvl = ld.split(": ", 1)
            s3[name][f"{ld}, model 5+ above"] = money(x[(x["c_" + c] == lvl) & (x["c_disagreement"] == "model 5+ above")])
    res["step3"] = s3
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("rungs", "leads")}, default=str))


if __name__ == "__main__":
    main()
