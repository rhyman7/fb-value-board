"""Pre-set test: NFL receivers whose WOPR is high for their projection, judged on Typical and the floor rungs.

    PFF_DATA_DIR=<pff-data clone> python experiments/wopr_standing_test.py

Rule written before running (Project doc claude/plan-wopr-standing-test-2026-11-18.md). Lead: a side
table of experiments/air_yards_wopr_test.py (WR/TE projected 25+ yards, 2019-25, game-day): at the same
projection the high-WOPR third out-produced the low third. The 2019-25 holdout was therefore already
read for this idea; 2026 weeks 1-10 are the unseen check.

The change, after the board's final receiving-yards projection, WR and TE:
    new = proj x (1 + b x z)
  WOPR  1.5 x target share + 0.7 x air-yards share (air_yards_wopr_test.inputs, wopr_raw)
  z     his WOPR minus a straight line of WOPR on projected yards, fitted by position on earlier seasons
        (from 2014; played, projected 10+ yards), over the spread of those misses, held in [-2.5, 2.5]
  b     one number, least squares on earlier seasons only (actual - proj on proj x z), held in [0, 0.2],
        fitted on each table's own history
Versions: A WR/TE projected 10+ yards; B 25+ yards only.
Measures (15), WR/TE who played, projected 10+ yards:
  RMSE, game-day and bet-time, tuning 2014-18 and holdout 2019-25 (4)
  Typical: Brier at each player's own current Typical line (the half-yard line where the prop model
    fitted on earlier seasons gives the current projection an even chance), candidate projection
    against current, game-day and bet-time, tuning and holdout (4)
  Brier at the 2023-25 DK/FD receiving-yards main lines, WR/TE: close with game-day, pricing snapshot
    with bet-time (2)
  Brier on the 2025 alternate rungs the books read at 65-95%, WR/TE, pricing snapshot, bet-time (1)
  2026 weeks 1-10 (b and the z line from 2014-2025): RMSE and Typical Brier, game-day and bet-time (4)
Rule: the version is chosen on tuning alone (better on all four tuning measures, then the lowest
  bet-time tuning Typical Brier). Kept only if better on all 15 and each holdout RMSE gain >= 1 paired
  SE; a version worse on any 2026 measure is not kept whatever the rest says. Mixed, or under 1 SE: a
  trade-off for Ryan. No version better on all four tuning measures: nothing is carried further.
Reported, not used: b by season; the high / middle / low thirds at bet time, on 2014-18 and on 2026;
  how far projections move.

Writes experiments/out/wopr_standing_test.json.
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

from nflmodel import config as C, props as P  # noqa: E402
import air_yards_wopr_test as AW  # noqa: E402

OUT = HERE / "out" / "wopr_standing_test.json"
CACHE = C.RAW / "wopr_standing_now.pkl"
TUNE, HOLD, FRESH = (2014, 2018), (2019, 2025), (2026, 2026)
VERSIONS = {"A": 10.0, "B": 25.0}
GROUPS = ("WR", "TE")
ST = "rec_yds"


def current() -> dict:
    """The board's current projections, game-day and bet-time, with wopr_raw (cached)."""
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    T = AW.tables()
    feats = AW.inputs(T["gameday"])[["game_id", "player_id", "wopr_raw"]]
    out = {}
    for k in ("gameday", "bettime"):
        p = AW.project(T[k], T["R"], {}, [])
        out[k] = p.merge(feats, on=["game_id", "player_id"], how="left").fillna({"wopr_raw": 0.0}).reset_index(drop=True)
        print("projected", k, len(out[k]), flush=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def live(p):
    return (p["played"].astype(bool) & (p["active"] == 1) & p["group"].isin(GROUPS) & p[ST].notna()).to_numpy()


def standing(p: pd.DataFrame) -> np.ndarray:
    proj, w, season, grp = p[ST + "_proj"].to_numpy(float), p["wopr_raw"].to_numpy(float), p["season"].to_numpy(), p["group"].to_numpy()
    base = live(p) & (proj >= 10)
    z = np.zeros(len(p))
    for s in sorted(set(season[season >= 2015])):
        for g in GROUPS:
            k = base & (grp == g) & (season < s) & (season >= 2014)
            if k.sum() < 500:
                continue
            c1, c0 = np.polyfit(proj[k], w[k], 1)
            sd = float((w[k] - (c0 + c1 * proj[k])).std())
            m = (season == s) & (grp == g)
            z[m] = np.clip((w[m] - (c0 + c1 * proj[m])) / sd, -2.5, 2.5)
    return z


def adjusted(p: pd.DataFrame, z: np.ndarray, lo: float):
    proj, y, season = p[ST + "_proj"].to_numpy(float), p[ST].to_numpy(float), p["season"].to_numpy()
    el = p["group"].isin(GROUPS).to_numpy() & (proj >= lo)
    h = np.where(el, proj * z, 0.0)
    fit = live(p) & el
    new, bs = proj.copy(), {}
    for s in sorted(set(season[season >= 2015])):
        k = fit & (season < s) & (season >= 2014)
        b = float(np.clip((h[k] @ (y[k] - proj[k])) / max(1e-9, h[k] @ h[k]), 0, 0.2)) if k.any() else 0.0
        new[season == s] = proj[season == s] + b * h[season == s]
        bs[int(s)] = round(b, 4)
    c = p.copy()
    c[ST + "_proj"] = new
    return c, bs


def mask(p, per):
    return live(p) & (p[ST + "_proj"].to_numpy(float) >= 10) & p["season"].between(*per).to_numpy()


def rmse(base, cand, per) -> dict:
    k = mask(base, per)
    y = base[ST].to_numpy(float)[k]
    e0, e1 = (base[ST + "_proj"].to_numpy(float)[k] - y) ** 2, (cand[ST + "_proj"].to_numpy(float)[k] - y) ** 2
    r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
    cl = pd.Series(e1 - e0).groupby(base["game_id"].to_numpy()[k]).sum()
    return {"n": int(k.sum()), "now": round(r0, 4), "candidate": round(r1, 4), "diff": round(r1 - r0, 5),
            "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / k.sum() / (2 * r0)), 5)}


_MODELS: dict = {}


def prop_model(base, key, season):
    from props_calibration import FAMILY
    if (key, season) not in _MODELS:
        m = P.PropModel(FAMILY).fit(base[(base.season < season) & (base.season >= 2013)], stats=["receptions", ST])
        saved = P.PropModel.load()
        if saved is not None:
            m.iso = saved.iso
        _MODELS[(key, season)] = m
    return _MODELS[(key, season)]


def typical_rows(base, cand, key, per) -> pd.DataFrame:
    """Each scored player's current Typical line, and the model's over chance there for both projections."""
    out = []
    k = mask(base, per)
    for s in sorted(set(base["season"].to_numpy()[k])):
        m = prop_model(base, key, int(s))
        for g in GROUPS:
            if (ST, g) not in m.yards:
                continue
            sel = k & (base["season"].to_numpy() == s) & (base["group"].to_numpy() == g)
            d, c = base[sel].reset_index(drop=True), cand[sel].reset_index(drop=True)
            if d.empty:
                continue
            mu = P.mean(d, ST)
            L = np.floor(mu[:, None] * np.linspace(0.05, 1.7, 67)[None, :]) + 0.5
            over, _, under = m.probs(d, ST, g, L)
            dec = over / np.maximum(over + under, 1e-9)
            typ = np.array([np.interp(0.5, dec[i][::-1], L[i][::-1]) for i in range(len(d))])
            line = np.floor(typ) + 0.5
            o0, _, u0 = m.probs(d, ST, g, line)
            o1, _, u1 = m.probs(c, ST, g, line)
            out.append(pd.DataFrame({"game_id": d["game_id"], "season": s, "y": d[ST].to_numpy(float), "line": line, "typical": typ,
                                     "proj": mu, "wopr_raw": d["wopr_raw"].to_numpy(float),
                                     "p0": np.ravel(o0 / (o0 + u0)), "p1": np.ravel(o1 / (o1 + u1))}))
    return pd.concat(out, ignore_index=True)


def typical_brier(base, cand, key, per) -> dict:
    x = typical_rows(base, cand, key, per)
    hit = (x["y"] > x["line"]).astype(float)
    d = (x["p1"] - hit) ** 2 - (x["p0"] - hit) ** 2
    cl = d.groupby(x["game_id"]).sum()
    return {"n": int(len(x)), "now": round(float(((x["p0"] - hit) ** 2).mean()), 5), "candidate": round(float(((x["p1"] - hit) ** 2).mean()), 5),
            "diff": round(float(d.mean()), 6), "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(x)), 6), "reached_typical": round(float(hit.mean()), 4)}


def thirds(base, key, per) -> dict:
    """Reported: at the same projection (25+ yards), the low / middle / high WOPR thirds."""
    x = typical_rows(base, base, key, per)
    x = x[x["proj"] >= 25].copy()
    x["band"] = pd.qcut(x["proj"], 10, labels=False, duplicates="drop")
    x["third"] = x.groupby("band")["wopr_raw"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    x["hit"] = (x["y"] > x["line"]).astype(float)
    x["floor"] = (x["y"] >= 0.6 * x["proj"]).astype(float)
    out = {str(t): {"n": int(len(g)), "avg_proj": round(float(g["proj"].mean()), 1), "avg_actual": round(float(g["y"].mean()), 1),
                    "reached_typical": round(float(g["hit"].mean()), 4), "cleared_60pct": round(float(g["floor"].mean()), 4)} for t, g in x.groupby("third")}
    hi, lo = x[x["third"] == "high"], x[x["third"] == "low"]
    for c in ("hit", "floor"):
        out[f"high minus low, {'typical' if c == 'hit' else '60pct'}"] = {
            "diff": round(float(hi[c].mean() - lo[c].mean()), 4), "se": round(float(np.sqrt(hi[c].var() / len(hi) + lo[c].var() / len(lo))), 4)}
    out["high minus low, actual minus projected"] = {
        "diff": round(float((hi.y - hi.proj).mean() - (lo.y - lo.proj).mean()), 2),
        "se": round(float(np.sqrt((hi.y - hi.proj).var() / len(hi) + (lo.y - lo.proj).var() / len(lo))), 2)}
    return out


def main():
    N = current()
    Z = {k: standing(N[k]) for k in N}
    res = {"tuning": {}}
    cands = {}
    for v, lo in VERSIONS.items():
        cands[v] = {k: adjusted(N[k], Z[k], lo) for k in N}
        res["tuning"][v] = {**{f"RMSE {k}": rmse(N[k], cands[v][k][0], TUNE) for k in N},
                            **{f"Typical Brier {k}": typical_brier(N[k], cands[v][k][0], k, TUNE) for k in N}}
        print(v, json.dumps(res["tuning"][v]), flush=True)
    res["b_by_season"] = {v: {k: cands[v][k][1] for k in N} for v in VERSIONS}
    ok = [v for v in VERSIONS if all(m["diff"] < 0 for m in res["tuning"][v].values())]
    res["better_on_all_four_tuning"] = ok
    if not ok:
        res["verdict"] = "No version better on all four tuning measures: nothing carried further, nothing kept."
    else:
        pick = min(ok, key=lambda v: res["tuning"][v]["Typical Brier bettime"]["candidate"])
        c = {k: cands[pick][k][0] for k in N}
        res["chosen"] = pick
        res["holdout"] = {**{f"RMSE {k}": rmse(N[k], c[k], HOLD) for k in N}, **{f"Typical Brier {k}": typical_brier(N[k], c[k], k, HOLD) for k in N}}
        print("holdout", json.dumps(res["holdout"]), flush=True)
        res["lines"] = {"main lines, close, game-day": AW.brier(N["gameday"], c["gameday"], "close", (ST,), GROUPS),
                        "main lines, pricing snapshot, bet-time": AW.brier(N["bettime"], c["bettime"], "flag", (ST,), GROUPS),
                        "floor rungs 65-95%, pricing snapshot, bet-time": AW.floor_rungs(N["bettime"], c["bettime"], GROUPS)}
        print("lines", json.dumps(res["lines"]), flush=True)
        res["2026 weeks 1-10"] = {**{f"RMSE {k}": rmse(N[k], c[k], FRESH) for k in N}, **{f"Typical Brier {k}": typical_brier(N[k], c[k], k, FRESH) for k in N}}
        ms = list(res["tuning"][pick].values()) + list(res["holdout"].values()) + list(res["lines"].values()) + list(res["2026 weeks 1-10"].values())
        better = [m["diff"] < 0 for m in ms]
        se_ok = all(-m["diff"] >= m["se"] for n, m in res["holdout"].items() if n.startswith("RMSE"))
        fresh_ok = all(m["diff"] < 0 for m in res["2026 weeks 1-10"].values())
        res["better_on"] = f"{sum(better)} of {len(better)}"
        res["kept"] = bool(all(better) and se_ok)
        res["verdict"] = ("Kept (provisional)." if res["kept"] else "Worse on a 2026 measure: not kept." if not fresh_ok
                          else "Better on every measure but a holdout RMSE gain is under 1 SE: a trade-off for Ryan." if all(better)
                          else "Mixed: a trade-off for Ryan, not kept by the rule.")
        mv = np.abs(c["bettime"][ST + "_proj"] - N["bettime"][ST + "_proj"]).to_numpy()[mask(N["bettime"], HOLD)]
        res["projection_move_bettime_2019_25"] = {"mean_abs_yards": round(float(mv.mean()), 2), "p90": round(float(np.quantile(mv, 0.9)), 2), "max": round(float(mv.max()), 2)}
    res["reported_thirds"] = {"game-day 2014-18": thirds(N["gameday"], "gameday", TUNE), "bet-time 2014-18": thirds(N["bettime"], "bettime", TUNE),
                              "game-day 2019-25": thirds(N["gameday"], "gameday", HOLD), "bet-time 2019-25": thirds(N["bettime"], "bettime", HOLD),
                              "bet-time 2026 weeks 1-10": thirds(N["bettime"], "bettime", FRESH)}
    print(json.dumps(res, default=str), flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
