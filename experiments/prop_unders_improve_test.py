"""Can the NFL prop-unders rule be improved? (Nov 20, 2026)

    python experiments/prop_unders_improve_test.py [--rows FILE]

Rules written first: Project doc claude/plan-nfl-prop-unders-improvement-2026-11-20.md. The track is
the Recommended list's "NFL prop unders, model agrees" (nflmodel/recommended.py: the board's over
chance below 35%; receptions and receiving yards, a running back's rushing yards and attempts).

Question 1: is the 2026 live log a break from 2023-25, or noise (z of the difference in returns)?
Question 2: five fixed candidates against the rule as it stands, on the rows of
props_over_lean_test.py (2023-24 tuning, 2025 holdout; pricing snapshot and close):
  V1 skip unders priced shorter than -150       V2 over chance 10+ points below the books' fair
  V3 below 35% and 10+ points below the books   V4 below 30%        V5 skip tiny lines
Kept only if better than the baseline on all four measures, by 1 standard error on the holdout at
both snapshots, above zero on the holdout, and keeping 40%+ of the baseline's holdout bets.

Writes experiments/out/prop_unders_improve_test.json. Changes nothing on the boards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parent / "out" / "prop_unders_improve_test.json"
TUNE, HOLD = (2023, 2024), (2025,)
AGREE, GAP, STRICT, JUICE = 0.35, 0.10, 0.30, -150.0
TINY = {"receptions": 2.5, "rec_yds": 20.5, "rush_yds": 20.5, "carries": 8.5}


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def cl_se(v: pd.Series, games: pd.Series) -> float:
    s = v.groupby(games).sum()
    return float(s.std(ddof=1) * np.sqrt(len(s)) / len(v)) if len(s) > 1 and len(v) else float("nan")


def in_rule(x: pd.DataFrame) -> pd.Series:
    """The track's markets: receptions and receiving yards, a running back's rushing yards and attempts."""
    return x["stat"].isin(["receptions", "rec_yds"]) | (x["stat"].isin(["rush_yds", "carries"]) & (x["group"] == "RB"))


def masks(x: pd.DataFrame) -> dict:
    base = x["ours"] < AGREE
    juice_ok = dec(x["under"]) >= dec(JUICE)
    gap = x["ours"] <= x["fair"] - GAP
    tiny = x["line"] < x["stat"].map(TINY)
    return {"baseline": base, "V1_no_heavy_juice": base & juice_ok, "V2_gap_only": gap, "V3_both": base & gap,
            "V4_below_30": x["ours"] < STRICT, "V5_no_tiny_lines": base & ~tiny}


def stat(x: pd.DataFrame) -> dict:
    if not len(x):
        return {"n": 0, "ret": None, "se": None, "win": None}
    return {"n": int(len(x)), "ret": round(float(x["u_ret"].mean()), 4), "se": round(cl_se(x["u_ret"], x["game_id"]), 4),
            "win": round(float((x["hit"] == 0).mean()), 4)}


def compare(x: pd.DataFrame, m: dict, name: str) -> dict:
    """Candidate against baseline on one snapshot and one set of seasons."""
    b, c = x[m["baseline"]], x[m[name]]
    out = {"baseline": stat(b), "cand": stat(c)}
    if not len(b) or not len(c):
        return {**out, "gap": None, "gap_se": None}
    if name == "V2_gap_only":                    # not a subset: the two standard errors combined
        gap, gse = c["u_ret"].mean() - b["u_ret"].mean(), float(np.hypot(out["cand"]["se"], out["baseline"]["se"]))
    else:                                        # a subset: the bets kept against the bets dropped
        d = x[m["baseline"] & ~m[name]]
        if not len(d):
            return {**out, "gap": None, "gap_se": None}
        gap, gse = c["u_ret"].mean() - d["u_ret"].mean(), float(np.hypot(out["cand"]["se"], cl_se(d["u_ret"], d["game_id"])))
        out["dropped"] = stat(d)
    return {**out, "gap": round(float(gap), 4), "gap_se": round(gse, 4)}


def verdict(r: dict) -> dict:
    cells = [r[s][k] for s in ("pricing", "close") for k in ("tune", "hold")]
    better = [c["cand"]["ret"] is not None and c["baseline"]["ret"] is not None and c["cand"]["ret"] > c["baseline"]["ret"] for c in cells]
    hold = [r[s]["hold"] for s in ("pricing", "close")]
    by_se = [h["gap"] is not None and h["gap_se"] and h["gap"] / h["gap_se"] >= 1 for h in hold]
    above = [h["cand"]["ret"] is not None and h["cand"]["ret"] > 0 for h in hold]
    share = min((h["cand"]["n"] / h["baseline"]["n"]) if h["baseline"]["n"] else 0 for h in hold)
    return {"better_on": int(sum(better)), "of": 4, "holdout_1se": bool(all(by_se)), "holdout_above_zero": bool(all(above)),
            "share_of_baseline": round(share, 3), "kept": bool(all(better) and all(by_se) and all(above) and share >= 0.40)}


def live_rows(season: int = 2026) -> pd.DataFrame:
    """The live Props files as rows shaped like the backfill's (graded main lines, the two books)."""
    rows = []
    for f in sorted((ROOT / "docs" / "props").glob(f"{season}_w*.json")):
        for p in json.loads(f.read_text()).get("props") or []:
            if p.get("alt") or p.get("snap") != "flag" or p.get("res") not in ("win", "loss") or p.get("po") is None or p.get("pu") is None:
                continue
            best = None
            for bk, pr in (p.get("bk") or {}).items():
                if bk in ("draftkings", "fanduel") and pr and len(pr) > 1 and pr[1] is not None:
                    if best is None or dec(pr[1]) > dec(best):
                        best = float(pr[1])
            if best is None or p["po"] + p["pu"] <= 0:
                continue
            rows.append({"week": int(f.stem.split("_w")[1]), "game_id": p["g"], "stat": p["s"], "group": p["pos"], "line": float(p["L"]),
                         "ours": p["po"] / (p["po"] + p["pu"]), "fair": p.get("fo"), "under": best, "hit": 1.0 if p["res"] == "win" else 0.0})
    x = pd.DataFrame(rows)
    if len(x):
        x["u_ret"] = np.where(x["hit"] == 0, dec(x["under"]) - 1, -1.0)
        x["fair"] = pd.to_numeric(x["fair"], errors="coerce")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None, help="a pickle of props_over_lean_test's (rows, pff); built when not given")
    a = ap.parse_args()
    if a.rows:
        rows, pff = pd.read_pickle(a.rows)
    else:
        import props_over_lean_test as T
        if not T.ROWS_CACHE.exists():
            raise SystemExit("run experiments/props_over_lean_test.py first: it builds the rows this test reads")
        rows, pff = pd.read_pickle(T.ROWS_CACHE)
    rep = {"pff_route_inputs": pff, "tune": list(TUNE), "hold": list(HOLD), "candidates": {}}
    data = {}
    for snap in ("pricing", "close"):
        x = rows[snap].rename(columns={"fair_over": "fair"})       # the books' vig-free over chance (power method, as the board)
        x = x[in_rule(x) & x["under"].notna() & x["fair"].notna()].copy()
        data[snap] = x
    hist = stat(data["pricing"][masks(data["pricing"])["baseline"]])
    rep["baseline_by_season"] = {snap: {int(s): stat(y[masks(y)["baseline"]]) for s, y in data[snap].groupby("season")} for snap in data}
    for name in list(masks(data["pricing"]))[1:]:
        r = {}
        for snap, x in data.items():
            r[snap] = {k: compare(x[x["season"].isin(ss)], masks(x[x["season"].isin(ss)]), name) for k, ss in (("tune", TUNE), ("hold", HOLD))}
        r["verdict"] = verdict(r)
        rep["candidates"][name] = r
    lv = live_rows()
    if len(lv):
        lv = lv[in_rule(lv)]
        m = masks(lv.assign(fair=lv["fair"].fillna(0.5)))
        b = lv[m["baseline"]]
        live = {"weeks": sorted(int(w) for w in lv["week"].unique()), "baseline": stat(b),
                "by_week": {int(w): stat(y) for w, y in b.groupby("week")},
                "by_stat": {f"{s} {g}": stat(y) for (s, g), y in b.groupby(["stat", "group"])},
                "candidates": {k: stat(lv[v]) for k, v in m.items() if k != "baseline"}}
        if live["baseline"]["n"] and hist["n"]:
            live["z_vs_2023_25"] = round((live["baseline"]["ret"] - hist["ret"]) / float(np.hypot(live["baseline"]["se"], hist["se"])), 2)
        rep["live_2026"] = live
    rep["history_pricing"] = hist
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps({"history_pricing": hist, "live_2026": rep.get("live_2026"),
                      "verdicts": {k: v["verdict"] for k, v in rep["candidates"].items()}}, indent=1))


if __name__ == "__main__":
    main()
