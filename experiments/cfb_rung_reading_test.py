"""Pre-set test: the college ladder reading refitted on 2024 + 2025 (Nov 13, 2026).

    python experiments/cfb_rung_reading_test.py

Rule written before running (Project doc claude/plan-cfb-rung-reading-test-2026-11-13.md):

Candidate: the current fit (cfb_props_real_lines.py, ladders: over-only DK/FD alternate yardage rungs
ten minutes before kickoff, the better book's price, raw implied chance, isotonic regression of
cleared on implied, read at the same grid and interpolated) on 2024 and 2025 pooled instead of 2025
alone. Current: parlays.CFB_RUNG_CAL.

Measures, candidate minus current, clustered by game: (1) Brier, all rungs; (2) Brier, floor rungs
(current reading 65-95%); (3) calibration miss in the 80%+ band (current reading >= 80%):
|share cleared - average reading|. On 2024 and 2025 (tuning; in-sample for the candidate) and the
2026 live rungs (holdout: docs/cfb/props/2026_w*.json, graded rungs, the better DK/FD price the board
logged when it first priced the rung).

Rule: kept only if the candidate is better on all three measures in 2024, 2025 and 2026, and on the
2026 all-rungs Brier by at least 1 SE. Mixed results go to Ryan. Reported only: each single season's
fit scored on the other season; the readings at 80/85/90/95% implied.

Writes experiments/out/cfb_rung_reading_test.json. Changes nothing on the boards.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import odds as O, parlays as PA, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402

OUT = HERE / "out" / "cfb_rung_reading_test.json"
GRID = np.array([x for x, _ in PA.CFB_RUNG_CAL])
FLOOR = (0.65, 0.95)
BOOKS = ("draftkings", "fanduel")


def rungs(season: int) -> pd.DataFrame:
    """The same rung sample the current reading was fitted on, for one season."""
    R.set_season(season)
    pproj, pinfo, games = R.projections()
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    a = s[s["market"].str.endswith("_alternate") & (s["side"] == "over")]
    a = a.groupby(["game_id", "nname", "stat", "line"], as_index=False)["price"].max()
    a["implied"] = O.implied_prob(a["price"])
    r = R.model_chances(a.merge(R.player_frame(pproj, pinfo), on=["game_id", "nname"], how="inner"), P.PropModel.load())
    r = r[r["y"] != r["line"]]
    return pd.DataFrame({"game_id": r["game_id"].to_numpy(), "implied": np.asarray(r["implied"], float),
                         "hit": (r["y"] > r["line"]).astype(float).to_numpy()})


def live_2026() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "docs" / "cfb" / "props" / "2026_w*.json"))):
        for p in json.load(open(f))["props"]:
            if not p["alt"] or p.get("y") is None or p["y"] == p["L"]:
                continue
            prices = [v[0] for b, v in (p.get("bk") or {}).items() if b in BOOKS and v and v[0] is not None]
            if not prices:
                continue
            best = max(prices, key=lambda a: float(O.american_to_decimal(a)))
            rows.append({"game_id": p["g"], "implied": float(O.implied_prob(best)), "hit": float(p["y"] > p["L"])})
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame) -> list:
    iso = IsotonicRegression(out_of_bounds="clip").fit(df["implied"], df["hit"])
    return [(float(g), round(float(v), 4)) for g, v in zip(GRID, iso.predict(GRID))]


def read(cal, implied) -> np.ndarray:
    return np.interp(np.asarray(implied, float), *zip(*cal))


def compare(df: pd.DataFrame, cand, cur) -> dict:
    hit, g = df["hit"].to_numpy(), df["game_id"].to_numpy()
    pc, pn = read(cur, df["implied"]), read(cand, df["implied"])
    d1, s1 = R.clustered_diff(pn, pc, hit, g)
    fl = (pc >= FLOOR[0]) & (pc <= FLOOR[1])
    d2, s2 = R.clustered_diff(pn[fl], pc[fl], hit[fl], g[fl])
    top = pc >= 0.8
    miss_cur = abs(hit[top].mean() - pc[top].mean()) if top.any() else None
    miss_new = abs(hit[top].mean() - pn[top].mean()) if top.any() else None
    return {"rungs": int(len(df)), "games": int(pd.Series(g).nunique()),
            "all_brier": {"current": round(D.brier(pc, hit), 5), "candidate": round(D.brier(pn, hit), 5), "diff": round(d1, 6), "se": round(s1, 6)},
            "floor_brier": {"rungs": int(fl.sum()), "diff": round(d2, 6), "se": round(s2, 6),
                            "cleared": round(float(hit[fl].mean()), 4), "read_current": round(float(pc[fl].mean()), 4), "read_candidate": round(float(pn[fl].mean()), 4)},
            "top_80": {"rungs": int(top.sum()), "cleared": round(float(hit[top].mean()), 4) if top.any() else None,
                       "read_current": round(float(pc[top].mean()), 4) if top.any() else None,
                       "read_candidate": round(float(pn[top].mean()), 4) if top.any() else None,
                       "miss_current": round(float(miss_cur), 4) if miss_cur is not None else None,
                       "miss_candidate": round(float(miss_new), 4) if miss_new is not None else None}}


def better(c: dict) -> list[bool]:
    return [c["all_brier"]["diff"] < 0, c["floor_brier"]["diff"] < 0,
            c["top_80"]["miss_candidate"] is not None and c["top_80"]["miss_candidate"] < c["top_80"]["miss_current"]]


def main():
    cur = [tuple(x) for x in PA.CFB_RUNG_CAL]
    d24, d25, d26 = rungs(2024), rungs(2025), live_2026()
    cand = fit(pd.concat([d24, d25], ignore_index=True))
    res = {"current": cur, "candidate": cand, "refit_2025_alone": fit(d25), "fit_2024_alone": fit(d24),
           "seasons": {"2024": compare(d24, cand, cur), "2025": compare(d25, cand, cur), "2026_live": compare(d26, cand, cur)},
           "reported_only": {"fit_2024_scored_2025_vs_current": compare(d25, fit(d24), cur),
                             "fit_2025_scored_2024_vs_current": compare(d24, fit(d25), cur)},
           "reading_at": {str(x): {"current": round(float(read(cur, [x])[0]), 4), "candidate": round(float(read(cand, [x])[0]), 4)}
                          for x in (0.8, 0.85, 0.9, 0.95)}}
    checks = {s: better(c) for s, c in res["seasons"].items()}
    h = res["seasons"]["2026_live"]["all_brier"]
    res["verdict"] = {"checks": checks, "measures_better": int(sum(sum(v) for v in checks.values())), "measures": 9,
                      "holdout_by_1se": bool(h["diff"] <= -h["se"]),
                      "kept": bool(all(all(v) for v in checks.values()) and h["diff"] <= -h["se"])}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("seasons", "reading_at", "verdict")}, indent=1, default=float))


if __name__ == "__main__":
    main()
