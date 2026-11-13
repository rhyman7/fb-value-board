"""Pre-set test: smoothing college main-line overs and ladder rungs together (Nov 13, 2026).

    python experiments/cfb_ladder_smoothing_test.py

Rule written before running (Project doc claude/plan-cfb-ladder-smoothing-test-2026-11-13.md):

Current (live, parlays.monotone): pool-adjacent violators, equal weights, over each player's over legs
(main-line over and over-only rungs) so the chance falls as the line rises.
Candidate: the main-line over keeps its own chance (books' no-vig, nudged as live); the rungs are made
monotone around it. Implemented as the same pool-adjacent violators with the main lines weighted so
heavily they do not move (the closest non-increasing sequence that keeps the main lines fixed: rungs
below a main line that read under it are raised to it, rungs above that read over it are lowered to
it, and rungs otherwise pool among themselves).

Legs: over-only DK/FD rungs read through CFB_RUNG_CAL, averaged over the books (as cfb_book_legs);
two-sided main lines at the books' no-vig chance nudged 20% / 3 points toward the board's model
(CFB_NUDGE_STATS). 2024 and 2025 lines (ten minutes before kickoff, game-day projections, college
widths) are tuning; 2026 live (docs/cfb/props/2026_w*.json, first-priced chances) is the holdout.

Measures: Brier of the shown chance, candidate minus current, clustered by game, on player-games
where the versions differ: (1) main-line overs, (2) rungs, (3) all over legs. Kept only if better on
all 9 (three seasons), and on 2026 all-over-legs by at least 1 SE. Mixed results go to Ryan.

Writes experiments/out/cfb_ladder_smoothing_test.json. Changes nothing on the boards.
"""
from __future__ import annotations

import glob
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

from nflmodel import odds as O, parlays as PA, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402

OUT = HERE / "out" / "cfb_ladder_smoothing_test.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
BOOKS = ("draftkings", "fanduel")
BIG = 1e6


def wpav(y, w):
    """Weighted pool-adjacent violators: the closest non-increasing sequence."""
    blocks = []                                   # [value, weight, count]
    for v, ww in zip(y, w):
        blocks.append([v, ww, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    out = []
    for v, _, n in blocks:
        out += [v] * n
    return out


def nudge(po, fair):
    gap = po - fair
    return fair + np.clip(PA.CFB_W_MAIN * gap, -PA.CFB_PULL_MAX, PA.CFB_PULL_MAX) if abs(gap) <= PA.CONFLICT else fair


def smooth(legs: pd.DataFrame) -> pd.DataFrame:
    """legs: one player-game-stat's over legs (L, w, main, hit, game_id). Adds cur and cand."""
    legs = legs.sort_values(["L", "main"]).reset_index(drop=True)
    y = legs["w"].tolist()
    legs["cur"] = wpav(y, [1.0] * len(y))
    legs["cand"] = wpav(y, [BIG if m else 1.0 for m in legs["main"]])
    return legs


def score(x: pd.DataFrame) -> dict:
    groups = [smooth(g) for _, g in x.groupby(["game_id", "pid", "stat"]) if g["main"].any() and (~g["main"]).any()]
    d = pd.concat(groups, ignore_index=True)
    changed = (d["cur"] - d["cand"]).abs() > 1e-9
    key = d.groupby(["game_id", "pid", "stat"]).ngroup()
    differs = pd.Series(changed.to_numpy()).groupby(key).transform("any").to_numpy()
    s = d[differs]
    out = {"player_games": int(key.nunique()), "differ": int(pd.Series(key[differs]).nunique()),
           "legs_changed": int(changed.sum()), "mean_change": round(float((d["cur"] - d["cand"]).abs()[changed].mean()), 4) if changed.any() else 0.0,
           "main_over_shift_current": round(float((d["cur"] - d["w"])[d["main"]].mean()), 4)}
    for name, m in (("main_overs", s["main"]), ("rungs", ~s["main"]), ("all_overs", pd.Series(True, index=s.index))):
        z = s[m.to_numpy()]
        diff, se = R.clustered_diff(z["cand"].to_numpy(), z["cur"].to_numpy(), z["hit"].to_numpy(), z["game_id"].to_numpy())
        out[name] = {"legs": int(len(z)), "diff": round(diff, 6), "se": round(se, 6), "z": round(diff / se, 2) if se else None}
    return out


def season_legs(season: int) -> pd.DataFrame:
    R.set_season(season)
    pproj, pinfo, games = R.projections()
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s = s[s["stat"].isin(PA.CFB_NUDGE_STATS) & s["book"].isin(BOOKS)]
    rows = R.player_frame(pproj, pinfo)
    model = P.PropModel.load(PARAMS)
    # rungs: per book reading, averaged
    a = s[s["market"].str.endswith("_alternate") & (s["side"] == "over")].copy()
    a = a.groupby(["game_id", "nname", "stat", "line", "book"], as_index=False)["price"].last()
    a["read"] = np.interp(O.implied_prob(a["price"]), *zip(*PA.CFB_RUNG_CAL))
    a = a.groupby(["game_id", "nname", "stat", "line"], as_index=False)["read"].mean()
    # main lines: books' no-vig, nudged toward the model
    m = s[~s["market"].str.endswith("_alternate")]
    w = m.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair"].mean()
    x = R.model_chances(w.merge(rows, on=["game_id", "nname"], how="inner"), model)
    x["po"] = x["our_over"] / (x["our_over"] + x["our_under"])
    main = pd.DataFrame({"game_id": x["game_id"], "pid": x["nname"], "stat": x["stat"], "L": x["line"],
                         "w": [nudge(p, f) for p, f in zip(x["po"], x["fair"])], "main": True, "y": x["y"]})
    ra = a.merge(rows, on=["game_id", "nname"], how="inner")
    ra["y"] = np.nan
    for st in ra["stat"].unique():
        m_ = (ra["stat"] == st).to_numpy()
        ra.loc[m_, "y"] = np.asarray(P.actual(ra[m_], st), float)
    rung = pd.DataFrame({"game_id": ra["game_id"], "pid": ra["nname"], "stat": ra["stat"], "L": ra["line"], "w": ra["read"], "main": False, "y": ra["y"]})
    legs = pd.concat([main, rung], ignore_index=True)
    legs = legs[legs["y"] != legs["L"]]
    legs["hit"] = (legs["y"] > legs["L"]).astype(float)
    return legs


def live_legs() -> pd.DataFrame:
    out = []
    for f in sorted(glob.glob(str(ROOT / "docs" / "cfb" / "props" / "2026_w*.json"))):
        for p in json.load(open(f))["props"]:
            if p["s"] not in PA.CFB_NUDGE_STATS or p.get("y") is None or p["y"] == p["L"]:
                continue
            bk = {b: v for b, v in (p.get("bk") or {}).items() if b in BOOKS and v}
            if p["alt"]:
                reads = [float(np.interp(O.implied_prob(v[0]), *zip(*PA.CFB_RUNG_CAL))) for v in bk.values() if v[0] is not None]
                if not reads:
                    continue
                w = float(np.mean(reads))
            else:
                if p.get("fo") is None or p.get("po") is None or p.get("pu") is None or p["po"] + p["pu"] <= 0:
                    continue
                w = float(nudge(p["po"] / (p["po"] + p["pu"]), p["fo"]))
            out.append({"game_id": p["g"], "pid": p["id"], "stat": p["s"], "L": p["L"], "w": w, "main": not p["alt"],
                        "y": p["y"], "hit": float(p["y"] > p["L"])})
    return pd.DataFrame(out)


def main():
    res = {"seasons": {}}
    for label, legs in (("2024", season_legs(2024)), ("2025", season_legs(2025)), ("2026_live", live_legs())):
        res["seasons"][label] = score(legs)
        print(label, json.dumps(res["seasons"][label]), flush=True)
    checks = {k: [v[m]["diff"] < 0 for m in ("main_overs", "rungs", "all_overs")] for k, v in res["seasons"].items()}
    h = res["seasons"]["2026_live"]["all_overs"]
    res["verdict"] = {"checks": checks, "measures_better": int(sum(sum(c) for c in checks.values())), "measures": 9,
                      "holdout_by_1se": bool(h["diff"] <= -h["se"]),
                      "kept": bool(all(all(c) for c in checks.values()) and h["diff"] <= -h["se"])}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
