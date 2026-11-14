"""Pre-set test: correcting the college QB rushing chance (Nov 13, 2026).

    python experiments/cfb_qb_rush_calibration_test.py     (PFF data at ../pff-data, as the board)

Rule written before running (Project doc claude/plan-cfb-qb-rush-calibration-test-2026-11-13.md):

Candidates correct only the model's chance of the over for college QB rushing yards, fitted
walk-forward without betting lines: for scored season S, on college QB player-games 2017..S-1 (appeared,
projection >= 10 yards), each at three synthetic lines (projection, 0.7x, 1.3x, rounded to .5), board
projections, college prop widths.
  A  logit shift: chance' = sigmoid(logit(chance) + a), a minimising Brier (grid -1.5..1.5 by 0.01)
  B  isotonic: chance' = isotonic fit of cleared on chance
Measures: Brier at the DK/FD QB rushing main lines, new minus current, clustered by game: 2024 (tuning)
and 2025 (holdout), game-day and bet-time (4); 2026 live (logged chances; correction fitted on
2017-2025), must not be worse. Reported: with the Parlays nudge; QB rushing ladder rungs.
Rule: kept only if better on all 4, the 2025 holdout by 1+ SE for both kinds, 2026 live not worse; if
both pass, the lower 2025 game-day Brier. Mixed results go to Ryan.
(The board's projections here come from one walk-forward over all seasons, players.walk_forward with
its default first season, used both to fit the correction and to score.)

Writes experiments/out/cfb_qb_rush_calibration_test.json. Changes nothing on the boards.
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

from cfbmodel import players as PL  # noqa: E402
from nflmodel import odds as O, parlays as PA, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_gap_diag as X  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = HERE / "out" / "cfb_qb_rush_calibration_test.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
GRID = np.round(np.arange(-1.5, 1.501, 0.01), 2)


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ex(z):
    return 1 / (1 + np.exp(-z))


def half(x):
    return np.floor(np.asarray(x, float)) + 0.5


def synthetic(proj, model, seasons):
    d = proj[proj["season"].isin(seasons) & (proj["group"] == "QB") & proj["played"].astype(bool) & (proj["appeared"] == 1)
             & (proj["rush_yds_proj"] >= 10)].reset_index(drop=True)
    L = np.column_stack([half(d["rush_yds_proj"] * f) for f in (1.0, 0.7, 1.3)])
    over, push, under = model.probs(d, "rush_yds", "QB", L)
    ch = (over / (over + under)).ravel()
    y = np.asarray(P.actual(d, "rush_yds"), float)
    hit = (y[:, None] > L).astype(float).ravel()
    return ch, hit


def fit(ch, hit):
    a = GRID[int(np.argmin([((ex(lg(ch) + g) - hit) ** 2).mean() for g in GRID]))]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(ch, hit)
    return {"A": lambda p, a=a: ex(lg(p) + a), "B": lambda p, iso=iso: iso.predict(np.asarray(p, float))}, float(a)


def nudge(p, fair):
    gap = p - fair
    return np.where(np.abs(gap) <= PA.CONFLICT, fair + np.clip(PA.CFB_W_MAIN * gap, -PA.CFB_PULL_MAX, PA.CFB_PULL_MAX), fair)


def diff(new, cur, hit, game):
    d, se = R.clustered_diff(np.asarray(new), np.asarray(cur), np.asarray(hit), np.asarray(game))
    return {"diff": round(d, 6), "se": round(se, 6), "z": round(d / se, 2) if se else None}


def qb_lines(proj, pinfo, games, season, model):
    w = X.lines(season, games)
    R.set_season(season)
    rows = R.player_frame(proj, pinfo)
    x = R.model_chances(w.merge(rows, on=["game_id", "nname"], how="inner"), model)
    x = x[(x["y"] != x["line"]) & (x["group"] == "QB")].copy()
    x["hit"] = (x["y"] > x["line"]).astype(float)
    x["ours"] = x["our_over"] / (x["our_over"] + x["our_under"])
    return x


def qb_rungs(proj, pinfo, games, season, model):
    R.set_season(season)
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    a = s[s["market"].str.endswith("_alternate") & (s["side"] == "over") & (s["stat"] == "rush_yds")]
    a = a.groupby(["game_id", "nname", "stat", "line"], as_index=False)["price"].max()
    x = R.model_chances(a.merge(R.player_frame(proj, pinfo), on=["game_id", "nname"], how="inner"), model)
    x = x[(x["y"] != x["line"]) & (x["group"] == "QB")].copy()
    x["hit"] = (x["y"] > x["line"]).astype(float)
    x["ours"] = x["our_over"] / (x["our_over"] + x["our_under"])
    return x


def live_2026():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "docs" / "cfb" / "props" / "2026_w*.json"))):
        for p in json.load(open(f))["props"]:
            if p["alt"] or p["s"] != "rush_yds" or p.get("pos") != "QB" or p.get("y") is None or p["y"] == p["L"]:
                continue
            if p.get("fo") is None or p.get("po") is None or p.get("pu") is None or p["po"] + p["pu"] <= 0:
                continue
            rows.append({"game_id": p["g"], "ours": p["po"] / (p["po"] + p["pu"]), "fair": p["fo"], "hit": float(p["y"] > p["L"])})
    return pd.DataFrame(rows)


def main():
    t, pinfo, games = G.table()
    if not (t.get("route_share", 0) > 0).any():
        print("WARNING: no PFF data (../pff-data): the projections are not the board's")
    model = P.PropModel.load(PARAMS)
    tb = t.copy()
    m = tb["played"].astype(bool)
    tb.loc[m, "active"] = tb.loc[m, "expected"].astype(int)
    res = {"fits": {}, "lines": {}, "nudge": {}, "rungs": {}}
    for kind, tab in (("game_day", t), ("bet_time", tb)):
        proj = PL.walk_forward(tab)
        for season in (2024, 2025):
            corr, a = fit(*synthetic(proj, model, range(2017, season)))
            res["fits"][f"{kind} {season}"] = {"A_shift": a}
            x = qb_lines(proj, pinfo, games, season, model)
            r = {"props": int(len(x)), "games": int(x["game_id"].nunique()), "current_mean_chance": round(float(x["ours"].mean()), 4),
                 "cleared": round(float(x["hit"].mean()), 4), "brier_books": round(float(((x["fair"] - x["hit"]) ** 2).mean()), 5),
                 "brier_current": round(float(((x["ours"] - x["hit"]) ** 2).mean()), 5)}
            for c, f in corr.items():
                p = f(x["ours"])
                r[c] = {"mean_chance": round(float(np.mean(p)), 4), "brier": round(float(((p - x["hit"]) ** 2).mean()), 5),
                        **diff(p, x["ours"], x["hit"], x["game_id"])}
                res["nudge"].setdefault(f"{kind} {season}", {})[c] = diff(nudge(p, x["fair"]), nudge(x["ours"], x["fair"]), x["hit"], x["game_id"])
            res["lines"][f"{kind} {season}"] = r
            if kind == "game_day":
                rr = qb_rungs(proj, pinfo, games, season, model)
                res["rungs"][str(season)] = {"rungs": int(len(rr)), **{c: diff(f(rr["ours"]), rr["ours"], rr["hit"], rr["game_id"]) for c, f in corr.items()}}
            print(kind, season, json.dumps(r), flush=True)
        if kind == "game_day":
            corr26, a26 = fit(*synthetic(proj, model, range(2017, 2026)))
            res["fits"]["live 2026"] = {"A_shift": a26}
            lv = live_2026()
            res["live_2026"] = {"props": int(len(lv)), "games": int(lv["game_id"].nunique()) if len(lv) else 0}
            if len(lv):
                for c, f in corr26.items():
                    res["live_2026"][c] = diff(f(lv["ours"]), lv["ours"], lv["hit"], lv["game_id"])
    verdict = {}
    for c in ("A", "B"):
        ok = all(res["lines"][k][c]["diff"] < 0 for k in res["lines"])
        hold = all(res["lines"][f"{k} 2025"][c]["diff"] <= -res["lines"][f"{k} 2025"][c]["se"] for k in ("game_day", "bet_time"))
        live_ok = ("live_2026" not in res) or (c not in res["live_2026"]) or res["live_2026"][c]["diff"] <= 0
        verdict[c] = {"all_four_better": bool(ok), "holdout_1se": bool(hold), "live_not_worse": bool(live_ok), "passes": bool(ok and hold and live_ok)}
    passing = [c for c in verdict if verdict[c]["passes"]]
    verdict["kept"] = min(passing, key=lambda c: res["lines"]["game_day 2025"][c]["brier"]) if passing else None
    res["verdict"] = verdict
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("fits", "verdict", "live_2026", "rungs", "nudge")}, indent=1, default=float))


if __name__ == "__main__":
    main()
