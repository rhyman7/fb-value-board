"""Pre-set test: the three college player-model changes from the ChatGPT review (Nov 19, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/chatgpt_player_changes_test.py

Rule written before running (Project doc claude/plan-chatgpt-player-changes-test-2026-11-19.md). The
review's code was not seen (it worked on copies of the repos); the versions are rebuilt from its write-up.

TE / WR pull-down: for projections above 15 (TE) / 20 (WR) receiving yards, new = projection - c, c the
  average of (projection - result) on earlier seasons' rows above the same bar; receptions scaled by the
  same ratio. Reported, not used: c as the middle miss (the version an average-miss score would pick).
QB big-underdog term: new = projection - b * max(0, -margin - 7), margin the board's projected margin for
  his team, b by least squares through the origin on earlier seasons' misses. Projected starters
  (att_share >= 0.5) who threw a pass. Reported, not used: the same term for favourites of more than 7.
The review's blended "second estimate" is not a projection change and is not tested (the board already
moves the books' college main-line chance 20% towards the model).

Rows: played and recorded a stat (as the review scored). Every constant is fitted on earlier seasons only.
Measures per change (8): RMSE, game-day and bet-time, tuning 2017-19 and holdout 2021-25; Brier at the
DK/FD main line, 2024 and 2025, game-day and bet-time (widths unchanged; a player with a line who took a
snap and recorded nothing is an under). Kept only if better on all 8, each holdout RMSE gain >= 1 SE.
MAE, bias and 2026 so far are reported, not used. Writes scores only, never PFF rows.
"""
from __future__ import annotations

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

from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402
import cfb_shutout_test as S  # noqa: E402

OUT = HERE / "out" / "chatgpt_player_changes_test.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
TUNE, HOLD = [2017, 2018, 2019], [2021, 2022, 2023, 2024, 2025]
BAR = {"TE": 15.0, "WR": 20.0}
KEY = ["game_id", "player_id"]
DOG = 7.0


def rec_rows(cur, g):
    return (cur["played"].astype(bool) & (cur["appeared"] == 1) & (cur["group"] == g) & (cur["rec_yds_proj"] > BAR[g])).to_numpy()


def qb_rows(cur):
    return (cur["played"].astype(bool) & (cur["appeared"] == 1) & (cur["group"] == "QB") & (cur["att_share"] >= 0.5) & (cur["pass_att"] > 0)).to_numpy()


def pull_down(cur, how="mean"):
    new, info = cur.copy(), {}
    sea = cur["season"].to_numpy()
    miss = (cur["rec_yds_proj"] - cur["rec_yds"]).to_numpy(float)
    shift = np.zeros(len(cur))
    for g in BAR:
        r = rec_rows(cur, g)
        above = ((cur["group"] == g) & (cur["rec_yds_proj"] > BAR[g])).to_numpy()
        for s in sorted(set(sea)):
            tr = r & (sea < s)
            if tr.sum() < 500:
                continue
            c = float(np.mean(miss[tr]) if how == "mean" else np.median(miss[tr]))
            shift[above & (sea == s)] = c
            info[f"{g} {s}"] = round(c, 2)
    ratio = np.where(cur["rec_yds_proj"] > 0, (cur["rec_yds_proj"] - shift) / cur["rec_yds_proj"].replace(0, np.nan), 1.0)
    new["rec_yds_proj"] = cur["rec_yds_proj"] - shift
    new["receptions_proj"] = cur["receptions_proj"] * np.nan_to_num(ratio, nan=1.0)
    return new, info


def qb_term(cur, side="dog"):
    new, info = cur.copy(), {}
    sea = cur["season"].to_numpy()
    m = cur["margin"].to_numpy(float)
    h = np.nan_to_num(np.maximum(0, (-m if side == "dog" else m) - DOG))
    miss = (cur["pass_yds_proj"] - cur["pass_yds"]).to_numpy(float)
    r = qb_rows(cur)
    adj = np.zeros(len(cur))
    for s in sorted(set(sea)):
        tr = r & (sea < s) & (h > 0)
        if tr.sum() < 200:
            continue
        b = float((h[tr] * miss[tr]).sum() / (h[tr] ** 2).sum())
        te = (cur["group"] == "QB").to_numpy() & (sea == s)
        adj[te] = b * h[te]
        info[str(s)] = {"yards_per_point": round(b, 3), "rows": int(tr.sum())}
    new["pass_yds_proj"] = cur["pass_yds_proj"] - adj
    new["hinge"] = h
    return new, info


def errors(cur, new, rows, stat, seasons, only=None):
    k = rows & cur["season"].isin(seasons).to_numpy()
    if only is not None:
        k = k & only
    d = cur.loc[k, KEY + [stat, f"{stat}_proj"]].merge(new[KEY + [f"{stat}_proj"]].rename(columns={f"{stat}_proj": "new"}), on=KEY)
    y, a, b = d[stat].to_numpy(float), d[f"{stat}_proj"].to_numpy(float), d["new"].to_numpy(float)
    if not len(d):
        return {"n": 0}
    diff, se = G.paired((b - y) ** 2, (a - y) ** 2, d["game_id"].to_numpy())
    mdiff, mse = G.paired(np.abs(b - y), np.abs(a - y), d["game_id"].to_numpy())
    return {"n": int(len(d)), "changed": int((a != b).sum()), "rmse_now": round(float(np.sqrt(((a - y) ** 2).mean())), 3),
            "rmse_new": round(float(np.sqrt(((b - y) ** 2).mean())), 3), "mse_diff": round(diff, 3), "mse_se": round(se, 3), "z": round(diff / se, 2) if se else 0.0,
            "mae_now": round(float(np.abs(a - y).mean()), 3), "mae_new": round(float(np.abs(b - y).mean()), 3), "mae_z": round(mdiff / mse, 2) if mse else 0.0,
            "bias_now": round(float((y - a).mean()), 2), "bias_new": round(float((y - b).mean()), 2)}


def brier(cur, new, pinfo, games, g, stat, season, cols, changed_only=False) -> dict:
    R.set_season(season)
    s = R.snapshots()
    s = s.merge(R.match_events(s, games), left_on="event_id", right_index=True, how="inner")
    s["nname"], s["stat"] = s["player"].map(PM.norm_name), s["market"].map(PM.MARKET_STAT)
    main = s[~s["market"].str.endswith("_alternate") & (s["stat"] == stat)]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    took = cur[(cur["appeared"] == 1) | (cur["anysnap"] == 1)]
    ren = {c: f"n_{c}" for c in cols}
    fr = R.player_frame(took, pinfo).merge(new[KEY + cols].rename(columns=ren), on=KEY, how="inner")
    base_ = w.merge(fr[fr["group"] == g], on=["game_id", "nname"], how="inner")
    if changed_only:
        base_ = base_[(base_[cols[0]] - base_[ren[cols[0]]]).abs() > 1e-9]
    model = P.PropModel.load(PARAMS)
    outs = {}
    for name in ("current", "new"):
        dd = base_.copy()
        if name == "new":
            for c in cols:
                dd[c] = dd[ren[c]]
        x = R.model_chances(dd, model)
        if x.empty:
            return {"props": 0}
        outs[name] = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
    idx = outs["current"].index.intersection(outs["new"].index)
    a, b = outs["current"].loc[idx], outs["new"].loc[idx]
    if not len(a):
        return {"props": 0}
    hit = (a["y"] > a["line"]).astype(float).to_numpy()
    pc, pn = (a["our_over"] / (a["our_over"] + a["our_under"])).to_numpy(), (b["our_over"] / (b["our_over"] + b["our_under"])).to_numpy()
    d, se = R.clustered_diff(pn, pc, hit, a["game_id"].to_numpy())
    return {"props": int(len(a)), "now": round(D.brier(pc, hit), 5), "new": round(D.brier(pn, hit), 5), "diff": round(d, 6), "se": round(se, 6),
            "books": round(D.brier(a["fair"].to_numpy(), hit), 5), "over_rate": round(float(hit.mean()), 4), "model_over_now": round(float(pc.mean()), 4),
            "model_over_new": round(float(pn.mean()), 4), "thin": bool(len(a) < 200)}


def verdict(r):
    rm = [r["tuning"][k] for k in ("gd", "bt")] + [r["holdout"][k] for k in ("gd", "bt")]
    br = [x for x in r["brier"].values() if x.get("props")]
    better = all(x["mse_diff"] < 0 for x in rm) and all(x["diff"] < 0 for x in br) and len(br) == 4
    one_se = all(x["z"] <= -1 for x in (r["holdout"][k] for k in ("gd", "bt")))
    n = sum(x["mse_diff"] < 0 for x in rm) + sum(x["diff"] < 0 for x in br)
    r["better_on"] = f"{n} of {len(rm) + len(br)}"
    return "KEPT (provisional)" if better and one_se else ("Trade-off: better on all 8, a holdout gain under 1 SE" if better else f"Not kept under the rule: better on {n} of {len(rm) + len(br)}")


def main():
    t, pinfo, games, frame, pg = S.base()
    snaps, _ = S.snaps(t, pinfo)
    F = {}
    for bt in (False, True):
        cur = S.walk("cur", t, bt)
        cur = cur.merge(snaps[KEY].assign(anysnap=1), on=KEY, how="left")
        cur["anysnap"] = cur["anysnap"].fillna(0).astype(int)
        F["bt" if bt else "gd"] = cur
    res = {"plan": "claude/plan-chatgpt-player-changes-test-2026-11-19.md", "changes": {}}
    for g in ("TE", "WR"):
        r = {"tuning": {}, "holdout": {}, "brier": {}, "median_version": {"tuning": {}, "holdout": {}, "brier": {}}, "season_2026": {}}
        for k, cur in F.items():
            new, info = pull_down(cur, "mean")
            med, minfo = pull_down(cur, "median")
            rows = rec_rows(cur, g)
            r["tuning"][k], r["holdout"][k] = errors(cur, new, rows, "rec_yds", TUNE), errors(cur, new, rows, "rec_yds", HOLD)
            r["season_2026"][k] = {"mean": errors(cur, new, rows, "rec_yds", [2026]), "median": errors(cur, med, rows, "rec_yds", [2026])}
            r["median_version"]["tuning"][k], r["median_version"]["holdout"][k] = errors(cur, med, rows, "rec_yds", TUNE), errors(cur, med, rows, "rec_yds", HOLD)
            for season in (2024, 2025):
                r["brier"][f"{season} {k}"] = brier(cur, new, pinfo, games, g, "rec_yds", season, ["rec_yds_proj", "receptions_proj"])
                r["median_version"]["brier"][f"{season} {k}"] = brier(cur, med, pinfo, games, g, "rec_yds", season, ["rec_yds_proj", "receptions_proj"])
            if k == "gd":
                r["shift_by_season"] = {a: b for a, b in info.items() if a.startswith(g)}
                r["median_shift_by_season"] = {a: b for a, b in minfo.items() if a.startswith(g)}
        r["verdict"] = verdict(r)
        res["changes"][f"{g} pull-down"] = r
        print(g, r["verdict"], flush=True)
    r = {"tuning": {}, "holdout": {}, "brier": {}, "affected_only": {"tuning": {}, "holdout": {}, "brier": {}}, "favourites_reported": {"tuning": {}, "holdout": {}}, "season_2026": {}}
    for k, cur in F.items():
        new, info = qb_term(cur, "dog")
        fav, finfo = qb_term(cur, "fav")
        rows = qb_rows(cur)
        hit = (new["hinge"] > 0).to_numpy()
        r["tuning"][k], r["holdout"][k] = errors(cur, new, rows, "pass_yds", TUNE), errors(cur, new, rows, "pass_yds", HOLD)
        r["affected_only"]["tuning"][k], r["affected_only"]["holdout"][k] = errors(cur, new, rows, "pass_yds", TUNE, hit), errors(cur, new, rows, "pass_yds", HOLD, hit)
        r["favourites_reported"]["tuning"][k], r["favourites_reported"]["holdout"][k] = errors(cur, fav, rows, "pass_yds", TUNE), errors(cur, fav, rows, "pass_yds", HOLD)
        r["season_2026"][k] = errors(cur, new, rows, "pass_yds", [2026], hit)
        for season in (2024, 2025):
            r["brier"][f"{season} {k}"] = brier(cur, new, pinfo, games, "QB", "pass_yds", season, ["pass_yds_proj"])
            r["affected_only"]["brier"][f"{season} {k}"] = brier(cur, new, pinfo, games, "QB", "pass_yds", season, ["pass_yds_proj"], changed_only=True)
        if k == "gd":
            r["slope_by_season"], r["favourite_slope_by_season"] = info, finfo
    r["verdict"] = verdict(r)
    res["changes"]["QB big-underdog term"] = r
    print("QB", r["verdict"], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
