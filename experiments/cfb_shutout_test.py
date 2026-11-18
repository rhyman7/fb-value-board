"""Pre-set test: counting shut-outs in the college receiving projections (Nov 18, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_shutout_test.py

Rule written before running (Project doc claude/plan-cfb-shutout-test-2026-11-18.md). Lead: the
Weekly Review's college Players page with shut-outs counted (2026 weeks 1-11): tight ends projected
over 15 yards reach Typical 44%, wide receivers 48%. The college chain knows a player "appeared"
only when he has a box-score line, so a game played without a catch or target is a game missed.

On the field = a box-score line, or PFF offensive snaps of at least half the median of his earlier
weeks that season (two weeks minimum; with fewer, any snap). A shut-out = on the field, no line.

A. new = proj x (1 - p0) for receiving yards and receptions; p0 = chance of a shut-out given on the
   field, per position, a logistic curve on log(projected receptions), fitted on earlier seasons only.
B. Shut-outs become zero rows in the player-game history (WR, TE, RB) and the chain is rebuilt.
   (Process note: the first run's B changed nothing because the chain drops rows with no attempt,
   carry or catch; fixed with a negligible reception count and the whole test rerun.)

Judged for TE and WR separately. Rows: on the field, current projection over 15 yards (TE) or 20
(WR); a shut-out scores 0. The current projections are the baseline on the same rows.
Measures per position (8): RMSE on receiving yards, game-day and bet-time, tuning 2017-19 and
holdout 2021-25; Brier at the DK/FD main receiving-yards lines, 2024 and 2025, game-day and bet-time
(a player with a line who took any offensive snap and recorded nothing is an under; widths unchanged).
Version chosen on tuning (better on both tuning RMSE, then the lower game-day one). Kept only if
better on all 8, each holdout RMSE gain >= 1 SE. Writes scores only, never PFF rows.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from cfbmodel import players as PL  # noqa: E402
from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_pff_role_test as T  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = HERE / "out" / "cfb_shutout_test.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
TUNE, HOLD = [2017, 2018, 2019], [2021, 2022, 2023, 2024, 2025]
MIN = {"TE": 15.0, "WR": 20.0}
KEY = ["game_id", "player_id"]
TMP = Path("/tmp")
EPS = 1e-6


def base():
    """The board's tables and the frame they were built from (cached)."""
    from cfbmodel import data, model, players_data
    from cfbmodel.ratings import CFBRatingEngine
    t, pinfo, games = G.table()
    fc = TMP / "cfb_shut_frame.pkl"
    if fc.exists():
        frame = pd.read_pickle(fc)
    else:
        g2, tg, qb = data.load_all(refresh=False, verbose=False)
        frame, _ = model.walk_forward(model.build_frame(g2, CFBRatingEngine(tg, qb, games=g2)))
        frame.to_pickle(fc)
    pg, _ = players_data.load_player_games(verbose=False)
    return t, pinfo, games, frame, pg


def snaps(t, pinfo) -> pd.DataFrame:
    """(game_id, player_id) -> PFF offensive snaps and whether they were his usual share."""
    pw = T.pff_player_weeks(list(range(2017, 2026)))
    m, st = T.map_onto_board(pw, t, pinfo)
    m = m.dropna(subset=["board_id"])
    m = m[m["snap_counts_total"] > 0].copy()
    m["player_id"] = m["board_id"].astype(np.int64)
    m = m.sort_values(["player_id", "season", "week"]).drop_duplicates(["game_id", "player_id"])
    g = m.groupby(["player_id", "season"])["snap_counts_total"]
    usual = g.transform(lambda s: s.shift().expanding(2).median())
    m["usual_ok"] = usual.isna() | (m["snap_counts_total"] >= 0.5 * usual)
    return m[["game_id", "player_id", "team", "season", "snap_counts_total", "usual_ok"]].rename(columns={"snap_counts_total": "snaps"}), st


def tables(t, bt: bool):
    if not bt:
        return t
    tb = t.copy()
    m = tb["played"].astype(bool)
    tb.loc[m, "active"] = tb.loc[m, "expected"].astype(int)
    return tb


def walk(name, t, bt):
    c = TMP / f"cfb_shut_{name}_{'bt' if bt else 'gd'}.pkl"
    old = TMP / f"cfb_boost_cur_{'bet-time' if bt else 'game-day'}.pkl"
    if name == "cur" and old.exists():
        return pd.read_pickle(old)
    if c.exists():
        return pd.read_pickle(c)
    p = PL.walk_forward(tables(t, bt))
    p.to_pickle(c)
    return p


def version_b(t, pinfo, games, frame, pg, S):
    """The table rebuilt with shut-outs as zero rows in the player-game history."""
    c = TMP / "cfb_shut_tableB.pkl"
    if c.exists():
        return pd.read_pickle(c)
    grp = t.drop_duplicates("player_id").set_index("player_id")["group"]
    tid = t.drop_duplicates(["season", "team"]).set_index(["season", "team"])["team_id"]
    have = set(zip(pg["game_id"], pg["player_id"]))
    z = S[S["usual_ok"] & S["player_id"].map(grp).isin(["WR", "TE", "RB"])]
    z = z[[k not in have for k in zip(z["game_id"], z["player_id"])]].copy()
    z["team_id"] = [tid.get(k) for k in zip(z["season"], z["team"])]
    z = z.dropna(subset=["team_id"])
    add = pd.DataFrame(0.0, index=range(len(z)), columns=[c_ for c_ in pg.columns if c_ not in ("season", "game_id", "team_id", "player_id")])
    # players.player_rows keeps a row only when attempts + carries + receptions > 0, so a true zero row is
    # dropped (found on the first run: version B changed nothing). A negligible reception count makes the
    # game count as an appearance with zero yards without touching the model's code.
    add["receptions"] = EPS
    add.insert(0, "player_id", z["player_id"].to_numpy()); add.insert(0, "team_id", z["team_id"].astype(pg["team_id"].dtype).to_numpy())
    add.insert(0, "game_id", z["game_id"].to_numpy()); add.insert(0, "season", z["season"].to_numpy())
    tB = PL.build(games, frame, pd.concat([pg, add[pg.columns]], ignore_index=True), pinfo, None, None)
    tB.attrs["zero_rows"] = int(len(add))
    tB.to_pickle(c)
    return tB


def version_a(cur: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """cur with rec_yds_proj and receptions_proj multiplied by (1 - p0), p0 fitted on earlier seasons."""
    new = cur.copy()
    p0 = np.zeros(len(cur))
    info = {}
    x = np.log(cur["receptions_proj"].clip(lower=0.05).to_numpy(float))
    ok = (cur["played"].astype(bool) & (cur["onfield"] == 1) & (cur["receptions_proj"] > 0)).to_numpy()
    y = (cur["appeared"] == 0).to_numpy()
    sea = cur["season"].to_numpy()
    for g in ("TE", "WR", "RB"):
        gm = (cur["group"] == g).to_numpy()
        for s in sorted(set(sea)):
            tr = ok & gm & (sea < s) & (sea >= 2017)
            if tr.sum() < 500 or y[tr].sum() < 20:
                continue
            lr = LogisticRegression(C=1e6).fit(x[tr, None], y[tr])
            te = gm & (sea == s)
            p0[te] = lr.predict_proba(x[te, None])[:, 1]
            info[f"{g} {s}"] = {"intercept": round(float(lr.intercept_[0]), 3), "slope": round(float(lr.coef_[0][0]), 3), "train": int(tr.sum()), "train_rate": round(float(y[tr].mean()), 4)}
    new["p0"] = p0
    for c in ("rec_yds_proj", "receptions_proj"):
        new[c] = cur[c] * (1 - p0)
    return new, info


def rows(cur, g):
    return cur["played"].astype(bool) & (cur["onfield"] == 1) & (cur["group"] == g) & (cur["rec_yds_proj"] > MIN[g])


def rmse(cur, new, g, seasons) -> dict:
    d = cur.loc[rows(cur, g) & cur["season"].isin(seasons), KEY + ["rec_yds", "rec_yds_proj"]].merge(
        new[KEY + ["rec_yds_proj"]].rename(columns={"rec_yds_proj": "new"}), on=KEY)
    e0, e1 = (d["rec_yds_proj"] - d["rec_yds"]).to_numpy() ** 2, (d["new"] - d["rec_yds"]).to_numpy() ** 2
    diff, se = G.paired(e1, e0, d["game_id"].to_numpy())
    r0 = float(np.sqrt(e0.mean()))
    return {"n": int(len(d)), "now": round(r0, 4), "candidate": round(float(np.sqrt(e1.mean())), 4), "mse_diff": round(diff, 4), "mse_se": round(se, 4),
            "z": round(diff / se, 2) if se else 0.0, "bias_now": round(float((d["rec_yds"] - d["rec_yds_proj"]).mean()), 2),
            "bias_candidate": round(float((d["rec_yds"] - d["new"]).mean()), 2)}


def brier(cur, new, pinfo, games, g, season) -> dict:
    R.set_season(season)
    s = R.snapshots()
    s = s.merge(R.match_events(s, games), left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    main = s[~s["market"].str.endswith("_alternate") & (s["stat"] == "rec_yds")]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    took = cur[(cur["appeared"] == 1) | (cur["anysnap"] == 1)]             # a prop is void if he did not play
    fr = R.player_frame(took, pinfo).merge(new[KEY + ["rec_yds_proj", "receptions_proj"]].rename(columns={"rec_yds_proj": "n_s", "receptions_proj": "n_v"}), on=KEY, how="inner")
    base_ = w.merge(fr[fr["group"] == g], on=["game_id", "nname"], how="inner")
    model = P.PropModel.load(PARAMS)
    outs = {}
    for name in ("current", "new"):
        dd = base_.copy()
        if name == "new":
            dd["rec_yds_proj"], dd["receptions_proj"] = dd["n_s"], dd["n_v"]
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
    return {"props": int(len(a)), "shut_outs_with_a_line": int((a["appeared"] == 0).sum()), "now": round(D.brier(pc, hit), 5), "candidate": round(D.brier(pn, hit), 5),
            "diff": round(d, 6), "se": round(se, 6), "books": round(D.brier(a["fair"].to_numpy(), hit), 5), "thin": bool(len(a) < 200)}


def main():
    t0 = time.time()
    t, pinfo, games, frame, pg = base()
    S, mapst = snaps(t, pinfo)
    print("snaps", len(S), mapst, round(time.time() - t0), flush=True)
    tB = version_b(t, pinfo, games, frame, pg, S)
    print("table B built", round(time.time() - t0), flush=True)
    res = {"plan": "claude/plan-cfb-shutout-test-2026-11-18.md", "pff_match": mapst, "positions": {}}
    F = {}
    for bt in (False, True):
        cur = walk("cur", t, bt)
        cur = cur.merge(S[KEY + ["usual_ok"]].assign(anysnap=1), on=KEY, how="left")
        cur["anysnap"] = cur["anysnap"].fillna(0).astype(int)
        cur["shut"] = ((cur["appeared"] == 0) & cur["played"].astype(bool) & (cur["usual_ok"] == True)).astype(int)  # noqa: E712
        cur["onfield"] = ((cur["appeared"] == 1) | (cur["shut"] == 1)).astype(int)
        A, infoA = version_a(cur)
        B = walk("B", tB, bt)
        F["bt" if bt else "gd"] = (cur, A, B)
        if not bt:
            res["p0_fit"] = infoA
        print("projections", "bet-time" if bt else "game-day", round(time.time() - t0), flush=True)
    cur = F["gd"][0]
    hold = cur["season"].isin(HOLD) & cur["played"].astype(bool)
    for g in ("TE", "WR", "RB"):
        on = hold & (cur["onfield"] == 1) & (cur["group"] == g) & (cur["rec_yds_proj"] > MIN.get(g, 10.0))
        d = cur[on]
        band = pd.cut(d["receptions_proj"], [0, 1, 1.5, 2, 3, 4, 99])
        res.setdefault("shut_out_rate_holdout", {})[g] = {"n": int(len(d)), "rate": round(float(d["shut"].mean()), 4),
            "by_projected_receptions": {str(k): {"n": int(len(x)), "actual": round(float(x["shut"].mean()), 4), "A_predicts": round(float(F["gd"][1].loc[x.index, "p0"].mean()), 4)} for k, x in d.groupby(band, observed=True)}}
    for g in ("TE", "WR"):
        r = {"tuning": {}}
        for v, i in (("A", 1), ("B", 2)):
            r["tuning"][v] = {k: rmse(F[k][0], F[k][i], g, TUNE) for k in ("gd", "bt")}
        ok = [v for v in ("A", "B") if all(r["tuning"][v][k]["mse_diff"] < 0 for k in ("gd", "bt"))]
        r["better_on_both_tuning"] = ok
        if not ok:
            r["verdict"] = "No version better on both tuning measures: nothing carried to the holdout, nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["gd"]["candidate"])
            i = 1 if pick == "A" else 2
            r["chosen"] = pick
            r["holdout"] = {k: rmse(F[k][0], F[k][i], g, HOLD) for k in ("gd", "bt")}
            r["brier"] = {f"{season} {k}": brier(F[k][0], F[k][i], pinfo, games, g, season) for season in (2024, 2025) for k in ("gd", "bt")}
            other = "B" if pick == "A" else "A"
            r["other_version_reported"] = {"holdout": {k: rmse(F[k][0], F[k][3 - i], g, HOLD) for k in ("gd", "bt")}, "version": other,
                                           "brier": {f"{season} {k}": brier(F[k][0], F[k][3 - i], pinfo, games, g, season) for season in (2024, 2025) for k in ("gd", "bt")}}
            br = [x for x in r["brier"].values() if x.get("props")]
            better = all(x["mse_diff"] < 0 for x in list(r["tuning"][pick].values()) + list(r["holdout"].values())) and all(x["diff"] < 0 for x in br) and len(br) == 4
            one_se = all(x["z"] <= -1 for x in r["holdout"].values())
            r["verdict"] = "KEPT (provisional)" if better and one_se else ("Trade-off: better on all 8, a holdout gain under 1 SE" if better else "Trade-off: mixed across the 8 measures")
        res["positions"][g] = r
        print(g, json.dumps(r, default=float), flush=True)
    MIN["RB"] = 10.0
    res["reported"] = {"RB": {v: {k: rmse(F[k][0], F[k][i], "RB", HOLD) for k in ("gd", "bt")} for v, i in (("A", 1), ("B", 2))}}
    res["zero_rows_added_B"] = tB.attrs.get("zero_rows")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
