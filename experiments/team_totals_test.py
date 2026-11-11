"""NFL receiving and passing projections that add up within a team (review item A4, Nov 11, 2026).

    PFF_DATA_DIR=../pff-data python experiments/team_totals_test.py

Rule written before the run (Project doc claude/plan-team-totals-test-2026-11-11.md). Seen before
the rule: receivers' projected yards average 103% of their quarterback's projected passing yards
(sd 12 points), and when the receivers' sum is above his number they come in short by 0.72 of the
gap, he over by 0.28.

Candidates, per team-game, players expected to play, where the team has a projected starter
(50+ projected passing yards); scale factors limited to 0.7-1.4:
  A  receivers' receiving yards scaled so the team sum is the quarterback's projected passing
     yards, receptions to his projected completions; the quarterback unchanged
  B  target = QB + a x (receivers - QB), a fitted on earlier seasons only; receivers and the
     quarterback (passing yards, completions) both scaled to it
Measures: RMSE on 2014-18 and 2019 on, paired by player-game (rec_yds, receptions: WR/TE/RB with
1+ projected reception; pass_yds, completions: quarterbacks with 10+ projected attempts); Brier at
the closing DraftKings/FanDuel lines 2023-25, those four markets pooled.
Keep rule (Nov 11, 2026): every RMSE measure a candidate changes lower on both periods, each
holdout gain at least 1 standard error, pooled Brier lower. Both pass: the lower pooled Brier.

The projections are the board's (players.walk_forward, the boosted step, the Questionable step),
built once and cached in data/raw/team_totals_proj.pkl. Writes experiments/out/team_totals_test.json.
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
from nflmodel import config as C, propbackfill as B, props as P  # noqa: E402

CACHE = C.RAW / "team_totals_proj.pkl"
PAIRS = (("rec_yds", "pass_yds"), ("receptions", "completions"))
CLIP = (0.7, 1.4)


def build() -> pd.DataFrame:
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))["ml"]
    from nflmodel import data, extra, game_roles, injuries, model, pff_qb, players, players_data, players_ml, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=pff_qb.load(qb, sorted(games.season.unique())))
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = players.load_availability(refresh_current=False, verbose=False)
    rush = injuries.load_pass_rush(games, refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = players.player_rows(players.team_games(f0, pg), pg, avail, pinfo)
    frame, _ = model.walk_forward(game_roles.attach(base, game_roles.team_features(rows, R, avail)))
    table = players.build(frame, pg, pinfo, avail, inj, rush, games=games, overrides=None)
    pall = players.walk_forward(table, first_test=C.FIRST_TEST_SEASON)
    ml = players_ml.questionable(players_ml.walk_forward(players_ml.add_features(pall, R), first=C.FIRST_TEST_SEASON), first=C.FIRST_TEST_SEASON, snaps=R.get("snaps"))
    ml = ml.merge(players.season_average(table), on=["game_id", "player_id"], how="left")
    C.RAW.mkdir(parents=True, exist_ok=True)
    pickle.dump({"ml": ml}, open(CACHE, "wb"))
    return ml


def team_sums(p: pd.DataFrame, rec: str, qb: str):
    act = (p["active"] == 1).to_numpy()
    isqb = (p["group"] == "QB").to_numpy()
    key = [p["game_id"], p["team"]]
    S = pd.Series(np.where(act & ~isqb, p[rec + "_proj"], 0.0), index=p.index).groupby(key).transform("sum")
    Q = pd.Series(np.where(act & isqb, p[qb + "_proj"], 0.0), index=p.index).groupby(key).transform("sum")
    Sa = pd.Series(np.where(~isqb, p[rec].fillna(0), 0.0), index=p.index).groupby(key).transform("sum")
    yq = pd.Series(np.where(act & isqb, p["pass_yds_proj"], 0.0), index=p.index).groupby(key).transform("sum")
    ok = (yq >= 50) & (S > 0) & (Q > 0)
    return S, Q, Sa, ok, act, isqb


def reconcile(p: pd.DataFrame, how: str) -> tuple[pd.DataFrame, dict]:
    p = p.copy()
    info = {}
    for rec, qb in PAIRS:
        S, Q, Sa, ok, act, isqb = team_sums(p, rec, qb)
        if how == "A":
            T = Q
        else:
            T = Q.copy()
            a_by = {}
            for s in sorted(p["season"].unique()):
                tr = (p["season"] < s) & ok & p["played"].astype(bool)
                t = p[tr].assign(S=S[tr], Q=Q[tr], Sa=Sa[tr]).drop_duplicates(["game_id", "team"])
                if len(t) < 300:
                    a = 0.28
                else:
                    g, e = (t["S"] - t["Q"]).to_numpy(float), (t["Sa"] - t["Q"]).to_numpy(float)   # actual team total against the QB's number
                    a = float(np.clip(g @ e / (g @ g), 0.0, 1.0))
                a_by[int(s)] = round(a, 3)
                m = p["season"] == s
                T[m] = Q[m] + a * (S[m] - Q[m])
            info[rec] = a_by
        fr = (T / S.where(S > 0)).clip(*CLIP).where(ok, 1.0).fillna(1.0)
        p[rec + "_proj"] = np.where(act & ~isqb, p[rec + "_proj"] * fr, p[rec + "_proj"])
        if how == "B":
            fq = (T / Q.where(Q > 0)).clip(*CLIP).where(ok, 1.0).fillna(1.0)
            p[qb + "_proj"] = np.where(act & isqb, p[qb + "_proj"] * fq, p[qb + "_proj"])
    return p, info


def rmse_table(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    out = {}
    live = base["played"].astype(bool) & (base["active"] == 1) & (base["season"] >= 2014)
    sel = {"rec_yds": live & base["group"].isin(["WR", "TE", "RB"]) & (base["receptions_proj"] >= 1),
           "receptions": live & base["group"].isin(["WR", "TE", "RB"]) & (base["receptions_proj"] >= 1),
           "pass_yds": live & (base["group"] == "QB") & (base["pass_att_proj"] >= 10),
           "completions": live & (base["group"] == "QB") & (base["pass_att_proj"] >= 10)}
    for st, m in sel.items():
        for per, pm in (("2014-18", base["season"] < 2019), ("2019 on", base["season"] >= 2019)):
            k = m & pm & base[st].notna()
            e0, e1 = (base.loc[k, st + "_proj"] - base.loc[k, st]) ** 2, (cand.loc[k, st + "_proj"] - cand.loc[k, st]) ** 2
            d = (e1 - e0).to_numpy(float)
            r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
            se = float(d.std(ddof=1) / np.sqrt(len(d)) / (2 * r0)) if len(d) > 1 else None      # SE of the RMSE change (delta method)
            out[f"{st} {per}"] = {"n": int(k.sum()), "now": round(r0, 4), "candidate": round(r1, 4), "diff": round(r1 - r0, 4), "se": round(se, 4),
                                  "changed": bool(np.abs(d).sum() > 0)}
    return out


def lines(tables: dict) -> dict:
    import props_real_lines as RL
    from props_calibration import FAMILY
    saved = P.PropModel.load()
    stats = ("rec_yds", "receptions", "pass_yds", "completions")
    got = {k: [] for k in tables}
    for season in (2023, 2024, 2025):
        snaps = B.load(season, "close")
        if snaps.empty:
            continue
        for k, t in tables.items():
            m = P.PropModel(FAMILY).fit(t[(t.season < season) & (t.season >= 2013)])
            if saved is not None:
                m.iso = saved.iso
            x = RL.priced(t, m, snaps, season)
            got[k].append(x[x["stat"].isin(stats)])
    out = {}
    key = ["game_id", "nname", "market", "line", "book"]
    base = pd.concat(got["now"], ignore_index=True)
    for k in tables:
        x = pd.concat(got[k], ignore_index=True)
        x = x.merge(base[key + ["our_over", "our_under"]].rename(columns={"our_over": "b_over", "our_under": "b_under"}), on=key, how="inner")
        x = x[x["y"] != x["line"]]
        hit = (x["y"] > x["line"]).astype(float)
        ours, was, mkt = x["our_over"] / (x["our_over"] + x["our_under"]), x["b_over"] / (x["b_over"] + x["b_under"]), x["fair_over"]
        def one(mask):
            d = ((ours - hit) ** 2 - (was - hit) ** 2)[mask]
            cl = d.groupby(x.loc[mask, "game_id"]).sum()
            return {"n": int(mask.sum()), "brier": round(float(((ours - hit) ** 2)[mask].mean()), 5), "books": round(float(((mkt - hit) ** 2)[mask].mean()), 5),
                    "vs_now": round(float(d.mean()), 5), "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / max(mask.sum(), 1)), 5)}
        out[k] = {"pooled": one(pd.Series(True, index=x.index)), **{st: one(x["stat"] == st) for st in stats}}
    return out


def main():
    base = build()
    res = {"ratio_now": {}}
    S, Q, _, ok, _, _ = team_sums(base, "rec_yds", "pass_yds")
    r = (S / Q)[ok & base["played"].astype(bool) & (base["season"] >= 2014)].groupby([base["game_id"], base["team"]]).first()
    res["ratio_now"] = {"mean": round(float(r.mean()), 3), "sd": round(float(r.std()), 3), "team_games": int(len(r))}
    tables = {"now": base}
    for how in ("A", "B"):
        cand, info = reconcile(base, how)
        tables[how] = cand
        res[how] = {"rmse": rmse_table(base, cand), "fitted_a": info}
    res["lines"] = lines(tables)
    for how in ("A", "B"):
        rm = [v for v in res[how]["rmse"].values() if v["changed"]]
        hold = [v for k, v in res[how]["rmse"].items() if v["changed"] and k.endswith("2019 on")]
        res[how]["keep"] = bool(all(v["diff"] < 0 for v in rm) and all(v["diff"] <= -v["se"] for v in hold) and res["lines"][how]["pooled"]["vs_now"] < 0)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "team_totals_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
