"""A probability-weighted starting quarterback in the NFL game model (Nov 11, 2026).

    python experiments/uncertain_qb_test.py          (with ../pff-data: the production model)

THE RULE (approved by Ryan and written before the first run, Nov 11, 2026; Project doc
claude/plan-uncertain-qb-test-2026-11-11.md)
-------------------------------------------------------------------------------------------
Lead (an outside review of the repo): the game model takes one starter per team. When last game's
starter is Questionable the board keeps him until the daily QB check switches it, and historically
he started 60% of the time (qshade.py). Does averaging the two possible starters forecast better?

  flagged team-game   regular season, 2016 on, played: the team's starter from its last game (the
                      engine's last_qb, what the board assumes) is on that week's final injury
                      report for that team as Questionable (roles.practice, report level 1)
  B0  baseline        the production game model (coefficients fitted walk-forward on the usual
                      frame) predicting with the incumbent as that team's quarterback and the
                      bet-time offensive-line input (Out / Doubtful and reserve lists; the
                      "bet-time" version from review_fix_tests T1). Everything else as the usual
                      frame. This is what the board shows before the starter is confirmed.
  W   candidate       the same model with the incumbent and with the backup, averaged by the
                      chance the incumbent starts.
        backup        players._pick_starter's fallback: among the team's quarterbacks expected to
                      play before the week's news (players.build(pregame=True), active_pre) other
                      than the incumbent, the most decayed pass attempts, then the largest share.
                      No such quarterback: an unknown one (the replacement prior).
        chance        the start rate of earlier flagged incumbents (seasons before this one, from
                      2012) at the same Friday practice level (full / limited / none), shrunk to
                      the overall earlier rate with 10 cases; no practice level: the overall rate.
        averaged      margin and total: the weighted mean of the two forecasts (four when both
                      teams are flagged). Win and cover chances: the weighted mean of the chances,
                      not the chance at the averaged margin.
  chances             normal around the raw model's margin, with the standard deviation of the
                      usual walk-forward's margin misses over the previous ten seasons. The same
                      for B0 and W. Raw model, not the blend: the closing line knows the starter.

Measures, flagged games only, W against B0, paired by game, with standard errors
(tuning 2016-2018, holdout 2019 on):
  1  average margin miss
  2  average total miss
  3  Brier score of the home win chance (ties left out)
  4  Brier score of the home cover chance at nfelo's opening spread (pushes and games without an
     opener left out)
KEEP (CONVENTIONS.md, Nov 11, 2026): W better on all four in both periods and the holdout
improvement at least 1 paired standard error on each; provisional, rechecked on the next full
season. Anything else is not kept; a mixed result goes to Ryan as a trade-off.
Reported, not in the rule: all games 2019 on (they differ only in the flagged ones); how often the
incumbent started by practice level; how often the backup pick was the quarterback who started when
the incumbent sat; the same four measures knowing the actual starter (the usual backtest's view).

SECOND QUESTION (a bet-sheet rule, not a model change). Every played regular-season game from 2016
with an opening spread; the bet-time forecast (B0 in flagged games, the usual forecast with the
bet-time line input elsewhere); the model's side against the opener, graded at -110, pushes out.
Board-flagged games (an incumbent Questionable after a limited or no Friday practice, the
qshade.qb_flags trigger) against all other games: record, return, and the weight the model earns
against the opener (the slope of result - opener on model - opener).
PROPOSE HOLDING model-tier bets on a flagged game until the starter check clears it only if, in
the holdout, the flagged games' return is lower than the other games' by at least 1 standard error
of the difference, and in tuning it is not higher. That would change the bet sheet: Ryan's call,
asked separately. Price-tier and opener-tier rows are not part of it.

Props: no player projection is touched, so no prop moves.

Known limits: about 45 flagged games in tuning; the report is Friday's, so this is the late-week
case; the opener stands in for the line the board bets into mid-week.

Writes experiments/out/uncertain_qb_test.json.
"""
from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
FIRST, TUNE_TO, HOLD = 2016, 2018, 2019
RATE_FROM, SHRINK, SD_LOOKBACK = 2012, 10, 10
PERIODS = {"tuning 2016-18": lambda s: (s >= FIRST) & (s <= TUNE_TO), "holdout 2019 on": lambda s: s >= HOLD}


def paired(a, b) -> dict:
    """mean and standard error of a - b, and n (rows with either missing are dropped)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return {"diff": None, "se": None, "n": int(len(d))}
    return {"diff": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4), "n": int(len(d))}


def load():
    from nflmodel import data, extra, game_roles, injuries, model, pff_qb, players, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    team_extra, qb_extra = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False)
    prior = pff_qb.load(qb, sorted(games.season.unique()))
    print("quarterback prior:", "PFF grades" if prior is not None else "flat (no PFF data: NOT the production model)", flush=True)
    engine = ExtRatingEngine(tg, qb, team_extra, qb_extra, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = players.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = players.player_rows(players.team_games(f0, pg), pg, avail, pinfo)
    wf = game_roles.attach(base, game_roles.team_features(rows, R, avail))
    # the bet-time offensive-line input (review_fix_tests T1): a game-day inactive does not count
    src = inspect.getsource(injuries._season)
    a = "rw = [w for w in roster_weeks.get(team, []) if w <= wk]"
    b = "for path in paths.values():\n        path.unlink(missing_ok=True)\n"
    assert a in src and b in src, "injuries._season changed: update the bet-time patch"
    ns = dict(vars(injuries))
    exec(src.replace(a, "rw = [w for w in roster_weeks.get(team, []) if w < wk]").replace(b, "").replace("def _season(", "def _season_bettime("), ns)
    parts = []
    for s in range(injuries.FIRST, int(games[games.played].season.max()) + 1):
        try:
            parts.append(ns["_season_bettime"](s, games))
        except Exception as exc:
            print(f"  bet-time injuries {s}: skipped ({exc})")
    bt = model.build_frame(games, engine, injuries=pd.concat(parts, ignore_index=True))[["game_id", "d_chg_ol"]].rename(columns={"d_chg_ol": "d_chg_ol_bt"})
    wf = wf.merge(bt, on="game_id", how="left")
    wf["d_chg_ol_bt"] = wf["d_chg_ol_bt"].fillna(0.0)
    try:
        pr = injuries.load_pass_rush(games, verbose=False)
    except Exception:
        pr = None
    ptab = players.build(f0, pg, pinfo, avail, inj, pr, games=games, pregame=True)
    return dict(games=games, engine=engine, wf=wf, R=R, ptab=ptab, qb=qb, prior=prior, openers=data.load_openers(refresh=False))


def flagged_cases(L) -> pd.DataFrame:
    """One row per flagged team-game: the incumbent, his practice level, whether he started, the backup."""
    games, engine, R = L["games"], L["engine"], L["R"]
    g = games[games.played & (games.season >= RATE_FROM)]
    if "game_type" in g:
        g = g[g["game_type"].eq("REG")]
    lq = engine.last_qb.sort_values("order")
    pr = R["practice"]
    pr = pr[pr["report"] == 1].set_index(["season", "week", "team", "player_id"])["practice"]
    pt = L["ptab"]
    act = (pt["active_pre"] if "active_pre" in pt else pt["active"]) == 1
    q = pt[(pt["group"] == "QB") & act]
    qb_by = {k: d for k, d in q.groupby(["game_id", "team"])}
    out = []
    for r in g.itertuples(index=False):
        for side in ("home", "away"):
            team, actual = getattr(r, side + "_team"), getattr(r, side + "_qb_id")
            prev = lq[(lq["team"] == team) & (lq["order"] < r.order)]
            if not len(prev) or not isinstance(actual, str):
                continue
            inc = prev["qb_id"].iloc[-1]
            key = (r.season, r.week, team, inc)
            if key not in pr.index:
                continue
            lvl = pr.loc[key]
            lvl = float(lvl.iloc[0]) if isinstance(lvl, pd.Series) else float(lvl)
            d = qb_by.get((r.game_id, team))
            backup = None
            if d is not None:
                d = d[d["player_id"] != inc]
                if len(d):
                    backup = d.sort_values(["pass_att_sum", "att_share_raw"], ascending=False, na_position="last")["player_id"].iloc[0]
            out.append({"game_id": r.game_id, "season": int(r.season), "week": int(r.week), "side": side, "team": team, "incumbent": inc,
                        "practice": None if np.isnan(lvl) else int(lvl), "started": int(actual == inc), "actual": actual,
                        "backup": backup, "backup_right": None if actual == inc else int(backup == actual)})
    c = pd.DataFrame(out)
    # the chance the incumbent starts: earlier seasons only, by practice level, shrunk to the overall rate
    ch = []
    for r in c.itertuples():
        past = c[c.season < r.season]
        if not len(past):
            ch.append(np.nan)
            continue
        overall = past.started.mean()
        if r.practice is None or pd.isna(r.practice):
            ch.append(overall)
            continue
        same = past[past.practice == r.practice]
        ch.append((same.started.sum() + SHRINK * overall) / (len(same) + SHRINK))
    c["chance"] = ch
    return c


def scenario_features(L, c: pd.DataFrame) -> dict:
    """The engine's features for the flagged games under each choice of quarterback.
    II: every flagged side its incumbent; KK: its backup; KI: home backup, away incumbent; IK: the
    reverse; AA: who actually started (a check against the usual frame, and the reported oracle)."""
    games, engine = L["games"], L["engine"]
    sub = games[games.game_id.isin(c.game_id)].copy()
    pick = {(r.game_id, r.side): r for r in c.itertuples()}
    out = {}
    for name in ("AA", "II", "KK", "KI", "IK"):
        g = sub.copy()
        if name != "AA":
            for side, how in (("home", name[0]), ("away", name[1])):
                col = side + "_qb_id"
                vals = []
                for gid, cur in zip(g["game_id"], g[col]):
                    r = pick.get((gid, side))
                    vals.append(cur if r is None else r.incumbent if how == "I" else (r.backup if isinstance(r.backup, str) else "unknown-backup"))
                g[col] = vals
        out[name] = engine.features(g).set_index("game_id")
    return out


def main() -> None:
    from nflmodel import model
    L = load()
    wf, games = L["wf"], L["games"]
    c = flagged_cases(L)
    c = c[c.chance.notna()]
    res = {"qb_prior": "pff" if L["prior"] is not None else "flat", "cases_from_2012": int(len(c))}
    test = c[c.season >= FIRST]
    res["flagged_team_games"] = int(len(test))
    res["start_rate"] = {"all": round(float(test.started.mean()), 3),
                         **{{0: "full", 1: "limited", 2: "none"}[k]: {"n": int(len(d)), "started": round(float(d.started.mean()), 3), "avg_chance_used": round(float(d.chance.mean()), 3)}
                            for k, d in test.dropna(subset=["practice"]).groupby("practice")}}
    sat = test[test.started == 0]
    res["backup_pick_right_when_incumbent_sat"] = {"n": int(len(sat)), "right": round(float(sat.backup_right.mean()), 3) if len(sat) else None}

    feats = scenario_features(L, test)
    MF, TF = model.MARGIN_FEATS, model.TOTAL_FEATS
    base = wf.set_index("game_id")
    for col in MF + TF:
        if col not in base:
            base[col] = 0.0
    eng_cols = [k for k in feats["AA"].columns if k in base.columns and k not in ("home_qb_id", "away_qb_id")]
    chk = (feats["AA"][eng_cols].astype(float) - base.loc[feats["AA"].index, eng_cols].astype(float)).abs().max().max()
    assert chk < 1e-6, f"recomputed features differ from the usual frame ({chk})"

    def X(rows: pd.DataFrame, cols):
        cols = [k if k != "d_chg_ol" else "d_chg_ol_bt" for k in cols]
        return rows[cols].to_numpy(float)

    def with_feats(name, ids):
        x = base.loc[ids].copy()
        x[eng_cols] = feats[name].loc[ids, eng_cols].to_numpy()
        return x

    # the usual walk-forward (for the spread of its misses) and the per-season fits
    usual, _ = model.walk_forward(wf)
    usual = usual[usual.played]
    usual["err"] = usual["result"] - usual["model_margin"]
    ch = {(r.game_id, r.side): r.chance for r in test.itertuples()}
    rows = []
    allg = []
    for s in sorted(v for v in base.season.unique() if v >= FIRST):
        tr = base[(base.season < s) & base.played]
        m = model.Linear(20.0, intercept=False).fit(tr[MF].to_numpy(float), tr["result"].to_numpy(float))
        t = model.Linear(20.0).fit(tr[TF].to_numpy(float), tr["total"].to_numpy(float))
        past = usual[(usual.season < s) & (usual.season >= s - SD_LOOKBACK)]
        sd = float(np.sqrt((past["err"] ** 2).mean()))
        ids = [g for g in test[test.season == s].game_id.unique()]
        te = base[(base.season == s) & base.played]
        if "game_type" in te:
            te = te[te["game_type"].eq("REG")]
        bet = pd.DataFrame({"game_id": te.index, "season": s, "m": m.predict(X(te, MF)), "result": te["result"].to_numpy(float)})   # the usual forecast, bet-time line input
        if ids:
            P = {n: (m.predict(X(with_feats(n, ids), MF)), t.predict(X(with_feats(n, ids), TF))) for n in ("AA", "II", "KK", "KI", "IK")}
            for i, gid in enumerate(ids):
                ph, pa = ch.get((gid, "home"), 1.0), ch.get((gid, "away"), 1.0)
                w = {"II": ph * pa, "KI": (1 - ph) * pa, "IK": ph * (1 - pa), "KK": (1 - ph) * (1 - pa)}
                g = base.loc[gid]
                rows.append({"game_id": gid, "season": int(s), "sd": sd, "result": float(g["result"]), "total": float(g["total"]),
                             "m_b0": P["II"][0][i], "t_b0": P["II"][1][i], "m_aa": P["AA"][0][i], "t_aa": P["AA"][1][i],
                             "m_w": sum(w[n] * P[n][0][i] for n in w), "t_w": sum(w[n] * P[n][1][i] for n in w),
                             **{f"w_{n}": w[n] for n in w}, **{f"m_{n}": P[n][0][i] for n in w}})
            b0 = {r["game_id"]: r["m_b0"] for r in rows if r["season"] == s}
            bet["m"] = [b0.get(g, v) for g, v in zip(bet["game_id"], bet["m"])]
        allg.append(bet)
    d = pd.DataFrame(rows).merge(L["openers"][["game_id", "open_spread"]], on="game_id", how="left")
    res["flagged_games"] = int(len(d))

    # chances: the weighted mean of the chances under each quarterback
    combos = ("II", "KI", "IK", "KK")
    d["win"] = np.where(d.result > 0, 1.0, np.where(d.result < 0, 0.0, np.nan))
    d["cov"] = np.where(d.open_spread.isna() | (d.result == d.open_spread), np.nan, (d.result > d.open_spread).astype(float))
    d["pw_b0"], d["pw_aa"] = ndtr(d.m_b0 / d.sd), ndtr(d.m_aa / d.sd)
    d["pw_w"] = sum(d[f"w_{n}"] * ndtr(d[f"m_{n}"] / d.sd) for n in combos)
    d["pc_b0"], d["pc_aa"] = ndtr((d.m_b0 - d.open_spread) / d.sd), ndtr((d.m_aa - d.open_spread) / d.sd)
    d["pc_w"] = sum(d[f"w_{n}"] * ndtr((d[f"m_{n}"] - d.open_spread) / d.sd) for n in combos)
    loss = {"margin miss": lambda v: (d[f"m_{v}"] - d.result).abs(), "total miss": lambda v: (d[f"t_{v}"] - d.total).abs(),
            "win Brier": lambda v: (d[f"pw_{v}"] - d.win) ** 2, "cover Brier (opener)": lambda v: (d[f"pc_{v}"] - d.cov) ** 2}
    res["measures"], better, se_ok = {}, [], []
    for per, fn in PERIODS.items():
        k = fn(d.season)
        res["measures"][per] = {}
        for name, f in loss.items():
            b0, w, aa = f("b0")[k], f("w")[k], f("aa")[k]
            p = paired(w, b0)
            res["measures"][per][name] = {"B0": round(float(b0.mean()), 4), "W": round(float(w.mean()), 4), **p,
                                          "knowing the starter (not in the rule)": round(float(aa.mean()), 4)}
            better.append(p["diff"] is not None and p["diff"] < 0)
            if per.startswith("holdout"):
                se_ok.append(p["diff"] is not None and p["se"] and -p["diff"] >= p["se"])
    res["keep"] = bool(all(better) and all(se_ok))
    res["verdict"] = "KEEP (provisional)" if res["keep"] else "not kept" if not any(better) else "not kept: mixed or under 1 SE (a trade-off for Ryan)"

    # all games 2019 on (not in the rule): they differ only in the flagged games
    A = pd.concat(allg, ignore_index=True).merge(L["openers"][["game_id", "open_spread"]], on="game_id", how="left")
    mw = d.set_index("game_id")["m_w"]
    A["m_w"] = [mw.get(g, v) for g, v in zip(A.game_id, A.m)]
    h = A[A.season >= HOLD]
    res["all_games_2019_on"] = {"n": int(len(h)), "B0": round(float((h.m - h.result).abs().mean()), 4), "W": round(float((h.m_w - h.result).abs().mean()), 4),
                                **paired((h.m_w - h.result).abs(), (h.m - h.result).abs())}

    # ---- second question: the model's side against the opener, board-flagged games against the rest
    board = set(test[test.practice.isin([1, 2])].game_id)
    S = A[A.open_spread.notna()].copy()
    S["edge"] = S.m - S.open_spread
    S["act"] = S.result - S.open_spread
    S = S[(S.edge != 0) & (S.act != 0)]
    S["won"] = (np.sign(S.edge) == np.sign(S.act)).astype(float)
    S["ret"] = np.where(S.won == 1, 100 / 110, -1.0)
    S["flag"] = S.game_id.isin(board)

    def block(x: pd.DataFrame) -> dict:
        if len(x) < 2:
            return {"n": int(len(x))}
        w = float(x.edge @ x.act / (x.edge @ x.edge))
        r = x.act - w * x.edge
        return {"n": int(len(x)), "record": f"{int(x.won.sum())}-{int(len(x) - x.won.sum())}", "win_pct": round(float(x.won.mean()), 4),
                "return": round(float(x.ret.mean()), 4), "return_se": round(float(x.ret.std(ddof=1) / np.sqrt(len(x))), 4),
                "weight": round(w, 3), "weight_se": round(float(np.sqrt((r ** 2 * x.edge ** 2).sum()) / (x.edge @ x.edge)), 3)}

    res["second_question"] = {}
    for per, fn in PERIODS.items():
        x = S[fn(S.season)]
        f, o = block(x[x.flag]), block(x[~x.flag])
        gap = se = None
        if "return" in f and "return" in o:
            gap, se = round(f["return"] - o["return"], 4), round(float(np.hypot(f["return_se"], o["return_se"])), 4)
        res["second_question"][per] = {"board-flagged": f, "other games": o, "return gap (flagged - other)": gap, "gap_se": se}
    t_, h_ = (res["second_question"][p] for p in PERIODS)
    res["propose_hold"] = bool(h_["gap_se"] and h_["return gap (flagged - other)"] <= -h_["gap_se"]
                               and t_["return gap (flagged - other)"] is not None and t_["return gap (flagged - other)"] <= 0)
    res["props"] = "no player projection is touched; no prop moves"

    OUT.mkdir(exist_ok=True)
    (OUT / "uncertain_qb_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
