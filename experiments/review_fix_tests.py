"""Tests for the model fixes from the Nov 11, 2026 review (T1, T2, T4 for the NFL; T3, T4 for college).

    PFF_DATA_DIR=../pff-data python experiments/review_fix_tests.py nfl
    PFF_DATA_DIR=../pff-data python experiments/review_fix_tests.py cfb

Rules, written before any run (Project doc claude/plan-review-fixes-tests-2026-11-11.md). The keep
rule: kept only if better on every measure listed and worse on none; a mixed result is Ryan's call.

T1  NFL d_chg_ol built the same way for played and upcoming games. Today the played-game version
    counts game-day inactives, the upcoming-game version only Out/Doubtful and longer-term lists.
    Candidate: the played-game version by the upcoming-game rule (the roster of the latest earlier
    week; a game-day inactive does not count). Both are scored on the prediction the board can make:
    the bet-time input at prediction time. Measure: average margin miss, 2014-18 and 2019 on.
    Keep the candidate if lower on both.
T2  NFL totals cap (config.TOTAL_WEIGHT_CAP, 0.05). Candidates 0.15, 0.25, none. Measures: the
    market-adjusted total's average miss on 2014-18 and 2019 on, and the return on the totals the
    version flags as positive EV on 2019 on. Keep the smallest cap that lowers the miss on both
    periods and whose flagged totals returned more than zero on 2019 on; else the cap stays.
T3  College first-FBS-season feature (d_newfbs: home minus away, 1 in a team's first season as an
    FBS team in the games file, 2015 on). Measure: average margin miss, all games, 2017-19 and
    2021-25. Keep if lower on both. Reported, not in the rule: games with a first-year team; 2026.
T4  The blend weight sees this season: the last eight seasons plus this season's games before the
    week being priced. Measure: the market-adjusted margin's average miss, NFL 2014-18 and 2019 on,
    college 2017-19 and 2021-25. Keep per board if lower on both.

Writes experiments/out/review_fix_tests_<league>.json.
"""
from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"


def paired(a, b):
    """mean and standard error of a - b (per-game misses), and n."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    return {"diff": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4), "n": int(len(d))}


def miss(pred, actual):
    return np.abs(np.asarray(pred, float) - np.asarray(actual, float))


def weight_in_season(df, model_col, line_col, actual_col, lookback, first_oos, blend_weight):
    """proj = line + w (model - line), with w from the last `lookback` seasons plus this season's
    games in earlier weeks (T4). Returns (projection, weight per row)."""
    proj, wcol = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    for (s, o), wk in df.groupby(["season", "order"]):
        past = df[df.played & (((df.season < s) & (df.season >= max(first_oos, s - lookback))) | ((df.season == s) & (df.order < o)))]
        w = blend_weight(*(past[c].to_numpy(float) for c in (model_col, line_col, actual_col)))
        has = wk[line_col].notna()
        proj.loc[wk.index] = np.where(has, wk[line_col] + w * (wk[model_col] - wk[line_col]), wk[model_col])
        wcol.loc[wk.index] = w
    return proj, wcol


# ------------------------------------------------------------------------------------------ NFL
def nfl() -> dict:
    from nflmodel import backtest, bets as B, config as C, data, extra, game_roles, injuries, model, pff_qb, players, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    team_extra, qb_extra = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False)
    prior = pff_qb.load(qb, sorted(games.season.unique()))
    print("quarterback prior:", "PFF grades" if prior is not None else "flat (no PFF data: NOT the production model)")
    engine = ExtRatingEngine(tg, qb, team_extra, qb_extra, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = players.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = players.player_rows(players.team_games(f0, pg), pg, avail, pinfo)
    wf = game_roles.attach(base, game_roles.team_features(rows, R, avail))
    out = {"qb_prior": "pff" if prior is not None else "flat"}
    per = {"2014-18": lambda d: (d.season >= 2014) & (d.season < 2019), "2019 on": lambda d: d.season >= 2019}

    # ---- T1: the bet-time version of the injury groups for played games
    src = inspect.getsource(injuries._season)
    a = "rw = [w for w in roster_weeks.get(team, []) if w <= wk]"
    b = "for path in paths.values():\n        path.unlink(missing_ok=True)\n"
    assert a in src and b in src, "injuries._season changed: update the T1 patch"
    ns = dict(vars(injuries))
    exec(src.replace(a, "rw = [w for w in roster_weeks.get(team, []) if w < wk]").replace(b, "").replace("def _season(", "def _season_bettime("), ns)
    parts = []
    for s in range(injuries.FIRST, int(games[games.played].season.max()) + 1):
        try:
            parts.append(ns["_season_bettime"](s, games))
        except Exception as exc:
            print(f"  bet-time injuries {s}: skipped ({exc})")
    inj_bt = pd.concat(parts, ignore_index=True)
    bt = model.build_frame(games, engine, injuries=inj_bt)[["game_id", "d_chg_ol"]].rename(columns={"d_chg_ol": "d_chg_ol_bt"})
    x = wf.merge(bt, on="game_id", how="left")
    x["d_chg_ol_bt"] = x["d_chg_ol_bt"].fillna(0.0)
    MF = model.MARGIN_FEATS
    MF_bt = [m if m != "d_chg_ol" else "d_chg_ol_bt" for m in MF]
    x["m_now"], x["m_cand"] = np.nan, np.nan
    for s in sorted(v for v in x.season.unique() if v >= model.FIRST_OOS):
        tr, te = x[(x.season < s) & x.played], x.season == s
        m_now = model.Linear(20.0, intercept=False).fit(tr[MF].to_numpy(float), tr["result"].to_numpy(float))       # fitted on the full-information input
        m_cand = model.Linear(20.0, intercept=False).fit(tr[MF_bt].to_numpy(float), tr["result"].to_numpy(float))    # fitted on the bet-time input
        x.loc[te, "m_now"] = m_now.predict(x.loc[te, MF_bt].to_numpy(float))     # both predict from the bet-time input
        x.loc[te, "m_cand"] = m_cand.predict(x.loc[te, MF_bt].to_numpy(float))
        if s == x.season.max():
            out["t1_coef"] = {"now": m_now.raw_coefs(MF)["d_chg_ol"], "candidate": m_cand.raw_coefs(MF_bt)["d_chg_ol_bt"]}
    p = x[x.played & x.spread_line.notna() & (x.season >= 2014)]
    out["t1"] = {k: {"now": round(float(miss(p[f(p)].m_now, p[f(p)].result).mean()), 4), "candidate": round(float(miss(p[f(p)].m_cand, p[f(p)].result).mean()), 4),
                     **paired(miss(p[f(p)].m_cand, p[f(p)].result), miss(p[f(p)].m_now, p[f(p)].result))} for k, f in per.items()}
    out["t1"]["share_of_games_input_differs"] = round(float((p.d_chg_ol != p.d_chg_ol_bt).mean()), 3)
    out["t1"]["keep"] = all(out["t1"][k]["diff"] < 0 for k in per)

    # ---- T2: the totals cap
    out["t2"] = {}
    for cap in (0.05, 0.15, 0.25, 1.0):
        C.TOTAL_WEIGHT_CAP = cap
        f, info = model.walk_forward(wf)
        graded, _ = backtest.grade_all(games, f)
        g = graded[(graded.source == "proj") & (graded.market == "Total") & (graded.ev > 0) & (graded.season >= C.HOLDOUT_FROM) & graded.profit.notna()]
        pl = f[f.played & f.total_line.notna() & (f.season >= 2014)]
        out["t2"][str(cap)] = {**{k: round(float(miss(pl[fn(pl)].proj_total, pl[fn(pl)].total).mean()), 4) for k, fn in per.items()},
                               "line": {k: round(float(miss(pl[fn(pl)].total_line, pl[fn(pl)].total).mean()), 4) for k, fn in per.items()},
                               "flagged_holdout": {"n": int(len(g)), "roi": round(float(g.profit.mean()), 4) if len(g) else None,
                                                   "se": round(float(g.profit.std(ddof=1) / np.sqrt(len(g))), 4) if len(g) > 1 else None,
                                                   "win_pct": round(float((g.outcome > 0).sum() / max(((g.outcome != 0).sum()), 1)), 4) if len(g) else None},
                               "weights": {int(s): info[s]["w_total"] for s in sorted(info) if s >= 2014}}
        if cap == 0.05:
            base_f = f
    C.TOTAL_WEIGHT_CAP = 0.05
    now = out["t2"]["0.05"]
    out["t2"]["keep"] = next((c for c in ("0.15", "0.25", "1.0") if all(out["t2"][c][k] < now[k] for k in per)
                              and (out["t2"][c]["flagged_holdout"]["roi"] or -1) > 0), None)

    # ---- T4: the blend weight sees this season
    f = base_f.copy()
    f["proj_in"], f["w_in"] = weight_in_season(f, "model_margin", "spread_line", "result", C.BLEND_LOOKBACK, model.FIRST_OOS, model._blend_weight)
    if C.FINAL_WEEK_MARKET:
        fw = f["final_week"].fillna(False).astype(bool) & f.spread_line.notna()
        f.loc[fw, "proj_in"] = f.loc[fw, "spread_line"]
    p = f[f.played & f.spread_line.notna() & (f.season >= 2014)]
    out["t4"] = {k: {"now": round(float(miss(p[fn(p)].proj_margin, p[fn(p)].result).mean()), 4), "candidate": round(float(miss(p[fn(p)].proj_in, p[fn(p)].result).mean()), 4),
                     "line": round(float(miss(p[fn(p)].spread_line, p[fn(p)].result).mean()), 4),
                     **paired(miss(p[fn(p)].proj_in, p[fn(p)].result), miss(p[fn(p)].proj_margin, p[fn(p)].result))} for k, fn in per.items()}
    out["t4"]["keep"] = all(out["t4"][k]["diff"] < 0 for k in per)
    cur = f[f.season == f.season.max()]
    out["t4"]["this_season_weight"] = {"now": round(float(cur.w_margin.iloc[0]), 3), "candidate_latest_week": round(float(cur.sort_values("order").w_in.iloc[-1]), 3)}
    return out


# -------------------------------------------------------------------------------------- college
def cfb() -> dict:
    from cfbmodel import config as C, data, extra, model, pff_returning as PFFRET
    from cfbmodel.ratings import CFBRatingEngine
    from nflmodel.model import _blend_weight
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games["wind_source"] = np.where(games["played"], "recorded", "")
    inputs = extra.load_inputs(games, refresh_current=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games, extra=inputs["rated"])
    base = PFFRET.add(model.build_frame(games, engine, inputs=inputs), verbose=True)
    out = {"pffret": bool((base["d_pffret"] != 0).any())}
    per = {"2017-19": lambda d: (d.season >= 2017) & (d.season <= 2019), "2021-25": lambda d: (d.season >= 2021) & (d.season <= 2025)}

    # ---- T3: first FBS season
    long = pd.concat([games[["season", "home_team"]].rename(columns={"home_team": "team"}), games[["season", "away_team"]].rename(columns={"away_team": "team"})])
    first = long[long.team != C.POOL].groupby("team").season.min()
    first_data = int(games.season.min())
    new = {(t, int(s)) for t, s in first.items() if s > first_data}
    out["t3_new_teams"] = sorted(f"{t} {s}" for t, s in new)
    isnew = lambda team, season: np.array([(t, int(s)) in new for t, s in zip(team, season)], float)
    x = base.copy()
    x["d_newfbs"] = isnew(x.home_team, x.season) - isnew(x.away_team, x.season)
    f_now, _ = model.walk_forward(x)
    f_cand, info = model.walk_forward(x, margin_feats=model.MARGIN_FEATS + ["d_newfbs"])
    key = ["game_id"]
    m = f_now[key + ["season", "played", "spread_line", "result", "model_margin", "proj_margin", "w_margin", "order"]].merge(
        f_cand[key + ["model_margin", "proj_margin", "d_newfbs"]], on=key, suffixes=("", "_c"))
    p = m[m.played & m.spread_line.notna()]
    def block(d):
        return {"n": int(len(d)), "now": round(float(miss(d.model_margin, d.result).mean()), 4), "candidate": round(float(miss(d.model_margin_c, d.result).mean()), 4),
                "blend_now": round(float(miss(d.proj_margin, d.result).mean()), 4), "blend_candidate": round(float(miss(d.proj_margin_c, d.result).mean()), 4),
                "line": round(float(miss(d.spread_line, d.result).mean()), 4), **(paired(miss(d.model_margin_c, d.result), miss(d.model_margin, d.result)) if len(d) > 1 else {})}
    out["t3"] = {k: block(p[fn(p)]) for k, fn in per.items()}
    out["t3"]["2026 (not in the rule)"] = block(p[p.season == 2026])
    out["t3"]["first-year games, 2017-25"] = block(p[(p.d_newfbs != 0) & (p.season >= 2017) & (p.season <= 2025)])
    out["t3"]["first-year games, 2026"] = block(p[(p.d_newfbs != 0) & (p.season == 2026)])
    out["t3"]["coef_by_season"] = {int(s): info[s]["margin_coefs"]["d_newfbs"] for s in sorted(info)}
    out["t3"]["keep"] = all(out["t3"][k]["diff"] < 0 for k in per)

    # ---- T4: the blend weight sees this season
    f = f_now.copy()
    f["proj_in"], f["w_in"] = weight_in_season(f, "model_margin", "spread_line", "result", C.BLEND_LOOKBACK, model.FIRST_OOS, _blend_weight)
    p = f[f.played & f.spread_line.notna()]
    def b4(d):
        return {"now": round(float(miss(d.proj_margin, d.result).mean()), 4), "candidate": round(float(miss(d.proj_in, d.result).mean()), 4),
                "line": round(float(miss(d.spread_line, d.result).mean()), 4), **paired(miss(d.proj_in, d.result), miss(d.proj_margin, d.result))}
    out["t4"] = {k: b4(p[fn(p)]) for k, fn in per.items()}
    out["t4"]["2026 (not in the rule)"] = b4(p[p.season == 2026])
    out["t4"]["keep"] = all(out["t4"][k]["diff"] < 0 for k in per)
    cur = f[f.season == f.season.max()].sort_values("order")
    out["t4"]["this_season_weight"] = {"now": round(float(cur.w_margin.iloc[0]), 3), "candidate_by_week": {int(o % 100): round(float(w), 3) for o, w in cur.groupby("order").w_in.first().items()}}
    return out


if __name__ == "__main__":
    league = sys.argv[1] if len(sys.argv) > 1 else "nfl"
    res = nfl() if league == "nfl" else cfb()
    OUT.mkdir(exist_ok=True)
    (OUT / f"review_fix_tests_{league}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
