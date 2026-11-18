"""Pre-set test 3 from the "Hidden Stats" guide: interception luck from PFF turnover-worthy plays (NFL).

    PFF_DATA_DIR=<pff-data clone> python experiments/twp_int_luck_test.py

Rule written before running (Project doc claude/plan-air-yards-wopr-twp-tests-2026-11-18.md).

3a, the quarterback's interceptions projection (today: pass attempts x his own interception rate
shrunk to the league's, no extra calibration terms):
  twp_gap   his turnover-worthy plays per dropback (PFF passing_summary), the decay and matching of
            nflmodel/pff_passing.py (8-game half-life, last season at 60%), shrunk to the 2014-18
            league rate by 200 dropbacks, rate / league - 1; zero with no history
  Entered in players.CAL_EXTRA["ints"] (raw projection x column). One version.
  Measures (6): interceptions RMSE, projected starters who started with 10+ projected attempts,
  game-day and bet-time, tuning 2015-18 (2014 is PFF's burn-in) and holdout 2019-25 (4); Brier of the
  chance of 1+ interception (Poisson on the projection) against what happened, game-day, both
  periods (2). The history has no interception lines, so no book-line measure.
3b, the game model. As of each game, decayed over earlier games (same decay), each shrunk toward zero
  by 200 dropbacks (set before the run: the plan gave no shrink, and a backup's handful of dropbacks
  would otherwise swamp the term):
  QB side    (interceptions thrown - league rate x turnover-worthy plays) per dropback, the game's
             starting quarterback (the id the game model's own quarterback term uses)
  defense    (interceptions made - league rate x (interceptions + dropped interceptions)) per opponent
             dropback, summed from PFF defense_coverage
  League rates come from earlier seasons only (2014 uses its own; it is burn-in).
  Version A: d_il_qb in the margin model and s_il_qb in the total model (home minus away, and sum).
  Version B: A plus d_il_def and s_il_def.
  Added next to the production features (model.walk_forward margin_feats / total_feats, as
  experiments/turnover_free_test.py does), on the production frame (injuries and role terms attached).
  Measures (4): the raw model's average miss on margin and on total, tuning 2015-18 and holdout
  2019-25, paired by game.
Rule (3a and 3b each on its own): 3b's version is chosen on tuning alone (better on both tuning
  measures, then the lower tuning margin miss). Kept only if better on every measure and each holdout
  gain >= 1 paired SE. Mixed, or under 1 SE: a trade-off for Ryan. No version better on both tuning
  measures: nothing is carried to the holdout.
Reported, not used: the record against the closing spread of team-games in the top and bottom fifth
  of net interception luck (defense minus quarterback), 2015-25; the 2026 live-log interception lines.

Writes experiments/out/twp_int_luck_test.json (scores only, never PFF rows).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from cfbmodel import pff_data  # noqa: E402
from nflmodel import config as C, pff_passing as PP, pff_qb, pff_routes, players as PL  # noqa: E402
import air_yards_wopr_test as AW  # noqa: E402

OUT = HERE / "out" / "twp_int_luck_test.json"
TUNE, HOLD = (2015, 2018), (2019, 2025)
SHRINK = 200.0


def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


# ------------------------------------------------------------------ 3a

def twp_gap(table: pd.DataFrame) -> np.ndarray:
    last = int(table["season"].max())
    keys = table[["game_id", "player_id", "season", "week", "t", "played", "active"]].copy()
    pw = keys[keys["played"].astype(bool) & (keys["active"] == 1)][["player_id", "season", "week", "t"]].drop_duplicates()
    pa = _num(PP._load("passing_summary", last), ["turnover_worthy_plays", "dropbacks"])
    pa = pa.merge(pff_routes.pff_ids(), left_on="player_id", right_on="pff_id", how="inner")
    lg = pa[pa.season.isin(PP.LEAGUE)]
    rate = float(lg.turnover_worthy_plays.sum() / lg.dropbacks.sum())
    p = pa.groupby(["nfl_id", "season", "week"], as_index=False)[["turnover_worthy_plays", "dropbacks"]].sum().rename(columns={"nfl_id": "player_id"})
    p = p.merge(pw, on=["player_id", "season", "week"])
    s = PP.decayed(keys, "player_id", p, "player_id", ["turnover_worthy_plays", "dropbacks"])
    return np.where(s[:, 1] > 0, ((s[:, 0] + SHRINK * rate) / (s[:, 1] + SHRINK)) / rate - 1, 0.0), rate


def qb_mask(p, per):
    return (p["played"].astype(bool) & (p["active"] == 1) & (p["group"] == "QB") & (p["pass_att_proj"] >= 10)
            & (p["started"] == 1) & p["season"].between(*per) & p["ints"].notna()).to_numpy()


def paired(e0, e1, games, root=True) -> dict:
    m0, m1 = float(e0.mean()), float(e1.mean())
    cl = pd.Series(e1 - e0).groupby(np.asarray(games)).sum()
    se = float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(e0))
    if root:
        r0, r1 = np.sqrt(m0), np.sqrt(m1)
        return {"n": int(len(e0)), "now": round(r0, 5), "candidate": round(r1, 5), "diff": round(r1 - r0, 6), "se": round(se / (2 * r0), 6)}
    return {"n": int(len(e0)), "now": round(m0, 5), "candidate": round(m1, 5), "diff": round(m1 - m0, 6), "se": round(se, 6)}


def run_3a() -> dict:
    T = AW.tables()
    R = T["R"]
    gap, rate = twp_gap(T["gameday"])
    g = T["gameday"].assign(twp_gap=gap)
    b = T["bettime"].merge(g[["game_id", "player_id", "twp_gap"]], on=["game_id", "player_id"], how="left")
    b["twp_gap"] = b["twp_gap"].fillna(0.0)
    tabs = {"gameday": g, "bettime": b}
    w = g[g.season.between(*HOLD) & g.played.astype(bool) & (g.active == 1) & (g.group == "QB") & (g.started == 1)]
    res = {"league_twp_per_dropback_2014_18": round(rate, 4), "coverage_starters_2019_25": round(float((w.twp_gap != 0).mean()), 3),
           "spread_of_twp_gap": round(float(w.twp_gap.std()), 3)}
    now = {k: AW.project(tabs[k], R, {}, []) for k in tabs}
    cand = {k: AW.candidate(now[k], AW.project(tabs[k], R, {"ints": ["twp_gap"]}, []), ("ints",)) for k in tabs}
    ms = {}
    for k in tabs:
        for name, per in (("tuning 2015-18", TUNE), ("holdout 2019-25", HOLD)):
            m = qb_mask(now[k], per)
            y = now[k]["ints"].to_numpy(float)[m]
            p0, p1 = now[k]["ints_proj"].to_numpy(float)[m], cand[k]["ints_proj"].to_numpy(float)[m]
            gm = now[k]["game_id"].to_numpy()[m]
            ms[f"RMSE {k}, {name}"] = paired((p0 - y) ** 2, (p1 - y) ** 2, gm)
            if k == "gameday":
                hit = (y >= 1).astype(float)
                c0, c1 = 1 - np.exp(-np.clip(p0, 0, None)), 1 - np.exp(-np.clip(p1, 0, None))
                ms[f"Brier 1+ interception, {name}"] = {**paired((c0 - hit) ** 2, (c1 - hit) ** 2, gm, root=False), "threw_one": round(float(hit.mean()), 4)}
    res["measures"] = ms
    better = [m["diff"] < 0 for m in ms.values()]
    se_ok = all(-m["diff"] >= m["se"] for n, m in ms.items() if "holdout" in n and n.startswith("RMSE"))
    res["better_on"] = f"{sum(better)} of {len(better)}"
    res["kept"] = bool(all(better) and se_ok)
    res["verdict"] = "Kept (provisional)." if res["kept"] else ("Better on every measure but a holdout gain is under 1 SE: a trade-off for Ryan."
                                                                if all(better) else "Mixed or worse: not kept by the rule.")
    return res


# ------------------------------------------------------------------ 3b

def luck_tables(games, tg, qb):
    """Per quarterback-game and per defense-game interception luck (numerator) and dropbacks."""
    last = int(min(games.season.max(), 2026))
    pa = pff_data.load("passing_summary", range(2014, last + 1), league="nfl")
    pa = _num(pa, ["dropbacks", "interceptions", "turnover_worthy_plays"])
    pa["grades_pass"] = pd.to_numeric(pa["grades_pass"], errors="coerce")
    pa = pa[pa["dropbacks"] > 0].copy()
    pa["week"] = pa["week"].astype(int).map(lambda w: pff_qb.PLAYOFF_WEEK.get(w, w))
    pa = pa[pa["week"].between(1, 22)]
    pa["season"] = pa["season"].astype(int)
    pa["order"] = pa["season"] * 100 + pa["week"]
    # the production matcher (ids, then name within team): its rows carry the nflverse quarterback id
    m = pff_qb.qb_games(qb)[["nfl_id", "season", "order", "dropbacks", "grades_pass"]].drop_duplicates(["season", "order", "dropbacks", "grades_pass"], keep=False)
    q = pa.merge(m, on=["season", "order", "dropbacks", "grades_pass"], how="inner")
    q = q.groupby(["nfl_id", "season", "order", "franchise_id"], as_index=False)[["interceptions", "turnover_worthy_plays", "dropbacks"]].sum()
    # league interceptions per turnover-worthy play, earlier seasons only
    by = q.groupby("season")[["interceptions", "turnover_worthy_plays"]].sum().sort_index().cumsum()
    k = (by["interceptions"] / by["turnover_worthy_plays"]).shift().bfill()
    q["luck"] = q["interceptions"] - q["season"].map(k) * q["turnover_worthy_plays"]
    # PFF franchise -> board team, by the quarterbacks' board team that season
    qt = qb[["qb_id", "order", "team"]].rename(columns={"qb_id": "nfl_id"}).drop_duplicates(["nfl_id", "order"])
    fm = q.merge(qt, on=["nfl_id", "order"]).groupby(["season", "franchise_id", "team"])["dropbacks"].sum().reset_index() \
          .sort_values("dropbacks").drop_duplicates(["season", "franchise_id"], keep="last")[["season", "franchise_id", "team"]]
    dc = pff_data.load("defense_coverage", range(2014, last + 1), league="nfl")
    dc = _num(dc, ["interceptions", "dropped_ints"])
    dc["week"] = dc["week"].astype(int).map(lambda w: pff_qb.PLAYOFF_WEEK.get(w, w))
    dc = dc[dc["week"].between(1, 22)]
    dc["season"] = dc["season"].astype(int)
    dc["order"] = dc["season"] * 100 + dc["week"]
    d = dc.groupby(["season", "order", "franchise_id"], as_index=False)[["interceptions", "dropped_ints"]].sum().merge(fm, on=["season", "franchise_id"])
    opp = tg[["team", "order", "game_id"]].merge(tg[["game_id", "team", "n_pass"]].rename(columns={"team": "o", "n_pass": "opp_db"}), on="game_id")
    opp = opp[opp["team"] != opp["o"]][["team", "order", "opp_db"]]
    d = d.merge(opp, on=["team", "order"])
    d["chances"] = d["interceptions"] + d["dropped_ints"]
    byd = d.groupby("season")[["interceptions", "chances"]].sum().sort_index().cumsum()
    kd = (byd["interceptions"] / byd["chances"]).shift().bfill()
    d["luck"] = d["interceptions"] - d["season"].map(kd) * d["chances"]
    info = {"int_per_twp_by_season": {int(s): round(float(v), 3) for s, v in k.items()},
            "int_per_chance_by_season": {int(s): round(float(v), 3) for s, v in kd.items()},
            "qb_games": int(len(q)), "defense_games": int(len(d))}
    return q.rename(columns={"order": "t"}), d.rename(columns={"order": "t"}), info


def frame_with_luck():
    from nflmodel import data, extra, game_roles, injuries, model, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False)
    prior = pff_qb.load(qb, sorted(games.season.unique()))
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = PL.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = PL.player_rows(PL.team_games(f0, pg), pg, avail, pinfo)
    f = game_roles.attach(base, game_roles.team_features(rows, R, avail))
    q, d, info = luck_tables(games, tg, qb)
    g = games[["game_id", "season", "order", "home_team", "away_team", "home_qb_id", "away_qb_id"]].rename(columns={"order": "t"})
    for side in ("home", "away"):
        sq = PP.decayed(g.rename(columns={f"{side}_qb_id": "k"}), "k", q, "nfl_id", ["luck", "dropbacks"])
        sd = PP.decayed(g.rename(columns={f"{side}_team": "k"}), "k", d, "team", ["luck", "opp_db"])
        g[f"{side}_il_qb"] = sq[:, 0] / (sq[:, 1] + SHRINK)
        g[f"{side}_il_def"] = sd[:, 0] / (sd[:, 1] + SHRINK)
    for c in ("il_qb", "il_def"):
        g["d_" + c], g["s_" + c] = g[f"home_{c}"] - g[f"away_{c}"], g[f"home_{c}"] + g[f"away_{c}"]
    keep = ["game_id"] + [c for c in g.columns if "_il_" in c]
    return f.merge(g[keep], on="game_id", how="left").fillna({c: 0.0 for c in keep[1:]}), info


def run_3b() -> dict:
    from nflmodel import model
    f, info = frame_with_luck()
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)

    def out(mf, tf):
        o, _ = model.walk_forward(f, margin_feats=mf, total_feats=tf)
        o = o[o.played & o.model_margin.notna() & (o.season <= 2025)].copy()
        o["em"], o["et"] = (o.model_margin - o.result).abs(), (o.model_total - o.total).abs()
        return o.set_index("game_id")

    base = out(MF, TF)
    versions = {"A (QB side)": (MF + ["d_il_qb"], TF + ["s_il_qb"]),
                "B (QB side + defense)": (MF + ["d_il_qb", "d_il_def"], TF + ["s_il_qb", "s_il_def"])}
    res = {"inputs": info, "spread_of_terms_2019_25": {c: round(float(f[f.season.between(*HOLD)][c].std()), 5) for c in ("d_il_qb", "d_il_def")}}

    def cmp(o, per):
        k = base.season.between(*per)
        r = {}
        for col, lab in (("em", "margin"), ("et", "total")):
            a, b = base.loc[k, col].to_numpy(float), o.loc[base.index[k], col].to_numpy(float)
            dd = b - a
            r[lab] = {"n": int(k.sum()), "now": round(float(a.mean()), 4), "candidate": round(float(b.mean()), 4),
                      "diff": round(float(dd.mean()), 5), "se": round(float(dd.std(ddof=1) / np.sqrt(len(dd))), 5)}
        return r

    outs = {v: out(*fs) for v, fs in versions.items()}
    res["tuning"] = {v: cmp(o, TUNE) for v, o in outs.items()}
    ok = [v for v in versions if all(m["diff"] < 0 for m in res["tuning"][v].values())]
    res["better_on_both_tuning"] = ok
    if not ok:
        res["verdict"] = "No version better on both tuning measures: nothing carried to the holdout, nothing kept."
    else:
        pick = min(ok, key=lambda v: res["tuning"][v]["margin"]["candidate"])
        res["chosen"], res["holdout"] = pick, cmp(outs[pick], HOLD)
        ms = list(res["tuning"][pick].values()) + list(res["holdout"].values())
        better = [m["diff"] < 0 for m in ms]
        se_ok = all(-m["diff"] >= m["se"] for m in res["holdout"].values())
        res["better_on"] = f"{sum(better)} of {len(better)}"
        res["kept"] = bool(all(better) and se_ok)
        res["verdict"] = "Kept (provisional)." if res["kept"] else ("Better on every measure but a holdout gain is under 1 SE: a trade-off for Ryan."
                                                                    if all(better) else "Mixed: a trade-off for Ryan, not kept by the rule.")
    # reported: "fade the lucky team" against the closing spread, 2015-25
    b = f[f.played & f.season.between(2015, 2025) & f.spread_line.notna()].copy()
    tm = pd.concat([
        pd.DataFrame({"season": b.season, "net": b.home_il_def - b.home_il_qb, "cover": np.sign(b.result - b.spread_line)}),
        pd.DataFrame({"season": b.season, "net": b.away_il_def - b.away_il_qb, "cover": -np.sign(b.result - b.spread_line)})], ignore_index=True)
    tm = tm[tm.cover != 0]
    lo, hi = tm.net.quantile(0.2), tm.net.quantile(0.8)
    rep = {}
    for name, x in (("luckiest fifth", tm[tm.net >= hi]), ("unluckiest fifth", tm[tm.net <= lo]), ("everyone else", tm[(tm.net > lo) & (tm.net < hi)])):
        w = int((x.cover > 0).sum())
        rep[name] = {"record_ats": f"{w}-{len(x) - w}", "cover_rate": round(w / len(x), 4), "se": round(float(np.sqrt(0.25 / len(x))), 4)}
    res["reported_ats_2015_25"] = rep
    return res


def live_int_lines() -> dict:
    p = C.DATA / "props" / "prop_log.csv"
    if not p.exists():
        return {}
    x = pd.read_csv(p, usecols=lambda c: c in ("market", "side", "season", "game_id", "player_id", "line"))
    x = x[x["market"] == "player_pass_interceptions"]
    return {"rows": int(len(x)), "lines": int(x.drop_duplicates(["game_id", "player_id", "line"]).shape[0])}


def main():
    if not pff_data.available():
        sys.exit("PFF data not found (../pff-data or PFF_DATA_DIR)")
    res = {}
    which = sys.argv[1:] or ["3b", "3a"]
    if OUT.exists():
        res = json.loads(OUT.read_text())
    if "3b" in which:
        res["3b game model"] = run_3b()
        print("3b", json.dumps(res["3b game model"], default=str), flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=str))
    if "3a" in which:
        res["3a interceptions projection"] = run_3a()
        res["3a interceptions projection"]["live_log_interception_lines_2026"] = live_int_lines()
        print("3a", json.dumps(res["3a interceptions projection"], default=str), flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
