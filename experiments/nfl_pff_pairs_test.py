"""Pre-set test: the college PFF pairs (Nov 13, 2026) on the NFL player projections.

    python experiments/nfl_pff_pairs_test.py          (PFF data at ../pff-data or PFF_DATA_DIR)

Rule written before running (Project doc claude/plan-nfl-pff-pairs-test-2026-11-13.md):

Candidates, as of each game from earlier games only (8-game half-life, last season at 60%, nothing
older), league constants from 2014-2018, 0 with no history:
  pff_yprr        yards per route run (receiving_summary), 100 routes prior, rate / league - 1
  pff_pass_grade  the QB's PFF passing grade (passing_summary), dropback-weighted, 50 dropbacks prior,
                  (grade - league) / league spread
  pff_opp_cov     the opponent's PFF coverage grade (team_overview, per game), 2 games prior,
                  (grade - league) / league spread
Passing pair: pff_pass_grade + pff_opp_cov in players.CAL_EXTRA["pass_yds"].
Receiving pair: pff_yprr + pff_opp_cov in CAL_EXTRA["rec_yds"], pff_opp_cov in CAL_EXTRA["receptions"],
  and both as features of the boosted step (players_ml) for WR/RB receptions and receiving yards.
Measures (the Nov 12 template, db_name_match_test.py): the board's full pipeline, game-day and bet-time
  tables, paired by player-game. RMSE 2014-18 and 2019 on, each kind: passing yards (QBs who played,
  10+ projected attempts); receiving yards and receptions (WR/TE/RB who played, 1+ projected
  reception). Brier at the 2023-25 DK/FD main lines: closing lines with game-day projections, pricing
  snapshot with bet-time; passing yards / receiving yards + receptions pooled.
Rule: each pair kept only if better on every one of its measures, each holdout RMSE gain >= 1 SE.

Writes experiments/out/nfl_pff_pairs_test.json (scores only, never PFF rows).
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

from cfbmodel import pff_data  # noqa: E402
from nflmodel import config as C, pff_routes, players as PL, players_ml as ML, propbackfill as B, props as P  # noqa: E402

OUT = HERE / "out" / "nfl_pff_pairs_test.json"
CACHE = C.RAW / "nfl_pff_pairs_tables.pkl"
HL, CARRY = 8.0, 0.6
LEAGUE = (2014, 2015, 2016, 2017, 2018)
PLAYOFF = {28: 19, 29: 20, 30: 21, 32: 22}
PAIRS = {
    "passing": {"cal": {"pass_yds": ["pff_pass_grade", "pff_opp_cov"]}, "ml": [], "stats": ("pass_yds",)},
    "receiving": {"cal": {"rec_yds": ["pff_yprr", "pff_opp_cov"], "receptions": ["pff_opp_cov"]},
                  "ml": ["pff_yprr", "pff_opp_cov"], "stats": ("rec_yds", "receptions")},
}


# ------------------------------------------------------------------ inputs

def decayed(t, key, hist, hkey, cols):
    h = hist.sort_values([hkey, "t"])
    keyed = {k: (g["t"].to_numpy(), g["season"].to_numpy(), g[cols].to_numpy(float)) for k, g in h.groupby(hkey)}
    out = np.zeros((len(t), len(cols)))
    tt, kk, ss = t["t"].to_numpy(), t[key].to_numpy(), t["season"].to_numpy()
    for i in range(len(t)):
        k = keyed.get(kk[i])
        if k is None:
            continue
        ht, hs, v = k
        sel = (ht < tt[i]) & (hs >= ss[i] - 1)
        if not sel.any():
            continue
        j = np.arange(sel.sum())[::-1]
        w = 0.5 ** (j / HL) * np.where(hs[sel] == ss[i], 1.0, CARRY)
        out[i] = (w[:, None] * np.nan_to_num(v[sel])).sum(axis=0)
    return out


def _load(table, last):
    p = pff_data.load(table, range(2014, last + 1), league="nfl")
    p["week"] = p["week"].astype(int).map(lambda w: PLAYOFF.get(w, w))
    p = p[p["week"].between(1, 22)].copy()
    p["season"] = p["season"].astype(int)
    return p


def inputs(table: pd.DataFrame) -> pd.DataFrame:
    """pff_yprr, pff_pass_grade, pff_opp_cov for every (game_id, player_id) of the table."""
    last = int(table["season"].max())
    ids = pff_routes.pff_ids()
    keys = table[["game_id", "player_id", "team", "opp", "season", "week", "t", "played", "active"]].copy()
    played = keys[keys["played"].astype(bool) & (keys["active"] == 1)][["player_id", "season", "week", "t"]].drop_duplicates()
    out = keys[["game_id", "player_id"]].copy()
    rec = _load("receiving_summary", last).merge(ids, left_on="player_id", right_on="pff_id", how="inner")
    for c in ("yards", "routes"):
        rec[c] = pd.to_numeric(rec[c], errors="coerce").fillna(0.0)
    r = rec.groupby(["nfl_id", "season", "week"], as_index=False)[["yards", "routes"]].sum().rename(columns={"nfl_id": "player_id"})
    r = r.merge(played, on=["player_id", "season", "week"])
    lg = r[r.season.isin(LEAGUE)]
    lg_y = lg.yards.sum() / lg.routes.sum()
    s = decayed(keys, "player_id", r, "player_id", ["yards", "routes"])
    out["pff_yprr"] = np.where(s[:, 1] > 0, ((s[:, 0] + 100 * lg_y) / (s[:, 1] + 100)) / lg_y - 1, 0.0)
    pa = _load("passing_summary", last).merge(ids, left_on="player_id", right_on="pff_id", how="inner")
    for c in ("grades_pass", "dropbacks"):
        pa[c] = pd.to_numeric(pa[c], errors="coerce")
    pa = pa.dropna(subset=["grades_pass", "dropbacks"])
    pa["g_db"] = pa["grades_pass"] * pa["dropbacks"]
    q = pa[pa.season.isin(LEAGUE) & (pa.dropbacks >= 15)]
    lg_g, sd_g = float(np.average(q.grades_pass, weights=q.dropbacks)), float(q.grades_pass.std())
    p = pa.groupby(["nfl_id", "season", "week"], as_index=False)[["g_db", "dropbacks"]].sum().rename(columns={"nfl_id": "player_id"})
    p = p.merge(played, on=["player_id", "season", "week"])
    s = decayed(keys, "player_id", p, "player_id", ["g_db", "dropbacks"])
    out["pff_pass_grade"] = np.where(s[:, 1] > 0, (((s[:, 0] + 50 * lg_g) / (s[:, 1] + 50)) - lg_g) / sd_g, 0.0)
    # team_overview franchise -> board team: majority of the franchise's linked receivers' board team
    link = rec[["season", "week", "franchise_id", "nfl_id"]].merge(
        keys[["player_id", "season", "week", "team"]].rename(columns={"player_id": "nfl_id"}), on=["nfl_id", "season", "week"])
    fm = link.groupby(["season", "franchise_id", "team"]).size().rename("n").reset_index().sort_values("n").drop_duplicates(["season", "franchise_id"], keep="last")
    to = _load("team_overview", last)
    to["grades_coverage_defense"] = pd.to_numeric(to["grades_coverage_defense"], errors="coerce")
    to = to.dropna(subset=["grades_coverage_defense"]).merge(fm[["season", "franchise_id", "team"]], on=["season", "franchise_id"])
    tg = keys[["game_id", "team", "opp", "season", "week", "t", "played"]].drop_duplicates(["game_id", "team"])
    to = to.merge(tg[tg["played"].astype(bool)][["team", "season", "week", "t"]].drop_duplicates(["team", "season", "week"]), on=["team", "season", "week"])
    to["one"] = 1.0
    lg = to[to.season.isin(LEAGUE)]
    m, sd = float(lg.grades_coverage_defense.mean()), float(lg.grades_coverage_defense.std())
    so = decayed(tg.assign(team=tg["opp"]), "team", to, "team", ["grades_coverage_defense", "one"])
    tg = tg.assign(pff_opp_cov=np.where(so[:, 1] > 0, (((so[:, 0] + 2 * m) / (so[:, 1] + 2)) - m) / sd, 0.0))
    look = dict(zip(zip(tg["game_id"], tg["team"]), tg["pff_opp_cov"]))
    out["pff_opp_cov"] = [look.get(k, 0.0) for k in zip(keys["game_id"], keys["team"])]
    return out


# ------------------------------------------------------------------ tables and variants

def tables() -> dict:
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    from nflmodel import data, extra, game_roles, injuries, model, pff_qb, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False)
    prior = pff_qb.load(qb, sorted(games.season.unique()))
    print("quarterback prior:", "PFF grades" if prior is not None else "flat (NOT the production model)", flush=True)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = PL.load_availability(refresh_current=False, verbose=False)
    rush = injuries.load_pass_rush(games, refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = PL.player_rows(PL.team_games(f0, pg), pg, avail, pinfo)
    frame, _ = model.walk_forward(game_roles.attach(base, game_roles.team_features(rows, R, avail)))
    out = {"R": R}
    for when, pre in (("gameday", False), ("bettime", True)):
        t = PL.build(frame, pg, pinfo, avail, inj, rush, games=games, overrides=None, pregame=pre)
        if when == "gameday":
            feats = inputs(t)
        out[when] = t.merge(feats, on=["game_id", "player_id"], how="left")
        for c in ("pff_yprr", "pff_pass_grade", "pff_opp_cov"):
            out[when][c] = out[when][c].fillna(0.0)
    C.RAW.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def project(table: pd.DataFrame, R: dict, cal: dict, ml_extra: list[str]) -> pd.DataFrame:
    saved = {k: list(v) for k, v in PL.CAL_EXTRA.items()}
    for st, cols in cal.items():
        PL.CAL_EXTRA[st] = saved[st] + cols
    try:
        pall = PL.walk_forward(table, first_test=C.FIRST_TEST_SEASON)
    finally:
        PL.CAL_EXTRA.clear()
        PL.CAL_EXTRA.update(saved)
    x = ML.add_features(pall, R)
    feats = ML.feats_for(x) + ml_extra
    ml = ML.questionable(ML.walk_forward(x, first=C.FIRST_TEST_SEASON, feats=feats), first=C.FIRST_TEST_SEASON, snaps=R.get("snaps"))
    return ml.merge(PL.season_average(table), on=["game_id", "player_id"], how="left").reset_index(drop=True)


def rmse(base, cand, stats) -> dict:
    assert (base[["game_id", "player_id"]].to_numpy() == cand[["game_id", "player_id"]].to_numpy()).all()
    out = {}
    live = base["played"].astype(bool) & (base["active"] == 1) & (base["season"] >= 2014)
    rec = live & base["group"].isin(["WR", "TE", "RB"]) & (base["receptions_proj"] >= 1)
    qbs = live & (base["group"] == "QB") & (base["pass_att_proj"] >= 10)
    for st in stats:
        m = qbs if st == "pass_yds" else rec
        for per, pm in (("2014-18", base["season"] < 2019), ("2019 on", base["season"] >= 2019)):
            k = m & pm & base[st].notna()
            e0, e1 = (base.loc[k, st + "_proj"] - base.loc[k, st]) ** 2, (cand.loc[k, st + "_proj"] - cand.loc[k, st]) ** 2
            d = (e1 - e0).to_numpy(float)
            r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
            cl = pd.Series(d).groupby(base.loc[k, "game_id"].to_numpy()).sum()
            out[f"{st} {per}"] = {"n": int(k.sum()), "now": round(r0, 4), "candidate": round(r1, 4), "diff": round(r1 - r0, 5),
                                  "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(d) / (2 * r0)), 5)}
    return out


def brier(base, cand, kind, stats) -> dict:
    import props_real_lines as RL
    from props_calibration import FAMILY
    saved = P.PropModel.load()
    got = {"now": [], "cand": []}
    for season in (2023, 2024, 2025):
        snaps = B.load(season, kind)
        if snaps.empty:
            continue
        for k, t in (("now", base), ("cand", cand)):
            mdl = P.PropModel(FAMILY).fit(t[(t.season < season) & (t.season >= 2013)])
            if saved is not None:
                mdl.iso = saved.iso
            x = RL.priced(t, mdl, snaps, season)
            got[k].append(x[x["stat"].isin(stats)])
    key = ["game_id", "nname", "market", "line", "book"]
    b0 = pd.concat(got["now"], ignore_index=True)
    x = pd.concat(got["cand"], ignore_index=True).merge(
        b0[key + ["our_over", "our_under"]].rename(columns={"our_over": "b_over", "our_under": "b_under"}), on=key, how="inner")
    x = x[x["y"] != x["line"]]
    hit = (x["y"] > x["line"]).astype(float)
    ours, was, mkt = x["our_over"] / (x["our_over"] + x["our_under"]), x["b_over"] / (x["b_over"] + x["b_under"]), x["fair_over"]
    d = (ours - hit) ** 2 - (was - hit) ** 2
    cl = d.groupby(x["game_id"]).sum()
    return {"n": int(len(x)), "now": round(float(((was - hit) ** 2).mean()), 5), "candidate": round(float(((ours - hit) ** 2).mean()), 5),
            "books": round(float(((mkt - hit) ** 2).mean()), 5), "diff": round(float(d.mean()), 6),
            "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(x)), 6)}


def main():
    if not pff_data.available():
        sys.exit("PFF data not found (../pff-data or PFF_DATA_DIR)")
    T = tables()
    R = T["R"]
    g = T["gameday"]
    w = g[g.season.between(2019, 2025) & g.played.astype(bool) & (g.active == 1)]
    res = {"coverage_2019_25": {"pff_yprr (WR/TE/RB)": round(float((w[w.group.isin(["WR", "TE", "RB"])].pff_yprr != 0).mean()), 3),
                                "pff_pass_grade (QB)": round(float((w[w.group == "QB"].pff_pass_grade != 0).mean()), 3),
                                "pff_opp_cov": round(float((w.pff_opp_cov != 0).mean()), 3)}}
    print("coverage", res["coverage_2019_25"], flush=True)
    now = {k: project(T[k], R, {}, []) for k in ("gameday", "bettime")}
    for name, cfg in PAIRS.items():
        cand = {k: project(T[k], R, cfg["cal"], cfg["ml"]) for k in ("gameday", "bettime")}
        r = {"rmse": {k: rmse(now[k], cand[k], cfg["stats"]) for k in ("gameday", "bettime")},
             "brier": {"closing lines, game-day": brier(now["gameday"], cand["gameday"], "close", cfg["stats"]),
                       "pricing snapshot, bet-time": brier(now["bettime"], cand["bettime"], "flag", cfg["stats"])}}
        ok = True
        for k in ("gameday", "bettime"):
            for st in cfg["stats"]:
                a, b = r["rmse"][k][f"{st} 2014-18"], r["rmse"][k][f"{st} 2019 on"]
                ok &= a["diff"] < 0 and b["diff"] < 0 and -b["diff"] >= b["se"]
        ok &= all(v["diff"] < 0 for v in r["brier"].values())
        r["kept"] = bool(ok)
        res[name] = r
        print(name, json.dumps(r, default=str), flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
