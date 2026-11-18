"""Pre-set tests 1 and 2 from the "Hidden Stats" guide: unrealized air yards and air-yards share / WOPR (NFL).

    PFF_DATA_DIR=<pff-data clone> python experiments/air_yards_wopr_test.py

Rule written before running (Project doc claude/plan-air-yards-wopr-twp-tests-2026-11-18.md). The board's
receiving projections run on target share (plus PFF route share in the boosted step); air yards are
collected per player-game and used nowhere. aDOT was in the Nov 6 PFF receiving chain (not kept).

Inputs, as of each game from earlier games he played (players.asof: 6-game half-life, last season at
60%); league constants from 2014-18 only; zero with no history.
Test 1, receiving yards, WR and TE (zero for RBs):
  A air_unreal  share of his air yards that went uncaught (air yards on targets not caught / all his
                air yards), shrunk to his position's league rate by 300 air yards, rate / league - 1
  B air_gap3    (air yards - receiving yards) / air yards over his last 3 games played this season;
                zero with under 3 games or under 100 air yards
  Entered in players.CAL_EXTRA["rec_yds"] (raw projection x column) and as a feature of the boosted
  step; only the receiving-yards projection is taken from the candidate run.
  Measures (6): receiving-yards RMSE, WR/TE who played with 1+ projected reception, game-day and
  bet-time, tuning 2014-18 and holdout 2019-25; Brier at the 2023-25 DK/FD receiving-yards main lines,
  closing lines with game-day projections and the pricing snapshot with bet-time.
Test 2, receptions and receiving yards, WR/TE/RB:
  A ay_minus    air-yards share - target share (air-yards share built like tgt_share: his share of the
                team's air yards in games he was active, same decay and shrink)
  B wopr        1.5 x target share + 0.7 x air-yards share, value / position league average - 1
  Entered in CAL_EXTRA["rec_yds"] and ["receptions"] and as a feature of the boosted step; the
  receptions and receiving-yards projections are taken from the candidate run.
  Measures (11): RMSE for receiving yards and receptions, WR/TE/RB who played with 1+ projected
  reception, game-day and bet-time, tuning and holdout (8); Brier at the main lines, receiving yards
  2023-25 and receptions 2025 pooled, closing and pricing (2); Brier on the floor rungs, the 2025
  alternate receiving-yards rungs the books read at 65-95%, pricing snapshot (1).
Rule (each test on its own): the version is chosen on tuning alone (better than the current model on
  every tuning RMSE measure, then the lowest game-day tuning receiving-yards RMSE). Kept only if better
  on every measure and each holdout RMSE gain >= 1 paired SE. Mixed, or under 1 SE: a trade-off for
  Ryan. No version better on every tuning measure: nothing is carried to the holdout.
Reported, not used: test 1, the top fifth of air_unreal (average miss before and after, receiving-TD
  rate against the board's projection); test 2, at the same projection, whether the high-WOPR third
  clears 80% and 60% of his projection (stand-ins for his Typical number and a floor rung) more often
  than the low third.

Writes experiments/out/air_yards_wopr_test.json.
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

from nflmodel import config as C, players as PL, players_ml as ML, propbackfill as B, props as P  # noqa: E402

OUT = HERE / "out" / "air_yards_wopr_test.json"
CACHE = C.RAW / "air_yards_wopr_tables.pkl"
LEAGUE = (2014, 2018)
TUNE, HOLD = (2014, 2018), (2019, 2025)
REC = ("WR", "TE", "RB")
TESTS = {
    "1 unrealized air yards": {"versions": {"A": "air_unreal", "B": "air_gap3"}, "cal": ("rec_yds",), "take": ("rec_yds",),
                               "groups": ("WR", "TE"), "rungs": False},
    "2 air-yards share / WOPR": {"versions": {"A": "ay_minus", "B": "wopr"}, "cal": ("rec_yds", "receptions"),
                                 "take": ("rec_yds", "receptions"), "groups": REC, "rungs": True},
}


# ------------------------------------------------------------------ tables

def tables() -> dict:
    """The board's player tables before projection, game-day and bet-time, and the role data (cached)."""
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
        out[when] = PL.build(frame, pg, pinfo, avail, inj, rush, games=games, overrides=None, pregame=pre)
        print(when, len(out[when]), flush=True)
    C.RAW.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def uncaught_air(seasons) -> pd.DataFrame:
    """Air yards on targets not caught, per receiver per game, from play-by-play (the box-score cache
    keeps only the total). Same plays as players_data: pass attempts, no sacks, spikes or two-point tries."""
    from nflmodel.data import _download
    got = []
    for s in seasons:
        f = C.RAW / f"air_uncaught_{s}.csv"
        if not f.exists():
            raw = C.RAW / f"pbp_air_{s}.parquet"
            _download(C.PBP_URL.format(season=s), raw)
            p = pd.read_parquet(raw, columns=["game_id", "posteam", "pass_attempt", "sack", "qb_spike", "two_point_attempt",
                                              "complete_pass", "receiver_player_id", "air_yards"])
            raw.unlink()
            p = p[(p["two_point_attempt"].fillna(0) == 0) & p["posteam"].notna() & (p["pass_attempt"] == 1)
                  & (p["sack"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0) & p["receiver_player_id"].notna()]
            p = p[p["complete_pass"].fillna(0) == 0]
            p.groupby(["game_id", "receiver_player_id"])["air_yards"].sum().rename("air_unc").reset_index() \
                .rename(columns={"receiver_player_id": "player_id"}).to_csv(f, index=False)
            print("  uncaught air yards", s, flush=True)
        got.append(pd.read_csv(f, dtype={"player_id": str}))
    return pd.concat(got, ignore_index=True)


# ------------------------------------------------------------------ inputs

def inputs(t: pd.DataFrame) -> pd.DataFrame:
    """air_unreal, air_gap3, ay_minus, wopr for every (game_id, player_id) of the game-day table."""
    a = t[["game_id", "player_id", "team", "season", "week", "t", "played", "active", "group", "air_yards", "rec_yds",
           "tgt_share_raw"]].copy()
    a["player_id"] = a["player_id"].astype(str)
    a = a.merge(uncaught_air(sorted(int(s) for s in a["season"].unique())), on=["game_id", "player_id"], how="left")
    use = (a["active"] == 1) & a["played"].astype(bool)
    a["air_unc"] = a["air_unc"].fillna(0.0)
    team_air = a[use].groupby(["game_id", "team"])["air_yards"].sum().rename("team_air").reset_index()
    a = a.merge(team_air, on=["game_id", "team"], how="left")
    a["ay_sh"] = np.where(use & (a["team_air"] > 0), a["air_yards"] / a["team_air"], np.nan)
    a["one"] = np.where(use, 1.0, np.nan)
    for c in ("air_yards", "air_unc", "rec_yds"):
        a[c] = a[c].where(use)
    f = PL.asof(a, "player_id", ["air_yards", "air_unc", "ay_sh"], ["one"], PL.PLAYER_HL)
    out = f[["game_id", "player_id", "group", "season", "tgt_share_raw"]].copy()
    pos = a[use].groupby("group")["ay_sh"].mean()
    out["ay_share"] = (f["ay_sh_sum"].to_numpy(float) + out["group"].map(pos).fillna(0.0).to_numpy(float) * 0.5) / (f["one_sum"].to_numpy(float) + 1.0)
    lg = a[use & a["season"].between(*LEAGUE)]
    # 1A: uncaught share of his air yards, 300 air yards of his position's league rate
    rate = (lg.groupby("group")["air_unc"].sum() / lg.groupby("group")["air_yards"].sum())
    r = out["group"].map(rate).to_numpy(float)
    air, unc = f["air_yards_sum"].to_numpy(float), f["air_unc_sum"].to_numpy(float)
    ok = out["group"].isin(["WR", "TE"]).to_numpy() & (air > 0)
    out["air_unreal"] = np.where(ok, ((unc + 300 * r) / (air + 300)) / r - 1, 0.0)
    # 1B: the article's gap over his last 3 games played this season
    # a game he was out of, or an upcoming one, reads the last three he played this season
    a2 = a.sort_values(["player_id", "t"]).copy()
    a2["air_p"], a2["rec_p"] = a2["air_yards"], a2["rec_yds"]
    gp = a2.groupby(["player_id", "season"])
    a2["n_prev"] = gp["one"].transform(lambda s: s.shift().fillna(0).cumsum())

    def last3(s):
        v = s.shift()
        return v.dropna().rolling(3, min_periods=3).sum().reindex(v.index).ffill()
    a2["air3"] = gp["air_p"].transform(last3)
    a2["rec3"] = gp["rec_p"].transform(last3)
    gap = np.where((a2["n_prev"] >= 3) & (a2["air3"] >= 100) & a2["group"].isin(["WR", "TE"]), (a2["air3"] - a2["rec3"]) / a2["air3"], 0.0)
    out = out.merge(a2.assign(air_gap3=gap)[["game_id", "player_id", "air_gap3"]], on=["game_id", "player_id"], how="left")
    # 2: air-yards share, shrunk as tgt_share is (half his position's average, one game)
    isrec = out["group"].isin(REC).to_numpy()
    out["ay_share"] = np.where(isrec, out["ay_share"], 0.0)
    out["ay_minus"] = np.where(isrec, out["ay_share"] - out["tgt_share_raw"].fillna(0.0), 0.0)
    w = 1.5 * out["tgt_share_raw"].fillna(0.0) + 0.7 * out["ay_share"]
    out["wopr_raw"] = np.where(isrec, w, 0.0)
    k = out.merge(a[["game_id", "player_id", "one"]], on=["game_id", "player_id"], how="left")
    lgm = out[(k["one"].to_numpy() == 1) & out["season"].between(*LEAGUE).to_numpy() & isrec].groupby("group")["wopr_raw"].mean()
    out["wopr"] = np.where(isrec, out["wopr_raw"] / out["group"].map(lgm).to_numpy(float) - 1, 0.0)
    cols = ["air_unreal", "air_gap3", "ay_minus", "wopr", "wopr_raw", "ay_share"]
    out[cols] = out[cols].fillna(0.0)
    return out[["game_id", "player_id"] + cols]


# ------------------------------------------------------------------ projections and measures

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


def candidate(base: pd.DataFrame, run: pd.DataFrame, take) -> pd.DataFrame:
    """The current projections with only the tested stats taken from the candidate run."""
    assert (base[["game_id", "player_id"]].to_numpy() == run[["game_id", "player_id"]].to_numpy()).all()
    c = base.copy()
    for st in take:
        c[st + "_proj"] = run[st + "_proj"].to_numpy()
    return c


def rmse(base, cand, st, groups, per) -> dict:
    k = (base["played"].astype(bool) & (base["active"] == 1) & base["group"].isin(groups) & (base["receptions_proj"] >= 1)
         & base["season"].between(*per) & base[st].notna()).to_numpy()
    y = base[st].to_numpy(float)[k]
    e0, e1 = (base[st + "_proj"].to_numpy(float)[k] - y) ** 2, (cand[st + "_proj"].to_numpy(float)[k] - y) ** 2
    r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
    cl = pd.Series(e1 - e0).groupby(base["game_id"].to_numpy()[k]).sum()
    return {"n": int(k.sum()), "now": round(r0, 4), "candidate": round(r1, 4), "diff": round(r1 - r0, 5),
            "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / k.sum() / (2 * r0)), 5)}


def _models(base, cand, season):
    from props_calibration import FAMILY
    saved = P.PropModel.load()
    out = []
    for t in (base, cand):
        m = P.PropModel(FAMILY).fit(t[(t.season < season) & (t.season >= 2013)])
        if saved is not None:
            m.iso = saved.iso
        out.append(m)
    return out


def brier(base, cand, kind, stats, groups=None) -> dict:
    import props_real_lines as RL
    got = {"now": [], "cand": []}
    for season in (2023, 2024, 2025):
        snaps = B.load(season, kind)
        if snaps.empty:
            continue
        for k, t, m in zip(("now", "cand"), (base, cand), _models(base, cand, season)):
            x = RL.priced(t, m, snaps, season)
            x = x[x["stat"].isin(stats)]
            got[k].append(x if groups is None else x[x["group"].isin(groups)])
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


def floor_rungs(base, cand, groups=None) -> dict:
    """Brier on the alternate receiving-yards rungs the books read at 65-95% (pricing snapshot, bet-time)."""
    from nfl_pff_prop_inputs_test import rungs
    key = ["game_id", "player_id", "stat", "line", "book"]
    got = {"now": [], "cand": []}
    for season in (2023, 2024, 2025):
        snaps = B.load(season, "flag")
        if snaps.empty:
            continue
        for k, t, m in zip(("now", "cand"), (base, cand), _models(base, cand, season)):
            r = rungs(t, m, snaps, season, "player_reception_yds_alternate")
            if not r.empty:
                got[k].append(r if groups is None else r[r["group"].isin(groups)])
    a = pd.concat(got["now"], ignore_index=True)
    b = pd.concat(got["cand"], ignore_index=True)
    a, b = a[a["y"] != a["line"]].drop_duplicates(key), b[b["y"] != b["line"]].drop_duplicates(key)
    j = a[key + ["y", "our_over", "books"]].merge(b[key + ["our_over"]], on=key, suffixes=("_c", "_n"))
    hit = (j["y"] > j["line"]).astype(float)
    d = (j["our_over_n"] - hit) ** 2 - (j["our_over_c"] - hit) ** 2
    cl = d.groupby(j["game_id"]).sum()
    return {"n": int(len(j)), "now": round(float(((j["our_over_c"] - hit) ** 2).mean()), 5),
            "candidate": round(float(((j["our_over_n"] - hit) ** 2).mean()), 5), "books": round(float(((j["books"] - hit) ** 2).mean()), 5),
            "diff": round(float(d.mean()), 6), "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(j)), 6),
            "cleared": round(float(hit.mean()), 4)}


# ------------------------------------------------------------------ reported, not used

def report_air(base, cand, col) -> dict:
    """The top fifth of air_unreal among WR/TE with 1+ projected reception, 2019-25."""
    k = base["played"].astype(bool) & (base["active"] == 1) & base["group"].isin(["WR", "TE"]) & (base["receptions_proj"] >= 1) \
        & base["season"].between(*HOLD) & (base[col] != 0)
    x, c = base[k], cand[k]
    top = x[col] >= x[col].quantile(0.8)
    out = {}
    for name, m in (("top fifth", top), ("the rest", ~top)):
        y = x.loc[m, "rec_yds"]
        td = (x.loc[m, "rec_td"] > 0).astype(float)
        ptd = 1 - np.exp(-x.loc[m, "rec_td_proj"].clip(lower=0))
        out[name] = {"n": int(m.sum()), "avg_actual_minus_proj_now": round(float((y - x.loc[m, "rec_yds_proj"]).mean()), 2),
                     "se": round(float((y - x.loc[m, "rec_yds_proj"]).std(ddof=1) / np.sqrt(m.sum())), 2),
                     "after": round(float((y - c.loc[m, "rec_yds_proj"]).mean()), 2),
                     "rec_td_rate": round(float(td.mean()), 4), "board_rec_td_chance": round(float(ptd.mean()), 4)}
    return out


def report_wopr(base) -> dict:
    """At the same projection: does the high-WOPR third clear his typical game and a floor more often?"""
    k = base["played"].astype(bool) & (base["active"] == 1) & base["group"].isin(["WR", "TE"]) & (base["rec_yds_proj"] >= 25) \
        & base["season"].between(*HOLD)
    x = base[k].copy()
    x["band"] = pd.qcut(x["rec_yds_proj"], 10, labels=False, duplicates="drop")
    x["third"] = x.groupby("band")["wopr_raw"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    x["typical"] = (x["rec_yds"] >= 0.8 * x["rec_yds_proj"]).astype(float)   # stand-in for his Typical number
    x["floor"] = (x["rec_yds"] >= 0.6 * x["rec_yds_proj"]).astype(float)
    x["ratio"] = x["rec_yds"] / x["rec_yds_proj"]
    out = {}
    for th, g in x.groupby("third"):
        out[str(th)] = {"n": int(len(g)), "avg_proj": round(float(g["rec_yds_proj"].mean()), 1), "avg_actual": round(float(g["rec_yds"].mean()), 1),
                        "cleared_80pct": round(float(g["typical"].mean()), 4), "cleared_60pct": round(float(g["floor"].mean()), 4),
                        "sd_of_actual_over_proj": round(float(g["ratio"].std()), 3)}
    hi, lo = x[x["third"] == "high"], x[x["third"] == "low"]
    for c in ("typical", "floor"):
        se = float(np.sqrt(hi[c].var() / len(hi) + lo[c].var() / len(lo)))
        out[f"high minus low, {c}"] = {"diff": round(float(hi[c].mean() - lo[c].mean()), 4), "se": round(se, 4)}
    return out


# ------------------------------------------------------------------ run

def main():
    T = tables()
    R = T["R"]
    feats = inputs(T["gameday"])
    for k in ("gameday", "bettime"):
        T[k] = T[k].drop(columns=[c for c in feats.columns if c in T[k].columns and c not in ("game_id", "player_id")]) \
                   .merge(feats, on=["game_id", "player_id"], how="left")
        T[k][list(feats.columns[2:])] = T[k][list(feats.columns[2:])].fillna(0.0)
    g = T["gameday"]
    w = g[g.season.between(2019, 2025) & g.played.astype(bool) & (g.active == 1)]
    res = {"coverage_2019_25": {c: round(float((w[w.group.isin(["WR", "TE"]) if c.startswith("air") else w.group.isin(REC)][c] != 0).mean()), 3)
                                for c in ("air_unreal", "air_gap3", "ay_minus", "wopr")},
           "spread_2019_25": {c: round(float(w[w.group.isin(["WR", "TE"])][c].std()), 4) for c in ("air_unreal", "air_gap3", "ay_minus", "wopr")}}
    print(json.dumps(res), flush=True)
    now = {k: project(T[k], R, {}, []) for k in ("gameday", "bettime")}
    print("current projections done", flush=True)
    for name, cfg in TESTS.items():
        r, cands = {"tuning": {}}, {}
        for v, col in cfg["versions"].items():
            cands[v] = {k: candidate(now[k], project(T[k], R, {st: [col] for st in cfg["cal"]}, [col]), cfg["take"]) for k in ("gameday", "bettime")}
            r["tuning"][v] = {f"{st} {k}": rmse(now[k], cands[v][k], st, cfg["groups"], TUNE) for st in cfg["take"] for k in ("gameday", "bettime")}
            print(name, v, col, json.dumps(r["tuning"][v]), flush=True)
        ok = [v for v in cfg["versions"] if all(m["diff"] < 0 for m in r["tuning"][v].values())]
        r["better_on_every_tuning_measure"] = ok
        if not ok:
            r["verdict"] = "No version better on every tuning measure: nothing carried to the holdout, nothing kept."
        else:
            pick = min(ok, key=lambda v: r["tuning"][v]["rec_yds gameday"]["candidate"])
            c = cands[pick]
            r["chosen"] = f"{pick} ({cfg['versions'][pick]})"
            r["holdout"] = {f"{st} {k}": rmse(now[k], c[k], st, cfg["groups"], HOLD) for st in cfg["take"] for k in ("gameday", "bettime")}
            r["brier"] = {"closing lines, game-day": brier(now["gameday"], c["gameday"], "close", cfg["take"]),
                          "pricing snapshot, bet-time": brier(now["bettime"], c["bettime"], "flag", cfg["take"])}
            if cfg["rungs"]:
                r["brier"]["floor rungs 65-95%, pricing snapshot"] = floor_rungs(now["bettime"], c["bettime"])
            ms = list(r["tuning"][pick].values()) + list(r["holdout"].values()) + list(r["brier"].values())
            better = [m["diff"] < 0 for m in ms]
            se_ok = all(-m["diff"] >= m["se"] for m in r["holdout"].values())
            r["better_on"] = f"{sum(better)} of {len(better)}"
            r["kept"] = bool(all(better) and se_ok)
            r["verdict"] = "Kept (provisional)." if r["kept"] else ("Better on every measure but a holdout gain is under 1 SE: a trade-off for Ryan."
                                                                     if all(better) else "Mixed: a trade-off for Ryan, not kept by the rule.")
        if name.startswith("1"):
            v = "A"
            r["reported_top_fifth_air_unreal_2019_25"] = report_air(now["gameday"], cands[v]["gameday"], "air_unreal")
        else:
            r["reported_wopr_thirds_2019_25"] = report_wopr(now["gameday"])
        res[name] = r
        print(name, json.dumps(r, default=str), flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
