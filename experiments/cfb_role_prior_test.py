"""Early-season role prior for college running backs, from CFBD background data.

    python experiments/cfb_role_prior_test.py

Nov 3, 2026. The college prop diagnostic (cfb_props_gap_diag.py) put 78% of the rushing-yards
gap to the books in weeks 3-8, in props where the model's role for a back sits 30%+ from the
books'. A back's share of his team's carries starts each season from a flat prior: half the
position's average share, worth one game, against his decayed history (last season at 60%).
What a book knows in September that the chain does not: his recruiting rating, class year,
whether he transferred in, how much of last year's rushing left the roster, and where his history
ranks him among the team's backs.

Candidate: a player-specific prior share for running backs, a ridge regression of a back's carry
share over his team's first eight games on
    hist_share   his decayed carry share entering the season (car_share_sum / one_sum, 0 if none)
    hist_n       how much history that is (one_sum, capped at 8)
    stars, rating   247 composite from CFBD recruiting (through the roster's recruitIds)
    class_year   roster year 1-4
    transfer     in the CFBD portal for this season with this team as destination
    team_ret     the team's returning rushing usage (CFBD /player/returning)
    rank         rank of hist_share among the team's backs with rows this season
    same-group averages where a feature is missing
fitted on the seasons before each test season (2015 on). The prior replaces the flat one with a
weight of w pseudo-games: car_share_raw = (car_share_sum + prior x w) / (one_sum + w), running
backs only; w chosen on the tuning seasons from {1, 2, 4, 8, 16}. Everything downstream (share
normalisation, team volume, calibration) is unchanged.

Rules, written before running:
1. Score A, backtest: RMSE of RB rushing yards and of RB carries (chain or projection >= 20 yards
   / >= 4 carries) in weeks 1-8, tuning 2017-2019 and holdout 2021 on, paired difference from the
   current projections, standard errors clustered by game. w is chosen on tuning rushing yards.
2. Score B, lines: Brier at the 2025 DraftKings/FanDuel rushing-yards main lines in weeks 3-8
   (cfb_rush_group_calib_test.py's frame), new vs current, clustered by game. All weeks reported.
3. Keep only if rushing-yards RMSE and carries RMSE both improve on the holdout by 2+ SE and are
   not worse on the tuning seasons, and the weeks 3-8 Brier at the 2025 lines improves by 2+ SE.
   Otherwise nothing changes. Receiving (WR/TE/RB catch share, same features) is run the same way
   and reported as information, not for the decision.

Writes experiments/out/cfb_role_prior_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from cfbmodel import config as C, players as PL  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402
from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "cfb_role_prior_test.json"
CFBD = C.DATA / "cfbd"
EARLY = 8
W_GRID = (1, 2, 4, 8, 16)
FIT_FROM = 2015
FEATS = ["hist_share", "hist_n", "stars", "rating", "class_year", "transfer", "team_ret", "rank", "no_recruit"]


def _name(first, last):
    return (pd.Series(first).fillna("").astype(str) + " " + pd.Series(last).fillna("").astype(str)).map(PM.norm_name)


def background(seasons) -> pd.DataFrame:
    """One row per (season, player_id): stars, rating, class_year, transfer flag, CFBD team name."""
    rows = []
    for s in seasons:
        ro = pd.read_csv(CFBD / f"roster_{s}.csv", low_memory=False)
        ro["player_id"] = pd.to_numeric(ro["id"], errors="coerce")
        ro = ro[ro["player_id"].notna()].copy()
        ro["player_id"] = ro["player_id"].astype(np.int64)
        ro["recruit_id"] = ro["recruitIds"].map(lambda v: (json.loads(v) or [None])[0] if isinstance(v, str) else None)
        ro["recruit_id"] = pd.to_numeric(ro["recruit_id"], errors="coerce")
        rec = pd.concat([pd.read_csv(CFBD / f"recruits_{y}.csv", low_memory=False) for y in range(max(2010, s - 6), s + 1)
                         if (CFBD / f"recruits_{y}.csv").exists()], ignore_index=True)
        rec = rec.drop_duplicates("id").set_index("id")
        ro["stars"] = ro["recruit_id"].map(rec["stars"])
        ro["rating"] = ro["recruit_id"].map(rec["rating"])
        # recruits with the athlete id, where the roster has no recruit id
        by_ath = rec[rec["athleteId"].notna()].assign(athleteId=lambda d: pd.to_numeric(d["athleteId"], errors="coerce")).dropna(subset=["athleteId"]).drop_duplicates("athleteId").set_index("athleteId")
        miss = ro["stars"].isna()
        ro.loc[miss, "stars"] = ro.loc[miss, "player_id"].map(by_ath["stars"])
        ro.loc[miss, "rating"] = ro.loc[miss, "player_id"].map(by_ath["rating"])
        ro["class_year"] = pd.to_numeric(ro["year"], errors="coerce").where(lambda v: v.between(1, 4))
        ro["transfer"] = 0.0
        pf = CFBD / f"portal_{s}.csv"
        if pf.exists():
            po = pd.read_csv(pf, low_memory=False)
            po["key"] = _name(po["firstName"], po["lastName"]) + "|" + po["destination"].fillna("").astype(str)
            ro["key"] = _name(ro["firstName"], ro["lastName"]) + "|" + ro["team"].astype(str)
            ro["transfer"] = ro["key"].isin(set(po["key"])).astype(float)
        ro["season"] = s
        rows.append(ro[["season", "player_id", "team", "stars", "rating", "class_year", "transfer"]].rename(columns={"team": "cfbd_team"}).drop_duplicates(["season", "player_id"]))
    return pd.concat(rows, ignore_index=True)


def returning(seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        f = CFBD / f"returning_{s}.csv"
        if f.exists():
            r = pd.read_csv(f)
            out.append(pd.DataFrame({"season": s, "cfbd_team": r["team"], "team_ret": r["rushingUsage"], "team_ret_rec": r["receivingUsage"]}))
    return pd.concat(out, ignore_index=True)


def features(t: pd.DataFrame, bg: pd.DataFrame, ret: pd.DataFrame, share: str, groups: tuple) -> pd.DataFrame:
    """One row per (season, team, player) with a row in the table, group in `groups`: features
    known before the season and the target (carry/catch share over the team's first EARLY weeks)."""
    x = t[t["group"].isin(groups) & t["proj_total"].notna()].sort_values("t")
    first = x.groupby(["season", "team", "player_id"]).head(1)
    f = first[["season", "team", "player_id", "group", f"{share}_sum", "one_sum"]].copy()
    f["hist_share"] = (f[f"{share}_sum"] / f["one_sum"].replace(0, np.nan)).fillna(0.0)
    f["hist_n"] = f["one_sum"].clip(upper=8)
    f = f.merge(bg, on=["season", "player_id"], how="left")
    # CFBD team name for each board team: the roster team most of its players carry
    tm = f.dropna(subset=["cfbd_team"]).groupby(["season", "team"])["cfbd_team"].agg(lambda v: v.value_counts().index[0]).rename("team_cfbd").reset_index()
    f = f.drop(columns="cfbd_team").merge(tm, on=["season", "team"], how="left").merge(ret.rename(columns={"cfbd_team": "team_cfbd"}), on=["season", "team_cfbd"], how="left")
    if share == "rec_share":
        f["team_ret"] = f["team_ret_rec"]
    f["rank"] = f.groupby(["season", "team"])["hist_share"].rank(ascending=False, method="min").clip(upper=5)
    f["no_recruit"] = f["stars"].isna().astype(float)
    f["transfer"] = f["transfer"].fillna(0.0)
    # target: share over the first EARLY weeks in which he appeared, weighted by team volume
    vol = "team_car" if share == "car_share" else "team_comp"
    early = x[(x["week"] <= EARLY) & (x["appeared"] == 1) & x["played"].astype(bool)]
    num = "carries" if share == "car_share" else "receptions"
    tgt = early.groupby(["season", "team", "player_id"]).agg(n=(num, "sum"), v=(vol, "sum")).reset_index()
    tgt["target"] = tgt["n"] / tgt["v"].replace(0, np.nan)
    f = f.merge(tgt[["season", "team", "player_id", "target"]], on=["season", "team", "player_id"], how="left")
    f["target"] = f["target"].fillna(0.0)      # a row this season, never appeared early: no share
    return f


def fit_priors(f: pd.DataFrame, first_test: int) -> pd.DataFrame:
    """Walk-forward ridge: for each season from first_test, fitted on FIT_FROM..s-1."""
    out = []
    for s in sorted(f["season"].unique()):
        if s < first_test:
            continue
        tr = f[(f["season"] < s) & (f["season"] >= FIT_FROM)]
        cur = f[f["season"] == s].copy()
        if len(tr) < 500:
            continue
        fill = tr[FEATS].mean()
        Xtr, Xc = tr[FEATS].fillna(fill), cur[FEATS].fillna(fill)
        m = PL.Lin(5.0).fit(Xtr.to_numpy(float), tr["target"].to_numpy(float))
        cur["prior"] = np.clip(m.predict(Xc.to_numpy(float)), 0.0, 0.9)
        cur["coef"] = [dict(zip(FEATS, np.round(m.b / m.sd, 4)))] * len(cur)
        out.append(cur)
    return pd.concat(out, ignore_index=True)


def apply_prior(t: pd.DataFrame, pri: pd.DataFrame, share: str, groups: tuple, w: float) -> pd.DataFrame:
    t2 = t.merge(pri[["season", "team", "player_id", "prior"]], on=["season", "team", "player_id"], how="left")
    sel = t2["prior"].notna() & t2["group"].isin(groups)
    new = (t2[f"{share}_sum"] + t2["prior"] * w) / (t2["one_sum"] + w)
    t2[f"{share}_raw"] = np.where(sel, new, t2[f"{share}_raw"])
    return t2.drop(columns="prior")


def score_early(cur: pd.DataFrame, new: pd.DataFrame, stat: str, groups: tuple, floor: float) -> dict:
    key = ["game_id", "player_id"]
    m = cur[key + ["season", "week", "group", "appeared", "played", stat, stat + "_p", stat + "_proj"]].merge(
        new[key + [stat + "_proj"]].rename(columns={stat + "_proj": "new"}), on=key)
    d = m[m.group.isin(groups) & (m.appeared == 1) & m.played.astype(bool) & m[stat].notna() & (m.week <= EARLY)]
    d = d[(d[stat + "_p"] >= floor) | (d[stat + "_proj"] >= floor)]
    res = {}
    for per, x in (("tuning 2017-19", d[d.season <= 2019]), ("holdout 2021+", d[d.season >= 2021])):
        e_old = (x[stat + "_proj"] - x[stat]).to_numpy() ** 2
        e_new = (x["new"] - x[stat]).to_numpy() ** 2
        diff, se = G.paired(e_new, e_old, x.game_id.to_numpy())
        res[per] = {"n": int(len(x)), "rmse_current": float(np.sqrt(e_old.mean())), "rmse_new": float(np.sqrt(e_new.mean())),
                    "mse_diff": diff, "se": se, "z": diff / se if se else 0.0,
                    "bias_current": float((x[stat + "_proj"] - x[stat]).mean()), "bias_new": float((x["new"] - x[stat]).mean())}
    return res


def score_lines(cur, new, pinfo, games, stat: str, vol: str, weeks=None) -> dict:
    key = ["game_id", "player_id"]
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    rows = R.player_frame(cur, pinfo).merge(new[key + [stat + "_proj", vol + "_proj"]].rename(columns={stat + "_proj": "new_proj", vol + "_proj": "new_vol"}), on=key, how="left")
    main_ = s[~s["alt"] & (s["stat"] == stat)]
    w = main_.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    base = w.merge(rows, on=["game_id", "nname"], how="inner")
    if weeks is not None:
        base = base[base["week"].between(*weeks)]
    model = P.PropModel.load()
    outs = {}
    for name, cols in (("current", None), ("new", ("new_proj", "new_vol"))):
        dd = base.copy()
        if cols:
            dd[stat + "_proj"], dd[vol + "_proj"] = dd[cols[0]], dd[cols[1]]
        x = R.model_chances(dd, model)
        x = x[x["y"] != x["line"]].set_index(["game_id", "nname", "line"], drop=False).sort_index()
        outs[name] = x
    idx = outs["current"].index.intersection(outs["new"].index)
    a, b = outs["current"].loc[idx], outs["new"].loc[idx]
    hit = (a["y"] > a["line"]).astype(float).to_numpy()
    pa = (a["our_over"] / (a["our_over"] + a["our_under"])).to_numpy()
    pb = (b["our_over"] / (b["our_over"] + b["our_under"])).to_numpy()
    diff, se = R.clustered_diff(pb, pa, hit, a.index.get_level_values(0).to_numpy())
    dbk, sebk = R.clustered_diff(pb, a["fair"].to_numpy(), hit, a.index.get_level_values(0).to_numpy())
    return {"props": int(len(idx)), "brier_books": D.brier(a["fair"].to_numpy(), hit), "brier_current": D.brier(pa, hit),
            "brier_new": D.brier(pb, hit), "new_minus_current": diff, "se": se, "z": diff / se if se else 0.0,
            "new_minus_books": dbk, "books_se": sebk}


def main():
    t, pinfo, games = G.table()
    seasons = sorted(int(s) for s in t["season"].unique())
    bg = background([s for s in seasons if (CFBD / f"roster_{s}.csv").exists()])
    ret = returning(seasons)
    cur_path = Path("/tmp/cfb_pproj_all.pkl")
    cur = pd.read_pickle(cur_path) if cur_path.exists() else PL.walk_forward(t)
    cur.to_pickle(cur_path)
    res = {"rule": "keep only if RB rushing-yards and carries RMSE (weeks 1-8) improve on the holdout by 2+ SE and not worse on tuning, "
                   "and the weeks 3-8 Brier at the 2025 rushing lines improves by 2+ SE", "early_weeks": EARLY, "feats": FEATS}

    # ---- rushing: running backs ------------------------------------------------------------
    f = features(t, bg, ret, "car_share", ("RB",))
    pri = fit_priors(f, C.FIRST_TEST_SEASON)
    res["rushing"] = {"rows_with_prior": int(len(pri)), "coverage": {"stars": float(pri["stars"].notna().mean()), "class_year": float(pri["class_year"].notna().mean()),
                      "transfer_rate": float(pri["transfer"].mean()), "team_ret": float(pri["team_ret"].notna().mean())},
                      "coef_2025": pri.loc[pri.season == 2025, "coef"].iloc[0] if (pri.season == 2025).any() else None,
                      "prior_fit": {}}
    # how well the prior itself predicts the early share, against the flat prior and the history alone
    for per, x in (("tuning 2017-19", pri[pri.season <= 2019]), ("holdout 2021+", pri[pri.season >= 2021])):
        flat = 0.5 * x.groupby("season")["target"].transform("mean")
        res["rushing"]["prior_fit"][per] = {"n": int(len(x)), "rmse_prior": float(np.sqrt(((x.prior - x.target) ** 2).mean())),
                                             "rmse_hist_share": float(np.sqrt(((x.hist_share - x.target) ** 2).mean())),
                                             "rmse_flat": float(np.sqrt(((flat - x.target) ** 2).mean()))}
    grid = {}
    for w in W_GRID:
        new = PL.walk_forward(apply_prior(t, pri, "car_share", ("RB",), w))
        grid[w] = {"rush_yds": score_early(cur, new, "rush_yds", ("RB",), 20.0), "carries": score_early(cur, new, "carries", ("RB",), 4.0), "_proj": new}
        print(f"w={w}: rush tuning {grid[w]['rush_yds']['tuning 2017-19']['mse_diff']:+.3f} (z {grid[w]['rush_yds']['tuning 2017-19']['z']:+.2f}) "
              f"holdout {grid[w]['rush_yds']['holdout 2021+']['mse_diff']:+.3f} (z {grid[w]['rush_yds']['holdout 2021+']['z']:+.2f})", flush=True)
    best_w = min(W_GRID, key=lambda w: grid[w]["rush_yds"]["tuning 2017-19"]["mse_diff"])
    new = grid[best_w].pop("_proj")
    res["rushing"]["grid"] = {str(w): {k: v for k, v in g.items() if k != "_proj"} for w, g in grid.items()}
    res["rushing"]["w"] = best_w
    res["rushing"]["backtest"] = {"rush_yds": grid[best_w]["rush_yds"], "carries": grid[best_w]["carries"]}
    res["rushing"]["lines_2025_weeks_3_8"] = score_lines(cur, new, pinfo, games, "rush_yds", "carries", weeks=(3, 8))
    res["rushing"]["lines_2025_all"] = score_lines(cur, new, pinfo, games, "rush_yds", "carries")
    bt = res["rushing"]["backtest"]
    ok_bt = all(bt[s]["holdout 2021+"]["z"] <= -2 and bt[s]["tuning 2017-19"]["mse_diff"] <= 0 for s in ("rush_yds", "carries"))
    ok_ln = res["rushing"]["lines_2025_weeks_3_8"]["z"] <= -2
    res["kept"] = bool(ok_bt and ok_ln)

    # ---- receiving, as information ------------------------------------------------------------
    fr = features(t, bg, ret, "rec_share", ("WR", "TE", "RB"))
    prr = fit_priors(fr, C.FIRST_TEST_SEASON)
    newr = PL.walk_forward(apply_prior(t, prr, "rec_share", ("WR", "TE", "RB"), best_w))
    res["receiving_info"] = {"w": best_w, "rec_yds": score_early(cur, newr, "rec_yds", ("WR", "TE", "RB"), 15.0),
                             "lines_2025_all": score_lines(cur, newr, pinfo, games, "rec_yds", "receptions")}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "feats"}, indent=1, default=float))


if __name__ == "__main__":
    main()
