"""Pre-set test: prop spread that widens after a role change (Nov 17, 2026).

    python experiments/role_change_spread_test.py --league cfb|nfl

Rule written before running (Project doc claude/plan-role-change-spread-test-2026-11-17.md):

u = |the player's share in his last appearance - his decayed share coming in| (carries share for
carries and rushing yards; the share behind his receiving role for receptions and receiving yards:
catches in college, targets in the NFL, as each board's role column), capped at 0.30, 0 with fewer
than two earlier appearances or none since last season. One added term in the existing fits:
  counts  alpha = exp(a + b log(mu) + c role + d u)
  yards   per-play shape k1 = exp(a + b log(volume) + c role + e u)
for RB carries and rushing yards and WR/TE/RB receptions and receiving yards. QBs are not fitted
or scored. The baseline is the same family refitted on the same rows with d = e = 0. The exit
mixture is refitted in both; the isotonic correction (and the college QB rushing shift) are the
board's saved ones, the same in both.

College (14): log-likelihood per stat, tuning 2017-19 (leave one season out) and holdout 2021-25
  (each season fitted on 2017 to the season before), game-day table (8); Brier at the 2024 and 2025
  DK/FD main lines, receiving and rushing yards pooled, game-day and bet-time (4); floor rungs (books'
  reading 65-95%), 2024 and 2025, game-day (2).
NFL (11): log-likelihood per stat, tuning 2014-18 (leave one season out; the walk-forward table
  starts in 2014) and holdout 2019 on (8); Brier at the 2023-25 main lines, close with game-day and
  pricing snapshot with bet-time (2); floor rungs, 2025 ladders, game-day (1).
Kept in a league only if better on all its measures, each holdout log-likelihood gain 1+ SE.
Writes scores only (experiments/out/role_change_spread_test_<league>.json).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import odds as O, parlays as PA, propdist as D, propmarket as PM, props as P  # noqa: E402

U_CAP = 0.30
HL = 4.0                      # decay of the share coming in, in appearances
BAND = (0.65, 0.95)
GROUPS = {"receptions": ("WR", "TE", "RB"), "carries": ("RB",), "rec_yds": ("WR", "TE", "RB"), "rush_yds": ("RB",)}
UCOL = {"receptions": "u_rec", "rec_yds": "u_rec", "carries": "u_car", "rush_yds": "u_car"}
LINE_STATS = ("rec_yds", "rush_yds")
MODE = {"on": True}


# ------------------------------------------------------------------ the candidate's distributions

@dataclass
class CountFitU(D.CountFit):
    d: float = 0.0

    def alpha(self, mu, role):
        r = np.asarray(role, float)
        share, u = (r[:, 0], r[:, 1]) if r.ndim == 2 else (r, 0.0)
        return np.exp(np.clip(self.a + self.b * np.log(np.maximum(mu, 0.05)) + self.c * share + self.d * u, -20, 5))


@dataclass
class YardsFitU(D.YardsFit):
    e: float = 0.0

    def k1(self, vol_mu, role):
        r = np.asarray(role, float)
        share, u = (r[:, 0], r[:, 1]) if r.ndim == 2 else (r, 0.0)
        return np.exp(np.clip(self.a + self.b * np.log(np.maximum(vol_mu, 0.05)) + self.c * share + self.e * u, -6, 6))


def fit_count_u(y, mu, role, use_u: bool, start: CountFitU | None = None) -> CountFitU:
    """props_calibration's NB + exit fit (same start and optimizer), with d free only when use_u. The
    candidate starts from the baseline's optimum with d = 0 (start), so its in-sample fit can only be
    as good or better: a cold six-parameter Nelder-Mead start stalled below the baseline (process note)."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 0.02)

    def unpack(th):
        a, b, c = th[0], th[1], th[2]
        d = th[3] if use_u else 0.0
        q, f = D._sig(th[-2]) * 0.3, D._sig(th[-1])
        return CountFitU(a, b, c, q, f, "nb", len(y), d)

    th0 = [np.log(0.1), -0.5, 0.0] + ([0.0] if use_u else []) + [-2.0, -1.0]
    if start is not None:
        lg = lambda p: float(np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))))
        th0 = [start.a, start.b, start.c] + ([0.0] if use_u else []) + [lg(start.q / 0.3), lg(start.f)]
    res = optimize.minimize(lambda th: -unpack(th).loglik(y, mu, role).mean(), np.array(th0, float), method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
    return unpack(res.x)


def fit_yards_u(y, vol_pmf, ypp, vol_mu, role, use_u: bool, start: YardsFitU | None = None) -> YardsFitU:
    y, ypp, vol_mu = (np.asarray(v, float) for v in (y, ypp, vol_mu))

    def unpack(th):
        return YardsFitU(th[0], th[1], th[2], len(y), th[3] if use_u else 0.0)

    th0 = [0.0, 0.0, 0.0] + ([0.0] if use_u else [])
    if start is not None:
        th0 = [start.a, start.b, start.c] + ([0.0] if use_u else [])
    res = optimize.minimize(lambda th: -D.compound_loglik(y, vol_pmf, ypp, unpack(th).k1(vol_mu, role)).mean(), np.array(th0, float),
                            method="Nelder-Mead", options={"maxiter": 600 if not use_u else 900, "xatol": 1e-3, "fatol": 1e-6})
    return unpack(res.x)


_ROLE = P.role


def role_u(df, stat, group):
    r = _ROLE(df, stat, group)
    if stat in UCOL and group != "QB":
        u = df[UCOL[stat]].fillna(0.0).to_numpy(float) if UCOL[stat] in df else np.zeros(len(df))
        return np.column_stack([r, u])
    return r


P.role = role_u          # the props module reads role() at call time: share and u travel together


def fit_pair(df: pd.DataFrame, family: dict, saved: P.PropModel) -> tuple[P.PropModel, P.PropModel]:
    """The baseline, then the candidate warm-started from it on the same rows."""
    base = fit_model(df, family, saved, False)
    return base, fit_model(df, family, saved, True, base)


def fit_model(df: pd.DataFrame, family: dict, saved: P.PropModel, use_u: bool, warm: P.PropModel | None = None) -> P.PropModel:
    """PropModel.fit's loop for the candidate's stats and groups (20,000-row sample, 8,000 for yards)."""
    m = P.PropModel(family)
    rng = np.random.default_rng(0)
    for stat in ("receptions", "carries", "rec_yds", "rush_yds"):
        s = P.SPECS[stat]
        for g in GROUPS[stat]:
            d = df[P.eligible(df, stat, g)]
            if len(d) < 200:
                continue
            if len(d) > 20000:
                d = d.iloc[rng.choice(len(d), 20000, replace=False)]
            if s.kind == "count":
                m.counts[(stat, g)] = fit_count_u(P.actual(d, stat), P.mean(d, stat), P.role(d, stat, g), use_u,
                                                  warm.counts.get((stat, g)) if warm else None)
            else:
                vol = m._vol_pmf(d, stat, g)
                mu_v = np.maximum(P.mean(d, s.vol), 0.05)
                k = min(len(d), 8000)
                m.yards[(stat, g)] = fit_yards_u(P.actual(d, stat)[:k], vol[:k], (P.mean(d, stat) / mu_v)[:k], mu_v[:k],
                                                 P.role(d, stat, g)[:k], use_u, warm.yards.get((stat, g)) if warm else None)
    m.iso, m.shift = dict(saved.iso), dict(saved.shift)
    return m


# ------------------------------------------------------------------ u

def add_u(t: pd.DataFrame, catch_col: str) -> pd.DataFrame:
    """u_car and u_rec on every row, from earlier appearances only."""
    t = t.copy()
    t["_i"] = np.arange(len(t))
    pl = t["played"].astype(bool)
    touch = t.reindex(columns=["carries", "receptions", "targets", "pass_att"]).fillna(0).sum(axis=1) > 0
    h = t[pl & touch][["_i", "player_id", "game_id", "team", "season", "t", "carries", catch_col]].copy()
    tot = t[pl].groupby(["game_id", "team"])[["carries", catch_col]].sum()
    h = h.join(tot, on=["game_id", "team"], rsuffix="_team")
    h["s_car"] = h["carries"] / h["carries_team"].where(h["carries_team"] > 0)
    h["s_rec"] = h[catch_col] / h[catch_col + "_team"].where(h[catch_col + "_team"] > 0)
    h = h.sort_values(["player_id", "t"])
    g = h.groupby("player_id")
    h["n_app"] = g.cumcount() + 1
    for c in ("car", "rec"):
        dec = g["s_" + c].transform(lambda s: s.fillna(0).ewm(halflife=HL).mean())
        h["after_" + c] = (h["s_" + c].fillna(0) - dec).abs().clip(upper=U_CAP)
    q = t[["_i", "player_id", "t", "season"]].sort_values("t")
    hh = h[["player_id", "t", "season", "n_app", "after_car", "after_rec"]].rename(columns={"t": "h_t", "season": "h_season"}).sort_values("h_t")
    m = pd.merge_asof(q, hh, left_on="t", right_on="h_t", by="player_id", allow_exact_matches=False, direction="backward")
    m = m.set_index("_i").reindex(t["_i"])
    ok = (m["n_app"].fillna(0).to_numpy() >= 2) & (m["h_season"].to_numpy() >= t["season"].to_numpy() - 1)
    t["u_car"] = np.where(ok, m["after_car"].fillna(0).to_numpy(), 0.0)
    t["u_rec"] = np.where(ok, m["after_rec"].fillna(0).to_numpy(), 0.0)
    return t.drop(columns="_i")


# ------------------------------------------------------------------ scores

def clustered(d, game) -> tuple[float, float]:
    d = np.asarray(d, float)
    cl = pd.Series(d).groupby(np.asarray(game)).sum()
    return float(d.mean()), float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(d))


def loglik_rows(df: pd.DataFrame, m: P.PropModel, stat: str) -> pd.DataFrame:
    out = []
    s = P.SPECS[stat]
    for g in GROUPS[stat]:
        d = df[P.eligible(df, stat, g)].reset_index(drop=True)
        if d.empty:
            continue
        y = P.actual(d, stat)
        if s.kind == "count":
            ll = m.counts[(stat, g)].loglik(y, np.maximum(P.mean(d, stat), 0.02), P.role(d, stat, g))
        else:
            vol = m._vol_pmf(d, stat, g)
            mu_v = np.maximum(P.mean(d, s.vol), 0.05)
            ll = D.compound_loglik(y, vol, P.mean(d, stat) / mu_v, m.yards[(stat, g)].k1(mu_v, P.role(d, stat, g)))
        out.append(pd.DataFrame({"game_id": d["game_id"].to_numpy(), "player_id": d["player_id"].to_numpy(), "group": g,
                                 "u": d[UCOL[stat]].to_numpy(float), "ll": ll}))
    return pd.concat(out, ignore_index=True)


def ll_block(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    x = a.merge(b[["game_id", "player_id", "ll"]], on=["game_id", "player_id"], suffixes=("_base", "_u"))
    def one(z):
        dd, se = clustered(z["ll_u"] - z["ll_base"], z["game_id"])
        return {"rows": int(len(z)), "ll_base": round(float(z["ll_base"].mean()), 5), "ll_u": round(float(z["ll_u"].mean()), 5),
                "gain": round(dd, 6), "se": round(se, 6), "z": round(dd / se, 2) if se else None}
    return {"all": one(x), "u >= 0.15": one(x[x["u"] >= 0.15]) if (x["u"] >= 0.15).sum() > 30 else None}


def chance(x: pd.DataFrame) -> np.ndarray:
    return (x["our_over"] / (x["our_over"] + x["our_under"])).to_numpy()


def brier_pair(a: pd.DataFrame, b: pd.DataFrame, key: list, books_col: str) -> dict:
    x = a.merge(b[key + ["our_over", "our_under"]], on=key, suffixes=("", "_u"))
    x = x[x["y"] != x["line"]]
    hit = (x["y"] > x["line"]).astype(float).to_numpy()
    p0 = chance(x)
    p1 = (x["our_over_u"] / (x["our_over_u"] + x["our_under_u"])).to_numpy()
    g = x["game_id"].to_numpy()
    dd, se = clustered((p1 - hit) ** 2 - (p0 - hit) ** 2, g)
    db, sb = clustered((p1 - hit) ** 2 - (x[books_col].to_numpy() - hit) ** 2, g)
    dc, _ = clustered((p0 - hit) ** 2 - (x[books_col].to_numpy() - hit) ** 2, g)
    out = {"props": int(len(x)), "games": int(len(set(g))), "brier_base": round(D.brier(p0, hit), 5), "brier_u": round(D.brier(p1, hit), 5),
           "u_minus_base": round(dd, 6), "se": round(se, 6), "z": round(dd / se, 2) if se else None,
           "base_minus_books": round(dc, 5), "u_minus_books": round(db, 5), "books_se": round(sb, 5),
           "mean_abs_move": round(float(np.abs(p1 - p0).mean()), 4)}
    for st in sorted(x["stat"].unique()):
        k = (x["stat"] == st).to_numpy()
        d2, s2 = clustered(((p1 - hit) ** 2 - (p0 - hit) ** 2)[k], g[k])
        out[st] = {"props": int(k.sum()), "u_minus_base": round(d2, 6), "se": round(s2, 6)}
    return out


# ------------------------------------------------------------------ leagues

def cfb_tables():
    import cfb_rush_group_calib_test as G
    from cfbmodel import players as PL
    t, pinfo, games = G.table()
    out = {}
    for kind in ("game-day", "bet-time"):
        cache = Path(f"/tmp/cfb_boost_cur_{kind}.pkl")
        if cache.exists():
            cur = pd.read_pickle(cache)
        else:
            tb = t.copy()
            if kind == "bet-time":
                m = tb["played"].astype(bool)
                tb.loc[m, "active"] = tb.loc[m, "expected"].astype(int)
            cur = PL.walk_forward(tb)
            cur.to_pickle(cache)
        cur["tgt_share"] = cur.get("rec_share", 0.0)
        out[kind] = add_u(cur, "receptions")
    return out, pinfo, games


def cfb_lines(tables, models, pinfo, games) -> tuple[dict, dict]:
    import cfb_props_real_lines as R
    judged, rungs = {}, {}
    key = ["game_id", "nname", "stat", "line"]
    for season in (2024, 2025):
        R.set_season(season)
        s = R.snapshots()
        gid = R.match_events(s, games)
        s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
        s["nname"] = s["player"].map(PM.norm_name)
        s["stat"] = s["market"].map(PM.MARKET_STAT)
        main = s[~s["market"].str.endswith("_alternate") & s["stat"].isin(LINE_STATS)]
        w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
        w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
        w = w.groupby(key, as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
        base, cand = models[("hold", season)]
        for kind, tab in tables.items():
            rows = R.player_frame(tab, pinfo)
            x = w.merge(rows, on=["game_id", "nname"], how="inner")
            a, b = R.model_chances(x, base), R.model_chances(x, cand)
            judged[f"{season} main lines {kind}"] = brier_pair(a, b, key, "fair")
        # floor rungs, game-day
        alt = s[s["market"].str.endswith("_alternate") & (s["side"] == "over") & s["stat"].isin(LINE_STATS)]
        alt = alt.groupby(key, as_index=False)["price"].max()
        alt["books"] = np.interp(O.implied_prob(alt["price"]), *zip(*PA.CFB_RUNG_CAL))
        alt = alt[alt["books"].between(*BAND)]
        x = alt.merge(R.player_frame(tables["game-day"], pinfo), on=["game_id", "nname"], how="inner")
        a, b = R.model_chances(x, base), R.model_chances(x, cand)
        rungs[f"{season} floor rungs game-day"] = brier_pair(a, b, key, "books")
    return judged, rungs


def nfl_tables():
    T = pickle.load(open("/tmp/rcs/nfl_tables.pkl", "rb"))
    out = {}
    for kind, k in (("game-day", "gameday"), ("bet-time", "bettime")):
        out[kind] = add_u(T[k], "targets")
    return out, None, None


def nfl_lines(tables, models) -> tuple[dict, dict]:
    import props_real_lines as RL
    from nflmodel import propbackfill as B
    key = ["game_id", "nname", "market", "line", "book"]
    judged, rungs = {}, {}
    for kind, snapkind in (("game-day", "close"), ("bet-time", "flag")):
        got_a, got_b = [], []
        for season in (2023, 2024, 2025):
            snaps = B.load(season, snapkind)
            if snaps.empty:
                continue
            snaps = snaps[~snaps["market"].str.endswith("_alternate")]
            base, cand = models[("hold", season)]
            a, b = RL.priced(tables[kind], base, snaps, season), RL.priced(tables[kind], cand, snaps, season)
            got_a.append(a[a["stat"].isin(LINE_STATS)]), got_b.append(b[b["stat"].isin(LINE_STATS)])
        judged[f"2023-25 main lines {kind} ({snapkind})"] = brier_pair(pd.concat(got_a, ignore_index=True), pd.concat(got_b, ignore_index=True), key, "fair_over")
    # floor rungs: 2025 ladders at the close, game-day
    snaps = B.load(2025, "close")
    alt = snaps[snaps["market"].str.endswith("_alternate")].copy()
    alt["side"] = alt["side"].str.lower()
    alt = alt[alt["side"] == "over"]
    alt["stat"] = alt["market"].map(PM.MARKET_STAT)
    alt = alt[alt["stat"].isin(LINE_STATS)]
    alt = alt.groupby(["game_id", "player", "market", "stat", "line"], as_index=False)["price"].max()
    alt["books"] = np.interp(O.implied_prob(alt["price"]), *zip(*PA.RUNG_CAL))
    alt = alt[alt["books"].between(*BAND)]
    alt["side"], alt["book"] = "over", "best"
    two = pd.concat([alt.assign(side="over"), alt.assign(side="under", price=100)], ignore_index=True)   # priced() needs two sides
    base, cand = models[("hold", 2025)]
    a, b = RL.priced(tables["game-day"], base, two, 2025), RL.priced(tables["game-day"], cand, two, 2025)
    a = a.merge(alt[["game_id", "market", "line", "books"]].assign(nname=alt["player"].map(PM.norm_name)), on=["game_id", "nname", "market", "line"])
    rungs["2025 floor rungs game-day"] = brier_pair(a, b, key, "books")
    return judged, rungs


LEAGUE = {
    "cfb": {"tune": [2017, 2018, 2019], "hold": [2021, 2022, 2023, 2024, 2025], "first": 2017, "params": ROOT / "data" / "cfb" / "prop_params.json"},
    "nfl": {"tune": [2014, 2015, 2016, 2017, 2018], "hold": list(range(2019, 2027)), "first": 2014, "params": ROOT / "data" / "prop_params.json"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(LEAGUE), required=True)
    a = ap.parse_args()
    L = LEAGUE[a.league]
    out_path = HERE / "out" / f"role_change_spread_test_{a.league}.json"
    t0 = time.time()
    tables, pinfo, games = cfb_tables() if a.league == "cfb" else nfl_tables()
    gd = tables["game-day"]
    gd_played = gd[gd["played"].astype(bool)]
    saved = P.PropModel.load(L["params"])
    family = saved.family
    res = {"plan": "claude/plan-role-change-spread-test-2026-11-17.md", "league": a.league,
           "u_share": {k: round(float((gd_played[c] >= 0.15).mean()), 4) for k, c in (("carries u >= 0.15", "u_car"), ("catch u >= 0.15", "u_rec"))},
           "loglik": {}, "coef": {}}
    models = {}
    ll = {st: {"tune": [], "hold": []} for st in GROUPS}
    jobs = [("tune", s, [x for x in L["tune"] if x != s]) for s in L["tune"]] + \
           [("hold", s, list(range(L["first"], s))) for s in L["hold"]]
    for per, season, train_seasons in jobs:
        tr = gd_played[gd_played["season"].isin(train_seasons)]
        te = gd_played[gd_played["season"] == season]
        if te.empty:
            continue
        base, cand = fit_pair(tr, family, saved)
        models[(per, season)] = (base, cand)
        res["coef"][f"{per} {season}"] = {f"{k[0]}|{k[1]}": round(float(v.d), 3) for k, v in cand.counts.items()} | \
                                          {f"{k[0]}|{k[1]}": round(float(v.e), 3) for k, v in cand.yards.items()}
        for st in GROUPS:
            ll[st][per].append((loglik_rows(te, base, st), loglik_rows(te, cand, st)))
        print(per, season, "fitted", round(time.time() - t0), "s", res["coef"][f"{per} {season}"], flush=True)
    for st in GROUPS:
        for per, label in (("tune", "tuning"), ("hold", "holdout")):
            A = pd.concat([x[0] for x in ll[st][per]], ignore_index=True)
            Bb = pd.concat([x[1] for x in ll[st][per]], ignore_index=True)
            res["loglik"][f"{st} {label}"] = ll_block(A, Bb)
        print(st, {k: v["all"] for k, v in res["loglik"].items() if k.startswith(st + " ")}, flush=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    if a.league == "cfb":
        lines, rungs = cfb_lines(tables, models, pinfo, games)
    else:
        lines, rungs = nfl_lines(tables, models)
    res["lines"], res["rungs"] = lines, rungs
    for k, v in {**lines, **rungs}.items():
        print(k, {x: v[x] for x in ("props", "u_minus_base", "se", "z", "base_minus_books", "u_minus_books")}, flush=True)
    ok_ll = all(v["all"]["gain"] > 0 for k, v in res["loglik"].items()) and \
        all(v["all"]["z"] is not None and v["all"]["z"] >= 1 for k, v in res["loglik"].items() if k.endswith("holdout"))
    ok_lines = all(v["u_minus_base"] < 0 for v in lines.values()) and all(v["u_minus_base"] < 0 for v in rungs.values())
    res["measures"] = {"loglik": len(res["loglik"]), "lines": len(lines), "rungs": len(rungs)}
    res["kept"] = bool(ok_ll and ok_lines)
    res["loglik_pass"], res["lines_pass"] = bool(ok_ll), bool(ok_lines)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    print("loglik", ok_ll, "lines/rungs", ok_lines, "kept", res["kept"], round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
