"""Pricing NFL floor rungs off the books' own main line: a pre-set test. No model projection goes in.

    PFF_DATA_DIR=<pff-data clone> python experiments/market_ladder_test.py freeze
    PFF_DATA_DIR=<pff-data clone> python experiments/market_ladder_test.py score   # after week 13 is graded

Plan written before any run (Project doc claude/plan-market-ladder-test-2026-11-18.md). The rung set is
the floor-rung diagnosis's (experiments/floor_rung_diagnosis.py; PFF is needed only to rebuild that set).

Rungs: DK/FD alternate passing, rushing and receiving yards, over only, the books' reading
(parlays.RUNG_CAL) 65-95%, for player-games where DK or FD posts a two-sided main line.
Fair main-line chance: each book's main over/under de-vigged with the power method (odds.no_vig_power).
A rung's curve chance is worked out from each book's main line and averaged over the books posting one.
Versions (shape of the outcome spread, per stat):
  A  normal, sd = k * line**b, centred so the main line's fair chance holds.
  B  empirical: the distribution of actual / main line, by stat and main-line third, shifted
     (additively, in the ratio) so the main line's fair chance holds.
  C  gamma, coefficient of variation = k * line**b, scaled so the main line's fair chance holds.
freeze (2025 only): five blocks of consecutive weeks, each scored with a shape fitted on the other
  four. Chosen: lowest cross-fitted Brier. Refitted on all of 2025 and written to
  experiments/frozen/market_ladder_shape.json. No 2026 data is read for the fit. Tuning measures
  (1 Brier v the books' reading, 3 return, 5 closing line value) are recorded with it.
score (holdout 2026 weeks 8-13, the live prop log; weeks 8-10 and 11-13 also reported apart):
  2 Brier v the books' reading, 4 return, 6 closing line value.
  Flagged rung: curve EV at the better DK/FD price >= +3%.
  Price decision: kept only if the curve beats the books' reading on 1 and 2, the holdout by 1+ paired
  SE (clustered by game); one of the two, or under 1 SE -> trade-off. Curve loses on 1 -> no price
  decision on the holdout.
  Strategy decision: 3-6 all above zero and the holdout 30+ bets with return and CLV above zero by 2 SE
  -> staked strategy (Ryan's say-so); all above zero short of that -> live record, no stake.
  Reported, not used: 25%-boost three-leg parlay, DK v FD disagreement, cuts (stat, position, rung /
  main line, team projected to lose by 7+).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import config as C, odds as O, propbackfill as B, propmarket as PM  # noqa: E402

FROZEN = HERE / "frozen" / "market_ladder_shape.json"
OUT = HERE / "out" / "market_ladder_test.json"
STATS = ("pass_yds", "rush_yds", "rec_yds")
MAIN = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds"}
BOOKS = ("draftkings", "fanduel")
EV_FLAG = 0.03
HOLDOUT = tuple(range(8, 14))
QGRID = np.linspace(0, 1, 401)


# ---------- the books' main lines ----------

def _fair(m: pd.DataFrame, keys: list) -> pd.DataFrame:
    w = m.pivot_table(index=[*keys, "book", "line"], columns="side", values="price", aggfunc="last").reset_index()
    w = w.dropna(subset=["over", "under"])
    w["p_main"] = O.no_vig_power(w["over"].to_numpy(), w["under"].to_numpy())[0]
    w = w.assign(d=(w["p_main"] - 0.5).abs()).sort_values("d").drop_duplicates([*keys, "book"])   # one main line per book
    return w.rename(columns={"line": "main"})[[*keys, "book", "main", "p_main"]]


def mains_2025() -> pd.DataFrame:
    f = B.load(2025, "flag")
    m = f[f["market"].isin(MAIN) & f["book"].isin(BOOKS)].copy()
    m["stat"], m["nname"], m["side"] = m["market"].map(MAIN), m["player"].map(PM.norm_name), m["side"].str.lower()
    m = m.sort_values("updated")
    return _fair(m, ["game_id", "nname", "stat"])


def mains_2026() -> pd.DataFrame:
    log = pd.read_csv(C.DATA / "props" / "prop_log.csv", low_memory=False)
    m = log[(log["snapshot"] == "flag") & log["market"].isin(MAIN) & log["book"].isin(BOOKS)].copy()
    m["stat"], m["side"] = m["market"].map(MAIN), m["side"].str.lower()
    m = m.sort_values("time")
    return _fair(m, ["game_id", "player_id", "stat"])


def attach(x: pd.DataFrame, mains: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per rung row, with a list of (main, p_main) per book posting a main line."""
    g = mains.groupby(["game_id", key, "stat"]).apply(lambda d: list(zip(d["main"], d["p_main"]))).rename("mains").reset_index()
    return x.merge(g, on=["game_id", key, "stat"], how="inner")


# ---------- the curve ----------

def _cdf_over(version: str, prm: dict, r, L, p, stat):
    r, L, p = np.asarray(r, float), np.asarray(L, float), np.clip(np.asarray(p, float), 0.02, 0.98)
    if version == "A":
        sd = prm["k"] * L ** prm["b"]
        mu = L - sd * stats.norm.isf(p)
        return stats.norm.sf((r - mu) / sd)
    if version == "C":
        cv = np.clip(prm["k"] * L ** prm["b"], 0.05, 3.0)
        a = 1 / cv ** 2
        scale = L / stats.gamma.isf(p, a)
        return stats.gamma.sf(np.maximum(r, 0) / scale, a)
    # B: empirical ratio, by main-line third
    band = np.searchsorted(np.asarray(prm["cuts"]), L)
    out = np.empty(len(r))
    for b in np.unique(band):
        k = band == b
        q = np.asarray(prm["q"][int(b)])                         # quantiles of actual / main at QGRID
        delta = np.interp(1 - p[k], QGRID, q) - 1.0              # shift so P(R > 1) = p
        out[k] = 1 - np.interp(r[k] / L[k] + delta, q, QGRID)
    return out


def curve(x: pd.DataFrame, shape: dict) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for st in STATS:
        k = np.flatnonzero((x["stat"] == st).to_numpy())
        if not len(k):
            continue
        prm, v = shape["params"][st], shape["version"]
        rr = x["line"].to_numpy()[k]
        acc, cnt = np.zeros(len(k)), np.zeros(len(k))
        for j in range(len(BOOKS)):
            L = np.array([m[j][0] if len(m) > j else np.nan for m in x["mains"].to_numpy()[k]])
            p = np.array([m[j][1] if len(m) > j else np.nan for m in x["mains"].to_numpy()[k]])
            ok = ~np.isnan(L)
            if ok.any():
                acc[ok] += _cdf_over(v, prm, rr[ok], L[ok], p[ok], st)
                cnt[ok] += 1
        out[k] = acc / cnt
    return np.clip(out, 0.001, 0.999)


def fit(x: pd.DataFrame, version: str) -> dict:
    params = {}
    for st in STATS:
        g = x[x["stat"] == st]
        if version == "B":
            one = g.drop_duplicates(["game_id", "player_id", "stat"])
            L = one["mains"].map(lambda m: np.mean([a for a, _ in m])).to_numpy()
            R = one["y"].to_numpy() / L
            cuts = list(np.quantile(L, [1 / 3, 2 / 3]))
            band = np.searchsorted(cuts, L)
            params[st] = {"cuts": [round(float(c), 2) for c in cuts], "q": [[round(float(v), 4) for v in np.quantile(R[band == b], QGRID)] for b in range(3)]}
            continue
        y = g["hit"].to_numpy()

        def loss(t):
            prm = {"k": float(np.exp(t[0])), "b": float(t[1])}
            pr = curve(g.assign(stat=st), {"version": version, "params": {st: prm, **{s: prm for s in STATS if s != st}}})
            return float(((pr - y) ** 2).mean())
        t0 = [np.log(0.4), 1.0] if version == "A" else [np.log(0.6), 0.0]
        r = optimize.minimize(loss, t0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 400})
        params[st] = {"k": round(float(np.exp(r.x[0])), 6), "b": round(float(r.x[1]), 6)}
    return {"version": version, "params": params}


def blocks(x: pd.DataFrame) -> pd.Series:
    weeks = sorted(x["week"].unique())
    lab = {int(w): i for i, c in enumerate(np.array_split(np.array(weeks), 5)) for w in c}
    return x["week"].map(lab)


def crossfit(x: pd.DataFrame, version: str) -> np.ndarray:
    blk, out = blocks(x), np.full(len(x), np.nan)
    for b in sorted(blk.unique()):
        tr, te = (blk != b).to_numpy(), (blk == b).to_numpy()
        out[te] = curve(x[te], fit(x[tr], version))
    return out


# ---------- measures ----------

def vs_books(c, x) -> dict:
    y, g = x["hit"].to_numpy(), x["game_id"].to_numpy()
    bc, bb = (c - y) ** 2, (x["books"].to_numpy() - y) ** 2
    d = pd.Series(bb - bc).groupby(g).sum()
    n = len(y)
    return {"rungs": int(n), "games": int(len(d)), "brier_curve": round(float(bc.mean()), 5), "brier_books": round(float(bb.mean()), 5),
            "gain": round(float(d.sum() / n), 5), "se": round(float(d.std(ddof=1) * np.sqrt(len(d)) / n), 5),
            "curve_says": round(float(np.mean(c)), 4), "books_say": round(float(x["books"].mean()), 4), "cleared": round(float(y.mean()), 4)}


def flagged(x: pd.DataFrame, c) -> pd.DataFrame:
    x = x.assign(curve=c, dec=O.american_to_decimal(x["price"].to_numpy()))
    best = x.sort_values("dec").drop_duplicates(["game_id", "player_id", "stat", "line"], keep="last")
    return best[best["curve"] * best["dec"] - 1 >= EV_FLAG]


def money(f: pd.DataFrame) -> dict:
    import floor_rung_diagnosis as D
    return D.money(f)


# ---------- data ----------

def load():
    import floor_rung_diagnosis as D
    p = D.projections()
    a = D.cuts(D.rungs_2025(p))
    b = D.rungs_2026(p)
    b["proj"] = pd.to_numeric(b["proj"], errors="coerce")
    b = D.cuts(b)
    return a, b


def prep_2025(a: pd.DataFrame) -> pd.DataFrame:
    a = a[a["books"].between(0.65, 0.95)].copy()
    return attach(a, mains_2025(), "nname").reset_index(drop=True)


def prep_2026(b: pd.DataFrame) -> pd.DataFrame:
    b = b[b["books"].between(0.65, 0.95)].copy()
    b["y"] = b["actual"]
    return attach(b, mains_2026(), "player_id").reset_index(drop=True)


def freeze(a: pd.DataFrame) -> dict:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} already exists: the shape is frozen. Delete it only with Ryan's say-so.")
    x = prep_2025(a)
    tune = {}
    for v in ("A", "B", "C"):
        c = crossfit(x, v)
        tune[v] = vs_books(c, x)
        tune[v]["money_flagged"] = money(flagged(x, c))
    best = min(tune, key=lambda v: tune[v]["brier_curve"])
    shape = fit(x, best)
    rec = {"written": pd.Timestamp.now(tz="America/New_York").isoformat(timespec="minutes"),
           "plan": "claude/plan-market-ladder-test-2026-11-18.md", "fit_rows_2025": int(len(x)), "fit_games_2025": int(x["game_id"].nunique()),
           "rungs_without_a_main_line_dropped_2025": int(a["books"].between(0.65, 0.95).sum() - len(x)),
           "tuning_crossfit_2025": tune, "chosen": best, "shape": shape,
           "price_decision_goes_to_holdout": bool(tune[best]["gain"] > 0)}
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps(rec, indent=1))
    return rec


def score(b: pd.DataFrame) -> dict:
    rec = json.loads(FROZEN.read_text())
    h = prep_2026(b)
    h = h[h["week"].isin(HOLDOUT)].reset_index(drop=True)
    h["curve"] = curve(h, rec["shape"])
    t = rec["tuning_crossfit_2025"][rec["chosen"]]
    ho = vs_books(h["curve"].to_numpy(), h)
    fh = flagged(h, h["curve"].to_numpy())
    mh = money(fh)
    res = {"frozen": {k: rec[k] for k in ("written", "chosen", "price_decision_goes_to_holdout")}, "tuning": t, "holdout": ho, "holdout_money": mh,
           "holdout_parts": {f"weeks {lo}-{hi}": vs_books(g["curve"].to_numpy(), g) for (lo, hi), g in
                             (((8, 10), h[h["week"] <= 10]), ((11, 13), h[h["week"] >= 11])) if len(g)}}
    if rec["price_decision_goes_to_holdout"]:
        both = t["gain"] > 0 and ho["gain"] > 0
        res["price_decision"] = "kept (provisional)" if both and ho["gain"] >= ho["se"] else ("trade-off for Ryan" if (t["gain"] > 0) != (ho["gain"] > 0) or both else "not kept")
    else:
        res["price_decision"] = "not taken to the holdout (the curve lost to the books' reading on tuning)"
    tm = t["money_flagged"]
    vals = [tm.get("return_per_dollar", -1), mh.get("return_per_dollar", -1), (tm.get("price_clv") or {}).get("mean", -1), (mh.get("price_clv") or {}).get("mean", -1)]
    bar = mh.get("bets", 0) >= 30 and mh["return_per_dollar"] >= 2 * (mh["se"] or 9) and (mh.get("price_clv") or {}).get("mean", -1) >= 2 * ((mh.get("price_clv") or {}).get("se") or 9)
    res["strategy_decision"] = "staked strategy, with Ryan's say-so" if all(v > 0 for v in vals) and bar else ("live record, no stake" if all(v > 0 for v in vals) else "not a strategy")
    # reported only
    dk = h[h["book"] == "draftkings"][["game_id", "player_id", "stat", "line", "price"]]
    fd = h[h["book"] == "fanduel"][["game_id", "player_id", "stat", "line", "price"]]
    j = dk.merge(fd, on=["game_id", "player_id", "stat", "line"], suffixes=("_dk", "_fd"))
    gap = O.implied_prob(j["price_dk"]) - O.implied_prob(j["price_fd"])
    rep = {"dk_v_fd_same_rung": {"pairs": int(len(j)), "gap_3pts_or_more": int((gap.abs() >= 0.03).sum())}, "cuts": {}}
    for c in ("c_stat", "c_position", "c_rung_vs_main", "c_margin"):
        rep["cuts"][c[2:]] = {str(k): {**vs_books(g["curve"].to_numpy(), g), "money_flagged": money(flagged(g, g["curve"].to_numpy()))} for k, g in h.groupby(c) if len(g) >= 50}
    res["reported"] = rep
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("freeze", "score"):
        raise SystemExit(__doc__)
    a, b = load()
    out = freeze(a) if mode == "freeze" else score(b)
    if mode == "score":
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "shape"}, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
