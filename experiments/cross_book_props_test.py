"""Pricing one book's yardage props off the other book's main line (NFL, DraftKings and FanDuel).

    python experiments/cross_book_props_test.py . freeze     # fits the spread on 2023-24, tuning measures
    python experiments/cross_book_props_test.py . holdout    # 2025 (Part M); Part R tuning is 2025

Rule written before any run: Project doc claude/plan-cross-book-props-test-2026-11-19.md.
Reference: the other book's main line, power-method fair chance, carried to this book's line through a
gamma curve (coefficient of variation = k * line**b per stat, fitted on 2023-24 main lines and results).
X1 plain: EV >= +3% at this book's price. X2: the same with the earlier seasons' over lean (a logit
intercept per stat; 2024 from 2023, 2025 from 2023-24). X0 control: the book's own fair chance with
that shade, no other book. One bet per player-stat-snapshot (highest EV).
Part M main lines: tuning 2023-24, holdout 2025. Part R alternate rungs (over only): tuning 2025,
holdout 2026 weeks 8-13 (live log, scored after week 13).
Live record only if return and closing line value are above zero on tuning and holdout (X2 also above
X0); stakes only at 30+ holdout bets with both above zero by 2 SE.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize, special, stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import book_consistency_common as K

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
FROZEN = OUT / "cross_book_spread_frozen.json"
STATS = ("pass_yds", "rush_yds", "rec_yds")
EV_FLAG = 0.03


def _ascale(L, p, prm):
    cv = np.clip(prm["k"] * np.asarray(L, float) ** prm["b"], 0.05, 3.0)
    a = 1 / cv ** 2
    return a, np.asarray(L, float) / stats.gamma.isf(np.clip(p, 0.02, 0.98), a)


def ref_over(line, L, p, prm):
    a, sc = _ascale(L, p, prm)
    return np.clip(stats.gamma.sf(np.maximum(np.asarray(line, float), 0) / sc, a), 0.001, 0.999)


def fit_spread(x):
    out = {}
    for st in STATS:
        g = x[x["stat"] == st]
        L, p, y = g["line"].to_numpy(float), g["p_over"].to_numpy(float), np.maximum(g["y"].to_numpy(float), 0.5)
        def nll(v):
            prm = {"k": np.exp(v[0]), "b": v[1]}
            a, sc = _ascale(L, p, prm)
            return -np.sum(stats.gamma.logpdf(y, a, scale=sc))
        r = optimize.minimize(nll, [np.log(2.0), -0.3], method="Nelder-Mead")
        out[st] = {"k": float(np.exp(r.x[0])), "b": float(r.x[1]), "rows": int(len(g))}
    return out


def lean(x):
    """Logit intercept per stat: results against the book's own fair chance."""
    out = {}
    for st in STATS:
        g = x[(x["stat"] == st) & (x["y"] != x["line"])]
        off, hit = special.logit(g["p_over"].clip(0.02, 0.98)), (g["y"] > g["line"]).astype(float).to_numpy()
        f = lambda a: -np.sum(hit * -np.logaddexp(0, -(off + a)) + (1 - hit) * -np.logaddexp(0, off + a))
        out[st] = float(optimize.minimize_scalar(f, bounds=(-1, 1), method="bounded").x)
    return out


def shade(p, stat, ln):
    return special.expit(special.logit(p) + pd.Series(stat).map(ln).to_numpy()) if ln else p


def candidates_main(x, shape, ln, own=False):
    """Rows: a side at a book, with the reference chance and EV. own=True: X0 (the book's own fair)."""
    a = x.copy()
    if own:
        a["p_ref"] = a["p_over"]
    else:
        b = x[["game_id", "nname", "stat", "book", "line", "p_over"]].rename(columns={"book": "book_b", "line": "line_b", "p_over": "p_b"})
        a = a.merge(b, on=["game_id", "nname", "stat"])
        a = a[a["book"] != a["book_b"]].copy()
        a["p_ref"] = np.nan
        for st in STATS:
            k = a["stat"] == st
            a.loc[k, "p_ref"] = ref_over(a.loc[k, "line"], a.loc[k, "line_b"], a.loc[k, "p_b"], shape[st])
        a["gap"] = a["line"] - a["line_b"]
    a["p_ref"] = shade(a["p_ref"].to_numpy(), a["stat"].to_numpy(), ln)
    rows = []
    for side in ("over", "under"):
        d = a.copy(); d["side"] = side
        d["p_side"] = d["p_ref"] if side == "over" else 1 - d["p_ref"]
        d["price"] = d[side]; d["ev"] = d["p_side"] * K.dec(d["price"]) - 1
        d["win"] = np.where(d["y"] == d["line"], np.nan, (d["y"] > d["line"]) == (side == "over"))
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def candidates_rungs(alt, mains, shape, ln):
    b = mains[["game_id", "nname", "stat", "book", "line", "p_over"]].rename(columns={"book": "book_b", "line": "line_b", "p_over": "p_b"})
    a = alt.merge(b, on=["game_id", "nname", "stat"])
    a = a[a["book"] != a["book_b"]].copy()
    a["p_ref"] = np.nan
    for st in STATS:
        k = a["stat"] == st
        a.loc[k, "p_ref"] = ref_over(a.loc[k, "line"], a.loc[k, "line_b"], a.loc[k, "p_b"], shape[st])
    a["p_ref"] = shade(a["p_ref"].to_numpy(), a["stat"].to_numpy(), ln)
    a["side"], a["p_side"], a["gap"] = "over", a["p_ref"], a["line"] - a["line_b"]
    a["ev"] = a["p_side"] * K.dec(a["price"]) - 1
    a["win"] = np.where(a["y"] == a["line"], np.nan, a["y"] > a["line"])
    return a


def pick(c):
    f = c[c["ev"] >= EV_FLAG].sort_values("ev", ascending=False).drop_duplicates(["game_id", "nname", "stat"]).copy()
    f["profit"] = np.where(f["win"].isna(), 0.0, np.where(f["win"] == 1, K.dec(f["price"]) - 1, -1.0))
    return f


def clv(f, close, shape):
    """The bet's EV at the closing reference (each book's closing main carried to the bet's line, averaged)."""
    if f.empty:
        return f.assign(clv=np.nan)
    c = close[["game_id", "nname", "stat", "book", "line", "p_over"]].rename(columns={"book": "cb", "line": "cl", "p_over": "cp"})
    m = f.reset_index(names="bet").merge(c, on=["game_id", "nname", "stat"])
    m["pc"] = np.nan
    for st in STATS:
        k = m["stat"] == st
        if k.any():
            m.loc[k, "pc"] = ref_over(m.loc[k, "line"], m.loc[k, "cl"], m.loc[k, "cp"], shape[st])
    pc = m.groupby("bet")["pc"].mean()
    f = f.copy(); f["pc"] = pc.reindex(f.index)
    f["clv"] = np.where(f["side"] == "over", f["pc"], 1 - f["pc"]) * K.dec(f["price"]) - 1
    return f


def summ(f, cuts=True):
    if f.empty:
        return {"bets": 0}
    w = f["win"].dropna()
    o = {"bets": int(len(f)), "record": f"{int((w == 1).sum())}-{int((w == 0).sum())}", "ret": K.mean_se(f["profit"], f["game_id"]),
         "avg_decimal": round(float(K.dec(f["price"]).mean()), 3), "avg_ev_claimed": round(float(f["ev"].mean()), 4)}
    if "clv" in f:
        c = f.dropna(subset=["clv"]); o["clv"] = K.mean_se(c["clv"], c["game_id"])
    if cuts:
        o["by_side"] = {s: K.mean_se(f[f["side"] == s]["profit"], f[f["side"] == s]["game_id"]) for s in f["side"].unique()}
        o["by_stat"] = {s: K.mean_se(f[f["stat"] == s]["profit"], f[f["stat"] == s]["game_id"]) for s in STATS}
        o["by_book"] = {s: K.mean_se(f[f["book"] == s]["profit"], f[f["book"] == s]["game_id"]) for s in K.BOOKS}
        if "gap" in f:
            same = f["gap"] == 0
            o["same_line"] = K.mean_se(f[same]["profit"], f[same]["game_id"]); o["different_line"] = K.mean_se(f[~same]["profit"], f[~same]["game_id"])
        p = 1 / (1 + (1 / (f["profit"].mean() + 1) - 1)) if False else None
        wr, d = float((w == 1).mean()), float(K.dec(f["price"]).mean())
        o["three_leg_from_average_leg"] = {"no_boost": round(wr ** 3 * d ** 3 - 1, 4), "boost_25pct": round(wr ** 3 * (1 + 1.25 * (d ** 3 - 1)) - 1, 4)}
    return o


def run_main(seasons, D, shape, leans):
    res = {}
    for kind in ("flag", "close"):
        x = pd.concat([D[(s, kind)] for s in seasons]); close = pd.concat([D[(s, "close")] for s in seasons])
        for name in ("X1", "X2", "X0"):
            parts = []
            for s in seasons:
                ln = None if name == "X1" else leans.get(s)
                if name != "X1" and ln is None:
                    continue
                parts.append(pick(candidates_main(D[(s, kind)], shape, ln, own=(name == "X0"))))
            f = pd.concat(parts) if parts else pd.DataFrame()
            if kind == "flag" and len(f):
                f = clv(f, close, shape)
            res[f"{name}_{kind}"] = summ(f)
            if name == "X2" and kind == "flag" and len(f):
                x0 = pd.concat([pick(candidates_main(D[(s, kind)], shape, leans[s], own=True)) for s in seasons if leans.get(s)])
                k = f.merge(x0[["game_id", "nname", "stat", "side"]], on=["game_id", "nname", "stat", "side"])
                res["X2_flag"]["also_an_X0_bet"] = int(len(k))
        res[f"pairs_{kind}"] = int(len(x.merge(x, on=["game_id", "nname", "stat"]).query("book_x < book_y")))
        pr = x.merge(x, on=["game_id", "nname", "stat"]).query("book_x < book_y")
        res[f"pairs_different_line_{kind}"] = int((pr["line_x"] != pr["line_y"]).sum())
    return res


if __name__ == "__main__":
    mode = sys.argv[2]
    D = {(s, k): K.with_results(K.mains(s, k), s) for s in (2023, 2024, 2025) for k in ("flag", "close")}
    if mode == "freeze":
        fitx = pd.concat([D[(2023, "flag")], D[(2024, "flag")]])
        shape = fit_spread(fitx)
        leans = {2024: lean(D[(2023, "flag")]), 2025: lean(fitx)}
        res = {"shape": shape, "leans": leans, "tuning_main_2023_24": run_main([2023, 2024], D, shape, leans),
               "tuning_main_2024_only_X1": run_main([2024], D, shape, {})}
        FROZEN.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    else:
        fr = json.loads(FROZEN.read_text()); shape = fr["shape"]; leans = {int(k): v for k, v in fr["leans"].items()}
        res = {"holdout_main_2025": run_main([2025], D, shape, leans)}
        # Part R: 2025 rungs (tuning)
        alt = K.with_results(K.alts(2025, "flag"), 2025)
        R = {}
        for name, ln in (("X1", None), ("X2", leans[2025])):
            f = clv(pick(candidates_rungs(alt, D[(2025, "flag")], shape, ln)), D[(2025, "close")], shape)
            R[name] = summ(f)
            R[name]["below_other_main"] = summ(f[f["gap"] < 0], cuts=False); R[name]["above_other_main"] = summ(f[f["gap"] >= 0], cuts=False)
            R[name]["price_minus200_or_shorter"] = summ(f[K.dec(f["price"]) <= 1.5], cuts=False)
        R["rungs_with_other_books_main"] = int(len(candidates_rungs(alt, D[(2025, "flag")], shape, None)))
        res["tuning_rungs_2025"] = R
        (OUT / "cross_book_props_test.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
