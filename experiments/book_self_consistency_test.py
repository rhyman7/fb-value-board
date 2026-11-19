"""The book arguing with itself: a quarterback's passing-yards line against his receivers' lines at the same book.

    python experiments/book_self_consistency_test.py .

Rule written before any run: Project doc claude/plan-book-self-consistency-test-2026-11-19.md.
R = sum of posted receiving lines / QB passing line (one QB line, 3+ receiver lines, same team, same book).
A: expected R = fit-season mean R by number of receivers posted (3,4,5,6,7+), per book.
B: expected R = c * s, s = those players' share of the team's receiving yards in its earlier games
   that season (3+ earlier games), c fitted on the fit seasons.
G = R - expected R. Indicated: G high -> receivers under, QB over; G low -> the reverse.
Flagged: G in the outer thirds (cuts from the fit seasons' G only).
Baseline: logit(book fair over) + intercept per market. Test: + slope * G. Walk-forward
(2024 from a 2023 fit; 2025 from a 2023-24 fit). Version chosen on 2024 (pooled Brier), then 2025 scored.
Price input kept only if better on Brier for receiving and passing on 2024 and 2025, 2025 by 1+ SE.
Strategy record only if flagged return (both markets) and closing line value are above zero on both.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize, special
sys.path.insert(0, str(Path(__file__).resolve().parent))
import book_consistency_common as K

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)


def table(season, kind):
    x = K.with_results(K.mains(season, kind), season)
    x = x[x["stat"].isin(["pass_yds", "rec_yds"])].copy()
    pg = K.results(season)
    # each team's earlier-game receiving yards by player
    team_games = pg.groupby(["posteam", "week"])["rec_yds"].sum().rename("team_rec").reset_index()
    rows = []
    for (gid, team, book), g in x.groupby(["game_id", "posteam", "book"]):
        q, r = g[g["stat"] == "pass_yds"], g[g["stat"] == "rec_yds"]
        if len(q) != 1 or len(r) < 3:
            continue
        wk = int(g["week"].iloc[0])
        Lq = float(q["line"].iloc[0])
        R = float(r["line"].sum()) / Lq
        prev = pg[(pg["posteam"] == team) & (pg["week"] < wk)]
        ng = prev["week"].nunique()
        s = float(prev[prev["player_id"].isin(r["player_id"])]["rec_yds"].sum() / prev["rec_yds"].sum()) if ng >= 3 and prev["rec_yds"].sum() > 0 else np.nan
        rows.append({"season": season, "game_id": gid, "team": team, "book": book, "n": len(r), "R": R, "s": s})
    t = pd.DataFrame(rows)
    props = x.merge(t, left_on=["season", "game_id", "posteam", "book"], right_on=["season", "game_id", "team", "book"], how="inner")
    return t, props


def expected(fit_t, t, version):
    if version == "A":
        nb = lambda d: d["n"].clip(upper=7)
        m = fit_t.groupby([fit_t["book"], nb(fit_t)])["R"].mean()
        return pd.Series([m.get((b, n), np.nan) for b, n in zip(t["book"], nb(t))], index=t.index)
    f = fit_t.dropna(subset=["s"])
    c = float((f["R"] * f["s"]).sum() / (f["s"] ** 2).sum())
    return c * t["s"]


def fit_logit(d, with_g):
    d = d[d["y"] != d["line"]]
    off, hit, g = special.logit(d["p_over"].clip(0.02, 0.98)), (d["y"] > d["line"]).astype(float).to_numpy(), d["G"].to_numpy()
    def nll(b):
        z = off + b[0] + (b[1] * g if with_g else 0)
        return -np.sum(hit * -np.logaddexp(0, -z) + (1 - hit) * -np.logaddexp(0, z))
    return optimize.minimize(nll, [0.0, 0.0] if with_g else [0.0], method="BFGS").x


def pred(d, b):
    z = special.logit(d["p_over"].clip(0.02, 0.98)) + b[0] + (b[1] * d["G"] if len(b) > 1 else 0)
    return special.expit(z)


def score(fit_seasons, test_season, version, T, P, PC):
    ft = pd.concat([T[s] for s in fit_seasons]); out = {}
    def addG(props, t_src):
        t = t_src.copy(); t["G"] = t["R"] - expected(ft, t, version)
        return props.merge(t[["season", "game_id", "team", "book", "G"]], on=["season", "game_id", "team", "book"]).dropna(subset=["G"])
    fitp = pd.concat([addG(P[s], T[s]) for s in fit_seasons]); testp = addG(P[test_season], T[test_season])
    tg = pd.concat([T[s].assign(G=T[s]["R"] - expected(ft, T[s], version)) for s in fit_seasons])["G"].dropna()
    lo, hi = np.quantile(tg, [1 / 3, 2 / 3])
    out["cuts"] = [round(float(lo), 4), round(float(hi), 4)]
    close = PC[test_season][["game_id", "nname", "stat", "book", "line", "p_over"]].rename(columns={"p_over": "p_close"})
    sse = 0.0; nn = 0
    for stat in ("rec_yds", "pass_yds"):
        f, t = fitp[fitp["stat"] == stat], testp[testp["stat"] == stat]
        t = t[t["y"] != t["line"]].copy()
        b0, b1 = fit_logit(f, False), fit_logit(f, True)
        hit = (t["y"] > t["line"]).astype(float)
        e0, e1 = (pred(t, b0) - hit) ** 2, (pred(t, b1) - hit) ** 2
        d = K.mean_se(e1 - e0, t["game_id"])
        out[stat] = {"props": len(t), "slope": round(float(b1[1]), 3), "intercept": round(float(b0[0]), 3),
                     "brier_base": round(float(e0.mean()), 5), "brier_test": round(float(e1.mean()), 5), "diff": d["mean"] if False else round(float((e1 - e0).mean()), 5), "se": round(d["se"], 5) if d["se"] == d["se"] else None,
                     "over_rate": round(float(hit.mean()), 4), "fair_over": round(float(t["p_over"].mean()), 4)}
        sse += float(e1.sum() - e0.sum()); nn += len(t)
        # money: indicated side on flagged gaps
        over_side = (t["G"] < lo) if stat == "rec_yds" else (t["G"] > hi)
        under_side = (t["G"] > hi) if stat == "rec_yds" else (t["G"] < lo)
        m = t[over_side | under_side].copy()
        m["side_over"] = over_side[m.index]
        price = np.where(m["side_over"], m["over"], m["under"]); win = np.where(m["side_over"], m["y"] > m["line"], m["y"] < m["line"])
        m["profit"] = np.where(win, K.dec(price) - 1, -1.0)
        m = m.merge(close, on=["game_id", "nname", "stat", "book", "line"], how="left")
        pc = np.where(m["side_over"], m["p_close"], 1 - m["p_close"])
        m["clv"] = pc * K.dec(np.where(m["side_over"], m["over"], m["under"])) - 1
        c = m.dropna(subset=["clv"])
        out[stat]["money"] = {"ret": K.mean_se(m["profit"], m["game_id"]), "clv_same_line": K.mean_se(c["clv"], c["game_id"]),
                              "overs": int(m["side_over"].sum()), "unders": int((~m["side_over"]).sum()),
                              "ret_overs": K.mean_se(m[m["side_over"]]["profit"], m[m["side_over"]]["game_id"]),
                              "ret_unders": K.mean_se(m[~m["side_over"]]["profit"], m[~m["side_over"]]["game_id"]),
                              "by_book": {bk: K.mean_se(m[m["book"] == bk]["profit"], m[m["book"] == bk]["game_id"]) for bk in K.BOOKS}}
        # control: every prop in qualifying team-games, same sides split, unflagged
        u = t[~(over_side | under_side)]
        out[stat]["unflagged_under_ret"] = K.mean_se(np.where(u["y"] < u["line"], K.dec(u["under"]) - 1, -1.0), u["game_id"])
        out[stat]["unflagged_over_ret"] = K.mean_se(np.where(u["y"] > u["line"], K.dec(u["over"]) - 1, -1.0), u["game_id"])
    out["pooled_diff"] = round(sse / nn, 6)
    return out


if __name__ == "__main__":
    T, P, PC = {}, {}, {}
    for s in (2023, 2024, 2025):
        T[s], P[s] = table(s, "flag")
        PC[s] = K.mains(s, "close")
    res = {"team_game_books": {s: int(len(T[s])) for s in T}, "with_share": {s: int(T[s]["s"].notna().sum()) for s in T},
           "R_mean": {s: round(float(T[s]["R"].mean()), 4) for s in T}}
    res["tuning_2024"] = {v: score([2023], 2024, v, T, P, PC) for v in ("A", "B")}
    chosen = min(("A", "B"), key=lambda v: res["tuning_2024"][v]["pooled_diff"])
    res["chosen"] = chosen
    (OUT / "book_self_consistency_frozen.json").write_text(json.dumps({"chosen": chosen, "tuning": res["tuning_2024"]}, indent=1))
    if "--holdout" in sys.argv:
        res["holdout_2025"] = score([2023, 2024], 2025, chosen, T, P, PC)
    (OUT / "book_self_consistency_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
