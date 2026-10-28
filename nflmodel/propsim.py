"""Same-game combinations: player outcomes that move together.

A quarterback's passing yards and his top receiver's yards go over together far more often
than two independent props would; a receiver's catches and his yards even more so. Single
props don't need any of this; any combination of props from one game does.

The structure is a Gaussian copula. Each leg (a player's stat) has its own distribution
from props.PropModel, so every single-prop probability is exactly the model's. What the
copula adds is one correlation per pair of legs, looked up by the kind of pair:

  self   two stats of the same player (receptions and receiving yards)
  team   two players on the same team, by role: QB (the projected starter), R1, R2, R3
         (wide receivers and tight ends by target share), RB1 (the lead back by carries)
  opp    the same roles on opposite teams (both quarterbacks' passing yards)

Each correlation is fitted on the tuning seasons so that the chance both legs go over,
summed over past pairs of that kind, matches how often both did. That targets the thing a
combination bet pays on. With fewer than MIN_PAIRS past pairs, a kind gets no correlation.
Correlations are shrunk toward zero by sample size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from . import propdist as D, props as P

LEG_STATS = {   # the stats a combination can use, by role
    "QB": ("pass_yds", "pass_td", "completions", "pass_att", "rush_yds"),
    "R": ("rec_yds", "receptions", "anytime_td"),
    "RB1": ("rush_yds", "carries", "rec_yds", "receptions", "anytime_td"),
}
MIN_PAIRS = 300
SHRINK = 300.0          # rho x n / (n + SHRINK)


# --------------------------------------------------------------------------
# Joint probabilities
# --------------------------------------------------------------------------

def bvn_upper(a, b, rho):
    """P(Z1 > a, Z2 > b) for standard normals with correlation rho (Owen's T)."""
    h, k = -np.asarray(a, float), -np.asarray(b, float)
    h, k = np.broadcast_arrays(h, k)
    rho = np.broadcast_to(np.asarray(rho, float), h.shape)
    s = np.sqrt(np.maximum(1 - rho ** 2, 1e-12))
    hs = np.where(np.abs(h) < 1e-9, 1e-9, h)
    ks = np.where(np.abs(k) < 1e-9, 1e-9, k)
    ah = (ks - rho * hs) / (hs * s)
    ak = (hs - rho * ks) / (ks * s)
    c = np.where(hs * ks > 0, 0.0, 0.5)
    phi2 = 0.5 * stats.norm.cdf(hs) + 0.5 * stats.norm.cdf(ks) - special.owens_t(hs, ah) - special.owens_t(ks, ak) - c
    return np.clip(phi2, 0, 1)


def joint_over(pa, pb, rho):
    """Chance both legs go over, given each one's chance and the pair's correlation."""
    za = stats.norm.ppf(1 - np.clip(pa, 1e-6, 1 - 1e-6))
    zb = stats.norm.ppf(1 - np.clip(pb, 1e-6, 1 - 1e-6))
    return bvn_upper(za, zb, rho)


# --------------------------------------------------------------------------
# Legs and their roles
# --------------------------------------------------------------------------

def roles(df: pd.DataFrame) -> pd.Series:
    """QB (projected starter), R1-R3 (WR/TE by target share), RB1 (top back by carries)."""
    r = pd.Series(pd.NA, index=df.index, dtype=object)
    qb = (df["group"] == "QB") & (df["att_share"] > 0.5)
    r[qb] = "QB"
    key = [df["game_id"], df["team"]]
    rec = df["group"].isin(["WR", "TE"]) & (df["active"] == 1)
    rk = df["tgt_share"].where(rec).groupby(key).rank(ascending=False, method="first")
    for i in (1, 2, 3):
        r[rec & (rk == i)] = f"R{i}"
    rb = (df["group"] == "RB") & (df["active"] == 1)
    rk = df["car_share"].where(rb).groupby(key).rank(ascending=False, method="first")
    r[rb & (rk == 1)] = "RB1"
    return r


def _stats_for(role):
    return LEG_STATS["R"] if str(role).startswith("R") and role != "RB1" else LEG_STATS.get(role, ())


def legs(df: pd.DataFrame, model: P.PropModel, played: bool = True) -> pd.DataFrame:
    """Every leg in these games at its middle synthetic line (the standard line for
    touchdowns): chance of the over and, for played games, whether it went over."""
    df = df.copy()
    df["role"] = roles(df)
    out = []
    for role in ("QB", "R1", "R2", "R3", "RB1"):
        for stat in _stats_for(role):
            g = {"QB": "QB", "RB1": "RB"}.get(role)
            sub = df[df["role"] == role]
            for grp in ([g] if g else ["WR", "TE"]):
                if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
                    continue
                d = sub[P.eligible(sub, stat, grp, played=played)].reset_index(drop=True)
                if d.empty:
                    continue
                L = P.test_lines(P.mean(d, stat), stat)
                mid = L[:, L.shape[1] // 2] if not P.SPECS[stat].fixed_lines else L[:, 0]
                p, _, _ = model.probs(d, stat, grp, mid)
                rec = {"game_id": d["game_id"].to_numpy(), "team": d["team"].to_numpy(), "player_id": d["player_id"].to_numpy(),
                       "season": d["season"].to_numpy() if "season" in d else 0, "role": role, "stat": stat, "group": grp,
                       "line": mid, "p": p}
                if played:
                    rec["hit"] = (P.actual(d, stat) > mid).astype(float)
                out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pair_key(kind, a_stat, a_role, b_stat, b_role):
    a, b = f"{a_stat}@{a_role}", f"{b_stat}@{b_role}"
    if kind == "self":
        a, b = a_stat, b_stat
    return "|".join([kind] + sorted([a, b]))


def pairs(lg: pd.DataFrame) -> pd.DataFrame:
    """All pairs of legs in the same game, labelled by kind."""
    x = lg.reset_index(drop=True).assign(i=lambda d: np.arange(len(d)))
    j = x.merge(x, on="game_id", suffixes=("_a", "_b"))
    j = j[j["i_a"] < j["i_b"]]
    self_ = j["player_id_a"] == j["player_id_b"]
    team = (j["team_a"] == j["team_b"]) & ~self_
    j["kind"] = np.where(self_, "self", np.where(team, "team", "opp"))
    j["key"] = [pair_key(k, sa, ra, sb, rb) for k, sa, ra, sb, rb in zip(j["kind"], j["stat_a"], j["role_a"], j["stat_b"], j["role_b"])]
    return j


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_rhos(pr: pd.DataFrame) -> dict:
    """One correlation per kind of pair, matching the chance both went over."""
    out = {}
    for key, g in pr.groupby("key"):
        n = len(g)
        if n < MIN_PAIRS:
            continue
        target = (g["hit_a"] * g["hit_b"]).mean()
        f = lambda r: joint_over(g["p_a"].to_numpy(), g["p_b"].to_numpy(), r).mean() - target
        lo, hi = f(-0.6), f(0.95)
        if lo > 0:
            rho = -0.6
        elif hi < 0:
            rho = 0.95
        else:
            rho = optimize.brentq(f, -0.6, 0.95, xtol=1e-4)
        out[key] = [round(float(rho * n / (n + SHRINK)), 4), int(n)]
    return out


def rho_for(rhos: dict, kind, a_stat, a_role, b_stat, b_role) -> float:
    v = rhos.get(pair_key(kind, a_stat, a_role, b_stat, b_role))
    return float(v[0]) if v else 0.0


# --------------------------------------------------------------------------
# Live use
# --------------------------------------------------------------------------

def nearest_corr(R: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so the matrix is a valid correlation matrix."""
    w, v = np.linalg.eigh((R + R.T) / 2)
    R2 = v @ np.diag(np.maximum(w, 1e-6)) @ v.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def combo_prob(players: pd.DataFrame, model: P.PropModel, legs_in: list[tuple], n: int = 40000, seed: int = 0) -> dict:
    """Chance every leg of a same-game combination wins.

    players: one game's board rows (with role columns from roles()). legs_in: [(row label,
    stat, line, "over"|"under")]. Returns the joint chance, the independent product and each
    leg's own chance."""
    rhos = (model.meta.get("correlation") or {}).get("pairs", {})
    k = len(legs_in)
    R = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            ia, sa = legs_in[a][0], legs_in[a][1]
            ib, sb = legs_in[b][0], legs_in[b][1]
            ra, rb = players.loc[ia], players.loc[ib]
            kind = "self" if ra["player_id"] == rb["player_id"] else ("team" if ra["team"] == rb["team"] else "opp")
            r = rho_for(rhos, kind, sa, ra["role"], sb, rb["role"])
            # an under leg is the mirror image: flip the sign of its correlations
            if (legs_in[a][3] == "under") != (legs_in[b][3] == "under"):
                r = -r
            R[a, b] = R[b, a] = r
    R = nearest_corr(R)
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(k), R, size=n)
    win = np.ones(n, bool)
    singles = []
    for j, (i, stat, line, side) in enumerate(legs_in):
        row = players.loc[[i]].reset_index(drop=True)
        over, push, under = model.probs(row, stat, row["group"].iloc[0], np.array([line]))
        p = float(over[0] if side == "over" else under[0])
        singles.append(p)
        win &= z[:, j] > stats.norm.ppf(1 - p)      # leg j wins in the top p of its own scale
    return {"joint": float(win.mean()), "independent": float(np.prod(singles)), "legs": singles}


# --------------------------------------------------------------------------
# Evaluation (experiments/props_calibration.py)
# --------------------------------------------------------------------------

SHOW = [   # pairs reported in the calibration report
    ("QB pass yds & R1 rec yds", "team|pass_yds@QB|rec_yds@R1"),
    ("QB pass yds & R2 rec yds", "team|pass_yds@QB|rec_yds@R2"),
    ("R1 & R2 rec yds", "team|rec_yds@R1|rec_yds@R2"),
    ("QB pass yds & RB1 rush yds", "team|pass_yds@QB|rush_yds@RB1"),
    ("Both QBs' pass yds", "opp|pass_yds@QB|pass_yds@QB"),
    ("QB pass TDs & R1 anytime TD", "team|anytime_td@R1|pass_td@QB"),
    ("Same player: receptions & rec yds", "self|rec_yds|receptions"),
    ("Same player: carries & rush yds", "self|carries|rush_yds"),
    ("Same player: rec yds & anytime TD", "self|anytime_td|rec_yds"),
    ("QB: completions & pass yds", "self|completions|pass_yds"),
]


def evaluate(proj: pd.DataFrame, family: dict, tune: list[int], holdout_from: int) -> dict:
    """Fit correlations on the tuning seasons; on the holdout, score 'both over' for each
    kind of pair (independent product against the copula) and a three-leg stack."""
    tr = proj[proj.season.isin(tune)]
    m_t = P.PropModel(family).fit(tr)
    lt = legs(tr, m_t)
    rhos = fit_rhos(pairs(lt))
    corr = {"pairs": rhos, "min_pairs": MIN_PAIRS, "shrink": SHRINK, "fitted_on": [tune[0], tune[-1]]}

    ho = proj[proj.season >= holdout_from]
    m_h = P.PropModel(family).fit(proj[proj.season < holdout_from])
    lh = legs(ho, m_h)
    ph = pairs(lh)
    ph["rho"] = [rhos.get(k, [0.0])[0] for k in ph["key"]]
    ph["p_ind"] = ph["p_a"] * ph["p_b"]
    ph["p_cop"] = joint_over(ph["p_a"].to_numpy(), ph["p_b"].to_numpy(), ph["rho"].to_numpy())
    ph["hit"] = ph["hit_a"] * ph["hit_b"]
    rows = []
    for label, key in SHOW:
        g = ph[ph["key"] == key]
        if g.empty:
            continue
        cl = ((g["p_cop"] - g["hit"]) ** 2 - (g["p_ind"] - g["hit"]) ** 2).groupby(g["game_id"]).sum()
        rows.append({"pair": label, "n": int(len(g)), "rho": round(float(g["rho"].iloc[0]), 3),
                     "pred_independent": round(float(g["p_ind"].mean()), 4), "pred_copula": round(float(g["p_cop"].mean()), 4),
                     "actual": round(float(g["hit"].mean()), 4),
                     "actual_se": round(float(np.sqrt(g["hit"].mean() * (1 - g["hit"].mean()) / len(g))), 4),
                     "brier_independent": round(D.brier(g["p_ind"], g["hit"]), 5), "brier_copula": round(D.brier(g["p_cop"], g["hit"]), 5),
                     "brier_diff_se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(g)), 5)})
    allp = ph[ph["rho"] != 0]
    overall = {"n": int(len(allp)), "brier_independent": round(D.brier(allp["p_ind"], allp["hit"]), 5),
               "brier_copula": round(D.brier(allp["p_cop"], allp["hit"]), 5),
               "log_loss_independent": round(D.log_loss(allp["p_ind"], allp["hit"]), 5),
               "log_loss_copula": round(D.log_loss(allp["p_cop"], allp["hit"]), 5)}

    # a three-leg stack: QB pass yds, R1 and R2 rec yds, all over
    trip = []
    q = lh[(lh.role == "QB") & (lh.stat == "pass_yds")]
    r1 = lh[(lh.role == "R1") & (lh.stat == "rec_yds")]
    r2 = lh[(lh.role == "R2") & (lh.stat == "rec_yds")]
    t = q.merge(r1, on=["game_id", "team"], suffixes=("", "_1")).merge(r2, on=["game_id", "team"], suffixes=("", "_2"))
    if len(t):
        R = np.eye(3)
        R[0, 1] = R[1, 0] = rhos.get("team|pass_yds@QB|rec_yds@R1", [0])[0]
        R[0, 2] = R[2, 0] = rhos.get("team|pass_yds@QB|rec_yds@R2", [0])[0]
        R[1, 2] = R[2, 1] = rhos.get("team|rec_yds@R1|rec_yds@R2", [0])[0]
        z = np.random.default_rng(0).multivariate_normal(np.zeros(3), nearest_corr(R), size=20000)
        P3 = np.column_stack([t["p"], t["p_1"], t["p_2"]])
        thr = stats.norm.ppf(1 - np.clip(P3, 1e-6, 1 - 1e-6))
        pc = np.array([np.mean((z > th).all(1)) for th in thr])
        hit = (t["hit"] * t["hit_1"] * t["hit_2"]).to_numpy()
        trip = {"n": int(len(t)), "pred_independent": round(float(P3.prod(1).mean()), 4), "pred_copula": round(float(pc.mean()), 4),
                "actual": round(float(hit.mean()), 4), "brier_independent": round(D.brier(P3.prod(1), hit), 5),
                "brier_copula": round(D.brier(pc, hit), 5)}

    # single props are untouched: the copula only joins legs, each keeps its own probability
    return {"correlation": corr, "pairs": rows, "all_pairs_with_a_fitted_rho": overall, "three_leg_stack": trip,
            "single_props": "unchanged by construction: each leg's chance is the model's own (see combo_prob)"}
