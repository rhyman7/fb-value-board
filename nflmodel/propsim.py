"""Same-game simulation: player outcomes that move together with the game.

A quarterback's passing yards and his receivers' yards rise and fall together, and all
of them with how many points the team scores. Single props don't need this; any
combination of props from one game does.

The structure is a Gaussian copula with two factors per team:

  S   team scoring: (points - points implied by the board's projected margin and
      total) / its spread. The two teams' S are drawn together, with the correlation
      seen in past games.
  P   the passing game beyond the scoreboard: the starting quarterback's passing-yards
      residual with S taken out.

Each player-stat has loadings (l_S, l_P), fitted on the tuning seasons. In a simulated
game a player's normal score is z = l_S S + l_P P + sqrt(1 - l_S^2 - l_P^2) e, and his
outcome is his own distribution's quantile at Phi(z). Every player's single-prop
distribution is therefore exactly the one props.PropModel gives: the simulation changes
the joint outcomes and nothing else (checked in evaluate()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats

from . import config as C, propdist as D, props as P

CORR_STATS = {   # the player-stats the copula covers, by position group
    "pass_yds": ("QB",), "pass_td": ("QB",), "completions": ("QB",),
    "rec_yds": ("WR", "TE", "RB"), "receptions": ("WR", "TE", "RB"),
    "rush_yds": ("RB", "QB"), "carries": ("RB",), "anytime_td": ("WR", "TE", "RB"),
}


# --------------------------------------------------------------------------
# Normal scores of past outcomes
# --------------------------------------------------------------------------

def pit_z(model: P.PropModel, d: pd.DataFrame, stat: str, g: str, rng) -> np.ndarray:
    """Randomized probability integral transform of the outcome, as a normal score."""
    f = model.cdf_fn(d, stat, g)
    y = np.maximum(np.round(P.actual(d, stat)), 0)
    hi = f(y)[:, 0]
    lo = f(y - 1)[:, 0]
    u = lo + rng.uniform(size=len(y)) * (hi - lo)
    return stats.norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))


def team_points(games_path=None) -> pd.DataFrame:
    path = games_path or (C.RAW / "games.csv")
    g = pd.read_csv(path, usecols=["game_id", "home_team", "away_team", "home_score", "away_score"])
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(C.TEAM_ALIASES)
    h = g.rename(columns={"home_team": "team", "home_score": "pts"})[["game_id", "team", "pts"]]
    a = g.rename(columns={"away_team": "team", "away_score": "pts"})[["game_id", "team", "pts"]]
    return pd.concat([h, a], ignore_index=True)


def scores_table(proj: pd.DataFrame, model: P.PropModel, seed: int = 0) -> pd.DataFrame:
    """Long table of player-stat normal scores with their team's S and P factors."""
    rng = np.random.default_rng(seed)
    parts = []
    for stat, groups in CORR_STATS.items():
        for g in groups:
            if (stat, g) not in model.counts and (stat, g) not in model.yards:
                continue
            d = proj[P.eligible(proj, stat, g)].reset_index(drop=True)
            if d.empty:
                continue
            parts.append(pd.DataFrame({"game_id": d["game_id"], "team": d["team"], "opp": d["opp"], "season": d["season"],
                                       "player_id": d["player_id"], "stat": stat, "group": g,
                                       "z": pit_z(model, d, stat, g, rng), "imp_pts": d["imp_pts"].to_numpy()}))
    z = pd.concat(parts, ignore_index=True)
    tp = team_points()
    z = z.merge(tp, on=["game_id", "team"], how="left")
    z = z[z["pts"].notna()]
    return z


def fit_loadings(z: pd.DataFrame) -> dict:
    """S spread and cross-team correlation, then loadings per player-stat by least squares."""
    team = z.drop_duplicates(["game_id", "team"])[["game_id", "team", "opp", "pts", "imp_pts"]].copy()
    resid = team["pts"] - team["imp_pts"]
    sd = float(resid.std())
    team["S"] = (resid - resid.mean()) / sd
    opp = team[["game_id", "team", "S"]].rename(columns={"team": "opp", "S": "S_opp"})
    tt = team.merge(opp, on=["game_id", "opp"], how="inner")
    r_cross = float(np.corrcoef(tt["S"], tt["S_opp"])[0, 1])
    qb = z[(z.stat == "pass_yds")][["game_id", "team", "z"]].drop_duplicates(["game_id", "team"]).rename(columns={"z": "zqb"})
    team = team.merge(qb, on=["game_id", "team"], how="left")
    rho_qb = float(np.corrcoef(team.dropna()["S"], team.dropna()["zqb"])[0, 1])
    team["P"] = (team["zqb"] - rho_qb * team["S"]) / np.sqrt(1 - rho_qb ** 2)
    x = z.merge(team[["game_id", "team", "S", "P"]], on=["game_id", "team"], how="left").dropna(subset=["S"])
    load = {}
    for (stat, g), d in x.groupby(["stat", "group"]):
        if stat == "pass_yds":
            load[f"{stat}|{g}"] = [round(rho_qb, 4), round(float(np.sqrt(1 - rho_qb ** 2)), 4), int(len(d))]
            continue
        dd = d.dropna(subset=["P"])
        X = dd[["S", "P"]].to_numpy()
        b = np.linalg.lstsq(X, dd["z"].to_numpy(), rcond=None)[0]
        norm = np.hypot(*b)
        if norm > 0.95:
            b = b * 0.95 / norm
        load[f"{stat}|{g}"] = [round(float(b[0]), 4), round(float(b[1]), 4), int(len(dd))]
    return {"pts_sd": round(sd, 3), "cross_team_S": round(r_cross, 4), "qb_S": round(rho_qb, 4), "loadings": load}


# --------------------------------------------------------------------------
# Joint probabilities
# --------------------------------------------------------------------------

def bvn_upper(a, b, rho):
    """P(Z1 > a, Z2 > b) for standard normals with correlation rho (Owen's T)."""
    h, k = -np.asarray(a, float), -np.asarray(b, float)
    rho = np.broadcast_to(np.asarray(rho, float), h.shape)
    s = np.sqrt(np.maximum(1 - rho ** 2, 1e-12))
    hs = np.where(np.abs(h) < 1e-9, 1e-9, h)
    ks = np.where(np.abs(k) < 1e-9, 1e-9, k)
    ah = (k - rho * hs) / (hs * s)
    ak = (h - rho * ks) / (ks * s)
    c = np.where((hs * ks > 0) | ((hs * ks == 0) & (hs + ks >= 0)), 0.0, 0.5)
    phi2 = 0.5 * stats.norm.cdf(h) + 0.5 * stats.norm.cdf(k) - special.owens_t(hs, ah) - special.owens_t(ks, ak) - c
    return np.clip(phi2, 0, 1)


def pair_corr(load: dict, key1: str, key2: str, same_team: bool, cross_s: float) -> float:
    l1, l2 = load.get(key1), load.get(key2)
    if l1 is None or l2 is None:
        return 0.0
    if same_team:
        return l1[0] * l2[0] + l1[1] * l2[1]
    return l1[0] * l2[0] * cross_s


# --------------------------------------------------------------------------
# Simulation of one game (live use)
# --------------------------------------------------------------------------

def simulate(players: pd.DataFrame, model: P.PropModel, legs: list[tuple], n: int = 20000, seed: int = 0) -> np.ndarray:
    """Simulated outcomes for legs [(row index in players, stat), ...] of one game.
    Returns (n, len(legs)) integer outcomes. Uses model.meta['correlation']."""
    corr = model.meta.get("correlation") or {}
    load = corr.get("loadings", {})
    rng = np.random.default_rng(seed)
    teams = sorted(players["team"].unique())
    r = corr.get("cross_team_S", 0.0)
    Ssamp = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=n)
    S = {t: Ssamp[:, i % 2] for i, t in enumerate(teams)}
    Pf = {t: rng.standard_normal(n) for t in teams}
    out = np.zeros((n, len(legs)))
    for j, (i, stat) in enumerate(legs):
        row = players.loc[[i]]
        g = row["group"].iloc[0]
        lS, lP = (load.get(f"{stat}|{g}") or [0.0, 0.0])[:2]
        t = row["team"].iloc[0]
        z = lS * S[t] + lP * Pf[t] + np.sqrt(max(1 - lS ** 2 - lP ** 2, 0)) * rng.standard_normal(n)
        u = stats.norm.cdf(z)
        kmax = P.SPECS[stat].kmax if P.SPECS[stat].kind == "count" else 700
        grid = np.arange(kmax + 1, dtype=float)
        cdf = model.cdf_fn(row.reset_index(drop=True), stat, g)(grid[None, :])[0]
        out[:, j] = np.searchsorted(cdf, u)          # smallest k with F(k) >= u
    return out


def combo_prob(players, model, legs_with_lines, n: int = 20000, seed: int = 0) -> float:
    """Chance every leg wins, legs_with_lines = [(row index, stat, line, "over"|"under"), ...]."""
    sims = simulate(players, model, [(i, st) for i, st, _, _ in legs_with_lines], n, seed)
    ok = np.ones(n, bool)
    for j, (_, _, line, side) in enumerate(legs_with_lines):
        ok &= sims[:, j] > line if side == "over" else sims[:, j] < line
    return float(ok.mean())


# --------------------------------------------------------------------------
# Evaluation (experiments/props_calibration.py)
# --------------------------------------------------------------------------

PAIRS = [   # (stat, group, role in team) x (stat, group, role): same-game pairs scored
    ("QB pass yds & his top receiver's rec yds", ("pass_yds", "QB", 0), ("rec_yds", None, 1), True),
    ("QB pass yds & his lead back's rush yds", ("pass_yds", "QB", 0), ("rush_yds", "RB", 1), True),
    ("Two receivers' rec yds, same team", ("rec_yds", None, 1), ("rec_yds", None, 2), True),
    ("Both quarterbacks' pass yds", ("pass_yds", "QB", 0), ("pass_yds", "QB", 0), False),
    ("Top receiver's anytime TD & his QB's pass yds", ("anytime_td", None, 1), ("pass_yds", "QB", 0), True),
]


def evaluate(proj: pd.DataFrame, family: dict, tune: list[int], holdout_from: int) -> dict:
    """Fit loadings on the tuning seasons; on the holdout, score joint 'both over' events
    at the middle synthetic line, independent product against the copula; and check the
    simulation reproduces single-prop probabilities."""
    tr = proj[proj.season.isin(tune)]
    m_t = P.PropModel(family).fit(tr)
    zt = scores_table(tr, m_t)
    fit = fit_loadings(zt)
    out = {"loadings": fit, "pairs": []}

    ho = proj[proj.season >= holdout_from]
    m_h = P.PropModel(family).fit(proj[proj.season < holdout_from])
    m_h.meta["correlation"] = fit
    # each leg at its middle synthetic line
    legs = []
    for stat, groups in CORR_STATS.items():
        for g in groups:
            if (stat, g) not in m_h.counts and (stat, g) not in m_h.yards:
                continue
            d = ho[P.eligible(ho, stat, g)].reset_index(drop=True)
            if d.empty:
                continue
            L = P.test_lines(P.mean(d, stat), stat)
            mid = L[:, L.shape[1] // 2] if not P.SPECS[stat].fixed_lines else L[:, 0]
            over, _, _ = m_h.probs(d, stat, g, mid, calibrate=False)
            legs.append(pd.DataFrame({"game_id": d["game_id"], "team": d["team"], "player_id": d["player_id"], "stat": stat,
                                      "group": g, "mu": P.mean(d, stat), "p": over, "hit": (P.actual(d, stat) > mid).astype(float),
                                      "share": d["tgt_share"] if stat in ("rec_yds", "receptions", "anytime_td") else d["car_share"]}))
    legs = pd.concat(legs, ignore_index=True)
    legs["rank"] = 0
    nonqb = legs["group"] != "QB"
    legs.loc[nonqb, "rank"] = legs[nonqb].groupby(["game_id", "team", "stat"])["share"].rank(ascending=False, method="first").astype(int)
    load = fit["loadings"]
    for label, (s1, g1, r1), (s2, g2, r2), same in PAIRS:
        a = legs[(legs.stat == s1) & (legs["rank"] == r1) & ((legs.group == g1) if g1 else legs.group.isin(["WR", "TE"]))]
        b = legs[(legs.stat == s2) & (legs["rank"] == r2) & ((legs.group == g2) if g2 else legs.group.isin(["WR", "TE"]))]
        if same:
            j = a.merge(b, on=["game_id", "team"], suffixes=("_a", "_b"))
            j = j[j["player_id_a"] != j["player_id_b"]]
        else:
            j = a.merge(b, on="game_id", suffixes=("_a", "_b"))
            j = j[j["team_a"] < j["team_b"]]
        if j.empty:
            continue
        rho = np.array([pair_corr(load, f"{s1}|{ga}", f"{s2}|{gb}", same, fit["cross_team_S"]) for ga, gb in zip(j["group_a"], j["group_b"])])
        za, zb = stats.norm.ppf(1 - j["p_a"].clip(1e-4, 1 - 1e-4)), stats.norm.ppf(1 - j["p_b"].clip(1e-4, 1 - 1e-4))
        p_ind = j["p_a"] * j["p_b"]
        p_cop = bvn_upper(za, zb, rho)
        hit = j["hit_a"] * j["hit_b"]
        out["pairs"].append({"pair": label, "n": int(len(j)), "rho": round(float(rho.mean()), 3),
                             "pred_independent": round(float(p_ind.mean()), 4), "pred_copula": round(float(p_cop.mean()), 4),
                             "actual": round(float(hit.mean()), 4),
                             "brier_independent": round(D.brier(p_ind, hit), 5), "brier_copula": round(D.brier(p_cop, hit), 5),
                             "logloss_independent": round(D.log_loss(p_ind, hit), 5), "logloss_copula": round(D.log_loss(p_cop, hit), 5)})

    # single props through the simulation: same probabilities as the analytic ones
    rng = np.random.default_rng(1)
    chk = []
    for stat in ("pass_yds", "rec_yds", "rush_yds", "receptions", "anytime_td"):
        sub = legs[legs.stat == stat]
        sub = sub.iloc[rng.choice(len(sub), min(150, len(sub)), replace=False)]
        tab = ho.set_index(["game_id", "player_id"])
        for r in sub.itertuples():
            row = tab.loc[[(r.game_id, r.player_id)]].reset_index().iloc[[0]]
            L = P.test_lines(P.mean(row, stat), stat)
            line = L[0, L.shape[1] // 2] if not P.SPECS[stat].fixed_lines else L[0, 0]
            sims = simulate(row, m_h, [(0, stat)], n=4000, seed=int(rng.integers(1e9)))[:, 0]
            chk.append({"stat": stat, "analytic": float(r.p), "simulated": float((sims > line).mean()), "hit": r.hit})
    chk = pd.DataFrame(chk)
    out["single_prop_check"] = {
        "n": int(len(chk)), "mean_abs_diff": round(float((chk["analytic"] - chk["simulated"]).abs().mean()), 4),
        "brier_analytic": round(D.brier(chk["analytic"], chk["hit"]), 5), "brier_simulated": round(D.brier(chk["simulated"], chk["hit"]), 5),
        "note": "4,000 simulated games per row; differences are Monte Carlo noise (about 0.006 at 50%)"}
    return out
