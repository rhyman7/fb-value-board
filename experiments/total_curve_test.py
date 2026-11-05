"""Does the margin curve's width (and key-number shape) depend on the game's total?

The board prices every spread side, alternate spread, teaser leg and derivative off one margin
curve per season (`bets.fit_distribution`): one set of key-number multipliers and one width for
every game, whatever its total. Bettors who price teasers and alternate lines condition on the
total: a 38-point game lands on 3 and 7 more often and spreads out less than a 54-point game.

Candidates (each fitted walk-forward on the ten seasons before the test season, closing lines,
exactly as production fits its curve):
  base   production: global multipliers, one width
  A      width scales with the closing total: sd_i = sd * (total_i / mean_total) ** beta
  B      A, plus key-number multipliers fitted separately in three total bands
         (under 41, 41-47.5, 48+), each shrunk toward the global ones
  C      A, plus width grows with the size of the spread: sd_i *= 1 + gamma * (|spread_i| - 3) / 10

Measures (log loss, lower is better, paired per game, one standard error by game):
  alts    results against the closing spread moved by +-0.5 ... +-6.5 (fit_scale's SHIFTS)
  teaser  both sides' 6-point teaser legs (pushes set aside)
  wong    Wong legs only: underdogs +1.5 to +2.5 teased to +7.5 to +8.5, favourites -7.5 to -8.5
          teased to -1.5 to -2.5
NFL: tuning seasons 2014-2018, holdout 2019-2025. College (--cfb; cfbfastR closing lines from 2014, eight
seasons of history as production, so the first test seasons use fewer): tuning 2017-2019, holdout 2020-2025,
bands under 48.5 / 48.5-58 / 58.5+. 2026 to date is reported, not judged.

Rule, written before running: a candidate is kept (CONVENTIONS.md "Model tests", the keep rule)
only if it is better than base on all three measures in both the tuning and the holdout seasons.
If more than one passes, the lowest holdout `alts` loss wins. Nothing here changes the board;
a pass is a proposal to Ryan.

python experiments/total_curve_test.py [--cfb]   (results in experiments/out/total_curve_test_<league>.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nflmodel import data as DATA            # noqa: E402
from nflmodel import distributions as D      # noqa: E402
from nflmodel import odds as O               # noqa: E402

LEAGUE = "cfb" if "--cfb" in sys.argv else "nfl"
if LEAGUE == "nfl":
    MG = D.MARGINS
    BANDS = [(-1, 40.75), (40.75, 47.75), (47.75, 200)]
    SD0, SD_BOUNDS, LOOKBACK = 11.8, (9.0, 16.0), 10
    FIRST, TUNE, HOLD = 2014, (2014, 2018), (2019, 2025)
else:
    from cfbmodel import bets as CB
    MG = CB.MARGINS
    BANDS = [(-1, 48.25), (48.25, 58.25), (58.25, 300)]
    SD0, SD_BOUNDS, LOOKBACK = 16.0, (10.0, 24.0), 8
    FIRST, TUNE, HOLD = 2017, (2017, 2019), (2020, 2025)
TEASE = 6.0


def pmf(loc, sd, mult):
    loc = np.atleast_1d(loc)[:, None]
    sd = np.atleast_1d(sd)[:, None]
    p = D._cdf((MG[None, :] + 0.5 - loc) / sd) - D._cdf((MG[None, :] - 0.5 - loc) / sd)
    p = p * mult
    return p / p.sum(axis=1, keepdims=True)


def anchor(line, target, sd, mult):
    w = 8.0 if LEAGUE == "nfl" else 12.0
    lo, hi = line - w, line + w
    for _ in range(14 if LEAGUE == "nfl" else 16):
        mid = (lo + hi) / 2
        over, push, _ = D.over_push_under(pmf(mid, sd, mult), MG, line)
        low = over / np.maximum(1 - push, 1e-9) < target
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return (lo + hi) / 2




def curve(g, model, fair=None):
    """Margin pmf per game under a fitted model dict."""
    s, t = g.spread_line.to_numpy(float), g.total_line.to_numpy(float)
    sd = model["sd"] * (t / model["t0"]) ** model.get("beta", 0.0)
    sd = sd * np.clip(1 + model.get("gamma", 0.0) * (np.abs(s) - 3) / 10, 0.5, None)
    if "band_mults" in model:
        m = np.vstack([model["band_mults"][_band(x)] for x in t])
    else:
        m = np.broadcast_to(model["mult"], (len(g), len(MG)))
    target = 0.5 if fair is None else fair
    loc = anchor(s, target, sd, m)
    return pmf(loc, sd, m)


def _band(t):
    for i, (lo, hi) in enumerate(BANDS):
        if lo < t <= hi:
            return i
    return len(BANDS) - 1


def losses(g, p):
    """Per-game log losses: alts (mean over shifts), teaser (mean over both legs), wong (nan if none)."""
    s, r = g.spread_line.to_numpy(float), g.result.to_numpy(float)
    alt = np.zeros(len(g))
    cnt = np.zeros(len(g))
    for k in D.SHIFTS:
        line = s + k
        over, push, _ = D.over_push_under(p, MG, line)
        pc = np.clip(over / np.maximum(1 - push, 1e-9), 1e-6, 1 - 1e-6)
        ok = np.abs(r - line) > 1e-9
        hit = r > line
        ll = -np.where(hit, np.log(pc), np.log(1 - pc))
        alt += np.where(ok, ll, 0)
        cnt += ok
    alt = alt / np.maximum(cnt, 1)
    tz = np.zeros(len(g)); tc = np.zeros(len(g))
    wz = np.zeros(len(g)); wc = np.zeros(len(g))
    for side in (1, -1):                       # 1 = home leg, -1 = away leg
        line = s - side * TEASE                # home needs result > s - 6; away needs result < s + 6
        over, push, under = D.over_push_under(p, MG, line)
        pc = (over if side == 1 else under) / np.maximum(1 - push, 1e-9)
        pc = np.clip(pc, 1e-6, 1 - 1e-6)
        ok = np.abs(r - line) > 1e-9
        hit = (r > line) if side == 1 else (r < line)
        ll = -np.where(hit, np.log(pc), np.log(1 - pc))
        tz += np.where(ok, ll, 0); tc += ok
        # this leg's spread from its own side: home leg is -s (favourite negative)
        own = -s if side == 1 else s
        wong = ((own >= 1.5) & (own <= 2.5)) | ((own <= -7.5) & (own >= -8.5))
        wz += np.where(ok & wong, ll, 0); wc += ok & wong
    return alt, tz / np.maximum(tc, 1), np.where(wc > 0, wz / np.maximum(wc, 1), np.nan)


def nll_alts(g, model):
    a, _, _ = losses(g, curve(g, model))
    return float(a.sum())


def fit(h):
    """Fit base, A, B, C on history h."""
    mm = D.fit_multipliers(h.spread_line, h.result, SD0, MG, fold=True, shrink=8)
    sd = D.fit_scale(h.spread_line, h.result, MG, mm, bounds=SD_BOUNDS)
    mm = D.fit_multipliers(h.spread_line, h.result, sd, MG, fold=True, shrink=8)
    t0 = float(h.total_line.mean())
    base = {"sd": sd, "t0": t0, "mult": mm}

    def best_sd(extra):
        f = lambda x: nll_alts(h, {**base, **extra, "sd": x})
        return float(minimize_scalar(f, bounds=SD_BOUNDS, method="bounded", options={"xatol": 0.05}).x)

    # A: grid on beta, sd refitted for each
    bestA = None
    for beta in np.arange(0.0, 1.21, 0.1 if LEAGUE == "nfl" else 0.2):
        sdb = best_sd({"beta": beta})
        v = nll_alts(h, {**base, "beta": beta, "sd": sdb})
        if bestA is None or v < bestA[0]:
            bestA = (v, beta, sdb)
    A = {**base, "beta": float(bestA[1]), "sd": bestA[2]}

    # B: A's widths, multipliers by total band, shrunk toward the global multipliers
    band_mults = []
    t = h.total_line.to_numpy(float)
    s = h.spread_line.to_numpy(float)
    sd_i = A["sd"] * (t / t0) ** A["beta"]
    for lo, hi in BANDS:
        k = (t > lo) & (t <= hi)
        exp = pmf(s[k], sd_i[k], np.ones(len(MG))).sum(axis=0)
        obs = np.zeros(len(MG))
        idx = np.clip(np.round(h.result.to_numpy(float)[k]).astype(int) - MG[0], 0, len(MG) - 1)
        np.add.at(obs, idx, 1.0)
        exp, obs = exp + exp[::-1], obs + obs[::-1]
        prior = 20.0                            # pseudo-games pulling each band toward the global shape
        band_mults.append((obs + prior * mm) / (exp + prior))
    B = {**A, "band_mults": band_mults}

    # C: A plus spread-size width
    bestC = None
    for gamma in np.arange(-0.2, 0.41, 0.1 if LEAGUE == "nfl" else 0.2):
        sdc = best_sd({"beta": A["beta"], "gamma": gamma})
        v = nll_alts(h, {**base, "beta": A["beta"], "gamma": gamma, "sd": sdc})
        if np.isfinite(v) and (bestC is None or v < bestC[0]):
            bestC = (v, gamma, sdc)
    C = {**base, "beta": A["beta"], "gamma": float(bestC[1]), "sd": bestC[2]}
    return {"base": base, "A": A, "B": B, "C": C}


def load():
    if LEAGUE == "nfl":
        g = DATA.load_games()
        return g[g.played & g.spread_line.notna() & g.total_line.notna()].reset_index(drop=True)
    import pandas as pd
    root = Path(__file__).resolve().parents[1] / "data" / "cfb" / "cache"
    g = pd.concat([pd.read_csv(f) for f in sorted(root.glob("games_20*.csv"))], ignore_index=True)
    g = g[g.home_score.notna() & g.spread_line.notna() & g.total_line.notna()].copy()
    g["result"] = g.home_score - g.away_score
    g["spread_line"], g["total_line"] = CB.half(g.spread_line), CB.half(g.total_line)
    return g.reset_index(drop=True)


def main():
    g = load()
    rows = {k: {"alts": [], "teaser": [], "wong": [], "season": []} for k in ("base", "A", "B", "C")}
    params = {}
    for season in range(FIRST, 2027):
        h = g[(g.season < season) & (g.season >= season - LOOKBACK)]
        te = g[g.season == season].reset_index(drop=True)
        if te.empty:
            continue
        models = fit(h)
        params[season] = {k: {kk: (float(vv) if np.isscalar(vv) else None) for kk, vv in m.items()
                              if kk in ("sd", "beta", "gamma", "t0")} for k, m in models.items()}
        for k, m in models.items():
            a, t, w = losses(te, curve(te, m))
            rows[k]["alts"] += list(a); rows[k]["teaser"] += list(t); rows[k]["wong"] += list(w)
            rows[k]["season"] += [season] * len(te)
        print(season, params[season]["A"], params[season]["C"].get("gamma"), flush=True)

    out = {"params": params, "results": {}}
    seas = np.array(rows["base"]["season"])
    periods = {"tuning": (seas >= TUNE[0]) & (seas <= TUNE[1]),
               "holdout": (seas >= HOLD[0]) & (seas <= HOLD[1]),
               "2026 to date": seas == 2026}
    for k in ("A", "B", "C"):
        out["results"][k] = {}
        for pname, mask in periods.items():
            res = {}
            for meas in ("alts", "teaser", "wong"):
                b = np.array(rows["base"][meas])[mask]; c = np.array(rows[k][meas])[mask]
                ok = ~np.isnan(b) & ~np.isnan(c)
                d = c[ok] - b[ok]
                res[meas] = {"base": round(float(b[ok].mean()), 5), "cand": round(float(c[ok].mean()), 5),
                             "diff": round(float(d.mean()), 5),
                             "se": round(float(d.std(ddof=1) / np.sqrt(ok.sum())), 5), "n": int(ok.sum())}
            out["results"][k][pname] = res
        tune, hold = out["results"][k]["tuning"], out["results"][k]["holdout"]
        out["results"][k]["passes"] = all(p[m]["diff"] < 0 for p in (tune, hold) for m in ("alts", "teaser", "wong"))
    Path(__file__).with_name("out").mkdir(exist_ok=True)
    Path(__file__).with_name("out").joinpath(f"total_curve_test_{LEAGUE}.json").write_text(json.dumps(out, indent=1))
    for k in ("A", "B", "C"):
        print(k, "passes" if out["results"][k]["passes"] else "fails")
        for pname in periods:
            print("  ", pname, {m: (v["diff"], v["se"]) for m, v in out["results"][k][pname].items()})


if __name__ == "__main__":
    main()
