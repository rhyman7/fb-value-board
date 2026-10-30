"""Do the derivative curves match history? (October 2026)

    python experiments/derivs_test.py            # both leagues
    python experiments/derivs_test.py --league nfl

No closing derivative lines exist in the repo's history (the market log began on Oct 29,
2026 and The Odds API's history costs 10 credits a market), so the test is of the curves
themselves: walk-forward, season by season, the full-game curve is fitted on the seasons
before (bets.fit_distribution) and the half curve likewise (derivs.fit_half), each game is
priced from its closing full-game line and prices alone (bets.curves, source "line"), and
the derived chances are scored against what happened:

  Team totals   P(home points > L) for the implied team total L = floor((total + spread)/2)
                + 0.5 and the same line moved 3.5 and 7 points each way (both teams), from
                the joint pmf (derivs.team_pmf) and from the normal approximation with the
                pmfs' mean and variance (normal=True). Log loss, Brier and the hit rate by
                predicted-chance bucket; a curve that matches history has hit rates equal
                to its chances.
  First half    P(first-half margin > L) and P(first-half total > L) at the implied
                first-half lines (ratio x spread, the straight-line total fit) and at the
                same shifts, from the half curve and from a plain normal with the fit's own
                residual width (plain_sd) and no key numbers. Also how often first-half
                margins land on 0, 3, 7 and 10 against what each curve says.

A curve is good enough to price a book's derivative line when its hit rates sit within a
point or two of its chances across the buckets; the one with the lower log loss is the one
the boards should use. Results go in FEATURES.md ("Derivative markets").
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import bets as B, data as ND, derivs as DV, distributions as D  # noqa: E402

warnings.filterwarnings("ignore")
SHIFTS = (-7.0, -3.5, 0.0, 3.5, 7.0)
BINS = [0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0]


def _score(p, hit) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    hit = np.asarray(hit, bool)
    ll = -np.mean(np.where(hit, np.log(p), np.log(1 - p)))
    return {"n": int(len(p)), "logloss": round(float(ll), 4), "brier": round(float(np.mean((p - hit) ** 2)), 4)}


def _calib(p, hit) -> str:
    p, hit = np.asarray(p, float), np.asarray(hit, bool)
    cells = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 50:
            continue
        cells.append(f"{100 * p[m].mean():.0f}→{100 * hit[m].mean():.0f} ({m.sum()})")
    return "; ".join(cells)


def run(league: str) -> list[str]:
    if league == "nfl":
        games = ND.load_games()
        half = ND.load_half_scores(verbose=False)
        MG, TT, snap = D.MARGINS, D.TOTALS, None
        fit_full = lambda s: B.fit_distribution(games, s)
        fit_half = lambda s: DV.fit_half(games, half, s)
        curves = lambda df, dist: B.curves(df, dist, "line")
        seasons = range(2014, int(games.season.max()) + 1)
        g = games
    else:
        from cfbmodel import bets as CB, data as CD
        games, _, _ = CD.load_all(verbose=False)
        games = games[games.fbs_game == 1]
        half = CD.load_half_scores(verbose=False)
        MG, TT, snap = CB.MARGINS, CB.TOTALS, CB.half
        fit_full = lambda s: CB.fit_distribution(games, s)
        fit_half = lambda s: DV.fit_half(games, half, s, lookback=8, margins=MG, totals=TT, snap=snap, sd_bounds=((5.0, 16.0), (5.0, 16.0)))
        curves = lambda df, dist: B.curves(df, dist, "line", MG, TT, snap)
        seasons = range(2017, int(games.season.max()) + 1)
        g = games
    half = half.drop_duplicates("game_id")
    pts = DV.team_support(MG, TT)
    tt = {"joint": ([], []), "normal": ([], [])}          # (p, hit) at every implied line and shift
    hm = {"curve": ([], []), "plain": ([], [])}            # first-half margin
    ht = {"curve": ([], []), "plain": ([], [])}            # first-half total
    key_obs, key_exp = {k: 0 for k in (0, 3, 7, 10)}, {"curve": {k: 0.0 for k in (0, 3, 7, 10)}, "plain": {k: 0.0 for k in (0, 3, 7, 10)}}
    n_half = 0
    for s in seasons:
        played = g[g.played & g.spread_line.notna() & g.total_line.notna() & (g.season == s)].reset_index(drop=True)
        if played.empty:
            continue
        dist, hdist = fit_full(s), fit_half(s)
        cv = curves(played, dist)
        home_pts, away_pts = played.home_score.to_numpy(float), played.away_score.to_numpy(float)
        sp, to = cv["spread"], cv["total"]
        for i in range(len(played)):
            for side, actual, sign in (("home", home_pts[i], 1), ("away", away_pts[i], -1)):
                q = DV.team_pmf(cv["pm"][i], cv["pt"][i], MG, TT, side)
                qn = DV.team_pmf(cv["pm"][i], cv["pt"][i], MG, TT, side, normal=True)
                base = np.floor((to[i] + sign * sp[i]) / 2) + 0.5
                for k in SHIFTS:
                    L = base + k
                    if L < 0.5:
                        continue
                    tt["joint"][0].append(D.over_push_under(q[None, :], pts, L)[0][0]); tt["joint"][1].append(actual > L)
                    tt["normal"][0].append(D.over_push_under(qn[None, :], pts, L)[0][0]); tt["normal"][1].append(actual > L)
        if hdist is None:
            continue
        hh = played.merge(half, on="game_id")
        if hh.empty:
            continue
        hc = DV.half_curves(hh, dist, hdist, {}, "line", MG, TT, snap)
        m1, t1 = (hh.home_h1 - hh.away_h1).to_numpy(float), (hh.home_h1 + hh.away_h1).to_numpy(float)
        sd_m, sd_t = hdist["plain_sd"]
        n_half += len(hh)
        for i in range(len(hh)):
            mu_m, mu_t = hc["loc_m"][i], hc["loc_t"][i]
            for k in SHIFTS:
                Lm = np.floor(mu_m) + 0.5 + k
                hm["curve"][0].append(D.over_push_under(hc["pm"][i:i + 1], MG, Lm)[0][0]); hm["curve"][1].append(m1[i] > Lm)
                hm["plain"][0].append(1 - D._cdf((Lm - mu_m) / sd_m)); hm["plain"][1].append(m1[i] > Lm)
                Lt = np.floor(mu_t) + 0.5 + k
                if Lt < 0.5:
                    continue
                ht["curve"][0].append(D.over_push_under(hc["pt"][i:i + 1], TT, Lt)[0][0]); ht["curve"][1].append(t1[i] > Lt)
                ht["plain"][0].append(1 - D._cdf((Lt - mu_t) / sd_t)); ht["plain"][1].append(t1[i] > Lt)
            for k in key_obs:
                key_obs[k] += int(abs(m1[i]) == k)
                key_exp["curve"][k] += float(hc["pm"][i][MG == k][0] + (hc["pm"][i][MG == -k][0] if k else 0.0))
                key_exp["plain"][k] += float((D._cdf((k + 0.5 - mu_m) / sd_m) - D._cdf((k - 0.5 - mu_m) / sd_m)) + ((D._cdf((-k + 0.5 - mu_m) / sd_m) - D._cdf((-k - 0.5 - mu_m) / sd_m)) if k else 0.0))
    lines = [f"## {league.upper()}: seasons {seasons.start}-{seasons.stop - 1}, walk-forward", "",
             "| Curve | Lines scored | Log loss | Brier | Predicted → hit rate by bucket (n) |", "|---|---|---|---|---|"]
    for name, (p, hit) in tt.items():
        sc = _score(p, hit)
        lines.append(f"| Team total, {name} pmf | {sc['n']:,} | {sc['logloss']} | {sc['brier']} | {_calib(p, hit)} |")
    for label, store in (("First-half margin", hm), ("First-half total", ht)):
        for name, (p, hit) in store.items():
            if not p:
                continue
            sc = _score(p, hit)
            lines.append(f"| {label}, {'half curve' if name == 'curve' else 'plain normal'} | {sc['n']:,} | {sc['logloss']} | {sc['brier']} | {_calib(p, hit)} |")
    if n_half:
        lines += ["", f"First-half margins on key numbers ({n_half:,} games): observed vs the half curve vs a plain normal",
                  "| Margin | Observed | Half curve | Plain normal |", "|---|---|---|---|"]
        for k in key_obs:
            lines.append(f"| {k} | {100 * key_obs[k] / n_half:.1f}% | {100 * key_exp['curve'][k] / n_half:.1f}% | {100 * key_exp['plain'][k] / n_half:.1f}% |")
    if hdist:
        lines += ["", f"Latest half fit ({seasons.stop - 1}): margin ratio {hdist['ratio_margin']:.3f}, total = {hdist['total_fit'][0]:.2f} + {hdist['total_fit'][1]:.3f} x total, "
                  f"widths {hdist['sd_margin']:.2f} / {hdist['sd_total']:.2f} (plain {hdist['plain_sd'][0]:.2f} / {hdist['plain_sd'][1]:.2f}), {hdist['n']:,} games."]
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", choices=["nfl", "cfb"])
    a = p.parse_args(argv)
    for lg in ([a.league] if a.league else ["nfl", "cfb"]):
        print("\n".join(run(lg)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
