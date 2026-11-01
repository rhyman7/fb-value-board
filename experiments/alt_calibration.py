"""Are the moved lines the Parlays page prices calibrated? Spreads and totals, NFL and college.

    python experiments/alt_calibration.py

The Parlays page prices every spread within 7 points of the market and every total within 7,
read off the key-number curve anchored to the market's line (alts.py). derivs_test.py checked
the curve for team totals and halves; this checks it for the full-game moved lines themselves.
Walk-forward: each season's curve fitted on earlier seasons (bets.fit_distribution), centred on
the closing line (no model), every played game scored at the line moved 1 to 7 points each
way in half points. A curve is calibrated when the share of covers in each predicted band
matches the band; pushes on whole numbers are left out of the share (as the books drop the leg).

Writes experiments/out/alt_calibration.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from nflmodel import bets as B, data as ND, distributions as D  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "alt_calibration.json"
SHIFTS = np.arange(-7.0, 7.5, 0.5)
BANDS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def rows(league):
    if league == "nfl":
        games = ND.load_games()
        MG, TT, snap = D.MARGINS, D.TOTALS, None
        fit = lambda s: B.fit_distribution(games, s)
        seasons = range(2014, int(games.season.max()) + 1)
    else:
        from cfbmodel import bets as CB, data as CD
        games, _, _ = CD.load_all(verbose=False)
        games = games[games.fbs_game == 1]
        MG, TT, snap = CB.MARGINS, CB.TOTALS, CB.half
        fit = lambda s: CB.fit_distribution(games, s)
        seasons = range(2017, int(games.season.max()) + 1)
    out = []
    for s in seasons:
        g = games[games.played & games.spread_line.notna() & games.total_line.notna() & (games.season == s)].reset_index(drop=True)
        if g.empty:
            continue
        cv = B.curves(g, fit(s), "line", MG, TT, snap)
        margin = (g.home_score - g.away_score).to_numpy(float)
        total = (g.home_score + g.away_score).to_numpy(float)
        sp, to = cv["spread"], cv["total"]
        for k in SHIFTS:
            Ls = sp + k
            h, push, a = D.over_push_under(cv["pm"], cv["margins"], Ls)
            for i in range(len(g)):
                if margin[i] == Ls[i]:
                    continue
                p = h[i] / (1 - push[i]) if push[i] < 1 else np.nan
                out.append(("spread", s, float(k), float(Ls[i]), float(p), float(margin[i] > Ls[i])))
            Lt = to + k
            o, pt, u = D.over_push_under(cv["pt"], cv["totals"], Lt)
            for i in range(len(g)):
                if total[i] == Lt[i]:
                    continue
                p = o[i] / (1 - pt[i]) if pt[i] < 1 else np.nan
                out.append(("total", s, float(k), float(Lt[i]), float(p), float(total[i] > Lt[i])))
    return pd.DataFrame(out, columns=["market", "season", "shift", "line", "p", "hit"]).dropna()


def report(df):
    rep = {}
    for mk, x in df.groupby("market"):
        b = pd.cut(x.p, BANDS, include_lowest=True)
        cal = x.groupby(b, observed=True).agg(n=("hit", "size"), said=("p", "mean"), happened=("hit", "mean")).reset_index()
        cal["gap"] = cal.happened - cal.said
        cal["se"] = np.sqrt(cal.happened * (1 - cal.happened) / cal.n)
        by_shift = x.assign(a=x["shift"].abs()).groupby("a").apply(
            lambda y: pd.Series({"n": len(y), "said": y.p.mean(), "happened": y.hit.mean(), "brier": ((y.p - y.hit) ** 2).mean()})).reset_index()
        rep[mk] = {"bands": [{"band": str(r["p"]), "n": int(r.n), "said": round(r.said, 4), "happened": round(r.happened, 4),
                              "gap": round(r.gap, 4), "se": round(r.se, 4)} for _, r in cal.iterrows()],
                   "by_abs_shift": [{"shift": float(r.a), "n": int(r.n), "said": round(r.said, 4), "happened": round(r.happened, 4),
                                     "brier": round(r.brier, 5)} for _, r in by_shift.iterrows()],
                   "brier": round(float(((x.p - x.hit) ** 2).mean()), 5)}
    return rep


def main():
    rep = {}
    for lg in ("nfl", "cfb"):
        df = rows(lg)
        rep[lg] = report(df)
        print(f"== {lg}: {len(df)} lines")
        for mk, r in rep[lg].items():
            print(f"  {mk}: Brier {r['brier']}")
            for b in r["bands"]:
                print(f"    {b['band']:>14} n={b['n']:>6}  said {b['said']:.3f}  happened {b['happened']:.3f}  gap {b['gap']:+.3f} ± {b['se']:.3f}")
    OUT.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
