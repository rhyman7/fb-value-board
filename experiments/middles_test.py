"""Middles between DraftKings and FanDuel: how often, and are they worth betting?

    python experiments/middles_test.py

When the two books hang different numbers on the same game, betting the better side at each wins
both bets if the result lands between the two lines (a "middle"), and otherwise wins one and loses
one, paying the vig. Ryan bets only DraftKings and FanDuel.

Spreads: book A has the home team -3, book B -4.5: home -3 at A and away +4.5 at B both win when
the home team wins by exactly 4; a result on 3 or 4.5 pushes one side. Totals: over the lower total,
under the higher one. Only gaps where the lines actually differ; each game counted once per
snapshot and market (the widest gap).

Data: the saved market backfill (data/market/history/<league>/<season>/, DraftKings, FanDuel and
Pinnacle, 2024-2025): the Monday-noon snapshot ("open") and the last snapshot before kickoff
("close"). Graded two ways, per $1 on each side:
  realized  what the pair actually returned, with a standard error clustered by game
  expected  the pair's value under a fair distribution of results: Pinnacle's no-vig line at the
            same snapshot as the centre, and how often past games landed on each margin / total
            around their closing line (NFL 2006-2023, college 2014-2023; integers, so key numbers
            like 3 and 7 count as often as they really occur)

Rule, set before running (Nov 2, 2026): a kind of middle is worth flagging only if both its
expected value and its realized return are above zero, the realized by two standard errors.
Writes experiments/out/middles_test.json.
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
sys.path.insert(0, str(HERE))
import pinnacle_grade as PG  # noqa: E402
from nflmodel import odds as O  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "middles_test.json"
BOOKS = ("draftkings", "fanduel")
SEASONS = (2024, 2025)
GAPS = (0.5, 1.0, 1.5, 2.0, 3.0)


def residuals(games: pd.DataFrame, last: int) -> dict:
    """Empirical frequency of (result - closing line) for spreads and totals, by integer offset,
    keyed on the closing line rounded to the half point (pooled across lines nearby)."""
    g = games[games.played & games.spread_line.notna() & games.total_line.notna() & (games.season <= last)]
    return {"spread": (g.result.to_numpy(float), g.spread_line.to_numpy(float)),
            "total": (g.total.to_numpy(float), g.total_line.to_numpy(float))}


def outcome_probs(ref, center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Possible results and their chances for a game whose fair line is `center`: past games whose
    closing line was within `width` of it, shifted so their line sits on `center` (integers kept)."""
    res, line = ref
    m = np.abs(line - center) <= width
    if m.sum() < 200:
        m = np.argsort(np.abs(line - center))[:200]
    shift = np.round(center - line[m])                  # keep results integer
    vals = res[m] + shift
    u, c = np.unique(vals, return_counts=True)
    return u, c / c.sum()


def pair_profit(kind: str, lo: float, hi: float, p_lo: float, p_hi: float, r: np.ndarray) -> np.ndarray:
    """Per $1 each side. spread: home at line lo (wins r > lo), away at hi (wins r < hi).
    total: over lo (wins r > lo), under hi (wins r < hi). lo < hi."""
    d1, d2 = O.american_to_decimal(p_lo), O.american_to_decimal(p_hi)
    a = np.where(r > lo, d1 - 1, np.where(r == lo, 0.0, -1.0))
    b = np.where(r < hi, d2 - 1, np.where(r == hi, 0.0, -1.0))
    return a + b


def fair_center(q: dict, kind: str) -> float | None:
    p = q.get("pinnacle")
    if not p:
        return None
    return p.get("spread_line") if kind == "spread" else p.get("total_line")


def middles(qs: dict, games: pd.DataFrame, ref: dict, league: str, snap: str) -> pd.DataFrame:
    res = games.set_index("game_id")
    rows = []
    for gid, q in qs.items():
        if gid not in res.index or not all(b in q for b in BOOKS):
            continue
        g = res.loc[gid]
        if not bool(g.played):
            continue
        dk, fd = q["draftkings"], q["fanduel"]
        for kind, lk, hk, ak in (("spread", "spread_line", "home_spread_odds", "away_spread_odds"),
                                 ("total", "total_line", "over_odds", "under_odds")):
            if dk.get(lk) is None or fd.get(lk) is None or dk[lk] == fd[lk]:
                continue
            # spread_line is home-favoured positive; the home side covers when result > spread_line
            lo_b, hi_b = (dk, fd) if dk[lk] < fd[lk] else (fd, dk)
            lo, hi = lo_b[lk], hi_b[lk]
            p_lo, p_hi = lo_b.get(hk), hi_b.get(ak)
            if p_lo is None or p_hi is None:
                continue
            actual = float(g.result if kind == "spread" else g.total)
            c = fair_center(q, kind)
            ev = np.nan
            if c is not None:
                vals, pr = outcome_probs(ref[kind], c, 1.0 if kind == "spread" else 1.5)
                ev = float((pr * pair_profit(kind, lo, hi, p_lo, p_hi, vals)).sum())
            rows.append({"league": league, "snapshot": snap, "game_id": gid, "season": int(g.season), "kind": kind,
                         "lo": lo, "hi": hi, "gap": hi - lo, "price_lo": p_lo, "price_hi": p_hi,
                         "lo_book": "DK" if lo_b is dk else "FD", "ev": ev,
                         "hit": int(lo < actual < hi),
                         "profit": float(pair_profit(kind, lo, hi, p_lo, p_hi, np.array([actual]))[0])})
    return pd.DataFrame(rows)


def summary(x: pd.DataFrame) -> dict:
    n = len(x)
    if not n:
        return {"n": 0}
    g = x.groupby("game_id").profit.sum()
    se = float(g.std(ddof=1) * np.sqrt(len(g)) / n) if len(g) > 1 else float("nan")
    return {"n": n, "games": int(x.game_id.nunique()), "hit_rate": round(float(x.hit.mean()), 4),
            "expected_per_pair": round(float(x.ev.mean()), 4), "realized_per_pair": round(float(x.profit.mean()), 4),
            "realized_se": round(se, 4), "avg_gap": round(float(x.gap.mean()), 2)}


def main():
    out, allx = {}, []
    for league in ("nfl", "cfb"):
        lg = PG.League(league)
        games = lg.games()
        games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True)
        ref = residuals(games, 2023)
        for s in SEASONS:
            names = lg.team_names(s)
            open_q, close_q = PG.quotes(league, s, games, names)
            for snap, qs in (("open", open_q), ("close", close_q)):
                x = middles(qs, games, ref, league, snap)
                allx.append(x)
                print(league, s, snap, "games quoted by both books:",
                      sum(1 for q in qs.values() if all(b in q for b in BOOKS)), "middles:", len(x), flush=True)
    x = pd.concat(allx, ignore_index=True)
    x.to_csv(HERE / "out" / "middles_test.csv", index=False)
    for league in ("nfl", "cfb"):
        for snap in ("open", "close"):
            for kind in ("spread", "total"):
                y = x[(x.league == league) & (x.snapshot == snap) & (x.kind == kind)]
                key = f"{league} {snap} {kind}"
                out[key] = {f"gap {g}+": summary(y[y.gap >= g]) for g in GAPS}
                if kind == "spread":
                    key3 = y[(y.lo < 3) & (y.hi > 3) | (y.lo < -3) & (y.hi > -3) | (y.lo < 7) & (y.hi > 7) | (y.lo < -7) & (y.hi > -7)]
                    out[key]["across 3 or 7"] = summary(key3)
    for k, v in out.items():
        print(k)
        for g, s in v.items():
            print("   ", g, s)
    OUT.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
