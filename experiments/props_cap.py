"""How far should the model move a prop's chance off the books'? (the Parlays page's PULL_MAX)

    python experiments/props_cap.py

nflmodel/parlays.py prices an NFL prop leg as the books' vig-free chance moved toward the model
by the blend weight (props_blend.py) and at most PULL_MAX (3 points, a judgment call). This
fits that ceiling instead, on the same data and folds as props_blend.py:

    capped = book + clip(w x (ours - book), -cap, +cap)

  fold 1   fit on 2023, test on 2024 (yardage markets)
  fold 2   fit on 2023-24, test on 2025 (the fold that decides)

w is each market's least-squares weight on the training seasons (the pooled yardage weight for
markets with no training season), exactly as props_blend.py. For every cap on the grid:

  - Brier score on the training and test seasons, minus the book's, clustered by game
  - the sides it would make +EV at the pricing snapshot (1+ point over the book's vig-free
    chance, positive EV at the book's price, the live liquidity and role rules): record, flat
    return, and closing line value carried to the bet's line (the live log's measure)

Rule, set before running: the cap with the lowest pooled Brier on the training seasons is
chosen. It replaces 3 points if, on 2025, its Brier is no worse than 3 points' (within one
standard error of the difference counts as a tie, and a tie keeps the smaller cap, which claims
less). The bets are reported, not used to choose: at these sizes they cannot separate caps.

Writes experiments/out/props_cap.json.
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
import props_blend as PB  # noqa: E402
from nflmodel import propbackfill as B, props as P  # noqa: E402

warnings.filterwarnings("ignore")
CAPS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 1.0)
CURRENT = 0.03


def capped(x: pd.DataFrame, weights: dict, pooled: float, cap: float) -> pd.Series:
    w = x["stat"].map(weights).fillna(pooled)
    return x["mkt"] + np.clip(w * (x["ours"] - x["mkt"]), -cap, cap)


def brier(x: pd.DataFrame, p: pd.Series) -> dict:
    sq = lambda q: (q - x["hit"]) ** 2
    d = sq(p) - sq(x["mkt"])
    return {"n": int(len(x)), "brier": round(float(sq(p).mean()), 6), "minus_book": round(float(d.mean()), 6),
            "se": round(PB._se_by_game(d, x["game_id"]), 6)}


def as_capped(fl: pd.DataFrame, weights: dict, pooled: float, cap: float) -> pd.DataFrame:
    b = fl.copy()
    ours = b["our_over"] / (b["our_over"] + b["our_under"])
    w = b["stat"].map(weights).fillna(pooled)
    p = b["fair_over"] + np.clip(w * (ours - b["fair_over"]), -cap, cap)
    rest = 1 - b["our_push"]
    b["our_over"], b["our_under"] = p * rest, (1 - p) * rest
    return b


def main():
    from props_calibration import FAMILY, load_proj
    cache = PB.R.C.RAW / "props_blend_priced.pkl"
    if cache.exists():
        priced = pd.read_pickle(cache)
    else:
        priced = {}
    proj = load_proj(None)
    saved = P.PropModel.load()
    close_models = {}
    pre = None
    for season in PB.SEASONS:
        def fitted(table):
            m = P.PropModel(FAMILY).fit(table[(table.season < season) & (table.season >= 2013)])
            if saved is not None:
                m.iso = saved.iso
            return m
        close_models[season] = fitted(proj)
        if season in priced:
            continue
        pre = pre if pre is not None else PB.R.load_pregame()
        fl = PB.priced_season(pre, fitted(pre), B.load(season, "flag"), season)
        cl = PB.priced_season(proj, close_models[season], B.load(season, "close"), season)
        priced[season] = (fl, cl)
        print(f"{season}: {len(fl)} pricing-snapshot props, {len(cl)} at the close", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(priced, cache)

    report = {"rule": "lowest training Brier picks the cap; it replaces 0.03 if its 2025 Brier is no worse "
                      "(a difference within one SE is a tie, and a tie keeps the smaller cap)", "folds": {}}
    for name, (train, test) in PB.FOLDS.items():
        tr = PB.frame(pd.concat([priced[s][0] for s in train], ignore_index=True))
        tr = tr[tr["stat"] != "anytime_td"]
        weights = {st: PB.fit_w(g) for st, g in tr.groupby("stat")}
        pooled = PB.fit_w(tr[tr["stat"].isin(PB.YARDAGE)])
        fl, cl = priced[test]
        fl, cl = fl[fl["stat"] != "anytime_td"], cl[cl["stat"] != "anytime_td"]
        if name == "fold1":
            fl, cl = fl[fl["stat"].isin(PB.YARDAGE)], cl[cl["stat"].isin(PB.YARDAGE)]
        te = PB.frame(fl)
        rows = []
        for cap in CAPS:
            r = {"cap": cap, "train": brier(tr, capped(tr, weights, pooled, cap)), "test": brier(te, capped(te, weights, pooled, cap))}
            if cap > 0:
                s = PB.flagged(as_capped(fl, weights, pooled, cap), cl, 1.0, 0.01, close_models[test])
                r["bets"] = {k: s.get(k) for k in ("bets", "record", "return_flat", "return_se", "clv_carried", "clv_carried_se", "beat_close_carried") if k in s}
            rows.append(r)
        # the paired difference against the current cap, on the test season
        base = capped(te, weights, pooled, CURRENT)
        for r in rows:
            p = capped(te, weights, pooled, r["cap"])
            d = (p - te["hit"]) ** 2 - (base - te["hit"]) ** 2
            r["test_minus_current"] = round(float(d.mean()), 6)
            r["test_minus_current_se"] = round(PB._se_by_game(d, te["game_id"]), 6)
        best = min(rows, key=lambda r: r["train"]["brier"])
        report["folds"][name] = {"train": list(train), "test": test, "weights": {k: round(v, 3) for k, v in weights.items()},
                                 "pooled_w": round(pooled, 3), "rows": rows, "train_best_cap": best["cap"]}
        print(f"\n== {name}: train {list(train)} -> test {test}; weights {report['folds'][name]['weights']}, pooled {pooled:.3f}")
        print(f"{'cap':>6} {'train-book':>11} {'test-book':>11} {'±':>8} {'test-3pt':>10} {'±':>8}   bets  return       clv")
        for r in rows:
            b = r.get("bets") or {}
            print(f"{r['cap']:>6.3f} {r['train']['minus_book']:>11.5f} {r['test']['minus_book']:>11.5f} {r['test']['se']:>8.5f} "
                  f"{r['test_minus_current']:>10.5f} {r['test_minus_current_se']:>8.5f}   {b.get('bets', '-'):>4}  "
                  f"{b.get('return_flat', float('nan')) if b else float('nan'):>+.3f}±{b.get('return_se') or 0:.3f}  "
                  f"{(b.get('clv_carried') or float('nan')):>+.3f}±{(b.get('clv_carried_se') or 0):.3f}")
        print("training-best cap:", best["cap"])
    f2 = report["folds"]["fold2"]
    best = next(r for r in f2["rows"] if r["cap"] == f2["train_best_cap"])
    worse = best["test_minus_current"] > best["test_minus_current_se"]
    tie = abs(best["test_minus_current"]) <= best["test_minus_current_se"]
    keep = CURRENT if worse or (tie and best["cap"] > CURRENT) else best["cap"]
    report["verdict"] = {"train_best_cap": f2["train_best_cap"], "test_minus_current": best["test_minus_current"],
                         "se": best["test_minus_current_se"], "cap_to_use": keep}
    PB.OUT.mkdir(exist_ok=True)
    (PB.OUT / "props_cap.json").write_text(json.dumps(report, indent=1, default=float))
    print("\nverdict:", report["verdict"])


if __name__ == "__main__":
    main()
