"""The WR1 / WR2+ matchup split, rechecked where it would be used: the New projections and the books.

    python experiments/wr1_split_recheck.py

experiments/wr1_matchup_test.py found, on the chain projections, that a defense's matchup rating
moves secondary receivers far more than the top one (and that a defense's effect on WR1s is not a
stable trait): applying the WR rating at a separate strength for WR1 and WR2+, plus each role's own
rating, lowered the receiving-yards miss by 0.059 ± 0.005 on the holdout. The board shows the New
projections for wide receivers' receptions and receiving yards (players_ml.ADOPT), so:

1. New projections. Built as experiments/dvp_ml_test.py builds them (walk-forward from 2019),
   defenses rated on those projections, coefficients fitted on 2019-2021, scored on 2022 on.
   Base: the board's single WR rating (nflmodel/dvp.py); candidate: the split.
2. Against the books. 2023-2025 DraftKings / FanDuel main lines for WR receiving yards and
   receptions (props_blend's priced frames). logit(chance) = logit(book's vig-free chance) + c x
   signal, c fitted on 2023-2024, scored on 2025 by Brier against the book. Signals: the board's
   WR rating, and the split's log adjustment (chain ratings and coefficients, fitted 2013-2018).

Rule, set before running (Nov 2, 2026): the split goes on the board only if (1) lowers the
average miss on 2022 on by more than 2 standard errors for that stat, and (2) its 2025 Brier is
no worse than the board's current WR rating against the books. If it fails (2) but passes (1) it
may still go in as a projection improvement, but only with Ryan's say-so, since it moves the
numbers shown. Writes experiments/out/wr1_split_recheck.json.
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
import wr1_matchup_test as W  # noqa: E402
from nflmodel import config as C, data, dvp, players_ml, roles  # noqa: E402
from props_calibration import load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "wr1_split_recheck.json"


def fit_split(d, rt, stat, fit_mask):
    """Coefficients: board base b, and per role (b_wr, b_role)."""
    col = stat + "_proj"
    y, x0 = d[stat].to_numpy(float), d[col].to_numpy(float)
    r_wr = W.attach(d, rt, "WR", stat)
    r_role = np.where(d.role == "WR1", W.attach(d, rt, "WR1", stat), W.attach(d, rt, "WR2", stat))
    xb = x0 * r_wr
    b = float((xb[fit_mask] * (y - x0)[fit_mask]).sum() / (xb[fit_mask] ** 2).sum())
    beta = {}
    for role in ("WR1", "WR2"):
        m = (d.role == role).to_numpy() & fit_mask
        X = np.column_stack([x0 * r_wr, x0 * r_role])
        beta[role] = np.linalg.lstsq(X[m], (y - x0)[m], rcond=None)[0]
    return b, beta, r_wr, r_role


def log_adjust(d, b, beta, r_wr, r_role):
    base = b * r_wr
    split = np.where(d.role == "WR1", beta["WR1"][0] * r_wr + beta["WR1"][1] * r_role,
                     beta["WR2"][0] * r_wr + beta["WR2"][1] * r_role)
    return base, split


def new_projections() -> dict:
    pall = W.label_roles(load_proj(None))
    games = data.load_games()
    R = roles.load_all(games, verbose=False)
    ml = players_ml.questionable(players_ml.walk_forward(players_ml.add_features(pall, R), first=C.HOLDOUT_FROM),
                                 first=C.FIRST_TEST_SEASON)
    ml = ml[(ml.season >= C.HOLDOUT_FROM) & (ml.group == "WR")]
    ml = W.label_roles(ml.drop(columns=["role"], errors="ignore"))
    rt = W.role_ratings(ml)
    out = {}
    played = ml.played.fillna(False).astype(bool) & (ml.active == 1)
    for stat in W.STATS:
        col = stat + "_proj"
        d = ml[played & (ml[col] > 0)].copy()
        fitm = d.season.between(C.HOLDOUT_FROM, 2021).to_numpy()
        test = (d.season >= 2022).to_numpy()
        b, beta, r_wr, r_role = fit_split(d, rt, stat, fitm)
        lb, ls = log_adjust(d, b, beta, r_wr, r_role)
        x0, y = d[col].to_numpy(float), d[stat].to_numpy(float)
        base, cand = x0 * np.exp(lb), x0 * np.exp(ls)
        diff = np.abs(cand - y) - np.abs(base - y)
        g = pd.Series(d.game_id.to_numpy())
        r = {"b_base": round(b, 3), "coefs": {k: [round(float(v), 3) for v in vv] for k, vv in beta.items()},
             "n": int(test.sum()), "base_mae": round(float(np.abs(base - y)[test].mean()), 3),
             "change": round(float(diff[test].mean()), 4),
             "se": round(float(PB._se_by_game(pd.Series(diff[test]), g[test].reset_index(drop=True))), 4)}
        for role in ("WR1", "WR2"):
            m = test & (d.role == role).to_numpy()
            r[f"change_{role}"] = round(float(diff[m].mean()), 4)
        r["passes"] = bool(r["change"] < -2 * r["se"])
        out[stat] = r
        print("New", stat, r, flush=True)
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_c(z0, s, y):
    c = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-(z0 + c * s)))
        g, h = float((s * (y - p)).sum()), float((s * s * p * (1 - p)).sum())
        if h <= 0:
            break
        c += g / h
    return c


def vs_books() -> dict:
    pall = W.label_roles(load_proj(None))
    rt = W.role_ratings(pall)
    priced = pd.read_pickle(C.RAW / "props_blend_priced.pkl")
    played = pall.played.fillna(False).astype(bool) & (pall.active == 1) & (pall.group == "WR")
    roles_ = pall[["game_id", "player_id", "role"]].dropna()
    out = {}
    for stat in W.STATS:
        col = stat + "_proj"
        d = pall[played & (pall[col] > 0)]
        b, beta, _, _ = fit_split(d, rt, stat, d.season.isin(W.TUNE).to_numpy())

        def frame(seasons):
            fl = pd.concat([priced[s][0] for s in seasons], ignore_index=True)
            fl = fl[(fl.stat == stat) & (fl.group == "WR")].drop(columns=["role"], errors="ignore")
            fl = fl.merge(roles_, on=["game_id", "player_id"], how="inner")
            x = PB.frame(fl)
            r_wr = W.attach(x, rt, "WR", stat)
            r_role = np.where(x.role == "WR1", W.attach(x, rt, "WR1", stat), W.attach(x, rt, "WR2", stat))
            lb, ls = log_adjust(x, b, beta, r_wr, r_role)
            return x.assign(s_base=lb, s_split=ls)

        tr, te = frame((2023, 2024)), frame((2025,))
        res = {"n_2025": int(len(te)), "games_2025": int(te.game_id.nunique())}
        sq = lambda p: (p - te.hit.to_numpy()) ** 2
        briers = {}
        for sig in ("s_base", "s_split"):
            c = fit_c(logit(tr.mkt.to_numpy()), tr[sig].to_numpy(), tr.hit.to_numpy())
            p = 1 / (1 + np.exp(-(logit(te.mkt.to_numpy()) + c * te[sig].to_numpy())))
            briers[sig] = sq(p)
            dd = pd.Series(briers[sig] - sq(te.mkt.to_numpy()))
            res[sig] = {"c": round(c, 3), "brier_minus_book": round(float(dd.mean()), 6),
                        "se": round(PB._se_by_game(dd, te.game_id.reset_index(drop=True)), 6)}
        dd = pd.Series(briers["s_split"] - briers["s_base"])
        res["split_minus_base"] = {"brier": round(float(dd.mean()), 6),
                                   "se": round(PB._se_by_game(dd, te.game_id.reset_index(drop=True)), 6)}
        res["no_worse_than_base"] = bool(res["split_minus_base"]["brier"] <= 0)
        out[stat] = res
        print("books", stat, res, flush=True)
    return out


def main():
    res = {"new_projections": new_projections(), "vs_books_2025": vs_books()}
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
