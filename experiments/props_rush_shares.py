"""Rushing yards with the books' split of the carries.

    python experiments/props_rush_shares.py --season 2025

The model's biggest rushing misses come from guessing how a team splits its carries after
injuries or a role change. Here the split comes from the books instead: for every running
back the books post a rushing-yards line for, the mean that makes our own distribution agree
with the book's vig-free price at that line is turned into carries (that mean divided by
the model's yards per carry for him). Each back's share is his carries over the team total
of those backs. The model still supplies the team's running-back carries (pace, game script,
opponent) and each back's yards per carry (his efficiency, the defense); only the split is
the books'. Backs with no line keep the model's carries. Quarterbacks are left alone.

Scored on real 2025 lines at the pricing snapshot (pregame projections) and the close.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import propbackfill as B, propdist as D, props as P  # noqa: E402
import props_real_lines as R  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"


def implied_mean(model, row: pd.DataFrame, line: float, fair_over: float) -> float:
    """The rushing-yards mean at which our distribution gives the book's chance of the over."""
    def f(mu):
        r = row.assign(rush_yds_proj=mu, carries_proj=mu / max(row["ypc_m"].iloc[0], 0.5))
        return model.probs(r, "rush_yds", "RB", np.array([line]), calibrate=False)[0][0] - fair_over
    lo, hi = 1.0, 250.0
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return optimize.brentq(f, lo, hi, xtol=0.05)


def book_split(x: pd.DataFrame, model, proj: pd.DataFrame) -> pd.DataFrame:
    """x: one row per back with a line (game, team, player, line, fair_over, projection columns).
    Returns x with carries and yards rebuilt from the books' split."""
    x = x.copy()
    x["ypc_m"] = (x["rush_yds_proj"] / x["carries_proj"].clip(lower=0.5)).clip(2.0, 8.0)
    x["mu_book"] = [implied_mean(model, x.iloc[[i]], L, p) for i, (L, p) in enumerate(zip(x["line"], x["fair_over"]))]
    x["car_book"] = x["mu_book"] / x["ypc_m"]
    out = []
    for (g, t), d in x.groupby(["game_id", "team"]):
        d = d.copy()
        team = proj[(proj["game_id"] == g) & (proj["team"] == t) & (proj["group"] == "RB")]
        lined = set(d["player_id"])
        unlined = team.loc[~team["player_id"].isin(lined), "carries_proj"].sum()
        rb_total = team["carries_proj"].sum()                 # the model's carries for all the team's backs
        pool = max(rb_total - unlined, 0.5)
        ok = d["car_book"].notna()
        if ok.sum() == 0:
            d["carries_new"] = d["carries_proj"]
        else:
            share = d["car_book"].where(ok) / d.loc[ok, "car_book"].sum()
            d["carries_new"] = np.where(ok, pool * share * (ok.sum() / len(d)) + 0, d["carries_proj"])
            if (~ok).any():      # a back whose price could not be matched keeps his own carries
                d.loc[ok, "carries_new"] = (pool - d.loc[~ok, "carries_proj"].sum()).clip(min=0.5) * share[ok]
        d["share_model"] = d["carries_proj"] / max(rb_total, 0.5)
        d["share_book"] = d["carries_new"] / max(rb_total, 0.5)
        d["rush_yds_new"] = d["carries_new"] * d["ypc_m"]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def score(x: pd.DataFrame, model) -> dict:
    """Ours (model), ours (books' split) and the books' chance, on the same lines."""
    x = x[x["y"] != x["line"]].reset_index(drop=True)
    hit = (x["y"] > x["line"]).astype(float).to_numpy()
    old = x["our_over"] / (x["our_over"] + x["our_under"])
    r = x.assign(rush_yds_proj=x["rush_yds_new"], carries_proj=x["carries_new"])
    o, _, u = model.probs(r, "rush_yds", "RB", x["line"].to_numpy(float))
    new = o / (o + u)
    mkt = x["fair_over"].to_numpy()
    def se(a, b):
        d = (a - hit) ** 2 - (b - hit) ** 2
        g = pd.Series(d).groupby(x["game_id"].to_numpy()).sum()
        return round(float(d.mean()), 5), round(float(g.std(ddof=1) * np.sqrt(len(g)) / len(d)), 5)
    d_old, s_old = se(old.to_numpy(), mkt)
    d_new, s_new = se(new, mkt)
    car = x.dropna(subset=["carries"]) if "carries" in x else x
    return {"n": int(len(x)), "games": int(x["game_id"].nunique()),
            "brier_model": round(D.brier(old, hit), 5), "brier_book_split": round(D.brier(new, hit), 5), "brier_books": round(D.brier(mkt, hit), 5),
            "model_minus_books": [d_old, s_old], "book_split_minus_books": [d_new, s_new],
            "carries_miss_model": round(float(np.mean(np.abs(x["carries_proj"] - x["carries"]))), 2),
            "carries_miss_book_split": round(float(np.mean(np.abs(x["carries_new"] - x["carries"]))), 2),
            "yards_miss_model": round(float(np.mean(np.abs(x["rush_yds_proj"] - x["y"]))), 2),
            "yards_miss_book_split": round(float(np.mean(np.abs(x["rush_yds_new"] - x["y"]))), 2),
            "_new": new, "_x": x}


def bets(s: dict) -> dict:
    """The live flag rules (edge 5+ points, positive EV, both books) on the books' split."""
    x, new = s["_x"], s["_new"]
    rows = []
    for side in ("over", "under"):
        p = new if side == "over" else 1 - new
        fair = x["fair_over"] if side == "over" else 1 - x["fair_over"]
        price = x[side]
        ev = p * (np.vectorize(lambda o: o / 100 if o > 0 else 100 / -o)(price)) - (1 - p)
        ok = (p - fair >= 0.05) & (ev > 0) & (x["books_market"] >= 2)
        won = (x["y"] > x["line"]) if side == "over" else (x["y"] < x["line"])
        prof = np.where(won, np.vectorize(lambda o: o / 100 if o > 0 else 100 / -o)(price), -1.0)
        rows.append(pd.DataFrame({"game_id": x["game_id"][ok], "profit": prof[ok]}))
    b = pd.concat(rows)
    if b.empty:
        return {"bets": 0}
    g = b.groupby("game_id")["profit"].sum()
    return {"bets": int(len(b)), "won": int((b["profit"] > 0).sum()), "return": round(float(b["profit"].mean()), 4),
            "return_se": round(float(g.std(ddof=1) * np.sqrt(len(g)) / len(b)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--proj", type=Path, default=None)
    args = ap.parse_args()
    gd = load_proj(args.proj)
    pre = R.load_pregame()
    saved = P.PropModel.load()
    rep = {}
    for name, table, kind in (("pricing", pre, "flag"), ("close", gd, "close")):
        m = P.PropModel(FAMILY).fit(table[(table.season < args.season) & (table.season >= 2013)])
        m.iso = saved.iso
        x = R.priced(table, m, B.load(args.season, kind), args.season)
        x = x[(x["stat"] == "rush_yds") & (x["group"] == "RB")]
        # one row per back: the line both books share, else DraftKings'
        x = x.sort_values(["books_market", "book"], ascending=[False, True]).drop_duplicates(["game_id", "player_id"])
        x = book_split(x, m, table[table.season == args.season])
        s = score(x, m)
        s["bets_book_split"] = bets(s)
        rep[name] = {k: v for k, v in s.items() if not k.startswith("_")}
        print(name, json.dumps(rep[name], indent=1))
    OUT.mkdir(exist_ok=True)
    (OUT / f"props_rush_shares_{args.season}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
