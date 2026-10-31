"""The alternate yardage ladders (50+, 60+, 70+ ...) against real DraftKings and FanDuel prices.

    python experiments/props_ladders.py --season 2025

Needs the backfilled ladder prices: the **Prop price backfill** Action with markets
`player_pass_yds_alternate,player_rush_yds_alternate,player_reception_yds_alternate`
(data/props/history/<season>/, the same pricing and closing snapshots as the main lines).
The books post each rung Over only ("70+" is Over 69.5), often at different rungs.

Priced exactly as experiments/props_real_lines.py prices the main lines: pregame projections
at the pricing snapshot, game-day projections at the close, distribution widths fitted only on
seasons before this one. A rung's book chance is its price with the live code's assumed
margin taken out (propmarket.ALT_MARGIN, 10%, power method); the raw implied chance is kept
too, so the margin the books really charge on rungs can be read off the results.

Reported, at the pricing snapshot:

1. Calibration. For the books (raw and margin-free) and the model, predicted against how
   often the rung cleared, by band, and Brier scores on the same rungs. The bands above 75%
   answer "does the model know a player will very likely clear a low number".
2. Model bets. Rungs where the model's chance gives an EV of EDGE or more at the posted
   price (one per player and stat: the best), inside and outside the live price band
   (propmarket.MAX_ODDS), graded on results.
3. Book against book (no model). A rung both books post: the better price, when its EV
   against the other book's margin-free chance is EDGE or more.
4. Ladder gap (the live tag, propmarket.market_tags): a rung whose margin-free chance is more
   than LADDER_GAP below what the same book's main line implies through the model's shape.

Every bet carries closing line value two ways: price CLV (decimal price taken over the same
book's closing price for the same rung, minus 1: did we beat the close) and EV at the closing
margin-free chance. Standard errors are clustered by game.

The rule, set before the data exists: a kind of bet earns a place on the board only if, on a
season it was not fitted on, both its return and its price CLV are above zero by more than two
standard errors. Nothing here is fitted, so 2025 counts as out of sample.

Writes experiments/out/props_ladders_<season>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import config as C, odds as O, propbackfill as B, propdist as D, propmarket as PM, props as P  # noqa: E402
import props_real_lines as R  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out"
LADDERS = ["player_pass_yds_alternate", "player_rush_yds_alternate", "player_reception_yds_alternate"]
EDGES = (0.02, 0.05)
BANDS = [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 1.0001]


def _se(v, groups) -> float:
    v = pd.Series(np.asarray(v, float))
    if len(v) < 2:
        return float("nan")
    g = v.groupby(np.asarray(groups)).sum()
    return float(g.std(ddof=1) * np.sqrt(len(g)) / len(v))


def priced_rungs(proj: pd.DataFrame, model, snaps: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per game x player x ladder rung x book: price, book chances, ours, result."""
    s = snaps[snaps["market"].isin(LADDERS) & (snaps["side"].str.lower() == "over")].copy()
    if s.empty:
        return pd.DataFrame()
    info = pd.read_csv(C.CACHE / f"player_info_{season}.csv", dtype={"player_id": str})
    rows = proj[(proj["season"] == season) & proj["played"].astype(bool)].merge(info[["player_id", "name"]], on="player_id", how="left")
    rows["nname"] = rows["name"].map(PM.norm_name)
    rows = rows[~rows.duplicated(["game_id", "nname"], keep=False)]
    s["nname"] = s["player"].map(PM.norm_name)
    s = s.sort_values("snapshot").drop_duplicates(["game_id", "nname", "market", "line", "book"], keep="last")
    s = s[["game_id", "nname", "market", "line", "book", "price"]].merge(rows, on=["game_id", "nname"], how="inner")
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    out = []
    for (stat, grp), g in s.groupby(["stat", "group"]):
        if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
            continue
        g = g[P.eligible(g, stat, grp)].reset_index(drop=True)
        if g.empty:
            continue
        over, push, _ = model.probs(g, stat, grp, g["line"].to_numpy(float))
        g["ours"], g["our_push"] = over, push
        g["y"] = P.actual(g, stat)
        out.append(g)
    if not out:
        return pd.DataFrame()
    x = pd.concat(out, ignore_index=True)
    x["raw"] = O.implied_prob(x["price"])
    x["book"] = x["book"].astype(str)
    x["fair"] = O.one_sided_fair(x["price"], PM.ALT_MARGIN, "power")
    x["dec"] = O.american_to_decimal(x["price"])
    x["hit"] = (x["y"] > x["line"]).astype(float)
    return x


def calibration(x: pd.DataFrame) -> dict:
    bands = {}
    for name, col in (("book_raw", "raw"), ("book_fair", "fair"), ("model", "ours")):
        t = []
        for _, g in x.groupby(pd.cut(x[col], BANDS), observed=True):
            if len(g) >= 20:
                t.append({"band": f"{g[col].min():.2f}-{g[col].max():.2f}", "n": int(len(g)),
                          "predicted": round(float(g[col].mean()), 4), "actual": round(float(g["hit"].mean()), 4)})
        bands[name] = t
    sq = lambda p: (p - x["hit"]) ** 2
    d = sq(x["ours"]) - sq(x["fair"])
    high = x[x["fair"] >= 0.75]
    dh = (high["ours"] - high["hit"]) ** 2 - (high["fair"] - high["hit"]) ** 2
    return {"n": int(len(x)), "games": int(x["game_id"].nunique()),
            "brier_model": round(float(sq(x["ours"]).mean()), 5), "brier_book_fair": round(float(sq(x["fair"]).mean()), 5),
            "model_minus_book": round(float(d.mean()), 5), "se": round(_se(d, x["game_id"]), 5),
            "high_rungs": {"n": int(len(high)), "hit": round(float(high["hit"].mean()), 4) if len(high) else None,
                           "book_fair": round(float(high["fair"].mean()), 4) if len(high) else None,
                           "model": round(float(high["ours"].mean()), 4) if len(high) else None,
                           "model_minus_book": round(float(dh.mean()), 5) if len(high) else None,
                           "se": round(_se(dh, high["game_id"]), 5) if len(high) > 1 else None},
            "bands": bands}


def with_close(b: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    """Add the same book's closing price for the same rung: price CLV and EV at the close."""
    c = cl[["game_id", "player_id", "market", "line", "book", "dec", "fair"]].rename(columns={"dec": "close_dec", "fair": "close_fair"})
    b = b.merge(c, on=["game_id", "player_id", "market", "line", "book"], how="left")
    b["clv_price"] = b["dec"] / b["close_dec"] - 1
    b["clv_ev"] = b["close_fair"] * b["dec"] - 1
    return b


def grade(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0}
    b = b.assign(profit=np.where(b["hit"] > 0, b["dec"] - 1, -1.0))
    c = b.dropna(subset=["clv_price"])
    return {"bets": int(len(b)), "won": int((b["hit"] > 0).sum()),
            "avg_price": round(float(b["price"].median()), 0),
            "return": round(float(b["profit"].mean()), 4), "return_se": round(_se(b["profit"], b["game_id"]), 4),
            "claimed": round(float(b["claim"].mean()), 4),
            "clv_bets": int(len(c)),
            "clv_price": round(float(c["clv_price"].mean()), 4) if len(c) else None,
            "clv_price_se": round(_se(c["clv_price"], c["game_id"]), 4) if len(c) > 1 else None,
            "beat_close": round(float((c["clv_price"] > 0).mean()), 3) if len(c) else None,
            "clv_ev": round(float(c["clv_ev"].mean()), 4) if len(c) else None,
            "clv_ev_se": round(_se(c["clv_ev"], c["game_id"]), 4) if len(c) > 1 else None}


def passes(r: dict) -> bool:
    return bool(r.get("bets", 0) >= 30 and r["return"] > 2 * r["return_se"]
                and r.get("clv_price") is not None and r["clv_price"] > 2 * (r.get("clv_price_se") or np.inf))


def model_bets(x: pd.DataFrame, edge: float, in_band: bool) -> pd.DataFrame:
    b = x.assign(claim=x["ours"] * x["dec"] - 1)
    b = b[b["claim"] >= edge]
    if in_band:
        b = b[b["price"].abs() <= PM.MAX_ODDS]
    return b.sort_values("claim", ascending=False).drop_duplicates(["game_id", "player_id", "stat"])


def cross_book(x: pd.DataFrame, edge: float) -> pd.DataFrame:
    """A rung both books post: take the better price when it beats the other book's chance."""
    key = ["game_id", "player_id", "market", "line"]
    two = x[x.groupby(key)["book"].transform("nunique") >= 2]
    if two.empty:
        return two.assign(claim=pd.Series(dtype=float))
    other = two[key + ["book", "fair"]].rename(columns={"book": "other", "fair": "other_fair"})
    m = two.merge(other, on=key)
    m = m[m["book"] != m["other"]]
    m["claim"] = m["other_fair"] * m["dec"] - 1
    m = m[m["claim"] >= edge]
    return m.sort_values("claim", ascending=False).drop_duplicates(["game_id", "player_id", "stat"])


def ladder_gap(x: pd.DataFrame, mains: pd.DataFrame) -> pd.DataFrame:
    """The live ladder_gap tag on the over side (rungs are posted Over only): the rung's
    margin-free chance is more than LADDER_GAP below what the same book's main line implies
    through the model's shape, so the rung is cheap."""
    if mains.empty:
        return x.iloc[0:0].assign(claim=pd.Series(dtype=float))
    m = mains.assign(p_main=mains["our_over"] / (mains["our_over"] + mains["our_under"]))
    m = m[["game_id", "player_id", "stat", "book", "fair_over", "p_main"]].drop_duplicates(["game_id", "player_id", "stat", "book"])
    j = x.merge(m, on=["game_id", "player_id", "stat", "book"], how="inner")
    p_rung = j["ours"] / (1 - j["our_push"]).clip(lower=1e-9)
    j["implied"] = (j["fair_over"] + p_rung - j["p_main"]).clip(0, 1)
    j["gap"] = j["fair"] - j["implied"]
    j = j[j["gap"] < -PM.LADDER_GAP]
    j["claim"] = j["implied"] * j["dec"] - 1
    return j.sort_values("gap").drop_duplicates(["game_id", "player_id", "stat"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    a = ap.parse_args(argv)
    from props_calibration import FAMILY, load_proj
    proj = load_proj(None)
    pre = R.load_pregame()
    saved = P.PropModel.load()

    def fitted(table):
        m = P.PropModel(FAMILY).fit(table[(table.season < a.season) & (table.season >= 2013)])
        if saved is not None:
            m.iso = saved.iso
        return m
    m_pre, m_day = fitted(pre), fitted(proj)
    flag, close = B.load(a.season, "flag"), B.load(a.season, "close")
    x = priced_rungs(pre, m_pre, flag, a.season)
    if x.empty:
        print(f"No ladder prices saved for {a.season}: run the Prop price backfill with markets {','.join(LADDERS)}")
        return None
    cl = priced_rungs(proj, m_day, close, a.season)
    mains = R.priced(pre, m_pre, flag[flag["market"].isin(B.YARDAGE)], a.season)

    rep = {"season": a.season, "alt_margin": PM.ALT_MARGIN,
           "rule": "return > 2 SE and price CLV > 2 SE, 30+ bets, on a season not fitted on",
           "calibration": calibration(x), "calibration_by_stat": {st: calibration(g) for st, g in x.groupby("stat")},
           "model_bets": {}, "cross_book": {}, "ladder_gap": None}
    for e in EDGES:
        for band in (True, False):
            k = f"ev{int(e * 100)}_{'in_band' if band else 'any_price'}"
            rep["model_bets"][k] = grade(with_close(model_bets(x, e, band), cl))
        rep["cross_book"][f"ev{int(e * 100)}"] = grade(with_close(cross_book(x, e), cl))
    rep["ladder_gap"] = grade(with_close(ladder_gap(x, mains), cl))
    # by how sure the book is: low rungs (75%+), the middle, long shots
    hb = model_bets(x, EDGES[0], False)
    rep["model_bets_by_book_chance"] = {
        name: grade(with_close(hb[(hb["fair"] >= lo) & (hb["fair"] < hi)], cl))
        for name, lo, hi in (("75%+", 0.75, 1.01), ("40-75%", 0.40, 0.75), ("under 40%", 0, 0.40))}
    rep["verdict"] = {k: passes(v) for grp in ("model_bets", "cross_book") for k, v in rep[grp].items()} | {"ladder_gap": passes(rep["ladder_gap"])}
    OUT.mkdir(exist_ok=True)
    (OUT / f"props_ladders_{a.season}.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 220)
    c = rep["calibration"]
    print(f"\n{c['n']} rungs, {c['games']} games. Brier model {c['brier_model']}, book {c['brier_book_fair']}, "
          f"model minus book {c['model_minus_book']} ± {c['se']}. Rungs the book puts at 75%+: {c['high_rungs']}")
    for k in ("book_raw", "book_fair", "model"):
        print(f"\n{k}:\n", pd.DataFrame(c["bands"][k]).to_string(index=False))
    for grp in ("model_bets", "cross_book", "model_bets_by_book_chance"):
        print(f"\n== {grp}\n", pd.DataFrame(rep[grp]).T.to_string())
    print("\n== ladder_gap\n", rep["ladder_gap"])
    print("\nverdict:", rep["verdict"])
    return rep


if __name__ == "__main__":
    main()
