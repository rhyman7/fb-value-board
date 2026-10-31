"""A market-anchored prop blend, fitted on one set of seasons and tested on the next.

    python experiments/props_blend.py

The blend (PROPS.md, "A market-anchored blend"): for each two-sided prop at a book's line,

    blend = book + w x (ours - book)

with `book` the book's vig-free over chance (power method) and `ours` the model's, both with
pushes set aside. w = 0 is the book; w = 1 is the model.

Data: the backfilled DraftKings and FanDuel prices (data/props/history/<season>/), the
pricing snapshot (first daily run within three days of kickoff) and the close (ten minutes
before), priced exactly as experiments/props_real_lines.py does: pregame projections at the
pricing snapshot, game-day projections at the close, distribution widths fitted only on
seasons before the one scored.

w is chosen by least squares (minimum Brier score; closed form) on the training seasons'
pricing snapshots, one weight per market and one pooled over the yardage markets, clipped
to [0, 1]. Two out-of-sample folds:

  fold 1   fit on 2023, test on 2024 (yardage markets: the only ones backfilled for both)
  fold 2   fit on 2023 and 2024, test on 2025 (the fold that decides)

Receptions, rush attempts and anytime TD are backfilled for 2025 only, so they have no
training season: they are tested at the pooled yardage weight (fixed before looking), and
their in-sample best weight is reported for information only. Anytime TD is posted Yes
only, so its vig-free chance assumes the live code's 8% two-way margin
(propmarket.ONE_SIDED_MARGIN), only Yes is ever a bet, and it is reported on its own, left
out of the pooled numbers and the verdict.

Reported on the test season, per market and pooled:
  - Brier score: book, ours, blend; blend minus book with a standard error clustered by game
  - the sides the blend would flag at the pricing snapshot (blend chance at least EDGE
    points over the book's vig-free chance, positive EV at that book's price, and the
    live liquidity and role rules): record, flat return, and closing line value (the bet's
    EV at the closing vig-free price) two ways: only where the bet's exact line is still
    posted at the close (as props_real_lines.py), and for every bet with the closing chance
    carried to the bet's line by the model (carried_clv, the live log's definition), each
    with a standard error clustered by game
  - for every prop, whether the closing vig-free chance moved toward the blend's side

The rule for turning flags back on, set before running this (PROPS.md "To earn them back"):
on fold 2, the pooled blend must beat the book's Brier score by more than two standard
errors AND the flagged sides' average closing line value must be above zero by more than two
standard errors (the carried measure: the exact-line one leaves out most bets whose line moved
toward them, which was found on the first run and is why both are reported). Anything less is not a pass. One look-ahead stays in both snapshots: the game
model is anchored to the closing spread and total (PROPS.md); a blend that only passes because
of it would show up as a large CLV with no Brier gain.

Writes experiments/out/props_blend.json.
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
from nflmodel import odds as O, propbackfill as B, propmarket as PM, props as P  # noqa: E402
import props_real_lines as R  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out"
SEASONS = (2023, 2024, 2025)
YARDAGE = {"pass_yds", "rush_yds", "rec_yds"}
EDGES = (0.01, 0.02, 0.03)        # blend minus book, in probability, for a flagged side
FOLDS = {"fold1": ((2023,), 2024), "fold2": ((2023, 2024), 2025)}


def _se_by_game(values: pd.Series, games: pd.Series) -> float:
    """Standard error of a mean, clustered by game."""
    n = len(values)
    if n < 2:
        return float("nan")
    g = values.groupby(games.to_numpy()).sum()
    return float(g.std(ddof=1) * np.sqrt(len(g)) / n)


def frame(fl: pd.DataFrame) -> pd.DataFrame:
    """Priced props with pushes set aside: ours, mkt (the book's vig-free over chance), hit."""
    x = fl[fl["y"] != fl["line"]].copy()
    x["ours"] = x["our_over"] / (x["our_over"] + x["our_under"])
    x["mkt"] = x["fair_over"]
    x["hit"] = (x["y"] > x["line"]).astype(float)
    return x


def fit_w(x: pd.DataFrame) -> float:
    """The weight with the lowest Brier score: sum d(hit - book) / sum d^2, d = ours - book."""
    d = x["ours"] - x["mkt"]
    den = float((d ** 2).sum())
    return float(np.clip((d * (x["hit"] - x["mkt"])).sum() / den, 0, 1)) if den > 0 else 0.0


def brier_report(x: pd.DataFrame, w: float) -> dict:
    blend = x["mkt"] + w * (x["ours"] - x["mkt"])
    sq = lambda p: (p - x["hit"]) ** 2
    diff = sq(blend) - sq(x["mkt"])
    diff_ours = sq(x["ours"]) - sq(x["mkt"])
    return {"n": int(len(x)), "games": int(x["game_id"].nunique()), "w": round(w, 3),
            "brier_book": round(float(sq(x["mkt"]).mean()), 5), "brier_ours": round(float(sq(x["ours"]).mean()), 5),
            "brier_blend": round(float(sq(blend).mean()), 5),
            "blend_minus_book": round(float(diff.mean()), 5), "se": round(_se_by_game(diff, x["game_id"]), 5),
            "ours_minus_book": round(float(diff_ours.mean()), 5), "se_ours": round(_se_by_game(diff_ours, x["game_id"]), 5)}


def as_blend(fl: pd.DataFrame, w: float) -> pd.DataFrame:
    """The priced props with ours replaced by the blend (the push chance kept)."""
    b = fl.copy()
    ours = b["our_over"] / (b["our_over"] + b["our_under"])
    p = b["fair_over"] + w * (ours - b["fair_over"])
    rest = 1 - b["our_push"]
    b["our_over"], b["our_under"] = p * rest, (1 - p) * rest
    return b


def carried_clv(b: pd.DataFrame, cl: pd.DataFrame, model) -> pd.Series:
    """Closing line value for every flagged side, the way the live log measures it
    (PROPS.md, "The log"): the closing vig-free chance at the same book (else either book),
    carried from the closing line to the bet's line with the model's own difference between
    the two lines. props_real_lines.bets only grades a bet whose exact line is still posted
    at the close, which leaves out most of the bets whose line moved toward them."""
    key = ["game_id", "player_id", "market"]
    c = cl.sort_values("line").drop_duplicates(key + ["book"])
    same = b[key + ["book"]].reset_index().merge(c, on=key + ["book"], how="inner")
    anyb = b[key].reset_index().merge(c.drop_duplicates(key).drop(columns="book"), on=key, how="inner")
    m = pd.concat([same, anyb[~anyb["index"].isin(same["index"])]], ignore_index=True)
    if m.empty:
        return pd.Series(np.nan, index=b.index)
    m["bet_line"] = m["index"].map(b["line"])
    parts = []
    for (stat, grp), g in m.groupby(["stat", "group"]):
        over, _, under = model.probs(g.reset_index(drop=True), stat, grp, g["bet_line"].to_numpy(float))
        parts.append(pd.Series(over / (over + under), index=g.index))
    ours_bet = pd.concat(parts)
    ours_close = m["our_over"] / (m["our_over"] + m["our_under"])
    p = (m["fair_over"] + ours_bet - ours_close).clip(0.01, 0.99)
    p.index = m["index"]
    bb = b.loc[p.index]
    fair = np.where(bb["side"] == "over", p, 1 - p)
    dec = O.american_to_decimal(bb["price"])
    return pd.Series(fair * (dec - 1) - (1 - fair - bb["our_push"]), index=p.index).reindex(b.index)


def flagged(fl: pd.DataFrame, cl: pd.DataFrame, w: float, edge: float, close_model=None) -> dict:
    """The sides the blend would flag at the pricing snapshot, graded, with closing line value
    two ways: `avg_clv_ev` only where the bet's exact line is still posted at the close
    (props_real_lines), `clv_carried` for every bet (carried_clv)."""
    keep = PM.MIN_EDGE
    PM.MIN_EDGE = edge
    try:
        b, s = R.bets(as_blend(fl, w), cl)
    finally:
        PM.MIN_EDGE = keep
    if not b.empty:
        s["return_se"] = round(_se_by_game(b["profit"], b["game_id"]), 4)
        c = b.dropna(subset=["clv_ev"])
        s["clv_se"] = round(_se_by_game(c["clv_ev"], c["game_id"]), 4) if len(c) > 1 else None
        s["overs"] = int((b["side"] == "over").sum())
        if close_model is not None:
            cv = carried_clv(b, cl, close_model)
            ok = cv.notna()
            s["clv_carried_bets"] = int(ok.sum())
            s["clv_carried"] = round(float(cv[ok].mean()), 4) if ok.any() else None
            s["clv_carried_se"] = round(_se_by_game(cv[ok], b.loc[ok, "game_id"]), 4) if ok.sum() > 1 else None
            s["beat_close_carried"] = round(float((cv[ok] > 0).mean()), 3) if ok.any() else None
    return s


def moves(x: pd.DataFrame, cl: pd.DataFrame, w: float) -> dict:
    """Every pricing-snapshot prop at the same book and line at the close: did the vig-free
    chance move toward the side the blend leaned to?"""
    c = cl[["game_id", "player_id", "market", "line", "book", "fair_over"]].rename(columns={"fair_over": "close_over"})
    m = x.merge(c, on=["game_id", "player_id", "market", "line", "book"], how="inner")
    blend = m["mkt"] + w * (m["ours"] - m["mkt"])
    lean = np.sign(blend - m["mkt"])
    mv = lean * (m["close_over"] - m["mkt"])
    m = m[lean != 0]
    mv = mv[lean != 0]
    return {"n": int(len(m)), "toward": round(float((mv > 0).mean()), 3) if len(m) else None,
            "away": round(float((mv < 0).mean()), 3) if len(m) else None,
            "avg_move": round(float(mv.mean()), 4) if len(m) else None,
            "se": round(_se_by_game(mv, m["game_id"]), 4) if len(m) > 1 else None}


def evaluate(fl: pd.DataFrame, cl: pd.DataFrame, weights: dict, pooled: float, close_model=None) -> dict:
    """Brier, flagged sides and line moves on one test season. weights: stat -> w."""
    x = frame(fl)
    xp = x[x["stat"] != "anytime_td"]               # assumed margin: reported alone, not pooled
    flp = fl[fl["stat"] != "anytime_td"]
    out = {"pooled": {"brier": brier_report(xp, pooled)}}
    # the pooled test uses each market's own weight where it has one
    wcol = xp["stat"].map(weights).fillna(pooled)
    xs = xp.assign(ours=xp["mkt"] + wcol * (xp["ours"] - xp["mkt"]))
    out["pooled"]["brier_own_w"] = brier_report(xs, 1.0) | {"w": "per market"}
    fl_own = pd.concat([as_blend(g, weights.get(st, pooled)) for st, g in flp.groupby("stat")], ignore_index=True)
    out["pooled"]["flagged"] = {f"{e:.2f}": flagged(fl_own, cl, 1.0, e, close_model) for e in EDGES}
    out["pooled"]["moves"] = moves(xs, cl, 1.0)
    for st, g in x.groupby("stat"):
        w = weights.get(st, pooled)
        f = fl[fl["stat"] == st]
        out[st] = {"brier": brier_report(g, w),
                   "flagged": {f"{e:.2f}": flagged(f, cl, w, e, close_model) for e in EDGES},
                   "moves": moves(g, cl, w)}
    return out


def with_no_side(snaps: pd.DataFrame) -> pd.DataFrame:
    """Anytime TD is posted Yes only. Add a No price that gives the market the live code's
    assumed two-way overround (PM.ONE_SIDED_MARGIN, 8%), so it can be priced like the others;
    the No side is never a bet (its price is taken away again after pricing)."""
    a = snaps[(snaps["market"] == "player_anytime_td") & (snaps["side"].str.lower() == "yes")]
    if a.empty:
        return snaps
    p_yes = 1 / O.american_to_decimal(a["price"])
    p_no = (1 + PM.ONE_SIDED_MARGIN - p_yes).clip(0.01, 0.99)
    dec = 1 / p_no
    no = a.assign(side="no", price=np.where(dec >= 2, (dec - 1) * 100, -100 / (dec - 1)))
    have_no = snaps[(snaps["market"] == "player_anytime_td") & (snaps["side"].str.lower() == "no")]
    key = ["game_id", "book", "player", "line"]
    if len(have_no):
        no = no.merge(have_no[key].drop_duplicates(), on=key, how="left", indicator=True)
        no = no[no["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([snaps, no], ignore_index=True)


def priced_season(proj, model, snaps, season):
    x = R.priced(proj, model, with_no_side(snaps), season)
    if len(x):
        x.loc[x["stat"] == "anytime_td", "under"] = np.nan          # never bet the made-up No price
    return x


def main():
    from props_calibration import FAMILY, load_proj
    proj = load_proj(None)
    pre = R.load_pregame()
    saved = P.PropModel.load()
    cache = R.C.RAW / "props_blend_priced.pkl"
    priced = pd.read_pickle(cache) if cache.exists() else {}
    close_models = {}
    for season in SEASONS:
        def fitted(table):
            m = P.PropModel(FAMILY).fit(table[(table.season < season) & (table.season >= 2013)])
            if saved is not None:
                m.iso = saved.iso
            return m
        close_models[season] = fitted(proj)
        if season in priced:
            continue
        fl = priced_season(pre, fitted(pre), B.load(season, "flag"), season)
        cl = priced_season(proj, close_models[season], B.load(season, "close"), season)
        priced[season] = (fl, cl)
        print(f"{season}: {len(fl)} pricing-snapshot props, {len(cl)} at the close, "
              f"markets {sorted(fl['stat'].unique()) if len(fl) else []}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(priced, cache)

    report = {"rule": "fold 2 pooled: blend beats book Brier by > 2 SE and flagged sides' CLV > 0 by > 2 SE",
              "clv_note": "the verdict uses clv_carried (every flagged bet, the live log's definition); avg_clv_ev "
                          "only grades bets whose exact line is still posted at the close, which drops most bets "
                          "whose line moved toward them"}
    for name, (train, test) in FOLDS.items():
        tr = frame(pd.concat([priced[s][0] for s in train], ignore_index=True))
        weights = {st: fit_w(g) for st, g in tr.groupby("stat")}
        pooled = fit_w(tr[tr["stat"].isin(YARDAGE)])
        fl, cl = priced[test]
        if name == "fold1":
            fl, cl = fl[fl["stat"].isin(YARDAGE)], cl[cl["stat"].isin(YARDAGE)]
        rep = {"train": list(train), "test": test, "weights": {k: round(v, 3) for k, v in weights.items()},
               "pooled_w": round(pooled, 3), "train_brier": {st: brier_report(g, weights[st]) for st, g in tr.groupby("stat")},
               "test_eval": evaluate(fl, cl, weights, pooled, close_models[test])}
        te = frame(fl)
        rep["test_in_sample_w"] = {st: round(fit_w(g), 3) for st, g in te.groupby("stat")}   # information only
        report[name] = rep

    p = report["fold2"]["test_eval"]["pooled"]
    br = p["brier_own_w"]
    best = max(p["flagged"].values(), key=lambda s: s.get("bets", 0))
    brier_pass = br["blend_minus_book"] < -2 * br["se"]
    clv_ok = [s for s in p["flagged"].values() if s.get("bets", 0) and s.get("clv_carried") is not None
              and s.get("clv_carried_se") and s["clv_carried"] > 2 * s["clv_carried_se"]]
    report["verdict"] = {"brier_pass": bool(brier_pass), "clv_pass_at_edges": [k for k, s in p["flagged"].items() if s in clv_ok],
                         "pass": bool(brier_pass and clv_ok), "largest_flag_set": best}
    OUT.mkdir(exist_ok=True)
    (OUT / "props_blend.json").write_text(json.dumps(report, indent=1, default=float))
    for name in FOLDS:
        r = report[name]
        print(f"\n== {name}: train {r['train']} -> test {r['test']}; weights {r['weights']}, pooled yardage {r['pooled_w']}")
        rows = {k: v["brier"] for k, v in r["test_eval"].items() if k != "pooled"}
        rows["pooled (yardage w)"] = r["test_eval"]["pooled"]["brier"]
        rows["pooled (own w)"] = r["test_eval"]["pooled"]["brier_own_w"]
        print(pd.DataFrame(rows).T.to_string())
        print("flagged, pooled:")
        print(pd.DataFrame(r["test_eval"]["pooled"]["flagged"]).T.to_string())
        print("moves, pooled:", r["test_eval"]["pooled"]["moves"])
        print("test-season in-sample w (information only):", r["test_in_sample_w"])
    print("\nverdict:", json.dumps(report["verdict"], indent=1, default=float))


if __name__ == "__main__":
    main()
