"""The prop model against real DraftKings and FanDuel lines from past seasons.

    python experiments/props_real_lines.py --season 2025

Needs the historical snapshots from `python run_props.py backfill` (data/props/history/).
For each priced prop: our chance at the book's line (the walk-forward projection, with
distribution widths fitted only on seasons before this one), the book's vig-free chance,
and the result. Reported at the pricing snapshot (the first daily run within three days of
kickoff) and at the close (ten minutes before kickoff):

  - Brier score and log loss, ours against the market's, on the same props
  - our calibration at real lines
  - the bets the live rules would have flagged at the pricing snapshot: record, return
    with its standard error, and closing line value (the bet's EV at the closing price)

Look-ahead to keep in mind: the walk-forward projection uses each game's actual
game-day actives and starting quarterback, which a pricing snapshot three days out did
not know. That flatters the pricing-snapshot comparison a little; the close does not
have this problem (inactives are out 90 minutes before kickoff).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, odds as O, propbackfill as B, propdist as D, propmarket as PM, props as P  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"


def priced(proj: pd.DataFrame, model: P.PropModel, snaps: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per game x player x market x line x book, with ours, the book's and the result."""
    if snaps.empty:
        return pd.DataFrame()
    info = pd.read_csv(C.CACHE / f"player_info_{season}.csv", dtype={"player_id": str})
    rows = proj[(proj["season"] == season) & proj["played"].astype(bool)].copy()
    rows = rows.merge(info[["player_id", "name"]], on="player_id", how="left")
    rows["nname"] = rows["name"].map(PM.norm_name)
    rows = rows[~rows.duplicated(["game_id", "nname"], keep=False)]
    s = snaps.copy()
    s["nname"] = s["player"].map(PM.norm_name)
    s["side"] = s["side"].str.lower().replace({"yes": "over", "no": "under"})
    s.loc[s["market"] == "player_anytime_td", "line"] = 0.5
    w = s.pivot_table(index=["game_id", "nname", "market", "line", "book"], columns="side", values="price", aggfunc="last").reset_index()
    w = w.dropna(subset=["over", "under"])             # two-sided only: no assumed margin needed
    w["fair_over"], _ = O.no_vig_power(w["over"], w["under"])
    w["fair_over_scaled"], _ = O.no_vig(w["over"], w["under"])
    w["hold"] = O.hold(w["over"], w["under"])
    w["books_market"] = w.groupby(["game_id", "nname", "market"])["book"].transform("nunique")
    w = w.merge(rows, on=["game_id", "nname"], how="inner")
    w["stat"] = w["market"].map(PM.MARKET_STAT)
    out = []
    for (stat, grp), g in w.groupby(["stat", "group"]):
        if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
            continue
        g = g[P.eligible(g, stat, grp)]
        if g.empty:
            continue
        g = g.reset_index(drop=True)
        over, push, under = model.probs(g, stat, grp, g["line"].to_numpy(float))
        g["our_over"], g["our_push"], g["our_under"] = over, push, under
        g["y"] = P.actual(g, stat)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def role_clear(d: pd.DataFrame) -> pd.Series:
    """The live rule, from what the projection table knew before the game."""
    status_ok = d["report"].isna() | d["report"].isin(["Probable"])
    games_ok = d["one_sum"].fillna(0) >= PM.MIN_GAMES
    shift = pd.Series(0.0, index=d.index)
    for now, hist in (("tgt_share", "tgt_share_raw"), ("car_share", "car_share_raw")):
        h = d[hist].fillna(0)
        shift = np.maximum(shift, ((d[now] - h).abs() / h.where(h > 0.05)).fillna(0))
    qb_ok = (d["group"] != "QB") | (d["att_share"] >= 0.99)
    return status_ok & games_ok & (shift <= PM.ROLE_SHIFT) & qb_ok


def compare(x: pd.DataFrame) -> dict:
    """Ours against the market on the same two-sided props, pushes left out."""
    x = x[x["y"] != x["line"]]
    hit = (x["y"] > x["line"]).astype(float)
    ours = x["our_over"] / (x["our_over"] + x["our_under"])
    mkt = x["fair_over"]
    dlt = (ours - hit) ** 2 - (mkt - hit) ** 2
    cl = dlt.groupby(x["game_id"]).sum()
    blend = 0.5 * ours + 0.5 * mkt
    return {"n": int(len(x)), "games": int(x["game_id"].nunique()),
            "brier_ours": round(D.brier(ours, hit), 5), "brier_market": round(D.brier(mkt, hit), 5),
            "brier_blend_half": round(D.brier(blend, hit), 5),
            "ours_minus_market": round(float(dlt.mean()), 5), "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(x)), 5),
            "log_loss_ours": round(D.log_loss(ours, hit), 5), "log_loss_market": round(D.log_loss(mkt, hit), 5),
            "over_rate": round(float(hit.mean()), 4), "market_over": round(float(mkt.mean()), 4), "ours_over": round(float(ours.mean()), 4)}


def bets(fl: pd.DataFrame, cl: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sides the live rules would have flagged at the pricing snapshot, graded, with CLV."""
    fl = fl.copy()
    fl["role_clear"] = role_clear(fl)
    rows = []
    for side in ("over", "under"):
        p_win = fl[f"our_{side}"]
        p_lose = fl["our_under" if side == "over" else "our_over"]
        fair = fl["fair_over"] if side == "over" else 1 - fl["fair_over"]
        price = fl[side]
        edge = p_win / (p_win + p_lose) - fair
        ev = O.expected_value(p_win, price, fl["our_push"])
        ok = ((edge >= PM.MIN_EDGE) & (ev > 0) & (fl["books_market"] >= PM.MIN_BOOKS) & (fl["hold"] <= PM.MAX_HOLD)
              & (price.abs() <= PM.MAX_ODDS) & fl["role_clear"])
        b = fl[ok].assign(side=side, price=price[ok], edge=edge[ok], ev=ev[ok], p_win=p_win[ok], fair=fair[ok])
        rows.append(b)
    b = pd.concat(rows, ignore_index=True)
    if b.empty:
        return b, {"bets": 0}
    # one bet per prop: the best price across the two books
    b = b.sort_values("ev", ascending=False).drop_duplicates(["game_id", "player_id", "market", "side"])
    win = np.where(b["side"] == "over", b["y"] > b["line"], b["y"] < b["line"])
    push = b["y"] == b["line"]
    dec = O.american_to_decimal(b["price"])
    b["profit"] = np.where(push, 0.0, np.where(win, dec - 1, -1.0))
    # closing line value: the closing vig-free chance at the same line and book, else the same line any book
    c = cl[["game_id", "player_id", "market", "line", "book", "fair_over"]].rename(columns={"fair_over": "close_over"})
    b = b.merge(c, on=["game_id", "player_id", "market", "line", "book"], how="left")
    alt = cl.groupby(["game_id", "player_id", "market", "line"])["fair_over"].mean().rename("close_any").reset_index()
    b = b.merge(alt, on=["game_id", "player_id", "market", "line"], how="left")
    co = b["close_over"].fillna(b["close_any"])
    b["close_fair"] = np.where(b["side"] == "over", co, 1 - co)
    b["clv_ev"] = b["close_fair"] * (dec - 1) - (1 - b["close_fair"] - b["our_push"])
    n = len(b)
    by_game = b.groupby("game_id")["profit"].sum()
    se = float(by_game.std(ddof=1) * np.sqrt(len(by_game)) / n) if n > 1 else np.nan
    clv = b["clv_ev"].dropna()
    cg = b.dropna(subset=["clv_ev"]).groupby("game_id")["clv_ev"].sum()
    s = {"bets": int(n), "record": f"{int((b['profit'] > 0).sum())}-{int((b['profit'] < 0).sum())}-{int((b['profit'] == 0).sum())}",
         "return_flat": round(float(b["profit"].mean()), 4), "return_se": round(se, 4),
         "avg_edge": round(float(b["edge"].mean()), 4), "avg_ev_claimed": round(float(b["ev"].mean()), 4),
         "clv_bets": int(len(clv)), "avg_clv_ev": round(float(clv.mean()), 4) if len(clv) else None,
         "clv_se": round(float(cg.std(ddof=1) * np.sqrt(len(cg)) / len(clv)), 4) if len(clv) > 1 else None,
         "beat_close": round(float((clv > 0).mean()), 3) if len(clv) else None}
    return b, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--proj", type=Path, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from props_calibration import FAMILY, load_proj
    proj = load_proj(args.proj)
    # widths fitted only on seasons before this one; the tuning-season correction from the saved parameters
    model = P.PropModel(FAMILY).fit(proj[(proj.season < args.season) & (proj.season >= 2013)])
    saved = P.PropModel.load()
    if saved is not None:
        model.iso = saved.iso
    fl = priced(proj, model, B.load(args.season, "flag"), args.season)
    cl = priced(proj, model, B.load(args.season, "close"), args.season)
    if fl.empty:
        print(f"No saved snapshots for {args.season}: run `python run_props.py backfill` first.")
        return
    report = {"season": args.season, "pricing": {}, "close": {}, "calibration": {}}
    for name, x in (("pricing", fl), ("close", cl)):
        report[name]["all"] = compare(x)
        for st, g in x.groupby("stat"):
            report[name][st] = compare(g)
        y = x[x["y"] != x["line"]]
        p = y["our_over"] / (y["our_over"] + y["our_under"])
        report["calibration"][name] = D.calibration_table(p, (y["y"] > y["line"]).astype(float))
    b, s = bets(fl, cl)
    report["bets"] = s
    report["bets_by_stat"] = {st: bets(g, cl)[1] for st, g in fl.groupby("stat")}
    OUT.mkdir(exist_ok=True)
    (OUT / f"props_real_lines_{args.season}.json").write_text(json.dumps(report, indent=1, default=float))
    b.to_csv(C.RAW / f"props_real_bets_{args.season}.csv", index=False)
    pd.set_option("display.width", 200)
    for name in ("pricing", "close"):
        print(f"\n== {name}: ours against the vig-free market (Brier, lower is better)")
        print(pd.DataFrame(report[name]).T.to_string())
    print("\n== bets the live rules would have flagged at the pricing snapshot")
    print(json.dumps(report["bets"], indent=1))
    print(pd.DataFrame(report["bets_by_stat"]).T.to_string())


if __name__ == "__main__":
    main()
