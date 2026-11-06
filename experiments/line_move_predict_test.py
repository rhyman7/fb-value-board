"""Predicting the NFL spread's move from the opener to the close, tested walk-forward (Nov 6, 2026).

    python experiments/line_move_predict_test.py

THE RULE (written before the first run, Nov 6, 2026)
-----------------------------------------------------
The board's one proven spread edge is betting the opener when the model disagrees with it: the
line tends to move toward the model (backtest.opening_report; "Disagrees by 2+ points"). Today
the board infers the move from that gap alone. This test fits the move itself.

  target    move = closing spread - opening spread (nflverse convention, positive = home favoured)
  frame     the opener-time model (pipeline.py): each team starts last week's quarterback, no
            late news (blind frame, walk-forward), games after a Monday-night game left out, as
            opening_report does. Openers: nfelo's file (2009 on).
  inputs    known at the open:
              gap        model margin - opening spread
              nfelo_gap  nfelo's pre-game rating gap without its quarterback adjustment
                         ((nfelo_dif_base - home_net_qb_mod) / 25) - opening spread
              qb_risk_h, qb_risk_a  last game's starter threw under 75% of his team's dropbacks
                         (left early or shared), home and away, 0/1
              open       the opening spread itself
  models    walk-forward, refitted each season on earlier seasons (2009 on), ordinary least squares:
              gap-only   move ~ gap                         (what the board implies today)
              full       move ~ gap + nfelo_gap + qb_risk_h + qb_risk_a + open

Measures (tuning 2014-2018, holdout 2019 on; four):
  1-2  average miss of the predicted move, full vs gap-only, tuning and holdout.
  3-4  return per $1 at the opener (the opener's price; -110 where missing) of the bets each rule
       picks: today's rule bets the model's side when |gap| >= 2; the full rule bets the side of
       the predicted move when |predicted move| >= T, with T chosen on the tuning seasons as the
       smallest half-point threshold that picks no more bets than today's rule does there.
       Tuning and holdout.
KEEP RULE (CONVENTIONS.md): kept only if better on all four and worse on none; a mixed result
goes to Ryan. Win rate, closing line value per bet and standard errors are reported, not used.
A kept result changes nothing by itself: putting it on the board (a "line likely to move" mark on
the opener bets) needs a plan and a mockup first.
Props check: not applicable (spreads only).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import backtest as BT, bets as B, config as C, data, extra, model, odds as O, pff_qb  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
TUNE = (2014, 2018)
FEATS = ["gap", "nfelo_gap", "qb_risk_h", "qb_risk_a", "open"]


def qb_risk(games: pd.DataFrame, qb: pd.DataFrame) -> pd.DataFrame:
    """Per (team, order of the game): did the team's starter in its previous game throw under
    75% of its dropbacks then?"""
    q = qb.groupby(["team", "order", "qb_id"], as_index=False)["dropbacks"].sum()
    tot = q.groupby(["team", "order"])["dropbacks"].transform("sum")
    q["share"] = q["dropbacks"] / tot
    top = q.sort_values("share").drop_duplicates(["team", "order"], keep="last")
    top["risk"] = (top["share"] < 0.75).astype(float)
    top = top.sort_values(["team", "order"])
    top["risk_next"] = top["risk"]          # applies to the team's next game
    return top[["team", "order", "risk_next"]]


def build(games, tg, qb) -> pd.DataFrame:
    te, qe = extra.load_extra(verbose=False)
    pr = pff_qb.load(qb, sorted(games.season.unique()))
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=pr)
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan)
    early, _ = model.walk_forward(model.build_frame(blind, eng))
    op = data.load_openers(refresh=False)
    f = early.merge(op, on="game_id", how="inner")
    mon = BT._came_off_monday(games)
    f["monday"] = f["game_id"].map(mon).fillna(False).astype(bool)
    f = f[f.played & f.spread_line.notna() & f.open_spread.notna() & ~f.monday].copy()
    n = pd.read_csv(C.RAW / "nfelo_games.csv", usecols=["game_id", "nfelo_dif_base", "home_net_qb_mod"])
    n["nfelo_spread"] = (n["nfelo_dif_base"] - n["home_net_qb_mod"].fillna(0)) / 25.0
    f = f.merge(n[["game_id", "nfelo_spread"]], on="game_id", how="left")
    f["gap"] = f["model_margin"] - f["open_spread"]
    f["nfelo_gap"] = (f["nfelo_spread"] - f["open_spread"]).fillna(0.0)
    f["open"] = f["open_spread"]
    f["move"] = f["spread_line"] - f["open_spread"]
    # previous game of each team: its order, then that game's starter share
    r = qb_risk(games, qb)
    order_of = games[["game_id", "season", "week"]].assign(order=lambda x: x.season * 100 + x.week)
    if "order" not in f:
        f = f.merge(order_of[["game_id", "order"]], on="game_id", how="left")
    for side, col in (("home_team", "qb_risk_h"), ("away_team", "qb_risk_a")):
        rr = r.rename(columns={"team": side}).sort_values("order")
        x = f[["game_id", side, "order"]].sort_values("order")
        m = pd.merge_asof(x, rr.rename(columns={"order": "prev_order"}), left_on="order", right_on="prev_order",
                          by=side, allow_exact_matches=False, direction="backward")
        f = f.merge(m[["game_id", "risk_next"]].rename(columns={"risk_next": col}), on="game_id", how="left")
        f[col] = f[col].fillna(0.0)
    return f


def walk(f: pd.DataFrame, feats: list[str]) -> np.ndarray:
    pred = np.full(len(f), np.nan)
    for s in sorted(f.season.unique()):
        tr = f[f.season < s]
        te = (f.season == s).to_numpy()
        if len(tr) < 300:
            continue
        X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in feats])
        beta = np.linalg.lstsq(X, tr["move"].to_numpy(float), rcond=None)[0]
        Xt = np.column_stack([np.ones(te.sum())] + [f.loc[te, c].to_numpy(float) for c in feats])
        pred[te] = Xt @ beta
    return pred


def graded(f: pd.DataFrame, side: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    x = f[mask].copy()
    s = side[mask]
    x["cover"] = np.sign(x["result"] - x["open_spread"]) * s
    price = np.where(s > 0, x["open_home_odds"], x["open_away_odds"])
    price = np.where(np.isnan(price), B.DEFAULT_PRICE, price)
    dec = O.american_to_decimal(price)
    x["profit"] = np.where(x["cover"] > 0, dec - 1, np.where(x["cover"] < 0, -1.0, 0.0))
    x["clv"] = (x["spread_line"] - x["open_spread"]) * s
    return x


def summ(x: pd.DataFrame) -> dict:
    w, l = int((x.cover > 0).sum()), int((x.cover < 0).sum())
    return {"n": len(x), "w": w, "l": l, "win": round(w / max(w + l, 1), 4), "roi": round(float(x.profit.mean()), 4),
            "roi_se": round(float(x.profit.std(ddof=1) / np.sqrt(max(len(x), 2))), 4), "clv": round(float(x.clv.mean()), 3)}


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    f = build(games, tg, qb)
    f = f[f.season >= 2009].reset_index(drop=True)
    print(f"games {len(f)} (2009 on), nfelo gap present {f.nfelo_spread.notna().mean():.1%}, qb risk flags {f[['qb_risk_h','qb_risk_a']].mean().round(3).to_dict()}")
    f["p_gap"] = walk(f, ["gap"])
    f["p_full"] = walk(f, FEATS)
    T, H = f.season.between(*TUNE).to_numpy(), (f.season >= C.HOLDOUT_FROM).to_numpy()
    res = {}
    for name, m in (("tune", T), ("hold", H)):
        x = f[m & f.p_full.notna().to_numpy()]
        e0, e1 = np.abs(x.p_gap - x.move), np.abs(x.p_full - x.move)
        d = (e1 - e0)
        res[name] = {"mae_gap": round(float(e0.mean()), 4), "mae_full": round(float(e1.mean()), 4),
                     "change": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4), "n": len(x)}
        print(f"{name}: move miss gap-only {res[name]['mae_gap']:.4f}  full {res[name]['mae_full']:.4f}  ({res[name]['change']:+.4f} ± {res[name]['se']:.4f}, n {len(x)})")
    # coefficients of the last fit, for reading
    tr = f[f.season < 2026]
    X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in FEATS])
    print("full model, fitted through 2025:", dict(zip(["const"] + FEATS, np.round(np.linalg.lstsq(X, tr.move.to_numpy(float), rcond=None)[0], 4))))

    cur_mask = (f.gap.abs() >= 2).to_numpy()
    cur_side = np.sign(f.gap.to_numpy())
    n_cur_tune = int((cur_mask & T).sum())
    thr = None
    for t in np.arange(0.5, 5.01, 0.5):
        if int(((f.p_full.abs() >= t).to_numpy() & T).sum()) <= n_cur_tune:
            thr = float(t)
            break
    new_mask = (f.p_full.abs() >= thr).to_numpy() & f.p_full.notna().to_numpy()
    new_side = np.sign(f.p_full.fillna(0).to_numpy())
    print(f"\nbets at the opener: today's rule |gap| >= 2 ({n_cur_tune} on tuning); full rule |predicted move| >= {thr}")
    out = {}
    for name, m in (("tune", T), ("hold", H)):
        a, b = summ(graded(f, cur_side, cur_mask & m)), summ(graded(f, new_side, new_mask & m))
        out[name] = {"current": a, "full": b}
        print(f"  {name}: today's {a['w']}-{a['l']} win {a['win']:.1%} roi {a['roi']:+.3f} ± {a['roi_se']:.3f} clv {a['clv']:+.2f} (n {a['n']})"
              f"  | full {b['w']}-{b['l']} win {b['win']:.1%} roi {b['roi']:+.3f} ± {b['roi_se']:.3f} clv {b['clv']:+.2f} (n {b['n']})")
    by = []
    for s in sorted(f.season.unique()):
        if s < TUNE[0]:
            continue
        m = (f.season == s).to_numpy()
        a, b = summ(graded(f, cur_side, cur_mask & m)), summ(graded(f, new_side, new_mask & m))
        by.append((s, a["roi"], b["roi"], round(float(np.abs(f.p_gap[m] - f.move[m]).mean() - np.abs(f.p_full[m] - f.move[m]).mean()), 4)))
    print(pd.DataFrame(by, columns=["season", "roi_today", "roi_full", "move_miss_gain"]).to_string(index=False))
    measures = [res["tune"]["mae_full"] < res["tune"]["mae_gap"], res["hold"]["mae_full"] < res["hold"]["mae_gap"],
                out["tune"]["full"]["roi"] > out["tune"]["current"]["roi"], out["hold"]["full"]["roi"] > out["hold"]["current"]["roi"]]
    print(f"\nbetter on {sum(measures)} of 4 -> {'KEEP' if all(measures) else 'not kept' if not any(measures) else 'mixed: Ryan’s call'}")


if __name__ == "__main__":
    main()
