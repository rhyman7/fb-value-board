"""Two untested ideas: the referee crew, and Week 18.

    python experiments/referee_week18_test.py

Referee. Crews differ in how many penalties they call, and penalties extend drives, so a crew
could move totals (and, less plausibly, the home side). For every game, the referee's earlier
games (all seasons before this game) give two walk-forward numbers, each shrunk toward 0 by
SHRINK games:

  ref_tot    average of (total points - closing total)
  ref_home   average of (home margin - closing spread)

Each is added on its own to the board's total or margin model (experiments/feature_test.py's
walk-forward), and also checked directly against the closing line: does the residual against
the close line up with the crew's past residual? A crew effect the market already prices
would show in the model and not against the close.

Week 18. Teams resting starters or with nothing to play for: the market hears about it, the
model's ratings do not. If so the model should trust the market more in Week 18. Checked as
the raw model's and the closing line's average miss in Week 18 against weeks 1-17, and the
blend weight that would have minimized the miss in Week 18 (projection = line + w x (model -
line)) against the weight the board uses.

Rule, as in feature_test.py: a feature is kept only if it lowers the raw model's average miss
on the tuning seasons (to 2018) and holds on the holdout (2019 on).

Writes experiments/out/referee_week18.json.
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
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "referee_week18.json"
SHRINK = 30.0


def referee_features(games: pd.DataFrame) -> pd.DataFrame:
    g = games[games["played"] & games["referee"].notna() & games["total_line"].notna() & games["spread_line"].notna()].copy()
    g = g.sort_values(["season", "week", "gameday", "gametime"])
    g["r_tot"] = g["total"] - g["total_line"]
    g["r_home"] = g["result"] - g["spread_line"]
    rows = []
    for ref, h in g.groupby("referee", sort=False):
        ct = h["r_tot"].cumsum().shift(1).fillna(0.0)
        ch = h["r_home"].cumsum().shift(1).fillna(0.0)
        n = np.arange(len(h))
        rows.append(pd.DataFrame({"game_id": h["game_id"], "ref_tot": ct / (n + SHRINK), "ref_home": ch / (n + SHRINK), "ref_n": n}))
    return pd.concat(rows, ignore_index=True)


def miss(o):
    return {"margin": float(np.mean(np.abs(o.model_margin - o.result))), "total": float(np.mean(np.abs(o.model_total - o.total)))}


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame = model.build_frame(games, engine)
    rf = referee_features(games)
    frame = frame.merge(rf, on="game_id", how="left")
    for c in ("ref_tot", "ref_home"):
        frame[c] = frame[c].fillna(0.0)
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)

    def run(m, t):
        out, _ = model.walk_forward(frame, margin_feats=m, total_feats=t)
        o = out[out.played & out.spread_line.notna()]
        return o, {"tune": miss(o[o.season < C.HOLDOUT_FROM]), "hold": miss(o[o.season >= C.HOLDOUT_FROM])}

    base_o, base = run(mf, tf)
    _, r_tot = run(mf, tf + ["ref_tot"])
    _, r_home = run(mf + ["ref_home"], tf)
    rep = {"base": base,
           "ref_tot": {k: round(r_tot[k]["total"] - base[k]["total"], 4) for k in ("tune", "hold")},
           "ref_home": {k: round(r_home[k]["margin"] - base[k]["margin"], 4) for k in ("tune", "hold")}}

    # against the close: does the crew's past residual predict this game's?
    x = games[games["played"]].merge(rf, on="game_id")
    x = x[x["ref_n"] >= 20]
    for col, res in (("ref_tot", x["total"] - x["total_line"]), ("ref_home", x["result"] - x["spread_line"])):
        for name, m in (("tune", x["season"] < C.HOLDOUT_FROM), ("hold", x["season"] >= C.HOLDOUT_FROM)):
            a, b = x.loc[m, col], res[m]
            slope = float(np.polyfit(a, b, 1)[0]) if a.std() > 0 else 0.0
            r = float(np.corrcoef(a, b)[0, 1])
            rep.setdefault("vs_close", {}).setdefault(col, {})[name] = {"n": int(m.sum()), "corr": round(r, 4),
                                                                         "corr_se": round(1 / np.sqrt(m.sum()), 4), "slope": round(slope, 3)}
        # the over/under record when the crew leans one way by a point or more (totals only)
        if col == "ref_tot":
            lean = x[col].abs() >= 1.0
            side = np.sign(x.loc[lean, col])
            out = np.sign(res[lean])
            rec = {"bets": int(((out != 0)).sum()), "won": int((side == out).sum()), "lost": int(((side != out) & (out != 0)).sum())}
            for name, m in (("tune", x.loc[lean, "season"] < C.HOLDOUT_FROM), ("hold", x.loc[lean, "season"] >= C.HOLDOUT_FROM)):
                s_, o_ = side[m], out[m]
                rec[name] = f"{int((s_ == o_).sum())}-{int(((s_ != o_) & (o_ != 0)).sum())}"
            rep["ref_tot_lean_1pt"] = rec

    # Week 18
    o = base_o[base_o.game_type.eq("REG")] if "game_type" in base_o else base_o
    w18, rest = o[o.week == 18], o[(o.week < 18)]
    def line_miss(d):
        return {"margin": float(np.mean(np.abs(d.spread_line - d.result))), "total": float(np.mean(np.abs(d.total_line - d.total)))}
    rep["week18"] = {"games": int(len(w18)), "model": miss(w18), "line": line_miss(w18),
                     "weeks_1_17": {"model": miss(rest), "line": line_miss(rest)}}
    ws = np.linspace(0, 0.6, 61)
    def best_w(d):
        e = [np.mean(np.abs(d.spread_line + w * (d.model_margin - d.spread_line) - d.result)) for w in ws]
        return float(ws[int(np.argmin(e))]), float(min(e))
    for name, d in (("week18", w18), ("weeks_1_17", rest)):
        for per, m in (("tune", d.season < C.HOLDOUT_FROM), ("hold", d.season >= C.HOLDOUT_FROM)):
            w, e = best_w(d[m])
            rep.setdefault("best_margin_weight", {}).setdefault(name, {})[per] = {"w": round(w, 2), "miss": round(e, 3), "n": int(m.sum())}
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
