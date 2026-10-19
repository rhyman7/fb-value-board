"""Quarterback-change fixes, tested walk-forward.

    python experiments/qb_test.py

Games where either team starts a different quarterback from its previous game
are scored separately: that is where the model has to price the change itself.
Settings are judged on 2011-2018 (out of sample); 2019 on is the check.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")


def qb_change_games(games: pd.DataFrame) -> set:
    long = pd.concat([games[["game_id", "season", "gameday", f"{s}_team", f"{s}_qb_id"]]
                      .set_axis(["game_id", "season", "gameday", "team", "qb"], axis=1) for s in ("home", "away")])
    long = long[long.qb.notna()].sort_values(["team", "gameday"])
    long["prev"] = long.groupby("team")["qb"].shift(1)
    return set(long.loc[long.prev.notna() & (long.qb != long.prev), "game_id"])


def score(frame, changed, mf):
    out, _ = model.walk_forward(frame, margin_feats=mf)
    o = out[out.played & out.spread_line.notna()]
    res = {}
    for name, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        c = x[x.game_id.isin(changed)]
        res[name] = dict(all=np.mean(np.abs(x.model_margin - x.result)), chg=np.mean(np.abs(c.model_margin - c.result)),
                         chg_mkt=np.mean(np.abs(c.spread_line - c.result)), n_chg=len(c),
                         # how far the model sits from the closing line in QB-change games (smaller = it saw what the market saw)
                         chg_gap=np.mean(np.abs(c.model_margin - c.spread_line)), w=x.w_margin.mean())
    return res


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    changed = qb_change_games(games)
    base_mf = list(model.MARGIN_FEATS)
    print("half-life  prior  feats        | tune all  chg (mkt) gap   | hold all  chg (mkt) gap   n_chg")
    for hl in (700, 1400, 2800, 5600):
        for prior in (200, 400):
            eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_half_life=hl, qb_prior=prior)
            frame = model.build_frame(games, eng)
            for name, mf in (("base", base_mf), ("+d_qbm", base_mf + ["d_qbm"])):
                r = score(frame, changed, mf)
                t, h = r["tune"], r["hold"]
                print(f"{hl:9d}  {prior:5.0f}  {name:12s} | {t['all']:.3f}  {t['chg']:.3f} ({t['chg_mkt']:.3f}) {t['chg_gap']:.2f}"
                      f" | {h['all']:.3f}  {h['chg']:.3f} ({h['chg_mkt']:.3f}) {h['chg_gap']:.2f}  {t['n_chg']}/{h['n_chg']}", flush=True)


if __name__ == "__main__":
    main()
