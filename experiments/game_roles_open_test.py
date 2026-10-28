"""Opening-line check for the two game-model role signals (d_depth1_out, d_snap_drop).

    python experiments/game_roles_open_test.py --proj path

game_roles_test.py built "starters out" from the game-day inactive list, which is not known
when openers are posted or when the board is built a few days out. This version uses only
what is known by Friday: roster status the week before (players.pregame_active: injured
reserve and the like are out) plus players listed Out or Doubtful on the week's report.
Scored, for the base model and with both signals, against the opener (side picked by the
model's margin against the opening spread): win rate at the opener, at the close, and how
often the line moved toward the pick. Seasons 2014 on; tuning before 2019, holdout after.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import config as C, data, extra, injuries, model, players as PL, roles  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402
import game_roles_test as G  # noqa: E402

warnings.filterwarnings("ignore")


def friday_active(proj, R, avail):
    rows = proj[["season", "week", "player_id"]].copy()
    rows["t"] = PL._t(rows["season"], rows["week"])
    a = avail[["season", "week", "player_id", "status"]].copy()
    a["t"] = PL._t(a["season"], a["week"])
    rows = rows.merge(a.drop_duplicates(["player_id", "t"], keep="last")[["player_id", "t", "status"]], on=["player_id", "t"], how="left")
    rows.index = proj.index
    pre = PL.pregame_active(rows, avail)
    pr = R["practice"][["season", "week", "team", "player_id", "report"]]
    rep = proj[["season", "week", "team", "player_id"]].merge(pr, on=["season", "week", "team", "player_id"], how="left")["report"].to_numpy()
    return ((pre == 1) & ~(np.nan_to_num(rep, nan=-1) >= 2)).astype(int)


def against_open(out, op):
    f = out.merge(op, on="game_id", how="inner")
    f = f[f.played & f.spread_line.notna() & f.open_spread.notna() & (f.season >= 2014)].copy()
    pick = np.sign(f.model_margin - f.open_spread)
    f = f[pick != 0]
    pick = pick[pick != 0]
    co = np.sign(f.result - f.open_spread) * pick
    cc = np.sign(f.result - f.spread_line) * pick
    mv = (f.spread_line - f.open_spread) * pick
    res = {}
    for name, m in (("tune", f.season < C.HOLDOUT_FROM), ("hold", f.season >= C.HOLDOUT_FROM)):
        res[name] = {"n": int(m.sum()), "open_win": float((co[m] > 0).sum() / ((co[m] != 0).sum())),
                     "close_win": float((cc[m] > 0).sum() / ((cc[m] != 0).sum())),
                     "toward": float((mv[m] > 0).mean()), "clv": float(mv[m].mean()),
                     "miss": float(np.mean(np.abs(f.model_margin[m] - f.result[m])))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    args = ap.parse_args()
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame = model.build_frame(games, eng, injuries=inj)
    proj = pd.read_pickle(args.proj)
    R = roles.load_all(games, verbose=False)
    avail = PL.load_availability(verbose=False)
    op = data.load_openers()
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    base, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    rows = [("base", against_open(base, op))]
    for label, act in (("game-day inactives", proj["active"]), ("known by Friday", friday_active(proj, R, avail))):
        fr = G.attach(frame, G.team_features(proj.assign(active=act), R))
        out, _ = model.walk_forward(fr, margin_feats=mf + ["d_depth1_out", "d_snap_drop"], total_feats=tf)
        rows.append((label, against_open(out, op)))
    for label, r in rows:
        print(f"{label:20s} " + " | ".join(f"{k} n {v['n']} miss {v['miss']:.3f} win@open {v['open_win']:.4f} win@close {v['close_win']:.4f} "
                                           f"toward {v['toward']:.3f} clv {v['clv']:+.3f}" for k, v in r.items()), flush=True)


if __name__ == "__main__":
    main()
