"""Shared pieces for the defensive-player experiments (Oct 2026):

    def_retest.py        Part A: the October defensive rejects against the current model
    def_ratings.py       Part B: per-defender ratings and the team-game features built on them
    def_test.py          Part B: those features in the margin and total models
    def_players_test.py  Part B: the same in the player projections

production_frame() rebuilds the game frame exactly as the live pipeline does (ratings,
offensive line, depth-chart starters out and snap shares falling), so a candidate is
always tested against today's model. It is cached in data/raw/ (not committed).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, game_roles, injuries, model, players as PL, players_data as PD, roles  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
CACHE = C.RAW / "def_experiments"


def production_frame(refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "frame.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    base = model.build_frame(games, eng, injuries=inj)
    f0, _ = model.walk_forward(base)
    pg, pinfo = PD.load_player_games(verbose=False)
    avail = PL.load_availability(verbose=False)
    R = roles.load_all(games, verbose=False)
    rows = PL.player_rows(PL.team_games(f0, pg), pg, avail, pinfo)
    frame = game_roles.attach(base, game_roles.team_features(rows, R, avail))
    frame.to_pickle(path)
    return frame


def games():
    return data.load_all(refresh=False, verbose=False)[0]


def per_season(frame, mf=None, tf=None) -> pd.DataFrame:
    """Walk-forward; mean absolute miss of the raw margin and total, per season."""
    out, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    o = out[out.played & out.spread_line.notna()]
    return o.assign(em=(o.model_margin - o.result).abs(), et=(o.model_total - o.total).abs()).groupby("season")[["em", "et"]].agg(["sum", "count"])


def compare(base: pd.DataFrame, cand: pd.DataFrame, tune: tuple, hold: tuple, which: str = "em") -> dict:
    """Change in mean miss (candidate minus base, negative = better) over a tuning and a
    holdout window of seasons (inclusive), plus season by season."""
    d = {}
    for name, (a, b) in (("tune", tune), ("hold", hold)):
        s = [x for x in base.index if a <= x <= b]
        mb = base.loc[s, (which, "sum")].sum() / base.loc[s, (which, "count")].sum()
        mc = cand.loc[s, (which, "sum")].sum() / cand.loc[s, (which, "count")].sum()
        d[name] = mc - mb
    seas = (cand[(which, "sum")] / cand[(which, "count")]) - (base[(which, "sum")] / base[(which, "count")])
    d["seasons"] = seas
    for name, (a, b) in (("pre2020", (2013, 2019)), ("post2020", (2020, 2100))):
        s = seas[(seas.index >= a) & (seas.index <= b)]
        sb = [x for x in base.index if a <= x <= b]
        mb = base.loc[sb, (which, "sum")].sum() / base.loc[sb, (which, "count")].sum()
        mc = cand.loc[sb, (which, "sum")].sum() / cand.loc[sb, (which, "count")].sum()
        d[name] = (mc - mb, int((s < 0).sum()), len(s))
    return d


def against_open(frame, mf=None, tf=None, tune=(2014, 2018), hold=(2019, 2100)) -> dict:
    """The model's side against the opening spread (as experiments/game_roles_open_test.py):
    win rate at the opener, tuning and holdout."""
    out, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    op = data.load_openers()
    f = out.merge(op, on="game_id", how="inner")
    f = f[f.played & f.spread_line.notna() & f.open_spread.notna()].copy()
    pick = np.sign(f.model_margin - f.open_spread)
    f, pick = f[pick != 0], pick[pick != 0]
    co = np.sign(f.result - f.open_spread) * pick
    res = {}
    for name, (a, b) in (("tune", tune), ("hold", hold)):
        m = (f.season >= a) & (f.season <= b)
        res[name] = (float((co[m] > 0).sum() / (co[m] != 0).sum()), int((co[m] != 0).sum()))
    return res


def fmt_row(label, d):
    pre, post = d["pre2020"], d["post2020"]
    return (f"{label:34s} tune {d['tune']:+.4f}  hold {d['hold']:+.4f}  | 2013-19 {pre[0]:+.4f} ({pre[1]}/{pre[2]} better)"
            f"  2020+ {post[0]:+.4f} ({post[1]}/{post[2]} better)")
