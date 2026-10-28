"""A gradient-boosted last step for the player projections.

The current model's last step is a small regression per stat that blends the volume x
efficiency chain with the player's recent average. This replaces it, for the stats and
positions where it tested better on both the tuning seasons (2014-2018) and since 2019
(experiments/players_roles_test.py), with a gradient-boosted model that also sees the role
inputs (nflmodel/roles.py): snap share, close-game shares, depth chart, background, team pace
and neutral pass rate. Walk-forward: each season is predicted by a model trained on earlier
seasons only. Shown on the board as the New projections, with the Questionable adjustment below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import roles

BASE_F = ["{s}_p", "{s}_proj", "{s}_recent", "tgt_share", "car_share", "tgt_share_raw", "car_share_raw", "proj_team_att", "proj_team_car",
          "imp_pts", "margin", "proj_total", "wind", "home", "def_ypc", "def_ypa", "def_ypt", "one_sum", "ypc", "ypt", "catch", "gl_car_rate", "rz_tgt_rate"]
ROLE_F = ["snap_avg", "snap_last", "snap_last2", "snap_trend", "car_close_avg", "tgt_close_avg", "new_team", "depth", "age", "draft_pick", "exp",
          "pace", "neutral_pass"]
FEATS = BASE_F + ROLE_F
# where it beat the current model on both the tuning seasons and since 2019 (Oct 2026)
ADOPT = {"carries": ("RB",), "rush_yds": ("RB",), "receptions": ("WR", "RB"), "rec_yds": ("WR", "RB")}
TRAIN_GROUPS = {"carries": ("RB", "QB"), "rush_yds": ("RB", "QB"), "receptions": ("WR", "TE", "RB"), "rec_yds": ("WR", "TE", "RB")}


def add_features(proj: pd.DataFrame, R: dict) -> pd.DataFrame:
    out = proj.merge(roles.features(proj, R), on=["game_id", "player_id"], how="left")
    for st in ADOPT:
        out[st + "_recent"] = out[st + "_sum"] / out["one_sum"].replace(0, np.nan)
    return out


# Questionable players who play get less than their usual usage (Oct 2026, players_more_test.py):
# a multiplier per position and practice status, from earlier seasons only, shrunk toward 1
# by n / (n + Q_SHRINK). Applied after the boosted step.
Q_STATS = ("carries", "rush_yds", "targets", "receptions", "rec_yds")
Q_GROUPS = ("RB", "WR", "TE")
Q_SHRINK = 200
Q_RANGE = (0.5, 1.2)


def questionable(p: pd.DataFrame, first: int = 2014, shrink: float = Q_SHRINK) -> pd.DataFrame:
    """Scale the projections of players listed Questionable (report_lvl 1), by position and
    practice status, with multipliers fitted on earlier seasons' Questionable players who
    played. Players not listed, or with no earlier sample, are left alone."""
    p = p.copy()
    if "report_lvl" not in p or "practice_lvl" not in p:
        return p
    q = (p["report_lvl"] == 1) & p["group"].isin(Q_GROUPS)
    played = p["played"].astype(bool) & (p["active"] == 1)
    p["q_mult"] = 1.0
    for st in Q_STATS:
        if st + "_proj" not in p or st not in p:
            continue
        mult = pd.Series(1.0, index=p.index)
        for s in sorted(p.loc[q, "season"].unique()):
            if s < first:
                continue
            tr = p[q & played & (p["season"] < s) & (p[st + "_proj"] > 0.5) & p[st].notna()]
            if tr.empty:
                continue
            g = tr.groupby(["group", "practice_lvl"])
            n = g.size()
            m = (1 + n / (n + shrink) * (g[st].sum() / g[st + "_proj"].sum() - 1)).clip(*Q_RANGE)
            te = q & (p["season"] == s)
            k = pd.MultiIndex.from_frame(p.loc[te, ["group", "practice_lvl"]])
            mult[te] = pd.Series(m.reindex(k).to_numpy(), index=p.index[te]).fillna(1.0)
        p[st + "_proj"] = p[st + "_proj"] * mult
        if st == "rec_yds":
            p["q_mult"] = mult
    return p


def _model():
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)


def walk_forward(proj: pd.DataFrame, first: int = 2014, adopt: dict | None = None) -> pd.DataFrame:
    """proj with <stat>_proj replaced, for the adopted stats and positions, by the boosted
    prediction from a model trained on earlier seasons. The current projection is kept as
    <stat>_proj_lin."""
    adopt = adopt or ADOPT
    p = proj.copy()
    played = p["played"].astype(bool) & (p["active"] == 1)
    for stat, groups in adopt.items():
        p[stat + "_proj_lin"] = p[stat + "_proj"]
        cols = [f.format(s=stat) for f in FEATS]
        new = p[stat + "_proj"].copy()
        for s in sorted(p["season"].unique()):
            if s < first:
                continue
            tr = p[played & (p["season"] < s) & p["group"].isin(TRAIN_GROUPS[stat])]
            te = p[(p["season"] == s) & p["group"].isin(groups)]
            if len(tr) < 1000 or te.empty:
                continue
            m = _model().fit(tr[cols].to_numpy(float), tr[stat].clip(lower=0).to_numpy(float))
            new.loc[te.index] = m.predict(te[cols].to_numpy(float))
        p[stat + "_proj"] = new
    return p
