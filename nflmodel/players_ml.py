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

from . import pff_routes, roles

BASE_F = ["{s}_p", "{s}_proj", "{s}_recent", "tgt_share", "car_share", "tgt_share_raw", "car_share_raw", "proj_team_att", "proj_team_car",
          "imp_pts", "margin", "proj_total", "wind", "home", "def_ypc", "def_ypa", "def_ypt", "one_sum", "ypc", "ypt", "catch", "gl_car_rate", "rz_tgt_rate"]
ROLE_F = ["snap_avg", "snap_last", "snap_last2", "snap_trend", "car_close_avg", "tgt_close_avg", "new_team", "depth", "age", "draft_pick", "exp",
          "pace", "neutral_pass"]
FEATS = BASE_F + ROLE_F
# PFF route share and targets per route (nflmodel/pff_routes.py), applied Nov 5, 2026 (Ryan's call;
# PLAYERS.md "PFF route share in the New projections"). Used when the private PFF data is there
# (the columns exist and are filled); without it the model runs on FEATS, as before.
ROUTE_F = list(pff_routes.COLS)
# where it beat the current model on both the tuning seasons and since 2019 (Oct 2026)
ADOPT = {"carries": ("RB",), "rush_yds": ("RB",), "receptions": ("WR", "RB"), "rec_yds": ("WR", "RB")}
TRAIN_GROUPS = {"carries": ("RB", "QB"), "rush_yds": ("RB", "QB"), "receptions": ("WR", "TE", "RB"), "rec_yds": ("WR", "TE", "RB")}


def add_features(proj: pd.DataFrame, R: dict, routes: bool = True) -> pd.DataFrame:
    out = proj.merge(roles.features(proj, R), on=["game_id", "player_id"], how="left")
    if routes:
        rf = pff_routes.features(proj)
        if len(rf):
            out = out.merge(rf, on=["game_id", "player_id"], how="left")
    for st in ADOPT:
        out[st + "_recent"] = out[st + "_sum"] / out["one_sum"].replace(0, np.nan)
    return out


# Questionable players who play get less than their usual usage (Oct 2026, players_more_test.py):
# a multiplier per position and practice status, from earlier seasons only, shrunk toward 1
# by n / (n + Q_SHRINK). Applied after the boosted step. Since Nov 9, 2026 it is fitted only on
# Questionable players who got on the field (an offensive snap or a stat): before 2019 the rosters
# don't mark game-day inactives, so players who sat looked like they played with zero stats and
# pulled the multiplier down (experiments/nfl_questionable_test.py, V3; PLAYERS.md).
Q_STATS = ("carries", "rush_yds", "targets", "receptions", "rec_yds")
Q_GROUPS = ("RB", "WR", "TE")
Q_SHRINK = 200
Q_RANGE = (0.5, 1.2)


def on_field(p: pd.DataFrame, snaps: pd.DataFrame) -> pd.Series:
    """1 where the player got on the field: an offensive snap (snaps: game_id, player_id,
    offense_snaps, as roles.load_all's "snaps") or any stat."""
    sn = snaps[snaps["offense_snaps"].fillna(0) > 0]
    on = pd.MultiIndex.from_frame(sn[["game_id", "player_id"]].astype(str))
    snap = pd.MultiIndex.from_frame(p[["game_id", "player_id"]].astype(str)).isin(on)
    stat = p.reindex(columns=["targets", "carries", "pass_att", "receptions"]).fillna(0).sum(axis=1).to_numpy() > 0
    return pd.Series((snap | stat).astype(int), index=p.index)


def questionable(p: pd.DataFrame, first: int = 2014, shrink: float = Q_SHRINK,
                 snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scale the projections of players listed Questionable (report_lvl 1), by position and
    practice status, with multipliers fitted on earlier seasons' Questionable players who
    played. snaps (roles.load_all's "snaps"): fit only on those who got on the field, as the
    board does since Nov 9, 2026; without it, on everyone marked active (the old fit). Players
    not listed, or with no earlier sample, are left alone."""
    p = p.copy()
    if "report_lvl" not in p or "practice_lvl" not in p:
        return p
    q = (p["report_lvl"] == 1) & p["group"].isin(Q_GROUPS)
    played = p["played"].astype(bool) & (p["active"] == 1)
    if snaps is not None:
        played = played & (on_field(p, snaps) == 1)
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


def feats_for(p: pd.DataFrame) -> list[str]:
    """FEATS, plus ROUTE_F when the PFF route columns are there and filled."""
    has = all(c in p.columns for c in ROUTE_F) and bool(p["route_avg"].notna().any())
    return FEATS + ROUTE_F if has else list(FEATS)


def walk_forward(proj: pd.DataFrame, first: int = 2014, adopt: dict | None = None, feats: list[str] | None = None) -> pd.DataFrame:
    """proj with <stat>_proj replaced, for the adopted stats and positions, by the boosted
    prediction from a model trained on earlier seasons. The current projection is kept as
    <stat>_proj_lin."""
    adopt = adopt or ADOPT
    feats = feats if feats is not None else feats_for(proj)
    p = proj.copy()
    played = p["played"].astype(bool) & (p["active"] == 1)
    for stat, groups in adopt.items():
        p[stat + "_proj_lin"] = p[stat + "_proj"]
        cols = [f.format(s=stat) for f in feats]
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
