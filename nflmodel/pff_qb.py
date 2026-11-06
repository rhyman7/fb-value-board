"""A quarterback's prior from his PFF passing grades, for the game model's quarterback rating.

Applied Nov 5, 2026 (Ryan's go-ahead; FEATURES.md "Quarterback value from PFF grades",
experiments/qb_pff_grade_test.py): better on all six measures of the keep rule (margin on all
games and on QB-change games, and total; tuning 2015-2018 and holdout 2019 on).

A quarterback's rating is his decayed EPA per dropback shrunk with 200 dropbacks toward a prior.
The prior used to be a flat -0.13 for everyone. Now it is his PFF value:

  score   each PFF game (passing_summary, regular season and playoffs, 2014 on) turned into an
          EPA-scale number by a linear map on PFF's pass grade, fitted for each season on earlier
          seasons' games (10+ dropbacks, weighted by dropbacks): the game's nflverse EPA per
          dropback against that season's average on grades_pass. About +0.014 EPA per dropback
          per grade point.
  value   his decayed (the engine's quarterback half-life), dropback-weighted score over games
          before the one being rated, shrunk toward -0.13 with PG (400) dropbacks.

A quarterback with no PFF games keeps -0.13, and so does everyone when the PFF data isn't there
(no checkout of rhyman7/pff-data): the board then rates quarterbacks exactly as before.
PFF players are matched to nflverse ids by nflverse's players file (pff_id), then by unique
name within the team-season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ratings import _time

FIRST = 2014
PG = 400.0                  # dropbacks of shrinkage toward QB_REPLACEMENT
MIN_DB = 10                 # games used to fit the map
MIN_FIT_GAMES = 200
PLAYOFF_WEEK = {28: 19, 29: 20, 30: 21, 32: 22}     # PFF's NFL playoff weeks -> nflverse's


def qb_games(qb: pd.DataFrame) -> pd.DataFrame:
    """PFF passing_summary rows with the nflverse quarterback id, the engine's order and the game's
    nflverse EPA per dropback; empty without the PFF data."""
    from cfbmodel import pff_data
    if not pff_data.available():
        return pd.DataFrame()
    from .pff_routes import pff_ids, _players
    from .propmarket import norm_name
    p = pff_data.load("passing_summary", range(FIRST, int(qb.season.max()) + 1), league="nfl")
    if p.empty:
        return p
    p = p[pd.to_numeric(p["dropbacks"], errors="coerce").fillna(0) > 0].copy()
    p["dropbacks"] = pd.to_numeric(p["dropbacks"], errors="coerce")
    p["grades_pass"] = pd.to_numeric(p["grades_pass"], errors="coerce")
    p["week"] = p["week"].astype(int).map(lambda w: PLAYOFF_WEEK.get(w, w))
    p = p[p["week"].between(1, 22)]
    p["season"] = p["season"].astype(int)
    p["order"] = p["season"] * 100 + p["week"]
    p = p.merge(pff_ids(), left_on="player_id", right_on="pff_id", how="left")
    names = _players().dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]
    q = qb[["season", "team", "qb_id"]].drop_duplicates()
    q["nn"] = q["qb_id"].map(names).map(norm_name)
    q = q.dropna(subset=["nn"]).drop_duplicates(["season", "team", "nn"], keep=False)
    p["nn"] = p["player"].map(norm_name)
    miss = p["nfl_id"].isna()
    fb = p.loc[miss, ["season", "team_name", "nn"]].reset_index().merge(
        q.rename(columns={"team": "team_name"}), on=["season", "team_name", "nn"], how="inner").set_index("index")["qb_id"]
    p.loc[fb.index, "nfl_id"] = fb
    p = p[p["nfl_id"].notna()]
    nv = qb.groupby(["qb_id", "order"])[["epa", "dropbacks"]].sum()
    nv = (nv["epa"] / nv["dropbacks"]).rename("epa_db").reset_index().rename(columns={"qb_id": "nfl_id"})
    p = p.merge(nv, on=["nfl_id", "order"], how="left")
    return p[["nfl_id", "season", "order", "dropbacks", "grades_pass", "epa_db"]].dropna(subset=["grades_pass"]).reset_index(drop=True)


def fit_map(p: pd.DataFrame, seasons) -> dict:
    """{season: (intercept, slope)} of game EPA per dropback (vs that season's average) on
    grades_pass, fitted on PFF games of earlier seasons."""
    t = p.dropna(subset=["epa_db"])
    t = t[t["dropbacks"] >= MIN_DB].copy()
    if t.empty:
        return {}
    lg = t.groupby("season").apply(lambda x: np.average(x["epa_db"], weights=x["dropbacks"]))
    t["rel"] = t["epa_db"] - t["season"].map(lg)
    out = {}
    for s in seasons:
        tr = t[t["season"] < s]
        if len(tr) < MIN_FIT_GAMES:
            continue
        w = np.sqrt(tr["dropbacks"].to_numpy(float))
        X = np.column_stack([np.ones(len(tr)), tr["grades_pass"].to_numpy(float)])
        out[int(s)] = np.linalg.lstsq(X * w[:, None], tr["rel"].to_numpy() * w, rcond=None)[0]
    return out


class Prior:
    """values(order_now, ids, half_life): each quarterback's PFF prior as of a game."""

    def __init__(self, games: pd.DataFrame, coefs: dict, replacement: float = C.QB_REPLACEMENT, pg: float = PG):
        self.p, self.coefs, self.rep, self.pg = games, coefs, replacement, pg
        self.order = games["order"].to_numpy() if len(games) else np.array([], int)

    def values(self, order_now: int, ids, half_life: float) -> np.ndarray:
        ids = list(ids)
        beta = self.coefs.get(int(order_now) // 100)
        if beta is None or not len(self.p):
            return np.full(len(ids), self.rep)
        m = self.order < order_now
        p = self.p[m]
        score = beta[0] + beta[1] * p["grades_pass"].to_numpy(float)
        w = 0.5 ** ((_time(order_now) - _time(p["order"].to_numpy())) / (half_life / 35.0)) * p["dropbacks"].to_numpy(float)
        df = pd.DataFrame({"id": p["nfl_id"].to_numpy(), "w": w, "ws": w * score}).groupby("id").sum()
        g = (df["ws"] + self.pg * self.rep) / (df["w"] + self.pg)
        return pd.Series(ids, dtype=object).map(g).fillna(self.rep).to_numpy(float)


def load(qb: pd.DataFrame, seasons) -> Prior | None:
    """The PFF prior for the engine, or None without the PFF data (the engine then uses -0.13)."""
    g = qb_games(qb)
    if g.empty:
        return None
    coefs = fit_map(g, seasons)
    return Prior(g, coefs) if coefs else None
