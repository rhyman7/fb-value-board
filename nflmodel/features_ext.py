"""Candidate features beyond the base model, for testing and (where they earn it) use.

Rated the same way as the base metrics (opponent-adjusted ridge, decayed):
  sack   sacks per dropback          (pass protection vs pass rush)
  expl   explosive plays per play    (pass 20+ yards, rush 10+)
  ed     early-down pass EPA per dropback
  st     net special-teams EPA per game
  pace   seconds per snap, neutral situations
  snaps  offensive snaps per game
  fum    fumble-recovery luck per game (fumbles lost minus half of all fumbles)
Quarterback:
  cpoe   completion % over expected, decayed and shrunk, as a starter-vs-usual gap
Game context (no ratings needed):
  travel distance, time zones crossed, West Coast team in an early Eastern kickoff,
  wind and cold (outdoor games), divisional games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ratings import METRICS, RatingEngine, _time

EXTRA_METRICS = ("sack", "expl", "ed", "st", "pace", "snaps", "fum")
EXTRA_RIDGE = {"sack": 300.0, "expl": 320.0, "ed": 240.0, "st": 8.0, "pace": 8.0, "snaps": 8.0, "fum": 8.0}
CPOE_PRIOR_ATT = 150.0

# Home stadium location and standard-time UTC offset, by current franchise code.
SITES = {
    "ARI": (33.53, -112.26, -7), "ATL": (33.76, -84.40, -5), "BAL": (39.28, -76.62, -5), "BUF": (42.77, -78.79, -5),
    "CAR": (35.23, -80.85, -5), "CHI": (41.86, -87.62, -6), "CIN": (39.10, -84.52, -5), "CLE": (41.51, -81.70, -5),
    "DAL": (32.75, -97.09, -6), "DEN": (39.74, -105.02, -7), "DET": (42.34, -83.05, -5), "GB": (44.50, -88.06, -6),
    "HOU": (29.68, -95.41, -6), "IND": (39.76, -86.16, -5), "JAX": (30.32, -81.64, -5), "KC": (39.05, -94.48, -6),
    "LA": (33.95, -118.34, -8), "LAC": (33.95, -118.34, -8), "LV": (36.09, -115.18, -8), "MIA": (25.96, -80.24, -5),
    "MIN": (44.97, -93.26, -6), "NE": (42.09, -71.26, -5), "NO": (29.95, -90.08, -6), "NYG": (40.81, -74.07, -5),
    "NYJ": (40.81, -74.07, -5), "PHI": (39.90, -75.17, -5), "PIT": (40.45, -80.02, -5), "SEA": (47.60, -122.33, -8),
    "SF": (37.40, -121.97, -8), "TB": (27.98, -82.50, -5), "TEN": (36.17, -86.77, -6), "WAS": (38.91, -76.86, -5),
}


def _km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


PRODUCTION_METRICS = ("snaps",)   # the extras that earned a place in the live model


class ExtRatingEngine(RatingEngine):
    """The base engine plus extra rated metrics. `extra_metrics` picks which ones are
    solved each week (all of them for testing; PRODUCTION_METRICS for the board)."""

    def __init__(self, team_games, qb_games, team_extra, qb_extra, extra_metrics=EXTRA_METRICS, **kw):
        tg = team_games.merge(team_extra.drop(columns=["season"]), on=["game_id", "team"], how="left")
        super().__init__(tg, qb_games, **kw)
        self.ridge.update(EXTRA_RIDGE)
        self.metrics = METRICS + tuple(extra_metrics)
        t = tg.sort_values("order").reset_index(drop=True)
        f = lambda c: t[c].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            db, png, edp = f("dropbacks"), f("plays_ng"), f("ed_pass")
            self.y["sack"] = np.nan_to_num(f("sacks") / db)
            self.y["expl"] = np.nan_to_num(f("expl") / png)
            self.y["ed"] = np.nan_to_num(f("ed_pass_epa") / edp)
        lg_pace = np.nanmean(f("sec_per_play"))
        self.y["st"] = np.nan_to_num(f("st_epa"))
        self.y["pace"] = np.where(np.isnan(f("sec_per_play")), lg_pace, f("sec_per_play"))
        self.y["snaps"] = np.nan_to_num(f("snaps"), nan=np.nanmean(f("snaps")))
        self.y["fum"] = np.nan_to_num(f("fumbles_lost") - 0.5 * f("fumbles"))
        ones = np.ones(len(t))
        self.n_obs.update({"sack": np.nan_to_num(db), "expl": np.nan_to_num(png), "ed": np.nan_to_num(edp),
                           "st": ones, "pace": ones, "snaps": ones, "fum": ones})
        # quarterback CPOE per game, aligned with the qb_games ordering
        q = qb_games.merge(qb_extra.drop(columns=["season"]), on=["game_id", "qb_id"], how="left").sort_values("order")
        self.c_order = q["order"].to_numpy()
        self.c_id = q["qb_id"].to_numpy()
        self.c_team = q["team"].map(self.tix).to_numpy()
        self.c_att = q["att_cpoe"].fillna(0).to_numpy(float)
        self.c_sum = q["cpoe_sum"].fillna(0).to_numpy(float)

    def ratings_at(self, order_now: int) -> dict:
        out = super().ratings_at(order_now)
        m = self.c_order < order_now
        age = _time(order_now) - _time(self.c_order[m])
        w = 0.5 ** (age / (C.QB_HALF_LIFE_DROPBACKS / 35.0))
        df = pd.DataFrame({"id": self.c_id[m], "a": w * self.c_att[m], "s": w * self.c_sum[m]}).groupby("id").sum()
        val = df["s"] / (df["a"] + CPOE_PRIOR_ATT)        # shrunk toward 0 (league average)
        out["cpoe"] = val.to_dict()
        recent = m & (self.c_order // 100 >= order_now // 100 - self.window)
        tw = self._weights(order_now, self.c_order[recent]) * self.c_att[recent]
        vals = pd.Series(self.c_id[recent]).map(val).fillna(0).to_numpy()
        num = np.bincount(self.c_team[recent], weights=tw * vals, minlength=self.n)
        den = np.bincount(self.c_team[recent], weights=tw, minlength=self.n)
        out["cpoe_base"] = np.divide(num, den, out=np.zeros(self.n), where=den > 0)
        return out

    def _extra_features(self, f, r, g, h, a):
        def gap(side, ti):
            qid = f[f"{side}_qb_id"]
            if not isinstance(qid, str):
                return 0.0
            return r["cpoe"].get(qid, 0.0) - r["cpoe_base"][ti]
        f["d_cpoe"] = gap("home", h) - gap("away", a)


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Travel, time zone, weather and divisional features for every game."""
    df = df.copy()
    neutral = df["neutral"].to_numpy() == 1
    hs = df["home_team"].map(SITES)
    as_ = df["away_team"].map(SITES)
    df["travel"] = [0.0 if n or h is None or a is None else _km(a, h) / 1000.0 for n, h, a in zip(neutral, hs, as_)]
    tz_h = hs.map(lambda x: x[2] if isinstance(x, tuple) else -5)
    tz_a = as_.map(lambda x: x[2] if isinstance(x, tuple) else -5)
    df["tz_east"] = np.where(neutral, 0, (tz_h - tz_a).clip(lower=0))      # away team flew east
    df["tz_west"] = np.where(neutral, 0, (tz_a - tz_h).clip(lower=0))
    early = df["gametime"].fillna("13:00").str[:2].astype(int) < 14
    df["west_early"] = (~neutral & early & (tz_a <= -7) & (tz_h == -5)).astype(int)
    outdoor = df["indoor"] == 0
    wind = df["wind"].where(outdoor, 0.0)
    df["wind_known"] = (outdoor & df["wind"].notna()).astype(int)
    df["wind"] = wind.fillna(wind[outdoor & df["played"]].median() if (outdoor & df["played"]).any() else 8.0)
    temp = df["temp"].where(outdoor, 68.0)
    df["temp"] = temp.fillna(60.0)
    df["wind_hi"] = (df["wind"] - 10).clip(lower=0)
    df["cold"] = (45 - df["temp"]).clip(lower=0)
    df["hfa_div"] = df["hfa"] * df["div_game"].fillna(0)
    return df
