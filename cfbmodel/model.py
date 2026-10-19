"""The college game model: the NFL method (nflmodel/model.py) with college features.

1. Raw model     ridge regression of margin and total on team ratings, market-implied
                 ratings from earlier lines, the quarterback gap, home field and rest,
                 refitted each season on earlier seasons only
2. Market blend  projection = line + w * (raw model - line), w measured walk-forward

Only games between two FBS teams are modelled: they are the ones with lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflmodel.model import Linear, _blend_weight

from . import config as C
from .ratings import CFBRatingEngine

BASE_MARGIN = ["d_pass", "d_rush", "d_sr", "d_pts", "d_mkt", "d_qb", "hfa", "rest_diff"]
BASE_TOTAL = ["s_pass", "s_rush", "s_sr", "s_pts", "s_mkt", "s_qb", "lg_pts", "indoor"]
# Added after testing on 2017-2019 (CFB_FEATURES.md): talent early in the season, returning
# production, talent, and the quarterback gap early in the season; for totals, returning
# defensive production, tempo (snaps per game) and wind above 10 mph.
MARGIN_FEATS = BASE_MARGIN + ["d_talent_early", "d_rtprod", "d_qb_early", "d_talent"]
TOTAL_FEATS = BASE_TOTAL + ["s_def_rtprod", "s_snaps", "wind_hi"]
ALPHA_MARGIN, ALPHA_TOTAL = 150.0, 600.0   # game-model ridge, chosen on 2017-2019 (NFL uses 20)
FIRST_OOS = C.FIRST_TEST_SEASON


def _km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Home field, rest, wind and the college-only pregame numbers (talent, returning production)."""
    df = df.copy()
    neutral = df["neutral"].to_numpy() == 1
    df["rest_diff"] = (df["home_rest"] - df["away_rest"]).clip(-7, 7).fillna(0) * (1 - df["playoff"])
    outdoor = df["indoor"] == 0
    wind = pd.to_numeric(df["wind"], errors="coerce").where(outdoor, 0.0)
    df["wind_known"] = (outdoor & wind.notna()).astype(int)
    med = wind[outdoor & df["played"]].median()
    med = 7.0 if not np.isfinite(med) else med
    df["wind"] = wind.fillna(med)
    df["wind_hi"] = (df["wind"] - 10).clip(lower=0)
    temp = pd.to_numeric(df["temp"], errors="coerce").where(outdoor, 68.0)
    df["cold"] = (45 - temp.fillna(65.0)).clip(lower=0)
    # Talent (247 composite) and returning production, as differences and sums.
    for a in ("talent", "rtprod", "off_rtprod", "def_rtprod", "ret_qb"):
        h, w = pd.to_numeric(df[f"home_{a}"], errors="coerce"), pd.to_numeric(df[f"away_{a}"], errors="coerce")
        fill = pd.concat([h, w]).median()
        df[f"d_{a}"] = (h.fillna(fill) - w.fillna(fill))
        df[f"s_{a}"] = (h.fillna(fill) + w.fillna(fill))
    df["d_talent"] = df["d_talent"] / 100.0
    df["s_talent"] = df["s_talent"] / 100.0
    # How much of the season has been played: early games lean on priors.
    wk = np.where(df["postseason"] == 1, 15, df["week"]).astype(float)
    df["early"] = np.clip((6.0 - wk) / 5.0, 0, 1)          # 1 in weeks 0-1, 0 from week 6
    df["d_talent_early"] = df["d_talent"] * df["early"]
    df["d_rtprod_early"] = df["d_rtprod"] * df["early"]
    df["d_mkt_early"] = df["d_mkt"] * df["early"]
    df["d_qb_early"] = df["d_qb"] * df["early"]
    df["d_pts_early"] = df["d_pts"] * df["early"]
    df["d_sr_early"] = df["d_sr"] * df["early"]
    df["hfa_conf"] = df["hfa"] * df["conference_game"]
    # Travel and time zones for the away team (none at a neutral site).
    km = _km(pd.to_numeric(df["away_lat"], errors="coerce"), pd.to_numeric(df["away_lon"], errors="coerce"),
             pd.to_numeric(df["lat"], errors="coerce"), pd.to_numeric(df["lon"], errors="coerce"))
    df["travel"] = np.where(neutral, 0.0, np.nan_to_num(km / 1000.0))
    df["elev"] = np.where(neutral, 0.0, pd.to_numeric(df["elevation"], errors="coerce").fillna(200).clip(lower=0) / 1000.0)
    return df


def build_frame(games: pd.DataFrame, engine: CFBRatingEngine, verbose: bool = False) -> pd.DataFrame:
    """Every FBS-vs-FBS game from FIRST_MODEL_SEASON on, with pre-game features attached."""
    g = games[(games.season >= C.FIRST_MODEL_SEASON) & (games.fbs_game == 1)].copy()
    feats = engine.features(g, verbose=verbose)
    df = g.drop(columns=["home_qb_id", "away_qb_id"]).merge(feats, on="game_id", how="inner")
    fbs = games[(games.fbs_game == 1) & (games.neutral == 0) & games.played]
    home_margin = fbs.groupby("season")["result"].mean()
    trail = {s: home_margin[(home_margin.index < s) & (home_margin.index >= s - 5)].mean() for s in df.season.unique()}
    df["hfa"] = df["season"].map(trail).fillna(home_margin.mean()) * (1 - df["neutral"])
    return add_context(df)


def walk_forward(df: pd.DataFrame, alpha: float = ALPHA_MARGIN, alpha_total: float = ALPHA_TOTAL, margin_feats: list[str] | None = None,
                 total_feats: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    MF = MARGIN_FEATS if margin_feats is None else margin_feats
    TF = TOTAL_FEATS if total_feats is None else total_feats
    df = df.copy()
    for col in ("model_margin", "model_total", "proj_margin", "proj_total", "w_margin", "w_total"):
        df[col] = np.nan
    info = {}
    for s in sorted(x for x in df.season.unique() if x >= FIRST_OOS):
        train = df[(df.season < s) & df.played]
        test = df.season == s
        m = Linear(alpha, intercept=False).fit(train[MF].to_numpy(float), train["result"].to_numpy(float))
        t = Linear(alpha_total).fit(train[TF].to_numpy(float), train["total"].to_numpy(float))
        df.loc[test, "model_margin"] = m.predict(df.loc[test, MF].to_numpy(float))
        df.loc[test, "model_total"] = t.predict(df.loc[test, TF].to_numpy(float))
        past = df[(df.season < s) & (df.season >= max(FIRST_OOS, s - C.BLEND_LOOKBACK)) & df.played]
        wm = _blend_weight(*(past[c].to_numpy(float) for c in ("model_margin", "spread_line", "result")))
        wt = min(_blend_weight(*(past[c].to_numpy(float) for c in ("model_total", "total_line", "total"))), C.TOTAL_WEIGHT_CAP)
        df.loc[test, "w_margin"], df.loc[test, "w_total"] = wm, wt
        df.loc[test, "proj_margin"] = df.loc[test, "model_margin"]
        df.loc[test, "proj_total"] = df.loc[test, "model_total"]
        hl = test & df["spread_line"].notna()
        df.loc[hl, "proj_margin"] = df.loc[hl, "spread_line"] + wm * (df.loc[hl, "model_margin"] - df.loc[hl, "spread_line"])
        ht = test & df["total_line"].notna()
        df.loc[ht, "proj_total"] = df.loc[ht, "total_line"] + wt * (df.loc[ht, "model_total"] - df.loc[ht, "total_line"])
        info[int(s)] = {"margin_coefs": m.raw_coefs(MF), "total_coefs": t.raw_coefs(TF),
                        "total_intercept": round(float(t.y0 - (t.mu / t.sd) @ t.beta), 4),
                        "margin_feats": list(MF), "total_feats": list(TF),
                        "margin_c": [float(x) for x in m.beta / m.sd], "total_c": [float(x) for x in t.beta / t.sd],
                        "w_margin": round(wm, 3), "w_total": round(wt, 3), "n_train": int(len(train))}
    return df[df.season >= FIRST_OOS], info
