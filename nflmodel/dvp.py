"""Defense against each position, as a last multiplier on the player projections.

The player model judges a defense by team-level rates (yards per carry and per attempt allowed;
nothing for receivers, where the team-level rate did not help: PLAYERS.md). The matchup
dashboards split pass defense by position, and that split does help
(experiments/def_vs_position.py, Nov 1, 2026): on the holdout seasons it lowered the average
miss on receiving yards for wide receivers, tight ends and backs, receptions for all three, and
backs' rushing yards and carries, each by more than two standard errors. Passing yards did not
pass and are left alone.

Rating. For each defense-game, the log of (yards the defense allowed to a position + 1) over
(what the projections expected those players to get + 1), so it is already adjusted for the
offenses it faced. A defense's rating before a game is the weighted average of its earlier
games' logs (HALF_LIFE games, last season's at LAST_SEASON weight, older ones dropped), shrunk
toward 0 by SHRINK pseudo-games.

Use. projection x exp(b x rating), with b per position and stat fitted by least squares on
earlier seasons' projections (`fit`). Only the pairs in APPLY change the board; the others
are measured but not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE = 6.0
LAST_SEASON = 0.5
SHRINK = 4.0
PAIRS = [("WR", "rec_yds"), ("TE", "rec_yds"), ("RB", "rec_yds"), ("WR", "receptions"), ("TE", "receptions"),
         ("RB", "receptions"), ("RB", "rush_yds"), ("QB", "pass_yds"), ("RB", "carries")]
APPLY = {p for p in PAIRS if p != ("QB", "pass_yds")}


def _eligible(p: pd.DataFrame, grp: str) -> pd.Series:
    m = p["group"] == grp
    if grp == "QB":
        m &= p["att_share"] > 0.5
    return m


def ratings(proj: pd.DataFrame, proj_suffix: str = "_proj") -> pd.DataFrame:
    """(game_id, opp, group, stat, rating) for every game in proj, played or not, from the
    defense's games before it. proj needs the actual stats for played games (played, active)
    and the projections `<stat><proj_suffix>`."""
    out = []
    played = proj["played"].fillna(False).astype(bool) & (proj["active"] == 1)
    for grp, stat in PAIRS:
        col = stat + proj_suffix
        if col not in proj:
            continue
        d = proj[_eligible(proj, grp) & proj[col].notna()]
        key = ["game_id", "season", "week", "opp"]
        allg = d[key].drop_duplicates()
        pl = d[played.reindex(d.index).fillna(False)]
        agg = pl.groupby(key).agg(act=(stat, "sum"), exp=(col, "sum")).reset_index()
        agg = agg[agg["exp"] > 1]
        agg["lr"] = np.log((agg["act"].clip(lower=0) + 1) / (agg["exp"] + 1))
        g = allg.merge(agg[key + ["lr"]], on=key, how="left").sort_values(["opp", "season", "week"])
        res = []
        for dfn, h in g.groupby("opp", sort=False):
            lr = h["lr"].to_numpy(float)
            season = h["season"].to_numpy(int)
            have = ~np.isnan(lr)
            idx = np.cumsum(have) - have          # played games before each row
            pos = np.flatnonzero(have)
            r = np.zeros(len(h))
            for i in range(len(h)):
                prev = pos[pos < i]
                prev = prev[season[prev] >= season[i] - 1]
                if len(prev):
                    k = idx[i] - 1 - idx[prev]    # played games ago
                    w = 0.5 ** (k / HALF_LIFE) * np.where(season[prev] == season[i], 1.0, LAST_SEASON)
                    r[i] = float((w * lr[prev]).sum() / (w.sum() + SHRINK))
            res.append(pd.DataFrame({"game_id": h["game_id"].to_numpy(), "opp": dfn, "group": grp, "stat": stat, "rating": r}))
        if res:
            out.append(pd.concat(res, ignore_index=True))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["game_id", "opp", "group", "stat", "rating"])


def attach(proj: pd.DataFrame, rt: pd.DataFrame, grp: str, stat: str) -> pd.Series:
    """The rating for each row of proj (0 where there is none)."""
    r = rt[(rt["group"] == grp) & (rt["stat"] == stat)][["game_id", "opp", "rating"]]
    m = proj[["game_id", "opp"]].merge(r, on=["game_id", "opp"], how="left")
    return pd.Series(m["rating"].fillna(0.0).to_numpy(), index=proj.index)


def fit(proj: pd.DataFrame, rt: pd.DataFrame, seasons, proj_suffix: str = "_proj") -> dict:
    """{(group, stat): b}, least squares of (actual - projection) on projection x rating."""
    played = proj["played"].fillna(False).astype(bool) & (proj["active"] == 1)
    out = {}
    for grp, stat in PAIRS:
        col = stat + proj_suffix
        if col not in proj:
            continue
        d = proj[played & _eligible(proj, grp) & proj[col].notna() & (proj[col] > 0) & proj["season"].isin(list(seasons))]
        if len(d) < 500:
            continue
        x = d[col] * attach(d, rt, grp, stat)
        den = float((x ** 2).sum())
        out[(grp, stat)] = float((x * (d[stat] - d[col])).sum() / den) if den > 0 else 0.0
    return out


def apply(proj: pd.DataFrame, rt: pd.DataFrame, b: dict, proj_suffix: str = "_proj", pairs=None) -> pd.DataFrame:
    """proj with `<stat><proj_suffix>` multiplied by exp(b x rating) for each pair in `pairs`
    (APPLY by default), and the rating kept as dvp_<stat> for the player card."""
    p = proj.copy()
    for grp, stat in (pairs or APPLY):
        col = stat + proj_suffix
        if (grp, stat) not in b or col not in p:
            continue
        m = _eligible(p, grp)
        r = attach(p[m], rt, grp, stat)
        p.loc[m, col] = p.loc[m, col] * np.exp(b[(grp, stat)] * r)
        p.loc[m, f"dvp_{stat}"] = r
    return p
