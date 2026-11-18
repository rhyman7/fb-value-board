"""WOPR standing: a nudge to WR/TE receiving yards for receivers whose WOPR is high or low for
their projection (Nov 18, 2026, Ryan's call on a 14-of-15 trade-off; provisional, January recheck).

Pre-set test: experiments/wopr_standing_test.py (FEATURES.md "NFL: receivers whose WOPR is high for
their projection"; Project docs claude/plan-wopr-standing-test-2026-11-18.md and
claude/wopr-standing-test-results-2026-11-18.md). Better on every bet-time measure and on 2026 weeks
1-10; flat at the closing lines with game-day projections and under 1 standard error on the game-day
holdout, so it was not kept by the rule and is on the board as Ryan's trade-off.

    new receiving yards = projection x (1 + b x z)

  WOPR  1.5 x target share + 0.7 x air-yards share. Air-yards share is built like tgt_share: his
        share of the team's air yards in games he was active and played (players.asof, the player
        half-life), shrunk by one game of half his position's average.
  z     his WOPR minus the usual WOPR for his position at his projection (a straight line), over
        the spread around that line, held to +/- CLIP
  b     one number

The line, its spread and b are constants in data/wopr_standing.json, fitted on the game-day
walk-forward projections of the seasons before 2026 by
`experiments/wopr_standing_test.py --write-constants` (version A: WR/TE projected MIN_PROJ+ yards).
Don't change them, the cut-off or the clip without a new pre-set test; refit only at the January
recheck. Receptions, running backs and every other stat are untouched. Applied after the boosted
step and the Questionable step and before defense-vs-position. Without the file, or with APPLIED
False, the board is what it was before.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

APPLIED = True
GROUPS = ("WR", "TE")
MIN_PROJ = 10.0
CLIP = 2.5
PATH = C.DATA / "wopr_standing.json"
STAT = "rec_yds_proj"


def wopr(table: pd.DataFrame) -> pd.DataFrame:
    """wopr_raw and ay_share for every (game_id, player_id) of the player table (players.build),
    as of each game from earlier games. Zero for other positions."""
    from .players import PLAYER_HL, asof
    a = table[["game_id", "player_id", "team", "season", "t", "played", "active", "group", "air_yards", "tgt_share_raw"]].copy()
    use = (a["active"] == 1) & a["played"].astype(bool)
    team_air = a[use].groupby(["game_id", "team"])["air_yards"].sum().rename("team_air").reset_index()
    a = a.merge(team_air, on=["game_id", "team"], how="left")
    a["ay_sh"] = np.where(use & (a["team_air"] > 0), a["air_yards"] / a["team_air"], np.nan)
    a["one"] = np.where(use, 1.0, np.nan)
    f = asof(a, "player_id", ["ay_sh"], ["one"], PLAYER_HL)
    pos = a[use].groupby("group")["ay_sh"].mean()
    out = f[["game_id", "player_id", "group"]].copy()
    ay = (f["ay_sh_sum"].to_numpy(float) + out["group"].map(pos).fillna(0.0).to_numpy(float) * 0.5) / (f["one_sum"].to_numpy(float) + 1.0)
    rec = out["group"].isin(("WR", "TE", "RB")).to_numpy()
    out["ay_share"] = np.where(rec, ay, 0.0)
    out["wopr_raw"] = np.where(rec, 1.5 * f["tgt_share_raw"].fillna(0.0).to_numpy(float) + 0.7 * out["ay_share"], 0.0)
    return out[["game_id", "player_id", "ay_share", "wopr_raw"]].fillna(0.0)


def load(path=None) -> dict | None:
    p = PATH if path is None else path
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def factor(group, proj, wopr_raw, k: dict) -> np.ndarray:
    """1 + b x z for each row; 1 for other positions and below the cut-off."""
    group, proj, w = np.asarray(group), np.asarray(proj, float), np.asarray(wopr_raw, float)
    out = np.ones(len(proj))
    for g in GROUPS:
        ln = k["line"].get(g)
        if not ln:
            continue
        m = (group == g) & (proj >= k.get("min_proj", MIN_PROJ)) & np.isfinite(proj) & np.isfinite(w)
        z = np.clip((w[m] - (ln["c0"] + ln["c1"] * proj[m])) / ln["sd"], -k.get("clip", CLIP), k.get("clip", CLIP))
        out[m] = 1.0 + k["b"] * z
    return out


def apply(proj: pd.DataFrame, table: pd.DataFrame, season: int, groups=GROUPS, k: dict | None = None) -> tuple[pd.DataFrame, int]:
    """proj with this season's receiving-yards projections moved for the given positions.
    Returns (proj, rows moved). proj is returned unchanged when off or without constants."""
    k = load() if k is None else k
    if not APPLIED or k is None or STAT not in proj:
        return proj, 0
    w = wopr(table)
    look = dict(zip(zip(w["game_id"], w["player_id"]), w["wopr_raw"]))
    out = proj.copy()
    sel = ((out["season"] == season) & out["group"].isin(groups)).to_numpy()
    wr = np.array([look.get(x, np.nan) for x in zip(out.loc[sel, "game_id"], out.loc[sel, "player_id"])], float)
    f = factor(out.loc[sel, "group"], out.loc[sel, STAT], wr, k)
    out.loc[sel, STAT] = out.loc[sel, STAT].to_numpy(float) * f
    return out, int((f != 1.0).sum())
