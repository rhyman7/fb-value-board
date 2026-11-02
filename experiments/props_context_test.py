"""Coaching changes and playing surface, checked on the player projections (the props side).

    python experiments/props_context_test.py

FEATURES.md tested both on the game models only (Nov 2, 2026: no). They could still matter for
player stats: a new head coach can reshuffle targets and carries, and turf may add yards per
carry or catch. Per player-game (nflverse games file for coaches and surface, as the game tests):

  new_hc       the player's team has a different head coach from the start of last season
  new_early    the same, weeks 1-4 only (when last season's usage weighs most in the shares)
  mid          the team changed head coach earlier this season
  turf         the game is on artificial turf (any roof)
  turf_out     on turf and outdoors (turf apart from domes, which the projections already know)

Two ways each can act on a projection, fitted per stat and position group on the fitting seasons
by least squares, scored on later seasons:
  shift   projection x exp(b x flag)                        (more or fewer yards)
  shrink  projection + b x flag x (group average - projection)   (roles reshuffle toward the middle)

Projections: the chain projections (players_proj.pkl; fit 2013-2018, score 2019 on) for every
stat, and the New projections the board shows for players_ml.ADOPT (fit 2019-2021, score 2022 on),
built as experiments/dvp_ml_test.py builds them.

Rule, set before running (Nov 2, 2026): a context feature counts for a stat only if it lowers the
average miss on the scored seasons by more than two standard errors (clustered by game) on the
projections the board shows for that stat, and is no worse on the fitting seasons. Only then would
it be checked against the books' 2025 lines. Writes experiments/out/props_context_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import props_blend as PB  # noqa: E402
from nflmodel import config as C, data, players_ml, roles  # noqa: E402
from props_calibration import load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "props_context_test.json"
GRASS = {"grass", "dessograss"}
PAIRS = [("QB", "pass_yds"), ("QB", "pass_att"), ("RB", "carries"), ("RB", "rush_yds"), ("RB", "receptions"), ("RB", "rec_yds"),
         ("WR", "receptions"), ("WR", "rec_yds"), ("TE", "receptions"), ("TE", "rec_yds")]
FLAGS = ["new_hc", "new_early", "mid", "turf", "turf_out"]


def context(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(C.TEAM_ALIASES)
    s = g["surface"].astype("string").str.strip().str.lower().replace({"": pd.NA})
    turf = s.map(lambda x: np.nan if pd.isna(x) else float(x not in GRASS))
    by_stadium = pd.Series(turf.values, index=g.index).groupby(g["stadium_id"]).transform(lambda x: x.fillna(x.mode().iloc[0] if x.notna().any() else np.nan))
    g["turf"] = turf.fillna(by_stadium).fillna(0.0)
    g["indoor_g"] = g["roof"].isin(["dome", "closed"]).astype(float)
    g["date"] = pd.to_datetime(g["gameday"])
    long = pd.concat([g[["game_id", "season", "week", "date", "turf", "indoor_g"]].assign(team=g.home_team, coach=g.home_coach),
                      g[["game_id", "season", "week", "date", "turf", "indoor_g"]].assign(team=g.away_team, coach=g.away_coach)])
    long = long.sort_values("date")
    first = long.groupby(["team", "season"])["coach"].first()
    start = [first.get((t, s)) for t, s in zip(long.team, long.season)]
    prev = [first.get((t, s - 1)) for t, s in zip(long.team, long.season)]
    long["new_hc"] = [float(p is not None and isinstance(p, str) and a != p) for a, p in zip(start, prev)]
    long["new_early"] = long.new_hc * (long.week <= 4)
    long["mid"] = (long.coach != pd.Series(start, index=long.index)).astype(float)
    long["turf_out"] = long.turf * (1 - long.indoor_g)
    return long[["game_id", "team"] + FLAGS]


def eligible(p: pd.DataFrame, grp: str) -> pd.Series:
    m = p["group"] == grp
    return m & (p["att_share"] > 0.5) if grp == "QB" else m


def evaluate(p: pd.DataFrame, fit_seasons, test_seasons, label: str, pairs) -> list[dict]:
    played = p["played"].fillna(False).astype(bool) & (p["active"] == 1)
    rows = []
    for grp, stat in pairs:
        col = stat + "_proj"
        if col not in p:
            continue
        d = p[played & eligible(p, grp) & p[col].notna() & (p[col] > 0)]
        y, x0 = d[stat].to_numpy(float), d[col].to_numpy(float)
        fit = d.season.isin(list(fit_seasons)).to_numpy()
        test = d.season.isin(list(test_seasons)).to_numpy()
        avg = d.loc[fit, stat].mean()
        g = pd.Series(d.game_id.to_numpy())
        for flag in FLAGS:
            f = d[flag].fillna(0).to_numpy(float)
            if f[fit].sum() < 30:
                continue
            for how in ("shift", "shrink"):
                xx = x0 * f if how == "shift" else f * (avg - x0)
                den = float((xx[fit] ** 2).sum())
                if den <= 0:
                    continue
                b = float((xx[fit] * (y - x0)[fit]).sum() / den)
                adj = x0 * np.exp(b * f) if how == "shift" else x0 + b * xx
                diff = np.abs(adj - y) - np.abs(x0 - y)
                r = {"projections": label, "group": grp, "stat": stat, "flag": flag, "how": how, "b": round(b, 4),
                     "flagged_share": round(float(f[test].mean()), 3), "n": int(test.sum()),
                     "mae": round(float(np.abs(x0 - y)[test].mean()), 3),
                     "fit_change": round(float(diff[fit].mean()), 4), "change": round(float(diff[test].mean()), 4),
                     "se": round(PB._se_by_game(pd.Series(diff[test]), g[test].reset_index(drop=True)), 4)}
                r["passes"] = bool(r["change"] < -2 * r["se"] and r["fit_change"] <= 0)
                rows.append(r)
    return rows


def main():
    games = data.load_games()
    ctx = context(games)
    pall = load_proj(None).merge(ctx, on=["game_id", "team"], how="left")
    print(f"context matched on {pall.new_hc.notna().mean():.1%} of player-games", flush=True)
    res = {"chain": evaluate(pall, range(2013, C.HOLDOUT_FROM), range(C.HOLDOUT_FROM, 2027), "chain", PAIRS)}
    R = roles.load_all(games, verbose=False)
    ml = players_ml.questionable(players_ml.walk_forward(players_ml.add_features(pall, R), first=C.HOLDOUT_FROM),
                                 first=C.FIRST_TEST_SEASON)
    ml = ml[ml.season >= C.HOLDOUT_FROM]
    adopt = [(g, s) for s, gs in players_ml.ADOPT.items() for g in gs]
    res["new"] = evaluate(ml, range(C.HOLDOUT_FROM, 2022), range(2022, 2027), "New", [p for p in PAIRS if p in adopt])
    shown = {(r["group"], r["stat"]) for r in res["new"]}
    res["board"] = [r for r in res["new"]] + [r for r in res["chain"] if (r["group"], r["stat"]) not in shown]
    res["passing_on_board"] = [r for r in res["board"] if r["passes"]]
    OUT.write_text(json.dumps(res, indent=1, default=float))
    t = pd.DataFrame(res["board"])
    pd.set_option("display.width", 220)
    print(t.sort_values("change").to_string(index=False))
    print("passing:", res["passing_on_board"])


if __name__ == "__main__":
    main()
