"""Correlations between game lines and player props, for same-game parlays.

propsim.py already fits one correlation per kind of prop pair (a quarterback's yards with his
top receiver's). A same-game parlay can also mix a side or a total with props, or a side with
the total, and those pairs had no correlation. This fits them the same way, so that the chance
both legs win, summed over past pairs of that kind, matches how often both did:

  game legs     each team's cover at the closing spread (pushes left out) and the over at the
                closing total; chance 0.5 each (the closing line is the market's coin flip)
  prop legs     propsim.legs(): each player's stat at its middle synthetic line, by role
  pair kinds    gl|<stat>@<role>|cover   the player's own team covers
                gl|<stat>@<role>|over    the game goes over
                gg|fav|over (by spread size: 0-3, 3.5-7, 7.5+)   the favourite covers and the game goes over

The other team's cover is the mirror image of the player's team's cover (flip the sign), and
an under is the mirror of an over; the parlay page applies both rules.

Fitted on the tuning seasons (2013-2018) and scored on the holdout (2019 on): predicted chance
both win, independent and with the correlation, against what happened. Writes
data/parlay_corr.json (fitted on every season through the last full one, for the board) and
experiments/out/parlay_corr.json (the report).

    python experiments/parlay_corr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import config as C, data, propsim as S, props as P  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "parlay_corr.json"
BOARD = C.ROOT / "data" / "parlay_corr.json"
TUNE = list(range(2013, C.HOLDOUT_FROM))


def game_legs(games: pd.DataFrame) -> pd.DataFrame:
    """Per game: home cover, away cover and over at the closing numbers (pushes dropped)."""
    g = games[games["result"].notna() & games["spread_line"].notna() & games["total_line"].notna()].copy()
    # spread_line is home-positive points the home team is favoured by; result is home margin
    cov = g["result"] - g["spread_line"]
    tot = g["total"] - g["total_line"]
    rows = []
    for side, sign, team in (("home", 1, g["home_team"]), ("away", -1, g["away_team"])):
        rows.append(pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "team": team,
                                  "cover": np.where(cov == 0, np.nan, (sign * cov > 0).astype(float)),
                                  "fav": (sign * g["spread_line"] > 0) | ((g["spread_line"] == 0) & (side == "home")),
                                  "spread": g["spread_line"].abs(),
                                  "over": np.where(tot == 0, np.nan, (tot > 0).astype(float))}))
    return pd.concat(rows, ignore_index=True)


def fit_kind(p_a, p_b, hit_a, hit_b):
    d = pd.DataFrame({"p_a": p_a, "p_b": p_b, "hit_a": hit_a, "hit_b": hit_b, "key": "k"}).dropna()
    r = S.fit_rhos(d)
    return r.get("k", [0.0, int(len(d))])


def prop_game_pairs(lg: pd.DataFrame, gl: pd.DataFrame) -> pd.DataFrame:
    j = lg.merge(gl[["game_id", "team", "cover", "over"]], on=["game_id", "team"], how="inner")
    rows = []
    for what in ("cover", "over"):
        x = j[j[what].notna()]
        rows.append(pd.DataFrame({"game_id": x["game_id"], "season": x["season"],
                                  "key": "gl|" + x["stat"] + "@" + x["role"] + "|" + what,
                                  "p_a": x["p"], "p_b": 0.5, "hit_a": x["hit"], "hit_b": x[what]}))
    return pd.concat(rows, ignore_index=True)


def bucket(s):
    return np.where(s <= 3, "0-3", np.where(s <= 7, "3.5-7", "7.5+"))


def game_game_pairs(gl: pd.DataFrame) -> pd.DataFrame:
    f = gl[gl["fav"] & gl["cover"].notna() & gl["over"].notna()]
    return pd.DataFrame({"game_id": f["game_id"], "season": f["season"], "key": "gg|fav|over|" + pd.Series(bucket(f["spread"]), index=f.index),
                         "p_a": 0.5, "p_b": 0.5, "hit_a": f["cover"], "hit_b": f["over"]})


def fit_all(pr: pd.DataFrame) -> dict:
    return S.fit_rhos(pr)


def score(pr: pd.DataFrame, rhos: dict) -> list[dict]:
    pr = pr.copy()
    pr["rho"] = [rhos.get(k, [0.0])[0] for k in pr["key"]]
    pr["p_ind"] = pr["p_a"] * pr["p_b"]
    pr["p_cop"] = S.joint_over(pr["p_a"].to_numpy(), pr["p_b"].to_numpy(), pr["rho"].to_numpy())
    pr["hit"] = pr["hit_a"] * pr["hit_b"]
    out = []
    for k, g in pr.groupby("key"):
        if len(g) < 200:
            continue
        a = g["hit"].mean()
        out.append({"key": k, "n": int(len(g)), "rho": round(float(g["rho"].iloc[0]), 3),
                    "independent": round(float(g["p_ind"].mean()), 4), "copula": round(float(g["p_cop"].mean()), 4),
                    "actual": round(float(a), 4), "se": round(float(np.sqrt(a * (1 - a) / len(g))), 4)})
    return sorted(out, key=lambda r: -abs(r["rho"]))


def main():
    proj = load_proj(None)
    games = data.load_games()
    gl = game_legs(games)
    last_full = int(games.loc[games["result"].notna() & (games["game_type"] == "SB"), "season"].max())

    def build(seasons, fit_before=None):
        d = proj[proj.season.isin(seasons)]
        m = P.PropModel(FAMILY).fit(proj[proj.season < fit_before] if fit_before else d)
        lg = S.legs(d, m).merge(proj[["game_id", "player_id", "season"]].drop_duplicates(), on=["game_id", "player_id"], how="left", suffixes=("_x", ""))
        if "season_x" in lg:
            lg = lg.drop(columns="season_x")
        return pd.concat([prop_game_pairs(lg, gl), game_game_pairs(gl[gl.season.isin(seasons)])], ignore_index=True)

    tune = build(TUNE)
    rh_t = fit_all(tune)
    hold_seasons = sorted(s for s in proj.season.unique() if s >= C.HOLDOUT_FROM)
    hold = build(hold_seasons, fit_before=C.HOLDOUT_FROM)
    rep = {"fitted_on": [TUNE[0], TUNE[-1]], "rhos_tuning": rh_t, "holdout": score(hold, rh_t)}

    every = build([s for s in proj.season.unique() if s <= last_full])
    rh_all = fit_all(every)
    BOARD.write_text(json.dumps({"fitted_on": [int(proj.season.min()), last_full], "shrink": S.SHRINK,
                                 "min_pairs": S.MIN_PAIRS, "pairs": rh_all}, indent=1))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"{'pair':<44}{'n':>7}{'rho':>7}{'indep':>8}{'copula':>8}{'actual':>8}")
    for r in rep["holdout"]:
        print(f"{r['key']:<44}{r['n']:>7}{r['rho']:>7.2f}{r['independent']:>8.3f}{r['copula']:>8.3f}{r['actual']:>8.3f} ±{r['se']:.3f}")
    print(f"wrote {BOARD} and {OUT}")


if __name__ == "__main__":
    main()
