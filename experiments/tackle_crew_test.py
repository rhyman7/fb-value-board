"""Do home stat crews credit tackles differently, and does that help project a player's tackles?

    python experiments/tackle_crew_test.py <dir with stats_player_week_<season>.csv> <dir with pbp p<season>.parquet>

Tackles are recorded by each home team's stat crew. Per game and defense (regular season,
2016-2025, neutral sites left out): solo + assisted tackles per opponent offensive play, as a log
ratio to that season's league rate. A stadium's crew factor uses only the VISITING defenses'
tackles there (the visitors change every week, so what stays is the crew).

1. Is it a stable trait? The crew factor in one season against the next.
2. Does it help? Defenders (LB/DB/DL) averaging 3+ tackles with 4+ games so far: projection =
   season-to-date tackles per game, times exp(b x (this game's crew factor - his average crew
   factor so far)), the crew factor from the two previous seasons shrunk by 8 games. Average miss
   against the projection alone, 2018-2025, standard error clustered by game.

Rule, set before running (Nov 2, 2026): worth pricing tackle props on only if (2) lowers the miss
by more than two standard errors. Prints its results (FEATURES.md "Niche props" has them).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(stats_dir: Path, pbp_dir: Path):
    fr, plays = [], []
    for s in range(2016, 2026):
        d = pd.read_csv(stats_dir / f"s{s}.csv", usecols=["season", "week", "season_type", "game_id", "team", "opponent_team",
                                                          "def_tackles_solo", "def_tackle_assists"])
        d = d[d.season_type == "REG"]
        fr.append(d.groupby(["season", "week", "game_id", "team", "opponent_team"])[["def_tackles_solo", "def_tackle_assists"]].sum().reset_index())
        p = pd.read_parquet(pbp_dir / f"p{s}.parquet", columns=["game_id", "posteam", "home_team", "location", "play_type"])
        p = p[p.play_type.isin(["pass", "run"])]
        x = p.groupby(["game_id", "posteam"]).size().rename("opp_plays").reset_index()
        plays.append(x.merge(p.groupby("game_id").agg(home=("home_team", "first"), loc=("location", "first")).reset_index(), on="game_id"))
    t = pd.concat(fr)
    t["tk"] = t.def_tackles_solo + t.def_tackle_assists
    t = t.merge(pd.concat(plays).rename(columns={"posteam": "opponent_team"}), on=["game_id", "opponent_team"])
    t = t[t["loc"] != "Neutral"]
    t["lr"] = np.log(t.tk / t.opp_plays / (t.tk / t.opp_plays).groupby(t.season).transform("mean"))
    t["crew"] = t.home
    v = t[t.team != t.home]
    cf = v.groupby(["crew", "season"]).lr.mean().unstack()
    a = pd.concat([cf[s] for s in range(2016, 2025)])
    b = pd.concat([cf[s + 1] for s in range(2016, 2025)])
    print("crew factor, one season vs the next: corr", round(float(np.corrcoef(a, b)[0, 1]), 3), "n", len(a))
    print("2023-25 average crew factor, lowest/highest:", np.exp(cf[[2023, 2024, 2025]].mean(axis=1)).round(3).agg(["min", "max"]).to_dict())

    cs = v.groupby(["crew", "season"]).lr.agg(["mean", "size"]).reset_index()
    def prior(c, s):
        x = cs[(cs.crew == c) & cs.season.between(s - 2, s - 1)]
        n = x["size"].sum()
        return float((x["mean"] * x["size"]).sum() / (n + 8)) if n else 0.0
    crew = {(c, s): prior(c, s) for c in t.crew.unique() for s in range(2018, 2026)}
    pl = []
    for s in range(2018, 2026):
        d = pd.read_csv(stats_dir / f"s{s}.csv", usecols=["player_id", "season", "week", "season_type", "game_id", "position_group",
                                                          "def_tackles_solo", "def_tackle_assists"])
        pl.append(d[(d.season_type == "REG") & d.position_group.isin(["LB", "DB", "DL"])])
    p = pd.concat(pl)
    p["tk"] = p.def_tackles_solo.fillna(0) + p.def_tackle_assists.fillna(0)
    p = p.merge(t[["game_id", "crew"]].drop_duplicates("game_id"), on="game_id")
    p["cf"] = [crew.get((c, s), 0.0) for c, s in zip(p.crew, p.season)]
    p = p.sort_values(["player_id", "season", "week"])
    g = p.groupby(["player_id", "season"])
    p["n_prev"] = g.cumcount()
    p["avg_prev"] = g.tk.transform(lambda x: x.shift().expanding().mean())
    p["cf_prev"] = g.cf.transform(lambda x: x.shift().expanding().mean())
    x = p[(p.n_prev >= 4) & (p.avg_prev >= 3)]
    base = x.avg_prev
    for bb in (0.5, 1.0, 1.5):
        d = (x.avg_prev * np.exp(bb * (x.cf - x.cf_prev)) - x.tk).abs() - (base - x.tk).abs()
        gs = d.groupby(x.game_id).sum()
        print(f"strength {bb}: n {len(x)}, miss {(base - x.tk).abs().mean():.3f}, change {d.mean():+.4f} ± {gs.std(ddof=1) * np.sqrt(len(gs)) / len(d):.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
