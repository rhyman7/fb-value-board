"""Score the shadow test of the defensive inputs (nflmodel/shadow.py; FEATURES.md, "Shadow test").

    python experiments/shadow_score.py

For every logged game that has been played: the raw model's miss on the margin and the total,
current against shadow; for every logged player who played: receiving yards (non-quarterbacks)
and interceptions (quarterbacks with 10+ attempts). Shown for the pre-registered windows, the
rest of 2026 (week 9 on) and 2027, with paired standard errors (players clustered by game).
The shadow version replaces the board's only if it is lower in both windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import data, players_data as PD, shadow  # noqa: E402

WINDOWS = {"2026, week 9 on": lambda d: (d.season == 2026) & (d.week >= 9), "2027": lambda d: d.season == 2027}


def paired(cur, sh, cluster=None):
    d = (sh - cur).to_numpy(float)
    if not len(d):
        return np.nan, np.nan, 0
    if cluster is None:
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan, len(d)
    g = pd.DataFrame({"d": d, "c": np.asarray(cluster)}).groupby("c")["d"].agg(["sum", "size"])
    m = g["sum"].sum() / g["size"].sum()
    return m, np.sqrt(((g["sum"] - m * g["size"]) ** 2).sum()) / g["size"].sum(), len(d)


def main():
    games = data.load_all(refresh=False, verbose=False)[0]
    res = games.loc[games.played, ["game_id", "result", "total"]]
    if not shadow.GAMES_LOG.exists():
        print("no shadow log yet")
        return
    gl = pd.read_csv(shadow.GAMES_LOG).merge(res, on="game_id")
    for name, sel in WINDOWS.items():
        x = gl[sel(gl)]
        for what, actual in (("margin", "result"), ("total", "total")):
            cur, sh = (x[f"{what}_current"] - x[actual]).abs(), (x[f"{what}_shadow"] - x[actual]).abs()
            m, se, n = paired(cur, sh)
            print(f"{name:16s} {what:7s} n {n:4d}  current {cur.mean():.3f}  shadow {sh.mean():.3f}  change {m:+.4f} ± {se:.4f}")
    if shadow.PLAYERS_LOG.exists():
        pg, _ = PD.load_player_games(verbose=False)
        pl = pd.read_csv(shadow.PLAYERS_LOG, dtype={"player_id": str}).merge(pg[["game_id", "player_id", "rec_yds", "ints", "pass_att"]],
                                                                            on=["game_id", "player_id"])
        pl = pl.merge(games[["game_id", "season", "week"]], on="game_id")
        for name, sel in WINDOWS.items():
            x = pl[sel(pl)]
            for st, rows in (("rec_yds", x[x.group != "QB"]), ("ints", x[(x.group == "QB") & (x.pass_att >= 10)])):
                cur, sh = (rows[f"{st}_current"] - rows[st]).abs(), (rows[f"{st}_shadow"] - rows[st]).abs()
                m, se, n = paired(cur, sh, rows["game_id"])
                print(f"{name:16s} {st:7s} n {n:4d}  current {cur.mean():.3f}  shadow {sh.mean():.3f}  change {m:+.4f} ± {se:.4f}")


if __name__ == "__main__":
    main()
