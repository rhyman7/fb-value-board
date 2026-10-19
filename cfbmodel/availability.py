"""Missing players, as far as public college data can tell.

College football has no public injury report, so a player's absence is read from box
scores: a regular who recorded no stat in his team's last game is presumed still out.
That is the only version of the signal a bettor has before kickoff.

For each team and game, per position group:
  core      players with a large share of the team's usage over its earlier games this
            season (rushing: share of carries, quarterbacks excluded; pass catchers: share
            of receptions), at least two appearances
  miss_*    the usage share of core players who did not appear in the team's previous game
  chg_*     that, minus the average of the same measure over the three games before
            (what the team's recent results, and so its ratings, already reflect)
  now_*     the share of core players who did not appear in this game. Known only after
            the game; reported as a ceiling on what better injury data could add, never
            used by the board.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .players_data import GROUP

CORE = {"rb": 0.20, "rec": 0.12}


def team_game_missing(pg: pd.DataFrame, info: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team) with miss_/chg_/now_ for rushing and pass catchers."""
    g = games[["game_id", "season", "order", "home_id", "away_id", "home_team", "away_team", "played"]]
    keep = ["game_id", "season", "order", "team_id", "team", "played"]
    tm = pd.concat([g.rename(columns={"home_id": "team_id", "home_team": "team"})[keep],
                    g.rename(columns={"away_id": "team_id", "away_team": "team"})[keep]])
    tm = tm[tm.team != "FCS"].sort_values(["team", "order"]).reset_index(drop=True)
    p = pg.merge(tm[["game_id", "team_id", "team", "order"]], on=["game_id", "team_id"], how="inner")
    pos = p["player_id"].map(info["position"]).map(GROUP)
    p["is_qb"] = (pos == "QB") | (p["pass_att"] >= 5)
    tot = p.groupby(["game_id", "team_id"])[["carries", "receptions"]].transform("sum")
    p["rb"] = np.where(~p["is_qb"] & (tot["carries"] > 0), p["carries"] / tot["carries"].replace(0, np.nan), 0.0)
    p["rec"] = np.where(tot["receptions"] > 0, p["receptions"] / tot["receptions"].replace(0, np.nan), 0.0)

    out = []
    for (team, season), t in tm.groupby(["team", "season"], sort=False):
        t = t.sort_values("order")
        gids = list(t["game_id"])
        done = dict(zip(t["game_id"], t["played"].astype(bool)))
        pt = p[(p["team"] == team) & p["game_id"].isin(gids)]
        share = {k: pt.pivot_table(index="player_id", columns="game_id", values=k, aggfunc="sum") for k in CORE}
        appeared = pt.pivot_table(index="player_id", columns="game_id", values="order", aggfunc="size").notna()
        for k in CORE:
            share[k] = share[k].reindex(columns=gids).reindex(appeared.index).fillna(0.0)
        appeared = appeared.reindex(columns=gids).fillna(False).astype(bool)
        hist = {k: [] for k in CORE}
        for i, gid in enumerate(gids):
            row = {"game_id": gid, "team": team}
            for k, thr in CORE.items():
                if i == 0:
                    row[f"miss_{k}"] = row[f"now_{k}"] = row[f"chg_{k}"] = 0.0
                    hist[k].append(0.0)
                    continue
                prev = [x for x in gids[:i] if done[x]]
                if not prev:
                    row[f"miss_{k}"] = row[f"now_{k}"] = row[f"chg_{k}"] = 0.0
                    hist[k].append(0.0)
                    continue
                n_app = appeared[prev].sum(axis=1)
                avg = share[k][prev].where(appeared[prev].to_numpy()).mean(axis=1).fillna(0.0)
                core = avg[(avg >= thr) & (n_app >= 2)]
                last = appeared[prev[-1]]
                miss = float(core[~last.reindex(core.index, fill_value=False).astype(bool).to_numpy()].sum())
                now = float(core[~appeared[gid].reindex(core.index, fill_value=False).astype(bool).to_numpy()].sum()) if done[gid] else np.nan
                base = float(np.mean(hist[k][-3:])) if hist[k] else 0.0
                row[f"miss_{k}"], row[f"now_{k}"], row[f"chg_{k}"] = miss, now, miss - base
                hist[k].append(miss)
            out.append(row)
    return pd.DataFrame(out)


def add_missing(frame: pd.DataFrame, miss: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away differences (d_*) and sums (s_*) of every missing-player measure."""
    f = frame.copy()
    cols = [c for c in miss.columns if c not in ("game_id", "team")]
    for side in ("home", "away"):
        m = miss.rename(columns={c: f"{side}_{c}" for c in cols}).rename(columns={"team": f"{side}_team"})
        f = f.merge(m, on=["game_id", f"{side}_team"], how="left")
    for c in cols:
        h, a = f[f"home_{c}"].fillna(0.0), f[f"away_{c}"].fillna(0.0)
        f[f"d_{c}"], f[f"s_{c}"] = h - a, h + a
    return f
