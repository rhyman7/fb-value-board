"""Does route participation improve the target projections? (A free test of what PFF would add.)

    python experiments/routes_test.py

Why. The player projections share a team's targets out by each player's recent target share.
How often a receiver is on the field when his team drops back to pass (his route share) is the
role measure scouts and books watch: a player whose routes jump this week usually sees his
targets follow. nflverse publishes who was on the field for every play (pbp_participation,
NGS 2016-2022, FTN 2023 on) but only after each season, so the board cannot use it live; PFF
sells routes run every week. This tests, on past seasons, whether it would help at all.

What. Per player and game: dropbacks on the field / team dropbacks (route share; for wide
receivers and tight ends on the field on a dropback is close to running a route). As of each
game: the share over the player's earlier games this season (recent games weighted more, as the
target share is) and in his last game. Then, for wide receivers, tight ends and backs who played:

    actual target share ~ the model's projected target share                    (current)
    actual target share ~ projected share + snap share (as of, last game)       (with snaps: free, weekly)
    actual target share ~ projected share + snaps + route share (as of, last game)
                          + targets per route (as of)                            (with routes: PFF, weekly)

fitted by least squares on the seasons before each test season (walk-forward), tested on
2019-2025. Reported: mean squared error of the target share and of the targets, overall and
for players whose route share moved in their last game (where it should matter most).
Writes experiments/out/routes_test.json.
"""
from __future__ import annotations

import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import config as C  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out"
SEASONS = list(range(2016, 2026))
TEST = list(range(2019, 2026))
PART = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{s}.parquet"
DIR = C.RAW / "part"


def _get(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fb-value-board"}), timeout=600) as r:
            path.write_bytes(r.read())
    return path


def routes(season: int) -> pd.DataFrame:
    """(game_id, player_id, team, routes, team_db) for one season."""
    cache = DIR / f"routes_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    part = pd.read_parquet(_get(PART.format(s=season), DIR / f"pp_{season}.parquet"),
                           columns=["nflverse_game_id", "play_id", "possession_team", "offense_players"])
    pbp_path = _get(C.PBP_URL.format(season=season), DIR / f"pbp_{season}.parquet")
    pbp = pd.read_parquet(pbp_path, columns=["game_id", "play_id", "posteam", "qb_dropback", "play_type"])
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["play_type"].isin(["pass", "run", "no_play"])][["game_id", "play_id", "posteam"]]
    m = part.merge(db, left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"])
    m = m[m["offense_players"].notna() & (m["offense_players"] != "")]
    team_db = m.groupby(["game_id", "posteam"]).size().rename("team_db").reset_index()
    ex = m[["game_id", "posteam", "offense_players"]].assign(player_id=m["offense_players"].str.split(";")).explode("player_id")
    r = ex.groupby(["game_id", "posteam", "player_id"]).size().rename("routes").reset_index()
    r = r.merge(team_db, on=["game_id", "posteam"]).rename(columns={"posteam": "team"})
    r.to_parquet(cache)
    pbp_path.unlink()                         # the play-by-play is large; the routes table is all that is kept
    return r


def asof(df: pd.DataFrame) -> pd.DataFrame:
    """Route share as of each game: the player's earlier games this season, recent ones weighted
    more (half-life about 4 games), and his last game."""
    df = df.sort_values(["player_id", "season", "t"]).copy()
    g = df.groupby(["player_id", "season"])
    df["rs"] = df["routes"] / df["team_db"]
    df["tpr"] = df["targets"] / df["routes"].replace(0, np.nan)
    df["rs_last"] = g["rs"].shift(1)
    df["rs_ew"] = g["rs"].transform(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean())
    df["ss_last"] = g["offense_pct"].shift(1)
    df["ss_ew"] = g["offense_pct"].transform(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean())
    df["tpr_ew"] = g["tpr"].transform(lambda s: s.shift(1).ewm(halflife=6, min_periods=1).mean())
    return df


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def main():
    proj = pd.read_pickle(C.RAW / "players_proj.pkl")              # the board's walk-forward player table
    parts = []
    for s in SEASONS:
        parts.append(routes(s))
        print(" ", s, flush=True)
    R = pd.concat(parts, ignore_index=True)
    d = proj[proj["played"].astype(bool) & proj["group"].isin(["WR", "TE", "RB"]) & (proj["season"].isin(SEASONS))]
    d = d.merge(R[["game_id", "player_id", "routes", "team_db"]], on=["game_id", "player_id"], how="left")
    d["routes"] = d["routes"].fillna(0)
    snaps = pd.concat([pd.read_csv(C.CACHE / f"snaps_{s}.csv", dtype={"player_id": str}) for s in SEASONS], ignore_index=True)
    d = d.merge(snaps[["game_id", "player_id", "offense_pct"]], on=["game_id", "player_id"], how="left")
    d["team_db"] = d.groupby(["game_id", "team"])["team_db"].transform("max")
    d = d[d["team_db"] > 0]
    d = asof(d)
    d["y"] = d["targets"] / d["team_att"].replace(0, np.nan)
    d = d.dropna(subset=["y", "tgt_share"])
    d = d[d["team_att"] >= 15]
    base_f = ["tgt_share"]
    for c in ("ss_ew", "ss_last", "rs_ew", "rs_last", "tpr_ew"):
        d[c + "_na"] = d[c].isna().astype(float)
        d[c] = d[c].fillna(d.groupby("group")[c].transform("mean"))
    snap_f = ["tgt_share", "ss_ew", "ss_last", "ss_ew_na", "ss_last_na"]
    new_f = snap_f + ["rs_ew", "rs_last", "tpr_ew", "rs_ew_na", "rs_last_na", "tpr_ew_na"]
    rep = {"by_season": {}, "by_group": {}}
    rows = []
    for s in TEST:
        tr, te = d[(d.season < s)], d[d.season == s]
        for grp in ("WR", "TE", "RB"):
            a, b = tr[tr.group == grp], te[te.group == grp]
            if len(a) < 500 or b.empty:
                continue
            p0 = pred(ols(a[base_f].to_numpy(), a["y"].to_numpy()), b[base_f].to_numpy())
            p1 = pred(ols(a[new_f].to_numpy(), a["y"].to_numpy()), b[new_f].to_numpy())
            ps = pred(ols(a[snap_f].to_numpy(), a["y"].to_numpy()), b[snap_f].to_numpy())
            rows.append(pd.DataFrame({"season": s, "group": grp, "y": b["y"].to_numpy(), "p0": p0, "p1": p1, "ps": ps,
                                      "raw": b["tgt_share"].to_numpy(), "att": b["team_att"].to_numpy(),
                                      "moved": (b["rs_last"] - b["rs_ew"]).abs().to_numpy() >= 0.15}))
    x = pd.concat(rows, ignore_index=True)
    se = lambda e: float((e ** 2).mean())
    for key, g in [("all", x)] + [(f"group {k}", v) for k, v in x.groupby("group")] + [("route share moved 15+ pts last game", x[x.moved])]:
        e0, e1, er, es = g.y - g.p0, g.y - g.p1, g.y - g.raw, g.y - g.ps
        t0, t1 = (g.y - g.p0) * g.att, (g.y - g.p1) * g.att
        rep["by_group"][key] = {"n": int(len(g)), "mse_share_model": round(se(er), 6), "mse_share_current": round(se(e0), 6),
                                "mse_share_snaps": round(se(es), 6), "gain_snaps": round(1 - se(es) / se(e0), 4),
                                "mse_share_routes": round(se(e1), 6), "improvement": round(1 - se(e1) / se(e0), 4),
                                "routes_over_snaps": round(1 - se(e1) / se(es), 4),
                                "rmse_targets_current": round(float(np.sqrt((t0 ** 2).mean())), 3),
                                "rmse_targets_routes": round(float(np.sqrt((t1 ** 2).mean())), 3)}
    for s, g in x.groupby("season"):
        rep["by_season"][int(s)] = {"snaps": round(1 - se(g.y - g.ps) / se(g.y - g.p0), 4), "snaps+routes": round(1 - se(g.y - g.p1) / se(g.y - g.p0), 4),
                                    "routes_over_snaps": round(1 - se(g.y - g.p1) / se(g.y - g.ps), 4)}
    OUT.mkdir(exist_ok=True)
    (OUT / "routes_test.json").write_text(json.dumps(rep, indent=1))
    print(pd.DataFrame(rep["by_group"]).T.to_string())
    print("improvement by season:", rep["by_season"])


if __name__ == "__main__":
    main()
