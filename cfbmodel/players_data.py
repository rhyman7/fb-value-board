"""College player box scores, cached per season in data/cfb/cache/.

player_games_<s>.csv   one row per player per game with any stat: official box-score
                       passing, rushing and receiving (college box scores count sacks as
                       quarterback runs, and that is how props settle), tackles, plus
                       targets and red-zone usage from play-by-play
player_info_<s>.csv    name and position (ESPN game rosters, most common listing that season)

Players are keyed by ESPN athlete id, the same id the play-by-play uses for passers.
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from . import config as C
from .data import _retry, current_season

FIRST = 2014
ESPN_POS = {1: "WR", 4: "C", 7: "TE", 8: "QB", 9: "RB", 10: "FB", 12: "NT", 22: "PK", 23: "P",
            29: "CB", 30: "LB", 31: "DE", 32: "DT", 35: "DB", 36: "S", 37: "DL", 45: "OL",
            46: "OT", 73: "G", 76: "PR", 78: "LS", 264: "EDGE"}
GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
DEF_POS = {"CB", "LB", "DE", "DT", "NT", "DB", "S", "DL", "EDGE"}
PBP_COLS = ["game_id", "pos_team_id", "pass_attempt", "rush", "pass", "kneel_down", "penalty_no_play", "sack",
            "receiver_player_id", "rusher_player_id", "start.yardsToEndzone"]


def _read(url: str, columns=None) -> pd.DataFrame:
    def go():
        req = urllib.request.Request(url, headers={"User-Agent": "cfb-model"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return pd.read_parquet(io.BytesIO(r.read()), columns=columns)
    return _retry(go)


def _num(x):
    return pd.to_numeric(x, errors="coerce").fillna(0.0)


def _reduce(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = _read(f"{C.BASE}/player_box/parquet/player_box_{season}.parquet")
    b = b[b["athlete_id"].notna() & b["team_id"].notna()].copy()
    b["athlete_id"] = pd.to_numeric(b["athlete_id"], errors="coerce")
    b = b[b["athlete_id"].notna()]
    b["athlete_id"] = b["athlete_id"].astype(np.int64)
    col = lambda c: _num(b[c]) if c in b else pd.Series(0.0, index=b.index)
    ca = b["completions/passingAttempts"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
    cat = b["category"]
    f = pd.DataFrame({
        "game_id": b["game_id"].astype(np.int64), "team_id": b["team_id"].astype(np.int64), "player_id": b["athlete_id"],
        "name": b["athlete_name"],
        "completions": np.where(cat == "passing", pd.to_numeric(ca[0], errors="coerce"), 0),
        "pass_att": np.where(cat == "passing", pd.to_numeric(ca[1], errors="coerce"), 0),
        "pass_yds": np.where(cat == "passing", col("passingYards"), 0),
        "pass_td": np.where(cat == "passing", col("passingTouchdowns"), 0),
        "ints": np.where(cat == "passing", col("interceptions"), 0),
        "carries": np.where(cat == "rushing", col("rushingAttempts"), 0),
        "rush_yds": np.where(cat == "rushing", col("rushingYards"), 0),
        "rush_td": np.where(cat == "rushing", col("rushingTouchdowns"), 0),
        "receptions": np.where(cat == "receiving", col("receptions"), 0),
        "rec_yds": np.where(cat == "receiving", col("receivingYards"), 0),
        "rec_td": np.where(cat == "receiving", col("receivingTouchdowns"), 0),
        "tackles": np.where(cat == "defensive", col("totalTackles"), 0),
        "sacks_made": np.where(cat == "defensive", col("sacks"), 0),
    })
    f = f.fillna(0)
    names = f.groupby("player_id")["name"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0])
    pg = f.drop(columns="name").groupby(["game_id", "team_id", "player_id"], as_index=False).sum()

    # targets and red-zone usage from play-by-play
    import pyarrow.parquet as pq
    C.RAW.mkdir(parents=True, exist_ok=True)
    path = C.RAW / f"pbp_pl_{season}.parquet"

    def dl():
        req = urllib.request.Request(f"{C.BASE}/{C.FILES['pbp'].format(s=season)}", headers={"User-Agent": "cfb-model"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
        pq.ParquetFile(path)       # raises if the download was cut off
    _retry(dl)
    have = set(pq.ParquetFile(path).schema.names)
    p = pd.read_parquet(path, columns=[c for c in PBP_COLS if c in have])
    path.unlink()
    flag = lambda c: p[c].fillna(False).astype(bool) if c in p else pd.Series(False, index=p.index)
    ok = ~flag("penalty_no_play") & ~flag("kneel_down") & p["pos_team_id"].notna()
    yte = pd.to_numeric(p.get("start.yardsToEndzone"), errors="coerce")
    p["rz"], p["gl"] = (yte <= 20).astype(int), (yte <= 5).astype(int)
    t = p[ok & flag("pass_attempt") & ~flag("sack") & p["receiver_player_id"].notna()]
    tgt = t.groupby([t["game_id"].astype(np.int64), t["pos_team_id"].astype(np.int64), pd.to_numeric(t["receiver_player_id"], errors="coerce")]).agg(
        targets=("rz", "size"), rz_targets=("rz", "sum"))
    r = p[ok & flag("rush") & ~flag("pass") & p["rusher_player_id"].notna()]
    car = r.groupby([r["game_id"].astype(np.int64), r["pos_team_id"].astype(np.int64), pd.to_numeric(r["rusher_player_id"], errors="coerce")]).agg(
        rz_carries=("rz", "sum"), gl_carries=("gl", "sum"))
    use = pd.concat([tgt, car], axis=1).fillna(0)
    use.index.names = ["game_id", "team_id", "player_id"]
    use = use.reset_index()
    use = use[use["player_id"].notna()]
    use["player_id"] = use["player_id"].astype(np.int64)
    pg = pg.merge(use, on=["game_id", "team_id", "player_id"], how="outer").fillna(0)
    pg.insert(0, "season", season)

    # positions from ESPN game rosters
    gr = _read(f"{C.BASE}/game_rosters/parquet/game_rosters_{season}.parquet", columns=["athlete_id", "position_href", "full_name"])
    gr["pos"] = pd.to_numeric(gr["position_href"].astype(str).str.extract(r"positions/(\d+)")[0], errors="coerce").map(ESPN_POS)
    gr["athlete_id"] = pd.to_numeric(gr["athlete_id"], errors="coerce")
    gr = gr[gr["athlete_id"].notna() & gr["pos"].notna()]
    pos = gr.groupby(gr["athlete_id"].astype(np.int64))["pos"].agg(lambda s: s.mode().iat[0])
    rname = gr.groupby(gr["athlete_id"].astype(np.int64))["full_name"].first()
    ids = pd.Series(pg["player_id"].unique())
    info = pd.DataFrame({"season": season, "player_id": ids,
                         "name": ids.map(names).fillna(ids.map(rname)),
                         "position": ids.map(pos)})
    return pg, info


def load_player_games(refresh_current: bool = False, verbose: bool = True, last_season: int | None = None):
    now = last_season or current_season()
    pgs, infos = [], []
    C.CACHE.mkdir(parents=True, exist_ok=True)
    for season in range(FIRST, now + 1):
        pp, ip = C.CACHE / f"player_games_{season}.csv", C.CACHE / f"player_info_{season}.csv"
        if (season == now and refresh_current) or not (pp.exists() and ip.exists()):
            if verbose:
                print(f"  player box scores {season} ...", flush=True)
            pg, info = _reduce(season)
            pg.round(2).to_csv(pp, index=False)
            info.to_csv(ip, index=False)
        pgs.append(pd.read_csv(pp))
        infos.append(pd.read_csv(ip))
    pg = pd.concat(pgs, ignore_index=True)
    info = pd.concat(infos, ignore_index=True).sort_values("season").groupby("player_id").tail(1).set_index("player_id")
    return pg, info
