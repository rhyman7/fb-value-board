"""Shared loaders for the Nov 19, 2026 book-consistency tests (stored DK/FD history, results cache)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from nflmodel import config as C, odds as O, propbackfill as B, propmarket as PM  # noqa: E402

MAIN = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds"}
ALT = {k + "_alternate": v for k, v in MAIN.items()}
BOOKS = ("draftkings", "fanduel")


def results(season):
    pg = pd.read_csv(C.CACHE / f"player_games_{season}.csv", dtype={"player_id": str})
    info = pd.read_csv(C.CACHE / f"player_info_{season}.csv", dtype={"player_id": str})
    pg = pg.merge(info[["player_id", "name", "position"]], on="player_id", how="left")
    pg["nname"] = pg["name"].map(PM.norm_name)
    pg = pg[~pg.duplicated(["game_id", "nname"], keep=False)]
    return pg


def mains(season, kind):
    """One two-sided main line per game x player x stat x book, with the power-method fair over chance."""
    f = B.load(season, kind)
    m = f[f["market"].isin(MAIN) & f["book"].isin(BOOKS)].copy()
    m["stat"], m["nname"], m["side"] = m["market"].map(MAIN), m["player"].map(PM.norm_name), m["side"].str.lower()
    m = m.sort_values("updated")
    w = m.pivot_table(index=["game_id", "nname", "stat", "book", "line"], columns="side", values="price", aggfunc="last").reset_index()
    w = w.dropna(subset=["over", "under"])
    w["p_over"] = O.no_vig_power(w["over"].to_numpy(), w["under"].to_numpy())[0]
    w = w.assign(d=(w["p_over"] - 0.5).abs()).sort_values("d").drop_duplicates(["game_id", "nname", "stat", "book"]).drop(columns="d")
    w["season"] = season
    return w


def alts(season, kind):
    f = B.load(season, kind)
    m = f[f["market"].isin(ALT) & f["book"].isin(BOOKS) & (f["side"].str.lower() == "over")].copy()
    m["stat"], m["nname"] = m["market"].map(ALT), m["player"].map(PM.norm_name)
    m = m.sort_values("updated").drop_duplicates(["game_id", "nname", "stat", "book", "line"], keep="last")
    m["season"] = season
    return m[["season", "game_id", "nname", "stat", "book", "line", "price"]]


def with_results(w, season):
    pg = results(season)
    cols = ["game_id", "nname", "player_id", "posteam", "week", "pass_yds", "rush_yds", "rec_yds", "pass_att", "position"]
    x = w.merge(pg[cols], on=["game_id", "nname"], how="inner")
    x["y"] = np.select([x["stat"] == "pass_yds", x["stat"] == "rush_yds"], [x["pass_yds"], x["rush_yds"]], x["rec_yds"])
    return x


def dec(american):
    return np.asarray(O.american_to_decimal(np.asarray(american, float)), float)


def mean_se(v, cl):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"n": 0}
    s = pd.Series(v).groupby(np.asarray(cl)).sum()
    se = float(s.std(ddof=1) * np.sqrt(len(s)) / len(v)) if len(s) > 1 else float("nan")
    return {"n": int(len(v)), "games": int(len(s)), "mean": round(float(v.mean()), 4), "se": round(se, 4)}
