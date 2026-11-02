"""The Likeliest TD Scorers tracker: each week's top 10, who scored, and the sportsbook price.

    python -m nflmodel.tdtracker

The weekly report (rhyman7/nfl, scripts/build_report.py) lists the players most likely to score a
rushing or receiving touchdown, ranked by the board's anytime-TD chance. This keeps the record of
that list: per week, the top TOP players under the report's own rules, whether each scored, and
DraftKings' and FanDuel's anytime-TD price before kickoff. Written to docs/feed/td_tracker_nfl.json,
which the report reads for its tracker page.

Sources:
  chance   the board's td_any as shown before kickoff (docs/archive/<season>_wNN.json players);
           weeks before the archive kept players come from data/td_tracker/nfl_<season>_seed.csv
           (rebuilt the board's way, experiments/td_tracker_backfill.py)
  rules    the report's (td_rows): 2+ games played, a rushing or receiving TD already this
           season, RB/FB 10+ carries plus catches a game, WR 3.5+ catches, TE 3+, a QB only as
           his team's projected starter and only for his rushing TDs; from the box scores of
           earlier weeks (data/cache/player_games_<season>.csv)
  result   the same box scores: a rushing or receiving TD in that game (None until it is on file)
  price    the pricing snapshot before kickoff ("flag"): the live prop log
           (data/props/prop_log.csv) where it has the player, else the price backfill
           (data/props/history/<season>/<game>_flag.csv)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C, odds as O
from .dashfeed import norm

TOP = 10
OUT = C.DOCS / "feed" / "td_tracker_nfl.json"
SEED = C.DATA / "td_tracker"
BOOKS = ("draftkings", "fanduel")


def _shown(season: int) -> pd.DataFrame:
    rows = []
    for f in sorted((C.DOCS / "archive").glob(f"{season}_w*.json")):
        d = json.loads(f.read_text())
        for g, ps in (d.get("players") or {}).items():
            for x in ps:
                rows.append({"week": int(x.get("week") or d["week"]), "game_id": g, "player_id": x["id"], "name": x["name"],
                             "pos": x["pos"], "team": x["team"], "opp": x["opp"], "status": x.get("status"),
                             "chance": x.get("td_any"), "pass_att_proj": (x.get("proj") or {}).get("pass_att", 0)})
    return pd.DataFrame(rows)


def _eligible(c: pd.DataFrame, pg: pd.DataFrame, week: int) -> pd.DataFrame:
    h = pg[pg.week < week].groupby("player_id").agg(gp=("game_id", "nunique"), car=("carries", "mean"), rec=("receptions", "mean"),
                                                    rtd=("rush_td", "sum"), ctd=("rec_td", "sum"))
    x = c.merge(h, left_on="player_id", right_index=True, how="inner")
    x = x[(x.gp >= 2) & ~x.status.isin(["Out", "Doubtful", "IR", "PUP", "Suspended"])]
    qbs = x[x.pos == "QB"].sort_values("pass_att_proj", ascending=False).drop_duplicates(["game_id", "team"])
    keep = ((x.pos.isin(["RB", "FB"]) & (x.car + x.rec >= 10) & (x.rtd + x.ctd > 0))
            | ((x.pos == "WR") & (x.rec >= 3.5) & (x.rtd + x.ctd > 0))
            | ((x.pos == "TE") & (x.rec >= 3) & (x.rtd + x.ctd > 0))
            | (x.index.isin(qbs.index) & (x.rtd > 0)))
    return x[keep]


def _prices(season: int) -> dict:
    """(game_id, normalized name) -> {book: American price} for the anytime-TD 'yes' side."""
    out: dict = {}
    hist = C.DATA / "props" / "history" / str(season)
    for f in sorted(hist.glob("*_flag.csv")):
        try:
            s = pd.read_csv(f)
        except Exception:
            continue
        s = s[(s.market == "player_anytime_td") & s.side.str.lower().isin(["yes", "over"]) & s.book.isin(BOOKS)]
        gid = f.name[: -len("_flag.csv")]
        for r in s.sort_values("snapshot").itertuples():
            out.setdefault((gid, norm(r.player)), {})[r.book] = float(r.price)
    log = C.DATA / "props" / "prop_log.csv"
    if log.exists():
        p = pd.read_csv(log, usecols=["snapshot", "season", "game_id", "player", "market", "side", "books"])
        p = p[(p.season == season) & (p.market == "player_anytime_td") & (p.snapshot == "flag") & (p.side == "over")]
        for r in p.itertuples():
            try:
                b = json.loads(r.books)
            except Exception:
                continue
            q = {k: float(v[0]) for k, v in b.items() if k in BOOKS and v and v[0] is not None}
            if q:
                out[(r.game_id, norm(r.player))] = q
    return out


def build(season: int | None = None, pg: pd.DataFrame | None = None) -> dict:
    if pg is None:
        from .players_data import current_season
        season = season or current_season()
        pg = pd.read_csv(C.CACHE / f"player_games_{season}.csv")
    season = season or int(pg.season.max())
    pg = pg[pg.season == season]
    played = set(pg.game_id)
    scored = pg.assign(td=pg.rush_td.fillna(0) + pg.rec_td.fillna(0)).groupby(["game_id", "player_id"]).td.sum()
    rush_only = pg.groupby(["game_id", "player_id"]).rush_td.sum()
    sh = _shown(season)
    seed_path = SEED / f"nfl_{season}_seed.csv"
    seed = pd.read_csv(seed_path, dtype={"player_id": str}) if seed_path.exists() else pd.DataFrame()
    prices = _prices(season)
    weeks = []
    for w in sorted(set(sh.week) | set(seed.get("week", []))):
        if len(seed) and w in set(seed.week):
            x = seed[seed.week == w].rename(columns={"position": "pos"}).copy()
            x["source"] = "rebuilt"
        else:
            c = sh[(sh.week == w) & sh.chance.notna()]
            if c.empty:
                continue
            x = _eligible(c, pg, w).copy()
            x["source"] = "shown"
        x = x.sort_values("chance", ascending=False).head(TOP)
        players = []
        for i, r in enumerate(x.itertuples(), 1):
            if r.game_id in played:
                got = rush_only.get((r.game_id, r.player_id), 0) if r.pos == "QB" else scored.get((r.game_id, r.player_id), 0)
                res = bool(got > 0)
            else:
                res = None
            q = prices.get((r.game_id, norm(r.name)), {})
            best = max(q, key=lambda b: O.american_to_decimal(q[b])) if q else None
            players.append({"rank": i, "name": r.name, "pos": r.pos, "team": r.team, "opp": r.opp, "game_id": r.game_id,
                            "chance": round(float(r.chance), 3), "scored": res,
                            "dk": q.get("draftkings"), "fd": q.get("fanduel"), "best": best,
                            "ret10": (round(10 * (float(O.american_to_decimal(q[best])) - 1), 2) if res else -10.0)
                            if (best and res is not None) else None})
        graded = [p for p in players if p["scored"] is not None]
        priced = [p for p in graded if p["ret10"] is not None]
        weeks.append({"week": int(w), "source": x["source"].iloc[0] if len(x) else None, "players": players,
                      "complete": len(graded) == len(players) and len(players) > 0,
                      "scored": sum(p["scored"] for p in graded), "graded": len(graded),
                      "expected": round(sum(p["chance"] for p in graded), 2),
                      "bets": len(priced), "net10": round(sum(p["ret10"] for p in priced), 2)})
    done = [w for w in weeks if w["complete"]]
    allp = [p for w in done for p in w["players"]]
    tot = lambda ps: {"players": len(ps), "scored": sum(bool(p["scored"]) for p in ps), "expected": round(sum(p["chance"] for p in ps), 1),
                      "bets": sum(p["ret10"] is not None for p in ps), "net10": round(sum(p["ret10"] or 0 for p in ps), 2)}
    return {"league": "nfl", "season": int(season), "top": TOP, "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "weeks": weeks, "season_total": tot(allp),
            "spots_1_5": tot([p for p in allp if p["rank"] <= 5]), "spots_6_10": tot([p for p in allp if p["rank"] > 5])}


def write(season: int | None = None, pg: pd.DataFrame | None = None) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(season, pg), separators=(",", ":"), default=float))
    return OUT


if __name__ == "__main__":
    p = write()
    d = json.loads(p.read_text())
    for w in d["weeks"]:
        print(w["week"], w["source"], f'{w["scored"]}/{w["graded"]}', "exp", w["expected"], "bets", w["bets"], "net $", w["net10"])
    print("season", d["season_total"])
