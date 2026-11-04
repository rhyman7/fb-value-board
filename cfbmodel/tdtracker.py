"""The college Likeliest TD Scorers tracker: each week's top 10, who scored, and the sportsbook price.

    python -m cfbmodel.tdtracker

The college weekly report (rhyman7/cfb, scripts/build_report.py) lists the players most likely to
score a rushing or receiving touchdown, ranked by the college board's anytime-TD chance. This keeps
the record of that list, the way nflmodel/tdtracker.py does for the NFL: per week, the top TOP
players under the report's own rules, whether each scored, and DraftKings' and FanDuel's
anytime-TD price before kickoff. Written to docs/feed/td_tracker_cfb.json, which the college report
reads for its tracker page.

Sources:
  chance   the board's td_any as frozen at kickoff (docs/cfb/archive/<season>_wNN.json players).
           Weeks whose games were all written by the board's walk-forward backfill (archive
           games' "backfilled") are marked "rebuilt"; the rest "shown".
  rules    the college report's (td_rows): 2+ games played, played in his team's last game,
           not listed out, a rushing or receiving TD already this season, RB/FB 9+ carries plus
           catches a game, WR 3+ catches, TE 2+, a QB only as his team's listed starter and only
           for his rushing TDs; from the box scores of earlier weeks
           (data/cfb/cache/player_games_<season>.csv, weeks from games_<season>.csv)
  result   the archive's box score for that game ("actuals"): a rushing or receiving TD
           (None until the game is on file)
  price    the last DraftKings / FanDuel anytime-TD snapshot before kickoff in the live college
           props (data/props/history_cfb_live/<season>/<event>.csv, from Nov 2, 2026); none before
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from nflmodel import config as NC, odds as O
from nflmodel.dashfeed import norm

from . import config as C

TOP = 10
OUT = NC.DOCS / "feed" / "td_tracker_cfb.json"
LIVE = NC.DATA / "props" / "history_cfb_live"
BOOKS = ("draftkings", "fanduel")


def _archive(season: int) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted((C.DOCS / "archive").glob(f"{season}_w*.json"))]


def _shown(arch: list[dict]) -> pd.DataFrame:
    rows = []
    for d in arch:
        games = d.get("games") or {}
        for g, ps in (d.get("players") or {}).items():
            gm = games.get(g, {})
            starters = {gm.get("away"): (gm.get("away_qb") or {}).get("name"), gm.get("home"): (gm.get("home_qb") or {}).get("name")}
            for x in ps:
                rows.append({"week": int(x.get("week") or d["week"]), "game": g, "espn": gm.get("game_id"),
                             "player_id": str(x["id"]), "name": x["name"], "pos": x["pos"], "team": x["team"], "opp": x["opp"],
                             "status": x.get("status"), "chance": x.get("td_any"),
                             "starter": starters.get(x["team"]), "backfilled": bool(gm.get("backfilled"))})
    return pd.DataFrame(rows)


def _box(season: int) -> pd.DataFrame:
    """Every player-game this season with its week and team abbreviation-free team id."""
    pg = pd.read_csv(C.CACHE / f"player_games_{season}.csv", dtype={"player_id": str})
    gw = pd.read_csv(C.CACHE / f"games_{season}.csv", usecols=["game_id", "week"])
    return pg.merge(gw, on="game_id", how="inner")


def _eligible(c: pd.DataFrame, pg: pd.DataFrame, week: int) -> pd.DataFrame:
    """The college report's rules, from box scores of earlier weeks."""
    e = pg[pg.week < week]
    if e.empty or c.empty:
        return c.iloc[0:0]
    h = e.groupby("player_id").agg(gp=("game_id", "nunique"), car=("carries", "mean"), rec=("receptions", "mean"),
                                   rtd=("rush_td", "sum"), ctd=("rec_td", "sum"), team_id=("team_id", "last"), last=("week", "max"))
    team_last = e.groupby("team_id").week.max()
    h["active"] = h["last"] == h["team_id"].map(team_last)
    x = c.merge(h, left_on="player_id", right_index=True, how="inner")
    x = x[(x.gp >= 2) & x.active & ~x.status.isin(["Out", "Doubtful", "IR", "PUP", "Suspended"])]
    tds = x.rtd + x.ctd
    keep = ((x.pos.isin(["RB", "FB"]) & (x.car + x.rec >= 9) & (tds > 0))
            | ((x.pos == "WR") & (x.rec >= 3) & (tds > 0))
            | ((x.pos == "TE") & (x.rec >= 2) & (tds > 0))
            | ((x.pos == "QB") & (x.name == x.starter) & (x.rtd > 0)))
    return x[keep]


def _prices(season: int, arch: list[dict]) -> dict:
    """(board game id, normalized name) -> {book: American price}: the anytime-TD 'yes' side in the
    last snapshot taken before kickoff."""
    files = [f for f in sorted((LIVE / str(season)).glob("*.csv")) if f.name != "fetched.csv"]
    if not files:
        return {}
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = s[(s.market == "player_anytime_td") & s.side.str.lower().isin(["yes", "over"]) & s.book.isin(BOOKS)]
    s = s[pd.to_datetime(s.snapshot, utc=True) < pd.to_datetime(s.commence, utc=True)]
    if s.empty:
        return {}
    from nflmodel.parlays import _cfb_games
    games = [g for d in arch for g in (d.get("games") or {}).values() if g.get("kickoff")]
    gid = _cfb_games({"slate": {"games": games}}, s)
    s = s[s.event_id.isin(gid)].assign(g=lambda t: t.event_id.map(gid), nn=lambda t: t.player.map(norm))
    out: dict = {}
    for r in s.sort_values("snapshot").itertuples():   # later snapshots overwrite earlier ones
        out.setdefault((r.g, r.nn), {})[r.book] = float(r.price)
    return out


def _result(arch: list[dict]) -> dict:
    """(board game id, player id) -> (rushing TDs, rushing + receiving TDs) for games on file."""
    out = {}
    for d in arch:
        for g, ps in (d.get("actuals") or {}).items():
            for pid, a in ps.items():
                r = float(a.get("rush_td") or 0)
                out[(g, str(pid))] = (r, r + float(a.get("rec_td") or 0))
    return out


def build(season: int | None = None) -> dict:
    if season is None:
        from .players_data import current_season
        season = current_season()
    arch = _archive(season)
    sh = _shown(arch)
    pg = _box(season)
    prices = _prices(season, arch)
    res = _result(arch)
    played = {g for d in arch for g in (d.get("actuals") or {})}
    weeks = []
    for w in sorted(set(sh.week)) if len(sh) else []:
        c = sh[(sh.week == w) & sh.chance.notna()]
        if c.empty:
            continue
        x = _eligible(c, pg, w).sort_values("chance", ascending=False).head(TOP)
        if x.empty:
            continue
        players = []
        for i, r in enumerate(x.itertuples(), 1):
            if r.game in played:
                rt, any_ = res.get((r.game, r.player_id), (0.0, 0.0))
                hit = bool((rt if r.pos == "QB" else any_) > 0)
            else:
                hit = None
            q = prices.get((r.game, norm(r.name)), {})
            best = max(q, key=lambda b: O.american_to_decimal(q[b])) if q else None
            players.append({"rank": i, "name": r.name, "pos": r.pos, "team": r.team, "opp": r.opp, "game_id": r.game,
                            "chance": round(float(r.chance), 3), "scored": hit,
                            "dk": q.get("draftkings"), "fd": q.get("fanduel"), "best": best,
                            "ret10": (round(10 * (float(O.american_to_decimal(q[best])) - 1), 2) if hit else -10.0)
                            if (best and hit is not None) else None})
        graded = [p for p in players if p["scored"] is not None]
        priced = [p for p in graded if p["ret10"] is not None]
        weeks.append({"week": int(w), "source": "rebuilt" if x.backfilled.all() else "shown", "players": players,
                      "complete": len(graded) == len(players) and len(players) > 0,
                      "scored": sum(p["scored"] for p in graded), "graded": len(graded),
                      "expected": round(sum(p["chance"] for p in graded), 2),
                      "bets": len(priced), "net10": round(sum(p["ret10"] for p in priced), 2)})
    done = [w for w in weeks if w["complete"]]
    allp = [p for w in done for p in w["players"]]
    tot = lambda ps: {"players": len(ps), "scored": sum(bool(p["scored"]) for p in ps), "expected": round(sum(p["chance"] for p in ps), 1),
                      "bets": sum(p["ret10"] is not None for p in ps), "net10": round(sum(p["ret10"] or 0 for p in ps), 2)}
    return {"league": "cfb", "season": int(season), "top": TOP, "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "weeks": weeks, "season_total": tot(allp),
            "spots_1_5": tot([p for p in allp if p["rank"] <= 5]), "spots_6_10": tot([p for p in allp if p["rank"] > 5])}


def write(season: int | None = None) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(season), separators=(",", ":"), default=float))
    return OUT


if __name__ == "__main__":
    p = write()
    d = json.loads(p.read_text())
    for w in d["weeks"]:
        print(w["week"], w["source"], f'{w["scored"]}/{w["graded"]}', "exp", w["expected"], "bets", w["bets"], "net $", w["net10"])
    print("season", d["season_total"], "1-5", d["spots_1_5"], "6-10", d["spots_6_10"])
