"""Keep every week's predictions: frozen at kickoff, graded once played.

docs/archive/<season>_w<WW>.json   one file per week:
    games    game_id -> the board's game entry as it stood at the last refresh
             before kickoff (lines, projection, win chances, every priced bet)
    players  game_id -> that game's player projections at the same moment
    teams    game_id -> each team's volume and who was out
    results  game_id -> final score, and each bet's result in units
    actuals  game_id -> player_id -> box score
docs/archive/index.json            the weeks on file, with a short record each

Each refresh overwrites a game's entry until kickoff and never after, so what is
kept is what the board showed before the game. Grading fills in afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ARCH = C.DOCS / "archive"
ET = "America/New_York"
BOX = ["pass_att", "completions", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td",
       "targets", "receptions", "rec_yds", "rec_td"]
GAME_KEYS = ["id", "week", "date", "time", "away", "home", "away_name", "home_name", "away_record", "home_record",
             "neutral", "early", "weight", "roof", "wind", "away_qb", "home_qb", "market", "model", "proj", "win", "bets",
             "opening_spread"]


def kickoff(gameday, gametime) -> pd.Timestamp:
    """Kickoff in UTC (times in the schedule are Eastern)."""
    t = gametime if isinstance(gametime, str) and len(gametime) >= 4 else "13:00"
    return pd.Timestamp(f"{pd.Timestamp(gameday).date()} {t}").tz_localize(ET).tz_convert("UTC")


def _path(season: int, week: int, root: Path = ARCH) -> Path:
    return root / f"{season}_w{week:02d}.json"


def _load(path: Path, season: int, week: int) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"season": season, "week": week, "games": {}, "players": {}, "teams": {}, "results": {}, "actuals": {}}


def _save(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, separators=(",", ":")))


def snapshot(out: dict, games: pd.DataFrame, now: pd.Timestamp | None = None, root: Path = ARCH) -> int:
    """Save the board's entry for each game that has not kicked off. Returns the number saved."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    slate = (out.get("slate") or {}).get("games") or []
    if not slate:
        return 0
    season = int(out["meta"]["season"])
    when = out["meta"].get("generated")
    sched = games.set_index("game_id")
    pl = (out.get("players") or {}).get("players") or []
    tm = (out.get("players") or {}).get("teams") or []
    files: dict[Path, dict] = {}
    n = 0
    for g in slate:
        gid = g["id"]
        if gid not in sched.index:
            continue
        r = sched.loc[gid]
        if kickoff(r["gameday"], r["gametime"]) <= now:      # kicked off: keep what was frozen
            continue
        week = int(g["week"])
        path = _path(season, week, root)
        d = files.setdefault(path, _load(path, season, week))
        ko = kickoff(r["gameday"], r["gametime"]).strftime("%Y-%m-%dT%H:%MZ")
        d["games"][gid] = {**{k: g.get(k) for k in GAME_KEYS}, "kickoff": ko, "frozen_at": when}
        gp = [p for p in pl if p["game"] == gid]
        if gp or gid not in d["players"]:                     # keep earlier projections if this run had none
            d["players"][gid] = gp
            d["teams"][gid] = [t for t in tm if t["game"] == gid]
        n += 1
    for path, d in files.items():
        _save(path, d)
    return n


def _bet_result(b: dict, g: dict, margin: float, total: float) -> tuple[str, float]:
    """Win, loss or push, and units won on one unit risked."""
    mk, side = b["market"], b["side"]
    if mk == "Spread":
        line = float(g["market"]["spread"])                   # home favoured by this much
        diff = (margin - line) * (1 if side == "home" else -1)
    elif mk == "Total":
        diff = (total - float(g["market"]["total"])) * (1 if side == "over" else -1)
    else:                                                      # Moneyline
        diff = margin * (1 if side == "home" else -1)
    if diff == 0:
        return "P", 0.0
    if diff < 0:
        return "L", -1.0
    o = float(b["odds"])
    return "W", round(o / 100 if o > 0 else 100 / -o, 4)


def grade(games: pd.DataFrame, pg: pd.DataFrame | None, root: Path = ARCH) -> int:
    """Add final scores, bet results and player box scores to archived games now played."""
    if not root.exists():
        return 0
    sched = games.set_index("game_id")
    box = None
    if pg is not None and len(pg):
        box = pg.set_index("game_id")
    n = 0
    for path in sorted(root.glob("*_w*.json")):
        d = json.loads(path.read_text())
        changed = False
        for gid, g in d["games"].items():
            if gid in d["results"] or gid not in sched.index:
                continue
            r = sched.loc[gid]
            if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
                continue
            if box is not None and gid not in box.index:           # wait for the box score too
                continue
            hs, as_ = float(r["home_score"]), float(r["away_score"])
            margin, total = hs - as_, hs + as_
            bets = []
            for b in g.get("bets") or []:
                res, units = _bet_result(b, g, margin, total)
                bets.append({"market": b["market"], "side": b["side"], "result": res, "units": units})
            d["results"][gid] = {"home_score": hs, "away_score": as_, "bets": bets,
                                 "home_qb": r.get("home_qb_id") if isinstance(r.get("home_qb_id"), str) else None,
                                 "away_qb": r.get("away_qb_id") if isinstance(r.get("away_qb_id"), str) else None}
            if box is not None:
                b = box.loc[[gid]]
                ids = {p["id"] for p in d["players"].get(gid, [])}
                b = b[b["player_id"].isin(ids)]
                d["actuals"][gid] = {row.player_id: {k: float(getattr(row, k)) for k in BOX} for row in b.itertuples()}
            changed, n = True, n + 1
        if changed:
            _save(path, d)
    return n


def summarize(d: dict) -> dict:
    """A week's record: the board's side of each spread and total, the flagged bets, player misses."""
    rec = {"spread": [0, 0, 0], "total": [0, 0, 0], "bets": 0, "bet_wlp": [0, 0, 0], "units": 0.0, "staked": 0.0, "graded": 0}
    for gid, g in d["games"].items():
        res = d["results"].get(gid)
        if not res:
            continue
        rec["graded"] += 1
        byk = {(b["market"], b["side"]): b for b in res["bets"]}
        for mk, key in (("Spread", "spread"), ("Total", "total")):
            sides = [b for b in g.get("bets") or [] if b["market"] == mk]
            if not sides:
                continue
            pick = max(sides, key=lambda b: b.get("prob") or 0)        # the side the board favours
            r = byk.get((mk, pick["side"]))
            if r:
                rec[key]["WLP".index(r["result"])] += 1
        for b in g.get("bets") or []:
            if (b.get("stake") or 0) > 0:
                r = byk.get((b["market"], b["side"]))
                if r:
                    rec["bets"] += 1
                    rec["bet_wlp"]["WLP".index(r["result"])] += 1
                    rec["staked"] += b["stake"]
                    rec["units"] += b["stake"] * r["units"]
    rec["units"], rec["staked"] = round(rec["units"], 4), round(rec["staked"], 4)
    return rec


def write_index(root: Path = ARCH) -> list:
    rows = []
    for path in sorted(root.glob("*_w*.json")):
        d = json.loads(path.read_text())
        rows.append({"file": path.name, "season": d["season"], "week": d["week"], "games": len(d["games"]),
                     "players": sum(len(v) for v in d["players"].values()), **summarize(d)})
    if rows:
        _save(root / "index.json", {"weeks": rows})
    return rows


def update(out: dict, games: pd.DataFrame, pg: pd.DataFrame | None, now: pd.Timestamp | None = None,
           root: Path = ARCH, verbose: bool = True) -> None:
    saved = snapshot(out, games, now, root)
    graded = grade(games, pg, root)
    write_index(root)
    if verbose:
        print(f"  archive: {saved} games saved before kickoff, {graded} newly graded")
