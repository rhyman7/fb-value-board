"""Keep every week's college predictions: frozen at kickoff, graded once played.
Same files and method as nflmodel/archive.py, under docs/cfb/archive/.

docs/cfb/archive/<season>_w<WW>.json   one file per week:
    games    board id -> the game's entry at the last refresh before kickoff
    players  board id -> that game's player projections at the same moment
    teams    board id -> each team's projected volume and regulars who missed its last game
    results  board id -> final score and each bet's result in units
    actuals  board id -> player id -> box score
docs/cfb/archive/index.json            the weeks on file, with each week's record

Grading covers the board's side of each spread and total, the flagged bets, and the
projection itself: winners picked and the average miss on margin and total, next to the
line's. Weeks before the board went live are backfilled from the walk-forward replay
(the same model, fitted only on earlier games) and marked "backfilled".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel.archive import closing_value

from . import config as C

ARCH = C.DOCS / "archive"
BOX = ["pass_att", "completions", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td", "receptions", "rec_yds", "rec_td"]
GAME_KEYS = ["id", "game_id", "week", "date", "time", "away", "home", "away_name", "home_name", "away_record", "home_record",
             "neutral", "early", "weight", "roof", "wind", "away_qb", "home_qb", "market", "model", "proj", "win", "bets",
             "opening_spread", "opening_total", "line_source", "anchor", "sharp_open", "books"]


def _path(season, week, root=ARCH):
    return root / f"{season}_w{int(week):02d}.json"


def _load(path, season, week):
    if path.exists():
        return json.loads(path.read_text())
    return {"season": season, "week": int(week), "games": {}, "players": {}, "teams": {}, "results": {}, "actuals": {}}


def _save(path, d):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, separators=(",", ":")))


def snapshot(out: dict, games: pd.DataFrame, now=None, root=ARCH, backfilled: bool = False, slate: list | None = None, players=None) -> int:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    slate = slate if slate is not None else ((out.get("slate") or {}).get("games") or [])
    if not slate:
        return 0
    players = players if players is not None else (out.get("players") or {})
    pl, tm = players.get("players") or [], players.get("teams") or []
    ko = games.set_index("game_id")["kickoff"]
    when = out["meta"].get("generated")
    files, n = {}, 0
    for g in slate:
        gid = g["game_id"]
        if gid not in ko.index:
            continue
        k = pd.Timestamp(ko[gid])
        if not backfilled and k <= now:
            continue
        season = int(str(g["id"])[:4])
        path = _path(season, g["week"], root)
        d = files.setdefault(path, _load(path, season, g["week"]))
        if backfilled and g["id"] in d["games"] and not d["games"][g["id"]].get("backfilled"):
            continue                                   # never overwrite a live snapshot with the replay
        d["games"][g["id"]] = {**{x: g.get(x) for x in GAME_KEYS}, "kickoff": k.strftime("%Y-%m-%dT%H:%MZ"),
                               "frozen_at": "replay" if backfilled else when, "backfilled": backfilled}
        gp = [p for p in pl if p["game"] == g["id"]]
        if gp or g["id"] not in d["players"]:
            d["players"][g["id"]] = gp
            d["teams"][g["id"]] = [t for t in tm if t["game"] == g["id"]]
        n += 1
    for path, d in files.items():
        _save(path, d)
    return n


def _bet_result(b, g, margin, total):
    from .bets import half
    mk, side = b["market"], b["side"]
    if mk == "Spread":
        diff = (margin - float(half(g["market"]["spread"]))) * (1 if side == "home" else -1)
    elif mk == "Total":
        diff = (total - float(half(g["market"]["total"]))) * (1 if side == "over" else -1)
    else:
        diff = margin * (1 if side == "home" else -1)
    if diff == 0:
        return "P", 0.0
    if diff < 0:
        return "L", -1.0
    o = float(b["odds"])
    return "W", round(o / 100 if o > 0 else 100 / -o, 4)


def grade(games: pd.DataFrame, pg: pd.DataFrame | None, root=ARCH, close: dict | None = None, curve=None) -> int:
    """close / curve: the sharp book's closing quotes and the line curve, for closing line value
    (nflmodel/archive.closing_value)."""
    if not root.exists():
        return 0
    close = close or {}
    sched = games.set_index("game_id")
    box = pg.set_index("game_id") if pg is not None and len(pg) else None
    n = 0
    for path in sorted(root.glob("*_w*.json")):
        d = json.loads(path.read_text())
        changed = False
        for key, g in d["games"].items():
            gid = g.get("game_id")
            if key in d["results"] or gid not in sched.index:
                continue
            r = sched.loc[gid]
            if pd.isna(r["home_score"]) or pd.isna(r["away_score"]):
                continue
            hs, as_ = float(r["home_score"]), float(r["away_score"])
            margin, total = hs - as_, hs + as_
            bets = []
            for b in g.get("bets") or []:
                res, units = _bet_result(b, g, margin, total)
                bets.append({"market": b["market"], "side": b["side"], "result": res, "units": units,
                             **closing_value(b, g, close.get(gid), curve)})
            d["results"][key] = {"home_score": hs, "away_score": as_, "bets": bets}
            if box is not None and gid in box.index:
                b = box.loc[[gid]]
                ids = {p["id"] for p in d["players"].get(key, [])}
                b = b[b["player_id"].astype(str).isin(ids)]
                d["actuals"][key] = {str(int(row.player_id)): {k: float(getattr(row, k)) for k in BOX} for row in b.itertuples()}
            changed, n = True, n + 1
        if changed:
            _save(path, d)
    return n


def summarize(d: dict) -> dict:
    rec = {"spread": [0, 0, 0], "total": [0, 0, 0], "bets": 0, "bet_wlp": [0, 0, 0], "units": 0.0, "staked": 0.0, "graded": 0,
           "clv_n": 0, "clv_sum": 0.0, "beat_close": 0,
           "winners": [0, 0], "line_winners": [0, 0], "miss": {"margin": 0.0, "line_margin": 0.0, "total": 0.0, "line_total": 0.0}}
    for key, g in d["games"].items():
        res = d["results"].get(key)
        if not res:
            continue
        rec["graded"] += 1
        byk = {(b["market"], b["side"]): b for b in res["bets"]}
        for mk, k in (("Spread", "spread"), ("Total", "total")):
            sides = [b for b in g.get("bets") or [] if b["market"] == mk]
            if sides:
                pick = max(sides, key=lambda b: b.get("prob") or 0)
                r = byk.get((mk, pick["side"]))
                if r:
                    rec[k]["WLP".index(r["result"])] += 1
        for b in g.get("bets") or []:
            if (b.get("stake") or 0) > 0:
                r = byk.get((b["market"], b["side"]))
                if r:
                    rec["bets"] += 1
                    rec["bet_wlp"]["WLP".index(r["result"])] += 1
                    rec["staked"] += b["stake"]
                    rec["units"] += b["stake"] * r["units"]
                    if r.get("clv") is not None:
                        rec["clv_n"] += 1
                        rec["clv_sum"] += r["clv"]
                        rec["beat_close"] += int(r["clv"] > 0)
        margin, total = res["home_score"] - res["away_score"], res["home_score"] + res["away_score"]
        if margin != 0:
            rec["winners"][0 if np.sign(g["proj"]["margin"]) == np.sign(margin) else 1] += 1
            rec["line_winners"][0 if np.sign(g["market"]["spread"]) == np.sign(margin) else 1] += 1
        rec["miss"]["margin"] += abs(g["proj"]["margin"] - margin)
        rec["miss"]["line_margin"] += abs(g["market"]["spread"] - margin)
        rec["miss"]["total"] += abs(g["proj"]["total"] - total)
        rec["miss"]["line_total"] += abs(g["market"]["total"] - total)
    if rec["graded"]:
        rec["miss"] = {k: round(v / rec["graded"], 2) for k, v in rec["miss"].items()}
    rec["units"], rec["staked"] = round(rec["units"], 4), round(rec["staked"], 4)
    rec["clv"] = round(rec["clv_sum"] / rec["clv_n"], 4) if rec["clv_n"] else None
    del rec["clv_sum"]
    rec["backfilled"] = bool(d["games"]) and all(g.get("backfilled") for g in d["games"].values())
    return rec


def write_index(root=ARCH) -> list:
    rows = []
    for path in sorted(root.glob("*_w*.json")):
        d = json.loads(path.read_text())
        rows.append({"file": path.name, "season": d["season"], "week": d["week"], "games": len(d["games"]),
                     "players": sum(len(v) for v in d["players"].values()), **summarize(d)})
    if rows:
        _save(root / "index.json", {"weeks": rows})
    return rows


def update(out, games, pg, now=None, root=ARCH, verbose=True, close=None, curve=None):
    saved = snapshot(out, games, now, root)
    graded = grade(games, pg, root, close, curve)
    write_index(root)
    if verbose:
        print(f"  archive: {saved} games saved before kickoff, {graded} newly graded")
