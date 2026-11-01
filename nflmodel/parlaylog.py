"""The Parlays page's own record: what it suggested before kickoff, graded afterwards.

The page builds parlays in the browser from docs/parlays/<league>.json, so nothing it shows is
kept. This keeps a fixed, repeatable version of what it would have suggested, the way the
ledger keeps the Bet sheet's bets, so the tools can be judged by results and not by claims.

Every time the leg files are written (each board rebuild, each props run), for this week's
games still to kick off:

  leg   every leg with a DraftKings or FanDuel price that is worth at least break-even at its
        better book, on the chance its value is scored on (the books' own number for NFL props),
        with a chance of 35-95%: the pool the Leg finder ranks first
  p2    the Leg finder's best 2-leg parlay by value, different games, one book (DraftKings and
  p3    FanDuel each), and the best 3-leg one: the leg with the highest value per game, top 2 or 3

Each row is logged the first time it appears, at the price it had then, and never rewritten
(data/parlays/<league>_log.csv). A parlay whose legs' games have all finished is graded from the
board's frozen weekly archive (scores and box scores): a lost leg loses it, a push or a player
who did not play drops the leg (as the books settle it), and the payout is the remaining legs'
prices multiplied, one unit staked. data/parlays/<league>_results.csv keeps the grades;
docs/parlays/record_<league>.json the summary the page shows.

Rows are flat one-unit bets on every suggestion, not the stakes the page sizes: the record asks
whether the suggestions win, not how a bankroll would have done.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DIR = C.DATA / "parlays"
COLS = ["logged_at", "league", "season", "week", "kind", "key", "book", "price", "chance", "chance_v", "value", "ko", "legs"]
RCOLS = ["key", "league", "kind", "graded_at", "result", "units", "legs_won", "legs_lost", "legs_void"]
P_LO, P_HI = 0.35, 0.95


def _dec(a) -> float:
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def _amer(d: float) -> int:
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def _leg_key(l: dict) -> str:
    return "|".join(str(x) for x in (l["g"], l["m"], l.get("c", {}).get("tm") or l.get("c", {}).get("p") or "", l.get("s") or "",
                                       l.get("d") or l.get("c", {}).get("k") or "", l.get("L")))


def _slim(l: dict, book: str) -> dict:
    return {"g": l["g"], "m": l["m"], "lab": l["lab"], "L": l.get("L"), "d": l.get("d"), "s": l.get("s"),
            "tm": l.get("c", {}).get("tm") or l.get("tm"), "p": l.get("c", {}).get("p"), "k": l.get("c", {}).get("k"),
            "book": book, "price": int(l["bk"][book]), "w": l["w"], "wv": l.get("wv", l["w"])}


def candidates(data: dict, now: dt.datetime) -> list[dict]:
    """This week's legs still to kick off, with a price, chance and value at each book."""
    games = data["games"]
    weeks = [g.get("week") for g in games.values()
             if g.get("week") is not None and dt.datetime.fromisoformat(g["ko"].replace("Z", "+00:00")) > now]
    this_week = min(weeks) if weeks else None
    out = []
    for l in data["legs"]:
        g = games.get(l["g"])
        if not g or dt.datetime.fromisoformat(g["ko"].replace("Z", "+00:00")) <= now:
            continue
        if this_week is not None and g.get("week") != this_week:
            continue
        if not (P_LO <= l["w"] <= P_HI):
            continue
        for b, a in (l.get("bk") or {}).items():
            if b in ("dk", "fd") and a is not None:
                pv = l.get("wv", l["w"])
                out.append({"leg": l, "book": b, "mult": pv * _dec(a) + (l.get("h") or 0), "ko": g["ko"], "week": g.get("week")})
    return out


def snapshot(data: dict, league: str, now: dt.datetime | None = None, root: Path = DIR) -> int:
    """Log this run's suggestions that are not in the log yet. Returns rows added."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cand = candidates(data, now)
    if not cand:
        return 0
    rows = []
    stamp = now.strftime("%Y-%m-%d %H:%M")
    season = data.get("season")
    # single legs: the better book's price
    best = {}
    for c in cand:
        k = _leg_key(c["leg"])
        if k not in best or c["mult"] > best[k]["mult"]:
            best[k] = c
    for k, c in best.items():
        if c["mult"] < 1:
            continue
        l = _slim(c["leg"], c["book"])
        rows.append({"kind": "leg", "key": f"leg|{k}|{c['book']}", "book": c["book"], "price": l["price"], "chance": l["w"],
                     "chance_v": l["wv"], "value": round(c["mult"] - 1, 4), "ko": c["ko"], "week": c["week"], "legs": [l]})
    # the best 2- and 3-leg parlays at each book: the best leg per game, then the top games
    for b in ("dk", "fd"):
        per_game = {}
        for c in cand:
            if c["book"] != b:
                continue
            if c["leg"]["g"] not in per_game or c["mult"] > per_game[c["leg"]["g"]]["mult"]:
                per_game[c["leg"]["g"]] = c
        top = sorted(per_game.values(), key=lambda c: -c["mult"])
        for n in (2, 3):
            if len(top) < n:
                continue
            pick = top[:n]
            legs = [_slim(c["leg"], b) for c in pick]
            d = float(np.prod([_dec(l["price"]) for l in legs]))
            pv = float(np.prod([l["wv"] for l in legs]))
            mult = float(np.prod([c["mult"] for c in pick]))
            key = f"p{n}|{b}|" + "+".join(sorted(_leg_key(c["leg"]) for c in pick))
            rows.append({"kind": f"p{n}", "key": key, "book": b, "price": _amer(d), "chance": round(float(np.prod([l["w"] for l in legs])), 4),
                         "chance_v": round(pv, 4), "value": round(mult - 1, 4), "ko": min(c["ko"] for c in pick),
                         "week": pick[0]["week"], "legs": legs})
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{league}_log.csv"
    have = set(pd.read_csv(path, usecols=["key"])["key"]) if path.exists() else set()
    new = [r for r in rows if r["key"] not in have]
    if not new:
        return 0
    df = pd.DataFrame([{**r, "logged_at": stamp, "league": league, "season": season, "legs": json.dumps(r["legs"], separators=(",", ":"))}
                       for r in new])[COLS]
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(df)


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def _archive(league: str, season: int, week: int) -> dict | None:
    d = C.DOCS / ("archive" if league == "nfl" else "cfb/archive")
    path = d / f"{season}_w{int(week):02d}.json"
    return json.loads(path.read_text()) if path.exists() else None


def leg_result(l: dict, arch: dict) -> str | None:
    """W, L, P (push or void), or None while the game has no result."""
    res = (arch.get("results") or {}).get(l["g"])
    if not res or res.get("home_score") is None:
        return None
    games = arch.get("games")
    g = next((x for x in games if x["id"] == l["g"]), None) if isinstance(games, list) else (games or {}).get(l["g"])
    if g is None:
        return None
    hs, as_ = float(res["home_score"]), float(res["away_score"])
    if l["m"] in ("spread", "ml"):
        mine, theirs = (hs, as_) if l["tm"] == g["home"] else (as_, hs)
        x = mine - theirs + (float(l["L"]) if l["m"] == "spread" else 0.0)
        return "W" if x > 0 else "L" if x < 0 else "P"
    if l["m"] == "total":
        x = hs + as_ - float(l["L"])
        x = x if l["d"] == "over" or l.get("k") == "over" else -x
        return "W" if x > 0 else "L" if x < 0 else "P"
    if l["m"] == "prop":
        act = ((arch.get("actuals") or {}).get(l["g"]) or {}).get(str(l["p"]))
        if not act:
            return "P"                                  # did not play (or not in the box score): void
        if l["s"] == "anytime_td":
            y = float(act.get("rush_td", 0) or 0) + float(act.get("rec_td", 0) or 0)
        else:
            y = act.get(l["s"])
            if y is None:
                return "P"
            y = float(y)
        x = y - float(l["L"])
        x = x if l.get("d") != "under" else -x
        return "W" if x > 0 else "L" if x < 0 else "P"
    return None


def grade(league: str, now: dt.datetime | None = None, root: Path = DIR) -> int:
    """Grade logged rows whose games have all finished. Returns rows graded this run."""
    now = now or dt.datetime.now(dt.timezone.utc)
    path, rpath = root / f"{league}_log.csv", root / f"{league}_results.csv"
    if not path.exists():
        return 0
    log = pd.read_csv(path)
    done = set(pd.read_csv(rpath, usecols=["key"])["key"]) if rpath.exists() else set()
    todo = log[~log["key"].isin(done)]
    cache, out = {}, []
    for r in todo.itertuples(index=False):
        if pd.isna(r.week) or pd.isna(r.season):
            continue
        k = (int(r.season), int(r.week))
        if k not in cache:
            cache[k] = _archive(league, *k)
        arch = cache[k]
        if not arch:
            continue
        legs = json.loads(r.legs)
        res = [leg_result(l, arch) for l in legs]
        if any(x is None for x in res):
            continue
        won = [l for l, x in zip(legs, res) if x == "W"]
        lost = sum(x == "L" for x in res)
        void = sum(x == "P" for x in res)
        if lost:
            result, units = "L", -1.0
        elif not won:
            result, units = "P", 0.0
        else:
            result, units = "W", float(np.prod([_dec(l["price"]) for l in won])) - 1
        out.append({"key": r.key, "league": league, "kind": r.kind, "graded_at": now.strftime("%Y-%m-%d %H:%M"),
                    "result": result, "units": round(units, 4), "legs_won": len(won), "legs_lost": int(lost), "legs_void": int(void)})
    if out:
        pd.DataFrame(out)[RCOLS].to_csv(rpath, mode="a", header=not rpath.exists(), index=False)
    return len(out)


def summary(league: str, root: Path = DIR) -> dict:
    path, rpath = root / f"{league}_log.csv", root / f"{league}_results.csv"
    if not path.exists():
        return {"league": league, "kinds": {}, "by_week": [], "pending": 0}
    log = pd.read_csv(path)
    res = pd.read_csv(rpath) if rpath.exists() else pd.DataFrame(columns=RCOLS)
    m = log.merge(res[["key", "result", "units"]], on="key", how="left")

    def block(x):
        g = x[x["result"].notna()]
        n = len(g)
        units = float(g["units"].sum()) if n else 0.0
        bets = int((g["result"] != "P").sum())
        se = float(g["units"].std(ddof=1) / np.sqrt(bets)) if bets > 1 else None
        return {"logged": int(len(x)), "graded": n, "w": int((g["result"] == "W").sum()), "l": int((g["result"] == "L").sum()),
                "p": int((g["result"] == "P").sum()), "units": round(units, 2), "roi": round(units / bets, 4) if bets else None,
                "roi_se": round(se, 4) if se is not None else None, "claimed": round(float(g["value"].mean()), 4) if n else None,
                "chance": round(float(g["chance_v"].mean()), 4) if n else None,
                "hit": round(float((g["result"] == "W").sum() / bets), 4) if bets else None}
    kinds = {k: block(m[m["kind"] == k]) for k in ("leg", "p2", "p3")}
    weeks = [{"season": int(s), "week": int(w), **{k: block(x[x["kind"] == k]) for k in ("leg", "p2", "p3")}}
             for (s, w), x in m.groupby(["season", "week"])]
    return {"league": league, "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "kinds": kinds,
            "by_week": weeks, "pending": int(m["result"].isna().sum())}


def update(data: dict, league: str) -> dict:
    """Log, grade and write the summary (docs/parlays/record_<league>.json)."""
    n_new = snapshot(data, league)
    n_graded = grade(league)
    s = summary(league)
    out = C.DOCS / "parlays" / f"record_{league}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, separators=(",", ":")))
    print(f"  parlay record {league}: {n_new} logged, {n_graded} graded, {s['pending']} waiting")
    return s
