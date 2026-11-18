"""Weekly Review, Players page: every projection against what the player did (Nov 18, 2026).

Writes `review/data/<league>/<season>_w<week>.players.json`: one row per player and stat
(QB passing and rushing yards; RB rushing and receiving; WR and TE receiving) with the board's
projection frozen at kickoff, the actual, and a status:

  played   counts on the scorecard
Left off the page and not counted (Ryan, Nov 18, 2026): a WR projected 20 yards or fewer, a TE
15 or fewer, a back's rushing 15 or fewer, a back's receiving under 10. Quarterbacks: every
projection is listed; one under COUNT_MIN yards (rushing, usually) is listed but not scored.
  left     left the game early: not counted against the model, shown on its own scorecard
  dnp      NFL: projected, no offensive snaps. Not counted. (No stat line but on the field
           counts as a real zero.)
  nostat   college: no box-score line and no target, so it can't be told whether he played.
           Not counted, and labelled. (A target without a catch counts as a real zero.)

Left early (Ryan's ask: these should not count against the model):
  * a starting quarterback who threw under 60% of his team's passes (the review's qb_exits);
  * NFL: offensive snap share under half the median of his earlier games this season (the
    floors' early-exit rule, nflmodel/floors.py; two earlier games minimum). The note says
    "injured" when the play-by-play names him as injured;
  * college (no public snap counts): touches (pass attempts + carries + targets) under 40% of
    the median of his earlier games (the floors' rule; two earlier games, median of 5+) AND no
    play-by-play involvement after halftime. An estimate, and labelled as one.

Met = actual >= projection. Display only.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

STATS = {"QB": ["pass_yds", "rush_yds"], "RB": ["rush_yds", "rec_yds"], "WR": ["rec_yds"], "TE": ["rec_yds"]}
COUNT_MIN = 10.0       # quarterbacks: a projection under this many yards is listed but not scored
# Ryan's minimums (Nov 18, 2026): projections at or under these are left off the page and not counted
# (a back's receiving yards: under 10). Quarterbacks are unchanged.
def on_page(pos: str, stat: str, proj: float) -> bool:
    if pos == "WR":
        return proj > 20
    if pos == "TE":
        return proj > 15
    if pos == "RB":
        return proj > 15 if stat == "rush_yds" else proj >= 10
    return True
SNAP_CUT, TOUCH_CUT = 0.5, 0.4


def _short(name: str) -> str:
    parts = str(name).split()
    return f"{parts[0][0]}.{' '.join(parts[1:])}" if len(parts) > 1 else str(name)


def _left_nfl(src, pl, week, p, gsis_snaps) -> dict | None:
    me = gsis_snaps.get(pl["id"])
    if me is None:
        return None
    cur = me[me["week"] == week]
    pre = me[(me["week"] < week) & (me["offense_pct"] > 0)]
    if not len(cur) or len(pre) < 2:
        return None
    now, med = float(cur.iloc[0]["offense_pct"]), float(pre["offense_pct"].median())
    if med < 0.25 or now >= SNAP_CUT * med:       # a regular (a quarter of the snaps or more) who played under half his usual
        return None
    hurt = p["desc"].fillna("").str.contains(re.escape(_short(pl["name"])) + r" was injured", regex=True).any()
    return {"why": "injured" if hurt else "snaps", "note": f"{round(100 * now)}% of snaps, usually {round(100 * med)}%" + (", injured" if hurt else "")}


def _left_cfb(src, pl, week, p, game_id) -> dict | None:
    pg = src.pg[src.season]
    try:
        pid = int(pl["id"])
    except (TypeError, ValueError):
        return None
    me = pg[pg["player_id"] == pid].merge(src.games[["game_id", "week"]], on="game_id")
    me = me.assign(touch=me["pass_att"].fillna(0) + me["carries"].fillna(0) + me["targets"].fillna(0))
    cur, pre = me[me["game_id"] == game_id], me[me["week"] < week]
    if not len(cur) or len(pre) < 2:
        return None
    now, med = float(cur.iloc[0]["touch"]), float(pre["touch"].median())
    if med < 5 or now >= TOUCH_CUT * med:
        return None
    if now == 0:
        return {"why": "none", "note": f"no touches (usually {med:g})"}
    ids = pd.concat([p[c].astype(str).str.replace(r"\.0$", "", regex=True) for c in
                     ("passer_player_id", "rusher_player_id", "receiver_player_id") if c in p], axis=1)
    mine = (ids == str(pid)).any(axis=1)
    per = pd.to_numeric(p.loc[mine, "period"], errors="coerce")
    if len(per) and per.max() >= 3:
        return None
    last = f"last touch in quarter {int(per.max())}" if len(per) and not np.isnan(per.max()) else "no touches recorded"
    return {"why": "touches", "note": f"{int(now)} touch{'' if int(now) == 1 else 'es'}, usually {med:g}; {last} (estimate)"}


def typical(league: str, A: dict) -> dict:
    """(player id, game, stat) -> the model's 50/50 yardage (nflmodel/typical.py) at the projection
    frozen at kickoff, priced as if he plays. Uses the widths as fitted today. {} if it can't load."""
    try:
        from nflmodel import typical as T
        players = [dict(p) for ps in (A.get("players") or {}).values() for p in ps]
        return T.medians(league, players, given_plays=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  typical-game numbers skipped ({type(exc).__name__}: {exc})")
        return {}


def build(src, A: dict, facts: dict, week: int, pbp_week: pd.DataFrame) -> dict:
    lg = src.league
    med = typical(lg, A)
    gsis_snaps = {}
    if lg == "nfl":
        sn = src.snaps[src.snaps["season"] == src.season].copy()
        sn["gsis"] = sn["pfr_player_id"].map(src.pfr2gsis)
        gsis_snaps = {k: v for k, v in sn.dropna(subset=["gsis"]).groupby("gsis")}
    rows = []
    for g in facts["games"]:
        gid = g["id"]
        key = gid if lg == "nfl" else int(A["games"][gid].get("game_id") or 0)
        p = pbp_week[pbp_week["game_id"] == key]
        exits = {q["qb"] for q in ((g.get("chance") or {}).get("qb_exits") or [])}
        exit_last = {re.split(r"[.\s]", e.strip())[-1] for e in exits}
        act = (A.get("actuals") or {}).get(gid, {})
        for pl in (A.get("players") or {}).get(gid, []):
            stats = STATS.get(pl["pos"])
            if not stats:
                continue
            x = act.get(pl["id"])
            status, note = "played", None
            if not x:
                # no stat line. NFL: snap counts say whether he played (then it is a real zero).
                # College: a target without a catch in the box score means he played; otherwise unknown.
                status, note = "dnp", (pl.get("status") or None)
                if lg == "nfl":
                    me = gsis_snaps.get(pl["id"])
                    cur = me[me["week"] == week] if me is not None else []
                    if len(cur) and float(cur.iloc[0]["offense_pct"]) > 0:
                        x = {}
                        left = _left_nfl(src, pl, week, p, gsis_snaps) if len(p) else None
                        status, note = ("left", left["note"]) if left else ("played", f"{round(100 * float(cur.iloc[0]['offense_pct']))}% of snaps, no stats")
                else:
                    try:
                        pg = src.pg[src.season]
                        row = pg[(pg["player_id"] == int(pl["id"])) & (pg["game_id"] == key)]
                        if len(row) and float(row.iloc[0][["pass_att", "carries", "targets"]].fillna(0).sum()) > 0:
                            x, status, note = {}, "played", "targeted, no stats"
                        else:
                            status = "nostat"
                    except (TypeError, ValueError):
                        status = "nostat"
            else:
                left = None
                if pl["pos"] == "QB" and pl["name"].split()[-1] in exit_last:
                    q = next(q for q in g["chance"]["qb_exits"] if re.split(r"[.\s]", q["qb"].strip())[-1] == pl["name"].split()[-1])
                    left = {"why": "injured" if q["why"] == "injured" else "qb", "note": f"left in quarter {q['qtr']}" + (", injured" if q["why"] == "injured" else "") + f"; threw {round(100 * q['share'])}% of the team's passes"}
                elif len(p):
                    try:
                        left = _left_nfl(src, pl, week, p, gsis_snaps) if lg == "nfl" else _left_cfb(src, pl, week, p, key)
                    except Exception:
                        left = None
                if left:
                    status, note = ("nostat" if left["why"] == "none" else "left"), left["note"]
            for st in stats:
                pj = (pl.get("proj") or {}).get(st)
                if pj is None or pj < 1 or not on_page(pl["pos"], st, float(pj)):
                    continue
                av = float((x or {}).get(st, 0.0) or 0.0)   # x == {} : played, nothing recorded
                rows.append({"n": pl["name"], "p": pl["pos"], "t": pl["team"], "o": g["home"] if pl["team"] == g["away"] else g["away"],
                             "g": g["slug"], "s": st, "pj": round(float(pj), 1), "a": av, "st": status,
                             "c": bool(pj >= COUNT_MIN), **({"note": note} if note else {}),
                             **({"md": med[(str(pl["id"]), gid, st)]} if (str(pl["id"]), gid, st) in med else {})})
    return {"league": lg, "season": facts["season"], "week": week, "count_min": COUNT_MIN, "card": scorecard(rows), "rows": rows}


def _tally(rs) -> dict:
    n = len(rs)
    met = sum(r["a"] >= r["pj"] for r in rs)
    tm = [r for r in rs if r.get("md") is not None]
    return {"n": n, "met": met, "short": n - met, "typ_n": len(tm), "typ_met": sum(r["a"] >= r["md"] for r in tm),
            "avg_miss": round(sum(r["a"] - r["pj"] for r in rs) / n, 1) if n else None,
            "avg_abs": round(sum(abs(r["a"] - r["pj"]) for r in rs) / n, 1) if n else None}


def scorecard(rows: list[dict]) -> dict:
    counted = [r for r in rows if r["st"] == "played" and r["c"]]
    left = [r for r in rows if r["st"] == "left" and r["c"]]
    dnp = [r for r in rows if r["st"] in ("dnp", "nostat") and r["c"]]
    return {"all": _tally(counted),
            "by_stat": {s: _tally([r for r in counted if r["s"] == s]) for s in ("pass_yds", "rush_yds", "rec_yds")},
            "by_pos": {p: _tally([r for r in counted if r["p"] == p]) for p in ("QB", "RB", "WR", "TE")},
            "left": {**_tally(left), "players": len({(r["n"], r["t"]) for r in left})},
            "dnp": {"n": len(dnp), "players": len({(r["n"], r["t"]) for r in dnp})}}
