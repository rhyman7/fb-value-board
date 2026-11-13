"""Paper record of teams missing two or more of their usual five offensive linemen.

Why (Ryan, Nov 13, 2026; Project docs `claude/pff-unit-ratings-check-2026-11-11.md` and
`claude/plan-unit-tracker-this-week-2026-11-13.md`): in 2015-2025 a team that actually played
without two or more line regulars scored 0.9 points under its closing implied team total and
covered 45.9% (415 team-games). That count used who took snaps, which is only known after the
game. The injury-report version of the flag added nothing to the game model (FEATURES.md, "Two
or more offensive-line regulars out at bet time"). This module records what was knowable before
kickoff, so the lead can be judged on lines that could have been bet. It records and grades;
it never changes the board, the model or any bet.

Every board build, for each game that has not kicked off:

  a team is *tracked* once two or more of its five linemen in data/cache/ol_core_<season>.csv
  are out (injury report Out / Doubtful, a reserve list, off the roster, or a confirmed-news
  row in data/overrides.csv); once tracked it stays in the log until kickoff, so a team that
  drops back under two is recorded as dropping out. A row is appended whenever the linemen out
  or the team's lines change (data/ol_watch/nfl_log.csv, append-only, never edited). Lines are
  the board's anchor spread and total; `team_total` is the implied team total, (total - own
  spread) / 2. `late` is 1 when at least one of the linemen out came from the overrides file
  (news after the injury report).

After the game, each tracked team is graded once (data/ol_watch/nfl_results.csv) if it still
had two or more out on its last row before kickoff: points against the implied team total and
cover against the spread, both at the first row with two or more out (`flag_`) and at the last
row before kickoff (`last_`). The summary (data/ol_watch/nfl_record.json) gives the count,
average points against the team total with its standard error, and the cover record, for all
flagged teams and split by `late`.

Reading rule, written before any results: this is a paper record; nothing here reaches the Bet
sheet or the Best bets card. Rest-of-2026 will hold about 25 to 30 flags, too few to judge.
It is read after the 2027 season on the combined record, and becomes a model test only if the
flagged teams fall short of the flag-time team total by at least 2 standard errors; the test
then follows the usual pre-set rule. The five linemen are the game model's (snap share over
the team's recent games, `nflmodel/injuries.py`), not the PFF definition of the Nov 11 check.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DIR = C.DATA / "ol_watch"
MIN_OUT = 2
COLS = ["at", "season", "week", "game_id", "kickoff", "team", "opp", "side", "n_out", "late", "names", "reasons",
        "spread", "total", "team_total"]
RCOLS = ["key", "season", "week", "game_id", "team", "opp", "n_out", "late", "names", "flag_at",
         "flag_spread", "flag_team_total", "last_spread", "last_team_total", "points", "margin",
         "vs_flag_total", "vs_last_total", "cover_flag", "cover_last", "graded_at"]


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" else float(x)


def _own(spread, side: str):
    """The team's own spread (positive = underdog) from a home-favoured-positive line."""
    s = _num(spread)
    return None if s is None else (-s if side == "home" else s)


def _same(a, b) -> bool:
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 1e-9


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(path, dtype={"game_id": str, "names": str, "reasons": str}, keep_default_na=False,
                       na_values={"spread": [""], "total": [""], "team_total": [""]})


def core_path(season: int) -> Path:
    return C.CACHE / f"ol_core_{season}.csv"


def record(slate_games: list[dict], season: int, kickoffs: dict, core: pd.DataFrame, root: Path = DIR,
           now: dt.datetime | None = None) -> int:
    """Append a row for every tracked team whose linemen out or lines changed. Returns rows written.
    core: data/cache/ol_core_<season>.csv (game_id, team, name, player_id, share, out, reason)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    path = root / "nfl_log.csv"
    log = _read(path)
    last = {(r["game_id"], r["team"]): r for r in log.to_dict("records")}   # later rows overwrite earlier
    core = core.assign(game_id=core["game_id"].astype(str))
    out_by = {k: x[x["out"].astype(int) == 1] for k, x in core.groupby(["game_id", "team"])}
    rows = []
    for g in slate_games:
        gid = str(g["id"])
        ko = kickoffs.get(g["id"], kickoffs.get(gid))
        if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
            continue
        mk = g.get("market") or {}
        for side, other in (("home", "away"), ("away", "home")):
            team = g.get(side)
            if (gid, team) not in out_by:
                continue                                   # the lineman file has no row for this team's game
            x = out_by[(gid, team)]
            prev = last.get((gid, team))
            if len(x) < MIN_OUT and prev is None:
                continue
            own, total = _own(mk.get("spread"), side), _num(mk.get("total"))
            tt = None if own is None or total is None else round((total - own) / 2, 2)
            row = {"at": now.strftime("%Y-%m-%d %H:%M"), "season": season, "week": g.get("week"), "game_id": gid,
                   "kickoff": pd.Timestamp(ko).strftime("%Y-%m-%dT%H:%MZ"), "team": team, "opp": g.get(other), "side": side,
                   "n_out": int(len(x)), "late": int((x["reason"].astype(str) == "override").any()),
                   "names": " / ".join(sorted(x["name"].astype(str))), "reasons": " / ".join(sorted(set(x["reason"].astype(str)))),
                   "spread": own, "total": total, "team_total": tt}
            if prev is not None and str(prev.get("names", "")) == row["names"] and int(prev["late"]) == row["late"] \
                    and _same(prev.get("spread"), own) and _same(prev.get("total"), total):
                continue
            rows.append(row)
    if rows:
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows)[COLS].to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def _cover(margin: float, own) -> str | None:
    if own is None:
        return None
    x = margin + own
    return "W" if x > 0 else "L" if x < 0 else "P"


def grade(scores: dict, root: Path = DIR, now: dt.datetime | None = None) -> int:
    """Grade tracked teams whose game has a final score. scores: game id -> (home, away)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    path, rpath = root / "nfl_log.csv", root / "nfl_results.csv"
    if not path.exists():
        return 0
    log = _read(path)
    done = set(pd.read_csv(rpath, usecols=["key"])["key"]) if rpath.exists() else set()
    out = []
    for (gid, team), x in log.groupby(["game_id", "team"], sort=False):
        key = f"{gid}|{team}"
        if key in done or gid not in scores:
            continue
        hs, as_ = scores[gid]
        if hs is None or pd.isna(hs) or as_ is None or pd.isna(as_):
            continue
        r = x.iloc[-1]                                     # last row before kickoff
        if int(r["n_out"]) < MIN_OUT:
            continue                                       # back under two by kickoff: not a flag
        run = x.iloc[::-1]                                 # the unbroken run of 2+ rows ending at kickoff
        flag = r
        for _, y in run.iterrows():
            if int(y["n_out"]) < MIN_OUT:
                break
            flag = y
        home = r["side"] == "home"
        pts, opp = (float(hs), float(as_)) if home else (float(as_), float(hs))
        margin = pts - opp
        f_tt, l_tt = _num(flag["team_total"]), _num(r["team_total"])
        out.append({"key": key, "season": r["season"], "week": r["week"], "game_id": gid, "team": team, "opp": r["opp"],
                    "n_out": int(r["n_out"]), "late": int(r["late"]), "names": r["names"], "flag_at": flag["at"],
                    "flag_spread": _num(flag["spread"]), "flag_team_total": f_tt, "last_spread": _num(r["spread"]),
                    "last_team_total": l_tt, "points": pts, "margin": margin,
                    "vs_flag_total": None if f_tt is None else round(pts - f_tt, 2),
                    "vs_last_total": None if l_tt is None else round(pts - l_tt, 2),
                    "cover_flag": _cover(margin, _num(flag["spread"])), "cover_last": _cover(margin, _num(r["spread"])),
                    "graded_at": now.strftime("%Y-%m-%d %H:%M")})
    if out:
        pd.DataFrame(out)[RCOLS].to_csv(rpath, mode="a", header=not rpath.exists(), index=False)
    return len(out)


def _part(x: pd.DataFrame) -> dict:
    def avg(col):
        v = x[col].dropna().astype(float)
        if not len(v):
            return {"n": 0, "mean": None, "se": None}
        return {"n": int(len(v)), "mean": round(float(v.mean()), 2),
                "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 2) if len(v) > 1 else None}

    def cov(col):
        w, l, p = (int((x[col] == c).sum()) for c in "WLP")
        return {"w": w, "l": l, "p": p, "rate": round(w / (w + l), 4) if w + l else None}

    return {"n": int(len(x)), "vs_flag_total": avg("vs_flag_total"), "vs_last_total": avg("vs_last_total"),
            "cover_flag": cov("cover_flag"), "cover_last": cov("cover_last")}


def summary(root: Path = DIR) -> dict:
    rpath = root / "nfl_results.csv"
    out = {"rule": "paper record only; read after the 2027 season; see nflmodel/olwatch.py"}
    if rpath.exists():
        r = pd.read_csv(rpath, dtype={"game_id": str})
        out.update({"all": _part(r), "late news among the linemen out": _part(r[r["late"] == 1]),
                    "injury report and roster only": _part(r[r["late"] == 0]),
                    "by_season": {str(s): _part(x) for s, x in r.groupby("season")}})
    out["built"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return out


def update(slate_games: list[dict], season: int, kickoffs: dict, scores: dict, root: Path = DIR,
           core: pd.DataFrame | None = None) -> dict:
    if core is None:
        p = core_path(season)
        core = pd.read_csv(p, dtype={"game_id": str}, keep_default_na=False) if p.exists() else None
    n_new = record(slate_games, season, kickoffs, core, root) if core is not None and len(core) else 0
    n_graded = grade(scores, root)
    s = summary(root)
    if (root / "nfl_log.csv").exists():
        (root / "nfl_record.json").write_text(json.dumps(s, indent=1))
    print(f"  offensive-line watch: {n_new} rows, {n_graded} graded")
    return s
