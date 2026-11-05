"""Live record of the Wong underdog teaser strategy (claude/betting-strategies.md, strategy 2).

Strategy (Ryan, Nov 5, 2026; from the consultant review of that day): NFL underdogs of +1.5 to
+2.5 teased 6 points to +7.5 / +8.5, in games totalled 49.5 or less, paired into two-team
teasers at -120 or better. Since 2019 those legs cleared 77.6% at the closing line (a two-team
teaser at -120 needs 73.9%); favourite Wong legs (-7.5 to -8.5) cleared 71.1% and are logged
only to keep watching that split. This module records and grades; it never changes the board.

Every board build, for each game that has not kicked off:

  a side is *tracked* while its own spread (Pinnacle's anchor line, the board's `market`) or
  its DraftKings or FanDuel line is in either Wong range; once tracked it stays in the log
  until kickoff, so a leg that moves out of range (+2.5 to +3) is recorded as moving out.
  A row is appended whenever the side's lines or total change (data/teasers/nfl_log.csv,
  append-only, never edited). `kind`: dog or fav, from DraftKings' line, else FanDuel's, else the
  anchor (the books are where it is bet); `qualifies`: 1 for an underdog leg in a game totalled
  49.5 or less at that moment.

After the game, each side is graded from its last row before kickoff (the line the strategy
would have bet) at the anchor, DraftKings and FanDuel lines plus 6, and at `bet_line`, the better
of the two books' lines that is in range (data/teasers/nfl_results.csv: W, L or P, pushes as no
action). The summary (data/teasers/nfl_record.json) gives the leg record by kind at `bet_line`, and
the two-team teasers: every pair of qualifying legs from different games in the same week, one unit
at -120 and at -130; a push in either leg refunds the pair (FanDuel's two-team rule).
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DIR = C.DATA / "teasers"
TEASE = 6.0
DOG = (1.5, 2.5)
FAV = (-8.5, -7.5)
MAX_TOTAL = 49.5
PRICES = (-120, -130)
COLS = ["at", "season", "week", "game_id", "kickoff", "side", "team", "kind", "qualifies", "in_range",
        "line", "dk_line", "fd_line", "total"]
RCOLS = ["key", "season", "week", "game_id", "side", "kind", "qualifies", "line", "dk_line", "fd_line", "bet_line",
         "margin", "result", "result_dk", "result_fd", "result_bet", "graded_at"]


def _own(spread, side: str):
    """The side's own spread (positive = underdog) from a home-favoured-positive line."""
    if spread is None or (isinstance(spread, float) and np.isnan(spread)):
        return None
    return -float(spread) if side == "home" else float(spread)


def kind_of(own) -> str | None:
    if own is None:
        return None
    if DOG[0] <= own <= DOG[1]:
        return "dog"
    if FAV[0] <= own <= FAV[1]:
        return "fav"
    return None


def _book_line(g: dict, book: str):
    b = (g.get("books") or {}).get(book) or {}
    x = b.get("spread_line")
    return None if x is None else float(x)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str}) if path.exists() else pd.DataFrame(columns=COLS)


def record(slate_games: list[dict], season: int, kickoffs: dict, root: Path = DIR,
           now: dt.datetime | None = None) -> int:
    """Append a row for every tracked side whose lines changed. Returns rows written."""
    now = now or dt.datetime.now(dt.timezone.utc)
    path = root / "nfl_log.csv"
    log = _read(path)
    last = {(r["game_id"], r["side"]): r for r in log.to_dict("records")}   # later rows overwrite earlier
    rows = []
    for g in slate_games:
        gid = str(g["id"])
        ko = kickoffs.get(g["id"], kickoffs.get(gid))
        if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
            continue
        mk = g.get("market") or {}
        if mk.get("spread") is None or mk.get("total") is None:
            continue
        for side in ("home", "away"):
            own = _own(mk["spread"], side)
            dk, fd = _own(_book_line(g, "draftkings"), side), _own(_book_line(g, "fanduel"), side)
            k = kind_of(dk) or kind_of(fd) or kind_of(own)     # the books he bets first, then the anchor
            in_range = any(kind_of(x) is not None for x in (own, dk, fd))
            prev = last.get((gid, side))
            if not in_range and prev is None:
                continue
            total = float(mk["total"])
            row = {"at": now.strftime("%Y-%m-%d %H:%M"), "season": season, "week": g.get("week"), "game_id": gid,
                   "kickoff": pd.Timestamp(ko).strftime("%Y-%m-%dT%H:%MZ"), "side": side, "team": g.get(side),
                   "kind": k or "", "qualifies": int(k == "dog" and total <= MAX_TOTAL), "in_range": int(in_range),
                   "line": own, "dk_line": dk, "fd_line": fd, "total": total}
            if prev is not None and all(_same(prev.get(c), row[c]) for c in ("line", "dk_line", "fd_line", "total", "in_range")):
                continue
            rows.append(row)
    if rows:
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows)[COLS].to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def _same(a, b) -> bool:
    na = a is None or (isinstance(a, float) and np.isnan(a)) or a == ""
    nb = b is None or (isinstance(b, float) and np.isnan(b)) or b == ""
    if na or nb:
        return na and nb
    return abs(float(a) - float(b)) < 1e-9


def _res(own_margin: float, line) -> str | None:
    if line is None or (isinstance(line, float) and np.isnan(line)):
        return None
    x = own_margin + float(line) + TEASE
    return "W" if x > 0 else "L" if x < 0 else "P"


def grade(scores: dict, root: Path = DIR, now: dt.datetime | None = None) -> int:
    """Grade tracked sides whose game has a final score. scores: game id -> (home, away)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    path, rpath = root / "nfl_log.csv", root / "nfl_results.csv"
    if not path.exists():
        return 0
    log = _read(path)
    done = set(pd.read_csv(rpath, usecols=["key"])["key"]) if rpath.exists() else set()
    out = []
    for (gid, side), x in log.groupby(["game_id", "side"], sort=False):
        key = f"{gid}|{side}"
        if key in done or gid not in scores:
            continue
        hs, as_ = scores[gid]
        if hs is None or pd.isna(hs):
            continue
        r = x.iloc[-1]                                     # last row before kickoff
        if not int(r["in_range"]):
            continue                                       # moved out of range: not a leg at kickoff
        margin = (float(hs) - float(as_)) * (1 if side == "home" else -1)
        books = [float(v) for v in (r["dk_line"], r["fd_line"]) if pd.notna(v) and kind_of(float(v)) is not None]
        bet = max(books) if books else None                # the better of DraftKings and FanDuel, in range
        out.append({"key": key, "season": r["season"], "week": r["week"], "game_id": gid, "side": side,
                    "kind": r["kind"] if isinstance(r["kind"], str) and r["kind"] else kind_of(r["dk_line"]) or kind_of(r["fd_line"]) or "",
                    "qualifies": int(r["qualifies"]), "line": r["line"], "dk_line": r["dk_line"], "fd_line": r["fd_line"],
                    "bet_line": bet, "margin": margin, "result": _res(margin, r["line"]), "result_dk": _res(margin, r["dk_line"]),
                    "result_fd": _res(margin, r["fd_line"]), "result_bet": _res(margin, bet), "graded_at": now.strftime("%Y-%m-%d %H:%M")})
    if out:
        pd.DataFrame(out)[RCOLS].to_csv(rpath, mode="a", header=not rpath.exists(), index=False)
    return len(out)


def _dec(a: float) -> float:
    return 1 + 100 / -a if a < 0 else 1 + a / 100


def summary(root: Path = DIR) -> dict:
    rpath = root / "nfl_results.csv"
    if not rpath.exists():
        return {"legs": {}, "pairs": {}}
    r = pd.read_csv(rpath, dtype={"game_id": str})

    def legs(x):
        w, l, p = (int((x["result_bet"].fillna(x["result"]) == c).sum()) for c in "WLP")
        return {"w": w, "l": l, "p": p, "hit": round(w / (w + l), 4) if w + l else None}

    out = {"legs": {"dog, total 49.5 or less (the strategy)": legs(r[(r["kind"] == "dog") & (r["qualifies"] == 1)]),
                    "dog, total 50+": legs(r[(r["kind"] == "dog") & (r["qualifies"] == 0)]),
                    "fav (watch only)": legs(r[r["kind"] == "fav"])}}
    q = r[(r["kind"] == "dog") & (r["qualifies"] == 1) & r["result_bet"].notna()]
    pairs = {str(p): {"n": 0, "w": 0, "l": 0, "p": 0, "units": 0.0} for p in PRICES}
    for _, wk in q.groupby(["season", "week"]):
        for a, b in itertools.combinations(wk.to_dict("records"), 2):
            if a["game_id"] == b["game_id"]:
                continue
            res = (a["result_bet"], b["result_bet"])
            for price in PRICES:
                s = pairs[str(price)]
                s["n"] += 1
                if "L" in res:
                    s["l"] += 1; s["units"] -= 1
                elif "P" in res:
                    s["p"] += 1
                else:
                    s["w"] += 1; s["units"] += _dec(price) - 1
    for s in pairs.values():
        s["units"] = round(s["units"], 2)
        s["roi"] = round(s["units"] / (s["w"] + s["l"]), 4) if s["w"] + s["l"] else None
    out["pairs"] = pairs
    out["built"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return out


def update(slate_games: list[dict], season: int, kickoffs: dict, scores: dict, root: Path = DIR) -> dict:
    n_new = record(slate_games, season, kickoffs, root)
    n_graded = grade(scores, root)
    s = summary(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "nfl_record.json").write_text(json.dumps(s, indent=1))
    print(f"  Wong teaser log: {n_new} rows, {n_graded} graded")
    return s
