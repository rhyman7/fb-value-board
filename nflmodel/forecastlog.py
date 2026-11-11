"""A log of every forecast the boards showed before kickoff (both boards; data only, never shown).

Why: docs/archive/ keeps one entry per game, the board as it stood at the last refresh before
kickoff, so the opener and midweek forecasts were only recoverable from git history, and the
quarterback's status was not kept at all. A test of the blend weight at the times the board is
actually used (Project doc claude/plan-uncertain-qb-test-2026-11-11.md) needs the model's
forecast, the line it was blended with and the starter's status together, at each refresh.

Every board build, for each game on the slate that has not kicked off, a row is appended to
data/forecast_log/<league>_<season>.csv whenever the line, either quarterback or his status
differs from the game's last row, or the forecast or weight has moved past a small tolerance
(TOL: 0.05 points, 0.005 of weight). Append-only: never edit or delete rows. Nothing reads it but tests of the model; a failure never stops a board build.

Columns
  at                 when the board was built (UTC)
  model_margin/total the raw model (home margin, total)
  market_spread/total the line the board blended with (home-favoured positive, as on the board)
  proj_margin/total  the blend the board showed
  weight, early      the model's weight in that blend; early = 1 for the 4+-days-out weight
  away_qb, home_qb   the game model's quarterbacks
  *_qb_report        NFL only: his level on this week's injury report (1 = Questionable; -1 = not
                     listed; blank = no report read). *_qb_practice: Friday practice (0 full,
                     1 limited, 2 none, -1 not listed). College boards leave these blank.
  source             live, or backfill (rebuilt from the git history of docs/data.json by
                     scripts/forecast_log_backfill.py; those rows know a quarterback's status
                     only where the board carried a qb_flag, i.e. Questionable after a limited
                     or no practice, from Nov 9, 2026)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DIR = C.DATA / "forecast_log"
NUM = ["model_margin", "model_total", "market_spread", "market_total", "proj_margin", "proj_total", "weight", "early",
       "away_qb_report", "away_qb_practice", "home_qb_report", "home_qb_practice"]
TXT = ["away_qb", "home_qb"]
COLS = ["at", "season", "week", "game_id", "kickoff", "away", "home", *NUM[:8], "away_qb", "home_qb", *NUM[8:], "source"]
# The model is refitted on every build, so its numbers wobble by a hundredth of a point between
# builds with no news. A row is written only when a forecast has moved this far from the game's
# last row; lines, quarterbacks, status and the early flag are compared exactly.
TOL = {"model_margin": 0.05, "model_total": 0.05, "proj_margin": 0.05, "proj_total": 0.05, "weight": 0.005}
PRACTICE_TEXT = {"limited Friday": 1, "did not practice Friday": 2}      # qshade.QB_PRACTICE, for the backfill


def path(league: str, season: int, root: Path = DIR) -> Path:
    return root / f"{league}_{int(season)}.csv"


def _key(name) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalpha())


def _blank(x) -> bool:
    return x is None or x == "" or (isinstance(x, float) and np.isnan(x))


def _same(a, b) -> bool:
    if _blank(a) or _blank(b):
        return _blank(a) and _blank(b)
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def qb_status(games: list[dict], report: dict, names: dict) -> dict:
    """{(game id, side): (report level, practice level)} for each game-model quarterback.
    report: {(game_id, player_id): (report level, practice level)} for this week's quarterbacks;
    names: {player_id: name}. Matched by game and name, as qshade.qb_flags does."""
    by = {}
    for (gid, pid), (rep, lvl) in report.items():
        try:
            by[(gid, _key(names.get(pid)))] = (int(rep), int(lvl))
        except (TypeError, ValueError):
            continue
    out = {}
    for g in games:
        for side in ("away", "home"):
            k = (g.get("id"), _key((g.get(side + "_qb") or {}).get("name")))
            if k in by:
                out[(g.get("id"), side)] = by[k]
    return out


def status_from_flags(games: list[dict]) -> dict:
    """The same shape from the board's qb_flag alone (the backfill: all the history kept)."""
    out = {}
    for g in games:
        for f in g.get("qb_flag") or []:
            side = "away" if f.get("team") == g.get("away") else "home" if f.get("team") == g.get("home") else None
            if side and f.get("practice") in PRACTICE_TEXT:
                out[(g.get("id"), side)] = (1, PRACTICE_TEXT[f["practice"]])
    return out


def rows(slate_games: list[dict], season: int, at: pd.Timestamp, status: dict | None = None, source: str = "live",
         kickoffs: dict | None = None) -> list[dict]:
    """One row per game on the slate that has not kicked off at `at`."""
    status, kickoffs = status or {}, kickoffs or {}
    out = []
    for g in slate_games:
        gid = str(g.get("id"))
        ko = g.get("kickoff") or kickoffs.get(gid)
        if not ko:
            continue
        ko = pd.Timestamp(ko)
        ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
        if ko <= at:
            continue
        mk, md, pj = g.get("market") or {}, g.get("model") or {}, g.get("proj") or {}
        if md.get("margin") is None:
            continue
        row = {"at": at.strftime("%Y-%m-%d %H:%M"), "season": int(season), "week": g.get("week"), "game_id": gid,
               "kickoff": ko.strftime("%Y-%m-%dT%H:%MZ"), "away": g.get("away"), "home": g.get("home"),
               "model_margin": md.get("margin"), "model_total": md.get("total"),
               "market_spread": mk.get("spread"), "market_total": mk.get("total"),
               "proj_margin": pj.get("margin"), "proj_total": pj.get("total"),
               "weight": g.get("weight"), "early": None if g.get("early") is None else int(bool(g.get("early"))),
               "away_qb": (g.get("away_qb") or {}).get("name"), "home_qb": (g.get("home_qb") or {}).get("name"),
               "source": source}
        for side in ("away", "home"):
            rep, lvl = status.get((g.get("id"), side), (None, None))
            row[side + "_qb_report"], row[side + "_qb_practice"] = rep, lvl
        out.append(row)
    return out


def changed(prev: dict | None, row: dict) -> bool:
    """A new row is due: anything exact changed, or a forecast drifted past its tolerance since the
    game's last logged row (so slow drift is still caught once it adds up)."""
    if prev is None or not all(_same(prev.get(c), row.get(c)) for c in NUM + TXT if c not in TOL):
        return True
    for c, tol in TOL.items():
        a, b = prev.get(c), row.get(c)
        if _blank(a) or _blank(b):
            if not (_blank(a) and _blank(b)):
                return True
        elif abs(float(a) - float(b)) >= tol - 1e-9:
            return True
    return False


def read(league: str, season: int, root: Path = DIR) -> pd.DataFrame:
    p = path(league, season, root)
    return pd.read_csv(p, dtype={"game_id": str, "away_qb": str, "home_qb": str}) if p.exists() else pd.DataFrame(columns=COLS)


def record(slate_games: list[dict], league: str, season: int, generated=None, status: dict | None = None,
           root: Path = DIR, source: str = "live", kickoffs: dict | None = None) -> int:
    """Append a row for every game whose forecast, line, weight, quarterback or status changed.
    generated: the board's build time ("2026-11-11 21:10 UTC"); now if missing. Returns rows written."""
    at = pd.Timestamp(str(generated).replace(" UTC", "")) if generated else pd.Timestamp(dt.datetime.now(dt.timezone.utc))
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    log = read(league, season, root)
    last = {r["game_id"]: r for r in log.to_dict("records")}             # later rows overwrite earlier
    new = [r for r in rows(slate_games, season, at, status, source, kickoffs) if changed(last.get(r["game_id"]), r)]
    if new:
        root.mkdir(parents=True, exist_ok=True)
        p = path(league, season, root)
        df = pd.DataFrame(new)[COLS]
        for c in ("early", *NUM[8:]):                                    # whole numbers, blank when unknown
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        df.to_csv(p, mode="a", header=not p.exists(), index=False)
    return len(new)
