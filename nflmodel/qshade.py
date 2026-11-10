"""The display shade for NFL Questionable players (Nov 9, 2026; PLAYERS.md "Questionable players:
a shade on top of the clean multiplier").

A Questionable RB/WR/TE's projection (proj) is what he does if he plays: his props, parlays and
best bets price from it, and a prop is void if he sits. The Players tab and matchups show his
expected line instead, counting part of his chance to sit: receptions, receiving yards, carries
and rushing yards x (1 - lambda (1 - chance)), with chance by position and last Friday practice
and lambda per stat from data/questionable_shade.json. Teammates don't change (sharing his
lost usage tested worse). The board JSON gets play_chance and proj_ev; proj is never touched,
so nothing that reads proj changes. The page (site/template.html, adoptNew) shows proj_ev in
place of proj and keeps the if-he-plays number for the card.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as C

PATH = C.DATA / "questionable_shade.json"
STATS = ("receptions", "rec_yds", "carries", "rush_yds")
GROUP = {"RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
PRACTICE = {0: "full practice", 1: "limited in practice", 2: "did not practice"}


def load(path: Path | None = None) -> dict | None:
    path = path or PATH
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def chance(c: dict, group: str, practice) -> float:
    """His chance to play: by position and last practice level, else the overall rate."""
    try:
        lvl = str(int(practice))
    except (TypeError, ValueError):
        lvl = None
    return float(c["chance"].get(group, {}).get(lvl, c["chance"]["overall"]))


def apply(players: list[dict], report: dict, consts: dict | None = None) -> int:
    """Add play_chance and proj_ev to each RB/WR/TE listed Questionable on this week's report (in
    place). report: {(game_id, player_id): (report level, last practice level)}, as
    players_ml.questionable reads them (roles.py: report 1 = Questionable). The board's status
    is not used: before this week's report is out it can still carry last week's. Returns how
    many."""
    consts = consts if consts is not None else load()
    if not consts:
        return 0
    lam = consts["lambda"]
    n = 0
    for p in players:
        g = GROUP.get(p.get("pos"))
        rep, lvl = report.get((p.get("game"), p.get("id")), (None, None))
        if g is None or rep != 1:
            continue
        c = chance(consts, g, lvl)
        proj = p.get("proj") or {}
        ev = {}
        for st in STATS:
            v = proj.get(st)
            if v is None:
                continue
            ev[st] = round(float(v) * (1 - float(lam.get(st, 0.0)) * (1 - c)), 1)
        if ev:
            if lvl in PRACTICE:
                p["practice"] = PRACTICE[lvl]
            p["play_chance"] = round(c, 3)
            p["proj_ev"] = ev
            n += 1
    return n


# Questionable starting quarterbacks (Nov 9, 2026). A team's starter from its last game who was
# listed Questionable started 60% of the time, 2016-2026 (167): 91% after a full Friday practice
# (34), 54% after a limited one (112), 43% after none (21). The board flags the game when the
# game model's quarterback is on this week's report as Questionable and was limited or didn't
# practice, so the spread and total aren't bet before the starter news. No number changes; the
# daily QB check switches the starter (and so clears the flag) once one is reported.
QB_STARTED = {1: 0.54, 2: 0.43}
QB_PRACTICE = {1: "limited Friday", 2: "did not practice Friday"}


def _key(name) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalpha())


def qb_flags(games: list[dict], report: dict, names: dict) -> int:
    """Add qb_flag (a list of {team, name, practice, started}) to each game in the slate whose
    game-model quarterback (away_qb / home_qb) is Questionable on this week's report after a
    limited or no practice. report: {(game_id, player_id): (report level, practice level)};
    names: {player_id: name}. Matched by game, team and name. Returns how many flags."""
    by = {}
    for (gid, pid), (rep, lvl) in report.items():
        by.setdefault((gid, _key(names.get(pid))), []).append((rep, lvl))
    n = 0
    for g in games:
        flags = []
        for side in ("away", "home"):
            q = g.get(side + "_qb") or {}
            hits = by.get((g.get("id"), _key(q.get("name"))), [])
            for rep, lvl in hits:
                try:
                    rep, lvl = int(rep), int(lvl)
                except (TypeError, ValueError):
                    continue
                if rep == 1 and lvl in QB_STARTED:
                    flags.append({"team": g.get(side), "name": q.get("name"), "practice": QB_PRACTICE[lvl], "started": QB_STARTED[lvl]})
                    break
        if flags:
            g["qb_flag"] = flags
            n += len(flags)
        else:
            g.pop("qb_flag", None)
    return n
