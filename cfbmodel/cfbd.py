"""Current lines from CollegeFootballData.com (CFBD), a second source for games not yet played.

Optional: used only when the CFBD_API_KEY environment variable is set (the repository secret
of the same name in the Update college board Action). ESPN's scoreboard (espn.py) is the
first source; where ESPN has no line for a game, the CFBD line fills it, taking the
"consensus" provider first, then DraftKings, then the first provider present, and the game's
`line_source` says which ("CFBD consensus", "CFBD DraftKings", ...). Without the key nothing
changes, and any failure prints one line and changes nothing.

GET https://api.collegefootballdata.com/lines?year={season}&week={week}&seasonType=regular
with `Authorization: Bearer <key>` returns a list of games (id, homeTeam, awayTeam, startDate,
lines: [{provider, spread, spreadOpen, overUnder, overUnderOpen, homeMoneyline, awayMoneyline,
formattedSpread}]). `spread` is from the home team's view (negative = home favoured), so the
board's spread_line (positive = home favoured) is -spread. The game id is the ESPN id the
board uses as game_id.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

URL = "https://api.collegefootballdata.com/lines?year={season}&week={week}&seasonType={st}"
ENV = "CFBD_API_KEY"
PREFERRED = ("consensus", "DraftKings")


def _num(x):
    if x is None or isinstance(x, bool):
        return np.nan
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _price(x):
    v = _num(x)
    return v if np.isfinite(v) and abs(v) >= 100 else np.nan     # an American price is at least 100


def pick_line(lines: list | None) -> dict | None:
    """The provider to use: consensus, else DraftKings, else the first one with a spread or total."""
    if not isinstance(lines, list):
        return None
    usable = [ln for ln in lines if isinstance(ln, dict) and (np.isfinite(_num(ln.get("spread"))) or np.isfinite(_num(ln.get("overUnder"))))]
    if not usable:
        return None
    by_name = {str(ln.get("provider") or "").lower(): ln for ln in reversed(usable)}
    for name in PREFERRED:
        if name.lower() in by_name:
            return by_name[name.lower()]
    return usable[0]


def parse_game(game: dict) -> dict | None:
    """One row in espn.py's shape (game_id, book, spread_line, total_line, prices), or None."""
    try:
        gid = int(game["id"])
    except (KeyError, TypeError, ValueError):
        return None
    ln = pick_line(game.get("lines"))
    if ln is None:
        return None
    spread = _num(ln.get("spread"))
    out = {"game_id": gid, "book": f"CFBD {ln.get('provider') or 'line'}",
           "spread_line": -spread if np.isfinite(spread) else np.nan,
           "total_line": _num(ln.get("overUnder")),
           "home_moneyline": _price(ln.get("homeMoneyline")), "away_moneyline": _price(ln.get("awayMoneyline")),
           "home_spread_odds": np.nan, "away_spread_odds": np.nan, "over_odds": np.nan, "under_odds": np.nan}
    return out


def parse(payload) -> pd.DataFrame:
    rows = []
    for game in payload if isinstance(payload, list) else []:
        try:
            x = parse_game(game)
        except Exception:
            x = None
        if x:
            rows.append(x)
    return pd.DataFrame(rows)


def enabled() -> bool:
    return bool(os.environ.get(ENV))


def current_lines(season: int, weeks: list[int], postseason: bool = False) -> pd.DataFrame:
    """CFBD's lines for the given weeks; empty without the key or on any failure."""
    key = os.environ.get(ENV)
    if not key:
        return pd.DataFrame()
    rows = []
    for w in sorted(set(int(x) for x in weeks)):
        try:
            req = urllib.request.Request(URL.format(season=int(season), week=w, st="postseason" if postseason else "regular"),
                                         headers={"Authorization": f"Bearer {key}", "User-Agent": "cfb-model", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read())
        except Exception as exc:
            print(f"  no CFBD lines for week {w} ({type(exc).__name__})")
            continue
        rows.append(parse(payload))
    rows = [x for x in rows if len(x)]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def apply(games: pd.DataFrame, lines: pd.DataFrame, have: set | None = None) -> tuple[pd.DataFrame, int]:
    """Use CFBD's line for unplayed games ESPN did not price (`have`: the game ids ESPN gave a
    spread). Like espn.apply, it replaces the cfbfastR consensus line for those games."""
    if lines is None or not len(lines):
        return games, 0
    g = games.copy()
    L = lines.drop_duplicates("game_id").set_index("game_id")
    L = L[L["spread_line"].notna() | L["total_line"].notna()]
    skip = set(have or ())
    up = ~g["played"] & g["game_id"].isin(L.index) & ~g["game_id"].isin(skip)
    n = 0
    for c in ("spread_line", "total_line", "home_moneyline", "away_moneyline"):
        if c not in L:
            continue
        v = g.loc[up, "game_id"].map(L[c])
        if c not in g:
            g[c] = np.nan
        ok = v.notna()
        g.loc[ok[ok].index, c] = v[ok]
        if c == "spread_line":
            n = int(ok.sum())
    if "line_source" not in g:
        g["line_source"] = np.nan
    g.loc[up, "line_source"] = g.loc[up, "game_id"].map(L["book"])
    return g, n
