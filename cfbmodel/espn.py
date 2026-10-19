"""Current lines and prices from ESPN's public scoreboard (one book, usually ESPN BET or
DraftKings). Used for games not yet played; the cfbfastR consensus line is the fallback.

ESPN's odds block has appeared in two shapes; both are read:
  older   details "OSU -7.5", overUnder, homeTeamOdds/awayTeamOdds {moneyLine, spreadOdds}, overOdds, underOdds
  newer   moneyline/pointSpread/total {home/away/over/under: {close|open: {line, odds}}}
Anything missing or malformed is skipped, and a failed request changes nothing.
"""
from __future__ import annotations

import json
import re
import urllib.request

import numpy as np
import pandas as pd

URL = ("https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
       "?groups=80&seasontype={st}&week={w}&limit=400")


def _num(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"[-+]?\d+(\.\d+)?", str(x).replace("EVEN", "+100"))
    return float(m.group()) if m else np.nan


def _price(side: dict | None):
    if not isinstance(side, dict):
        return np.nan, np.nan
    leg = side.get("close") or side.get("current") or side.get("open") or {}
    return _num(leg.get("line")), _num(leg.get("odds") or leg.get("american"))


def parse_event(ev: dict) -> dict | None:
    comp = (ev.get("competitions") or [{}])[0]
    odds = (comp.get("odds") or [None])[0]
    if not isinstance(odds, dict):
        return None
    teams = {c.get("homeAway"): (c.get("team") or {}) for c in comp.get("competitors") or []}
    home_abbr = (teams.get("home") or {}).get("abbreviation")
    out = {"game_id": int(ev["id"]), "book": ((odds.get("provider") or {}).get("name"))}
    # spread, home team's perspective, positive = home favoured
    spread = np.nan
    ps = odds.get("pointSpread")
    if isinstance(ps, dict):
        line, out["home_spread_odds"] = _price(ps.get("home"))
        _, out["away_spread_odds"] = _price(ps.get("away"))
        spread = -line
    det = odds.get("details")
    if np.isnan(spread) and isinstance(det, str):
        if det.strip().upper() in ("EVEN", "PK", "PICK"):
            spread = 0.0
        else:
            m = re.match(r"\s*([A-Za-z&.'\- ]+?)\s+([-+]?\d+(\.\d+)?)\s*$", det)
            if m and home_abbr:
                v = abs(float(m.group(2)))
                spread = v if m.group(1).strip().upper() == home_abbr.upper() else -v
    out["spread_line"] = spread
    tot = odds.get("total")
    total = _num(odds.get("overUnder"))
    if isinstance(tot, dict):
        line, out["over_odds"] = _price(tot.get("over"))
        _, out["under_odds"] = _price(tot.get("under"))
        if np.isnan(total):
            total = line
    else:
        out["over_odds"], out["under_odds"] = _num(odds.get("overOdds")), _num(odds.get("underOdds"))
    out["total_line"] = total
    ml = odds.get("moneyline")
    if isinstance(ml, dict):
        out["home_moneyline"] = _price(ml.get("home"))[1]
        out["away_moneyline"] = _price(ml.get("away"))[1]
    else:
        out["home_moneyline"] = _num((odds.get("homeTeamOdds") or {}).get("moneyLine"))
        out["away_moneyline"] = _num((odds.get("awayTeamOdds") or {}).get("moneyLine"))
    if "home_spread_odds" not in out:
        out["home_spread_odds"] = _num((odds.get("homeTeamOdds") or {}).get("spreadOdds"))
        out["away_spread_odds"] = _num((odds.get("awayTeamOdds") or {}).get("spreadOdds"))
    for k in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds", "home_moneyline", "away_moneyline"):
        v = out.get(k, np.nan)
        out[k] = v if np.isfinite(v) and abs(v) >= 100 else np.nan      # an American price is at least 100
    return out


def current_lines(weeks: list[int], postseason: bool = False) -> pd.DataFrame:
    rows = []
    for w in sorted(set(int(x) for x in weeks)):
        try:
            req = urllib.request.Request(URL.format(st=3 if postseason else 2, w=w), headers={"User-Agent": "cfb-model"})
            with urllib.request.urlopen(req, timeout=30) as r:
                sb = json.loads(r.read())
        except Exception as exc:
            print(f"  no ESPN lines for week {w} ({exc})")
            continue
        for ev in sb.get("events") or []:
            try:
                x = parse_event(ev)
            except Exception:
                x = None
            if x:
                rows.append(x)
    return pd.DataFrame(rows)


def apply(games: pd.DataFrame, lines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Use ESPN's spread, total and prices for unplayed games that have them."""
    if lines is None or not len(lines):
        return games, 0
    g = games.copy()
    L = lines.set_index("game_id")
    up = ~g["played"] & g["game_id"].isin(L.index)
    n = 0
    for c in ("spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds", "home_moneyline", "away_moneyline"):
        if c not in L:
            continue
        v = g.loc[up, "game_id"].map(L[c])
        if c not in g:
            g[c] = np.nan
        ok = v.notna()
        g.loc[ok[ok].index, c] = v[ok]
        if c == "spread_line":
            n = int(ok.sum())
    g.loc[up, "line_source"] = g.loc[up, "game_id"].map(L["book"]).fillna("ESPN")
    return g, n
