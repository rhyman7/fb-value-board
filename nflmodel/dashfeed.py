"""A small feed for the matchup dashboards (rhyman7/nfl, rhyman7/cfb).

docs/feed/nfl.json and docs/feed/cfb.json carry what the dashboards show from the value board:
each game's projected line and total (the board's final projection, `proj`), the bets the board
flags, and each player's projection and anytime-TD chance with this week's main prop line (NFL only; college props are
not priced). The dashboards are on the same site, so they read these files directly; the full
data.json is 0.5-3 MB, these are a few hundred KB at most.

Keys match the dashboards: NFL game ids are the same nflverse ids; college games are keyed by
the ESPN id the dashboard uses. Players are keyed "TEAM|normalized name".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SRC = {"nfl": DOCS / "data.json", "cfb": DOCS / "cfb" / "data.json"}
OUT = DOCS / "feed"
PROJ_STATS = ("pass_yds", "rush_yds", "rec_yds", "receptions", "targets", "carries", "pass_att")
# the defense factors behind each projection (1.0 = average; shrunk for sample size): yards per
# target allowed to the player's position (NFL), per carry, and per pass attempt
DEF_KEYS = {"def_ypt": "ypt", "def_ypc": "ypc", "def_ypa": "ypa"}
LINE_STATS = ("pass_yds", "rush_yds", "rec_yds", "receptions")


def norm(name: str) -> str:
    """Lower case, no punctuation, no Jr./Sr./II-V suffix, so 'Kenneth Walker III' == 'kenneth walker'."""
    s = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def _r(x, d=1):
    return None if x is None else round(float(x), d)


def _game(g: dict) -> dict:
    flagged = [{"m": b["market"], "pick": b["pick"], "odds": b.get("odds"), "ev": _r(b.get("ev"), 3), "tier": b.get("tier")}
               for b in g.get("bets") or [] if b.get("tier") and not b.get("missed")]   # the board's Bet sheet
    return {"id": g["id"], "away": g["away"], "home": g["home"],
            "margin": _r(g["proj"]["margin"]), "total": _r(g["proj"]["total"]),   # home minus away
            "mkt_spread": _r((g.get("market") or {}).get("spread")), "mkt_total": _r((g.get("market") or {}).get("total")),
            "bets": flagged, "open": _open(g)}


def _open(g: dict) -> dict | None:
    """The first line the board saw for the game (its line history): home-favored spread positive,
    total, and the date. NFL games carry it as g["open"]; college as opening_spread/opening_total too."""
    o = g.get("open") if isinstance(g.get("open"), dict) else {}
    sp = o.get("spread", g.get("opening_spread"))
    tot = o.get("total", g.get("opening_total"))
    if sp is None and tot is None:
        return None
    return {"spread": _r(sp), "total": _r(tot), "at": o.get("seen")}


def _lines(week: int | None, season: int | None) -> dict:
    """Main-line NFL props for the week: {(team, norm name, stat): {L, fo}} (fo = books' no-vig over chance)."""
    if week is None:
        return {}
    f = DOCS / "props" / f"{season}_w{int(week):02d}.json"
    if not f.exists():
        return {}
    out = {}
    for r in json.loads(f.read_text()).get("props", []):
        if r.get("alt") or r.get("s") not in LINE_STATS or r.get("L") is None:
            continue
        out[(r["t"], norm(r["n"]), r["s"])] = {"L": r["L"], "fo": _r(r.get("fo"), 3)}
    return out


def build(league: str, data: dict | None = None) -> dict:
    d = data if data is not None else json.loads(SRC[league].read_text())
    meta = d.get("meta", {})
    games = {}
    for g in (d.get("slate") or {}).get("games", []):
        key = str(g["game_id"]) if league == "cfb" and g.get("game_id") is not None else g["id"]
        games[key] = _game(g)
    pl = (d.get("players") or {}).get("players", [])
    week = next((p.get("week") for p in pl if p.get("week") is not None), None)
    lines = _lines(week, meta.get("season")) if league == "nfl" else {}
    players = {}
    for p in pl:
        k = f"{p['team']}|{norm(p['name'])}"
        row = {"n": p["name"], "pos": p["pos"], "g": p["game"], "st": p.get("status"),
               "p": {s: _r(p["proj"].get(s)) for s in PROJ_STATS if p["proj"].get(s)},
               "td": _r(p.get("td_any"), 3)}                    # chance of a rushing or receiving TD
        why = p.get("why") or {}
        f = {k: _r(why[w], 3) for w, k in DEF_KEYS.items() if why.get(w) is not None and abs(why[w] - 1) > 1e-9}
        if f:
            row["f"] = f
        ln = {s: lines[(p["team"], norm(p["name"]), s)] for s in LINE_STATS if (p["team"], norm(p["name"]), s) in lines}
        if ln:
            row["L"] = ln
        players[k] = row
    return {"league": league, "season": meta.get("season"), "week": week, "generated": meta.get("generated"),
            "games": games, "players": players}


def write(league: str, data: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{league}.json"
    path.write_text(json.dumps(build(league, data), separators=(",", ":")))
    return path


if __name__ == "__main__":
    for lg in ("nfl", "cfb"):
        print(write(lg), round(OUT.joinpath(f"{lg}.json").stat().st_size / 1024), "KB")
