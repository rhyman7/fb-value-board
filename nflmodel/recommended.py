"""The Recommended list: a public record of picks that meet rules fixed before the season.

Ryan's call, Nov 20, 2026 (Project doc `claude/plan-recommended-list-2026-11-20.md`, which holds
the rules as approved; change nothing here mid-season). The boards keep showing every lean as
information; this list is the short set of picks the site stands behind, published before
kickoff and graded at the price shown. Only this record is marketed.

A *track* is one kind of pick with one frozen rule and its own record. Every track starts
**on trial**; a track is called Recommended only after a season review (STATUS below is changed
by hand then, never by this code):

  cfb_spread_open   College spreads at the opener. Week 5 on: the model is OPEN_GAP points or
                    more off the opening spread, and still that far off the spread at the
                    better of DraftKings and FanDuel when the pick is published.
  nfl_spread_open   NFL spreads at the opener: the same rule (not in the final week, where the
                    board itself defers to the market, config.FINAL_WEEK_MARKET).
  nfl_price_agree   A DraftKings or FanDuel price beats the sharp book's no-vig number by
                    sheet.PRICE_MIN and the model is on the same side of the sharp line
                    (claude/betting-strategies.md, strategy 3). Spread, total or moneyline.
  nfl_wong_teaser   Wong underdog teasers (strategy 2): underdogs +1.5 to +2.5 at one of the two
                    books, total 49.5 or less, teased 6, two legs from different games at the
                    same book. Teaser prices are not fetched: graded at -120, the worst price
                    the rule allows.
  nfl_prop_under    NFL prop unders the model agrees with (strategy 4): a main-line under where
                    the board's over chance is below propboard.AGREE; receptions and receiving
                    yards, and a running back's rushing yards and attempts.
  cfb_total_open    College totals at the opener: the model TOTAL_GAP points or more off the
                    opening total, and still that far off the total at the better book.

Record rules (the same for every track):
  - a pick is written once, the first build its rule is met before kickoff, with the time,
    book, line and price, and never changed or removed (data/recommended/<league>_picks.csv,
    append-only, written by the Update and Player props Actions only);
  - DraftKings and FanDuel only (ledger.MY_BOOKS), the better of the two at that moment;
  - one pick per game and market per track (per player and stat for props);
  - graded at the published price on one flat unit (data/recommended/<league>_results.csv,
    append-only); a push is zero units and left out of the win rate; a prop the books void is
    no action;
  - nothing is backfilled: the record starts with the first build after this module merged.

Rolling total (Ryan, Nov 20, 2026): UNIT dollars on every pick. A bet that lands on two tracks
(same game, market and side) is one bet: it counts once in the totals, on the track that
published it first, and in each track's own record.

The summary goes to docs/recommended/<league>.js (a script, so the page and its PDF read it
from disk as well as from the site). The page is docs/recommended/index.html.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C, ledger as L, sheet as SH

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "recommended"
OUT = ROOT / "docs" / "recommended"
PROPS = ROOT / "docs" / "props"

UNIT = 10.0                       # dollars a pick, for the rolling total
MY_BOOKS = L.MY_BOOKS
OPEN_GAP = 2.0                    # points off the opening spread (both leagues)
TOTAL_GAP = 3.0                   # points off the opening total (college)
CFB_FROM_WEEK = 5
TEASE = 6.0
DOG = (1.5, 2.5)
MAX_TOTAL = 49.5
TEASER_ODDS = -120.0
PROP_STATS = {"receptions": None, "rec_yds": None, "rush_yds": "RB", "carries": "RB"}   # stat -> position it is limited to
NEED = 100                        # graded picks before a track can be reviewed

# key, league, name, the rule in a sentence, what was known before going live (never part of the record)
TRACKS = [
    ("cfb_spread_open", "cfb", "College spreads at the opener",
     "From week 5: the model is 2+ points off the opening spread, and still 2+ off the spread at the better of DraftKings and FanDuel. The pick is the model's side.",
     "Week 5 on, at the opening spread and -110: 53.9% in 2021-25 (721-616), 55.3% in 2026 before this record began (119-96). Not measured at these two books' prices."),
    ("cfb_total_open", "cfb", "College totals at the opener",
     "The model is 3+ points off the opening total, and still 3+ off the total at the better of DraftKings and FanDuel.",
     "56.3% in 2021-25 (640-497); 43.1% in 2026 before this record began (69-91). Starts far behind."),
    ("nfl_spread_open", "nfl", "NFL spreads at the opener",
     "The model is 2+ points off the opening spread, and still 2+ off the spread at the better of DraftKings and FanDuel. The pick is the model's side.",
     "54.7%, +4.3% (plus or minus 3.7%) since 2019. The edge is in the very first number."),
    ("nfl_price_agree", "nfl", "NFL price, model agrees",
     "DraftKings or FanDuel beats the sharp book's fair price by 1% or more, and the model is on the same side.",
     "24-14 in 2024-25, closing line value +4.4% (plus or minus 1.8%). Missed its pre-set test on one cell."),
    ("nfl_wong_teaser", "nfl", "Wong underdog teasers",
     "Underdogs of +1.5 to +2.5 teased 6 points, total 49.5 or less, two teams at -120 or better. Graded at -120.",
     "Legs cleared 79.3% since 2019 at totals of 49.5 or less (318 legs). A two-team teaser at -120 needs 73.9% a leg."),
    ("nfl_prop_under", "nfl", "NFL prop unders, model agrees",
     "Main-line under where the model gives the over less than 35%. Receptions, receiving yards, and a running back's rushing yards and attempts.",
     "+5.4% (plus or minus 1.9%) at the close on 4,920 props in 2023-25. 2026 before this record began was against it (51-72)."),
]
STATUS = {k: "trial" for k, *_ in TRACKS}      # trial | recommended | proven | paused: changed by hand at a season review only
# Held back before its first pick (Nov 20, 2026, Ryan's call after the pre-set test in the Project doc
# claude/nfl-prop-unders-improvement-results-2026-11-20.md: the rule's return rests on 2025 alone and
# no tested change improved it). A held track publishes nothing and has no record; it is listed so the
# page says why. It starts, on trial, only when Ryan says so.
STATUS["nfl_prop_under"] = "held"
HELD = {k for k, v in STATUS.items() if v == "held"}
NAMES = {k: n for k, _, n, *_ in TRACKS}

COLS = ["row_id", "track", "league", "at", "season", "week", "game_id", "data_id", "kickoff", "label", "market", "side",
        "pick", "book", "line", "odds", "model", "opener", "legs", "built"]
RES_COLS = L.RES_COLS


def picks_path(league: str, root: Path = DIR) -> Path:
    return root / f"{league}_picks.csv"


def results_path(league: str, root: Path = DIR) -> Path:
    return root / f"{league}_results.csv"


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype={"row_id": str, "game_id": str, "data_id": str, "at": str, "legs": str, "label": str, "pick": str})
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df[cols]


def _append(path: Path, rows: list[dict], cols: list[str]) -> None:
    """Append rows under the file's own header (a new file gets `cols`)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    have = list(pd.read_csv(path, nrows=0).columns) if path.exists() else cols
    pd.DataFrame(rows).reindex(columns=have).to_csv(path, mode="a", header=not path.exists(), index=False)


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def _dec(a: float) -> float:
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def _mine(b: dict) -> list[dict]:
    return [q for q in (b.get("books") or []) if str(q.get("book", "")).lower() in MY_BOOKS and _ok(q.get("odds"))]


def _book_gap(b: dict, g: dict, q: dict) -> float | None:
    """The model minus this book's own line, signed toward the bet's side (points)."""
    if not _ok(q.get("line")):
        return None
    s = SH._sgn(b["side"])
    if b["market"] == "Spread":
        return s * float(g["model"]["margin"]) + float(q["line"])       # a book's spread line is the side's own handicap
    if b["market"] == "Total":
        return s * (float(g["model"]["total"]) - float(q["line"]))
    return None


def _label(g: dict) -> str:
    return f"{g.get('away')} at {g.get('home')}"


def _row(track, league, stamp, season, g, ko, b, q, built, model=None) -> dict:
    gid = str(g["id"])
    data_id = g.get("game_id")
    return {"row_id": f"{track}|{gid}|{b['market']}|{b['side']}", "track": track, "league": league, "at": stamp, "season": season,
            "week": g.get("week"), "game_id": gid, "data_id": str(int(data_id)) if _ok(data_id) and league == "cfb" else str(data_id if _ok(data_id) else gid),
            "kickoff": pd.Timestamp(ko).strftime("%Y-%m-%dT%H:%MZ"), "label": _label(g), "market": b["market"], "side": b["side"],
            "pick": _pick_text(g, b, q), "book": q["book"], "line": q.get("line"), "odds": float(q.get("odds_net", q["odds"])),
            "model": model, "opener": SH._open_line(g, b["market"])[0], "legs": None, "built": built}


def _pick_text(g: dict, b: dict, q: dict) -> str:
    """The pick at the book's own line (the bet's `pick` is at the anchor's line)."""
    line = q.get("line")
    if b["market"] == "Spread" and _ok(line):
        team = g.get("home") if b["side"] == "home" else g.get("away")
        return f"{team} {'PK' if float(line) == 0 else format(float(line), '+g')}"
    if b["market"] == "Total" and _ok(line):
        return f"{'Over' if b['side'] == 'over' else 'Under'} {float(line):g}"
    if b["market"] == "Moneyline":
        return f"{g.get('home') if b['side'] == 'home' else g.get('away')} to win"
    return str(b.get("pick"))


def opener_picks(g: dict, league: str) -> list[tuple[str, dict, dict, float]]:
    """(track, bet, book quote, model number) for the opener tracks this game meets now."""
    out = []
    week = g.get("week")
    for b in g.get("bets") or []:
        mk = b["market"]
        if mk == "Spread":
            track, need = f"{league}_spread_open", OPEN_GAP
            if league == "cfb" and not (_ok(week) and int(week) >= CFB_FROM_WEEK):
                continue
            if league == "nfl" and g.get("final_week") and getattr(C, "FINAL_WEEK_MARKET", False):
                continue
        elif mk == "Total" and league == "cfb":
            track, need = "cfb_total_open", TOTAL_GAP
        else:
            continue
        if not _ok(b.get("opener_gap")) or float(b["opener_gap"]) < need:
            continue
        best = None
        for q in _mine(b):
            gap = _book_gap(b, g, q)
            if gap is None or gap < need:
                continue
            key = (gap, _dec(q.get("odds_net", q["odds"])))
            if best is None or key > best[0]:
                best = (key, q)
        if best:
            out.append((track, b, best[1], float(g["model"]["total" if mk == "Total" else "margin"])))
    return out


def price_picks(g: dict) -> list[tuple[str, dict, dict, float]]:
    """NFL price, model agrees: a DraftKings or FanDuel price that beats the sharp no-vig number
    by sheet.PRICE_MIN, on a side the model's number is on (gap above zero)."""
    out = []
    for b in g.get("bets") or []:
        if b["market"] not in ("Spread", "Total", "Moneyline") or not _ok(b.get("gap")) or float(b["gap"]) <= 0:
            continue
        qs = [q for q in _mine(b) if _ok(q.get("ev_line")) and float(q["ev_line"]) >= SH.PRICE_MIN]
        if qs:
            q = max(qs, key=lambda q: float(q["ev_line"]))
            out.append(("nfl_price_agree", b, q, float(g["model"]["total" if b["market"] == "Total" else "margin"])))
    return out


def teaser_legs(g: dict) -> dict:
    """{book: leg} for the Wong underdog legs this game offers now (a leg: side, team, the
    book's own line and the teased line)."""
    total = (g.get("market") or {}).get("total")
    if not _ok(total) or float(total) > MAX_TOTAL:
        return {}
    out: dict = {}
    for b in g.get("bets") or []:
        if b["market"] != "Spread":
            continue
        for q in _mine(b):
            if _ok(q.get("line")) and DOG[0] <= float(q["line"]) <= DOG[1]:
                bk = str(q["book"]).lower()
                leg = {"game_id": str(g["id"]), "side": b["side"], "team": g.get("home") if b["side"] == "home" else g.get("away"),
                       "line": float(q["line"]), "teased": float(q["line"]) + TEASE}
                if bk not in out or leg["line"] > out[bk]["line"]:
                    out[bk] = leg
    return out


def record_games(slate_games: list[dict], league: str, season: int, kickoffs: dict, built: str | None = None,
                 now: dt.datetime | None = None, root: Path = DIR) -> int:
    """Publish every pick the game-line tracks meet now, on games that have not kicked off.
    kickoffs: board game id -> UTC Timestamp. Returns the picks written."""
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    built = built or os.environ.get("GITHUB_SHA", "")[:7] or stamp
    path = picks_path(league, root)
    old = _read(path, COLS)
    have = set(old["row_id"])
    taken = {(r.track, r.game_id, r.market) for r in old.itertuples()}      # one pick per game and market per track
    rows = []
    live = []
    for g in slate_games:
        ko = kickoffs.get(g["id"], kickoffs.get(str(g["id"])))
        if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
            continue
        live.append((pd.Timestamp(ko), g))
        found = opener_picks(g, league) + (price_picks(g) if league == "nfl" else [])
        for track, b, q, model in found:
            key = (track, str(g["id"]), b["market"])
            if key in taken:
                continue
            row = _row(track, league, stamp, season, g, ko, b, q, built, model)
            if row["row_id"] in have:
                continue
            rows.append(row)
            taken.add(key)
            have.add(row["row_id"])
    if league == "nfl":
        rows += _teasers(live, old, season, stamp, built)
    _append(path, rows, COLS)
    return len(rows)


def _teasers(live: list, old: pd.DataFrame, season: int, stamp: str, built: str) -> list[dict]:
    """Pair this build's unused Wong legs, earliest kickoffs first, two from different games at
    the same book (the book with more pairs; FanDuel on a tie, which refunds a pushed leg)."""
    used = set()
    for s in old.loc[old["track"] == "nfl_wong_teaser", "legs"].dropna():
        used |= {leg["game_id"] for leg in json.loads(s)}
    by_book: dict = {bk: [] for bk in MY_BOOKS}
    for ko, g in sorted(live, key=lambda x: (x[0], str(x[1]["id"]))):
        if str(g["id"]) in used:
            continue
        for bk, leg in teaser_legs(g).items():
            by_book[bk].append({**leg, "kickoff": ko.strftime("%Y-%m-%dT%H:%MZ"), "week": g.get("week"), "label": _label(g)})
    book = max(by_book, key=lambda bk: (len(by_book[bk]) // 2, bk == "fanduel"))
    legs = by_book[book]
    rows = []
    for a, b in zip(legs[0::2], legs[1::2]):
        pair = [{k: x[k] for k in ("game_id", "side", "team", "line", "teased", "kickoff")} for x in (a, b)]
        rows.append({"row_id": f"nfl_wong_teaser|{a['game_id']}+{b['game_id']}", "track": "nfl_wong_teaser", "league": "nfl", "at": stamp,
                     "season": season, "week": a["week"], "game_id": a["game_id"], "data_id": a["game_id"], "kickoff": a["kickoff"],
                     "label": f"{a['label']}; {b['label']}", "market": "Teaser", "side": "pair",
                     "pick": f"{a['team']} {a['teased']:+g} and {b['team']} {b['teased']:+g}", "book": book, "line": None,
                     "odds": TEASER_ODDS, "model": None, "opener": None, "legs": json.dumps(pair, separators=(",", ":")), "built": built})
    return rows


# ---------------------------------------------------------------- props (NFL)

STAT_LABEL = {"receptions": "receptions", "rec_yds": "receiving yards", "rush_yds": "rushing yards", "carries": "rushing attempts"}


def _prop_weeks(props_dir: Path, season: int) -> dict:
    out = {}
    for f in sorted(props_dir.glob(f"{season}_w*.json")):
        try:
            out[int(f.stem.split("_w")[1])] = f
        except ValueError:
            continue
    return out


def _load_props(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text()).get("props") or []
    except (OSError, ValueError):
        return []


def prop_under(p: dict) -> tuple[str, float] | None:
    """(book, under price) when this Props record meets the track's rule, else None."""
    if p.get("alt") or p.get("snap") != "flag" or p.get("s") not in PROP_STATS:
        return None
    pos = PROP_STATS[p["s"]]
    if pos and p.get("pos") != pos:
        return None
    po, pu = p.get("po"), p.get("pu")
    if not _ok(po) or not _ok(pu) or po + pu <= 0:
        return None
    from .propboard import AGREE
    if po / (po + pu) >= AGREE:
        return None
    best = None
    for bk, pr in (p.get("bk") or {}).items():
        if bk.lower() in MY_BOOKS and isinstance(pr, (list, tuple)) and len(pr) > 1 and _ok(pr[1]):
            if best is None or _dec(pr[1]) > _dec(best[1]):
                best = (bk.lower(), float(pr[1]))
    return best


def record_props(season: int, built: str | None = None, now: dt.datetime | None = None, root: Path = DIR,
                 props_dir: Path = PROPS) -> int:
    """Publish the prop unders the newest two weeks of the Props section meet now."""
    if "nfl_prop_under" in HELD:
        return 0
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    built = built or os.environ.get("GITHUB_SHA", "")[:7] or stamp
    path = picks_path("nfl", root)
    have = set(_read(path, COLS)["row_id"])
    weeks = _prop_weeks(props_dir, season)
    rows = []
    for wk in sorted(weeks)[-2:]:
        for p in _load_props(weeks[wk]):
            hit = prop_under(p)
            if not hit or not p.get("ko"):
                continue
            ko = pd.Timestamp(p["ko"])
            ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko
            rid = f"nfl_prop_under|{p['g']}|{p['id']}|{p['s']}"
            if ko <= pd.Timestamp(now) or rid in have:
                continue
            away, home = (str(p["g"]).split("_") + ["", ""])[2:4]
            rows.append({"row_id": rid, "track": "nfl_prop_under", "league": "nfl", "at": stamp, "season": season, "week": wk,
                         "game_id": p["g"], "data_id": p["g"], "kickoff": ko.strftime("%Y-%m-%dT%H:%MZ"),
                         "label": f"{p.get('n')} ({p.get('t')}), {STAT_LABEL[p['s']]}", "market": p["s"], "side": "under",
                         "pick": f"Under {float(p['L']):g}", "book": hit[0], "line": float(p["L"]), "odds": hit[1],
                         "model": round(p["po"] / (p["po"] + p["pu"]), 4), "opener": None,
                         "legs": json.dumps({"player_id": p["id"], "game": f"{away} at {home}"}, separators=(",", ":")), "built": built})
            have.add(rid)
    _append(path, rows, COLS)
    return len(rows)


def grade_props(season: int, now: dt.datetime | None = None, root: Path = DIR, props_dir: Path = PROPS) -> int:
    """Grade prop picks from the Props section's own results (the over's result, reversed)."""
    picks = _read(picks_path("nfl", root), COLS)
    picks = picks[(picks["track"] == "nfl_prop_under") & (picks["season"].astype(float) == season)]
    done = set(_read(results_path("nfl", root), RES_COLS)["row_id"])
    picks = picks[~picks["row_id"].isin(done)]
    if picks.empty:
        return 0
    now = now or dt.datetime.now(dt.timezone.utc)
    weeks = _prop_weeks(props_dir, season)
    cache: dict = {}
    rows = []
    for r in picks.itertuples(index=False):
        wk = int(float(r.week))
        if wk not in weeks:
            continue
        if wk not in cache:
            cache[wk] = {(p["g"], p["id"], p["s"], float(p["L"])): p for p in _load_props(weeks[wk]) if not p.get("alt")}
        pid = json.loads(r.legs)["player_id"]
        p = cache[wk].get((r.game_id, pid, r.market, float(r.line)))
        res = (p or {}).get("res")
        if res not in ("win", "loss", "push", "void"):
            continue
        result = {"loss": "W", "win": "L", "push": "P", "void": "V"}[res]
        units = round(_dec(r.odds) - 1, 4) if result == "W" else -1.0 if result == "L" else 0.0
        clv = None
        if _ok(p.get("cfo")) and _ok(p.get("cL")) and float(p["cL"]) == float(r.line):
            clv = round((1 - float(p["cfo"])) * _dec(r.odds) - 1, 4)
        rows.append({"row_id": r.row_id, "graded_at": now.strftime("%Y-%m-%d %H:%M"), "home_score": None, "away_score": None,
                     "result": result, "units": units, "close_line": p.get("cL"), "close_prob": (1 - float(p["cfo"])) if _ok(p.get("cfo")) else None, "clv": clv})
    _append(results_path("nfl", root), rows, RES_COLS)
    return len(rows)


# ---------------------------------------------------------------- grading (game lines, teasers)

def grade_games(league: str, scores: dict, close: dict | None = None, curve=None, now: dt.datetime | None = None, root: Path = DIR) -> int:
    """Grade game-line picks the way the ledger grades its rows (ledger.grade_frame: the result,
    units at the published price, closing line value against the sharp close), and the teasers."""
    picks = _read(picks_path(league, root), COLS)
    if picks.empty:
        return 0
    rp = results_path(league, root)
    lines = picks[picks["market"].isin(["Spread", "Total", "Moneyline"])].copy()
    lines["odds_net"] = lines["odds"]
    lines["anchor_spread"] = np.nan
    lines["anchor_total"] = np.nan
    n = L.grade_frame(lines, rp, scores, close, curve, now) if len(lines) else 0
    return n + _grade_teasers(picks[picks["market"] == "Teaser"], rp, scores, now)


def _grade_teasers(picks: pd.DataFrame, rp: Path, scores: dict, now: dt.datetime | None) -> int:
    if picks.empty:
        return 0
    done = set(_read(rp, RES_COLS)["row_id"])
    stamp = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%d %H:%M")
    rows = []
    for r in picks.itertuples(index=False):
        if r.row_id in done:
            continue
        legs = json.loads(r.legs)
        out = []
        for leg in legs:
            sc = scores.get(leg["game_id"])
            if sc is None or not _ok(sc[0]) or not _ok(sc[1]):
                out = None
                break
            own = (float(sc[0]) - float(sc[1])) * (1 if leg["side"] == "home" else -1)
            d = own + float(leg["teased"])
            out.append("P" if d == 0 else "W" if d > 0 else "L")
        if out is None:
            continue
        result = "L" if "L" in out else "P" if "P" in out else "W"       # a lost leg loses; else a pushed leg refunds the pair
        units = round(_dec(r.odds) - 1, 4) if result == "W" else -1.0 if result == "L" else 0.0
        rows.append({"row_id": r.row_id, "graded_at": stamp, "home_score": None, "away_score": None, "result": result, "units": units,
                     "close_line": None, "close_prob": None, "clv": None})
    _append(rp, rows, RES_COLS)
    return len(rows)


# ---------------------------------------------------------------- the record

def _rec(x: pd.DataFrame) -> dict:
    """One record: picks, graded, W-L-P, units and dollars, the return with its standard error,
    closing line value with its standard error. A voided prop is no action and not counted."""
    x = x[x["result"] != "V"]
    g = x[x["result"].isin(["W", "L", "P"])]
    w, l, p = int((g.result == "W").sum()), int((g.result == "L").sum()), int((g.result == "P").sum())
    u = g["units"].astype(float)
    cv = g["clv"].astype(float).dropna()
    return {"n": int(len(x)), "graded": int(len(g)), "open": int(len(x) - len(g)), "w": w, "l": l, "p": p,
            "win_pct": round(w / (w + l), 4) if w + l else None,
            "units": round(float(u.sum()), 2) if len(g) else 0.0, "dollars": round(float(u.sum()) * UNIT, 2) if len(g) else 0.0,
            "roi": round(float(u.mean()), 4) if len(g) else None,
            "roi_se": round(float(u.std(ddof=1) / np.sqrt(len(g))), 4) if len(g) > 1 else None,
            "clv_n": int(len(cv)), "clv": round(float(cv.mean()), 4) if len(cv) else None,
            "clv_se": round(float(cv.std(ddof=1) / np.sqrt(len(cv))), 4) if len(cv) > 1 else None}


def _close_units(r) -> float | None:
    """The same side at the sharp book's closing line and -110, for someone betting late."""
    if r["market"] not in ("Spread", "Total") or not _ok(r.get("close_line")) or not _ok(r.get("home_score")) or not _ok(r.get("away_score")):
        return None
    margin, total = float(r["home_score"]) - float(r["away_score"]), float(r["home_score"]) + float(r["away_score"])
    line = float(r["close_line"])
    if r["market"] == "Spread":
        line = -line if r["side"] == "home" else line           # the close is stored home-favoured-positive
    res, u = L.settle(r["market"], r["side"], line, margin, total)
    return round(100 / 110, 4) if res == "W" else u


def _close_own(r) -> float | None:
    """The closing line in the pick's own terms (a spread as the side's handicap)."""
    if not _ok(r.get("close_line")):
        return None
    cl = float(r["close_line"])
    return (-cl if r["side"] == "home" else cl) if r["market"] == "Spread" else cl


def summary(league: str, root: Path = DIR, now: dt.datetime | None = None) -> dict:
    """Everything the page shows for one league: each track's record, the week-by-week record
    with the rolling dollar total, and every pick."""
    now = now or dt.datetime.now(dt.timezone.utc)
    picks = _read(picks_path(league, root), COLS)
    res = _read(results_path(league, root), RES_COLS).drop_duplicates("row_id")
    x = picks.merge(res.drop(columns=["graded_at"]), on="row_id", how="left")
    for c in ("units", "clv", "odds", "line", "close_line", "home_score", "away_score"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.sort_values(["kickoff", "at", "row_id"], kind="stable").reset_index(drop=True)
    # a bet on two tracks is one bet in the totals: the first to publish it carries it
    first = x.sort_values(["at", "row_id"], kind="stable").drop_duplicates(["game_id", "market", "side"])["row_id"]
    x["dup"] = ~x["row_id"].isin(set(first)) & x["market"].isin(["Spread", "Total", "Moneyline"])
    one = x[~x["dup"]]
    tracks = []
    for key, lg, name, rule, before in TRACKS:
        if lg != league:
            continue
        rec = _rec(x[x["track"] == key])
        tracks.append({"key": key, "name": name, "rule": rule, "before": before, "status": STATUS[key], **rec,
                       "to_go": max(0, NEED - rec["graded"])})
    weeks, run = [], 0.0
    for (s, w), y in one.groupby(["season", "week"], sort=True):
        rec = _rec(y)
        run = round(run + rec["dollars"], 2)
        weeks.append({"season": int(s), "week": int(w), **rec, "running": run})
    rows = []
    for r in x.to_dict("records"):
        graded = r["result"] in ("W", "L", "P", "V")
        rows.append({"id": r["row_id"], "track": r["track"], "at": r["at"], "season": int(r["season"]), "week": int(float(r["week"])),
                     "kickoff": r["kickoff"], "label": r["label"], "market": r["market"], "pick": r["pick"], "book": r["book"],
                     "odds": None if not _ok(r["odds"]) else float(r["odds"]),
                     "result": r["result"] if graded else None,
                     "units": float(r["units"]) if graded and _ok(r["units"]) else None,
                     "dollars": round(float(r["units"]) * UNIT, 2) if graded and _ok(r["units"]) else None,
                     "close": _close_own(r), "close_units": _close_units(r),
                     "clv": float(r["clv"]) if _ok(r["clv"]) else None, "dup": bool(r["dup"]),
                     "game": json.loads(r["legs"]).get("game") if r["track"] == "nfl_prop_under" and isinstance(r["legs"], str) else None})
    return {"league": league, "built": now.strftime("%Y-%m-%d %H:%M UTC"), "unit": UNIT, "need": NEED,
            "since": str(picks["at"].min())[:10] if len(picks) else None,
            "all": _rec(one), "tracks": tracks, "weeks": weeks, "picks": rows}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write(league: str, root: Path = DIR, out: Path = OUT, now: dt.datetime | None = None) -> Path:
    """docs/recommended/<league>.js: the summary as a script, so the page reads it from disk too."""
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_clean(summary(league, root, now)), separators=(",", ":"), allow_nan=False)
    path = out / f"{league}.js"
    path.write_text(f"window.RECOMMENDED=window.RECOMMENDED||{{}};window.RECOMMENDED.{league}={body};\n")
    return path


def update_games(slate_games: list[dict], league: str, season: int, kickoffs: dict, scores: dict, close: dict | None = None,
                 curve=None, built: str | None = None, root: Path = DIR, out: Path = OUT) -> str:
    """One board build's pass: publish, grade, rewrite the page's file. Returns a line for the log."""
    n = record_games(slate_games, league, season, kickoffs, built, root=root)
    g = grade_games(league, scores, close, curve, root=root)
    if league == "nfl":
        n += record_props(season, built, root=root)
        g += grade_props(season, root=root)
    write(league, root, out)
    return f"{n} picks published, {g} graded"


def update_props(season: int, root: Path = DIR, out: Path = OUT) -> str:
    """The props run's pass (run_props.py, after the Props section is rebuilt)."""
    n = record_props(season, root=root)
    g = grade_props(season, root=root)
    write("nfl", root, out)
    return f"{n} prop picks published, {g} graded"
