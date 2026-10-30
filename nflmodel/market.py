"""Game lines from several books, including a sharp reference, for both boards.

Why. The boards' projections are anchored to one number: the nflverse consensus (NFL) or
ESPN's scoreboard book (college). Neither is a sharp price. Pinnacle is the low-margin book
that most of the market is priced from, so anchoring to its no-vig price and shopping
DraftKings, FanDuel, BetMGM and Kalshi for anything that beats it is the standard way to
find +EV bets without needing the model to beat the market at all. The model then adds on top.

What. One call per league to The Odds API's /sports/{sport}/odds (h2h, spreads, totals; the
five books named below) costs 3 credits (markets x one region: up to ten named books count
as one region), so a snapshot three times a day for both leagues is about 20 credits a day.
Every row that changed since the last snapshot is appended to data/market/<league>_<YYYY-MM>.csv:

    snapshot, league, event_id, commence, home_name, away_name, book, market, side, line, price, updated

`side` is home/away (spreads, h2h) or over/under (totals); `line` is the outcome's point as
the book posts it for that side (home -3.5 means the home side's spread is -3.5), blank for
h2h. `price` is American odds (Kalshi's contract price converted by the API; fees are not
in it, see odds.kalshi_effective_odds).

How the boards use it (slate.build, cfbmodel/board.py): the latest row per book for each
upcoming game; Pinnacle's line and two-way prices become the anchor the projection is
blended from when Pinnacle has the game, and every other book's price is scored against the
projection at its own line (line shopping, with the key-number curve carrying a half point
between lines).

The key is the ODDS_API_KEY secret (shared with the player props); every call is logged to
data/props/api_usage.csv by the client. A failed or missing snapshot changes nothing: the
boards fall back to the lines they already use.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DIR = C.DATA / "market"
SPORTS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}
BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "kalshi"]
SHARP = "pinnacle"                       # the anchor, when it has the game
RETAIL = ["draftkings", "fanduel", "betmgm", "kalshi"]   # where the bets are placed
MARKETS = ["h2h", "spreads", "totals"]
COLS = ["snapshot", "league", "event_id", "commence", "home_name", "away_name",
        "book", "market", "side", "line", "price", "updated"]
# Board column for each (book market, side): the same names slate.LINE_COLS uses.
LINE_COL = {("spreads", "home"): "home_spread_odds", ("spreads", "away"): "away_spread_odds",
            ("totals", "over"): "over_odds", ("totals", "under"): "under_odds",
            ("h2h", "home"): "home_moneyline", ("h2h", "away"): "away_moneyline"}

NFL_NAME_TO_ABBR = {v: k for k, v in C.TEAM_NAMES.items()}
# College names The Odds API spells differently from ESPN's display names.
CFB_ALIASES = {
    "hawaii rainbow warriors": "hawai'i rainbow warriors", "ul monroe warhawks": "louisiana-monroe warhawks",
    "louisiana monroe warhawks": "louisiana-monroe warhawks", "app state mountaineers": "appalachian state mountaineers",
    "umass minutemen": "massachusetts minutemen", "miami oh redhawks": "miami (oh) redhawks",
    "miami fl hurricanes": "miami hurricanes", "san jose state spartans": "san josé state spartans",
    "southern miss golden eagles": "southern mississippi golden eagles", "uconn huskies": "connecticut huskies",
    "sam houston bearkats": "sam houston state bearkats", "fiu panthers": "florida international panthers",
    "usf bulls": "south florida bulls", "utsa roadrunners": "ut san antonio roadrunners",
}


def _norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ()\-]", "", str(s or "").lower())).strip()


# --------------------------------------------------------------------------
# Fetching and storing snapshots
# --------------------------------------------------------------------------

def fetch(client, league: str, books: list[str] | None = None, markets: list[str] | None = None) -> list[dict]:
    """Every upcoming game's lines from the books (one call; markets x 1 region credits)."""
    sport = SPORTS[league]
    params = {"markets": ",".join(markets or MARKETS), "bookmakers": ",".join(books or BOOKS),
              "oddsFormat": "american", "dateFormat": "iso"}
    return client.sport_odds(sport, params)


def flatten(events: list[dict], league: str, snapshot: str) -> pd.DataFrame:
    """One row per event x book x market x side from a /sports/{sport}/odds response."""
    rows = []
    for ev in events or []:
        home, away = ev.get("home_team"), ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    if o.get("price") is None:
                        continue
                    name = str(o.get("name", ""))
                    if mk.get("key") == "totals":
                        side = name.lower()
                        if side not in ("over", "under"):
                            continue
                    else:
                        side = "home" if name == home else "away" if name == away else None
                        if side is None:
                            continue
                    rows.append({"snapshot": snapshot, "league": league, "event_id": ev.get("id"),
                                 "commence": ev.get("commence_time"), "home_name": home, "away_name": away,
                                 "book": bk.get("key"), "market": mk.get("key"), "side": side,
                                 "line": o.get("point"), "price": float(o["price"]), "updated": mk.get("last_update")})
    return pd.DataFrame(rows, columns=COLS)


def _commence(ev: dict):
    try:
        t = pd.Timestamp(ev.get("commence_time"))
        return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
    except Exception:
        return None


def _month_path(league: str, when: str, root: Path = DIR) -> Path:
    return root / f"{league}_{when[:7]}.csv"


def load(league: str, root: Path = DIR, months: int = 2) -> pd.DataFrame:
    """The stored rows for a league from the last `months` monthly files."""
    files = sorted(root.glob(f"{league}_????-??.csv"))[-months:]
    if not files:
        return pd.DataFrame(columns=COLS)
    out = pd.concat([pd.read_csv(f, dtype={"event_id": str}) for f in files], ignore_index=True)
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out[COLS]


def store(rows: pd.DataFrame, league: str, root: Path = DIR) -> int:
    """Append the rows whose line or price differs from the last stored row for the same
    event, book, market and side (or that are new). Returns the number appended."""
    if rows.empty:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    old = load(league, root)
    key = ["event_id", "book", "market", "side"]
    if len(old):
        last = old.sort_values("snapshot").groupby(key).tail(1)[key + ["line", "price"]].rename(columns={"line": "l0", "price": "p0"})
        m = rows.merge(last, on=key, how="left")
        same = (np.isclose(m["line"].astype(float), m["l0"].astype(float), equal_nan=True)
                & np.isclose(m["price"].astype(float), m["p0"].astype(float), equal_nan=True))
        rows = m.loc[~same, COLS]
    if rows.empty:
        return 0
    path = _month_path(league, str(rows["snapshot"].iloc[0]), root)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return int(len(rows))


def snapshot(client, league: str, now: dt.datetime | None = None, root: Path = DIR) -> int:
    """Fetch and store one snapshot. Returns rows appended; -1 if the fetch failed."""
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    try:
        events = fetch(client, league)
    except Exception as exc:                      # budget, network, key
        print(f"  market snapshot {league}: skipped ({exc})")
        return -1
    # only games not yet kicked off: a book's in-game line is not a pre-game quote
    events = [e for e in (events or []) if _commence(e) is None or _commence(e) > now]
    rows = flatten(events, league, stamp)
    n = store(rows, league, root)
    print(f"  market snapshot {league}: {len(events or [])} games, {len(rows)} quotes, {n} new rows")
    return n


# --------------------------------------------------------------------------
# Matching to the boards' games and reading the latest quotes
# --------------------------------------------------------------------------

def match_games(rows: pd.DataFrame, games: pd.DataFrame, league: str, names: dict | None = None) -> pd.DataFrame:
    """Add game_id to each row: the board game with the same two teams whose kickoff is
    within two days of the event's commence time. `games` has game_id, home_team, away_team
    and kickoff (UTC) or gameday. NFL rows match on C.TEAM_NAMES; college rows need `names`,
    the board's team display name -> data team name (cfbmodel teams: name -> school)."""
    if rows.empty or games.empty:
        return rows.assign(game_id=pd.Series(dtype=object))
    r = rows.copy()
    if league == "nfl":
        r["h"], r["a"] = r["home_name"].map(NFL_NAME_TO_ABBR), r["away_name"].map(NFL_NAME_TO_ABBR)
    else:
        look = {_norm(k): v for k, v in (names or {}).items()}
        # aliases work either way round: whichever spelling the board uses, the other matches it
        for k, v in CFB_ALIASES.items():
            nk, nv = _norm(k), _norm(v)
            if nv in look and nk not in look:
                look[nk] = look[nv]
            elif nk in look and nv not in look:
                look[nv] = look[nk]
        r["h"], r["a"] = r["home_name"].map(lambda s: look.get(_norm(s))), r["away_name"].map(lambda s: look.get(_norm(s)))
    g = games.copy()
    when = g["kickoff"] if "kickoff" in g else pd.to_datetime(g["gameday"], utc=True)
    g = g.assign(when=pd.to_datetime(when, utc=True))[["game_id", "home_team", "away_team", "when"]]
    r["commence_ts"] = pd.to_datetime(r["commence"], utc=True, errors="coerce")
    r = r.reset_index(drop=True)
    r["_i"] = r.index
    # candidates: the teams as listed, and the other way round (a neutral-site game the book
    # lists the other way). The same two teams can meet twice in a season, so the candidate
    # nearest the event's kickoff wins, within two days.
    straight = r.merge(g, left_on=["h", "a"], right_on=["home_team", "away_team"]).assign(flip=False)
    swapped = r.merge(g, left_on=["a", "h"], right_on=["home_team", "away_team"]).assign(flip=True)
    cand = pd.concat([straight, swapped], ignore_index=True)
    cand["gap"] = (cand["commence_ts"] - cand["when"]).abs()
    cand = cand[cand["gap"] <= pd.Timedelta(days=2)].sort_values(["_i", "flip", "gap"]).groupby("_i").head(1)
    out = r[COLS].copy()
    out["game_id"] = out.index.map(dict(zip(cand["_i"], cand["game_id"])))
    flip = out.index.map(dict(zip(cand["_i"], cand["flip"]))).fillna(False).astype(bool)
    if flip.any():                                            # the book's home is the board's away
        out.loc[flip, "side"] = out.loc[flip, "side"].map({"home": "away", "away": "home"}).fillna(out.loc[flip, "side"])
    out["game_id"] = out["game_id"].where(out["game_id"].notna(), None)
    return out


def unmatched(rows: pd.DataFrame) -> list[str]:
    """Team names the books use that the board could not place (for the run log)."""
    if rows.empty or "game_id" not in rows:
        return []
    miss = rows[rows["game_id"].isna()]
    return sorted(set(miss["home_name"]) | set(miss["away_name"]))


def latest(rows: pd.DataFrame) -> dict:
    """{game_id: {book: {spread_line, home_spread_odds, ..., 'updated'}}} from the latest row
    per side. spread_line follows the boards' convention (positive = home favoured)."""
    out: dict = {}
    if rows.empty or "game_id" not in rows:
        return out
    r = rows[rows["game_id"].notna()].sort_values("snapshot")
    r = r.groupby(["game_id", "book", "market", "side"]).tail(1)
    for (gid, book), q in r.groupby(["game_id", "book"]):
        d: dict = {"updated": str(q["snapshot"].max())}
        for x in q.itertuples(index=False):
            col = LINE_COL.get((x.market, x.side))
            if col is None:
                continue
            d[col] = float(x.price)
            if x.market == "spreads" and x.side == "home" and pd.notna(x.line):
                d["spread_line"] = -float(x.line)
            if x.market == "spreads" and x.side == "away" and pd.notna(x.line):
                d.setdefault("spread_line", float(x.line))
                d["away_spread_point"] = float(x.line)
            if x.market == "totals" and pd.notna(x.line):
                d["total_line"] = float(x.line)
        out.setdefault(gid, {})[book] = d
    return out


def quotes_for(games: pd.DataFrame, league: str, names: dict | None = None, root: Path = DIR,
               before: pd.Timestamp | None = None) -> dict:
    """The stored quotes for the games given: {"latest": latest(rows), "first": first_seen(rows)},
    both empty when nothing is stored or matching fails. `before` keeps only snapshots taken
    before that time (for closing lines)."""
    empty = {"latest": {}, "first": {}, "path": {}}
    try:
        rows = load(league, root)
        if before is not None and len(rows):
            rows = rows[pd.to_datetime(rows["snapshot"], utc=True) < before]
        if len(rows):                              # a game that has kicked off has no pre-game quote any more
            now = pd.Timestamp.now(tz="UTC") if before is None else before
            rows = rows[pd.to_datetime(rows["commence"], utc=True, errors="coerce") > now]
        rows = match_games(rows, games, league, names)
    except Exception as exc:
        print(f"  market quotes {league}: skipped ({exc})")
        return empty
    miss = unmatched(rows)
    if miss:
        print(f"  market quotes {league}: {len(miss)} unmatched names: {', '.join(miss[:8])}{' ...' if len(miss) > 8 else ''}")
    return {"latest": latest(rows), "first": first_seen(rows), "path": path(rows)}


def as_of(latest_quotes: dict) -> str | None:
    """The newest snapshot among the quotes (latest()), for the boards' "data as of" line."""
    stamps = [q.get("updated") for g in (latest_quotes or {}).values() for q in g.values() if q.get("updated")]
    return max(stamps) if stamps else None


def path(rows: pd.DataFrame, book: str = SHARP, max_points: int = 12) -> dict:
    """{game_id: [{"at": snapshot, "spread": ..., "total": ...}, ...]}: how `book`'s spread
    and total moved, one point per snapshot where either changed (home-positive spread),
    thinned to the first, the last and at most max_points in between."""
    out = {}
    if rows.empty or "game_id" not in rows:
        return out
    r = rows[rows["game_id"].notna() & (rows["book"] == book) & rows["line"].notna()]
    r = r[((r["market"] == "spreads") & (r["side"] == "home")) | ((r["market"] == "totals") & (r["side"] == "over"))]
    for gid, q in r.groupby("game_id"):
        pts, cur = [], {"spread": None, "total": None}
        for snap, s in q.sort_values("snapshot").groupby("snapshot", sort=True):
            for x in s.itertuples(index=False):
                if x.market == "spreads":
                    cur["spread"] = -float(x.line)
                else:
                    cur["total"] = float(x.line)
            if not pts or pts[-1]["spread"] != cur["spread"] or pts[-1]["total"] != cur["total"]:
                pts.append({"at": str(snap), **cur})
        if len(pts) > max_points + 2:
            step = (len(pts) - 2) / max_points
            mid = [pts[1 + int(i * step)] for i in range(max_points)]
            pts = [pts[0], *mid, pts[-1]]
        out[gid] = pts
    return out


def first_seen(rows: pd.DataFrame, book: str = SHARP) -> dict:
    """{game_id: {'spread': ..., 'total': ..., 'seen': 'YYYY-MM-DD HH:MM'}}: the first stored
    quote from `book` for each game (the opener as this log saw it)."""
    out = {}
    if rows.empty or "game_id" not in rows:
        return out
    r = rows[rows["game_id"].notna() & (rows["book"] == book)].sort_values("snapshot")
    for gid, q in r.groupby("game_id"):
        sp = q[(q["market"] == "spreads") & (q["side"] == "home") & q["line"].notna()]
        to = q[(q["market"] == "totals") & (q["side"] == "over") & q["line"].notna()]
        out[gid] = {"seen": str(min(q["snapshot"])),
                    "spread": -float(sp["line"].iloc[0]) if len(sp) else None,
                    "total": float(to["line"].iloc[0]) if len(to) else None}
    return out


def anchor(df: pd.DataFrame, quotes: dict, book: str = SHARP) -> tuple[pd.DataFrame, pd.Series]:
    """Replace the anchor line and prices with `book`'s where it quotes the game. Returns the
    frame and a Series of the anchor source per row ('pinnacle' or 'consensus')."""
    df = df.copy()
    src = pd.Series("consensus", index=df.index)
    for i, gid in zip(df.index, df["game_id"]):
        q = (quotes.get(gid) or {}).get(book)
        if not q:
            continue
        have_spread = "spread_line" in q and "home_spread_odds" in q and "away_spread_odds" in q
        have_total = "total_line" in q and "over_odds" in q and "under_odds" in q
        if not (have_spread and have_total):
            continue
        for c in ("spread_line", "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds"):
            df.at[i, c] = q[c]
        for c in ("home_moneyline", "away_moneyline"):
            if c in q:
                df.at[i, c] = q[c]
        src[i] = book
    return df, src


def closing_quotes(games: pd.DataFrame, league: str, kickoffs: dict, names: dict | None = None,
                   book: str = SHARP, root: Path = DIR) -> dict:
    """{game_id: quote} from `book`'s last stored snapshot before each game's kickoff
    (kickoffs: game_id -> UTC Timestamp). The closing line value of every bet is measured
    against this."""
    out: dict = {}
    try:
        rows = match_games(load(league, root, months=3), games, league, names)
    except Exception as exc:
        print(f"  closing quotes {league}: skipped ({exc})")
        return out
    if rows.empty:
        return out
    rows = rows[rows["game_id"].notna() & (rows["book"] == book)].copy()
    rows["ts"] = pd.to_datetime(rows["snapshot"], utc=True)
    for gid, q in rows.groupby("game_id"):
        ko = kickoffs.get(gid)
        if ko is None:
            continue
        before = q[q["ts"] < pd.Timestamp(ko)]
        if before.empty:
            continue
        out[gid] = latest(before.assign(game_id=gid)).get(gid, {}).get(book)
    return {k: v for k, v in out.items() if v}
