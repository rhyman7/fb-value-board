"""Past game lines (Pinnacle, DraftKings, FanDuel) from The Odds API's historical endpoint.

Why. The boards' records are graded against the nflverse consensus (NFL) or one book's line
(college). The live market log (market.py) has Pinnacle only from Oct 30, 2026. This fetches
past seasons' lines at two kinds of moment so experiments/pinnacle_grade.py can grade the
"price beats Pinnacle" tier, the opener-gap edge and closing line value against Pinnacle's
real numbers.

What. One call to /v4/historical/sports/{sport}/odds returns every listed game's lines as
they stood at that minute (the snapshot at or before it). Cost: 10 x (markets returned) x 1
region (the three named books count as one), so 30 credits a call with spreads, totals and
moneylines. Two kinds of snapshot per league-season:

  open    Monday 12:00 UTC before each game (the latest one at least OPEN_LEAD before
          kickoff): the week's lines a day after they open, before most of the week's moves.
          One call covers the whole week.
  close   CLOSE_BEFORE before the first kickoff of each cluster of games starting within
          CLOSE_WINDOW minutes of each other (NFL 30: the Sunday 4:05 and 4:25 games share one;
          college 120, so a college "close" is 10 minutes to two hours before kickoff).

`plan` lists the snapshots and `estimate` their cost before anything is spent.

Saved as data/market/history/<league>/<season>/<kind>_<YYYYMMDDTHHMM>.csv (the requested
time, UTC), with market.COLS: `snapshot` is the API's own timestamp. A header-only file means
the API had nothing (or answered 404) and is never asked for again; any other error skips the
snapshot and the run goes on. A file that exists is never fetched again; asking for markets it does not hold fetches only those,
saved beside it as <kind>_<YYYYMMDDTHHMM>_<markets>.csv. A stopped run is
resumed by starting a NEW run with the same inputs (never GitHub's Re-run, which checks out
the old commit without the saved files). Games already under way at a snapshot carry live
prices; `load` drops them (commence at or before the snapshot).
"""
from __future__ import annotations

import csv

import pandas as pd

from . import config as C, market as M, oddsapi

HIST = C.DATA / "market" / "history"
BOOKS = ["pinnacle", "draftkings", "fanduel"]
MARKETS = ["spreads", "totals", "h2h"]
OPEN_HOUR = 12                                   # Monday, UTC
OPEN_LEAD = pd.Timedelta(hours=6)                # the open snapshot is at least this long before kickoff
CLOSE_BEFORE = pd.Timedelta(minutes=10)
CLOSE_WINDOW = {"nfl": 30, "cfb": 120}           # minutes; see the module docstring
FIRST_SEASON = 2020                              # the historical game odds start in June 2020


def schedule(league: str, season: int) -> pd.DataFrame:
    """game_id and kickoff (UTC) of every game the board prices that season."""
    if league == "nfl":
        from . import propbackfill
        return propbackfill.schedule(season)[["game_id", "home_team", "away_team", "kickoff"]].reset_index(drop=True)
    g = pd.read_csv(C.DATA / "cfb" / "cache" / f"games_{season}.csv")
    g = g[g["spread_line"].notna()]                         # FBS games with a line: the ones the board prices
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True, errors="coerce")
    return g[g["kickoff"].notna()][["game_id", "home_team", "away_team", "kickoff"]].sort_values("kickoff").reset_index(drop=True)


def open_time(kickoff: pd.Timestamp) -> pd.Timestamp:
    """The latest Monday OPEN_HOUR:00 UTC at least OPEN_LEAD before kickoff."""
    k = pd.Timestamp(kickoff).tz_convert("UTC")
    t = (k - OPEN_LEAD).normalize()
    t = t - pd.Timedelta(days=t.weekday()) + pd.Timedelta(hours=OPEN_HOUR)
    return t if t <= k - OPEN_LEAD else t - pd.Timedelta(days=7)


def close_times(kickoffs, window_min: int) -> list[pd.Timestamp]:
    """One close per cluster of kickoffs within `window_min` minutes of the cluster's first."""
    k = sorted(pd.to_datetime(pd.Series(list(kickoffs)), utc=True).dropna().unique())
    out, i = [], 0
    while i < len(k):
        first = pd.Timestamp(k[i])
        out.append(first - CLOSE_BEFORE)
        i += 1
        while i < len(k) and pd.Timestamp(k[i]) < first + pd.Timedelta(minutes=window_min):
            i += 1
    return out


def plan(games: pd.DataFrame, league: str, window_min: int | None = None, kinds=("open", "close")) -> pd.DataFrame:
    """The snapshots to take: kind and requested time (UTC), in time order."""
    rows = []
    if "open" in kinds:
        rows += [("open", t) for t in sorted({open_time(k) for k in games["kickoff"]})]
    if "close" in kinds:
        rows += [("close", t) for t in close_times(games["kickoff"], window_min or CLOSE_WINDOW[league])]
    p = pd.DataFrame(rows, columns=["kind", "at"])
    return p.sort_values(["at", "kind"]).reset_index(drop=True)


def file_name(kind: str, at: pd.Timestamp, markets=None) -> str:
    """kind_YYYYMMDDTHHMM.csv; with markets, a second file for the same moment that adds
    markets an earlier run did not fetch (kind_YYYYMMDDTHHMM_spreads.csv)."""
    tail = f"_{'-'.join(markets)}" if markets else ""
    return f"{kind}_{pd.Timestamp(at):%Y%m%dT%H%M}{tail}.csv"


def missing_markets(d, kind: str, at: pd.Timestamp, markets) -> list[str]:
    """The requested markets no saved file for this moment holds yet. A header-only file
    (the API had nothing) settles the moment for good."""
    stem = file_name(kind, at)[:-4]
    have = set()
    for f in d.glob(f"{stem}*.csv"):
        x = pd.read_csv(f, usecols=["market"])
        if x.empty:
            return []
        have |= set(x["market"].dropna())
    return [m for m in markets if m not in have]


def target(d, kind: str, at: pd.Timestamp, todo: list[str]):
    """Where a fetch of `todo` is saved: the plain name if nothing is saved for the moment yet."""
    base = d / file_name(kind, at)
    return base if not base.exists() else d / file_name(kind, at, todo)


def estimate(league: str, season: int, markets=MARKETS, window_min: int | None = None, kinds=("open", "close")) -> dict:
    """Snapshots still to fetch and their worst-case cost (every market returned)."""
    p = plan(schedule(league, season), league, window_min, kinds)
    d = HIST / league / str(season)
    miss = [missing_markets(d, r.kind, r.at, markets) for r in p.itertuples()]
    todo = p[[bool(m) for m in miss]]
    return {"league": league, "season": season, "snapshots": int(len(p)), "to_fetch": int(len(todo)),
            "open": int((todo.kind == "open").sum()), "close": int((todo.kind == "close").sum()),
            "credits_per_snapshot": 10 * len(markets), "credits": int(10 * sum(len(m) for m in miss))}


def _iso(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows is None or rows.empty:
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(M.COLS)
        return
    rows[M.COLS].to_csv(path, index=False)


def run(client, league: str, season: int, max_credits: int, markets=MARKETS, books=BOOKS,
        window_min: int | None = None, kinds=("open", "close"), verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    if season < FIRST_SEASON:
        raise ValueError(f"historical game odds start in {FIRST_SEASON}")
    out_dir = HIST / league / str(season)
    p = plan(schedule(league, season), league, window_min, kinds)
    per = 10 * len(markets)
    spent, fetched, skipped, probe_ok = 0, 0, 0, False
    say(f"  {league} {season}: {len(p)} snapshots planned, up to {per} credits each")
    for r in p.itertuples():
        todo = missing_markets(out_dir, r.kind, r.at, markets)
        if not todo:
            continue
        path, per = target(out_dir, r.kind, r.at, todo), 10 * len(todo)
        if spent + per > max_credits:
            say(f"  stopping: {spent} credits spent, cap {max_credits}")
            return {"spent": spent, "snapshots": fetched, "skipped": skipped, "complete": False}
        try:
            body = client.historical_sport_odds(M.SPORTS[league], list(todo), _iso(r.at), list(books))
        except oddsapi.BudgetError as e:
            say(f"  {e}")
            return {"spent": spent, "snapshots": fetched, "skipped": skipped, "complete": False}
        except oddsapi.ApiError as e:
            if "HTTP 404" in str(e):
                _write(path, None)
                say(f"  {r.kind} {_iso(r.at)}: not available (404), saved empty")
            else:
                say(f"  {r.kind} {_iso(r.at)}: {e}; skipped")
            skipped += 1
            continue
        cost = client.last_cost
        spent += cost
        if not probe_ok and cost > 0:
            if cost > per:
                _write(path, M.flatten((body or {}).get("data") or [], league, (body or {}).get("timestamp") or _iso(r.at)))
                say(f"  first snapshot cost {cost} credits, expected at most {per}: stopping to check")
                return {"spent": spent, "snapshots": fetched + 1, "skipped": skipped, "complete": False, "probe_cost": cost}
            probe_ok = True
        snap = (body or {}).get("timestamp") or _iso(r.at)
        rows = M.flatten((body or {}).get("data") or [], league, snap)
        _write(path, rows)
        fetched += 1
        say(f"  {r.kind} {_iso(r.at)} (snapshot {snap}): {rows['event_id'].nunique() if len(rows) else 0} games, "
            f"{len(rows)} prices, {cost} credits (total {spent}, left {client.remaining})")
    if skipped:
        say(f"  {skipped} snapshot(s) skipped (see above)")
    return {"spent": spent, "snapshots": fetched, "skipped": skipped, "complete": True}


def load(league: str, season: int) -> pd.DataFrame:
    """Every saved snapshot of a league-season: market.COLS plus kind and requested (UTC).
    Games already under way at the snapshot (live prices) are dropped."""
    d = HIST / league / str(season)
    parts = []
    for f in sorted(d.glob("*_*.csv")):
        kind, stamp = f.stem.split("_", 1)
        stamp = stamp.split("_", 1)[0]                     # kind_STAMP_spreads.csv adds markets to kind_STAMP.csv
        if kind not in ("open", "close"):
            continue
        x = pd.read_csv(f)
        if len(x):
            parts.append(x.assign(kind=kind, requested=pd.to_datetime(stamp, format="%Y%m%dT%H%M", utc=True)))
    if not parts:
        return pd.DataFrame(columns=M.COLS + ["kind", "requested"])
    x = pd.concat(parts, ignore_index=True)
    snap = pd.to_datetime(x["snapshot"], utc=True, errors="coerce")
    start = pd.to_datetime(x["commence"], utc=True, errors="coerce")
    return x[start > snap].reset_index(drop=True)
