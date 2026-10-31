"""Past prop prices from The Odds API's historical endpoint, for testing against real lines.

For every game of a season, two snapshots, at the times the live run would take them:

  flag    the first daily run (12:47 UTC) with kickoff three days away or less
  close   ten minutes before kickoff

Saved as data/props/history/<season>/<game_id>_<kind>.csv (one row per book x market x
player x side x line; a header-only file means the snapshot had no prices). Files that
exist are never fetched again, so a run that stops can be restarted. Cost: one credit per
week for the event list, and 10 per market returned per snapshot.
"""
from __future__ import annotations

import csv
import datetime as dt

import pandas as pd

from . import config as C, oddsapi
from .data import _download

HIST = C.DATA / "props" / "history"
FLAG_HOUR = (12, 47)                 # the daily props run, UTC
CLOSE_BEFORE = pd.Timedelta(minutes=10)
COLS = ["snapshot", "event_id", "commence", "home_name", "away_name", "book", "market", "updated", "player", "side", "line", "price"]
YARDAGE = ["player_pass_yds", "player_rush_yds", "player_reception_yds"]


def schedule(season: int) -> pd.DataFrame:
    path = C.RAW / "games.csv"
    if not path.exists():
        _download(C.GAMES_URL, path)
    g = pd.read_csv(path)
    g = g[g["season"] == season].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(C.TEAM_ALIASES)
    # kickoff: gameday + gametime are US Eastern
    t = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"))
    g["kickoff"] = t.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return g[["game_id", "season", "week", "home_team", "away_team", "kickoff"]].sort_values("kickoff")


def flag_time(kickoff: pd.Timestamp) -> pd.Timestamp:
    start = kickoff - pd.Timedelta(days=3)
    t = start.normalize() + pd.Timedelta(hours=FLAG_HOUR[0], minutes=FLAG_HOUR[1])
    return t if t > start else t + pd.Timedelta(days=1)


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _save(path, snapshot, ev):
    rows = oddsapi.flatten(ev) if ev else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, snapshot=snapshot))
    return len(rows)


def _fetched(out_dir) -> dict:
    """(game_id, kind) -> set of markets already asked for, from fetched.csv. Files saved
    before that log existed were all fetched with the three yardage markets."""
    f = {}
    path = out_dir / "fetched.csv"
    if path.exists():
        for r in pd.read_csv(path, dtype=str).itertuples():
            f.setdefault((r.game_id, r.kind), set()).add(r.market)
    return f


def _note_fetched(out_dir, game_id: str, kind: str, markets: list[str]) -> None:
    path = out_dir / "fetched.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["game_id", "kind", "market"])
        for m in markets:
            w.writerow([game_id, kind, m])


def _missing_markets(path, markets: list[str], fetched: set) -> list[str]:
    """The requested markets a saved snapshot hasn't been asked for yet. A missing file needs
    them all; a header-only file (a snapshot the API answered 404 for) needs none; an existing
    file only the markets neither in it nor in fetched.csv (a market the book had no lines for
    is still fetched, so it isn't billed again), so a season can be extended a market at a time."""
    if not path.exists():
        return list(markets)
    have = pd.read_csv(path, usecols=["market"])["market"]
    if have.empty:
        return []
    done = set(have) | fetched | (set(YARDAGE) if not fetched else set())
    return [m for m in markets if m not in done]


def _append(path, snapshot, ev) -> int:
    """Add a fetch's rows to an existing snapshot file."""
    rows = oddsapi.flatten(ev) if ev else []
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        for r in rows:
            w.writerow(dict(r, snapshot=snapshot))
    return len(rows)


def run(client: oddsapi.Client, season: int, markets: list[str], max_credits: int, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    games = schedule(season)
    name = {v: k for k, v in C.TEAM_NAMES.items()}
    out_dir = HIST / str(season)
    events_path = out_dir / "events.csv"
    known = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(columns=["game_id", "event_id", "commence"])
    spent, fetched, skipped, probe_ok = 0, 0, 0, False

    def room(cost):
        return spent + cost <= max_credits

    for week, wk in games.groupby(["week"]):
        week = week[0] if isinstance(week, tuple) else week
        need = []
        fetched_log = _fetched(out_dir)
        for g in wk.itertuples():
            for kind in ("flag", "close"):
                want = _missing_markets(out_dir / f"{g.game_id}_{kind}.csv", markets, fetched_log.get((g.game_id, kind), set()))
                if want:
                    need.append((g, kind, want))
        if not need:
            continue
        # event ids for the week (1 credit), unless already listed
        ids = dict(zip(known["game_id"], known["event_id"]))
        if any(g.game_id not in ids for g, _, _ in need):
            if not room(1):
                break
            first, last = wk["kickoff"].min(), wk["kickoff"].max()
            evs = client.historical_events(_iso(first - pd.Timedelta(days=4)), _iso(first - pd.Timedelta(hours=1)),
                                           _iso(last + pd.Timedelta(hours=1)))
            spent += 1
            add = []
            for e in evs:
                h, a = name.get(e.get("home_team")), name.get(e.get("away_team"))
                m = wk[(wk.home_team == h) & (wk.away_team == a)]
                if len(m):
                    add.append({"game_id": m.iloc[0]["game_id"], "event_id": e["id"], "commence": e["commence_time"]})
            if add:
                known = pd.concat([known, pd.DataFrame(add)], ignore_index=True).drop_duplicates("game_id", keep="last")
                out_dir.mkdir(parents=True, exist_ok=True)
                known.to_csv(events_path, index=False)
            ids = dict(zip(known["game_id"], known["event_id"]))
        for g, kind, want in need:
            path = out_dir / f"{g.game_id}_{kind}.csv"
            per_snap = 10 * len(want)
            eid = ids.get(g.game_id)
            if eid is None:
                say(f"  {g.game_id}: no event id found")
                continue
            if not room(per_snap):
                say(f"  stopping: {spent} credits spent, cap {max_credits}")
                return {"spent": spent, "snapshots": fetched, "skipped": skipped, "complete": False}
            at = flag_time(g.kickoff) if kind == "flag" else g.kickoff - CLOSE_BEFORE
            try:
                ev = client.historical_event_odds(eid, want, _iso(at))
            except oddsapi.BudgetError as e:
                say(f"  {e}")
                return {"spent": spent, "snapshots": fetched, "skipped": skipped, "complete": False}
            except oddsapi.ApiError as e:
                # 404: the API has no snapshot of this event at this minute (an event re-created
                # under a new id late in the week, usually a playoff game). Save an empty file so
                # the snapshot is never asked for again. Any other error: skip, keep going.
                if "HTTP 404" in str(e) and not path.exists():
                    _save(path, _iso(at), None)
                    say(f"  {g.game_id} {kind} at {_iso(at)}: not available (404), skipped")
                else:
                    say(f"  {g.game_id} {kind} at {_iso(at)}: {e}; skipped")
                skipped += 1
                continue
            cost = client.last_cost
            spent += cost
            if not probe_ok and cost > 0:
                if cost > per_snap:
                    say(f"  first snapshot cost {cost} credits, expected at most {per_snap}: stopping to check")
                    return {"spent": spent, "snapshots": fetched, "complete": False, "probe_cost": cost}
                probe_ok = True
            snap = ev.get("_snapshot") if ev else ""
            n = _append(path, snap, ev) if path.exists() else _save(path, snap, ev)
            _note_fetched(out_dir, g.game_id, kind, want)
            fetched += 1
            say(f"  {g.game_id} {kind} at {_iso(at)}: {n} prices ({', '.join(want)}), {cost} credits (total {spent}, left {client.remaining})")
    if skipped:
        say(f"  {skipped} snapshot(s) skipped (see above)")
    return {"spent": spent, "snapshots": fetched, "skipped": skipped, "complete": True}


def load(season: int, kind: str) -> pd.DataFrame:
    """Every saved snapshot of one kind for a season, with game_id."""
    d = HIST / str(season)
    parts = []
    for p in sorted(d.glob(f"*_{kind}.csv")):
        x = pd.read_csv(p)
        if len(x):
            x["game_id"] = p.name[: -len(f"_{kind}.csv")]
            parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS + ["game_id"])
