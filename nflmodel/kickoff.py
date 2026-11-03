"""Kickoff refresh: a fresh line snapshot and a board rebuild shortly before each kickoff.

The Best bets card (bestbets.py) is graded as it stood at the last build before each game, and
the price tier is "take it now". GitHub's own schedules start late or not at all, so the
refreshes don't rely on them landing on the minute: the Kickoff refresh Action
(.github/workflows/kickoff.yml) starts well ahead, asks this module for the times, sleeps until
each one, starts a snapshot of that league (Market lines, workflow_dispatch) and then that
league's board rebuild (Update board / Update college board).

Times: games are grouped into kickoff windows (a game within WINDOW_MIN of the window's first
kickoff joins it), per league. Each window gets a refresh LEADS minutes before its first
kickoff: 35 (the card is live about 25 minutes before, time to place a bet) and 20 (as close to
the close as is still usable). Refreshes of the two leagues within MERGE_MIN of each other are
one refresh of both.

After the last refresh of a window (the 20-minute one) the Action sends that window's Best bets
card, as it now stands and as the record will grade it (bestbets.csv), to the phone (ntfy, the
NTFY_TOPIC secret) and to the week's best-bets issue (GitHub emails the comment). A window with
nothing on the card sends nothing.

    python -m nflmodel.kickoff [--horizon MIN]   prints "epoch league label alert" per refresh
                                                 still ahead within the horizon, soonest first;
                                                 alert is "nfl@EPOCH,cfb@EPOCH" (the windows
                                                 whose last refresh this is) or "-"
    python -m nflmodel.kickoff --alert nfl@EPOCH[,cfb@EPOCH] [--push FILE] [--md FILE]
                                                 writes those windows' card (first line of the
                                                 push file is its title); writes nothing when
                                                 the card is empty
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from zoneinfo import ZoneInfo

import pandas as pd

from . import bestbets as BB, config as C, sheet as SH

LEADS = (35, 20)          # minutes before a window's first kickoff
WINDOW_MIN = 30           # games kicking off within this of the first one share its refreshes
MERGE_MIN = 5             # NFL and college refreshes this close become one refresh of both
PATHS = {"nfl": C.DOCS / "data.json", "cfb": C.DOCS / "cfb" / "data.json"}
CARDS = {"nfl": C.ROOT / "data" / "bestbets.csv", "cfb": C.ROOT / "data" / "cfb" / "bestbets.csv"}
ALERT_LEAD = min(LEADS)   # the window's last refresh sends its card
ET = ZoneInfo("America/New_York")
BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "kalshi": "Kalshi", "consensus": "Consensus"}
MY_BOOKS = ("draftkings", "fanduel")   # the only books Ryan bets: the alert shows only these
NAMES = {"nfl": "NFL", "cfb": "College"}


def _parse(ko: str) -> dt.datetime:
    return dt.datetime.strptime(ko.replace("Z", ""), "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc)


def kickoffs(paths: dict | None = None) -> dict[str, list[dt.datetime]]:
    """Each league's game kickoffs (UTC) from the published boards."""
    out = {}
    for lg, p in (paths or PATHS).items():
        p = Path(p)
        if not p.exists():
            continue
        games = (json.loads(p.read_text()).get("slate") or {}).get("games") or []
        out[lg] = sorted({_parse(g["kickoff"]) for g in games if g.get("kickoff")})
    return out


def windows(times: list[dt.datetime]) -> list[dt.datetime]:
    """The first kickoff of each window."""
    firsts = []
    for t in sorted(times):
        if not firsts or (t - firsts[-1]).total_seconds() / 60 > WINDOW_MIN:
            firsts.append(t)
    return firsts


def plan(now: dt.datetime, horizon_min: float = 330, ko: dict | None = None) -> list[tuple[dt.datetime, str, str, str]]:
    """(when, league, label, alert) for every refresh after now and within the horizon, soonest
    first. league is nfl, cfb or both; alert names the windows whose last refresh this is
    ("nfl@EPOCH,cfb@EPOCH", EPOCH the window's first kickoff) or is "-"."""
    ko = kickoffs() if ko is None else ko
    end = now + dt.timedelta(minutes=horizon_min)
    raw = []
    for lg, times in ko.items():
        for first in windows(times):
            for lead in LEADS:
                at = first - dt.timedelta(minutes=lead)
                if now < at <= end:
                    alert = f"{lg}@{int(first.timestamp())}" if lead == ALERT_LEAD else ""
                    raw.append((at, lg, f"{lg} {first:%a %H:%MZ} -{lead}m", alert))
    raw.sort()
    out = []
    for at, lg, label, alert in raw:
        if out and out[-1][1] != lg and out[-1][1] != "both" and (at - out[-1][0]).total_seconds() / 60 <= MERGE_MIN:
            prev = out[-1]
            out[-1] = (prev[0], "both", prev[2] + " + " + label, ",".join(x for x in (prev[3], alert) if x))
            continue
        out.append((at, lg, label, alert))
    return [(at, lg, label, alert or "-") for at, lg, label, alert in out]


def _num(x: float) -> str:
    return f"{x:g}"


def _pick(market: str, pick: str, line) -> str:
    """The pick at a book's own line."""
    if market == "Spread" and line is not None and pd.notna(line):
        return f"{pick.rsplit(' ', 1)[0]} {'+' if line > 0 else ''}{_num(line)}"
    if market == "Total" and line is not None and pd.notna(line):
        return f"{pick.split(' ', 1)[0]} {_num(line)}"
    return pick


def qualifies(b: dict, q: dict) -> bool:
    """Whether this book's price still makes the bet what the card calls it: a price bet needs the
    book to beat the sharp no-vig number by sheet.PRICE_MIN (the tier's own rule); every other
    tier needs the book's price to be +EV on the model."""
    if b.get("tier") == "price":
        return (q.get("ev_line") or -1.0) >= SH.PRICE_MIN
    return (q.get("ev") or -1.0) > 0


def _bet_line(r, b: dict | None) -> tuple[str, bool]:
    """One card entry at DraftKings or FanDuel, and whether it qualifies there. `r` is the
    record's row (bestbets.csv), `b` the same bet on the board (data.json) with every book's quote."""
    teams = str(r["game_id"]).split("_")[2:]
    game = f"{teams[0]}@{'_'.join(teams[1:])}: " if len(teams) >= 2 else ""
    pick = str(r["pick"])
    quotes = [q for q in ((b or {}).get("books") or []) if q.get("book") in MY_BOOKS and q.get("odds") is not None]
    if not quotes and r["book"] in MY_BOOKS:       # the board entry is missing: the record's own row
        quotes = [{"book": r["book"], "line": r["line"], "odds": r["odds"], "ev": r["ev"], "stake": r["stake"],
                   "ev_line": SH.PRICE_MIN}]
        b = b or {"tier": "opener"}
    if not quotes:
        return f"{game}{pick} · no DraftKings or FanDuel price (card: {BOOKS.get(str(r['book']), r['book'])})", False
    good = [q for q in quotes if qualifies(b, q)]
    if good:
        q = max(good, key=lambda q: q.get("ev") or 0.0)
        stake = q.get("stake") or 0.0
        if stake <= 0 and b.get("bucket_book") == q["book"]:
            stake = b.get("bucket_stake") or 0.0
        if stake <= 0 and r["book"] == q["book"] and pd.notna(r["stake"]):
            stake = float(r["stake"])
        parts = [f"{game}{_pick(r['market'], pick, q.get('line'))} {int(round(q['odds'])):+d}", BOOKS[q["book"]],
                 f"stake {stake * 100:.1f}%" if stake > 0 else "no stake"]
        if q.get("ev") is not None:
            parts.append(f"EV {q['ev'] * 100:+.1f}%")
        return " · ".join(parts), True
    best = max(quotes, key=lambda q: q.get("ev") if q.get("ev") is not None else -9)
    ev = f", EV {best['ev'] * 100:+.1f}%" if best.get("ev") is not None else ""
    why = "doesn't beat Pinnacle by enough" if b.get("tier") == "price" else "not +EV"
    return (f"{game}{pick} · NOT A BET at DraftKings or FanDuel ({why}; best there: {BOOKS[best['book']]} "
            f"{_pick(r['market'], pick, best.get('line'))} {int(round(best['odds'])):+d}{ev}; "
            f"card has it at {BOOKS.get(str(r['book']), r['book'])})"), False


def _board_bets(path: Path | None) -> dict:
    """(game id, market, side) -> the board's bet, with every book's quote."""
    if path is None or not Path(path).exists():
        return {}
    games = (json.loads(Path(path).read_text()).get("slate") or {}).get("games") or []
    return {(str(g.get("id")), b.get("market"), b.get("side")): b for g in games for b in g.get("bets") or []}


def card(league: str, first: dt.datetime, path: Path | None = None) -> pd.DataFrame:
    """The Best bets card for the window that starts at `first` (games within WINDOW_MIN of it),
    as it stands now: each bet's newest entry, kept when it is on the card."""
    df = BB._read(Path(path or CARDS[league]))
    if df.empty:
        return df
    ko = pd.to_datetime(df["kickoff"].astype(str).str.replace("Z", ""), errors="coerce", utc=True)
    lo, hi = pd.Timestamp(first), pd.Timestamp(first + dt.timedelta(minutes=WINDOW_MIN))
    d = BB._last(df[(ko >= lo) & (ko <= hi)])
    if d.empty:
        return d
    d = d[pd.to_numeric(d["on_card"], errors="coerce") == 1]
    return d.sort_values(["kickoff", "game_id", "market"], kind="stable")


def message(windows_: list[tuple[str, dt.datetime]], paths: dict | None = None,
            boards: dict | None = None) -> tuple[str, str] | None:
    """(title, body) for these windows' cards at DraftKings and FanDuel, or None when every card
    is empty. A card bet neither book qualifies is listed as not a bet there."""
    parts, heads, n, off = [], [], 0, 0
    for lg, first in windows_:
        d = card(lg, first, (paths or {}).get(lg))
        if d.empty:
            continue
        bets = _board_bets((boards or PATHS).get(lg))
        when = first.astimezone(ET).strftime("%a %-I:%M %p ET")
        heads.append(f"{NAMES[lg]} {when}")
        rows = [_bet_line(r, bets.get((str(r["game_id"]), r["market"], r["side"]))) for _, r in d.iterrows()]
        rows.sort(key=lambda x: not x[1])                 # the bets first, then the ones that aren't
        n += sum(ok for _, ok in rows)
        off += sum(not ok for _, ok in rows)
        parts.append("\n".join([f"{NAMES[lg]}, kickoffs from {when}:"] + [t for t, _ in rows]))
    if not parts:
        return None
    count = f"{n} bet{'s' if n != 1 else ''}" + (f", {off} not at DK/FD" if off else "")
    return f"Best bets: {' + '.join(heads)} ({count})", "\n\n".join(parts)


def _windows_arg(s: str) -> list[tuple[str, dt.datetime]]:
    out = []
    for tok in s.split(","):
        if "@" in tok:
            lg, ep = tok.split("@", 1)
            out.append((lg, dt.datetime.fromtimestamp(int(ep), dt.timezone.utc)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon", type=float, default=330, help="minutes ahead to plan (the job's time limit)")
    ap.add_argument("--alert", help="nfl@EPOCH[,cfb@EPOCH]: write these windows' Best bets card")
    ap.add_argument("--push", type=Path, default=Path("/tmp/bestbets_push.txt"))
    ap.add_argument("--md", type=Path, default=Path("/tmp/bestbets.md"))
    a = ap.parse_args(argv)
    if a.alert is not None:
        msg = message(_windows_arg(a.alert))
        if msg is None:
            print("nothing on the card for", a.alert)
            return 0
        title, body = msg
        a.push.write_text(title + "\n" + body + "\n")
        a.md.write_text(f"**{title}**\n\n" + body.replace("\n", "  \n") + "\n\nThe Best bets card at the last rebuild before kickoff, at DraftKings and FanDuel only. Lines move: check the book before betting.\n")
        print(title + "\n" + body)
        return 0
    for at, lg, label, alert in plan(dt.datetime.now(dt.timezone.utc), a.horizon):
        print(int(at.timestamp()), lg, label.replace(" ", "_"), alert)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
