"""The Best Bets record: only the bets on each week's Best bets card, as the card showed them
when each game kicked off. Both boards (data/bestbets.csv and data/cfb/bestbets.csv, with
bestbets_results.csv beside each).

Why a second record. The ledger (ledger.py) writes a bet down the first time anything flags it,
at that first price: it is the record of acting on the board's first word. The Best bets card
(#bestbets, best-bets.pdf) is what a bettor works from on the day, and it changes as news and
prices come in: bets drop off, books and prices change, stakes move with the caps. This record
is that card: for every game, the bets on the card at the last build before kickoff, at the
book, line, price and stake it named, graded on the result and on closing line value against
Pinnacle's close the same way as the ledger (ledger.grade_frame).

What is on the card (the page's ON_CARD, site/template.html): every bet with a tier, except
the model tier on the NFL board, which carries no stake while its haircut is 0 and is in Model
leans instead. Stake: sheet.taken_stake (the book's stake, else the bucket stake), after the
game and weekly caps. Book, line and price: ledger._taken.

bestbets.csv (append-only, written by the Update Actions): one row each time a card entry is
new, changes book, line, price or stake, or leaves the card, for games that have not kicked
off: row_id (game|market|side), at (UTC), season, week, game_id, data_id, kickoff, market,
side, pick, tier, book, line, odds, odds_net, ev, stake, anchor_spread, anchor_total, on_card
(1 on the card, 0 taken off). The record reads, per row_id, the last entry before kickoff; a
row whose last entry is on_card 0 was not on the final card and is not counted.
bestbets_results.csv (append-only): the ledger's result columns per graded row_id.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import ledger as L, sheet as SH

COLS = ["row_id", "at", "season", "week", "game_id", "data_id", "kickoff", "market", "side", "pick", "tier",
        "book", "line", "odds", "odds_net", "ev", "stake", "anchor_spread", "anchor_total", "on_card"]
SAME = ["book", "line", "odds", "stake"]          # a change in any of these is a new entry


def on_card(b: dict, league: str) -> bool:
    """A bet the Best bets card lists (the page's ON_CARD)."""
    t = b.get("tier")
    return bool(t) and not (t == "model" and league == "nfl")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path, dtype={"row_id": str, "game_id": str, "data_id": str, "at": str})
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    return df[COLS]


def _last(df: pd.DataFrame, before: dict | None = None) -> pd.DataFrame:
    """The last entry per row_id (before its kickoff, when `before` maps row_id -> kickoff)."""
    if df.empty:
        return df
    d = df
    if before is not None:
        d = d[[str(a).replace(" ", "T") < str(before.get(r, "9999")).replace("Z", "") for r, a in zip(d["row_id"], d["at"])]]
    return d.sort_values("at", kind="stable").groupby("row_id", sort=False).tail(1)


def _na(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _same(a: dict, b: dict) -> bool:
    for k in SAME:
        x, y = a.get(k), b.get(k)
        if _na(x) or _na(y):
            if _na(x) != _na(y):
                return False
            continue
        try:
            if abs(float(x) - float(y)) > 1e-6:
                return False
        except (TypeError, ValueError):
            if str(x) != str(y):
                return False
    return True


def record(slate_games: list[dict], season: int, kickoffs: dict, path: Path, league: str, now: dt.datetime | None = None) -> int:
    """Log the card for every game that has not kicked off: new entries, changes, and bets
    that left it. kickoffs: game id -> UTC Timestamp. Returns the entries written."""
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    last = {r["row_id"]: r for r in _last(_read(path)).to_dict("records")}
    rows = []
    for g in slate_games:
        gid = str(g["id"])
        ko = kickoffs.get(g["id"], kickoffs.get(gid))
        if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
            continue
        seen = set()
        for b in L.sheet_bets(g):
            rid = f"{gid}|{b['market']}|{b['side']}"
            if not on_card(b, league):
                continue
            seen.add(rid)
            t = L._taken(b, g)
            row = {"row_id": rid, "at": stamp, "season": season, "week": g.get("week"), "game_id": gid,
                   "data_id": str(g.get("game_id", gid)), "kickoff": pd.Timestamp(ko).strftime("%Y-%m-%dT%H:%MZ"),
                   "market": b["market"], "side": b["side"], "pick": b["pick"], "tier": b.get("tier"),
                   "book": t["book"], "line": t["line"], "odds": t["odds"], "odds_net": t["odds_net"], "ev": t["ev"],
                   "stake": round(SH.taken_stake(b), 4), "anchor_spread": g["market"]["spread"], "anchor_total": g["market"]["total"], "on_card": 1}
            prev = last.get(rid)
            if prev is None or int(prev.get("on_card") or 0) == 0 or not _same(prev, row):
                rows.append(row)
        for rid, prev in last.items():                  # on the card last time, gone now
            if rid.startswith(gid + "|") and rid not in seen and int(prev.get("on_card") or 0) == 1:
                rows.append({**{k: prev.get(k) for k in COLS}, "at": stamp, "on_card": 0})
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=COLS).to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def final(path: Path, now: dt.datetime | None = None) -> pd.DataFrame:
    """The card as it stood at each game's kickoff (games that have kicked off), on-card rows only."""
    df = _read(path)
    if df.empty:
        return df
    now = now or dt.datetime.now(dt.timezone.utc)
    ko = dict(zip(df["row_id"], df["kickoff"].astype(str)))
    f = _last(df, ko)
    f = f[(f["on_card"].astype(float) == 1) & (pd.to_datetime(f["kickoff"], utc=True) <= pd.Timestamp(now))]
    return f.reset_index(drop=True)


def grade(path: Path, results_path: Path, scores: dict, close: dict | None = None, curve=None,
          now: dt.datetime | None = None, halves: dict | None = None) -> int:
    """Grade the final card's rows whose games have a score (ledger.grade_frame)."""
    return L.grade_frame(final(path, now), results_path, scores, close, curve, now, halves)


def summary(path: Path, results_path: Path, now: dt.datetime | None = None) -> dict:
    """For data.json: the record overall, by week, by tier and by market, and every row (newest
    first); plus `pending`, this week's card bets not yet kicked off (counted when they do)."""
    f = final(path, now)
    df = _read(path)
    pending = 0
    if not df.empty:
        now_ts = pd.Timestamp(now or dt.datetime.now(dt.timezone.utc))
        cur = _last(df)
        pending = int(((cur["on_card"].astype(float) == 1) & (pd.to_datetime(cur["kickoff"], utc=True) > now_ts)).sum())
    if f.empty:
        empty = L._rec(pd.DataFrame(columns=COLS + ["result", "units", "clv"]))
        return {"since": None, "all": empty, "by_week": [], "by_tier": {}, "by_market": {}, "rows": [], "pending": pending}
    res = L._read(results_path, L.RES_COLS)
    x = f.merge(res.drop(columns=["graded_at"]), on="row_id", how="left")
    for c in ("stake", "ev", "units", "clv", "odds", "odds_net", "line"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    out = {"since": str(df["at"].min())[:10], "all": L._rec(x), "pending": pending,
           "by_week": [{"season": int(s), "week": int(w), **L._rec(y)} for (s, w), y in x.groupby(["season", "week"])],
           "by_tier": {t: L._rec(y) for t, y in x.groupby("tier")},
           "by_market": {m: L._rec(y) for m, y in x.groupby("market")}}
    rows = x.sort_values(["kickoff", "game_id", "market", "side"], ascending=[False, True, True, True])
    out["rows"] = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                   for r in rows.drop(columns=["anchor_spread", "anchor_total", "on_card"]).to_dict("records")]
    return out
