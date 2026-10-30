"""The bet ledger: every bet the board flagged, at the price it was first flagged, graded
against the result and the sharp book's closing line. Both boards (data/ledger.csv and
data/cfb/ledger.csv, with ledger_results.csv beside each).

Why. The archive freezes each game at kickoff, so its record is the board's last word. A
bettor acts on the first word: the ledger writes a bet down the first time the board flags
it (a stake on the shown basis, or a price that beats the sharp line) and never changes the
row. Grading adds the result and the closing line value against the sharp book's last quote
before kickoff. Realized return and CLV on these rows are the live record of the board.

ledger.csv (append-only, one row per game, market and side, written by the Update Actions):
    row_id, flagged_at (UTC), season, week, game_id (the board's id), data_id (the data's
    game id; the same as game_id on the NFL board), kickoff (UTC), market, side, pick, tier,
    book, line, odds, odds_net, ev, stake, anchor, anchor_spread, anchor_total, model_margin,
    model_total
ledger_results.csv (append-only, one row per graded row_id):
    row_id, graded_at, home_score, away_score, result (W/L/P), units (on one unit risked),
    close_line, close_prob, clv

Derivative bets (nflmodel/derivs.py, from Oct 30, 2026) are rows like any other, with market
"Team total" (side home_over, home_under, away_over, away_under; `line` the team's total),
"1H Spread", "1H Total" or "1H Moneyline". First-half rows are graded on the half-time score
(`halves` in grade; the row waits until it is on file) and carry no closing line value yet.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import odds as O

COLS = ["row_id", "flagged_at", "season", "week", "game_id", "data_id", "kickoff", "market", "side", "pick", "tier",
        "book", "line", "odds", "odds_net", "ev", "stake", "anchor", "anchor_spread", "anchor_total",
        "model_margin", "model_total"]
RES_COLS = ["row_id", "graded_at", "home_score", "away_score", "result", "units", "close_line", "close_prob", "clv"]
FLAG_TIERS = ("price", "opener", "wind")       # flagged even with no stake sized


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype={"row_id": str, "game_id": str, "data_id": str})
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df[cols]


def _taken(b: dict, g: dict) -> dict:
    """The book, line and price the bet would be placed at: the retail book with the best
    price (the one beating the sharp line, for a price bet); the anchor's own line and price
    only when no book quotes the side."""
    if b.get("books") and b.get("best") is not None:
        q = next((x for x in b["books"] if x["book"] == (b.get("beats_sharp") if b.get("tier") == "price" and b.get("beats_sharp") else b["best"])), None)
        if q:
            return {"book": q["book"], "line": q.get("line"), "odds": q["odds"], "odds_net": q.get("odds_net", q["odds"]),
                    "ev": q["ev"], "stake": q.get("stake", 0.0)}
    mk, side = b["market"], b["side"]
    sp, to = g["market"]["spread"], g["market"]["total"]
    line = (-sp if side == "home" else sp) if mk == "Spread" else to if mk == "Total" else None
    return {"book": g.get("anchor") or "consensus", "line": line, "odds": b["odds"], "odds_net": b["odds"], "ev": b["ev"], "stake": b.get("stake", 0.0)}


def sheet_bets(g: dict) -> list[dict]:
    """A game's bets and its derivative sides together (both carry a tier)."""
    return list(g.get("bets") or []) + list(g.get("derivs") or [])


def flagged(b: dict) -> bool:
    """A bet the board would place: a stake on the market-adjusted basis, or a price or
    opener or wind flag (sized or not)."""
    if not b.get("tier"):
        return False
    stake = max(float(b.get("stake") or 0), max((float(q.get("stake") or 0) for q in b.get("books") or []), default=0.0))
    return stake > 0 or b.get("tier") in FLAG_TIERS


def record(slate_games: list[dict], season: int, kickoffs: dict, path: Path, now: dt.datetime | None = None) -> int:
    """Append every newly flagged bet on games that have not kicked off. kickoffs: game id ->
    UTC Timestamp. Returns the rows added."""
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    have = set(_read(path, COLS)["row_id"]) if path.exists() else set()
    rows = []
    for g in slate_games:
        gid = str(g["id"])
        ko = kickoffs.get(g["id"], kickoffs.get(gid))
        if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
            continue
        for b in sheet_bets(g):
            rid = f"{gid}|{b['market']}|{b['side']}"
            if rid in have or not flagged(b):
                continue
            t = _taken(b, g)
            if not t["stake"] and b.get("bucket_stake"):
                t["stake"] = b["bucket_stake"]          # sized from the bucket's record (sheet.bucket_stakes)
            rows.append({"row_id": rid, "flagged_at": stamp, "season": season, "week": g.get("week"), "game_id": gid,
                         "data_id": str(g.get("game_id", gid)), "kickoff": pd.Timestamp(ko).strftime("%Y-%m-%dT%H:%MZ"), "market": b["market"], "side": b["side"],
                         "pick": b["pick"], "tier": b.get("tier") or "", **t,
                         "anchor": g.get("anchor"), "anchor_spread": g["market"]["spread"], "anchor_total": g["market"]["total"],
                         "model_margin": g["model"]["margin"], "model_total": g["model"]["total"]})
            have.add(rid)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=COLS).to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def settle(market: str, side: str, line, margin: float, total: float, half: tuple | None = None) -> tuple[str, float] | None:
    """W/L/P and units on one unit risked at even money (the price is applied by the caller).
    margin and total are the game's; `half` is (half-time margin, half-time total) for the
    first-half markets, and None when it is not on file yet (the row is left ungraded).
    Team totals read the team's points from the margin and total."""
    if market.startswith("1H "):
        if half is None:
            return None
        margin, total = half
        market = market[3:]
    if market == "Team total":
        team, ou = side.split("_")
        pts = (total + margin) / 2 if team == "home" else (total - margin) / 2
        diff = (1 if ou == "over" else -1) * (pts - float(line))
    else:
        s = 1 if side in ("home", "over") else -1
        if market == "Spread":
            diff = s * margin + float(line)
        elif market == "Total":
            diff = s * (total - float(line))
        else:
            diff = s * margin
    return ("P", 0.0) if diff == 0 else ("L", -1.0) if diff < 0 else ("W", 1.0)


def grade(path: Path, results_path: Path, scores: dict, close: dict | None = None, curve=None,
          now: dt.datetime | None = None, halves: dict | None = None) -> int:
    """Append a result for every ledger row whose game has a final score and no result yet.
    scores: data game id (str) -> (home_score, away_score); close: data game id (str) -> the
    sharp book's closing quote (market.closing_quotes); curve(df) -> bets.curves(..., "line")
    for the CLV; halves: data game id (str) -> (home half-time score, away half-time score),
    for the first-half rows (one with no half-time score on file waits)."""
    led = _read(path, COLS)
    if led.empty:
        return 0
    done = set(_read(results_path, RES_COLS)["row_id"]) if results_path.exists() else set()
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    close = close or {}
    halves = halves or {}
    rows = []
    for r in led.itertuples(index=False):
        if r.row_id in done or r.data_id not in scores:
            continue
        hs, as_ = scores[r.data_id]
        if hs is None or as_ is None or (isinstance(hs, float) and np.isnan(hs)):
            continue
        margin, total = float(hs) - float(as_), float(hs) + float(as_)
        h = halves.get(r.data_id)
        half = (float(h[0]) - float(h[1]), float(h[0]) + float(h[1])) if h is not None and all(x is not None and not (isinstance(x, float) and np.isnan(x)) for x in h) else None
        settled = settle(r.market, r.side, r.line, margin, total, half)
        if settled is None:
            continue
        res, u = settled
        if res == "W":
            o = float(r.odds_net)
            u = round(o / 100 if o > 0 else 100 / -o, 4)
        cv = _clv(r, close.get(r.data_id), curve)
        rows.append({"row_id": r.row_id, "graded_at": stamp, "home_score": hs, "away_score": as_, "result": res, "units": u, **cv})
    if rows:
        pd.DataFrame(rows, columns=RES_COLS).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
    return len(rows)


def _clv(r, close: dict | None, curve) -> dict:
    """The EV of the price taken at the sharp book's closing no-vig chance, carried to the
    bet's own line through the key-number curve (archive.closing_value, on the ledger's row)."""
    empty = {"close_line": None, "close_prob": None, "clv": None}
    if not close or curve is None:
        return empty
    mk, side = r.market, r.side
    need = {"Spread": ("spread_line", "home_spread_odds", "away_spread_odds"), "Total": ("total_line", "over_odds", "under_odds"),
            "Moneyline": ("home_moneyline", "away_moneyline")}.get(mk)
    if need is None or any(k not in close for k in need):          # derivative rows: no closing value yet
        return empty
    row = {"game_id": r.data_id, "spread_line": close.get("spread_line", r.anchor_spread), "total_line": close.get("total_line", r.anchor_total),
           **{k: close.get(k, np.nan) for k in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds", "home_moneyline", "away_moneyline")}}
    from . import bets as B
    cv = curve(pd.DataFrame([row]))
    line = None if mk == "Moneyline" else (-float(r.line) if (mk == "Spread" and side == "home") else float(r.line))
    p, push = B._side_prob(cv, 0, mk, side, line)
    return {"close_line": close.get("spread_line" if mk == "Spread" else "total_line"),
            "close_prob": round(p / (1 - push), 4), "clv": round(float(O.expected_value(p, float(r.odds_net), push)), 4)}


def _rec(x: pd.DataFrame) -> dict:
    g = x[x["result"].notna()]
    w, l, p = int((g.result == "W").sum()), int((g.result == "L").sum()), int((g.result == "P").sum())
    staked = float(g["stake"].fillna(0).sum())
    profit = float((g["stake"].fillna(0) * g["units"]).sum())
    clv = g[g["clv"].notna()]
    return {"n": int(len(x)), "graded": int(len(g)), "w": w, "l": l, "p": p,
            "win_pct": round(w / (w + l), 4) if w + l else None,
            "units": round(float(g["units"].sum()), 2) if len(g) else 0.0,
            "roi": round(float(g["units"].mean()), 4) if len(g) else None,
            "staked": round(staked, 4), "profit": round(profit, 4),
            "return": round(profit / staked, 4) if staked > 0 else None,
            "clv_n": int(len(clv)), "clv": round(float(clv["clv"].mean()), 4) if len(clv) else None,
            "beat_close": int((clv["clv"] > 0).sum()), "claimed_ev": round(float(x["ev"].mean()), 4) if len(x) else None}


def summary(path: Path, results_path: Path) -> dict:
    """The ledger with its results for data.json: rows (newest first) and records overall, by
    tier, by market and by week."""
    led = _read(path, COLS)
    if led.empty:
        return {"since": None, "rows": [], "all": _rec(led.assign(result=np.nan, units=np.nan, clv=np.nan)), "by_tier": {}, "by_market": {}, "by_week": []}
    res = _read(results_path, RES_COLS)
    x = led.merge(res.drop(columns=["graded_at"]), on="row_id", how="left")
    x["tier"] = x["tier"].fillna("").replace("", "model")
    out = {"since": str(led["flagged_at"].min())[:10], "all": _rec(x),
           "by_tier": {t: _rec(y) for t, y in x.groupby("tier")},
           "by_market": {m: _rec(y) for m, y in x.groupby("market")},
           "by_week": [{"season": int(s), "week": int(w), **_rec(y)} for (s, w), y in x.groupby(["season", "week"])]}
    rows = x.sort_values(["kickoff", "game_id", "market", "side"], ascending=[False, True, True, True])
    out["rows"] = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                   for r in rows.drop(columns=["anchor_spread", "anchor_total", "model_margin", "model_total"]).to_dict("records")]
    return out
