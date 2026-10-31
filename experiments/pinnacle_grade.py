"""Grade the boards' edges against Pinnacle's real open and close, from the market backfill.

    python experiments/pinnacle_grade.py --league nfl --seasons 2023 2024 2025
    python experiments/pinnacle_grade.py --league cfb --seasons 2024 2025

Needs data/market/history/<league>/<season>/ (the Market line backfill Action,
nflmodel/marketbackfill.py): an "open" snapshot each Monday 12:00 UTC and a "close" ten
minutes before each cluster of kickoffs, Pinnacle, DraftKings and FanDuel. Until now every
record in the repo was graded against the nflverse consensus (NFL) or one book's line
(college), and the live market log has Pinnacle only from Oct 30, 2026.

1. Price beats Pinnacle (the Bet sheet's `price` tier). At each snapshot kind, every
   DraftKings and FanDuel price on a spread, total or moneyline whose EV against Pinnacle's
   no-vig number at that book's own line (through the key-number curve: bets.shop's
   `ev_line`, no model) is at least sheet.PRICE_MIN, inside bets.FLAG_ODDS. One bet per game,
   market and side: the best `ev_line`. Graded on the result at the book's line and price,
   with closing line value against Pinnacle's last quote before kickoff (EV of the price
   taken at the closing no-vig chance carried to the bet's line). At the close snapshot the
   CLV is zero by construction; the result is what counts there.

2. Opener gap. The raw model (the board's walk-forward, run without the week's starting
   quarterbacks or injuries, as backtest.opening_report does) against Pinnacle's open: bet
   the model's side at Pinnacle's open line and price when it is off by sheet.OPENER_GAP
   points or more on spreads (NFL 2, college 3) or sheet.TOTAL_GAP on totals; graded on the
   result, with closing line value in points and in EV against Pinnacle's close. Side by side
   with the same games graded the old way (the consensus opener at -110 and the consensus
   close), so the two can be compared directly. Games where a team played in the days before
   (NFL: the Monday before; college: five days) are left out, as in the board's own test.

3. How far the consensus closing lines the backtests use sit from Pinnacle's close.

Not graded: derivative markets (team totals, first-half lines). The bulk historical endpoint
does not carry them; the per-event one does, at about 40 credits a game per snapshot.

Writes experiments/out/pinnacle_grade_<league>.json, and the graded bets to
data/raw/pinnacle_{price,opener}_bets_<league>.csv. The model run is cached in
data/raw/pinnacle_grade_frame_<league>.pkl: delete it after any change to the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from nflmodel import bets as NB, config as C, market as M, marketbackfill as MB, odds as O, sheet as S  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out"
RETAIL = ["draftkings", "fanduel"]
NEED = ("spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds")
PRICE_COLS = NEED + ("home_moneyline", "away_moneyline")


# --------------------------------------------------------------------------
# Loading: games, the backfill as quotes per game
# --------------------------------------------------------------------------

class League:
    """What differs between the boards: games, the outcome curve, team names."""

    def __init__(self, league: str):
        self.league = league
        if league == "nfl":
            self.bets, self.names = NB, None
        else:
            from cfbmodel import bets as CB
            self.bets, self.names = CB, {}

    def games(self) -> pd.DataFrame:
        if self.league == "nfl":
            from nflmodel import data
            g = data.load_all(refresh=False, verbose=False)[0]
            if "kickoff" not in g:
                t = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].fillna("13:00").astype(str), errors="coerce")
                g["kickoff"] = t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
            return g
        from cfbmodel import data
        return data.load_all(refresh=False, verbose=False)[0]

    def team_names(self, season: int) -> dict | None:
        """College: the books' display name -> the data's school name."""
        if self.league == "nfl":
            return None
        from cfbmodel import teams
        t = teams.load_teams(season)
        return {r.name: school for school, r in t.iterrows()}

    def shop(self, df, dist, quotes):
        if self.league == "nfl":
            return NB.shop(df, dist, quotes, RETAIL, source="line")
        return self.bets.shop(df, dist, quotes, RETAIL, source="line")

    def curve(self, df, dist):
        if self.league == "nfl":
            return NB.curves(df, dist, "line")
        from cfbmodel import bets as CB
        return NB.curves(df, dist, "line", CB.MARGINS, CB.TOTALS, CB.half)


def quotes(league: str, season: int, games: pd.DataFrame, names: dict | None) -> tuple[dict, dict]:
    """({game_id: {book: quote}} at each game's open snapshot, the same at its last close
    snapshot), in market.latest's shape (spread_line positive = home favoured)."""
    rows = MB.load(league, season)
    if rows.empty:
        return {}, {}
    g = games[games["season"] == season][["game_id", "home_team", "away_team", "kickoff"]].copy()
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True)
    g["open_at"] = g["kickoff"].map(MB.open_time)
    want = dict(zip(g["game_id"], g["open_at"]))
    opens, closes = [], []
    for (kind, req), x in rows.groupby(["kind", "requested"]):
        m = M.match_games(x, g, league, names)
        m = m[m["game_id"].notna()]
        if kind == "open":
            m = m[m["game_id"].map(want) == req]            # each game's own week's open
            opens.append(m)
        else:
            closes.append(m)
    cat = lambda parts: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=M.COLS + ["game_id"])
    return M.latest(cat(opens)), M.latest(cat(closes))


def anchor_rows(q: dict, book: str = M.SHARP) -> pd.DataFrame:
    """One row per game with `book`'s full quote (both spread and total sides priced)."""
    rows = []
    for gid, books in q.items():
        b = books.get(book) or {}
        if all(k in b for k in NEED):
            rows.append({"game_id": gid, **{k: b.get(k, np.nan) for k in PRICE_COLS}})
    return pd.DataFrame(rows, columns=["game_id", *PRICE_COLS])


# --------------------------------------------------------------------------
# Grading helpers
# --------------------------------------------------------------------------

def _se(values: pd.Series, groups) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    g = pd.Series(np.asarray(values, float)).groupby(np.asarray(groups)).sum()
    return float(g.std(ddof=1) * np.sqrt(len(g)) / n)


def outcome(market: str, side: str, shown_line, result, total) -> int:
    """+1 win, 0 push, -1 loss for a side at its shown line (a spread as the side's own number)."""
    if market == "Moneyline":
        v = result if side == "home" else -result
    elif market == "Spread":
        v = (result if side == "home" else -result) + shown_line
    else:
        v = (total - shown_line) if side == "over" else (shown_line - total)
    return int(np.sign(v))


def board_line(market: str, side: str, shown):
    """The line in the curves' convention (spread positive = home favoured) from a side's shown line."""
    if market == "Spread":
        return -shown if side == "home" else shown
    return shown


def clv(lg: League, dist: dict, close_q: dict, gid: str, market: str, side: str, shown, odds) -> float:
    """EV of `odds` at Pinnacle's closing no-vig chance carried to the bet's line."""
    c = (close_q.get(gid) or {}).get(M.SHARP) or {}
    if market == "Moneyline" and not ("home_moneyline" in c and "away_moneyline" in c):
        return np.nan
    if market != "Moneyline" and not all(k in c for k in NEED):
        return np.nan
    df = pd.DataFrame([{"game_id": gid, **{k: c.get(k, np.nan) for k in PRICE_COLS}}])
    cv = lg.curve(df, dist)
    p, push = NB._side_prob(cv, 0, market, side, board_line(market, side, shown))
    return float(O.expected_value(p, odds, push))


def record(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"n": 0}
    c = b.dropna(subset=["clv"])
    return {"n": int(len(b)), "w": int((b.res > 0).sum()), "l": int((b.res < 0).sum()), "p": int((b.res == 0).sum()),
            "return": round(float(b.profit.mean()), 4), "return_se": round(_se(b.profit, b.game_id), 4),
            "claimed": round(float(b.claimed.mean()), 4) if "claimed" in b else None,
            "clv": round(float(c.clv.mean()), 4) if len(c) else None,
            "clv_se": round(_se(c.clv, c.game_id), 4) if len(c) > 1 else None,
            "beat_close": round(float((c.clv > 0).mean()), 3) if len(c) else None}


def _profit(res: int, odds: float) -> float:
    return 0.0 if res == 0 else (float(O.american_to_decimal(odds)) - 1 if res > 0 else -1.0)


# --------------------------------------------------------------------------
# 1. Price beats Pinnacle
# --------------------------------------------------------------------------

def price_tier(lg: League, games: pd.DataFrame, season: int, snap_q: dict, close_q: dict, dist: dict) -> pd.DataFrame:
    df = anchor_rows(snap_q)
    if df.empty:
        return pd.DataFrame()
    played = games[games["played"]].set_index("game_id")
    df = df[df["game_id"].isin(played.index)].reset_index(drop=True)
    shopped = lg.shop(df, dist, snap_q)
    out = []
    for (gid, market, side), qs in shopped.items():
        ok = [q for q in qs if q["book"] in RETAIL and q.get("ev_line") is not None and q["ev_line"] >= S.PRICE_MIN]
        if not ok:
            continue
        q = max(ok, key=lambda x: x["ev_line"])
        r = played.loc[gid]
        res = outcome(market, side, q["line"], r["result"], r["total"])
        out.append({"season": season, "game_id": gid, "market": market, "side": side, "book": q["book"],
                    "line": q["line"], "odds": q["odds"], "claimed": q["ev_line"], "res": res,
                    "profit": _profit(res, q["odds"]),
                    "clv": clv(lg, dist, close_q, gid, market, side, q["line"], q["odds"])})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 2. Opener gap
# --------------------------------------------------------------------------

def model_frame(lg: League, games: pd.DataFrame) -> pd.DataFrame:
    """The raw walk-forward without the week's starters or injuries (cached in data/raw)."""
    path = C.RAW / f"pinnacle_grade_frame_{lg.league}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan)
    if lg.league == "nfl":
        from nflmodel import data, extra, model
        from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
        _, tg, qb = data.load_all(refresh=False, verbose=False)
        te, qe = extra.load_extra(verbose=False)
        eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
        frame, _ = model.walk_forward(model.build_frame(blind, eng))
    else:
        from cfbmodel import data, model
        from cfbmodel.ratings import CFBRatingEngine
        _, tg, qb = data.load_all(refresh=False, verbose=False)
        eng = CFBRatingEngine(tg, qb, games=games)
        frame, _ = model.walk_forward(model.build_frame(blind.assign(wind=np.nan), eng))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return frame


def early_games(lg: League, games: pd.DataFrame) -> pd.Series:
    if lg.league == "nfl":
        from nflmodel import backtest
        return games["game_id"].map(backtest._came_off_monday(games)).fillna(False).astype(bool)
    from cfbmodel import backtest
    return games["game_id"].map(backtest._short_week(games)).fillna(False).astype(bool)


def opener_gap(lg: League, frame: pd.DataFrame, season: int, open_q: dict, close_q: dict, dist: dict, skip: set) -> pd.DataFrame:
    """Every game with Pinnacle's open and close, the model's side on spread and total, graded
    two ways: at Pinnacle's open (line and price) against Pinnacle's close, and at the consensus
    opener (-110) against the consensus close the board's backtest uses."""
    f = frame[(frame["season"] == season) & frame["played"] & ~frame["game_id"].isin(skip)]
    out = []
    for r in f.itertuples():
        o = (open_q.get(r.game_id) or {}).get(M.SHARP) or {}
        c = (close_q.get(r.game_id) or {}).get(M.SHARP) or {}
        for market, mcol, lkey, okeys, cons_open, cons_close in (
                ("Spread", "model_margin", "spread_line", ("home_spread_odds", "away_spread_odds"), "open_spread", "spread_line"),
                ("Total", "model_total", "total_line", ("over_odds", "under_odds"), "open_total", "total_line")):
            if lkey not in o or lkey not in c or any(k not in o for k in okeys):
                continue
            model = getattr(r, mcol)
            if pd.isna(model):
                continue
            d = model - o[lkey]
            if d == 0:
                continue
            first = d > 0                                   # home (spread) or over (total)
            side = ("home" if first else "away") if market == "Spread" else ("over" if first else "under")
            odds = o[okeys[0] if first else okeys[1]]
            shown = (-o[lkey] if side == "home" else o[lkey]) if market == "Spread" else o[lkey]
            res = outcome(market, side, shown, r.result, r.total)
            row = {"season": season, "game_id": r.game_id, "market": market, "side": side, "gap": abs(d),
                   "open": o[lkey], "close": c[lkey], "odds": odds, "res": res, "profit": _profit(res, odds),
                   "clv_pts": (c[lkey] - o[lkey]) * (1 if first else -1),
                   "clv": clv(lg, dist, close_q, r.game_id, market, side, shown, odds)}
            # the old way, same game: the consensus opener at -110, the consensus close
            co, cc = getattr(r, cons_open, np.nan), getattr(r, cons_close, np.nan)
            if pd.notna(co) and pd.notna(cc) and model != co:
                f2 = model > co
                side2 = ("home" if f2 else "away") if market == "Spread" else ("over" if f2 else "under")
                shown2 = (-co if side2 == "home" else co) if market == "Spread" else co
                res2 = outcome(market, side2, shown2, r.result, r.total)
                row |= {"cons_gap": abs(model - co), "cons_res": res2, "cons_profit": _profit(res2, -110.0),
                        "cons_clv_pts": (cc - co) * (1 if f2 else -1)}
            out.append(row)
    return pd.DataFrame(out)


def gap_table(x: pd.DataFrame, market: str, steps) -> list[dict]:
    rows = []
    for t in steps:
        y = x[(x.market == market) & (x.gap >= t)]
        r = {"threshold": t, "pinnacle": record(y.assign(claimed=np.nan)) | {
            "clv_pts": round(float(y.clv_pts.mean()), 3) if len(y) else None}}
        if "cons_gap" in x:
            z = x[(x.market == market) & (x.cons_gap >= t)].dropna(subset=["cons_res"])
            z = z.assign(res=z.cons_res, profit=z.cons_profit, clv=np.nan)
            r["consensus"] = {k: v for k, v in record(z.assign(claimed=np.nan)).items() if not k.startswith(("clv", "beat", "claimed"))} | {
                "clv_pts": round(float(z.cons_clv_pts.mean()), 3) if len(z) else None}
        rows.append(r)
    return rows


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["nfl", "cfb"], required=True)
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--no-model", action="store_true", help="skip the opener-gap test (no model run)")
    a = ap.parse_args(argv)
    lg = League(a.league)
    games = lg.games()
    frame = None if a.no_model else model_frame(lg, games)
    if frame is not None and a.league == "nfl" and "open_spread" not in frame:
        from nflmodel import data
        frame = frame.merge(data.load_openers(refresh=False)[["game_id", "open_spread"]], on="game_id", how="left")
    skip = set(games.loc[early_games(lg, games), "game_id"])
    prices, gaps, cons = [], [], []
    for s in a.seasons:
        names = lg.team_names(s)
        open_q, close_q = quotes(a.league, s, games, names)
        if not open_q and not close_q:
            print(f"{a.league} {s}: no backfill under data/market/history/{a.league}/{s}/")
            continue
        dist = lg.bets.fit_distribution(games, s)
        for kind, q in (("open", open_q), ("close", close_q)):
            p = price_tier(lg, games, s, q, close_q, dist)
            if len(p):
                prices.append(p.assign(kind=kind))
        if frame is not None:
            gaps.append(opener_gap(lg, frame, s, open_q, close_q, dist, skip))
        # 3. the consensus close against Pinnacle's
        pc = anchor_rows(close_q)
        if len(pc):
            g = games[games["season"] == s][["game_id", "spread_line", "total_line"]].merge(pc, on="game_id", suffixes=("_cons", "_pin"))
            cons.append(g)
        print(f"{a.league} {s}: {len(open_q)} games with an open, {len(close_q)} with a close", flush=True)

    report = {"league": a.league, "seasons": a.seasons, "price_min": S.PRICE_MIN}
    if prices:
        P = pd.concat(prices, ignore_index=True)
        report["price_tier"] = {kind: {"all": record(x), **{m: record(y) for m, y in x.groupby("market")},
                                       "by_season": {int(s): record(y) for s, y in x.groupby("season")}}
                                for kind, x in P.groupby("kind")}
        P.to_csv(C.RAW / f"pinnacle_price_bets_{a.league}.csv", index=False)
    if gaps and any(len(g) for g in gaps):
        G = pd.concat(gaps, ignore_index=True)
        sp = S.OPENER_GAP[a.league]
        report["opener_gap"] = {"spread": gap_table(G, "Spread", (0, 1, sp, sp + 1)),
                                "total": gap_table(G, "Total", (0, 2, S.TOTAL_GAP, S.TOTAL_GAP + 2)),
                                "games_left_out_short_week": int(games[games["season"].isin(a.seasons)]["game_id"].isin(skip).sum())}
        G.to_csv(C.RAW / f"pinnacle_opener_bets_{a.league}.csv", index=False)
    if cons:
        X = pd.concat(cons, ignore_index=True)
        report["consensus_vs_pinnacle_close"] = {
            "games": int(len(X)),
            "spread_mean_abs": round(float((X.spread_line_cons - X.spread_line_pin).abs().mean()), 3),
            "total_mean_abs": round(float((X.total_line_cons - X.total_line_pin).abs().mean()), 3),
            "spread_differs": round(float(((X.spread_line_cons - X.spread_line_pin).abs() >= 0.5).mean()), 3)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"pinnacle_grade_{a.league}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main()
