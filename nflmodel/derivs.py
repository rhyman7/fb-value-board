"""Derivative markets priced off the boards' full-game curves, for both boards: team totals
and first-half spreads, totals and moneylines.

Why. Books derive these lines from the main line with crude rules (half the spread, a bit
under half the total, a team total from the implied score), and limits are low, so they are
mispriced more often than full-game lines. The boards already carry a key-number curve for
the margin and the total, anchored so that at the sharp book's line each side equals its
no-vig price (distributions.py, bets.curves). Two things follow from those curves cheaply:

Team totals      Home points are (total + margin) / 2 and away points (total - margin) / 2.
                 Treating the margin and total curves as independent, the chance of every
                 whole-number team score is the sum of the joint probabilities of the
                 (margin, total) pairs that give it; pairs of different parity are impossible
                 and are dropped. A normal approximation with the fitted widths is kept beside
                 it (`team_pmf(..., normal=True)`); experiments/derivs_test.py says which
                 matches history (FEATURES.md, "Derivative markets").

First half       A half-game curve fitted the same way distributions.py fits the full game,
                 on half-time scores (data.load_half_scores, cfbmodel.data.load_half_scores):
                 the first-half margin is centred on `ratio_margin` x the full-game centre and
                 the first-half total on a straight line in the full-game total (`total_fit`),
                 with the halves' own key-number multipliers and widths (a tie at the half is
                 common, and 7 matters more than 3). Where the sharp book quotes the first-half
                 line, the half curve is anchored to its no-vig price exactly as the full-game
                 curve is anchored to the full-game line; otherwise it sits at the implied
                 centre. The model's disagreement with the full-game line is carried to the
                 half scaled by the same ratios.

Pricing. Every book's derivative price is scored at its own line through the curve exactly
as bets.shop does for the main markets (`shop` here has the same output shape), with `prob`
and `ev` from the market-adjusted projection and `ev_line` from the sharp book's own
derivative quote carried to the book's line by the curve (so "beats Pinnacle" means the same
as on the main markets, and is only set when Pinnacle quotes that derivative). `entries`
turns them into bet-shaped records (`g["derivs"]`), each with `deriv: True`, that the bet
sheet lists when a price beats the sharp number (sheet.annotate gives them a `price` tier or
None: the model's opinion on derivatives has no backtest yet, so no opener or model tier) and
the ledger records like any other bet (ledger.record; first-half rows are graded on the
half-time score, team totals on the team's points).

Sides: first-half markets use home/away and over/under like the main markets; team totals
use home_over, home_under, away_over, away_under.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bets as B
from . import config as C
from . import distributions as D
from . import odds as O

# The Odds API market keys (per-event endpoint) and the board names they become
API_MARKETS = ["team_totals", "spreads_h1", "totals_h1", "h2h_h1"]
MARKETS = ["Team total", "1H Spread", "1H Total", "1H Moneyline"]
TEAM_SIDES = ["home_over", "home_under", "away_over", "away_under"]
LOOKBACK = 10            # seasons of half-time scores the half curve is fitted on


# --------------------------------------------------------------------------
# Team totals from the margin and total curves
# --------------------------------------------------------------------------

def team_support(margins: np.ndarray, totals: np.ndarray) -> np.ndarray:
    return np.arange(0, (int(totals[-1]) + int(margins[-1])) // 2 + 1)


def team_pmf(pm: np.ndarray, pt: np.ndarray, margins: np.ndarray, totals: np.ndarray, side: str = "home",
             normal: bool = False) -> np.ndarray:
    """The chance of every whole-number score for one team, from one game's margin pmf `pm`
    and total pmf `pt` (rows of bets.curves output). side: home or away.
    normal=True: a normal curve with the mean and variance the two pmfs imply, bucketed."""
    pts = team_support(margins, totals)
    s = 1.0 if side == "home" else -1.0
    if normal:
        em, et = float((pm * margins).sum()), float((pt * totals).sum())
        vm, vt = float((pm * margins ** 2).sum()) - em ** 2, float((pt * totals ** 2).sum()) - et ** 2
        mu, sd = (et + s * em) / 2.0, np.sqrt(max(vm + vt, 1e-6)) / 2.0
        q = D.pmf(mu, sd, pts)[0]
        return q / q.sum()
    M, T = np.meshgrid(margins, totals, indexing="ij")           # M[i, j] = margins[i], T[i, j] = totals[j]
    joint = pm[:, None] * pt[None, :]
    team = (T + s * M)
    ok = ((T + M) % 2 == 0) & (team >= 0) & (T - s * M >= 0)     # same parity, both scores non-negative
    idx = np.where(ok, team // 2, 0).astype(int)
    q = np.zeros(len(pts))
    np.add.at(q, idx[ok], joint[ok])
    tot = q.sum()
    return q / tot if tot > 0 else q


def shift(q: np.ndarray, support: np.ndarray, delta) -> np.ndarray:
    """A pmf moved `delta` points up its support, by a piecewise-linear cdf, so a derived
    curve can be re-centred the way distributions.anchor re-centres a parametric one."""
    delta = float(delta)
    if abs(delta) < 1e-12:
        return q
    edges = np.concatenate([[support[0] - 0.5], support + 0.5])
    cdf = np.concatenate([[0.0], np.cumsum(q)])
    f = np.interp(edges - delta, edges, cdf, left=0.0, right=1.0)
    out = np.diff(f)
    tot = out.sum()
    return out / tot if tot > 0 else q


def anchor_pmf(q: np.ndarray, support: np.ndarray, line: float, target: float, span: float = 12.0) -> float:
    """The shift that makes P(over the line | no push) equal `target` (bisection)."""
    lo, hi = -span, span
    for _ in range(24):
        mid = (lo + hi) / 2
        over, push, _ = D.over_push_under(shift(q, support, mid)[None, :], support, line)
        if over[0] / max(1 - push[0], 1e-9) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------
# The half-game curve
# --------------------------------------------------------------------------

def fit_half(games: pd.DataFrame, half: pd.DataFrame, before_season: int, lookback: int = LOOKBACK,
             margins: np.ndarray | None = None, totals: np.ndarray | None = None, snap=None,
             sd_bounds=((4.0, 12.0), (4.0, 12.0))) -> dict | None:
    """Key-number multipliers, widths and centring for the first half, from the seasons before
    `before_season` (walk-forward), measured against the full-game closing lines. `half` has
    game_id, home_h1, away_h1 (data.load_half_scores). None when fewer than 300 games."""
    MG = D.MARGINS if margins is None else margins
    TT = D.TOTALS if totals is None else totals
    h = games[games.played & games.spread_line.notna() & games.total_line.notna()
              & (games.season < before_season) & (games.season >= before_season - lookback)]
    h = h.merge(half[["game_id", "home_h1", "away_h1"]], on="game_id")
    if len(h) < 300:
        return None
    sp, tl = (snap(h.spread_line.to_numpy(float)), snap(h.total_line.to_numpy(float))) if snap else (h.spread_line.to_numpy(float), h.total_line.to_numpy(float))
    m1 = (h.home_h1 - h.away_h1).to_numpy(float)
    t1 = (h.home_h1 + h.away_h1).to_numpy(float)
    a_m = float((sp * m1).sum() / max((sp ** 2).sum(), 1e-9))        # through the origin: a pick is a pick in the half too
    b1, b0 = np.polyfit(tl, t1, 1)
    mu_m, mu_t = a_m * sp, b0 + b1 * tl
    sd0_m, sd0_t = float(np.std(m1 - mu_m)), float(np.std(t1 - mu_t))
    mm = D.fit_multipliers(mu_m, m1, sd0_m, MG, fold=True, shrink=8)
    tm = D.fit_multipliers(mu_t, t1, sd0_t, TT, fold=False, shrink=40)
    sd_m = D.fit_scale(mu_m, m1, MG, mm, bounds=sd_bounds[0])
    sd_t = D.fit_scale(mu_t, t1, TT, tm, bounds=sd_bounds[1])
    mm = D.fit_multipliers(mu_m, m1, sd_m, MG, fold=True, shrink=8)
    tm = D.fit_multipliers(mu_t, t1, sd_t, TT, fold=False, shrink=40)
    return {"margin": mm, "total": tm, "sd_margin": sd_m, "sd_total": sd_t, "ratio_margin": a_m,
            "total_fit": (float(b0), float(b1)), "plain_sd": (sd0_m, sd0_t), "n": int(len(h))}


def half_summary(hdist: dict | None) -> dict | None:
    """The half curve's settings for data.json (meta.half)."""
    if not hdist:
        return None
    return {"ratio_margin": round(hdist["ratio_margin"], 3), "total_fit": [round(x, 3) for x in hdist["total_fit"]],
            "sd_margin": round(hdist["sd_margin"], 2), "sd_total": round(hdist["sd_total"], 2), "n": hdist["n"]}


def _q(quotes: dict, gid, book: str) -> dict:
    return (quotes.get(gid) or {}).get(book) or {}


def half_curves(df: pd.DataFrame, dist: dict, hdist: dict, quotes: dict, source: str = "proj",
                margins=None, totals=None, snap=None, sharp: str = "pinnacle") -> dict:
    """First-half curves for the games in `df` (reset index first), in bets.curves' shape, so
    bets._side_prob reads them: pm and pt (one row per game), tie, h_win, a_win.
    source = "line": the market alone; "proj"/"model": slid by the disagreement scaled to
    the half. Centred on the sharp book's first-half quote where it has one, else on the
    implied centre from the full-game curve."""
    MG = D.MARGINS if margins is None else margins
    TT = D.TOTALS if totals is None else totals
    full = B.curves(df, dist, "line", MG, TT, snap)
    a_m, (b0, b1) = hdist["ratio_margin"], hdist["total_fit"]
    n = len(df)
    loc_m, loc_t = a_m * full["loc_m"], b0 + b1 * full["loc_t"]
    for i, gid in enumerate(df["game_id"]):
        q = _q(quotes, gid, sharp)
        if all(k in q for k in ("h1_spread_line", "h1_home_spread_odds", "h1_away_spread_odds")):
            fair, _ = O.no_vig(q["h1_home_spread_odds"], q["h1_away_spread_odds"])
            L = snap(q["h1_spread_line"]) if snap else q["h1_spread_line"]
            loc_m[i] = D.anchor(L, fair, hdist["sd_margin"], MG, hdist["margin"])[0]
        if all(k in q for k in ("h1_total_line", "h1_over_odds", "h1_under_odds")):
            fair, _ = O.no_vig(q["h1_over_odds"], q["h1_under_odds"])
            L = snap(q["h1_total_line"]) if snap else q["h1_total_line"]
            loc_t[i] = D.anchor(L, fair, hdist["sd_total"], TT, hdist["total"])[0]
    if source != "line":
        col = lambda name: df[name].to_numpy(float) if name in df else np.full(n, np.nan)
        d_m = np.nan_to_num(col(f"{source}_margin") - col("spread_line"))
        d_t = np.nan_to_num(col(f"{source}_total") - col("total_line"))
        loc_m, loc_t = loc_m + a_m * d_m, loc_t + b1 * d_t
    pm = D.pmf(loc_m, hdist["sd_margin"], MG, hdist["margin"])
    pt = D.pmf(loc_t, hdist["sd_total"], TT, hdist["total"])
    tie = pm[:, MG == 0].ravel()
    h_win = (pm * (MG[None, :] > 0)).sum(axis=1)
    return {"pm": pm, "pt": pt, "tie": tie, "h_win": h_win, "a_win": 1 - tie - h_win, "margins": MG, "totals": TT,
            "loc_m": loc_m, "loc_t": loc_t}


# --------------------------------------------------------------------------
# Shopping every book's derivative price
# --------------------------------------------------------------------------

def _quote(book, shown_line, price, net, p, push, p0, push0, lined: bool, haircut: float) -> dict:
    ev = float(O.expected_value(p, net, push))
    stake = float(min(O.kelly(B.stake_prob(p, push, p0 / (1 - push0), haircut), net, push) * C.KELLY_FRACTION, C.MAX_STAKE))
    return {"book": book, "line": None if shown_line is None else float(shown_line), "odds": float(price), "odds_net": round(net, 1),
            "prob": round(p / (1 - push), 4), "p_push": round(push, 4), "prob_line": round(p0 / (1 - push0), 4), "ev": round(ev, 4),
            "ev_line": round(float(O.expected_value(p0, net, push0)), 4) if lined and B.FLAG_ODDS[0] <= float(price) <= B.FLAG_ODDS[1] else None,
            "stake": round(stake, 4)}


def shop(df: pd.DataFrame, dist: dict, hdist: dict | None, quotes: dict, books: list[str], source: str = "proj",
         kalshi_maker: bool = False, margins=None, totals=None, snap=None, haircut: float = 1.0,
         sharp: str = "pinnacle") -> dict:
    """Every book's price on every derivative side, scored at that book's own line, in
    bets.shop's shape: {(game_id, market, side): [quote, ...]} best EV first, each quote
    {book, line, odds, odds_net, prob, p_push, prob_line, ev, ev_line, stake}. prob_line is
    the market curve's chance at the book's line; ev_line is set only when the sharp book
    quotes that derivative (its no-vig number carried to the book's line through the curve).
    hdist None: no first-half markets."""
    df = df.reset_index(drop=True)
    if df.empty:
        return {}
    MG = D.MARGINS if margins is None else margins
    TT = D.TOTALS if totals is None else totals
    cv, cv0 = B.curves(df, dist, source, MG, TT, snap), B.curves(df, dist, "line", MG, TT, snap)
    hv = hv0 = None
    if hdist:
        hv, hv0 = half_curves(df, dist, hdist, quotes, source, MG, TT, snap, sharp), half_curves(df, dist, hdist, quotes, "line", MG, TT, snap, sharp)
    pts = team_support(MG, TT)
    out: dict = {}
    for i, gid in enumerate(df["game_id"]):
        q = quotes.get(gid) or {}
        sq = q.get(sharp) or {}
        net_of = lambda book, price: float(O.kalshi_effective_odds(price, kalshi_maker)) if book == "kalshi" else float(price)

        # ---- team totals: a derived pmf per team, re-centred on the sharp book's quote where it has one
        for team in ("home", "away"):
            tq, tq0 = team_pmf(cv["pm"][i], cv["pt"][i], MG, TT, team), team_pmf(cv0["pm"][i], cv0["pt"][i], MG, TT, team)
            keys = (f"tt_{team}_line", f"tt_{team}_over_odds", f"tt_{team}_under_odds")
            lined = all(k in sq for k in keys)
            if lined:
                fair, _ = O.no_vig(sq[keys[1]], sq[keys[2]])
                L = snap(sq[keys[0]]) if snap else sq[keys[0]]
                d = anchor_pmf(tq0, pts, L, fair)
                tq, tq0 = shift(tq, pts, d), shift(tq0, pts, d)
            for book in books:
                b = q.get(book)
                if not b or not all(k in b for k in keys):
                    continue
                L = b[keys[0]]
                over, push, under = D.over_push_under(tq[None, :], pts, L)
                over0, push0, under0 = D.over_push_under(tq0[None, :], pts, L)
                for ou, p, p0 in (("over", over[0], over0[0]), ("under", under[0], under0[0])):
                    price = b[f"tt_{team}_{ou}_odds"]
                    out.setdefault((gid, "Team total", f"{team}_{ou}"), []).append(
                        _quote(book, L, price, net_of(book, price), p, push[0], p0, push0[0], lined, haircut))

        # ---- first half: the half curve read exactly like the full-game one
        if hv is None:
            continue
        lined = {"Spread": all(k in sq for k in ("h1_spread_line", "h1_home_spread_odds", "h1_away_spread_odds")),
                 "Total": all(k in sq for k in ("h1_total_line", "h1_over_odds", "h1_under_odds")),
                 "Moneyline": all(k in sq for k in ("h1_home_moneyline", "h1_away_moneyline"))}
        for book in books:
            b = q.get(book)
            if not b:
                continue
            for market, side, line_key, price_key in (
                    ("Spread", "home", "h1_spread_line", "h1_home_spread_odds"), ("Spread", "away", "h1_spread_line", "h1_away_spread_odds"),
                    ("Total", "over", "h1_total_line", "h1_over_odds"), ("Total", "under", "h1_total_line", "h1_under_odds"),
                    ("Moneyline", "home", None, "h1_home_moneyline"), ("Moneyline", "away", None, "h1_away_moneyline")):
                price = b.get(price_key)
                line = b.get(line_key) if line_key else None
                if price is None or (line_key and line is None):
                    continue
                L = (snap(line) if snap else line) if line is not None else None
                p, push = B._side_prob(hv, i, market, side, L)
                p0, push0 = B._side_prob(hv0, i, market, side, L)
                shown = (-L if side == "home" else L) if market == "Spread" else L
                out.setdefault((gid, "1H " + market, side), []).append(
                    _quote(book, shown, price, net_of(book, price), p, push, p0, push0, lined[market], haircut))
    for k in out:
        out[k].sort(key=lambda x: -x["ev"])
    return out


def _label(market: str, side: str, home: str, away: str, line) -> str:
    team = home if side.startswith("home") else away
    fmt = lambda x: "PK" if x == 0 else f"{x:+g}"
    if market == "Team total":
        return f"{team} {'over' if side.endswith('over') else 'under'} {line:g}"
    if market == "1H Spread":
        return f"1H {team} {fmt(line)}"
    if market == "1H Total":
        return f"1H {'Over' if side == 'over' else 'Under'} {line:g}"
    return f"1H {team} ML"


def entries(df: pd.DataFrame, shopped: dict, sharp: str = "pinnacle") -> dict:
    """{game_id: [bet-shaped records]} from shop()'s output (run with the sharp book among the
    books, so its own derivative quote is on file): one record per derivative side a retail
    book quotes, with the sharp book's quote as the reference price where it has one (else
    the best retail price), the retail quotes as `books`/`best*`/`beats_sharp` (slate.bet_books,
    the sharp book is never a place to bet) and deriv: True. sheet.annotate adds the tier."""
    from .slate import bet_books
    retail = {k: [x for x in v if x["book"] != sharp] for k, v in shopped.items()}
    out: dict = {}
    order = {m: i for i, m in enumerate(MARKETS)}
    for g in df.itertuples(index=False):
        gid = g.game_id
        recs = []
        for (gg, market, side), qs in retail.items():
            if gg != gid or not qs:
                continue
            books = bet_books(retail, gid, market, side)
            sq = next((x for x in shopped[(gg, market, side)] if x["book"] == sharp), None)
            ref = sq or qs[0]
            dec = O.american_to_decimal(ref["odds"])
            recs.append({"market": market, "side": side, "pick": _label(market, side, g.home_team, g.away_team, ref["line"]),
                         "line": ref["line"], "odds": int(round(ref["odds"])),
                         "prob": ref["prob"], "raw_prob": ref["prob"], "market_prob": ref["prob_line"],
                         "breakeven": round(float(1 / dec), 4), "edge": round(ref["prob"] - ref["prob_line"], 4),
                         "ev": ref["ev"], "raw_ev": ref["ev"], "fair": B_fair(ref["prob"]), "stake": ref["stake"],
                         "deriv": True, "sharp": {k: sq[k] for k in ("line", "odds", "prob", "prob_line", "ev")} if sq else None,
                         **books})
        if recs:
            out[gid] = sorted(recs, key=lambda r: (order.get(r["market"], 9), r["side"]))
    return out


def B_fair(p: float) -> int | None:
    if not 0 < p < 1:
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))
