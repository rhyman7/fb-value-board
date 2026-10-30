"""Price every bet on a set of games: model probability vs market, EV, stake.

Used by both the backtest (played games, with results graded) and the live
slate (upcoming games). One row per side of each market, six rows per game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr, ndtri

from . import config as C
from . import distributions as D
from . import odds as O

DEFAULT_PRICE = -110.0  # used when a spread or total has no recorded price


def fit_distribution(games: pd.DataFrame, before_season: int, lookback: int = 10) -> dict:
    """Key-number multipliers and curve widths from the seasons before a season.

    Measured against closing lines, the best available estimate of where each
    game was centred.
    """
    h = games[games.played & games.spread_line.notna() & games.total_line.notna()
              & (games.season < before_season) & (games.season >= before_season - lookback)]
    # Multipliers and width depend on each other a little: fit the multipliers at
    # a typical width, fit the width, then refresh the multipliers at that width.
    mm = D.fit_multipliers(h.spread_line, h.result, 11.8, D.MARGINS, fold=True, shrink=8)
    tm = D.fit_multipliers(h.total_line, h.total, 13.1, D.TOTALS, fold=False, shrink=40)
    sd_m = D.fit_scale(h.spread_line, h.result, D.MARGINS, mm)
    sd_t = D.fit_scale(h.total_line, h.total, D.TOTALS, tm, bounds=(10.0, 17.0))
    mm = D.fit_multipliers(h.spread_line, h.result, sd_m, D.MARGINS, fold=True, shrink=8)
    tm = D.fit_multipliers(h.total_line, h.total, sd_t, D.TOTALS, fold=False, shrink=40)

    # Winning outright is its own question: fit the margin-to-win-probability
    # curve directly on wins and losses, using every season available.
    a = games[games.played & games.spread_line.notna() & (games.season < before_season) & (games.result != 0)]
    won, mu = (a.result > 0).to_numpy(), a.spread_line.to_numpy(float)
    nll = lambda sd: -np.sum(np.where(won, log_ndtr(mu / sd), log_ndtr(-mu / sd)))
    sd_w = float(minimize_scalar(nll, bounds=(8.0, 16.0), method="bounded").x)
    return {"margin": mm, "total": tm, "sd_margin": sd_m, "sd_total": sd_t, "sd_win": sd_w, "n": int(len(h))}


def _fmt(x: float, signed: bool = True) -> str:
    if signed and x == 0:
        return "PK"
    return f"{x:+g}" if signed else f"{x:g}"


def curves(df: pd.DataFrame, dist: dict, source: str = "proj", margins=None, totals=None, snap=None) -> dict:
    """The outcome curves for each game in `df` (reset index first), centred on `source`:
    the margin and total pmfs (one row per game) and each side's win chance.

    source = "proj"   market-adjusted projection (the one to bet from)
    source = "model"  raw model, before it is pulled toward the market
    source = "line"   the market alone (no disagreement): what the anchor's no-vig price says
    margins, totals: the outcome supports (the NFL's by default); snap: rounds a line to one a
    book would post (the college board's half-point rounding).
    """
    MG = D.MARGINS if margins is None else margins
    TT = D.TOTALS if totals is None else totals

    def col(name):
        return df[name].to_numpy(float) if name in df else np.full(len(df), np.nan)

    def price_of(name, default=None):
        x = col(name)
        return x if default is None else np.where(np.isnan(x), default, x)

    raw_spread, raw_total = col("spread_line"), col("total_line")     # positive = home favoured
    spread, total = (snap(raw_spread), snap(raw_total)) if snap else (raw_spread, raw_total)
    d_margin = np.zeros(len(df)) if source == "line" else col(f"{source}_margin") - raw_spread    # disagreement with the line
    d_total = np.zeros(len(df)) if source == "line" else col(f"{source}_total") - raw_total

    # Anchor each curve to the market's own no-vig price, then slide it by the disagreement.
    fair_home, _ = O.no_vig(price_of("home_spread_odds", DEFAULT_PRICE), price_of("away_spread_odds", DEFAULT_PRICE))
    fair_over, _ = O.no_vig(price_of("over_odds", DEFAULT_PRICE), price_of("under_odds", DEFAULT_PRICE))
    loc_m = D.anchor(spread, fair_home, dist["sd_margin"], MG, dist["margin"]) + d_margin
    loc_t = D.anchor(total, fair_over, dist["sd_total"], TT, dist["total"]) + d_total
    pm = D.pmf(loc_m, dist["sd_margin"], MG, dist["margin"])
    pt = D.pmf(loc_t, dist["sd_total"], TT, dist["total"])

    # Moneyline: start from the market's no-vig win probability, expressed as a
    # margin on the win-probability curve, and slide by the same disagreement.
    tie = pm[:, MG == 0].ravel()
    fair_ml, _ = O.no_vig(col("home_moneyline"), col("away_moneyline"))
    implied = np.where(np.isnan(fair_ml), raw_spread, dist["sd_win"] * ndtri(np.clip(fair_ml, 1e-4, 1 - 1e-4)))
    h_win = ndtr((implied + d_margin) / dist["sd_win"]) * (1 - tie)
    a_win = 1 - tie - h_win
    return {"pm": pm, "pt": pt, "tie": tie, "h_win": h_win, "a_win": a_win, "spread": spread, "total": total,
            "margins": MG, "totals": TT}


def stake_prob(p_win, p_push, fair, haircut: float):
    """The win probability the stake is sized from: the model's, shrunk toward the market's
    no-vig number by `haircut` (the share of claimed edge that has shown up in results,
    backtest.edge_haircut; 1 = size from the model's probability as is). Pushes stay as the
    model has them; the probability is shrunk on the no-push basis."""
    model_p = p_win / (1 - p_push)
    return (fair + haircut * (model_p - fair)) * (1 - p_push)


def price_games(df: pd.DataFrame, dist: dict, source: str = "proj", haircut: float = 1.0) -> pd.DataFrame:
    """Return the long table of bets for the games in `df`.

    source = "proj"   market-adjusted projection (the one to bet from)
    source = "model"  raw model, before it is pulled toward the market
    haircut: stakes are sized from the model probability shrunk toward the market's by this
    factor (stake_prob); the probabilities and EV reported are the model's either way.
    """
    df = df.reset_index(drop=True)

    def col(name):
        return df[name].to_numpy(float) if name in df else np.full(len(df), np.nan)

    def price_of(name, default=None):
        x = col(name)
        return x if default is None else np.where(np.isnan(x), default, x)

    cv = curves(df, dist, source)
    spread, total, pm, pt = cv["spread"], cv["total"], cv["pm"], cv["pt"]
    h_cov, s_push, a_cov = D.over_push_under(pm, D.MARGINS, spread)
    over, t_push, under = D.over_push_under(pt, D.TOTALS, total)
    tie, h_win, a_win = cv["tie"], cv["h_win"], cv["a_win"]

    res = col("result")
    tot = col("total")
    markets = [
        # market, side, label, price column, p_win, p_push, opposite price col, outcome (+1 win, 0 push, -1 loss)
        ("Spread", "home", [f"{t} {_fmt(-s)}" for t, s in zip(df.home_team, spread)], "home_spread_odds", h_cov, s_push, "away_spread_odds", np.sign(res - spread)),
        ("Spread", "away", [f"{t} {_fmt(s)}" for t, s in zip(df.away_team, spread)], "away_spread_odds", a_cov, s_push, "home_spread_odds", np.sign(spread - res)),
        ("Moneyline", "home", [f"{t} ML" for t in df.home_team], "home_moneyline", h_win, tie, "away_moneyline", np.sign(res)),
        ("Moneyline", "away", [f"{t} ML" for t in df.away_team], "away_moneyline", a_win, tie, "home_moneyline", np.sign(-res)),
        ("Total", "over", [f"Over {_fmt(t, False)}" for t in total], "over_odds", over, t_push, "under_odds", np.sign(tot - total)),
        ("Total", "under", [f"Under {_fmt(t, False)}" for t in total], "under_odds", under, t_push, "over_odds", np.sign(total - tot)),
    ]
    out = []
    for market, side, label, pc, p_win, p_push, oc, outcome in markets:
        default = None if market == "Moneyline" else DEFAULT_PRICE
        price, other = price_of(pc, default), price_of(oc, default)
        dec = O.american_to_decimal(price)
        fair, _ = O.no_vig(price, other)
        model_p = p_win / (1 - p_push)                      # win probability with pushes set aside
        ev = O.expected_value(p_win, price, p_push)
        profit = np.where(outcome > 0, dec - 1, np.where(outcome < 0, -1.0, 0.0))
        profit = np.where(np.isnan(outcome), np.nan, profit)
        stake = np.minimum(O.kelly(stake_prob(p_win, p_push, fair, haircut), price, p_push) * C.KELLY_FRACTION, C.MAX_STAKE)
        line = {"Spread": np.where(side == "home", -spread, spread), "Total": total}.get(market, np.full(len(df), np.nan))
        out.append(pd.DataFrame({
            "game_id": df["game_id"], "season": df["season"], "week": df["week"],
            "market": market, "side": side, "pick": label, "line": line, "odds": price,
            "model_prob": model_p, "market_prob": fair, "breakeven": 1 / dec,
            "edge": model_p - fair, "ev": ev, "p_push": p_push, "stake": stake,
            "outcome": outcome, "profit": profit,
        }))
    b = pd.concat(out, ignore_index=True)
    return b[b["odds"].notna() & b["ev"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------
# Line shopping
# --------------------------------------------------------------------------

def _side_prob(cv: dict, i: int, market: str, side: str, line) -> tuple[float, float]:
    """(p_win, p_push) for one side of one market at a book's own line, from the curves."""
    if market == "Moneyline":
        return (float(cv["h_win"][i]) if side == "home" else float(cv["a_win"][i])), float(cv["tie"][i])
    if market == "Spread":
        h, push, a = D.over_push_under(cv["pm"][i:i + 1], cv["margins"], line)
        return (float(h[0]) if side == "home" else float(a[0])), float(push[0])
    o, push, u = D.over_push_under(cv["pt"][i:i + 1], cv["totals"], line)
    return (float(o[0]) if side == "over" else float(u[0])), float(push[0])


def shop(df: pd.DataFrame, dist: dict, quotes: dict, books: list[str], source: str = "proj",
         kalshi_maker: bool = False, margins=None, totals=None, snap=None, haircut: float = 1.0) -> dict:
    """Every book's price on every side, scored at that book's own line.

    quotes: {game_id: {book: {spread_line, home_spread_odds, ..., total_line, over_odds, ...}}}
    (nflmodel/market.latest). Returns {(game_id, market, side): [quote, ...]} sorted best EV
    first, each quote {book, line, odds, odds_net, prob, p_push, ev, ev_line, stake}:
      prob, ev     from `source` (the market-adjusted projection)
      ev_line      from the anchor's own no-vig price with no model at all: positive means
                   this book's price beats the sharp reference, whatever the model thinks
      odds_net     the price after Kalshi's fee (equal to odds elsewhere)
      stake        quarter Kelly from `prob` shrunk toward the anchor's no-vig number at this
                   line by `haircut` (stake_prob), capped
    """
    df = df.reset_index(drop=True)
    cv = curves(df, dist, source, margins, totals, snap)
    cv0 = curves(df, dist, "line", margins, totals, snap)
    out: dict = {}
    for i, gid in enumerate(df["game_id"]):
        q = quotes.get(gid) or {}
        for book in books:
            b = q.get(book)
            if not b:
                continue
            for market, side, line_key, price_key in (
                    ("Spread", "home", "spread_line", "home_spread_odds"), ("Spread", "away", "spread_line", "away_spread_odds"),
                    ("Total", "over", "total_line", "over_odds"), ("Total", "under", "total_line", "under_odds"),
                    ("Moneyline", "home", None, "home_moneyline"), ("Moneyline", "away", None, "away_moneyline")):
                price = b.get(price_key)
                line = b.get(line_key) if line_key else None
                if price is None or (line_key and line is None):
                    continue
                net = float(O.kalshi_effective_odds(price, kalshi_maker)) if book == "kalshi" else float(price)
                p, push = _side_prob(cv, i, market, side, line)
                p0, push0 = _side_prob(cv0, i, market, side, line)
                ev = float(O.expected_value(p, net, push))
                stake = float(min(O.kelly(stake_prob(p, push, p0 / (1 - push0), haircut), net, push) * C.KELLY_FRACTION, C.MAX_STAKE))
                shown = (-line if side == "home" else line) if market == "Spread" else line
                out.setdefault((gid, market, side), []).append({
                    "book": book, "line": None if shown is None else float(shown), "odds": float(price), "odds_net": round(net, 1),
                    "prob": round(p / (1 - push), 4), "p_push": round(push, 4), "ev": round(ev, 4),
                    "ev_line": round(float(O.expected_value(p0, net, push0)), 4), "stake": round(stake, 4)})
    for k in out:
        out[k].sort(key=lambda x: -x["ev"])
    return out
