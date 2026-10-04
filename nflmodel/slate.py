"""The live slate: projections and priced bets for games not yet played."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C
from . import bets as B
from . import odds as O

LINE_COLS = ["spread_line", "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds",
             "home_moneyline", "away_moneyline"]


def apply_book_lines(df: pd.DataFrame, path=None) -> tuple[pd.DataFrame, int]:
    """Price bets at your own book's numbers, if lines.csv is present.

    The projection stays anchored to the consensus line; only the prices the
    bets are graded against change. This is how a soft or stale number at one
    book shows up as value.
    """
    path = path or C.ROOT / "lines.csv"
    if not path.exists():
        return df, 0
    book = pd.read_csv(path).set_index("game_id")
    df = df.copy()
    hit = df.game_id.isin(book.index)
    for col in LINE_COLS:
        if col in book:
            vals = df.game_id.map(book[col])
            df[col] = vals.where(vals.notna(), df[col])
    return df, int(hit.sum())


HISTORY_COLS = ["game_id", "seen", "spread_line", "total_line", "home_moneyline", "away_moneyline"]


def track_lines(up: pd.DataFrame, path=None) -> dict:
    """Record each game's line the first time it is seen and whenever it moves.

    The data source only carries the current line, so the history is built up by
    this file, one run at a time. Returns the first tracked line for each game.
    """
    path = path or C.DATA / "line_history.csv"
    hist = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=HISTORY_COLS)
    last = hist.groupby("game_id").tail(1).set_index("game_id") if len(hist) else hist.set_index("game_id")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    new = []
    for g in up.itertuples(index=False):
        cur = [float(getattr(g, c)) if pd.notna(getattr(g, c)) else np.nan for c in HISTORY_COLS[2:]]
        if g.game_id in last.index:
            old = last.loc[g.game_id, HISTORY_COLS[2:]].astype(float).to_numpy()
            if np.allclose(old, cur, equal_nan=True):
                continue
        new.append([g.game_id, now] + cur)
    if new:
        hist = pd.concat([hist, pd.DataFrame(new, columns=HISTORY_COLS)], ignore_index=True)
        hist.to_csv(path, index=False)
    first = hist.groupby("game_id").head(1).set_index("game_id")
    moves = hist.groupby("game_id").size()
    return {gid: {"seen": str(r.seen)[:10], "spread": float(r.spread_line), "total": float(r.total_line), "moves": int(moves[gid]) - 1}
            for gid, r in first.iterrows()}


def fair_american(p: float) -> int | None:
    """The price at which a bet with win probability p breaks even."""
    if not 0 < p < 1:
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def build(frame: pd.DataFrame, dist: dict, engine, builder=None, records: dict | None = None, openers: pd.DataFrame | None = None) -> dict:
    up = frame[~frame.played & frame.spread_line.notna() & frame.total_line.notna()].copy()
    if up.empty:
        return {"weeks": [], "games": [], "book_lines": 0}
    opened = track_lines(up)
    true_open = dict(zip(openers.game_id, openers.open_spread)) if openers is not None and len(openers) else {}
    records = records or {}
    priced, n_book = apply_book_lines(up)
    adj = B.price_games(priced, dist, "proj")
    raw = B.price_games(priced, dist, "model")
    key = ["game_id", "market", "side"]
    b = adj.merge(raw[key + ["model_prob", "ev"]].rename(columns={"model_prob": "raw_prob", "ev": "raw_ev"}), on=key)

    games = []
    for g in up.sort_values(["week", "gameday", "gametime", "game_id"]).itertuples(index=False):
        gb = b[b.game_id == g.game_id]
        ml = gb[gb.market == "Moneyline"].set_index("side")
        bets = []
        for r in gb.itertuples(index=False):
            bets.append({
                "market": r.market, "side": r.side, "pick": r.pick, "odds": int(r.odds),
                "prob": round(r.model_prob, 4), "raw_prob": round(r.raw_prob, 4), "market_prob": round(r.market_prob, 4),
                "breakeven": round(r.breakeven, 4), "edge": round(r.edge, 4), "ev": round(r.ev, 4), "raw_ev": round(r.raw_ev, 4),
                "fair": fair_american(r.model_prob), "stake": round(r.stake, 4),
            })

        def qb(side):
            qid = getattr(g, f"{side}_qb_id")
            name = getattr(g, f"{side}_qb_name")
            if not isinstance(name, str):
                name = engine.q_name.get(qid, "Unknown")
            return {"name": name, "value": round(float(getattr(g, f"{side}_qb_val")), 3),
                    "delta": round(float(getattr(g, f"{side}_qb_delta")), 3)}

        when = pd.Timestamp(g.gameday)
        games.append({
            "id": g.game_id, "week": int(g.week), "date": when.strftime("%a %b %-d"),
            "time": g.gametime if isinstance(g.gametime, str) else "",
            "away": g.away_team, "home": g.home_team, "neutral": bool(g.neutral),
            "away_name": C.TEAM_NAMES.get(g.away_team, g.away_team), "home_name": C.TEAM_NAMES.get(g.home_team, g.home_team),
            "away_record": records.get(g.away_team, "0-0"), "home_record": records.get(g.home_team, "0-0"),
            "early": bool(getattr(g, "early", False)), "weight": round(float(g.w_margin), 3),
            "open": opened.get(g.game_id),
            "opening_spread": float(true_open[g.game_id]) if g.game_id in true_open else None,
            "detail": builder.detail(g) if builder is not None else None,
            "roof": g.roof if isinstance(g.roof, str) else "unknown",
            "away_qb": qb("away"), "home_qb": qb("home"),
            "market": {"spread": float(g.spread_line), "total": float(g.total_line)},
            "model": {"margin": round(float(g.model_margin), 2), "total": round(float(g.model_total), 2)},
            "proj": {"margin": round(float(g.proj_margin), 2), "total": round(float(g.proj_total), 2)},
            "win": {"adj": round(float(ml.loc["home", "model_prob"]), 4) if "home" in ml.index else None,
                    "raw": round(float(ml.loc["home", "raw_prob"]), 4) if "home" in ml.index else None,
                    "market": round(float(ml.loc["home", "market_prob"]), 4) if "home" in ml.index else None},
            "bets": bets,
        })
    return {"weeks": sorted({g["week"] for g in games}), "games": games, "book_lines": n_book}


def season_records(games: pd.DataFrame, season: int) -> dict:
    rec = {}
    for g in games[(games.season == season) & games.played].itertuples(index=False):
        for team, diff in ((g.home_team, g.result), (g.away_team, -g.result)):
            w, l, d = rec.get(team, (0, 0, 0))
            rec[team] = (w + (diff > 0), l + (diff < 0), d + (diff == 0))
    return {t: "-".join(str(int(v)) for v in (r if r[2] else r[:2])) for t, r in rec.items()}


def ratings_table(engine, r: dict, coefs: dict, games: pd.DataFrame, frame: pd.DataFrame, season: int) -> list[dict]:
    """Team ratings in points per game versus an average team, plus the pieces."""
    t = engine.table(r)
    pass_share = 0.58   # share of plays that are dropbacks, for a blended per-play number
    t["off_epa"] = pass_share * t.pass_off + (1 - pass_share) * t.rush_off
    t["def_epa"] = pass_share * t.pass_def + (1 - pass_share) * t.rush_def
    stats = sum(coefs[f"d_{m}"] * (t[f"{m}_off"] - t[f"{m}_def"]) for m in ("pass", "rush", "sr", "pts"))

    # Each team's quarterback for its next game (falls back to last week's starter).
    up = frame[~frame.played].sort_values("order")
    nxt = {}
    for g in up.itertuples(index=False):
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            if team not in nxt:
                name = getattr(g, f"{side}_qb_name")
                qid = getattr(g, f"{side}_qb_id")
                nxt[team] = (name if isinstance(name, str) else engine.q_name.get(qid, "Unknown"),
                             float(getattr(g, f"{side}_qb_val")), float(getattr(g, f"{side}_qb_delta")))
    t["qb"] = [nxt.get(x, (q, np.nan, 0.0))[0] for x, q in zip(t.team, t.qb)]
    t["qb_val"] = [nxt.get(x, (None, r["qb"].get(i, C.QB_REPLACEMENT), 0.0))[1] for x, i in zip(t.team, t.qb_id)]
    t["qb_adj"] = [nxt.get(x, (None, None, 0.0))[2] for x in t.team] 
    t["qb_adj"] = t["qb_adj"] * coefs["d_qb"]
    t["stats"] = stats
    t["market"] = coefs["d_mkt"] * t["mkt"]
    t["rating"] = t["stats"] + t["market"] + t["qb_adj"]
    t["rating"] -= t["rating"].mean()

    rec = season_records(games, season)
    t = t.sort_values("rating", ascending=False).reset_index(drop=True)
    return [{
        "rank": i + 1, "team": x.team,
        "name": C.TEAM_NAMES.get(x.team, x.team), "record": rec.get(x.team, "0-0"),
        "rating": round(x.rating, 2), "market_rating": round(x.mkt, 2),
        "off_epa": round(x.off_epa, 3), "def_epa": round(x.def_epa, 3),
        "pass_off": round(x.pass_off, 3), "rush_off": round(x.rush_off, 3),
        "pass_def": round(x.pass_def, 3), "rush_def": round(x.rush_def, 3),
        "pts_off": round(x.pts_off, 2), "pts_def": round(x.pts_def, 2),
        "qb": x.qb, "qb_val": None if pd.isna(x.qb_val) else round(x.qb_val, 3),
    } for i, x in enumerate(t.itertuples(index=False))]
