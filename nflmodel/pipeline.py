"""Run everything end to end and write the dashboard data."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import backtest, bets, config as C, data, matchup, model, slate
from .ratings import RatingEngine


def _clean(o):
    """Make numpy / pandas values JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(refresh: bool = True, with_backtest: bool = True, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    say("Loading data ...")
    games, tg, qb, stats = data.load_all(refresh=refresh, verbose=verbose, with_stats=True)
    season = int(games[games.played].season.max())

    say("Rating teams week by week ...")
    engine = RatingEngine(tg, qb, games=games)
    frame, info = model.walk_forward(model.build_frame(games, engine))

    say("Fitting key numbers ...")
    dist = bets.fit_distribution(games, season)
    upcoming = frame[~frame.played & frame.spread_line.notna()]
    order_now = int(upcoming.order.min()) if len(upcoming) else int(frame.order.max()) + 1
    r = engine.ratings_at(order_now)
    builder = matchup.MatchupBuilder(games, qb, stats, engine, info[season], season)

    out = {
        "meta": {
            "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "season": season, "week": int(order_now % 100),
            "games_through": games[games.played].gameday.max().strftime("%b %-d, %Y"),
            "seasons_of_data": f"{C.FIRST_SEASON}-{season}",
            "kelly_fraction": C.KELLY_FRACTION, "max_stake": C.MAX_STAKE,
            "blend": {"margin": info[season]["w_margin"], "total": info[season]["w_total"], "lookback": C.BLEND_LOOKBACK},
            "coefs": {"margin": info[season]["margin_coefs"], "total": info[season]["total_coefs"]},
            "sources": ["nflverse play-by-play", "nflverse games file (schedule, results, closing lines)"],
        },
        "slate": slate.build(frame[frame.season == season], dist, engine, builder, slate.season_records(games, season)),
        "ratings": slate.ratings_table(engine, r, info[season]["margin_coefs"], games, frame, season),
        "key_numbers": backtest.key_number_report(games, dist, season),
    }
    if with_backtest:
        say("Backtesting (walk-forward, about a minute) ...")
        graded, _ = backtest.grade_all(games, frame)
        out["backtest"] = backtest.summarize(frame, graded)
        graded.round(4).to_csv(C.DATA / "backtest_bets.csv", index=False)
    else:
        prev = C.DOCS / "data.json"
        if prev.exists():
            out["backtest"] = json.loads(prev.read_text()).get("backtest")

    out = _clean(out)
    C.DOCS.mkdir(exist_ok=True)
    (C.DOCS / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    build_page(out)
    say(f"Wrote {C.DOCS / 'data.json'} and {C.DOCS / 'index.html'}")
    return out


def build_page(payload: dict | None = None) -> None:
    """Inline the data into the dashboard template so the page works anywhere,
    including opened straight from disk."""
    template = C.SITE / "template.html"
    if not template.exists():
        return
    if payload is None:
        payload = json.loads((C.DOCS / "data.json").read_text())
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    (C.DOCS / "index.html").write_text(template.read_text().replace("/*__DATA__*/null", blob))
