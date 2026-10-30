"""Run everything end to end and write the dashboard data."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import archive, backtest, bets, config as C, data, extra, injuries, market, matchup, model, overrides, players, players_data, slate, weather
from .features_ext import PRODUCTION_METRICS, ExtRatingEngine


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
    ov = overrides.load()                         # hand-entered starters and outs for upcoming games
    games = overrides.apply_starters(games, ov, verbose=verbose)
    openers = data.load_openers(refresh=refresh)
    team_extra, qb_extra = extra.load_extra(refresh_current=refresh, verbose=verbose)
    games = weather.fill_wind(games, verbose=verbose)
    inj = injuries.load_injuries(games, refresh_current=refresh, verbose=verbose)
    season = int(games[games.played].season.max())

    say("Rating teams week by week ...")
    engine = ExtRatingEngine(tg, qb, team_extra, qb_extra, extra_metrics=PRODUCTION_METRICS, games=games)
    base = model.build_frame(games, engine, injuries=inj)
    frame, info = model.walk_forward(base)
    wf_input = base                               # the frame the game model is fitted on (shadow.py refits it)

    prev_path = C.DOCS / "data.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}

    say("Projecting players ...")
    pg, ptable, pproj = None, None, None
    try:
        pg, pinfo = players_data.load_player_games(refresh_current=refresh, verbose=verbose)
        avail = players.load_availability(refresh_current=refresh, verbose=verbose)
        rush = injuries.load_pass_rush(games, refresh_current=refresh, verbose=verbose)
        R = None
        try:                                      # the role signals for the game model (game_roles.py)
            from . import game_roles, roles
            say("Role signals for the game model ...")
            R = roles.load_all(games, refresh_current=refresh, verbose=False)
            rows = players.player_rows(players.team_games(frame, pg), pg, avail, pinfo)
            rows = overrides.apply_out(rows, ov) if ov is not None else rows
            wf_input = game_roles.attach(base, game_roles.team_features(rows, R, avail))
            frame, info = model.walk_forward(wf_input)
        except Exception as exc:
            print(f"  role signals skipped ({exc}); game model runs without them")
        ptable = players.build(frame, pg, pinfo, avail, inj, rush, games=games, overrides=ov)
        pproj = players.walk_forward(ptable, first_test=season)
        upcoming_ids = set(frame.loc[~frame.played & frame.spread_line.notna() & (frame.season == season), "game_id"])
        # the new projections (players_ml.py), shown next to the current ones: trained on every
        # earlier season's walk-forward projections, so the full walk-forward runs every time
        pall, new = None, None
        try:
            say("New projections (role inputs, boosted model) ...")
            from . import players_ml, roles
            R = R if R is not None else roles.load_all(games, refresh_current=refresh, verbose=False)
            pall = players.walk_forward(ptable, first_test=C.FIRST_TEST_SEASON)
            ml = players_ml.questionable(players_ml.walk_forward(players_ml.add_features(pall, R), first=season), first=C.FIRST_TEST_SEASON)
            new = ml[ml["season"] == season]
        except Exception as exc:
            print(f"  new projections skipped ({exc})")
        player_board = players.board(pproj, pinfo, upcoming_ids, new=new)
        if with_backtest:
            say("Backtesting player projections ...")
            pall = pall if pall is not None else players.walk_forward(ptable, first_test=C.FIRST_TEST_SEASON)
            player_board["backtest"] = players.backtest_summary(pall, ptable, C.HOLDOUT_FROM)
        elif prev.get("players") and prev["players"].get("backtest"):
            player_board["backtest"] = prev["players"]["backtest"]
    except Exception as exc:                      # the game board must not fail because of the player board
        print(f"  player projections skipped ({exc})")
        player_board = None

    try:                                          # shadow test of the defensive inputs: logged, never shown
        from . import shadow
        shadow.run(wf_input, frame, games, tg, ptable, pproj, season,
                   refresh=refresh, verbose=verbose)
    except Exception as exc:
        print(f"  shadow test skipped ({exc})")

    say("Fitting key numbers ...")
    dist = bets.fit_distribution(games, season)
    upcoming = frame[~frame.played & frame.spread_line.notna()]
    order_now = int(upcoming.order.min()) if len(upcoming) else int(frame.order.max()) + 1
    r = engine.ratings_at(order_now)
    builder = matchup.MatchupBuilder(games, qb, stats, engine, info[season], season)

    # ---- early lines: games several days out are priced with the opening-line weight
    opening = None
    if with_backtest:
        say("Backtesting (walk-forward, about a minute) ...")
        graded, dists = backtest.grade_all(games, frame)
        if len(openers):
            say("Testing against opening lines ...")
            # An opener is posted before late quarterback news, so re-run the model
            # assuming each team starts whoever started the week before.
            blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan)
            # ... and with no injury information, which mostly arrives after openers are posted.
            early, _ = model.walk_forward(model.build_frame(blind, engine))
            opening = backtest.opening_report(games, early, openers, dists)
    else:
        opening = prev.get("openers")
    w_early = float(opening["weight"]) if opening and opening.get("weight") is not None else float(info[season]["w_margin"])
    today = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    frame = frame.copy()
    frame["early"] = (~frame.played) & ((frame.gameday - today).dt.days >= C.EARLY_DAYS)
    e = frame["early"] & frame.spread_line.notna()
    frame.loc[e, "w_margin"] = w_early
    frame.loc[e, "proj_margin"] = frame.loc[e, "spread_line"] + w_early * (frame.loc[e, "model_margin"] - frame.loc[e, "spread_line"])

    out = {
        "meta": {
            "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "season": season, "week": int(order_now % 100),
            "games_through": games[games.played].gameday.max().strftime("%b %-d, %Y"),
            "seasons_of_data": f"{C.FIRST_SEASON}-{season}",
            "kelly_fraction": C.KELLY_FRACTION, "max_stake": C.MAX_STAKE,
            "colors": {k: list(v) for k, v in C.TEAM_COLORS.items()},
            "blend": {"margin": info[season]["w_margin"], "total": info[season]["w_total"], "lookback": C.BLEND_LOOKBACK,
                      "early": round(w_early, 3), "early_days": C.EARLY_DAYS},
            "coefs": {"margin": info[season]["margin_coefs"], "total": info[season]["total_coefs"]},
            "sources": ["nflverse play-by-play", "nflverse games file (schedule, results, closing lines)", "nfelo games file (opening lines)",
                        "The Odds API (DraftKings and FanDuel player props)"],
        },
        "slate": slate.build(frame[frame.season == season], dist, engine, builder, slate.season_records(games, season), openers,
                             quotes=market.quotes_for(frame[(frame.season == season) & ~frame.played], "nfl")),
        "ratings": slate.ratings_table(engine, r, info[season]["margin_coefs"], games, frame, season),
        "key_numbers": backtest.key_number_report(games, dist, season),
    }
    out["openers"] = opening
    out["players"] = player_board
    if with_backtest:
        out["backtest"] = backtest.summarize(frame, graded)
        out["backtest"]["wind_unders"] = backtest.wind_report(frame)
        graded.round(4).to_csv(C.DATA / "backtest_bets.csv", index=False)
    else:
        out["backtest"] = prev.get("backtest")

    out = _clean(out)
    C.DOCS.mkdir(exist_ok=True)
    (C.DOCS / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    try:                                          # keep each week's predictions, frozen at kickoff
        done = games[games.played & (games.season == season)]
        close = market.closing_quotes(done, "nfl", {g.game_id: archive.kickoff(g.gameday, g.gametime) for g in done.itertuples()})
        archive.update(out, games, pg, verbose=verbose, close=close, curve=lambda df: bets.curves(df, dist, "line"))
    except Exception as exc:
        print(f"  archive skipped ({exc})")
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
