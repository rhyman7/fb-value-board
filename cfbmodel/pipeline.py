"""Run the college board end to end and write docs/cfb/ (data.json and index.html)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from nflmodel.pipeline import _clean

from . import archive, backtest, bets, board, config as C, data, espn, model, players, players_data, teams as T, weather
from .ratings import CFBRatingEngine

PLAYER_PERIODS = {"tuning": (2017, 2019), "holdout": (C.HOLDOUT_FROM, 2100)}


def run(refresh: bool = True, with_backtest: bool = True, verbose: bool = True, live: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    say("Loading college data ...")
    games, tg, qb = data.load_all(refresh=refresh, verbose=verbose)
    season = int(games[games.played].season.max())
    up_weeks = sorted(games[~games.played & (games.season == season)].week.unique())[:2]
    n_espn = 0
    if live and up_weeks:
        games, n_espn = espn.apply(games, espn.current_lines(up_weeks))
        say(f"  ESPN lines for {n_espn} upcoming games")
        games = weather.fill_wind(games, verbose=verbose)
    else:
        games["wind_source"] = np.where(games["played"], "recorded", "")
    games.loc[~games.played & (games.wind_source != "forecast"), "wind"] = np.nan   # a typical day unless forecast

    say("Rating teams week by week ...")
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, info = model.walk_forward(model.build_frame(games, engine))

    prev_path = C.DOCS / "data.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    teams = T.load_teams(season, refresh=refresh)
    abbr = teams["abbr"].to_dict()
    names = teams["name"].to_dict()
    colors = {r.abbr: [r.color, r.alt] for r in teams.itertuples()}

    say("Fitting key numbers ...")
    dist = bets.fit_distribution(games[games.fbs_game == 1], season)

    # ---- backtest and opening lines --------------------------------------------------------
    opening = prev.get("openers")
    bt = prev.get("backtest")
    if with_backtest:
        say("Backtesting (walk-forward) ...")
        graded, dists = backtest.grade_all(games, frame[frame.season < season + 1])
        bt = backtest.summarize(frame, graded, dists)
        bt["wind_unders"] = backtest.wind_report(frame)
        blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan, wind=np.nan)
        early_frame, _ = model.walk_forward(model.build_frame(blind, engine))
        opening = backtest.opening_report(games, early_frame, dists)
        graded.round(4).to_csv(C.DATA / "backtest_bets.csv", index=False)

    w_early = float(opening["weight"]) if opening and opening.get("weight") is not None else float(info[season]["w_margin"])
    now = pd.Timestamp.now(tz="UTC")
    frame["early_line"] = (~frame.played) & ((pd.to_datetime(frame.kickoff, utc=True) - now).dt.days >= C.EARLY_DAYS)
    e = frame["early_line"] & frame.spread_line.notna()
    frame.loc[e, "w_margin"] = w_early
    frame.loc[e, "proj_margin"] = frame.loc[e, "spread_line"] + w_early * (frame.loc[e, "model_margin"] - frame.loc[e, "spread_line"])

    say("Building game pages ...")
    stats = board.season_stats(games, tg, season, refresh=refresh)
    builder = board.MatchupBuilder(games, qb, stats, engine, info[season], season, abbr)
    records = board.season_records(games, season)
    cur = frame[frame.season == season]
    slate = board.build_slate(cur, dist, engine, builder, records, abbr, names)
    upcoming = frame[~frame.played & frame.spread_line.notna() & (frame.season == season)]
    order_now = int(upcoming.order.min()) if len(upcoming) else int(frame.order.max()) + 1
    r = engine.ratings_at(order_now)

    say("Projecting players ...")
    pboard, pg, ptable = None, None, None
    try:
        pg, pinfo = players_data.load_player_games(refresh_current=refresh, verbose=verbose)
        ptable = players.build(games, frame, pg, pinfo)
        pproj = players.walk_forward(ptable, first_test=season)
        keys = {g["game_id"]: g["id"] for g in slate["games"]}
        pboard = players.board(pproj, ptable, pinfo, keys, abbr)
        if with_backtest:
            say("Backtesting player projections ...")
            pall = players.walk_forward(ptable, first_test=C.FIRST_TEST_SEASON)
            s = players.backtest_summary(pall, ptable, PLAYER_PERIODS)
            pboard["backtest"] = {"from": C.HOLDOUT_FROM, "rows": s["holdout"]["rows"], "tuning": s["tuning"]["rows"],
                                  "starter_right": s["holdout"]["starter_right"]}
        elif prev.get("players") and prev["players"].get("backtest"):
            pboard["backtest"] = prev["players"]["backtest"]
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  player projections skipped ({exc})")

    out = {
        "meta": {
            "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "season": season, "week": int(order_now % 100),
            "games_through": pd.Timestamp(games[games.played].kickoff.max()).tz_convert("America/New_York").strftime("%b %-d, %Y"),
            "seasons_of_data": f"{C.FIRST_SEASON}-{season}",
            "kelly_fraction": C.KELLY_FRACTION, "max_stake": C.MAX_STAKE, "colors": colors,
            "blend": {"margin": info[season]["w_margin"], "total": info[season]["w_total"], "lookback": C.BLEND_LOOKBACK,
                      "early": round(w_early, 3), "early_days": C.EARLY_DAYS},
            "coefs": {"margin": info[season]["margin_coefs"], "total": info[season]["total_coefs"]},
            "lines_from": f"ESPN for {n_espn} games, cfbfastR consensus otherwise" if n_espn else "cfbfastR consensus (an average across books)",
            "sources": ["cfbfastR play-by-play, schedules, box scores and lines (sportsdataverse)",
                        "ESPN scoreboard (current lines and prices)", "Open-Meteo (wind forecasts)"],
            "teams": int(len(teams)),
        },
        "slate": slate,
        "ratings": board.ratings_table(engine, r, info[season], games, frame[frame.season == season], season, teams),
        "key_numbers": backtest.key_number_report(games, dist, season + 1),
        "backtest": bt, "openers": opening, "players": pboard,
    }
    out = _clean(out)
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    try:
        week_now = int(order_now % 100)
        have_past = any(int(f.stem.split("_w")[1]) < week_now for f in archive.ARCH.glob(f"{season}_w*.json"))
        archive.update(out, games, pg, verbose=verbose)
        if not have_past or with_backtest:
            backfill(out, frame, games, dist, engine, abbr, names, season, ptable, pg, verbose)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  archive skipped ({exc})")
    build_page(out)
    say(f"Wrote {C.DOCS / 'data.json'} and {C.DOCS / 'index.html'}")
    return out


def backfill(out, frame, games, dist, engine, abbr, names, season, ptable, pg, verbose=True):
    """Past weeks of this season, from the walk-forward replay, for weeks the board did not see."""
    done = frame[(frame.season == season) & frame.played & frame.spread_line.notna() & frame.total_line.notna()].copy()
    if done.empty:
        return
    done["early_line"] = False
    ents = []
    for wk, x in done.groupby("week"):
        ents += board.entries(x, dist, engine, None, lambda g: board.season_records(games, season, before=g.kickoff), abbr, names, {})
    pb = {"players": [], "teams": []}
    if ptable is not None:
        try:
            pproj = players.walk_forward(ptable, first_test=season)
            keys = {g["game_id"]: g["id"] for g in ents}
            info = players_data.load_player_games(verbose=False)[1]
            pb = players.board(pproj, ptable, info, keys, abbr)
        except Exception as exc:
            print(f"  backfill players skipped ({exc})")
    n = archive.snapshot(out, games, root=archive.ARCH, backfilled=True, slate=ents, players=pb)
    g = archive.grade(games, pg)
    archive.write_index()
    if verbose:
        print(f"  backfill: {n} past games from the replay, {g} graded")


def build_page(payload: dict | None = None) -> None:
    template = C.SITE / "cfb-template.html"
    if not template.exists():
        return
    if payload is None:
        payload = json.loads((C.DOCS / "data.json").read_text())
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    (C.DOCS / "index.html").write_text(template.read_text().replace("/*__DATA__*/null", blob))
