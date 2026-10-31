"""Run the college board end to end and write docs/cfb/ (data.json and index.html)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from nflmodel import backtest as NBT, bestbets, derivs, ledger, market, sheet
from nflmodel.pipeline import _clean, haircut_factor

from . import archive, backtest, bets, board, config as C, cfbd, data, designations as DES, espn, model, players, players_data, starters as QBS, teams as T, weather
from .ratings import CFBRatingEngine

PLAYER_PERIODS = {"tuning": (2017, 2019), "holdout": (C.HOLDOUT_FROM, 2100)}


def run(refresh: bool = True, with_backtest: bool = True, verbose: bool = True, live: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    say("Loading college data ...")
    games, tg, qb = data.load_all(refresh=refresh, verbose=verbose)
    season = int(games[games.played].season.max())
    up_weeks = sorted(games[~games.played & (games.season == season)].week.unique())[:2]
    n_espn = n_cfbd = 0
    if live and up_weeks:
        espn_lines = espn.current_lines(up_weeks)
        games, n_espn = espn.apply(games, espn_lines)
        say(f"  ESPN lines for {n_espn} upcoming games")
        if cfbd.enabled():       # second source, only with the CFBD_API_KEY secret: games ESPN did not price
            try:
                have = set(espn_lines.loc[espn_lines.spread_line.notna(), "game_id"]) if len(espn_lines) else set()
                games, n_cfbd = cfbd.apply(games, cfbd.current_lines(season, up_weeks), have)
                say(f"  CFBD lines for {n_cfbd} more upcoming games")
            except Exception as exc:
                print(f"  CFBD lines skipped ({type(exc).__name__}: {exc})")
        games = weather.fill_wind(games, verbose=verbose)
    else:
        games["wind_source"] = np.where(games["played"], "recorded", "")
    games.loc[~games.played & (games.wind_source != "forecast"), "wind"] = np.nan   # a typical day unless forecast

    say("Rating teams week by week ...")
    engine = CFBRatingEngine(tg, qb, games=games)
    teams = T.load_teams(season, refresh=refresh)
    abbr = teams["abbr"].to_dict()
    # injury designations (Covers.com, docs/cfb/designations.json): upcoming games only
    des_all = DES.listed(DES.load(), abbr)
    des_all = DES.drop_played(des_all, DES.played_from_qbs(qb, games), verbose=verbose)   # quarterbacks played through
    des_out = DES.outs(des_all, abbr)
    if len(des_out):
        say(f"  injury designations: {len(des_out)} players out or doubtful")
    # this week's starting quarterbacks (the CFB dashboard's daily check), then injuries on top
    if live:
        QBS.fetch(verbose=verbose)
    qb_picks = QBS.resolve(QBS.load(), games, qb, verbose=verbose)
    games = QBS.apply(games, qb_picks, engine, verbose=verbose)
    games = DES.apply_qbs(games, qb, des_out, engine, verbose=verbose)
    frame, info = model.walk_forward(model.build_frame(games, engine))

    prev_path = C.DOCS / "data.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    names = teams["name"].to_dict()
    colors = {r.abbr: [r.color, r.alt] for r in teams.itertuples()}

    say("Fitting key numbers ...")
    dist = bets.fit_distribution(games[games.fbs_game == 1], season)
    half = data.load_half_scores(refresh=refresh, verbose=verbose)     # half-time scores, for the first-half curve (nflmodel/derivs.py)
    hdist = derivs.fit_half(games[games.fbs_game == 1], half, season, lookback=8, margins=bets.MARGINS, totals=bets.TOTALS, snap=bets.half,
                            sd_bounds=((5.0, 16.0), (5.0, 16.0)))

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
    # the edge haircut (nflmodel/backtest.edge_haircut, same graded table layout): stakes only
    haircut = NBT.edge_haircut(graded, season) if with_backtest else (prev.get("meta") or {}).get("haircut")
    h_factor = haircut_factor(haircut, verbose)
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
    quotes = market.quotes_for(cur[~cur.played], "cfb", names={v: k for k, v in names.items()})
    slate = board.build_slate(cur, dist, engine, builder, records, abbr, names, quotes=quotes, haircut=h_factor, hdist=hdist)
    upcoming = frame[~frame.played & frame.spread_line.notna() & (frame.season == season)]
    order_now = int(upcoming.order.min()) if len(upcoming) else int(frame.order.max()) + 1
    r = engine.ratings_at(order_now)

    say("Projecting players ...")
    pboard, pg, pinfo, ptable = None, None, None, None
    try:
        pg, pinfo = players_data.load_player_games(refresh_current=refresh, verbose=verbose)
        des_all = DES.drop_played(des_all, DES.played_from_box(pg, pinfo, games), verbose=verbose)  # everyone else
        des_out = DES.outs(des_all, abbr)
        ptable = players.build(games, frame, pg, pinfo, des_out, qb_picks)
        pproj = players.walk_forward(ptable, first_test=season)
        keys = {g["game_id"]: g["id"] for g in slate["games"]}
        pboard = players.board(pproj, ptable, pinfo, keys, abbr, des_all)
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
            "kelly_fraction": C.KELLY_FRACTION, "max_stake": C.MAX_STAKE, "max_game_stake": C.MAX_GAME_STAKE, "colors": colors,
            "haircut": haircut, "haircut_on": bool(C.EDGE_HAIRCUT), "haircut_factor": h_factor,
            "blend": {"margin": info[season]["w_margin"], "total": info[season]["w_total"], "lookback": C.BLEND_LOOKBACK,
                      "early": round(w_early, 3), "early_days": C.EARLY_DAYS},
            "coefs": {"margin": info[season]["margin_coefs"], "total": info[season]["total_coefs"]},
            "lines_from": lines_from(n_espn, n_cfbd),
            "sources": ["cfbfastR play-by-play, schedules, box scores and lines (sportsdataverse)",
                        "ESPN scoreboard (current lines and prices)",
                        *(["CollegeFootballData.com (current lines for games ESPN has not priced)"] if n_cfbd else []),
                        "Open-Meteo (wind forecasts)"],
            "teams": int(len(teams)), "half": derivs.half_summary(hdist),
        },
        "slate": slate,
        "ratings": board.ratings_table(engine, r, info[season], games, frame[frame.season == season], season, teams),
        "key_numbers": backtest.key_number_report(games, dist, season + 1),
        "backtest": bt, "openers": opening, "players": pboard,
    }
    des_checked = None
    try:
        des_checked = json.loads(DES.PATH.read_text()).get("checked") if DES.PATH.exists() else None
    except Exception:
        pass
    out["meta"]["inputs"] = sheet.inputs_stamp(out["meta"]["generated"], market=slate.get("market_as_of"), designations=des_checked)
    sheet.bucket_stakes(slate["games"], opening, None, C.KELLY_FRACTION, C.MAX_STAKE)
    slate["moves"] = sheet.line_moves(slate["games"], "cfb", opening)   # the Line movement section at build time
    out["meta"]["max_week_stake"] = C.MAX_WEEK_STAKE
    out["meta"]["week_cap"] = sheet.cap_week(slate["games"], C.MAX_WEEK_STAKE)   # after the bucket stakes, before the ledger
    try:                                          # the ledger (nflmodel/ledger.py): flagged bets at their first price, graded
        cur = games[games.season == season]
        kick = {str(int(g.game_id)): pd.Timestamp(g.kickoff) for g in cur.itertuples()}
        n_new = ledger.record(slate["games"], season, {g["id"]: kick[str(g["game_id"])] for g in slate["games"] if str(g["game_id"]) in kick},
                              C.DATA / "ledger.csv")
        done = cur[cur.played]
        close = market.closing_quotes(done, "cfb", {g.game_id: pd.Timestamp(g.kickoff) for g in done.itertuples()}, names={v: k for k, v in names.items()})
        scores = {str(int(g.game_id)): (g.home_score, g.away_score) for g in done.itertuples()}
        halves = {str(int(h.game_id)): (h.home_h1, h.away_h1) for h in half.itertuples()} if len(half) else {}
        n_graded = ledger.grade(C.DATA / "ledger.csv", C.DATA / "ledger_results.csv", scores, {str(k): v for k, v in close.items()},
                                lambda df: bets.curves(df, dist, "line"), halves=halves)
        out["ledger"] = ledger.summary(C.DATA / "ledger.csv", C.DATA / "ledger_results.csv")
        say(f"  ledger: {n_new} bets flagged, {n_graded} graded")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  ledger skipped ({exc})")
        out["ledger"] = prev.get("ledger")
    try:                                          # the Best Bets record (nflmodel/bestbets.py): the card as it stood at kickoff
        kk = {g["id"]: kick[str(g["game_id"])] for g in slate["games"] if str(g["game_id"]) in kick}
        bestbets.record(slate["games"], season, kk, C.DATA / "bestbets.csv", "cfb")
        n_bb = bestbets.grade(C.DATA / "bestbets.csv", C.DATA / "bestbets_results.csv", scores, {str(k): v for k, v in close.items()},
                              lambda df: bets.curves(df, dist, "line"), halves=halves)
        out["best_record"] = bestbets.summary(C.DATA / "bestbets.csv", C.DATA / "bestbets_results.csv")
        say(f"  best bets record: {n_bb} graded")
    except Exception as exc:
        print(f"  best bets record skipped ({exc})")
        out["best_record"] = prev.get("best_record")
    out = _clean(out)
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    try:
        week_now = int(order_now % 100)
        have_past = any(int(f.stem.split("_w")[1]) < week_now for f in archive.ARCH.glob(f"{season}_w*.json"))
        done = games[games.played & (games.season == season)]
        close = market.closing_quotes(done, "cfb", {g.game_id: pd.Timestamp(g.kickoff) for g in done.itertuples()},
                                      names={v: k for k, v in names.items()})
        archive.update(out, games, pg, verbose=verbose, close=close,
                       curve=lambda df: bets.curves(df, dist, "line"), info=pinfo)
        if not have_past or with_backtest:
            backfill(out, frame, games, dist, engine, abbr, names, season, ptable, pg, verbose)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  archive skipped ({exc})")
    build_page(out)
    say(f"Wrote {C.DOCS / 'data.json'} and {C.DOCS / 'index.html'}")
    return out


def lines_from(n_espn: int, n_cfbd: int = 0) -> str:
    """The meta line saying where the upcoming games' lines came from."""
    parts = ([f"ESPN for {n_espn} games"] if n_espn else []) + ([f"CFBD for {n_cfbd} games"] if n_cfbd else [])
    return ", ".join(parts) + ", cfbfastR consensus otherwise" if parts else "cfbfastR consensus (an average across books)"


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
