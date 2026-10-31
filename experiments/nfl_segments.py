"""Where the NFL model's edge lives: the walk-forward backtest split into segments.

    python experiments/nfl_segments.py            # ~2 min the first time, cached after
    python experiments/nfl_segments.py --rebuild  # rebuild the cached model runs

Research only: nothing here changes the board, its bets, tiers or stakes.

The NFL counterpart of experiments/cfb_segments.py. It rebuilds the two walk-forward runs the
board's full backtest uses (nflmodel/pipeline.py):
  * the production run (ratings, quarterbacks, injuries and the role signals) that
    backtest.grade_all, summarize and wind_report grade at the close, and
  * the opener run (each team assumed to start last week's quarterback, no injury
    information) that backtest.opening_report grades at the opener, leaving out games where
    a team played the Monday night before,
checks that they reproduce the board's headline numbers, then splits every bucket by segment,
for the holdout (2019 on) and the tuning seasons (2014-2018) separately.

Buckets (each graded at the price the data records for that line, -110 where it has none, as
the board's own reports do; return per bet with one standard error):
  Spreads   raw model 2+ and 3+ off the closing spread (the college study's columns);
            raw model 2+ and 3+ off the opening spread, graded at the opener and the same
            sides at the close (the board's `opener` tier)
  Totals    raw model 3+ off the closing total (no historical NFL opening totals);
            wind unders: outdoor games with a recorded 10-19 mph wind, under at the closing
            total (the board's `wind` tier)
  Edges     market-adjusted spread bets with EV above 0 at the close (the "model edges",
            the board's `model` tier)
Pinnacle: the same buckets on 2024-2025 graded against Pinnacle's real lines and prices from
the market backfill (data/market/history/nfl/, via experiments/pinnacle_grade.quotes): the
opener buckets at Pinnacle's Monday open, the rest at Pinnacle's close; model edges are
re-priced from the market-adjusted projection blended off Pinnacle's closing spread.

Segments (each game in exactly one group per split):
  Matchup        division game (nflverse `div_game`) / same conference, not division /
                 other conference (AFC and NFC by team)
  Slot           prime time (kickoff 7 pm ET or later on a Thursday, Sunday or Monday; one
                 group) / Sunday day / Saturday / neutral site (international games and the
                 Super Bowl) / other (Thanksgiving and holiday day games, Friday, Tuesday,
                 Wednesday). Neutral site is checked first, then the day.
  Rest           short week if either team has 4-5 days' rest, else long if either has 10+
                 (off a bye), else normal (nflverse `home_rest`, `away_rest`)
  Rest gap       home rest minus away rest: home 2+ days more / within 1 day / away 2+ more
  Roof           dome or closed / retractable open / outdoors (nflverse `roof`)
  Wind           indoors (dome, closed or open retractable: the model's `indoor` is 0 for an
                 open roof, but the wind tier only bets outdoors) / outdoors with a recorded
                 wind under 10, 10-19, 20+ mph / outdoors, no recorded wind
  Season part    weeks 1-4, 5-9, 10-14, 15-18, playoffs (as the board's table)
  Line size      the closing spread: pick'em to 3, 3.5-7, 7.5-10, 10.5+ (the same closing
                 number for every bucket, so a game sits in one group; opener bets were made
                 at a number that may sit in the next group)
  Favourite      home favourite / road favourite / pick'em (closing spread)
  Quarterback    a change for either team (this game's starter is not the one who started
                 the team's previous game, across seasons) / no change

The rule, set before running:
  A segment is a lead only if, in one of the four betting buckets the board uses (opener
  2+, opener 3+, totals 3+ at the close, wind unders, model edges), its return is above zero
  in BOTH periods, the holdout return is at least two standard errors above zero, and it is
  still above zero when graded against Pinnacle on 2024-2025 (direction at least; no
  Pinnacle bets in the segment means it cannot pass). A segment that holds every bet in the
  bucket (outdoors, or 10-19 mph, for the wind unders) is the bucket itself, not a test. The closing-spread 2+/3+ columns are
  shown, as in the college study, but are not betting buckets on the NFL board and are not
  screened. Every segment x bucket cell with any holdout bets counts as a test; under no
  edge anywhere a cell passes about 0.6% of the time (2.3% for the holdout bar, times about
  half for each direction check), so several hundred tests would throw up one or two by
  chance. Cells overlap (2+ contains 3+; splits share games) so they are not independent.
  Anything that only shows up in one period is noise. A segment that passes inside a bucket
  that is positive overall may just be the bucket's own edge: the output also gives the
  segment's holdout return minus the rest of the bucket's, which the rule does not use.

Writes experiments/out/nfl_segments.md and prints it. The two model runs are cached in
data/raw/nfl_segments_frames.pkl (git-ignored); delete it (or --rebuild) after any change to
the model.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import backtest, bets as B, config as C, data, extra, injuries, model, odds as O, overrides, weather  # noqa: E402
from nflmodel import market as M  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "nfl_segments.md"
CACHE = C.RAW / "nfl_segments_frames.pkl"
PIN_SEASONS = (2024, 2025)
PERIODS = {"Holdout (2019 on)": lambda s: s >= C.HOLDOUT_FROM,
           "Tuning seasons (2014-2018)": lambda s: (s >= C.FIRST_TEST_SEASON) & (s < C.HOLDOUT_FROM)}
AFC = {"BAL", "BUF", "CIN", "CLE", "DEN", "HOU", "IND", "JAX", "KC", "LAC", "SD", "LV", "OAK", "MIA", "NE", "NYJ", "PIT", "TEN"}


# --------------------------------------------------------------------------
# The model runs (as nflmodel/pipeline.py builds them on a full-backtest run)
# --------------------------------------------------------------------------

def build(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        return pd.read_pickle(CACHE)
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    ov = overrides.load()
    games = overrides.apply_starters(games, ov, verbose=False)
    openers = data.load_openers(refresh=False)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    base = model.build_frame(games, engine, injuries=inj)
    frame, _ = model.walk_forward(base)
    from nflmodel import game_roles, players, players_data, roles
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = players.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    rows = players.player_rows(players.team_games(frame, pg), pg, avail, pinfo)
    rows = overrides.apply_out(rows, ov) if ov is not None else rows
    frame, _ = model.walk_forward(game_roles.attach(base, game_roles.team_features(rows, R, avail)))
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan)
    early, _ = model.walk_forward(model.build_frame(blind, engine))
    graded, dists = backtest.grade_all(games, frame)
    out = {"games": games, "openers": openers, "frame": frame, "early": early, "graded": graded, "dists": dists}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------

def qb_change(games: pd.DataFrame) -> pd.Series:
    """game_id -> True when either team's starter differs from its previous game's."""
    p = games[games.played].sort_values("order")
    long = pd.concat([p[["game_id", "order"]].assign(team=p[f"{s}_team"], qb=p[f"{s}_qb_id"]) for s in ("home", "away")])
    long = long.sort_values(["team", "order"])
    prev = long.groupby("team")["qb"].shift(1)
    long["chg"] = long["qb"].notna() & prev.notna() & (long["qb"] != prev)
    return long.groupby("game_id")["chg"].any()


def segments(games: pd.DataFrame) -> pd.DataFrame:
    """One row per played game from 2014 with a label for every split."""
    g = games[games.played & (games.season >= C.FIRST_TEST_SEASON)].copy()
    hc, ac = g.home_team.isin(AFC), g.away_team.isin(AFC)
    g["seg_matchup"] = np.where(g.div_game == 1, "Division", np.where(hc == ac, "Same conference, not division", "Other conference"))
    hour = pd.to_numeric(g.gametime.str[:2], errors="coerce").fillna(13)
    prime = g.weekday.isin(["Thursday", "Sunday", "Monday"]) & (hour >= 19)
    g["seg_slot"] = np.select([g.neutral == 1, prime, g.weekday.eq("Sunday"), g.weekday.eq("Saturday")],
                              ["Neutral site", "Prime time (TNF, SNF, MNF)", "Sunday day", "Saturday"], "Other (holiday day games, Fri, Tue, Wed)")
    hr, ar = g.home_rest, g.away_rest
    g["seg_rest"] = np.select([(hr <= 5) | (ar <= 5), (hr >= 10) | (ar >= 10)], ["Short week (either team 4-5 days)", "Long (either team 10+ days)"], "Normal")
    gap = hr - ar
    g["seg_restgap"] = np.select([gap >= 2, gap <= -2], ["Home 2+ days more rest", "Away 2+ days more rest"], "Within 1 day")
    g["seg_roof"] = np.select([g.roof.isin(["dome", "closed"]), g.roof.eq("open")], ["Dome or closed", "Retractable, open"], "Outdoors")
    outdoor = ~g.roof.isin(["dome", "closed", "open"])
    w = g.wind
    g["seg_wind"] = np.select([~outdoor, w.isna(), w < 10, w < 20], ["Indoors (incl. open roof)", "Outdoors, no wind recorded", "Outdoors, under 10 mph", "Outdoors, 10-19 mph"],
                              "Outdoors, 20+ mph")
    wk = g.week
    g["seg_part"] = np.select([wk >= 19, wk <= 4, wk <= 9, wk <= 14], ["Playoffs", "Weeks 1-4", "Weeks 5-9", "Weeks 10-14"], "Weeks 15-18")
    a = g.spread_line.abs()
    g["seg_size"] = np.select([a <= 3, a <= 7, a <= 10], ["Pick'em to 3", "3.5 to 7", "7.5 to 10"], "10.5+")
    g["seg_fav"] = np.select([g.spread_line > 0, g.spread_line < 0], ["Home favourite", "Road favourite"], "Pick'em")
    g["seg_qb"] = np.where(g.game_id.map(qb_change(games)).fillna(False).astype(bool), "Quarterback change", "No change")
    return g[["game_id"] + [c for _, c, _ in SPLITS]].set_index("game_id")


SPLITS = [("Matchup", "seg_matchup", ["Division", "Same conference, not division", "Other conference"]),
          ("Slot", "seg_slot", ["Prime time (TNF, SNF, MNF)", "Sunday day", "Saturday", "Neutral site", "Other (holiday day games, Fri, Tue, Wed)"]),
          ("Rest", "seg_rest", ["Short week (either team 4-5 days)", "Normal", "Long (either team 10+ days)"]),
          ("Rest gap", "seg_restgap", ["Home 2+ days more rest", "Within 1 day", "Away 2+ days more rest"]),
          ("Roof", "seg_roof", ["Dome or closed", "Retractable, open", "Outdoors"]),
          ("Wind", "seg_wind", ["Indoors (incl. open roof)", "Outdoors, under 10 mph", "Outdoors, 10-19 mph", "Outdoors, 20+ mph", "Outdoors, no wind recorded"]),
          ("Season part", "seg_part", ["Weeks 1-4", "Weeks 5-9", "Weeks 10-14", "Weeks 15-18", "Playoffs"]),
          ("Line size", "seg_size", ["Pick'em to 3", "3.5 to 7", "7.5 to 10", "10.5+"]),
          ("Favourite", "seg_fav", ["Home favourite", "Road favourite", "Pick'em"]),
          ("Quarterback", "seg_qb", ["Quarterback change", "No change"])]


# --------------------------------------------------------------------------
# Bets: one table per bucket, columns game_id, season, res (+1/0/-1), profit
# --------------------------------------------------------------------------

def _profit(res, price) -> np.ndarray:
    price = np.where(np.isnan(np.asarray(price, float)), B.DEFAULT_PRICE, np.asarray(price, float))
    dec = O.american_to_decimal(price)
    return np.where(res > 0, dec - 1, np.where(res < 0, -1.0, 0.0))


def close_spreads(frame) -> pd.DataFrame:
    f = frame[frame.played & frame.spread_line.notna() & (frame.season >= C.FIRST_TEST_SEASON)].copy()
    f["gap"] = f.model_margin - f.spread_line
    f = f[f.gap != 0]
    pick = np.sign(f.gap)
    f["res"] = np.sign(f.result - f.spread_line) * pick
    f["profit"] = _profit(f.res, np.where(pick > 0, f.home_spread_odds, f.away_spread_odds))
    return f[["game_id", "season", "gap", "res", "profit"]]


def opener_spreads(early, openers, games) -> pd.DataFrame:
    """As backtest.opening_report: the opener run against the opener, Monday games left out;
    `res_close`/`profit_close` are the same sides at the closing spread and price."""
    f = early.merge(openers, on="game_id", how="inner")
    monday = f.game_id.map(backtest._came_off_monday(games)).fillna(False).to_numpy(bool)
    f = f[f.played & f.spread_line.notna() & (f.season >= C.FIRST_TEST_SEASON) & ~monday].copy()
    f["gap"] = f.model_margin - f.open_spread
    f = f[f.gap != 0]
    pick = np.sign(f.gap)
    f["res"] = np.sign(f.result - f.open_spread) * pick
    f["profit"] = _profit(f.res, np.where(pick > 0, f.open_home_odds, f.open_away_odds))
    f["res_close"] = np.sign(f.result - f.spread_line) * pick
    f["profit_close"] = _profit(f.res_close, np.where(pick > 0, f.home_spread_odds, f.away_spread_odds))
    return f[["game_id", "season", "gap", "res", "profit", "res_close", "profit_close"]]


def close_totals(frame) -> pd.DataFrame:
    f = frame[frame.played & frame.total_line.notna() & (frame.season >= C.FIRST_TEST_SEASON)].copy()
    f["gap"] = f.model_total - f.total_line
    f = f[f.gap != 0]
    pick = np.sign(f.gap)
    f["res"] = np.sign(f.total - f.total_line) * pick
    f["profit"] = _profit(f.res, np.where(pick > 0, f.over_odds, f.under_odds))
    return f[["game_id", "season", "gap", "res", "profit"]]


def wind_unders(frame, first: int = C.FIRST_TEST_SEASON) -> pd.DataFrame:
    """As backtest.wind_report: outdoor games with a recorded 10-19 mph wind, under at the close."""
    f = frame[frame.played & frame.total_line.notna() & (frame.indoor == 0) & (frame.wind_known == 1)
              & (frame.wind >= 10) & (frame.wind < 20) & (frame.season >= first)].copy()
    f["res"] = np.sign(f.total_line - f.total)
    f["profit"] = _profit(f.res, f.under_odds)
    return f[["game_id", "season", "res", "profit"]]


def model_edges(graded) -> pd.DataFrame:
    g = graded[(graded.source == "proj") & (graded.market == "Spread") & (graded.ev > 0)]
    return pd.DataFrame({"game_id": g.game_id, "season": g.season, "res": g.outcome, "profit": g.profit})


# --------------------------------------------------------------------------
# Pinnacle, 2024-2025
# --------------------------------------------------------------------------

def pinnacle(R: dict) -> dict[str, pd.DataFrame]:
    """The same buckets graded at Pinnacle's lines and prices (open for the opener buckets)."""
    import pinnacle_grade as PG
    games, frame, early, dists = R["games"], R["frame"], R["early"], R["dists"]
    monday = set(games.loc[games.game_id.map(backtest._came_off_monday(games)).fillna(False).astype(bool), "game_id"])
    if "kickoff" not in games:
        t = pd.to_datetime(games["gameday"].astype(str) + " " + games["gametime"].fillna("13:00").astype(str), errors="coerce")
        games = games.assign(kickoff=t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"))
    out = {k: [] for k in ("open", "total", "wind", "edges")}
    for s in PIN_SEASONS:
        oq, cq = PG.quotes("nfl", s, games, None)
        pin = lambda q, gid: (q.get(gid) or {}).get(M.SHARP) or {}
        # opener spreads: the opener run against Pinnacle's open
        e = early[(early.season == s) & early.played & ~early.game_id.isin(monday)]
        for r in e.itertuples():
            o = pin(oq, r.game_id)
            if not all(k in o for k in ("spread_line", "home_spread_odds", "away_spread_odds")):
                continue
            d = r.model_margin - o["spread_line"]
            if d == 0:
                continue
            res = np.sign(r.result - o["spread_line"]) * np.sign(d)
            out["open"].append({"game_id": r.game_id, "season": s, "gap": d, "res": res,
                                "profit": float(_profit(np.array([res]), [o["home_spread_odds"] if d > 0 else o["away_spread_odds"]])[0])})
        # totals and wind: the production run against Pinnacle's close
        f = frame[(frame.season == s) & frame.played]
        rows = []
        for r in f.itertuples():
            c = pin(cq, r.game_id)
            if all(k in c for k in ("total_line", "over_odds", "under_odds")):
                d = r.model_total - c["total_line"]
                if d != 0:
                    res = np.sign(r.total - c["total_line"]) * np.sign(d)
                    out["total"].append({"game_id": r.game_id, "season": s, "gap": d, "res": res,
                                         "profit": float(_profit(np.array([res]), [c["over_odds"] if d > 0 else c["under_odds"]])[0])})
                if r.indoor == 0 and r.wind_known == 1 and 10 <= r.wind < 20:
                    res = np.sign(c["total_line"] - r.total)
                    out["wind"].append({"game_id": r.game_id, "season": s, "res": res,
                                        "profit": float(_profit(np.array([res]), [c["under_odds"]])[0])})
            if all(k in c for k in ("spread_line", "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds")):
                rows.append({"game_id": r.game_id, **{k: c[k] for k in ("spread_line", "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds")}})
        # model edges: the market-adjusted projection blended off Pinnacle's closing spread
        if rows and s in dists:
            p = pd.DataFrame(rows)
            x = f.drop(columns=["spread_line", "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds"]).merge(p, on="game_id")
            x["proj_margin"] = x.spread_line + x.w_margin * (x.model_margin - x.spread_line)
            x["proj_total"] = x.total_line + x.w_total * (x.model_total - x.total_line)
            b = B.price_games(x, dists[s], "proj")
            b = b[(b.market == "Spread") & (b.ev > 0)]
            out["edges"].append(pd.DataFrame({"game_id": b.game_id, "season": s, "res": b.outcome, "profit": b.profit}))
    edges = pd.concat(out.pop("edges"), ignore_index=True) if out.get("edges") else pd.DataFrame(columns=["game_id", "season", "res", "profit"])
    res = {k: pd.DataFrame(v) for k, v in out.items()}
    res["edges"] = edges
    return res


# --------------------------------------------------------------------------
# Records and tables
# --------------------------------------------------------------------------

def stats(y: pd.DataFrame, res: str = "res", profit: str = "profit") -> dict:
    n = len(y)
    w, l, p = int((y[res] > 0).sum()), int((y[res] < 0).sum()), int((y[res] == 0).sum())
    roi = float(y[profit].mean()) if n else np.nan
    se = float(y[profit].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {"n": n, "w": w, "l": l, "p": p, "win": w / (w + l) if w + l else np.nan, "roi": roi, "se": se}


def fmt(s: dict, se: bool = True) -> str:
    if not s["n"]:
        return "-"
    if s["n"] < 2 or np.isnan(s["roi"]):
        return f"{s['w']}-{s['l']}"
    out = f"{s['w']}-{s['l']} {s['win']:.1%} {s['roi']:+.1%}"
    return out + (f" ± {s['se']:.1%}" if se else "")


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def label(x: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    return x.join(seg, on="game_id", how="inner")


BUCKETS = [  # (key, title, table, filter, result columns)
    ("open2", "Opener 2+", "open", lambda y: y.gap.abs() >= 2, ("res", "profit")),
    ("open3", "Opener 3+", "open", lambda y: y.gap.abs() >= 3, ("res", "profit")),
    ("total3", "Totals 3+ at the close", "total", lambda y: y.gap.abs() >= 3, ("res", "profit")),
    ("wind", "Wind unders 10-19 mph", "wind", None, ("res", "profit")),
    ("edges", "Model edges (adjusted spread, EV above 0)", "edges", None, ("res", "profit")),
]


def period_tables(period: str, sel, T: dict, seg_games: pd.DataFrame, close_frame: pd.DataFrame) -> list[str]:
    md = [f"### {period}", ""]
    cf = close_frame[sel(close_frame.season)]
    pick = lambda name: T[name][sel(T[name].season)]
    cs, os_, ct, wu, me = pick("close_sp"), pick("open"), pick("total"), pick("wind"), pick("edges")

    def rows(cols_fn, head):
        out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for title, col, order in SPLITS:
            for i, g in enumerate(order):
                x = cf[cf[col] == g]
                if not len(x):
                    continue
                out.append("| " + " | ".join([f"{title}: {g}" if i == 0 else g, str(len(x))] + cols_fn(lambda t: t[t[col] == g], x)) + " |")
        out.append("| " + " | ".join(f"**{v}**" for v in ["All games", str(len(cf))] + cols_fn(lambda t: t, cf)) + " |")
        return out

    def spreads(sub, x):
        c, o = sub(cs), sub(os_)
        return [f"{mae(x.model_margin, x.result):.2f} / {mae(x.spread_line, x.result):.2f}",
                fmt(stats(c[c.gap.abs() >= 2])), fmt(stats(c[c.gap.abs() >= 3])),
                fmt(stats(o[o.gap.abs() >= 2])), fmt(stats(o[o.gap.abs() >= 2], "res_close", "profit_close"), se=False),
                fmt(stats(o[o.gap.abs() >= 3])), fmt(stats(o[o.gap.abs() >= 3], "res_close", "profit_close"), se=False)]

    md += ["**Spreads** (W-L, win rate, return per bet ± s.e.)", ""]
    md += rows(spreads, ["Segment", "Games", "Miss: model / close", "vs close, 2+", "vs close, 3+", "vs opener, 2+",
                         "same sides at close", "vs opener, 3+", "same sides at close"])
    md += [""]

    def totals(sub, x):
        t = sub(ct)
        return [f"{mae(x.model_total, x.total):.2f} / {mae(x.total_line, x.total):.2f}",
                fmt(stats(t[t.gap.abs() >= 3])), fmt(stats(sub(wu))), fmt(stats(sub(me)))]

    md += ["**Totals, wind unders and model edges** (W-L, win rate, return per bet ± s.e.)", ""]
    md += rows(totals, ["Segment", "Games", "Total miss: model / close", "Totals 3+ at close", "Wind unders 10-19 mph", "Model edges"])
    md += [""]
    return md


def screen(T: dict, P: dict, seg_games: pd.DataFrame) -> tuple[list[dict], int]:
    """Every segment x bucket cell, with the rule applied."""
    out, tests = [], 0
    hold = lambda t: t[t.season >= C.HOLDOUT_FROM]
    tune = lambda t: t[(t.season >= C.FIRST_TEST_SEASON) & (t.season < C.HOLDOUT_FROM)]
    for key, title, table, flt, _ in BUCKETS:
        base = T[table] if flt is None else T[table][flt(T[table])]
        pb = P[table] if flt is None or P[table].empty else P[table][flt(P[table])]
        pb = label(pb, seg_games) if len(pb) else pb
        for split, col, order in SPLITS:
            for g in order:
                y = base[base[col] == g]
                h, t = stats(hold(y)), stats(tune(y))
                if not h["n"] or len(y) == len(base):     # no bets, or the segment is the whole bucket
                    continue
                tests += 1
                pz = pb[pb[col] == g] if len(pb) else pb
                p = stats(pz) if len(pz) else {"n": 0, "roi": np.nan}
                rest = hold(base[base[col] != g])
                diff = h["roi"] - float(rest.profit.mean()) if len(rest) else np.nan
                dse = np.sqrt(h["se"] ** 2 + float(rest.profit.std(ddof=1)) ** 2 / len(rest)) if len(rest) > 1 and h["n"] > 1 else np.nan
                passed = (h["n"] > 1 and t["n"] > 0 and h["roi"] > 0 and t["roi"] > 0 and h["roi"] >= 2 * h["se"]
                          and p["n"] > 0 and p["roi"] > 0)
                out.append({"bucket": title, "split": split, "segment": g, "hold": h, "tune": t, "pin": p,
                            "diff": diff, "diff_se": dse, "pass": passed, "hold_2se": h["n"] > 1 and h["roi"] >= 2 * h["se"]})
    return out, tests


# --------------------------------------------------------------------------

def reproduce(R: dict, T: dict) -> list[str]:
    s = backtest.summarize(R["frame"], R["graded"])
    o = T["open"]
    o2, o2h = o[o.gap.abs() >= 2], o[(o.gap.abs() >= 2) & (o.season >= C.HOLDOUT_FROM)]
    w_all = stats(wind_unders(R["frame"], first=0))
    ed = T["edges"][T["edges"].season >= C.HOLDOUT_FROM]
    a, so, sc, se = s["accuracy"], stats(o2), stats(o2h), stats(ed)
    cw = stats(o2, "res_close", "profit_close")
    return ["## The headline numbers, reproduced", "",
            "| | This run | Brief |", "|---|---|---|",
            f"| Games graded (2014 on) | {s['games']:,} | 3,403 |",
            f"| Average miss on the margin: model / closing line | {a['margin_mae']['model']:.2f} / {a['margin_mae']['market']:.2f} | 10.02 / 9.91 |",
            f"| Raw model 2+ off the opening spread | {so['n']} bets, {so['win']:.1%}, {so['roi']:+.1%} ± {so['se']:.1%} | 982, 54.5%, +4.0% ± 3.0% |",
            f"| ... of which since 2019 | {sc['n']}, {sc['roi']:+.1%} | 616, +4.0% |",
            f"| ... the same sides at the close | {cw['win']:.1%} | 51.8% |",
            f"| Wind unders 10-19 mph (all seasons, from 2011) | {w_all['w']}-{w_all['l']}-{w_all['p']} | 498-379-9 |",
            f"| Model edges (adjusted spreads, EV above 0) since 2019 | {se['n']:,} bets, {se['win']:.1%}, {se['roi']:+.1%} | 1,070, 50.9%, -1.3% |", ""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    R = build(a.rebuild)
    seg_games = segments(R["games"])
    raw = {"close_sp": close_spreads(R["frame"]), "open": opener_spreads(R["early"], R["openers"], R["games"]),
           "total": close_totals(R["frame"]), "wind": wind_unders(R["frame"]), "edges": model_edges(R["graded"])}
    T = {k: label(v, seg_games) for k, v in raw.items()}
    close_frame = label(R["frame"][R["frame"].played & R["frame"].spread_line.notna() & R["frame"].total_line.notna()
                                   & (R["frame"].season >= C.FIRST_TEST_SEASON)], seg_games)
    P = pinnacle(R)
    cells, tests = screen(T, P, seg_games)

    md = ["# NFL board: where the edge lives", "",
          "Walk-forward backtest split by segment (`python experiments/nfl_segments.py`; the segment definitions and the "
          "rule are in the script's docstring). Miss is the average absolute error of the raw model and of the closing "
          "line. Records are W-L, win rate and return per bet with one standard error, at the price the data records for "
          "each line (-110 where it has none). Opener records use the run that assumes last week's quarterbacks and no "
          "injury news and leave out games after a Monday night, as the board's opener report does; \"same sides at close\" "
          "bets those sides at the closing spread instead. Wind unders are outdoor games with a recorded 10-19 mph wind, "
          "under at the closing total. Model edges are the market-adjusted spread bets with EV above 0 at the close.", ""]
    md += reproduce(R, T)
    for period, sel in PERIODS.items():
        md += period_tables(period, sel, T, seg_games, close_frame)

    # Pinnacle by bucket and segment
    md += ["### Graded against Pinnacle, 2024-2025", "",
           "Opener buckets at Pinnacle's Monday open (line and price), the rest at Pinnacle's close; model edges re-priced "
           "from the projection blended off Pinnacle's closing spread.", ""]
    head = ["Segment"] + [t for _, t, _, _, _ in BUCKETS]
    md += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    PL = {k: (label(v, seg_games) if len(v) else v) for k, v in P.items()}
    for title, col, order in SPLITS + [("All games", None, [None])]:
        for i, g in enumerate(order):
            cells_ = []
            for key, _, table, flt, _ in BUCKETS:
                y = PL[table]
                if len(y) and flt is not None:
                    y = y[flt(y)]
                if len(y) and col is not None:
                    y = y[y[col] == g]
                cells_.append(fmt(stats(y)) if len(y) else "-")
            name = "**All games**" if col is None else (f"{title}: {g}" if i == 0 else g)
            md.append("| " + " | ".join([name] + cells_) + " |")
    md += [""]

    # the rule
    n_hold = sum(c["hold_2se"] for c in cells)
    passed = [c for c in cells if c["pass"]]
    md += ["## The rule", "",
           f"Tested {tests} segment x bucket cells (every cell with holdout bets, across the five betting buckets: opener 2+, "
           "opener 3+, totals 3+ at the close, wind unders, model edges). A lead needs a return above zero in both periods, "
           "a holdout return at least two standard errors above zero, and a return above zero against Pinnacle on 2024-2025. "
           f"With no edge anywhere about 0.6% of cells would pass ({tests * 0.0057:.1f} expected by chance here; "
           f"{tests * 0.0228:.1f} would clear the holdout bar alone). "
           f"{n_hold} cells clear the holdout bar; {len(passed)} pass the whole rule. "
           "Segments that hold every bet in a bucket (outdoors and 10-19 mph for the wind unders) are not tests and are left out. "
           "The Pinnacle seasons are the holdout's last two, so that check is a regrade of the same games at better lines, "
           "not a fresh sample.", ""]
    show = sorted([c for c in cells if c["hold_2se"]], key=lambda c: (not c["pass"], c["bucket"], c["split"]))
    if show:
        md += ["Cells that clear the holdout bar (\"vs rest\" is the holdout return minus the rest of the bucket's, ± s.e.; "
               "not part of the rule):", "",
               "| Bucket | Segment | Holdout | Tuning | Pinnacle 2024-25 | vs rest | Passes |", "|---|---|---|---|---|---|---|"]
        for c in show:
            md.append(f"| {c['bucket']} | {c['split']}: {c['segment']} | {fmt(c['hold'])} | {fmt(c['tune'])} | "
                      f"{fmt(c['pin']) if c['pin']['n'] else '-'} | {c['diff']:+.1%} ± {c['diff_se']:.1%} | {'yes' if c['pass'] else 'no'} |")
        md += [""]

    text = "\n".join(md)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
