"""The college prop numbers against real DraftKings and FanDuel lines, 2025.

    python experiments/cfb_props_real_lines.py

Lines: data/props/history_cfb/2025/ (nflmodel/cfbpropbackfill.py), one snapshot per game ten
minutes before kickoff: 451 games with props, weeks 3-16 (week 3 partly). Projections: the
college board's walk-forward player table for 2025 (cfbmodel.players.walk_forward with
first_test=2025, game-day roles; no injury designations, which the board only has for the
current week). Chances at each line: the NFL prop distributions (nflmodel.props.PropModel,
fitted on NFL seasons), exactly what the Parlays page uses for college legs today
(parlays.cfb_prop_legs). Results: the box scores in the same player table. Look-ahead left in:
the game model's market blend uses the closing spread and total; wind is the game's actual wind.

Rules, written before running (Nov 1, 2026):

1. Main lines (two-sided passing, rushing and receiving yards, passing TDs), pushes left out.
   Brier score of three chances on the same props, compared with the books' (average no-vig
   chance across the two books, power method), standard errors clustered by game:
     model     the board's chance alone (today's college legs)
     nudged    the books' chance moved 20% of the way to the model, at most 3 points, legs
               more than 25 points apart left out (the NFL legs' rule: parlays._pull, CONFLICT)
   - The model may price college props alone only if it is no worse than the books.
   - College props get the NFL nudge only if nudged beats the books by 2 standard errors.
   - Otherwise they are priced off the books alone (no model nudge).
2. Ladder rungs (over-only alternate yardage lines). The NFL reads a rung's price through
   RUNG_CAL (how 2025 NFL rungs at each price cleared). Calibration bands by that reading:
   the NFL reading holds for college if, in every band with 100+ rungs, the share that cleared
   is within 2 standard errors (clustered by game) of the band's average reading. Otherwise
   college ladders need their own reading, fitted on these 2025 rungs and checked on 2026's
   before they're trusted. The model's own chance at each rung is reported the same way.
3. Anytime TD (over-only "yes" prices, read with the NFL's assumed margin): Brier of the books,
   the model and the nudged chance (NFL TD weight), same decision rule as 1.

The verdict on the "not market-tested" label: it comes off only through the books' prices
(live college pulls); this test decides whether the model also gets a say (rule 1) and whether
the ladders can be read the NFL's way (rule 2).

Writes experiments/out/cfb_props_real_lines.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import market as M, odds as O, parlays as PA, propdist as D, propmarket as PM, props as P  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "props" / "history_cfb" / "2025"
OUT = Path(__file__).resolve().parent / "out" / "cfb_props_real_lines.json"
BANDS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def projections(cache=Path("/tmp/cfb_pproj_2025.pkl")):
    """The college walk-forward player table for 2025 (cached: it takes about a minute)."""
    from cfbmodel import data, model, players, players_data
    from cfbmodel.ratings import CFBRatingEngine
    if cache.exists():
        return pd.read_pickle(cache), pd.read_pickle(cache.with_name("cfb_pinfo.pkl")), pd.read_pickle(cache.with_name("cfb_games.pkl"))
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, engine))
    pg, pinfo = players_data.load_player_games(verbose=False)
    pproj = players.walk_forward(players.build(games, frame, pg, pinfo, None, None), first_test=2025)
    return pproj, pinfo, games


def snapshots() -> pd.DataFrame:
    parts = []
    for f in sorted(HIST.glob("*.csv")):
        if f.name == "events.csv":
            continue
        x = pd.read_csv(f)
        if len(x):
            parts.append(x)
    s = pd.concat(parts, ignore_index=True)
    s["side"] = s["side"].str.lower().replace({"yes": "over", "no": "under"})
    s.loc[s["market"] == "player_anytime_td", "line"] = 0.5
    return s


def match_events(s: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """event_id -> game_id, by the two teams (either way round) and kickoff within two days."""
    from cfbmodel import teams as T
    t = T.load_teams(2025, refresh=False)
    look = {M._norm(n): sch for sch, n in t["name"].items()}
    for k, v in M.CFB_ALIASES.items():
        nk, nv = M._norm(k), M._norm(v)
        if nv in look and nk not in look:
            look[nk] = look[nv]
        elif nk in look and nv not in look:
            look[nv] = look[nk]
    ev = s.drop_duplicates("event_id")[["event_id", "home_name", "away_name", "commence"]].copy()
    ev["h"], ev["a"] = ev["home_name"].map(lambda x: look.get(M._norm(x))), ev["away_name"].map(lambda x: look.get(M._norm(x)))
    ev["ts"] = pd.to_datetime(ev["commence"], utc=True)
    g = games[games.season == 2025][["game_id", "home_team", "away_team", "kickoff"]].copy()
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True)
    out = {}
    for e in ev.itertuples():
        c = g[((g.home_team == e.h) & (g.away_team == e.a)) | ((g.home_team == e.a) & (g.away_team == e.h))]
        c = c[(c.kickoff - e.ts).abs() <= pd.Timedelta(days=2)]
        if len(c):
            out[e.event_id] = c.iloc[0]["game_id"]
    return pd.Series(out, name="game_id")


def player_frame(pproj, pinfo) -> pd.DataFrame:
    rows = pproj[(pproj.season == 2025) & pproj["played"].astype(bool)].copy()
    rows["name"] = rows["player_id"].map(pinfo["name"])
    rows["nname"] = rows["name"].map(PM.norm_name)
    rows = rows[~rows.duplicated(["game_id", "nname"], keep=False)]
    rows["tgt_share"] = rows.get("rec_share", 0.0)          # as parlays.cfb_prop_legs does
    return rows


def model_chances(d: pd.DataFrame, model) -> pd.DataFrame:
    out = []
    for (stat, grp), g in d.groupby(["stat", "group"]):
        if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
            continue
        g = g[P.eligible(g, stat, grp)].reset_index(drop=True)
        if g.empty:
            continue
        over, push, under = model.probs(g, stat, grp, g["line"].to_numpy(float))
        g["our_over"], g["our_push"], g["our_under"] = over, push, under
        g["y"] = P.actual(g, stat)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def clustered_diff(a, b, hit, game) -> tuple[float, float]:
    """Mean of (a-hit)^2 - (b-hit)^2 and its standard error, clustered by game."""
    d = (a - hit) ** 2 - (b - hit) ** 2
    cl = pd.Series(d).groupby(np.asarray(game)).sum()
    return float(d.mean()), float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(d))


def compare(x: pd.DataFrame, weight: float, label: str) -> dict:
    hit = (x["y"] > x["line"]).astype(float).to_numpy()
    ours = (x["our_over"] / (x["our_over"] + x["our_under"])).to_numpy()
    mkt = x["fair"].to_numpy()
    gap = ours - mkt
    keep = np.abs(gap) <= PA.CONFLICT
    nudged = mkt + np.clip(weight * gap, -PA.PULL_MAX, PA.PULL_MAX)
    m_vs, m_se = clustered_diff(ours, mkt, hit, x["game_id"])
    n_vs, n_se = clustered_diff(nudged[keep], mkt[keep], hit[keep], x["game_id"].to_numpy()[keep])
    return {"what": label, "props": int(len(x)), "games": int(x["game_id"].nunique()), "over_rate": round(float(hit.mean()), 4),
            "brier_books": round(D.brier(mkt, hit), 5), "brier_model": round(D.brier(ours, hit), 5),
            "model_minus_books": round(m_vs, 5), "model_se": round(m_se, 5),
            "nudged_kept": int(keep.sum()), "nudged_minus_books": round(n_vs, 5), "nudged_se": round(n_se, 5),
            "model_alone_ok": bool(m_vs <= 0), "nudge_earned": bool(n_vs <= -2 * n_se)}


def bands(p, hit, game) -> list[dict]:
    rows = []
    df = pd.DataFrame({"p": p, "hit": hit, "g": game})
    df["band"] = pd.cut(df["p"], BANDS, include_lowest=True)
    for b, x in df.groupby("band", observed=True):
        if len(x) < 30:
            continue
        cl = x.groupby("g")["hit"].agg(["sum", "size"])
        rate = x["hit"].mean()
        se = float(np.sqrt(((cl["sum"] - rate * cl["size"]) ** 2).sum()) / len(x))
        rows.append({"band": str(b), "n": int(len(x)), "predicted": round(float(x["p"].mean()), 4), "cleared": round(float(rate), 4),
                     "se": round(se, 4), "within_2se": bool(abs(rate - x["p"].mean()) <= 2 * se) if len(x) >= 100 else None})
    return rows


def main():
    pproj, pinfo, games = projections()
    s = snapshots()
    gid = match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    rows = player_frame(pproj, pinfo)
    model = P.PropModel.load()
    res = {"events_with_props": int(s["event_id"].nunique()), "events_matched": int(gid.size),
           "player_rows_2025": int(len(rows))}

    # 1. main lines, two-sided
    main = s[~s["alt"] & (s["market"] != "player_anytime_td")]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index()
    w = w.dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    x = model_chances(w.merge(rows, on=["game_id", "nname"], how="inner"), model)
    x = x[x["y"] != x["line"]]
    res["main"] = {"all": compare(x, PA.W_MAIN if hasattr(PA, "W_MAIN") else 0.2, "main lines")}
    for st, g in x.groupby("stat"):
        res["main"][st] = compare(g, PA.W_MAIN if hasattr(PA, "W_MAIN") else 0.2, st)

    # 2. ladder rungs, over-only
    a = s[s["alt"] & (s["side"] == "over")].copy()
    a = a.groupby(["game_id", "nname", "stat", "line"], as_index=False)["price"].max()       # the better of the two books
    a["implied"] = O.implied_prob(a["price"])
    a["nfl_read"] = np.interp(a["implied"], *zip(*PA.RUNG_CAL))
    r = model_chances(a.merge(rows, on=["game_id", "nname"], how="inner"), model)
    r = r[r["y"] != r["line"]]
    hit = (r["y"] > r["line"]).astype(float).to_numpy()
    ours = (r["our_over"] / (r["our_over"] + r["our_under"])).to_numpy()
    bb = bands(r["nfl_read"].to_numpy(), hit, r["game_id"].to_numpy())
    res["ladders"] = {"rungs": int(len(r)), "games": int(r["game_id"].nunique()),
                      "nfl_reading": bb, "nfl_reading_holds": all(b["within_2se"] for b in bb if b["within_2se"] is not None),
                      "raw_implied": bands(r["implied"].to_numpy(), hit, r["game_id"].to_numpy()),
                      "model": bands(ours, hit, r["game_id"].to_numpy()),
                      "brier_nfl_read": round(D.brier(r["nfl_read"].to_numpy(), hit), 5), "brier_model": round(D.brier(ours, hit), 5)}
    m_vs, m_se = clustered_diff(ours, r["nfl_read"].to_numpy(), hit, r["game_id"])
    res["ladders"]["model_minus_nfl_read"], res["ladders"]["model_se"] = round(m_vs, 5), round(m_se, 5)

    # 3. anytime TD, over-only
    t = s[(s["market"] == "player_anytime_td") & (s["side"] == "over")].copy()
    t = t.groupby(["game_id", "nname", "stat", "line"], as_index=False)["price"].mean()
    t["fair"] = O.one_sided_fair(t["price"], PA.RUNG_MARGIN, "power")
    td = model_chances(t.merge(rows, on=["game_id", "nname"], how="inner"), model)
    res["anytime_td"] = compare(td, getattr(PA, "W_TD", 0.2), "anytime TD")
    # Added after the first run: the books' TD chance depends on the margin assumed (the NFL's
    # 10%). Give the books their best reading on these same props (in sample, generous to them)
    # and compare again; then correct both the same way, which leaves only how well each ranks.
    from sklearn.isotonic import IsotonicRegression
    hit = (td["y"] > 0.5).astype(float).to_numpy()
    ours = (td["our_over"] / (td["our_over"] + td["our_under"])).to_numpy()
    imp = np.asarray(O.implied_prob(td["price"]))
    best = min(((m, D.brier(np.asarray(O.one_sided_fair(td["price"], m, "power")), hit)) for m in np.arange(0.05, 0.61, 0.025)), key=lambda z: z[1])
    fb = IsotonicRegression(out_of_bounds="clip").fit(imp, hit).predict(imp)
    fm = IsotonicRegression(out_of_bounds="clip").fit(ours, hit).predict(ours)
    d1, s1 = clustered_diff(ours, fb, hit, td["game_id"])
    d2, s2 = clustered_diff(fm, fb, hit, td["game_id"])
    res["anytime_td"].update({"books_best_margin": round(float(best[0]), 3), "model_minus_books_recalibrated": round(d1, 5),
                              "recalibrated_se": round(s1, 5), "ranking_only_model_minus_books": round(d2, 5), "ranking_se": round(s2, 5)})
    # college ladder reading, fitted on these 2025 rungs (to be checked on 2026's before use)
    ri = np.asarray(r["implied"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(ri, hit_r := (r["y"] > r["line"]).astype(float).to_numpy())
    grid = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    res["ladders"]["cfb_reading_2025"] = [(float(g), round(float(v), 4)) for g, v in zip(grid, iso.predict(grid))]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
