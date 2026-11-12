"""Defensive backs the name match loses (NFL receiving projections). Nov 12, 2026.

    PFF_DATA_DIR=../pff-data python experiments/db_name_match_test.py

THE RULE (written before the first run; Project doc claude/plan-db-name-match-test-2026-11-12.md)
-------------------------------------------------------------------------------------------------
`injuries._season` matches a team's core players (named as the snap counts spell them) to the
injury report and the weekly roster by normalised full name. A player the files spell differently
counts as off the roster, so unavailable, every week. Fixed for the offensive line on Nov 11, 2026
(`injuries.ALIAS_GROUPS = ("ol",)`); this is the same question for the one other group a number on
the board reads: defensive backs (`opp_miss_db` in the calibration of receiving yards and
receptions, `players.CAL_EXTRA`). Linebackers are merged into the player table but no projection
uses them; the defensive line and running backs feed nothing; pass catchers feed the shadow test
(`shadow.MARGIN_PC`), whose inputs are fixed, and are left alone. The game model reads only the
offensive line, so no spread or total moves.

Now (B0): ALIAS_GROUPS = ("ol",). Candidate (C): ALIAS_GROUPS = ("ol", "db"). Both rebuilt for
every season from the same downloads, each fitted and projecting on its own input.

Measures: the board's projections (players.walk_forward, the boosted step, the Questionable step),
WR/TE/RB who played with 1+ projected reception, paired by player-game, 2014-18 and 2019 on:
  1-2  receiving-yards and receptions RMSE, game-day table
  3-4  the same, bet-time table (players.build(pregame=True))
  5    Brier at the closing DraftKings/FanDuel lines 2023-25, rec_yds + receptions pooled, game-day
  6    Brier at the pricing-snapshot lines 2023-25, the same two markets, bet-time table
KEEP (CONVENTIONS.md, Nov 11, 2026): C lower on 1-4 in both periods, each holdout gain at least
1 standard error, and both Briers lower. Provisional if kept. Anything else is not kept; because
B0 is a known error, a flat or mixed result goes to Ryan as a trade-off.
Reported, not in the rule: how often opp_miss_db differs by season; 2026 alone; passing yards and
completions (expected unchanged); how far this week's receiving projections would move.

Writes experiments/out/db_name_match_test.json. Changes nothing the board reads.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
from nflmodel import config as C, propbackfill as B, props as P  # noqa: E402

OUT = HERE / "out"
STASH = C.RAW / "inj_stash"                     # the nflverse files, kept so both versions read the same ones
CACHE = C.RAW / "db_name_match_tables.pkl"
VERSIONS = {"now": ("ol",), "cand": ("ol", "db")}
STATS = ("rec_yds", "receptions")


def build() -> dict:
    """{(version, 'gameday' | 'bettime'): board projections}, plus the two injury-group frames."""
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    from nflmodel import data, extra, game_roles, injuries, model, pff_qb, players, players_data, players_ml, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    last = int(games[games.played].season.max())

    real = injuries._download
    STASH.mkdir(parents=True, exist_ok=True)

    def kept(url, path):                        # _season deletes its downloads: serve them from a stash
        s = STASH / Path(path).name
        if not s.exists():
            real(url, s)
        shutil.copy(s, path)
    injuries._download = kept
    inj = {}
    for v, groups in VERSIONS.items():
        injuries.ALIAS_GROUPS = groups
        inj[v] = pd.concat([injuries._season(s, games) for s in range(injuries.FIRST, last + 1)], ignore_index=True).round(4)
    injuries.ALIAS_GROUPS = VERSIONS["now"]
    injuries._download = real
    same = [c for c in inj["now"].columns if c not in ("miss_db",)]
    assert inj["now"][same].equals(inj["cand"][same]), "the candidate moved a column other than miss_db"

    prior = pff_qb.load(qb, sorted(games.season.unique()))
    print("quarterback prior: PFF grades" if prior is not None else "quarterback prior: flat (NOT the production model)", flush=True)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj["now"])       # the game model reads only the offensive line: one frame
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = players.load_availability(refresh_current=False, verbose=False)
    rush = injuries.load_pass_rush(games, refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = players.player_rows(players.team_games(f0, pg), pg, avail, pinfo)
    frame, _ = model.walk_forward(game_roles.attach(base, game_roles.team_features(rows, R, avail)))
    out = {"inj": inj, "routes": bool(getattr(players_ml, "ROUTE_F", None))}
    for v in VERSIONS:
        for when, pre in (("gameday", False), ("bettime", True)):
            print(f"  projections: {v}, {when} ...", flush=True)
            table = players.build(frame, pg, pinfo, avail, inj[v], rush, games=games, overrides=None, pregame=pre)
            pall = players.walk_forward(table, first_test=C.FIRST_TEST_SEASON)
            ml = players_ml.questionable(players_ml.walk_forward(players_ml.add_features(pall, R), first=C.FIRST_TEST_SEASON),
                                         first=C.FIRST_TEST_SEASON, snaps=R.get("snaps"))
            out[(v, when)] = ml.merge(players.season_average(table), on=["game_id", "player_id"], how="left").reset_index(drop=True)
    C.RAW.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def rmse(base: pd.DataFrame, cand: pd.DataFrame, last: int) -> dict:
    assert (base[["game_id", "player_id"]].to_numpy() == cand[["game_id", "player_id"]].to_numpy()).all()
    out = {}
    live = base["played"].astype(bool) & (base["active"] == 1) & (base["season"] >= 2014)
    rec = live & base["group"].isin(["WR", "TE", "RB"]) & (base["receptions_proj"] >= 1)
    qbs = live & (base["group"] == "QB") & (base["pass_att_proj"] >= 10)
    for st, m in (("rec_yds", rec), ("receptions", rec), ("pass_yds", qbs), ("completions", qbs)):
        for per, pm in (("2014-18", base["season"] < 2019), ("2019 on", base["season"] >= 2019), (f"{last} alone", base["season"] == last)):
            k = m & pm & base[st].notna()
            e0, e1 = (base.loc[k, st + "_proj"] - base.loc[k, st]) ** 2, (cand.loc[k, st + "_proj"] - cand.loc[k, st]) ** 2
            d = (e1 - e0).to_numpy(float)
            r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
            out[f"{st} {per}"] = {"n": int(k.sum()), "now": round(r0, 4), "candidate": round(r1, 4), "diff": round(r1 - r0, 5),
                                  "se": round(float(d.std(ddof=1) / np.sqrt(len(d)) / (2 * r0)), 5),      # delta method
                                  "rows_changed": int((np.abs(d) > 0).sum())}
    return out


def brier(base: pd.DataFrame, cand: pd.DataFrame, kind: str) -> dict:
    import props_real_lines as RL
    from props_calibration import FAMILY
    saved = P.PropModel.load()
    got = {"now": [], "cand": []}
    for season in (2023, 2024, 2025):
        snaps = B.load(season, kind)
        if snaps.empty:
            continue
        for k, t in (("now", base), ("cand", cand)):
            mdl = P.PropModel(FAMILY).fit(t[(t.season < season) & (t.season >= 2013)])
            if saved is not None:
                mdl.iso = saved.iso
            x = RL.priced(t, mdl, snaps, season)
            got[k].append(x[x["stat"].isin(STATS)])
    key = ["game_id", "nname", "market", "line", "book"]
    b0 = pd.concat(got["now"], ignore_index=True)
    x = pd.concat(got["cand"], ignore_index=True).merge(
        b0[key + ["our_over", "our_under"]].rename(columns={"our_over": "b_over", "our_under": "b_under"}), on=key, how="inner")
    x = x[x["y"] != x["line"]]
    hit = (x["y"] > x["line"]).astype(float)
    ours, was, mkt = x["our_over"] / (x["our_over"] + x["our_under"]), x["b_over"] / (x["b_over"] + x["b_under"]), x["fair_over"]

    def one(mask):
        d = ((ours - hit) ** 2 - (was - hit) ** 2)[mask]
        cl = d.groupby(x.loc[mask, "game_id"]).sum()
        return {"n": int(mask.sum()), "now": round(float(((was - hit) ** 2)[mask].mean()), 5), "candidate": round(float(((ours - hit) ** 2)[mask].mean()), 5),
                "books": round(float(((mkt - hit) ** 2)[mask].mean()), 5), "diff": round(float(d.mean()), 6),
                "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / max(mask.sum(), 1)), 6)}
    return {"pooled": one(pd.Series(True, index=x.index)), **{st: one(x["stat"] == st) for st in STATS}}


def main() -> None:
    T = build()
    inj = T["inj"]
    last = int(T[("now", "gameday")].loc[T[("now", "gameday")]["played"].astype(bool), "season"].max())
    res = {"versions": {k: list(v) for k, v in VERSIONS.items()}, "pff_routes_on": T.get("routes")}
    m = inj["now"].merge(inj["cand"], on=["season", "game_id", "team"], suffixes=("", "_c"))
    res["input"] = {int(s): {"team_games": int(len(d)), "differ": round(float((~np.isclose(d.miss_db, d.miss_db_c)).mean()), 4),
                             "mean_missing_now": round(float(d.miss_db.mean()), 4), "mean_missing_candidate": round(float(d.miss_db_c.mean()), 4)}
                    for s, d in m.groupby("season")}
    res["rmse"] = {w: rmse(T[("now", w)], T[("cand", w)], last) for w in ("gameday", "bettime")}
    res["brier"] = {"closing lines, game-day": brier(T[("now", "gameday")], T[("cand", "gameday")], "close"),
                    "pricing snapshot, bet-time": brier(T[("now", "bettime")], T[("cand", "bettime")], "flag")}
    ok = True
    for w in ("gameday", "bettime"):
        for st in STATS:
            a, b = res["rmse"][w][f"{st} 2014-18"], res["rmse"][w][f"{st} 2019 on"]
            ok &= a["diff"] < 0 and b["diff"] < 0 and -b["diff"] >= b["se"]
    ok &= all(v["pooled"]["diff"] < 0 for v in res["brier"].values())
    res["kept"] = bool(ok)
    b, c = T[("now", "bettime")], T[("cand", "bettime")]
    up = (~b["played"].astype(bool)) & (b["season"] == b["season"].max()) & b["group"].isin(["WR", "TE", "RB"]) & (b["receptions_proj"] >= 1)
    if up.any():
        wk = b.loc[up, "week"].min()
        up &= b["week"] == wk
        d = (c.loc[up, "rec_yds_proj"] - b.loc[up, "rec_yds_proj"]).abs()
        res["this_week"] = {"week": int(wk), "players": int(up.sum()), "rec_yds_move_mean": round(float(d.mean()), 3), "rec_yds_move_max": round(float(d.max()), 3),
                            "moved_half_yard": int((d >= 0.5).sum())}
    OUT.mkdir(exist_ok=True)
    (OUT / "db_name_match_test.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
