"""Pre-set test 1 of Nov 26, 2026: count only new absences in the NFL game model's starters-out input.

    PFF_DATA_DIR=<pff-data clone> python experiments/new_absences_test.py

Rule written before running (Project doc claude/plan-injury-news-tests-2026-11-26.md). Lead: on the
Nov 26, 2026 board the model had Buffalo by 6.2 over Kansas City against a line of 2.5, 2.6 points of
it from d_depth1_out (two more Chiefs skill starters out than Bills). One had been out since Week 10;
the other was a fullback at 1% of the offensive snaps.

Today d_depth1_out (nflmodel/game_roles.py) is a plain count: depth-chart starters at QB, RB, WR and
TE who are out, home minus away, with "out" known by Friday (roster status the week before, plus Out
or Doubtful on the week's report). It does not ask whether the team has already been playing without
him, nor how much he plays.

Versions, each replacing d_depth1_out in the margin model (every count is home minus away and uses
the same Friday definition of out):
  A  new only        a starter counts only if he had an offensive snap in the team's previous game
                     of the season (in the team's first game of a season every starter out counts)
  B  beyond recent   this week's count minus the team's average count over its previous four games
                     of the season (fewer if fewer were played; zero in the first game)
  C  both apart      two inputs: starters newly out (A), and starters out who also missed the
                     previous game
  D  regulars only   a starter counts only if his average snap share is 40% or more (the level
                     game_roles.py already calls a regular). Added Nov 26 after the step 0 fact
                     check and before any scoring run: 15.5% of the starters counted out in 2014-25
                     were under that level (98% of fullbacks, 30% of running backs).
Measures (6), against the current model on the same games (played, with a closing spread):
  1-2  raw model margin miss, tuning 2014-18 and holdout 2019-25
  3-4  board (market-adjusted) margin miss, proj_margin, tuning and holdout
  5-6  Brier of the home cover chance at the closing spread, tuning and holdout
Details fixed here before the first run (they fill gaps in the plan's wording):
  - The inputs under test use Friday information in every version. The rest of the frame is the
    production frame the model is fitted on.
  - Cover chance: normal around proj_margin, with the standard deviation of (result - proj_margin)
    over that version's own walk-forward games of the previous ten seasons. Pushes are left out.
  - Standard errors are paired by game.
Rule: a version is chosen on tuning alone (better than the current model on all three tuning
  measures, then the lowest tuning raw margin miss). Kept only if better on all six and each holdout
  gain >= 1 paired SE. Mixed, or under 1 SE: a trade-off for Ryan. No version better on all three
  tuning measures: nothing is carried to the holdout and the holdout is not read.
2026 is reported for the chosen version only, never judged (the lead was found in it).
Props check, only if a version passes: the player walk-forward with the candidate's margin.

Writes experiments/out/new_absences_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

from nflmodel import game_roles, model, players as PL, roles  # noqa: E402

OUT = HERE / "out" / "new_absences_test.json"
TUNE, HOLD = (2014, 2018), (2019, 2025)
REGULAR = 0.4
RECENT = 4
COLS = ["n_all", "n_new", "n_cont", "n_beyond", "n_reg", "n_news", "snap_drop"]


def frames():
    """The production frame without the role inputs, plus what the counts are built from."""
    from nflmodel import data, extra, injuries as IJ, pff_qb, players_data, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = IJ.load_injuries(games, refresh_current=False, verbose=False)
    prior = pff_qb.load(qb, sorted(games.season.unique()))
    print("quarterback prior:", "PFF grades" if prior is not None else "flat (NOT the production model)", flush=True)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = PL.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = PL.player_rows(PL.team_games(f0, pg), pg, avail, pinfo)
    return base, rows, R, avail, data.load_openers(), prior is not None


def roster_active(table: pd.DataFrame, avail: pd.DataFrame) -> pd.Series:
    """1 if his roster status the week before has him available (no injury report): what the opener knew."""
    rows = table[["season", "week", "player_id"]].copy()
    rows["t"] = PL._t(rows["season"], rows["week"])
    a = avail[["season", "week", "player_id", "status"]].copy()
    a["t"] = PL._t(a["season"], a["week"])
    a = a.drop_duplicates(["player_id", "t"], keep="last")[["player_id", "t", "status"]]
    rows = rows.merge(a, on=["player_id", "t"], how="left")
    rows.index = table.index
    pre = PL.pregame_active(rows, avail).to_numpy() == 1
    if "played" in table and "active" in table:     # upcoming games: who the board already has out
        pre &= ~((~table["played"].astype(bool)) & (table["active"] == 0)).to_numpy()
    return pd.Series(pre.astype(int), index=table.index)


def team_counts(rows: pd.DataFrame, R: dict, avail: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per team-game: the current count and each version's."""
    f = roles.features(rows, R)[["game_id", "player_id", "depth", "snap_avg", "snap_trend"]]
    x = rows[["game_id", "team", "season", "t", "player_id", "position"]].copy()
    x["fri"] = game_roles.friday_active(rows, R, avail).to_numpy()
    x["pre"] = roster_active(rows, avail).to_numpy()
    x = x.merge(f, on=["game_id", "player_id"], how="left")
    # the team's previous game of the season, and whether he had an offensive snap in it
    tg = x.drop_duplicates(["game_id", "team"])[["game_id", "team", "season", "t"]].sort_values(["team", "season", "t"])
    tg["prev_game"] = tg.groupby(["team", "season"])["game_id"].shift()
    x = x.merge(tg[["game_id", "team", "prev_game"]], on=["game_id", "team"], how="left")
    sn = R["snaps"][R["snaps"]["offense_snaps"].fillna(0) > 0][["game_id", "player_id"]].assign(prev_snap=1)
    x = x.merge(sn.rename(columns={"game_id": "prev_game"}), on=["prev_game", "player_id"], how="left")
    first = x["prev_game"].isna()
    out = x["fri"] == 0
    d1 = x["depth"] == 1
    new = first | (x["prev_snap"] == 1)
    regular = (x["snap_avg"].fillna(0) >= REGULAR) & ~out
    key = [x["game_id"], x["team"]]
    t = pd.DataFrame({
        "n_all": (d1 & out).groupby(key).sum(),
        "n_new": (d1 & out & new).groupby(key).sum(),
        "n_reg": (d1 & out & (x["snap_avg"].fillna(0) >= REGULAR)).groupby(key).sum(),
        "n_news": (d1 & out & (x["pre"] == 1)).groupby(key).sum(),
        "snap_drop": (-(x["snap_trend"].fillna(0)).clip(upper=0)).where(regular, 0).groupby(key).sum(),
    }).reset_index()
    t["n_cont"] = t["n_all"] - t["n_new"]
    t = t.merge(tg[["game_id", "team", "season", "t"]], on=["game_id", "team"], how="left").sort_values(["team", "season", "t"])
    rec = t.groupby(["team", "season"])["n_all"].transform(lambda s: s.shift().rolling(RECENT, min_periods=1).mean())
    t["n_beyond"] = t["n_all"] - rec.fillna(0.0)
    k = d1 & out & x["season"].between(TUNE[0], HOLD[1])
    facts = {"starters_counted_out_2014_25": int(k.sum()),
             "share_who_also_missed_the_previous_game": round(float((~new)[k].mean()), 3),
             "share_under_40pct_snap_share": round(float((x["snap_avg"].fillna(0) < REGULAR)[k].mean()), 3),
             "share_out_on_this_weeks_report_only": round(float((x["pre"] == 1)[k].mean()), 3)}
    return t, facts


def attach(frame: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    f = frame.drop(columns=[c for c in frame if c.startswith(("d_n_", "h_n_", "a_n_")) or c in game_roles.FEATS])
    for side, team in (("h", "home_team"), ("a", "away_team")):
        s = t[["game_id", "team"] + COLS].rename(columns={"team": team} | {c: f"{side}_{c}" for c in COLS})
        f = f.merge(s, on=["game_id", team], how="left")
    for c in COLS:
        f["d_" + c] = f["h_" + c].fillna(0) - f["a_" + c].fillna(0)
    f["d_depth1_out"], f["d_snap_drop"] = f["d_n_all"], f["d_snap_drop"]
    return f.drop(columns=[f"{s}_{c}" for s in "ha" for c in COLS])


def phi(z):
    return 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))


def add_scores(o: pd.DataFrame) -> pd.DataFrame:
    """Per-game misses and the cover-chance Brier for one version's walk-forward output."""
    o = o[o.played & o.spread_line.notna() & o.model_margin.notna()].copy()
    o["raw"] = (o.model_margin - o.result).abs()
    o["board"] = (o.proj_margin - o.result).abs()
    o["brier"] = np.nan
    resid = o.result - o.proj_margin
    for s in sorted(o.season.unique()):
        past = resid[(o.season < s) & (o.season >= s - 10)]
        if len(past) < 200:
            continue
        k = o.season == s
        p = 1 - phi((o.loc[k, "spread_line"] - o.loc[k, "proj_margin"]) / past.std())
        cov = np.where(o.loc[k, "result"] == o.loc[k, "spread_line"], np.nan, (o.loc[k, "result"] > o.loc[k, "spread_line"]).astype(float))
        o.loc[k, "brier"] = (p - cov) ** 2
    return o.set_index("game_id")


def cmp(b: pd.DataFrame, o: pd.DataFrame, per, only=None) -> dict:
    k = b.season.between(*per)
    if only is not None:
        k &= b.index.isin(only)
    r = {}
    for col in ("raw", "board", "brier"):
        a_, c_ = b.loc[k, col].to_numpy(float), o.loc[b.index[k], col].to_numpy(float)
        ok = ~(np.isnan(a_) | np.isnan(c_))
        dd = c_[ok] - a_[ok]
        r[col] = {"n": int(ok.sum()), "now": round(float(a_[ok].mean()), 5), "candidate": round(float(c_[ok].mean()), 5),
                  "diff": round(float(dd.mean()), 6), "se": round(float(dd.std(ddof=1) / np.sqrt(len(dd))), 6)}
    return r


def judge(res: dict, base: pd.DataFrame, outs: dict, only=None) -> dict:
    """Apply the rule: tuning picks the version; the holdout is read only for the pick."""
    res["tuning"] = {v: cmp(base, o, TUNE, only) for v, o in outs.items()}
    print(json.dumps(res["tuning"]), flush=True)
    ok = [v for v in outs if all(m["diff"] < 0 for m in res["tuning"][v].values())]
    res["better_on_all_tuning_measures"] = ok
    if not ok:
        res["verdict"] = "No version better on all three tuning measures: nothing carried to the holdout, the holdout not read, nothing kept."
        return res
    pick = min(ok, key=lambda v: res["tuning"][v]["raw"]["candidate"])
    res["chosen"] = pick
    res["holdout"] = cmp(base, outs[pick], HOLD, only)
    res["reported_2026_not_judged"] = cmp(base, outs[pick], (2026, 2026), only)
    ms = list(res["tuning"][pick].values()) + list(res["holdout"].values())
    better = [m["diff"] < 0 for m in ms]
    se_ok = all(-m["diff"] >= m["se"] for m in res["holdout"].values())
    res["better_on"] = f"{sum(better)} of {len(better)}"
    res["passes_before_props_check"] = bool(all(better) and se_ok)
    res["verdict"] = ("Passes the six measures: the props check decides." if res["passes_before_props_check"] else
                      "Better on every measure but a holdout gain is under 1 SE: a trade-off for Ryan." if all(better)
                      else "Mixed: a trade-off for Ryan, not kept by the rule.")
    return res


def main():
    base, rows, R, avail, _, pff = frames()
    t, facts = team_counts(rows, R, avail)
    chk = game_roles.team_features(rows, R, avail).merge(t, on=["game_id", "team"])
    assert (chk["depth1_out"] == chk["n_all"]).all() and np.allclose(chk["snap_drop_x"], chk["snap_drop_y"]), "counts differ from game_roles.py"
    f = attach(base, t)
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    swap = lambda new: [c for c in MF if c != "d_depth1_out"] + new  # noqa: E731
    versions = {"A (new only)": swap(["d_n_new"]), "B (beyond recent)": swap(["d_n_beyond"]),
                "C (new and continuing apart)": swap(["d_n_new", "d_n_cont"]), "D (regulars only)": swap(["d_n_reg"])}
    hf = f[f.season.between(TUNE[0], HOLD[1]) & f.played]
    res = {"quarterback_prior_pff": pff, "inputs": facts | {
        "games_2014_25": int(len(hf)),
        "games_where_the_input_changes": {v: int((hf[m[-1] if len(m) == len(MF) else "d_n_new"] != hf["d_depth1_out"]).sum()) for v, m in versions.items()},
        "spread": {c: round(float(hf[c].std()), 3) for c in ("d_depth1_out", "d_n_new", "d_n_cont", "d_n_beyond", "d_n_reg")}}}
    print(json.dumps(res), flush=True)

    def run(mf):
        o, info = model.walk_forward(f, margin_feats=mf, total_feats=TF)
        return add_scores(o), info

    b, _ = run(MF)
    outs, infos = {}, {}
    for v, mf in versions.items():
        outs[v], infos[v] = run(mf)
    res = judge(res, b, outs)
    if "chosen" in res:
        v = res["chosen"]
        res["coefficients_2025_fit"] = {c: infos[v][2025]["margin_coefs"].get(c) for c in versions[v] if c.startswith("d_n_")}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
