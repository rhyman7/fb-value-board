"""Do head-coaching changes add anything to the game models?

    python experiments/coach_change_test.py

ELWAY (Silver Bulletin) adjusts for coaching changes and SP+ adjusts its preseason numbers for
them; the board does not. Coaches: NFL from nflverse's home_coach / away_coach; college from
Wikipedia's FBS coaching-change tables, 2014-2025 seasons (experiments/data/cfb_coach_changes.csv:
every firing, interim and hire with its date).

Per team and game:
  new    a different head coach from the start of last season (NFL: first game's coach this
         season vs last; college: any change dated from Aug 15 last year to Aug 31 this year)
  mid    the coach was changed during this season before this game (fired, interim or new)
Features (home minus away for the margin, home plus away for the total):
  d_new / s_new          new, all season
  d_new_early            new, weeks 1-4 only
  d_new_reg              new x last season's average point margin (negated): does a new coach's
                         team drift back to the middle more than the ratings already allow?
  d_mid / s_mid          mid-season change

Rule, set before running (Nov 2, 2026): a feature is kept only if adding it lowers the raw
model's average miss in both the tuning seasons and the holdout, the holdout by at least 2
paired standard errors.

Writes experiments/out/coach_change_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
OUT = HERE / "out" / "coach_change_test.json"
CFB_CHANGES = HERE / "data" / "cfb_coach_changes.csv"
ALIAS = {"FIU": "Florida International", "Hawaii": "Hawai'i", "Massachusetts": "Massachusetts",
         "Louisiana Monroe": "UL Monroe", "San Jose State": "San José State",
         "Appalachian State": "App State"}


def prev_margin(g: pd.DataFrame) -> dict:
    """(team, season) -> that team's average point margin the season before."""
    long = pd.concat([g.assign(team=g.home, m=g.hs - g.as_), g.assign(team=g.away, m=g.as_ - g.hs)])
    avg = long[long.played].groupby(["team", "season"])["m"].mean()
    return {(t, s + 1): v for (t, s), v in avg.items()}


def long_form(g: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([g[["game_id", "season", "week", "date"]].assign(side="home", team=g.home),
                      g[["game_id", "season", "week", "date"]].assign(side="away", team=g.away)], ignore_index=True)


def nfl_flags(games: pd.DataFrame) -> pd.DataFrame:
    g = games.rename(columns={"home_team": "home", "away_team": "away"})
    g["date"] = pd.to_datetime(g["gameday"])
    L = long_form(g)
    L["coach"] = np.where(L.side == "home", L.game_id.map(g.set_index("game_id").home_coach),
                          L.game_id.map(g.set_index("game_id").away_coach))
    L = L.sort_values("date")
    first = L.groupby(["team", "season"])["coach"].first()
    L["start_coach"] = [first.get((t, s)) for t, s in zip(L.team, L.season)]
    L["prev_start"] = [first.get((t, s - 1)) for t, s in zip(L.team, L.season)]
    L["new"] = (L.prev_start.notna() & (L.start_coach != L.prev_start)).astype(float)
    L["mid"] = (L.coach != L.start_coach).astype(float)
    return L


def cfb_flags(games: pd.DataFrame) -> pd.DataFrame:
    g = games.rename(columns={"home_team": "home", "away_team": "away"})
    g["date"] = pd.to_datetime(g["gameday"])
    ch = pd.read_csv(CFB_CHANGES, parse_dates=["date"])
    ch["team"] = ch.team.replace(ALIAS)
    teams = set(g.home) | set(g.away)
    missing = sorted(set(ch.team) - teams)
    if missing:
        print("  coach-change teams not matched:", missing)
    L = long_form(g)
    by = {t: d.date.to_numpy() for t, d in ch.groupby("team")}
    new, mid = [], []
    for t, s, dt in zip(L.team, L.season, L.date.to_numpy()):
        d = by.get(t)
        if d is None:
            new.append(0.0), mid.append(0.0)
            continue
        lo, hi = np.datetime64(f"{s - 1}-08-15"), np.datetime64(f"{s}-08-31")
        new.append(float(((d >= lo) & (d <= hi)).any()))
        mid.append(float(((d > hi) & (d < dt)).any()))
    L["new"], L["mid"] = new, mid
    return L


def features(games: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    g = games.rename(columns={"home_team": "home", "away_team": "away", "home_score": "hs", "away_score": "as_"})
    pm = prev_margin(g)
    flags = flags.assign(reg=[-pm.get((t, s), 0.0) for t, s in zip(flags.team, flags.season)])
    flags["reg"] = flags.reg * flags.new
    flags["early"] = flags.new * (flags.week <= 4)
    w = flags.pivot_table(index="game_id", columns="side", values=["new", "mid", "reg", "early"])
    f = pd.DataFrame(index=w.index)
    f["d_new"], f["s_new"] = w["new"]["home"] - w["new"]["away"], w["new"]["home"] + w["new"]["away"]
    f["d_new_early"] = w["early"]["home"] - w["early"]["away"]
    f["d_new_reg"] = w["reg"]["home"] - w["reg"]["away"]
    f["d_mid"], f["s_mid"] = w["mid"]["home"] - w["mid"]["away"], w["mid"]["home"] + w["mid"]["away"]
    return f.reset_index()


def paired(base, test, pred, actual, tune, hold):
    j = base[["game_id", "season", pred, actual]].merge(test[["game_id", pred]], on="game_id", suffixes=("", "_t")).dropna()
    j["d"] = (j[pred + "_t"] - j[actual]).abs() - (j[pred] - j[actual]).abs()
    res = {}
    for p, x in (("tune", j[j.season.isin(tune)]), ("hold", j[j.season >= hold])):
        res[p] = {"n": int(len(x)), "base_miss": round(float((x[pred] - x[actual]).abs().mean()), 4),
                  "change": round(float(x.d.mean()), 4), "se": round(float(x.d.std(ddof=1) / np.sqrt(len(x))), 4)}
    res["keep"] = bool(res["tune"]["change"] < 0 and res["hold"]["change"] <= -2 * res["hold"]["se"])
    return res


def run(league: str) -> dict:
    if league == "nfl":
        from nflmodel import config as C, data, extra, model
        from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
        games, tg, qb = data.load_all(refresh=False, verbose=False)
        te, qe = extra.load_extra(verbose=False)
        frame = model.build_frame(games, ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games))
        flags = nfl_flags(games)
        hold = C.HOLDOUT_FROM
        tune = list(range(int(frame.season.min()), hold))
    else:
        from cfbmodel import config as C, data, model
        from cfbmodel.ratings import CFBRatingEngine
        games, tg, qb = data.load_all(refresh=False, verbose=False)
        games = games[games.season <= 2025]
        frame = model.build_frame(games, CFBRatingEngine(tg, qb, games=games))
        flags = cfb_flags(games)
        hold, tune = C.HOLDOUT_FROM, [2017, 2018, 2019]
    f = features(games, flags)
    frame = frame.merge(f, on="game_id", how="left")
    cols = [c for c in f.columns if c != "game_id"]
    frame[cols] = frame[cols].fillna(0.0)
    frame["act_margin"] = frame["home_score"] - frame["away_score"]
    frame["act_total"] = frame["home_score"] + frame["away_score"]
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)

    def out(mf, tf):
        o, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
        return o[o.played & o.model_margin.notna()]

    played = frame[frame.season <= 2025]
    res = {"holdout_from": hold, "tuning_seasons": [min(tune), max(tune)],
           "team_games_new_coach_share": round(float(flags.new.mean()), 3),
           "team_games_after_midseason_change_share": round(float(flags.mid.mean()), 3),
           "games": int(len(played))}
    base = out(MF, TF)
    for name, mf, tf, pred, act in (
            ("margin + d_new", MF + ["d_new"], TF, "model_margin", "act_margin"),
            ("margin + d_new_early", MF + ["d_new_early"], TF, "model_margin", "act_margin"),
            ("margin + d_new_reg", MF + ["d_new_reg"], TF, "model_margin", "act_margin"),
            ("margin + d_mid", MF + ["d_mid"], TF, "model_margin", "act_margin"),
            ("margin + all", MF + ["d_new", "d_new_early", "d_new_reg", "d_mid"], TF, "model_margin", "act_margin"),
            ("total + s_new", MF, TF + ["s_new"], "model_total", "act_total"),
            ("total + s_mid", MF, TF + ["s_mid"], "model_total", "act_total")):
        res[name] = paired(base, out(mf, tf), pred, act, tune, hold)
        print(league, name, res[name], flush=True)
    return res


def main():
    res = {lg: run(lg) for lg in ("nfl", "cfb")}
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
