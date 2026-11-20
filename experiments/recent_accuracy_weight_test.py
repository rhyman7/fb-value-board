"""A game-by-game market weight from recent accuracy (nfelo's idea), college and NFL (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/recent_accuracy_weight_test.py

T3 of the Project doc claude/plan-batch-tests-2026-11-20.md (written before the run). Committed
before its first run.

nfelo leans on its model more for teams where the model has lately been more accurate than the
line. Here:
  recent edge of a team = over its last N played games (previous seasons included), the average of
                          |line - result| - |raw model - result|   (positive: the model was closer)
  a game's value        = the average of its two teams' recent edges (0 for a team with fewer than
                          N such games)
  candidate             = board number + c x (raw model - line) x recent edge
  c for season S        = least squares through the origin on the board's misses in earlier
                          walk-forward seasons (0 for the first season)
Versions: A N = 4; B N = 8; C N = 8 with the recent edge reduced to its sign.

Measures (6 per league): average miss, RMS miss and side of the closing line, tuning and holdout
(college 2017-19 / 2021-25; NFL 2014-18 / 2019-25), on played games with a closing spread, against
the board on the same games. Chosen on tuning alone (better on all three tuning measures, then the
lowest tuning average miss). Kept only if better on all six and the holdout average miss by at
least 1 paired SE. Otherwise a trade-off or nothing. Each league on its own.

Writes experiments/out/recent_accuracy_weight_test.json.
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
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

OUT = HERE / "out" / "recent_accuracy_weight_test.json"
VERSIONS = {"A": (4, False), "B": (8, False), "C": (8, True)}
PERIODS = {"cfb": ((2017, 2019), (2021, 2025)), "nfl": ((2014, 2018), (2019, 2025))}


def nfl_frame() -> pd.DataFrame:
    """The NFL board's walk-forward frame, as nflmodel/pipeline.py builds it (PFF quarterback prior,
    role signals)."""
    from nflmodel import data, extra, game_roles, injuries, model, overrides, pff_qb, players, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    ov = overrides.load()
    games = overrides.apply_starters(games, ov, verbose=False)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, refresh_current=False, verbose=False, overrides=ov)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games,
                             qb_prior_model=pff_qb.load(qb, sorted(games.season.unique())))
    base = model.build_frame(games, engine, injuries=inj)
    frame, _ = model.walk_forward(base)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = players.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    rows = players.player_rows(players.team_games(frame, pg), pg, avail, pinfo)
    rows = overrides.apply_out(rows, ov) if ov is not None else rows
    frame, _ = model.walk_forward(game_roles.attach(base, game_roles.team_features(rows, R, avail)))
    if "game_type" in frame:
        frame = frame[frame.game_type == "REG"]
    return frame


def cfb_frame() -> pd.DataFrame:
    import cfb_outside_ratings_test as T
    from cfbmodel import model
    close, _ = T.frames()
    wf, _ = model.walk_forward(close)
    return wf[wf.fbs_game == 1]


def recent_edge(f: pd.DataFrame, n: int, sign: bool) -> np.ndarray:
    """Per game: the average of the two teams' recent edges, from games strictly before it."""
    order = ["season", "week"] + (["gameday"] if "gameday" in f else [])
    g = f.sort_values(order)
    edge = (g.spread_line - g.result).abs() - (g.model_margin - g.result).abs()
    usable = (g.played & g.spread_line.notna() & g.model_margin.notna()).to_numpy()
    hist: dict = {}
    out = {}
    ev = edge.to_numpy(float)
    for i, r in enumerate(g.itertuples()):
        vals = []
        for team in (r.home_team, r.away_team):
            h = hist.get(team, [])
            vals.append(float(np.mean(h[-n:])) if len(h) >= n else 0.0)
        out[r.Index] = 0.5 * (vals[0] + vals[1])
        if usable[i]:
            for team in (r.home_team, r.away_team):
                hist.setdefault(team, []).append(ev[i])
    x = pd.Series(out).reindex(f.index).to_numpy(float)
    return np.sign(x) if sign else x


def candidate(f: pd.DataFrame, n: int, sign: bool) -> tuple[pd.Series, dict]:
    x = (f.model_margin - f.spread_line).to_numpy(float) * recent_edge(f, n, sign)
    y = (f.result - f.proj_margin).to_numpy(float)
    ok = (f.played & f.spread_line.notna() & f.proj_margin.notna() & f.model_margin.notna()).to_numpy() & np.isfinite(x) & np.isfinite(y)
    season = f.season.to_numpy()
    cand = f.proj_margin.to_numpy(float).copy()
    cs = {}
    for s in sorted(set(season)):
        k = ok & (season < s) & (x != 0)
        c = float((x[k] @ y[k]) / (x[k] @ x[k])) if k.sum() >= 200 else 0.0
        cs[int(s)] = round(c, 4)
        m = (season == s) & np.isfinite(x)
        cand[m] = cand[m] + c * x[m]
    return pd.Series(cand, index=f.index), cs


def measures(g: pd.DataFrame) -> dict:
    eb, ec = (g.proj_margin - g.result).abs(), (g.cand - g.result).abs()
    d = ec - eb
    res = g.result - g.spread_line
    sb, sc = np.sign(g.proj_margin - g.spread_line), np.sign(g.cand - g.spread_line)
    both = (sb != 0) & (sc != 0) & (res != 0)
    return {"n": int(len(g)), "mae_board": float(eb.mean()), "mae_cand": float(ec.mean()), "d_mae": float(d.mean()),
            "se_d_mae": float(d.std(ddof=1) / np.sqrt(len(g))),
            "rmse_board": float(np.sqrt(((g.proj_margin - g.result) ** 2).mean())), "rmse_cand": float(np.sqrt(((g.cand - g.result) ** 2).mean())),
            "n_side": int(both.sum()), "win_board": float((np.sign(res[both]) == sb[both]).mean()), "win_cand": float((np.sign(res[both]) == sc[both]).mean()),
            "sides_changed": int((sb[both] != sc[both]).sum())}


def better(m: dict) -> dict:
    return {"mae": m["mae_cand"] < m["mae_board"], "rmse": m["rmse_cand"] < m["rmse_board"], "win": m["win_cand"] > m["win_board"]}


def run(league: str, f: pd.DataFrame) -> dict:
    (tlo, thi), (hlo, hhi) = PERIODS[league]
    res = {"versions": {}}
    frames = {}
    for v, (n, sign) in VERSIONS.items():
        cand, cs = candidate(f, n, sign)
        g = f.assign(cand=cand)
        g = g[g.played & g.spread_line.notna() & g.proj_margin.notna() & g.cand.notna()]
        mt = measures(g[g.season.between(tlo, thi)])
        res["versions"][v] = {"tune": mt, "tune_better": better(mt), "c_by_season": cs}
        frames[v] = g
    passing = [v for v in VERSIONS if all(res["versions"][v]["tune_better"].values())]
    res["tune_passing"] = passing
    if not passing:
        res["verdict"] = "No version is better than the board on all three tuning measures. Nothing is carried to the holdout; nothing is kept."
    else:
        chosen = min(passing, key=lambda v: res["versions"][v]["tune"]["mae_cand"])
        g = frames[chosen]
        mh = measures(g[g.season.between(hlo, hhi)])
        bh = better(mh)
        se_ok = mh["d_mae"] < 0 and abs(mh["d_mae"]) >= mh["se_d_mae"]
        n_better = sum(res["versions"][chosen]["tune_better"].values()) + sum(bh.values())
        res.update({"chosen": chosen, "holdout": mh, "holdout_better": bh, "one_se": bool(se_ok), "2026": measures(g[g.season == 2026])})
        res["verdict"] = (f"Version {chosen}: better on all 6 and the holdout average miss by at least 1 SE. Passes the keep rule (provisional)."
                          if all(bh.values()) and se_ok else f"Version {chosen}: a trade-off ({n_better} of 6 better; holdout 1 SE: {bool(se_ok)}). Ryan's call.")
    return res


def main() -> None:
    out = {}
    for league, build in (("cfb", cfb_frame), ("nfl", nfl_frame)):
        out[league] = run(league, build())
        for v, r in out[league]["versions"].items():
            t = r["tune"]
            print(f"{league} {v} tuning: miss {t['mae_board']:.3f} -> {t['mae_cand']:.3f} ({t['d_mae']:+.4f} ± {t['se_d_mae']:.4f}), rms {t['rmse_board']:.3f} -> {t['rmse_cand']:.3f}, "
                  f"side {100 * t['win_board']:.1f}% -> {100 * t['win_cand']:.1f}% ({t['sides_changed']} changed); better on {sum(r['tune_better'].values())} of 3")
        print(league, out[league]["verdict"], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
