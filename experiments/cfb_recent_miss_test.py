"""A recent-miss nudge for the college game model (Nov 18, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_recent_miss_test.py

Lead: in college about a tenth of a team's recent average miss against the model shows up in
its next game (Project doc claude/team-rating-bias-check-2026-11-18.md). Plan, written before
the run: claude/plan-cfb-recent-miss-test-2026-11-18.md.

The change: the model's own number plus b x (how each team has done against that number lately).
  margin nudge = b_margin x (home team's recent average miss - away team's)
  total nudge  = b_total  x (home team's recent average total miss + away team's)
A team's miss = its actual margin minus the current model's walk-forward margin for it; for
totals, the game's total minus the model's. Only earlier games of the same season count.
b for season S is fitted by least squares (no intercept) on every walk-forward season before S
(2017 gets no nudge).

Versions:
  A  average over the last four games, three minimum
  B  season to date, shrunk: sum of misses / (games + 4)
  C  as A, each game's miss capped at +/-21 before averaging

Rule, set before running:
  - Measures: raw model average miss, margin and total, tuning 2017-19 and holdout 2021-25
    (2020 in neither). Four measures against the current model on the same games.
  - Chosen on tuning alone: among versions better on both tuning measures, the lowest tuning margin.
  - Kept only if the chosen version is better on all four and the holdout improvement is at least
    1 paired standard error on both margin and total. Provisional, rechecked on 2026 in January.
  - Mixed, or better everywhere but under 1 SE: a trade-off for Ryan.
  - No version better on both tuning measures: nothing is carried to the holdout, nothing kept.
  - Reported, not used: paired differences and SEs, season split, 2026 so far, coefficients, and
    the board's number (line moved toward the model at the season's weight) with and without.

Writes experiments/out/cfb_recent_miss_test.json.
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
OUT = HERE / "out" / "cfb_recent_miss_test.json"
TUNE, HOLD = (2017, 2019), (2021, 2025)
CAP = 21.0


def frame() -> pd.DataFrame:
    from cfbmodel import data, extra, model
    from cfbmodel.pipeline import CFBRatingEngine, PFFRET
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games["wind_source"] = np.where(games["played"], "recorded", "")
    inputs = extra.load_inputs(games, refresh_current=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games, extra=inputs["rated"])
    f, _ = model.walk_forward(PFFRET.add(model.build_frame(games, engine, inputs=inputs), verbose=True))
    f = f[f.played & f.model_margin.notna() & (f.postseason == 0)].copy()
    f["kick"] = pd.to_datetime(f["kickoff"])
    f["mm"] = f["result"] - f["model_margin"]
    f["mt"] = f["total"] - f["model_total"]
    return f.sort_values("kick").reset_index(drop=True)


def recent(f: pd.DataFrame, version: str) -> tuple[np.ndarray, np.ndarray]:
    """Per game: (home recent margin miss - away's, home recent total miss + away's)."""
    from cfbmodel import config as C
    h = f[["game_id", "season", "kick"]].assign(team=f.home_team, side="h", m=f.mm, t=f.mt)
    a = f[["game_id", "season", "kick"]].assign(team=f.away_team, side="a", m=-f.mm, t=f.mt)
    x = pd.concat([h, a]).sort_values(["season", "team", "kick"])
    if version == "C":
        x["m"], x["t"] = x.m.clip(-CAP, CAP), x.t.clip(-CAP, CAP)
    grp = x.groupby(["season", "team"])
    for col in ("m", "t"):
        if version in ("A", "C"):
            x["r_" + col] = grp[col].transform(lambda v: v.shift(1).rolling(4, min_periods=3).mean())
        else:
            n = grp.cumcount()
            x["r_" + col] = (grp[col].cumsum() - x[col]) / (n + 4)
    x.loc[x.team == C.POOL, ["r_m", "r_t"]] = 0.0       # the non-FBS pool is not one team
    x[["r_m", "r_t"]] = x[["r_m", "r_t"]].fillna(0.0)
    hh = x[x.side == "h"].set_index("game_id")
    aa = x[x.side == "a"].set_index("game_id")
    gid = f.game_id
    return (hh.r_m.reindex(gid).to_numpy() - aa.r_m.reindex(gid).to_numpy(),
            hh.r_t.reindex(gid).to_numpy() + aa.r_t.reindex(gid).to_numpy())


def nudged(f: pd.DataFrame, version: str) -> tuple[pd.DataFrame, dict]:
    dm, dt = recent(f, version)
    out = f[["game_id", "season"]].copy()
    out["nm"], out["nt"] = f.model_margin.to_numpy(), f.model_total.to_numpy()
    coefs = {}
    for s in sorted(f.season.unique()):
        past = (f.season < s).to_numpy()
        if not past.any():
            continue
        bm = float(dm[past] @ f.mm.to_numpy()[past] / max(1e-9, dm[past] @ dm[past]))
        bt = float(dt[past] @ f.mt.to_numpy()[past] / max(1e-9, dt[past] @ dt[past]))
        cur = (f.season == s).to_numpy()
        out.loc[cur, "nm"] = f.model_margin.to_numpy()[cur] + bm * dm[cur]
        out.loc[cur, "nt"] = f.model_total.to_numpy()[cur] + bt * dt[cur]
        coefs[int(s)] = [round(bm, 3), round(bt, 3)]
    return out, coefs


def score(f, n, lo, hi) -> dict:
    k = f.season.between(lo, hi).to_numpy()
    base_m, base_t = (f.result - f.model_margin).abs().to_numpy()[k], (f.total - f.model_total).abs().to_numpy()[k]
    new_m, new_t = (f.result - n.nm).abs().to_numpy()[k], (f.total - n.nt).abs().to_numpy()[k]
    dm, dt = new_m - base_m, new_t - base_t
    se = lambda d: float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"games": int(k.sum()), "base": [round(float(base_m.mean()), 3), round(float(base_t.mean()), 3)],
            "new": [round(float(new_m.mean()), 3), round(float(new_t.mean()), 3)],
            "diff": [round(float(dm.mean()), 4), round(float(dt.mean()), 4)], "se": [round(se(dm), 4), round(se(dt), 4)]}


def board(f, n, lo, hi) -> dict:
    """The board's number: the line moved toward the model at the season's blend weight."""
    k = (f.season.between(lo, hi) & f.spread_line.notna() & f.total_line.notna()).to_numpy()
    x, y = f[k], n[k]
    pm = lambda mod: x.spread_line + x.w_margin * (mod - x.spread_line)
    pt = lambda mod: x.total_line + x.w_total * (mod - x.total_line)
    r = lambda v: round(float(v), 3)
    return {"games": int(k.sum()),
            "margin": [r((x.result - pm(x.model_margin)).abs().mean()), r((x.result - pm(y.nm)).abs().mean())],
            "total": [r((x.total - pt(x.model_total)).abs().mean()), r((x.total - pt(y.nt)).abs().mean())],
            "line": [r((x.result - x.spread_line).abs().mean()), r((x.total - x.total_line).abs().mean())]}


def main():
    f = frame()
    res = {"rule": "see docstring", "versions": {}, "tuning": {}}
    built = {}
    for v in "ABC":
        built[v] = nudged(f, v)
        res["tuning"][v] = score(f, built[v][0], *TUNE)
    ok = [v for v in "ABC" if res["tuning"][v]["diff"][0] < 0 and res["tuning"][v]["diff"][1] < 0]
    res["better_on_both_tuning"] = ok
    if not ok:
        res["verdict"] = "No version is better on both tuning measures: nothing carried to the holdout, nothing kept."
    else:
        pick = min(ok, key=lambda v: res["tuning"][v]["new"][0])
        n, coefs = built[pick]
        h = score(f, n, *HOLD)
        res["chosen"], res["holdout"], res["coefs"] = pick, h, coefs
        better = h["diff"][0] < 0 and h["diff"][1] < 0
        one_se = -h["diff"][0] >= h["se"][0] and -h["diff"][1] >= h["se"][1]
        res["verdict"] = ("KEPT (provisional)" if better and one_se else
                          "Trade-off: better on all four, holdout under 1 SE on at least one" if better else
                          "Trade-off: mixed across the four measures")
        res["by_season"] = {int(s): score(f, n, s, s) for s in sorted(f.season.unique()) if s >= 2018}
        res["s2026"] = score(f, n, 2026, 2026)
        res["board"] = {"tuning": board(f, n, *TUNE), "holdout": board(f, n, *HOLD), "2026": board(f, n, 2026, 2026)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
