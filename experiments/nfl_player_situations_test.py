"""NFL player projections in four situations (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/nfl_player_situations_test.py

T4 of the Project doc claude/plan-batch-tests-2026-11-20.md (written before the run). The situations
are Ryan's Weekly Review ideas (Nov 18). Committed before its first run.

Each is a term added after the board's final projection, on pass_yds (QB), rush_yds (RB) and
rec_yds (WR/TE/RB) separately; rows and harness as experiments/nfl_underdog_term_test.py:
    new = max(0, proj + b * x)
  S1 first game back   x = proj when the player missed his team's previous game this season after
                       playing for that team earlier in it
  S2 last game's miss  x = last game's (actual - projection) for the same stat, this season, capped
                       at +/- 1 standard deviation of that stat's misses on earlier seasons
  S3 division game     x = proj when the opponent is a division rival
  S4 former team       x = proj when the opponent is a team he played for earlier (history from 2014)
  b: one number per stat and situation, least squares through the origin on earlier seasons only,
  either sign, 0 until 100 earlier rows with x != 0 exist. The situation flags come from who
  actually played earlier games, known before kickoff, and are the same for game-day and bet-time.

Measures per stat and situation (6): RMSE, game-day and bet-time, tuning 2014-18 and holdout
2019-25; Brier at the 2023-25 DK/FD main lines, closing lines with game-day projections and the
pricing snapshot with bet-time.
Rule: carried to the holdout only if better on both tuning RMSE measures. Kept only if better on
all six and each holdout RMSE gain >= 1 paired SE. Mixed, or under 1 SE: a trade-off for Ryan.
Twelve chances (4 x 3), so about one pass by luck is expected; any pass is provisional and is
reported with that warning.
Reported, not used: b by season, bias on the affected rows, 2026 so far.

Writes experiments/out/nfl_player_situations_test.json.
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

import nfl_lift_shrink_test as L  # noqa: E402

OUT = HERE / "out" / "nfl_player_situations_test.json"
STATS = L.STATS
TUNE, HOLD = L.TUNE, L.HOLD
SITUATIONS = ("S1 first game back", "S2 last game's miss", "S3 division game", "S4 former team")
DIVISIONS = [("BUF", "MIA", "NE", "NYJ"), ("BAL", "CIN", "CLE", "PIT"), ("HOU", "IND", "JAX", "TEN"), ("DEN", "KC", "LV", "LAC"),
             ("DAL", "NYG", "PHI", "WAS"), ("CHI", "DET", "GB", "MIN"), ("ATL", "CAR", "NO", "TB"), ("ARI", "LA", "SF", "SEA")]
DIV = {t: i for i, d in enumerate(DIVISIONS) for t in d}


def flags(g: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, player_id): first_back, division, former. From the game-day table's record of
    who played; everything used is from games before the row's own."""
    p = g[["game_id", "player_id", "team", "opp", "season", "t", "active", "played"]].copy()
    p["did"] = (p["active"] == 1) & p["played"].astype(bool)
    tg = p[["team", "season", "t"]].drop_duplicates().sort_values(["team", "season", "t"])
    tg["gnum"] = tg.groupby(["team", "season"]).cumcount()
    p = p.merge(tg, on=["team", "season", "t"], how="left")
    did = set(map(tuple, p.loc[p.did, ["player_id", "team", "season", "gnum"]].to_numpy()))
    first = p[p.did].groupby(["player_id", "team", "season"])["gnum"].min().to_dict()
    fb = np.zeros(len(p), bool)
    for i, r in enumerate(p[["player_id", "team", "season", "gnum"]].itertuples(index=False)):
        f0 = first.get((r.player_id, r.team, r.season))
        if r.gnum >= 2 and f0 is not None and f0 < r.gnum - 1 and (r.player_id, r.team, r.season, r.gnum - 1) not in did:
            fb[i] = True
    p["first_back"] = fb
    p["division"] = (p["team"].map(DIV) == p["opp"].map(DIV)) & p["team"].map(DIV).notna()
    # former team: any team he played for in an earlier game, other than his current one
    former = np.zeros(len(p), bool)
    q = p.sort_values(["player_id", "t"])
    seen: dict = {}
    for idx, pid, team, opp, d in zip(q.index, q["player_id"], q["team"], q["opp"], q["did"]):
        s = seen.setdefault(pid, set())
        if opp in s and opp != team:
            former[idx] = True
        if d:
            s.add(team)
    p["former"] = former
    return p[["game_id", "player_id", "first_back", "division", "former"]]


def last_miss(g: pd.DataFrame, st: str) -> pd.Series:
    """Last game's (actual - projection) for the stat, same player and season, from the last game he
    played; NaN without one. Uncapped."""
    a = g[(g["active"] == 1) & g["played"].astype(bool) & g[st].notna() & g[st + "_proj"].notna()].sort_values(["player_id", "t"])
    miss = (a[st] - a[st + "_proj"]).groupby([a["player_id"], a["season"]]).shift()
    out = pd.Series(np.nan, index=g.index)
    out.loc[a.index] = miss
    return out


def adjusted(p: pd.DataFrame, st: str, x: np.ndarray) -> tuple[np.ndarray, dict]:
    m = L.mask(p, st).to_numpy()
    proj, y, season = p[st + "_proj"].to_numpy(float), p[st].to_numpy(float), p["season"].to_numpy()
    new, bs = proj.copy(), {}
    for s in sorted(set(season[season >= 2014])):
        k = m & (season < s) & (season >= 2013) & (x != 0) & np.isfinite(x)
        b = float((x[k] @ (y[k] - proj[k])) / max(1e-9, x[k] @ x[k])) if k.sum() >= 100 else 0.0
        sel = season == s
        new[sel] = np.maximum(0.0, proj[sel] + b * np.nan_to_num(x[sel]))
        bs[int(s)] = round(b, 4)
    return new, bs


def capped(miss: np.ndarray, p: pd.DataFrame, st: str) -> np.ndarray:
    """Cap each season's misses at +/- 1 SD of that stat's misses on earlier seasons (masked rows)."""
    m = L.mask(p, st).to_numpy()
    season = p["season"].to_numpy()
    err = (p[st] - p[st + "_proj"]).to_numpy(float)
    out = np.nan_to_num(miss.copy())
    for s in sorted(set(season)):
        k = m & (season < s) & np.isfinite(err)
        sd = float(err[k].std()) if k.sum() >= 100 else np.inf
        sel = season == s
        out[sel] = np.clip(out[sel], -sd, sd)
    return out


def affected(p, new, st, lo, hi, x) -> dict:
    k = (L.mask(p, st) & p["season"].between(lo, hi)).to_numpy() & (x != 0) & np.isfinite(x)
    if not k.any():
        return {"n": 0}
    y, old, nw = p[st].to_numpy(float)[k], p[st + "_proj"].to_numpy(float)[k], new[k]
    return {"n": int(k.sum()), "bias_now": round(float((y - old).mean()), 2), "bias_new": round(float((y - nw).mean()), 2),
            "se": round(float((y - old).std(ddof=1) / np.sqrt(k.sum())), 2)}


def main():
    T = L.tables()
    F = flags(T["gameday"])
    tabs = {w: T[w].merge(F, on=["game_id", "player_id"], how="left") for w in ("gameday", "bettime")}
    assert all(len(tabs[w]) == len(T[w]) for w in tabs)
    res = {"plan": "claude/plan-batch-tests-2026-11-20.md (T4)"}
    import nfl_pff_pairs_test as T0
    for sit in SITUATIONS:
        for st in STATS:
            xs = {}
            for w in ("gameday", "bettime"):
                p = tabs[w]
                proj = p[st + "_proj"].to_numpy(float)
                if sit.startswith("S1"):
                    x = np.where(p["first_back"].fillna(False).to_numpy(bool), proj, 0.0)
                elif sit.startswith("S2"):
                    x = capped(last_miss(tabs["gameday"], st).to_numpy(float), p, st)
                elif sit.startswith("S3"):
                    x = np.where(p["division"].fillna(False).to_numpy(bool), proj, 0.0)
                else:
                    x = np.where(p["former"].fillna(False).to_numpy(bool), proj, 0.0)
                xs[w] = np.nan_to_num(x)
            cand = {w: adjusted(tabs[w], st, xs[w]) for w in ("gameday", "bettime")}
            r = {"tuning": {w: L.rmse(tabs[w], cand[w][0], st, *TUNE) for w in ("gameday", "bettime")},
                 "b_by_season": cand["gameday"][1],
                 "affected_tuning": affected(tabs["gameday"], cand["gameday"][0], st, *TUNE, xs["gameday"])}
            if not all(r["tuning"][w]["diff"] < 0 for w in ("gameday", "bettime")):
                r["verdict"] = "Not better on both tuning measures: nothing carried to the holdout, nothing kept."
            else:
                r["holdout"] = {w: L.rmse(tabs[w], cand[w][0], st, *HOLD) for w in ("gameday", "bettime")}
                frames = {}
                for w in ("gameday", "bettime"):
                    c = T[w].copy()
                    c[st + "_proj"] = cand[w][0]
                    frames[w] = c
                r["brier"] = {"closing lines, game-day": T0.brier(T["gameday"], frames["gameday"], "close", (st,)),
                              "pricing snapshot, bet-time": T0.brier(T["bettime"], frames["bettime"], "flag", (st,))}
                r["affected_holdout"] = affected(tabs["gameday"], cand["gameday"][0], st, *HOLD, xs["gameday"])
                r["season_2026_affected"] = affected(tabs["gameday"], cand["gameday"][0], st, 2026, 2026, xs["gameday"])
                r["season_2026_all"] = L.rmse(tabs["gameday"], cand["gameday"][0], st, 2026, 2026)
                six = list(r["tuning"].values()) + list(r["holdout"].values()) + list(r["brier"].values())
                n = sum(v["diff"] < 0 for v in six)
                one_se = all(-v["diff"] >= v["se"] for v in r["holdout"].values())
                r["better_on"] = f"{n} of 6"
                r["verdict"] = ("PASSES the keep rule (provisional; one of twelve chances)" if n == 6 and one_se
                                else "Trade-off: better on all six, a holdout gain under 1 SE" if n == 6 else f"Trade-off: better on {n} of 6")
            res[f"{sit} | {st}"] = r
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(res, indent=1, default=str))     # saved after every cell: the run is long
            t = r["tuning"]
            print(f"{sit} | {st}: tuning game-day {t['gameday']['diff']:+.4f} ± {t['gameday']['se']:.4f}, bet-time {t['bettime']['diff']:+.4f}; "
                  f"affected {r['affected_tuning'].get('n')} rows, bias {r['affected_tuning'].get('bias_now')}; {r['verdict']}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
