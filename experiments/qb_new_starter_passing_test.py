"""Passing yards for quarterbacks without an established run of starts.

    python experiments/qb_new_starter_passing_test.py

Nov 9, 2026 (PLAYERS.md "Quarterback rushing per start" noted it). Projected starters in games they
started, 2014 on, by the share of their recent active games that were starts: under 30% were
projected for 221 passing yards and threw for 204; 30-60% 226 against 208; established starters
243 against 245. A new starter's per-start form rests on few starts, so his number leans on the
chain, whose yards per attempt are shrunk toward the average of all quarterbacks, who are mostly
established starters. The last step is fitted on starters who started, so it learns the average
starter's relation between chain and result, not a new starter's.

THE CHANGE ("new-starter term"): one more column in the passing calibration (players.calib_X) for
the passing stats: raw projection x new, where new = 1 - min(decayed starts / decayed active
games, 1), 0 for an established starter and 1 for a quarterback who has never started in his
recent active games. The same regression, refitted walk-forward with the extra column. Only
quarterback passing rows can move (the column is 0 for everyone else).

MEASURES, written before the run. All four must be better than the board for the change to be
kept (the keep rule); mixed goes to Ryan. Rows: projected starters in games they started.
  1-2  RMSE of passing yards: tuning 2014-2018, holdout 2019 on
  3-4  Brier at the DraftKings/FanDuel passing-yards main lines, 2023-2025 pooled: pricing snapshot
       (pregame projections) and close
Widths fitted on seasons before the one scored, for each version on its own projections; the saved
correction the same in both. Standard errors paired, clustered by game.

REPORTED ALONGSIDE, not part of the rule: the same for pass attempts and completions (the column
is added to their fits too, since they share the calibration code; whether it helps them is
reported, and the production change would carry only the stats that pass); both versions by
start-share bucket; bias; 2026 alone.

Writes experiments/out/qb_new_starter_passing_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import players as PL, propbackfill as B, props as P  # noqa: E402
import props_real_lines as RL  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "qb_new_starter_passing_test.json"
TUNING, HOLDOUT_FROM, FIRST = (2014, 2018), 2019, 2014
LINE_SEASONS = (2023, 2024, 2025)
STATS = ("pass_yds", "pass_att", "completions")
BUCKETS, NAMES = [-0.01, 0.3, 0.6, 0.85, 1.01], ["under 30% starts", "30-60%", "60-85%", "85%+ (established)"]


def newness(p: pd.DataFrame) -> np.ndarray:
    return (1 - (p["start_sum"] / p["one_sum"].replace(0, np.nan)).fillna(0).clip(0, 1)).to_numpy(float)


def with_term(proj: pd.DataFrame) -> pd.DataFrame:
    """The board's table with the passing stats refitted walk-forward with the new-starter term.
    Keeps <stat>_cur for the board's numbers."""
    t = proj.copy()
    for st in STATS:
        t[st + "_cur"] = t[st + "_proj"]
    qb1 = (t["group"] == "QB") & (t["att_share"] > 0.5)
    started = qb1 & (t["started"] == 1) & (t["active"] == 1) & t["played"].astype(bool)
    for s in sorted(x for x in t["season"].unique() if x >= FIRST - 1):
        fit = started & (t["season"] < s) & (t["season"] >= s - 6)
        if fit.sum() < 200:
            continue
        cur = qb1 & (t["season"] == s)
        for st in STATS:
            Xf = np.column_stack([PL.calib_X(t[fit], st), PL.calib_X(t[fit], st)[:, 0] * newness(t[fit])])
            Xc = np.column_stack([PL.calib_X(t[cur], st), PL.calib_X(t[cur], st)[:, 0] * newness(t[cur])])
            m = PL.Lin(1.0).fit(Xf, t.loc[fit, st].to_numpy(float))
            t.loc[cur, st + "_proj"] = np.clip(m.predict(Xc), 0, None)
    return t


def paired(a, b, games):
    d = a - b
    g = d.groupby(games).sum()
    return float(d.mean()), float(g.std(ddof=1) * np.sqrt(len(g)) / len(d)) if len(g) > 1 else float("nan")


def rmse_report(t: pd.DataFrame) -> dict:
    q = t[(t["group"] == "QB") & (t["att_share"] > 0.5) & (t["started"] == 1) & t["played"].astype(bool) & (t["season"] >= FIRST)].copy()
    q["bucket"] = pd.cut(1 - newness(q), BUCKETS, labels=NAMES).astype(str)
    spans = {"tuning": q["season"].between(*TUNING), "holdout": q["season"] >= HOLDOUT_FROM, "2026": q["season"] == 2026}
    out = {}
    for st in STATS:
        out[st] = {}
        for span, m in spans.items():
            g = q[m]
            e0, e1 = (g[st + "_cur"] - g[st]) ** 2, (g[st + "_proj"] - g[st]) ** 2
            d, se = paired(e1, e0, g["game_id"])
            out[st][span] = {"n": int(len(g)), "current_rmse": round(float(np.sqrt(e0.mean())), 3), "new_rmse": round(float(np.sqrt(e1.mean())), 3),
                             "mse_change": round(d, 3), "mse_se": round(se, 3),
                             "current_bias": round(float((g[st] - g[st + "_cur"]).mean()), 2), "new_bias": round(float((g[st] - g[st + "_proj"]).mean()), 2)}
        out[st]["by_start_history"] = {b: {"n": int(len(g)), "actual": round(float(g[st].mean()), 1), "current": round(float(g[st + "_cur"].mean()), 1),
                                           "new": round(float(g[st + "_proj"].mean()), 1),
                                           "current_rmse": round(float(np.sqrt(((g[st + "_cur"] - g[st]) ** 2).mean())), 2),
                                           "new_rmse": round(float(np.sqrt(((g[st + "_proj"] - g[st]) ** 2).mean())), 2)} for b, g in q.groupby("bucket")}
    return out


def lines_report(t: pd.DataFrame, kind: str) -> dict:
    saved = P.PropModel.load()
    cur = t.copy()
    for st in STATS:
        cur[st + "_proj"] = t[st + "_cur"]
    priced = {}
    for name, v in (("current", cur), ("new", t)):
        parts = []
        for season in LINE_SEASONS:
            snaps = B.load(season, kind)
            snaps = snaps[snaps["market"].isin(["player_pass_yds", "player_pass_attempts", "player_pass_completions"])]
            if snaps.empty:
                continue
            m = P.PropModel(FAMILY).fit(v[(v.season < season) & (v.season >= 2014)], stats=list(STATS))
            m.iso = saved.iso
            x = RL.priced(v, m, snaps, season)
            if len(x):
                parts.append(x)
        priced[name] = pd.concat(parts, ignore_index=True)
    key = ["game_id", "player_id", "market", "line", "book"]
    out = {}
    for st in STATS:
        a = priced["current"][priced["current"]["stat"] == st]
        b = priced["new"][priced["new"]["stat"] == st][key + ["our_over", "our_under"]].rename(columns={"our_over": "o2", "our_under": "u2"})
        j = a.merge(b, on=key, how="inner")
        j = j[j["y"] != j["line"]]
        if j.empty:
            continue
        hit = (j["y"] > j["line"]).astype(float)
        p0, p1 = j["our_over"] / (j["our_over"] + j["our_under"]), j["o2"] / (j["o2"] + j["u2"])
        d, se = paired((p1 - hit) ** 2, (p0 - hit) ** 2, j["game_id"])
        out[st] = {"n": int(len(j)), "over_rate": round(float(hit.mean()), 4), "brier_books": round(float(((j["fair_over"] - hit) ** 2).mean()), 5),
                   "current": round(float(((p0 - hit) ** 2).mean()), 5), "new": round(float(((p1 - hit) ** 2).mean()), 5),
                   "brier_change": round(d, 5), "se": round(se, 5)}
    return out


def main():
    rep = {}
    close = with_term(load_proj(None))
    rep["rmse"] = rmse_report(close)
    print(json.dumps(rep["rmse"], indent=1), flush=True)
    rep["lines"] = {"close": lines_report(close, "close"), "pricing": lines_report(with_term(RL.load_pregame()), "flag")}
    print(json.dumps(rep["lines"], indent=1), flush=True)
    m = [rep["rmse"]["pass_yds"][sp]["mse_change"] < 0 for sp in ("tuning", "holdout")]
    m += [rep["lines"][sn]["pass_yds"]["brier_change"] < 0 for sn in ("pricing", "close")]
    rep["measures_better"], rep["kept_by_rule"] = int(sum(m)), bool(all(m))
    print("measures better:", rep["measures_better"], "of 4; kept by the rule:", rep["kept_by_rule"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
