"""Do long runs mislead the player model? Cap them and see.

    python experiments/capped_runs_test.py

Every run is counted at no more than that season's 97.5th percentile of rushing gains (about 18
yards), then scaled by the season's raw/capped ratio so the league average is unchanged. The
capped yards replace the raw ones as inputs (each player's yards per carry, and each defense's
yards per carry allowed); the outcome being predicted is still his actual rushing yards.

Keep rule, set before running: adopt it if the holdout seasons' average miss on rushing yards
(RBs and QBs, the board's rush_yds rows) improves by at least 2 paired standard errors and the
tuning seasons don't get worse. Prompted by a split-half check on 2023-2025 (first-half capped
yards per carry correlated 0.365 with second-half yards per carry, raw 0.342; receivers showed
nothing, so only runs are tested here).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model, players as PL, players_data as PD  # noqa: E402
from nflmodel.data import _download  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
CUT_Q = 0.975
COLS = ["game_id", "season_type", "rush_attempt", "two_point_attempt", "qb_kneel", "rushing_yards", "rusher_player_id"]


def capped_runs(seasons) -> pd.DataFrame:
    """(game_id, player_id) -> capped and scaled rushing yards; per-season cut and scale."""
    cache = C.CACHE / "capped_runs.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for s in seasons:
        raw = C.RAW / f"pbp_cap_{s}.parquet"
        _download(C.PBP_URL.format(season=s), raw)
        p = pd.read_parquet(raw, columns=COLS)
        raw.unlink()
        p = p[(p.rush_attempt == 1) & (p.two_point_attempt.fillna(0) == 0) & (p.qb_kneel.fillna(0) == 0)
              & p.rusher_player_id.notna() & p.rushing_yards.notna()]
        cut = p.rushing_yards.quantile(CUT_Q)
        p["cap"] = p.rushing_yards.clip(upper=cut)
        k = p.rushing_yards.sum() / p.cap.sum()
        g = p.groupby(["game_id", "rusher_player_id"]).agg(raw=("rushing_yards", "sum"), cap=("cap", "sum")).reset_index()
        g["rush_yds_cap"] = g["cap"] * k
        g["season"] = s
        rows.append(g.rename(columns={"rusher_player_id": "player_id"}))
        print(f"  {s}: cut {cut:.0f} yds, scale {k:.3f}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(cache, index=False)
    return out


def errors(proj: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
    ev = proj[proj.played.astype(bool) & (proj.active == 1) & proj.group.isin(["RB", "QB"])].copy()
    ev["actual"] = actual.reindex(ev.index)
    ev = ev[(ev["actual"].fillna(0) != 0) | (ev["rush_yds_proj"] > 0)]
    ev["err"] = (ev["rush_yds_proj"] - ev["actual"].fillna(0)).abs()
    return ev[["season", "group", "err"]]


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    av = PL.load_availability(verbose=False)

    cap = capped_runs(sorted(pg.season.unique()))
    pg2 = pg.merge(cap[["game_id", "player_id", "rush_yds_cap"]], on=["game_id", "player_id"], how="left")
    pg2["rush_yds"] = pg2["rush_yds_cap"].fillna(pg2["rush_yds"])
    pg2 = pg2.drop(columns="rush_yds_cap")

    base = PL.walk_forward(PL.build(frame, pg, info, av, inj, pr, games=games))
    test = PL.walk_forward(PL.build(frame, pg2, info, av, inj, pr, games=games))
    actual = base["rush_yds"]                        # the real yards, for both
    e0, e1 = errors(base, actual), errors(test, actual)
    j = e0.join(e1[["err"]], rsuffix="_cap", how="inner")
    j["d"] = j["err_cap"] - j["err"]                 # negative = capping helps
    res = {}
    for label, sub in (("tuning", j[j.season < C.HOLDOUT_FROM]), ("holdout", j[j.season >= C.HOLDOUT_FROM])):
        for grp, x in (("RB+QB", sub), ("RB", sub[sub.group == "RB"])):
            res[f"{label} {grp}"] = {"n": int(len(x)), "mae_raw": round(x.err.mean(), 3), "mae_capped": round(x.err_cap.mean(), 3),
                                     "change": round(x.d.mean(), 4), "se": round(x.d.std() / np.sqrt(len(x)), 4)}
    h, t = res["holdout RB+QB"], res["tuning RB+QB"]
    res["keep"] = bool(h["change"] <= -2 * h["se"] and t["change"] <= 0)
    OUT.mkdir(exist_ok=True)
    (OUT / "capped_runs_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
