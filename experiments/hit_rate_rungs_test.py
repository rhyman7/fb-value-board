"""Ryan's hit-rate method on the alternate yardage ladders, against real DraftKings / FanDuel prices.

    python experiments/hit_rate_rungs_test.py

The method: a player's "true probability" of clearing a rung (say 60+ receiving yards) is how often
he cleared it this season and last; when that is 90% or more and well above the book's implied
chance, bet it. Tested on the saved ladder prices (data/props/history/<season>/, 2023-2025; the
pricing snapshot before kickoff, posted Over only) with the players' own game logs. No model.

For every rung: the player's games before this one, this season and last (8 or more needed), and
the share in which he went over the rung. Reported:

1. Calibration by hit-rate band: hit rate going in, the book's implied chance (raw, and with the
   board's assumed 10% rung margin taken out), and how often the rung actually cleared.
2. The bets: per player, game and stat, the highest rung he cleared 90%+ of the time (the best
   price), at the better of the two books' prices; and the subset where the hit rate beats the
   raw implied chance by 5 points or more (an "edge" by the method's own math). Graded on results
   (return per $1) and on price closing line value against the same book's closing price.
   Standard errors clustered by game.

Rule, set before running (Nov 2, 2026), the same as for the other ladder bets
(experiments/props_ladders.py): a kind of bet earns a place only with 30+ bets and both its return
and its price CLV above zero by more than two standard errors.

Writes experiments/out/hit_rate_rungs_test.json.
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
from nflmodel import config as C, odds as O, propbackfill as B, propmarket as PM  # noqa: E402
from props_calibration import load_proj  # noqa: E402
from props_ladders import LADDERS, _se  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "hit_rate_rungs_test.json"
SEASONS = (2023, 2024, 2025)
MIN_PRIOR = 8
STAT = {"player_pass_yds_alternate": "pass_yds", "player_rush_yds_alternate": "rush_yds",
        "player_reception_yds_alternate": "rec_yds"}
BANDS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]


def rungs(proj: pd.DataFrame, season: int, kind: str) -> pd.DataFrame:
    s = B.load(season, kind)
    s = s[s["market"].isin(LADDERS) & (s["side"].str.lower() == "over")].copy()
    if s.empty:
        return s
    info = pd.read_csv(C.CACHE / f"player_info_{season}.csv", dtype={"player_id": str})
    rows = proj[proj.season == season].merge(info[["player_id", "name"]], on="player_id", how="left")
    rows["nname"] = rows["name"].map(PM.norm_name)
    rows = rows[~rows.duplicated(["game_id", "nname"], keep=False)]
    s["nname"] = s["player"].map(PM.norm_name)
    s = s.sort_values("snapshot").drop_duplicates(["game_id", "nname", "market", "line", "book"], keep="last")
    s = s[["game_id", "nname", "market", "line", "book", "price"]].merge(
        rows[["game_id", "nname", "player_id", "season", "week", "pass_yds", "rush_yds", "rec_yds"]], on=["game_id", "nname"])
    s["stat"] = s["market"].map(STAT)
    s["y"] = [r[st] for r, st in zip(s.to_dict("records"), s["stat"])]
    s["dec"] = O.american_to_decimal(s["price"])
    s["raw"] = O.implied_prob(s["price"])
    s["fair"] = O.one_sided_fair(s["price"], PM.ALT_MARGIN, "power")
    s["hit"] = (s["y"] > s["line"]).astype(float)
    return s


def add_prior(x: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Each rung's prior hit rate: the player's games this season (earlier weeks) and last season."""
    logs = {}
    for (pid, stat), g in proj.melt(id_vars=["player_id", "season", "week"], value_vars=["pass_yds", "rush_yds", "rec_yds"],
                                     var_name="stat").groupby(["player_id", "stat"]):
        logs[(pid, stat)] = g.sort_values(["season", "week"])[["season", "week", "value"]].to_numpy(float)
    n, rate = [], []
    for pid, stat, s, w, line in zip(x.player_id, x.stat, x.season, x.week, x.line):
        a = logs.get((pid, stat))
        if a is None:
            n.append(0), rate.append(np.nan)
            continue
        m = ((a[:, 0] == s) & (a[:, 1] < w)) | (a[:, 0] == s - 1)
        v = a[m, 2]
        n.append(len(v))
        rate.append(float((v > line).mean()) if len(v) else np.nan)
    return x.assign(n_prior=n, prior=rate)


def grade(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0}
    b = b.assign(profit=np.where(b.hit > 0, b.dec - 1, -1.0))
    c = b.dropna(subset=["clv"])
    r = {"bets": int(len(b)), "games": int(b.game_id.nunique()), "won": round(float(b.hit.mean()), 4),
         "hit_rate_going_in": round(float(b.prior.mean()), 4), "book_implied_raw": round(float(b.raw.mean()), 4),
         "book_fair": round(float(b.fair.mean()), 4), "median_price": float(b.price.median()),
         "return": round(float(b.profit.mean()), 4), "return_se": round(_se(b.profit, b.game_id), 4),
         "clv_bets": int(len(c)), "clv_price": round(float(c.clv.mean()), 4) if len(c) else None,
         "clv_price_se": round(_se(c.clv, c.game_id), 4) if len(c) > 1 else None}
    r["passes"] = bool(r["bets"] >= 30 and r["return"] > 2 * r["return_se"] and r["clv_price"] is not None
                       and r["clv_price"] > 2 * (r["clv_price_se"] or np.inf))
    return r


def main():
    proj = load_proj(None)
    proj = proj[proj.played.astype(bool)].copy()
    proj["player_id"] = proj["player_id"].astype(str)
    x = pd.concat([rungs(proj, s, "flag") for s in SEASONS], ignore_index=True)
    cl = pd.concat([rungs(proj, s, "close") for s in SEASONS], ignore_index=True)
    x = add_prior(x, proj)
    x = x[x.n_prior >= MIN_PRIOR]
    res = {"rungs": int(len(x)), "games": int(x.game_id.nunique()), "min_prior_games": MIN_PRIOR, "calibration": []}
    for _, g in x.groupby(pd.cut(x.prior, BANDS, right=False), observed=True):
        res["calibration"].append({"hit_rate_band": f"{g.prior.min():.2f}-{g.prior.max():.2f}", "rungs": int(len(g)),
                                   "hit_rate_going_in": round(float(g.prior.mean()), 4),
                                   "book_implied_raw": round(float(g.raw.mean()), 4), "book_fair": round(float(g.fair.mean()), 4),
                                   "actual": round(float(g.hit.mean()), 4),
                                   "actual_se": round(_se(g.hit - g.hit.mean(), g.game_id), 4)})
    # the bets: highest 90%+ rung per player-game-stat, better book's price
    c = cl[["game_id", "player_id", "market", "line", "book", "dec"]].rename(columns={"dec": "close_dec"})
    cand = x[x.prior >= 0.9].merge(c, on=["game_id", "player_id", "market", "line", "book"], how="left")
    cand["clv"] = cand.dec / cand.close_dec - 1
    best = (cand.sort_values(["line", "dec"], ascending=[False, False])
            .drop_duplicates(["game_id", "player_id", "stat"]))
    res["bets_90pct"] = grade(best)
    res["bets_90pct_edge5"] = grade(best[best.prior - best.raw >= 0.05])
    res["bets_95pct"] = grade(best[best.prior >= 0.95])
    res["bets_90pct_by_stat"] = {st: grade(g) for st, g in best.groupby("stat")}
    res["bets_90pct_by_season"] = {int(s): grade(g) for s, g in best.groupby("season")}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(res["calibration"]).to_string(index=False))
    for k in ("bets_90pct", "bets_90pct_edge5", "bets_95pct"):
        print(k, res[k])
    print(pd.DataFrame(res["bets_90pct_by_stat"]).T.to_string())
    print(pd.DataFrame(res["bets_90pct_by_season"]).T.to_string())


if __name__ == "__main__":
    main()
