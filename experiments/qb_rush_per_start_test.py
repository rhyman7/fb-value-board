"""A starting quarterback's rushing, judged on the games he started.

    python experiments/qb_rush_per_start_test.py

Nov 9, 2026 (PLAYERS.md "Quarterback rushing per start"). Found from one row: Carson Wentz, the
projected starter, at 0.6 carries against a book line of 2.5. The cause is general. A player's
share of the carries is averaged over the games he was active for, and a backup quarterback is
active every week without playing, so his share is mostly zeros. When he takes over, the starter
rule hands him every pass, and his passing form is already his per-start average, but his
rushing still uses the diluted share and the diluted per-game average. Since 2014, projected
starters with under 30% of their recent active games as starts were projected for 0.69 carries
and 3.9 yards and ran 3.04 times for 15.8 (320 starts); established starters are unbiased.

THE CHANGE ("per start"), for the projected starting quarterback only (att_share > 0.5), carries
and rushing yards only. It mirrors what passing already does:
  share      his share of the team's carries in the games he started: (decayed sum over starts +
             prior) / (decayed starts + 1), the prior being the average starter's share in earlier
             seasons. The player decay and season carry are the board's (players.asof).
  chain      carries = share x projected team carries; yards = carries x his yards per carry x the
             defense's, both as now
  form       his per-start average (carries, yards), in place of his per-active-game average; with
             under 0.3 decayed starts, the chain
  last step  the same small regression (players.Lin) on chain and form, fitted on projected
             starters who started, over the same earlier seasons the board's calibration uses
Every other row and stat keeps the board's projection exactly: nothing is refitted for them.
Walk-forward: a season sees only earlier ones.

MEASURES, written before the run. All eight must be better for "per start" than "current" for the
change to be kept (the keep rule, CONVENTIONS.md "Model tests"); mixed goes to Ryan. Rows:
projected starters in games they started.
  1-2  RMSE of rushing yards, on the board's box score: tuning 2014-2018, holdout 2019 on
  3-4  RMSE of carries, the same
  5-6  Brier at the DraftKings/FanDuel quarterback rushing-yards main lines, 2023-2025, graded on
       the official line (kneel-downs counted): pricing snapshot, close
  7-8  Brier at the quarterback rush-attempts main lines (2025), priced with the expected
       kneel-downs as the board now does (props.OFFICIAL_KEY) and graded on the official line:
       pricing snapshot, close
Widths are fitted on seasons before the one scored, for each version on its own projections; the
saved correction is the same in both. Standard errors: paired, clustered by game.

REPORTED ALONGSIDE, not part of the rule: both versions by how much of his recent active time
were starts (the change should be large for new starters and near nothing for established ones);
bias; 2026 alone; and passing yards by the same buckets, where the current projection looked
high for new starters (not changed here).

Writes experiments/out/qb_rush_per_start_test.json.
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
from nflmodel import kneels, players as PL, players_data as PD, propbackfill as B, props as P  # noqa: E402
import props_real_lines as RL  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "qb_rush_per_start_test.json"
TUNING, HOLDOUT_FROM, FIRST = (2014, 2018), 2019, 2014
LINE_SEASONS = (2023, 2024, 2025)
MARKETS = {"rush_yds": "player_rush_yds", "carries": "player_rush_attempts"}
BUCKETS = [-0.01, 0.3, 0.6, 0.85, 1.01]
BUCKET_NAMES = ["under 30% starts", "30-60%", "60-85%", "85%+ (established)"]


def per_start(proj: pd.DataFrame) -> pd.DataFrame:
    """The board's walk-forward table with the projected starters' carries and rushing yards
    replaced by the per-start version. Adds carries_cur / rush_yds_cur (the board's numbers)."""
    t = proj.copy()
    use = (t["active"] == 1) & t["played"].astype(bool)
    st = t["start"].where(use)
    share = np.where(t["team_car"] > 0, t["carries"] / t["team_car"], np.nan)
    t["car_share_st"] = pd.Series(share, index=t.index).where(use) * st
    t["carries_st"] = t["carries"].where(use) * st
    t["rush_yds_st"] = t["rush_yds"].where(use) * st
    a = PL.asof(t[["game_id", "player_id", "t", "season", "car_share_st", "carries_st", "rush_yds_st"]], "player_id",
                ["car_share_st", "carries_st", "rush_yds_st"], None, PL.PLAYER_HL)
    t = t.merge(a[["game_id", "player_id", "car_share_st_sum", "carries_st_sum", "rush_yds_st_sum"]], on=["game_id", "player_id"], how="left")
    t["carries_cur"], t["rush_yds_cur"] = t["carries_proj"], t["rush_yds_proj"]
    qb1 = (t["group"] == "QB") & (t["att_share"] > 0.5)
    started = qb1 & (t["started"] == 1) & use
    for s in sorted(x for x in t["season"].unique() if x >= FIRST - 1):
        earlier = started & (t["season"] < s)
        if earlier.sum() < 200:
            continue
        prior = float(pd.Series(share, index=t.index)[earlier].mean())          # the average starter's share, earlier seasons
        rows = qb1 & (t["season"] >= s - 6) & (t["season"] <= s)
        d = t[rows].copy()
        sh = (d["car_share_st_sum"].fillna(0) + prior) / (d["start_sum"].fillna(0) + 1.0)
        chain_c = sh * d["proj_team_car"]
        chain_y = chain_c * d["ypc"] * d["def_ypc"]
        enough = d["start_sum"] > 0.3
        form_c = (d["carries_st_sum"] / d["start_sum"]).where(enough, chain_c)
        form_y = (d["rush_yds_st_sum"] / d["start_sum"]).where(enough, chain_y)
        fit = (d["season"] < s) & (d["started"] == 1) & (d["active"] == 1) & d["played"].astype(bool)
        cur = d["season"] == s
        for stat, chain, form in (("carries", chain_c, form_c), ("rush_yds", chain_y, form_y)):
            X = np.column_stack([chain.to_numpy(float), form.to_numpy(float)])
            m = PL.Lin(1.0).fit(X[fit.to_numpy()], d.loc[fit, stat].to_numpy(float))
            t.loc[d.index[cur.to_numpy()], stat + "_proj"] = np.clip(m.predict(X[cur.to_numpy()]), 0, None)
    return t


def paired(a, b, games) -> tuple[float, float]:
    d = a - b
    g = d.groupby(games).sum()
    return float(d.mean()), float(g.std(ddof=1) * np.sqrt(len(g)) / len(d)) if len(g) > 1 else float("nan")


def rmse_report(t: pd.DataFrame) -> dict:
    rows = (t["group"] == "QB") & (t["att_share"] > 0.5) & (t["started"] == 1) & t["played"].astype(bool) & (t["season"] >= FIRST)
    q = t[rows].copy()
    q["bucket"] = pd.cut((q["start_sum"] / q["one_sum"].replace(0, np.nan)).fillna(0), BUCKETS, labels=BUCKET_NAMES).astype(str)
    spans = {"tuning": q["season"].between(*TUNING), "holdout": q["season"] >= HOLDOUT_FROM, "2026": q["season"] == 2026}
    out = {}
    for stat in ("rush_yds", "carries"):
        out[stat] = {}
        for span, m in spans.items():
            g = q[m]
            e0, e1 = (g[stat + "_cur"] - g[stat]) ** 2, (g[stat + "_proj"] - g[stat]) ** 2
            d, se = paired(e1, e0, g["game_id"])
            out[stat][span] = {"n": int(len(g)), "current_rmse": round(float(np.sqrt(e0.mean())), 4), "per_start_rmse": round(float(np.sqrt(e1.mean())), 4),
                               "rmse_change": round(float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())), 4), "mse_change": round(d, 4), "mse_se": round(se, 4),
                               "current_bias": round(float((g[stat] - g[stat + "_cur"]).mean()), 3), "per_start_bias": round(float((g[stat] - g[stat + "_proj"]).mean()), 3)}
        out[stat]["by_start_history"] = {
            b: {"n": int(len(g)), "actual": round(float(g[stat].mean()), 2), "current": round(float(g[stat + "_cur"].mean()), 2),
                "per_start": round(float(g[stat + "_proj"].mean()), 2),
                "current_rmse": round(float(np.sqrt(((g[stat + "_cur"] - g[stat]) ** 2).mean())), 3),
                "per_start_rmse": round(float(np.sqrt(((g[stat + "_proj"] - g[stat]) ** 2).mean())), 3)}
            for b, g in q.groupby("bucket")}
    out["pass_yds_by_start_history"] = {b: {"n": int(len(g)), "actual": round(float(g["pass_yds"].mean()), 1), "current": round(float(g["pass_yds_proj"].mean()), 1)}
                                        for b, g in q.groupby("bucket")}
    return out


def lines_report(t: pd.DataFrame, kind: str) -> dict:
    """Brier at the real quarterback rushing lines, both versions graded on the official line."""
    saved = P.PropModel.load()
    cur = t.copy()
    cur["carries_proj"], cur["rush_yds_proj"] = t["carries_cur"], t["rush_yds_cur"]
    V = {"current": cur, "per_start": t}
    off = t.assign(o_rush_yds=t["rush_yds"] + t["kneel_yds"], o_carries=t["carries"] + t["kneels"])[["game_id", "player_id", "o_rush_yds", "o_carries"]]
    priced = {}
    for name, v in V.items():
        parts = []
        for season in LINE_SEASONS:
            snaps = B.load(season, kind)
            snaps = snaps[snaps["market"].isin(MARKETS.values())]
            if snaps.empty:
                continue
            m = P.PropModel(FAMILY).fit(v[(v.season < season) & (v.season >= 2014)], stats=["carries", "rush_yds"])
            m.iso = saved.iso
            x = RL.priced(v, m, snaps, season)
            if len(x):
                parts.append(x[x["group"] == "QB"])
        priced[name] = pd.concat(parts, ignore_index=True)
    key = ["game_id", "player_id", "market", "line", "book"]
    out = {}
    for stat in MARKETS:
        a = priced["current"][priced["current"]["stat"] == stat].merge(off, on=["game_id", "player_id"], how="left")
        b = priced["per_start"][priced["per_start"]["stat"] == stat][key + ["our_over", "our_under"]].rename(columns={"our_over": "o2", "our_under": "u2"})
        j = a.merge(b, on=key, how="inner")
        if j.empty:
            continue
        j = j[j[f"o_{stat}"] != j["line"]]
        hit = (j[f"o_{stat}"] > j["line"]).astype(float)
        p0, p1 = j["our_over"] / (j["our_over"] + j["our_under"]), j["o2"] / (j["o2"] + j["u2"])
        d, se = paired((p1 - hit) ** 2, (p0 - hit) ** 2, j["game_id"])
        out[stat] = {"n": int(len(j)), "games": int(j["game_id"].nunique()), "over_rate": round(float(hit.mean()), 4),
                     "books_fair_over": round(float(j["fair_over"].mean()), 4), "brier_books": round(float(((j["fair_over"] - hit) ** 2).mean()), 5),
                     "current": {"brier": round(float(((p0 - hit) ** 2).mean()), 5), "our_over": round(float(p0.mean()), 4)},
                     "per_start": {"brier": round(float(((p1 - hit) ** 2).mean()), 5), "our_over": round(float(p1.mean()), 4)},
                     "brier_change": round(d, 5), "se": round(se, 5)}
    return out


def prepared(proj: pd.DataFrame) -> pd.DataFrame:
    pg, _ = PD.load_player_games(verbose=False)
    t = per_start(proj)
    t = kneels.project(kneels.attach(t, pg))
    ky = pg[["game_id", "player_id", "kneel_yds"]].drop_duplicates(["game_id", "player_id"])
    t = t.merge(ky, on=["game_id", "player_id"], how="left")
    t["kneel_yds"] = t["kneel_yds"].fillna(0.0).where(t["group"] == "QB", 0.0)
    return t


def main():
    rep = {}
    close = prepared(load_proj(None))
    rep["rmse"] = rmse_report(close)
    print(json.dumps(rep["rmse"], indent=1), flush=True)
    rep["lines"] = {"close": lines_report(close, "close")}
    print("close", json.dumps(rep["lines"]["close"], indent=1), flush=True)
    rep["lines"]["pricing"] = lines_report(prepared(RL.load_pregame()), "flag")
    print("pricing", json.dumps(rep["lines"]["pricing"], indent=1), flush=True)
    m = [rep["rmse"][st][sp]["mse_change"] < 0 for st in ("rush_yds", "carries") for sp in ("tuning", "holdout")]
    m += [rep["lines"][sn][st]["brier_change"] < 0 for st in ("rush_yds", "carries") for sn in ("pricing", "close")]
    rep["measures_better"], rep["kept_by_rule"] = int(sum(m)), bool(all(m))
    print("measures better:", rep["measures_better"], "of 8; kept by the rule:", rep["kept_by_rule"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
