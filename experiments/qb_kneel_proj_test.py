"""Quarterback rushing projections on the official line: add the kneel-downs he is expected to take.

    python experiments/qb_kneel_proj_test.py

Nov 9, 2026 (PROPS.md "Kneel-downs and rushing props"). The books settle a quarterback's rushing
yards and rush attempts on the official box score, kneel-downs included; the projections are for
his runs without them (players_data.SETTLE). This tests moving the two projections to the official
line by adding the kneels the starter is expected to take, and nothing else.

THE CHANGE ("adjusted"). For the projected starting quarterback (att_share > 0.5):
    carries_proj  += k
    rush_yds_proj += y
where k = a + b * wp is his expected kneel-downs and y = c * k their yards; wp is his team's
chance of winning from the game model's projected margin, Phi(margin / 13.5); a, b and c are
least-squares fits on projected starters' starts in every earlier season (2012 on), k floored at 0.
Walk-forward: a season only sees earlier ones. No other projection and no other position moves.

MEASURES, written before the run. All eight must be better for "adjusted" than for the board as
it is ("current") for the change to be kept (the keep rule, CONVENTIONS.md "Model tests"); a mixed
result goes to Ryan. Everything is graded on the official line (rush_yds + kneel_yds, carries +
kneels), for projected starters in games they started:
  1-2  RMSE of rushing yards: tuning 2014-2018, holdout 2019 on
  3-4  RMSE of carries: tuning, holdout
  5-6  Brier at the DraftKings/FanDuel quarterback rushing-yards main lines, 2023-2025 pooled:
       the pricing snapshot (pregame projections) and the close (game-day projections)
  7-8  Brier at the quarterback rush-attempts main lines (2025, the only season with them):
       pricing snapshot and close
For the lines, "current" is the board today: distribution widths fitted on the kneel-less box
score, the kneel-less mean. "adjusted" has the adjusted mean and widths refitted against the
official line. Widths and the kneel fit use only seasons before the one scored; the saved
isotonic correction is the same in both. Standard errors: paired, clustered by game.

REPORTED ALONGSIDE, not part of the rule:
  "flat"   the same with k = the earlier seasons' average kneels for a starter (no wp)
  "refit"  the current mean with widths refitted on the official line (how much is the widths)
  bias (official minus projected) for each version; over rate against the lines
  the 2026 season on its own

Writes experiments/out/qb_kneel_proj_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import config as C, players_data as PD, propbackfill as B, props as P  # noqa: E402
import props_real_lines as RL  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "qb_kneel_proj_test.json"
TUNING, HOLDOUT_FROM, FIRST_SCORED = (2014, 2018), 2019, 2014
LINE_SEASONS = (2023, 2024, 2025)
SD = 13.5
MARKETS = {"rush_yds": "player_rush_yds", "carries": "player_rush_attempts"}


def with_kneels(table: pd.DataFrame) -> pd.DataFrame:
    """The walk-forward table with each player-game's kneels and kneel yards from the box scores."""
    pg, _ = PD.load_player_games(verbose=False)
    k = pg[["game_id", "player_id", "kneels", "kneel_yds"]]
    t = table.merge(k, on=["game_id", "player_id"], how="left")
    t[["kneels", "kneel_yds"]] = t[["kneels", "kneel_yds"]].fillna(0.0)
    t["wp"] = norm.cdf(t["margin"].fillna(0.0) / SD)
    t["qb1"] = (t["group"] == "QB") & (t["att_share"] > 0.5)
    return t


def kneel_fit(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Expected kneels and kneel yards for projected starters, walk-forward: columns k, y (the wp
    version) and k_flat, y_flat (the earlier seasons' averages)."""
    t = t.copy()
    for c in ("k", "y", "k_flat", "y_flat"):
        t[c] = 0.0
    starts = t["qb1"] & (t["started"] == 1) & t["played"].astype(bool)
    info = {}
    for s in sorted(x for x in t.season.unique() if x >= FIRST_SCORED):
        tr = t[starts & (t.season < s)]
        X = np.c_[np.ones(len(tr)), tr["wp"]]
        a, b = np.linalg.lstsq(X, tr["kneels"].to_numpy(float), rcond=None)[0]
        c = float(tr["kneel_yds"].sum() / tr["kneels"].sum())
        m = t["qb1"] & (t.season == s)
        k = (a + b * t.loc[m, "wp"]).clip(lower=0.0)
        t.loc[m, "k"], t.loc[m, "y"] = k, c * k
        t.loc[m, "k_flat"] = float(tr["kneels"].mean())
        t.loc[m, "y_flat"] = c * float(tr["kneels"].mean())
        info[int(s)] = {"a": round(float(a), 4), "b": round(float(b), 4), "yards_per_kneel": round(c, 3),
                        "mean_kneels": round(float(tr["kneels"].mean()), 3), "starts": int(len(tr))}
    return t, info


def versions(t: pd.DataFrame) -> dict:
    """name -> table. "official" columns hold what the books settle on; each version's projections
    and (for the refitted ones) box-score columns are what the prop model is fitted and priced on."""
    qb = t["group"] == "QB"
    off = t.copy()                                   # official actuals for quarterbacks, current means
    off.loc[qb, "carries"] = t.loc[qb, "carries"] + t.loc[qb, "kneels"]
    off.loc[qb, "rush_yds"] = t.loc[qb, "rush_yds"] + t.loc[qb, "kneel_yds"]
    out = {"current": t, "refit": off}
    for name, kc, yc in (("adjusted", "k", "y"), ("flat", "k_flat", "y_flat")):
        v = off.copy()
        v["carries_proj"] = v["carries_proj"] + v[kc]
        v["rush_yds_proj"] = v["rush_yds_proj"] + v[yc]
        out[name] = v
    return out


def paired(a: pd.Series, b: pd.Series, games: pd.Series) -> tuple[float, float]:
    d = a - b
    g = d.groupby(games).sum()
    return float(d.mean()), float(g.std(ddof=1) * np.sqrt(len(g)) / len(d)) if len(g) > 1 else float("nan")


def rmse_report(V: dict, t: pd.DataFrame) -> dict:
    rows = t["qb1"] & (t["started"] == 1) & t["played"].astype(bool) & (t.season >= FIRST_SCORED)
    official = {"rush_yds": t["rush_yds"] + t["kneel_yds"], "carries": t["carries"] + t["kneels"]}
    spans = {"tuning": rows & t.season.between(*TUNING), "holdout": rows & (t.season >= HOLDOUT_FROM), "2026": rows & (t.season == 2026)}
    out = {}
    for stat, y in official.items():
        out[stat] = {}
        for span, m in spans.items():
            base = (V["current"].loc[m, f"{stat}_proj"] - y[m]) ** 2
            r = {"n": int(m.sum()), "current_rmse": round(float(np.sqrt(base.mean())), 4),
                 "current_bias": round(float((y[m] - V["current"].loc[m, f"{stat}_proj"]).mean()), 3)}
            for name in ("adjusted", "flat"):
                e = (V[name].loc[m, f"{stat}_proj"] - y[m]) ** 2
                d, se = paired(e, base, t.loc[m, "game_id"])          # mean squared error difference
                r[name] = {"rmse": round(float(np.sqrt(e.mean())), 4), "rmse_change": round(float(np.sqrt(e.mean()) - np.sqrt(base.mean())), 4),
                           "mse_change": round(d, 4), "mse_se": round(se, 4),
                           "bias": round(float((y[m] - V[name].loc[m, f"{stat}_proj"]).mean()), 3)}
            out[stat][span] = r
    return out


def lines_report(V: dict, t: pd.DataFrame, kind: str) -> dict:
    """Brier at the real quarterback rushing lines, every version graded on the official line."""
    saved = P.PropModel.load()
    off = (t.assign(o_rush_yds=t["rush_yds"] + t["kneel_yds"], o_carries=t["carries"] + t["kneels"])
           [["game_id", "player_id", "o_rush_yds", "o_carries"]])
    priced = {name: [] for name in V}
    for season in LINE_SEASONS:
        snaps = B.load(season, kind)
        snaps = snaps[snaps["market"].isin(MARKETS.values())]
        if snaps.empty:
            continue
        for name, v in V.items():
            m = P.PropModel(FAMILY).fit(v[(v.season < season) & (v.season >= 2014)])
            m.iso = saved.iso
            x = RL.priced(v, m, snaps, season)
            if len(x):
                priced[name].append(x[x["group"] == "QB"])
    out = {}
    key = ["game_id", "player_id", "market", "line", "book"]
    base = pd.concat(priced["current"], ignore_index=True).merge(off, on=["game_id", "player_id"], how="left")
    for stat in MARKETS:
        b = base[base["stat"] == stat].copy()
        if b.empty:
            continue
        b["y_off"] = b[f"o_{stat}"]
        b = b[b["y_off"] != b["line"]]
        hit = (b["y_off"] > b["line"]).astype(float)
        ours = b["our_over"] / (b["our_over"] + b["our_under"])
        r = {"n": int(len(b)), "games": int(b.game_id.nunique()), "over_rate_official": round(float(hit.mean()), 4),
             "over_rate_without_kneels": round(float((b["y"] > b["line"]).mean()), 4),
             "books_fair_over": round(float(b["fair_over"].mean()), 4), "brier_books": round(float(((b["fair_over"] - hit) ** 2).mean()), 5),
             "current": {"brier": round(float(((ours - hit) ** 2).mean()), 5), "our_over": round(float(ours.mean()), 4)}}
        for name in ("adjusted", "flat", "refit"):
            x = pd.concat(priced[name], ignore_index=True)
            x = x[x["stat"] == stat][key + ["our_over", "our_under"]].rename(columns={"our_over": "o2", "our_under": "u2"})
            j = b.merge(x, on=key, how="inner")
            h = (j["y_off"] > j["line"]).astype(float)
            cur = j["our_over"] / (j["our_over"] + j["our_under"])
            new = j["o2"] / (j["o2"] + j["u2"])
            d, se = paired((new - h) ** 2, (cur - h) ** 2, j["game_id"])
            r[name] = {"n": int(len(j)), "brier": round(float(((new - h) ** 2).mean()), 5), "brier_change": round(d, 5), "se": round(se, 5),
                       "our_over": round(float(new.mean()), 4)}
        out[stat] = r
    return out


def main():
    rep = {}
    tables = {"close": with_kneels(load_proj(None)), "pricing": with_kneels(RL.load_pregame())}
    t, info = kneel_fit(tables["close"])
    rep["kneel_fit"] = info
    V = versions(t)
    rep["rmse"] = rmse_report(V, t)
    print(json.dumps({"kneel_fit": {k: info[k] for k in list(info)[-3:]}, "rmse": rep["rmse"]}, indent=1), flush=True)
    rep["lines"] = {}
    for kind, name in (("flag", "pricing"), ("close", "close")):
        tk, _ = kneel_fit(tables[name])
        rep["lines"][name] = lines_report(versions(tk), tk, kind)
        print(name, json.dumps(rep["lines"][name], indent=1), flush=True)
    m = [rep["rmse"][st][sp]["adjusted"]["mse_change"] < 0 for st in ("rush_yds", "carries") for sp in ("tuning", "holdout")]
    m += [rep["lines"][sn][st]["adjusted"]["brier_change"] < 0 for st in ("rush_yds", "carries") for sn in ("pricing", "close")]
    rep["measures_better"] = int(sum(m))
    rep["kept_by_rule"] = bool(all(m))
    print("measures better:", rep["measures_better"], "of 8; kept by the rule:", rep["kept_by_rule"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
