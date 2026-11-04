"""Where the college prop gap lives: rushing and receiving yards against the 2025 DraftKings and
FanDuel main lines, split by the kind of miss.

    python experiments/cfb_props_gap_diag.py

Read-only diagnostic (Nov 3, 2026), no model change. Same props, projections and chances as
experiments/cfb_props_real_lines.py (the board's walk-forward 2025 player table, the NFL prop
distributions). Each prop's contribution to the gap is (model - hit)^2 - (books - hit)^2; the
splits say which props carry it. Flags that use the game's outcome (who actually appeared) are
labelled post hoc: they measure what the books knew at kickoff and the model could not.

Splits:
  distance   |log(projection / line)|: under 10%, 10-30%, over 30%. A 30% disagreement on a
             yardage line is a volume or role disagreement, not efficiency.
  expected   the prop player was expected by the model's rule (played the team's previous
             game) or not (post hoc: he appeared this game but was not expected).
  teammate   an expected regular teammate at the same position (10%+ of the team's carries or
             catches) did not appear in this game (post hoc).
  group      position group.
  margin     |projected margin| under 14 or 14+.
  week       weeks 3-8 against 9-16.
  spread     mean |p - 0.5| of the model's chance and the books'; a model more extreme than
             the books that is not more often right has a distribution that is too narrow.
  reliability  cleared share by band of the model's chance, with the books' chance in the same band.

Writes experiments/out/cfb_props_gap_diag.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.cfb_props_real_lines as R  # noqa: E402
from nflmodel import odds as O, propmarket as PM, props as P  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "cfb_props_gap_diag.json"
CACHE = Path("/tmp/cfb_pproj_2025.pkl")
STATS = ("rush_yds", "rec_yds")
SHARE = {"rush_yds": "car_share", "rec_yds": "rec_share"}
BANDS = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]


def projections():
    if CACHE.exists():
        return (pd.read_pickle(CACHE), pd.read_pickle(CACHE.with_name("cfb_pinfo.pkl")),
                pd.read_pickle(CACHE.with_name("cfb_games.pkl")))
    pproj, pinfo, games = R.projections(cache=Path("/nonexistent"))
    pproj.to_pickle(CACHE)
    pinfo.to_pickle(CACHE.with_name("cfb_pinfo.pkl"))
    games.to_pickle(CACHE.with_name("cfb_games.pkl"))
    return pproj, pinfo, games


def main_lines(s: pd.DataFrame) -> pd.DataFrame:
    main = s[~s["alt"] & (s["market"] != "player_anytime_td")]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index()
    w = w.dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    return w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})


def teammate_out(pproj: pd.DataFrame) -> pd.Series:
    """Per (game_id, team_id, group): did an expected regular at that position not appear?"""
    p = pproj[(pproj.season == 2025) & pproj["played"].astype(bool)].copy()
    rows = []
    for stat, sh in SHARE.items():
        reg = p[(p["expected"] == 1) & (p[sh].fillna(0) >= 0.10)]
        out = reg[reg["appeared"] == 0].groupby(["game_id", "team_id", "group"]).size().rename("n_out").reset_index()
        out["stat"] = stat
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def split(x: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    total = x["d"].sum()
    for v, g in x.groupby(col, observed=True, dropna=False):
        cl = g.groupby("game_id")["d"].sum()
        se = float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(g)) if len(cl) > 1 else float("nan")
        rows.append({col: str(v), "props": int(len(g)), "share_of_props": round(len(g) / len(x), 3),
                     "gap_per_prop": round(float(g["d"].mean()), 5), "se": round(se, 5),
                     "share_of_gap": round(float(g["d"].sum() / total), 3) if total else None,
                     "brier_model": round(float(((g["ours"] - g["hit"]) ** 2).mean()), 5),
                     "brier_books": round(float(((g["fair"] - g["hit"]) ** 2).mean()), 5),
                     "model_mean_minus_line_pct": round(float((g["proj"] / g["line"] - 1).mean() * 100), 1),
                     "over_rate": round(float(g["hit"].mean()), 3)})
    return rows


def reliability(x: pd.DataFrame) -> list[dict]:
    x = x.assign(band=pd.cut(x["ours"], BANDS, include_lowest=True))
    rows = []
    for b, g in x.groupby("band", observed=True):
        rows.append({"model_band": str(b), "props": int(len(g)), "model_mean": round(float(g["ours"].mean()), 3),
                     "books_mean": round(float(g["fair"].mean()), 3), "cleared": round(float(g["hit"].mean()), 3)})
    return rows


def main():
    pproj, pinfo, games = projections()
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    rows = R.player_frame(pproj, pinfo)
    model = P.PropModel.load()
    w = main_lines(s)
    x = R.model_chances(w.merge(rows, on=["game_id", "nname"], how="inner"), model)
    x = x[(x["y"] != x["line"]) & x["stat"].isin(STATS)].copy()
    x["hit"] = (x["y"] > x["line"]).astype(float)
    x["ours"] = x["our_over"] / (x["our_over"] + x["our_under"])
    x["d"] = (x["ours"] - x["hit"]) ** 2 - (x["fair"] - x["hit"]) ** 2
    x["proj"] = np.where(x["stat"] == "rush_yds", x["rush_yds_proj"], x["rec_yds_proj"])
    x["distance"] = pd.cut(np.abs(np.log(np.maximum(x["proj"], 1) / x["line"])), [0, 0.1, 0.3, 10], labels=["<10%", "10-30%", ">30%"])
    x["expected_by_model"] = np.where(x["expected"] == 1, "expected", "not expected (post hoc)")
    to = teammate_out(pproj)
    x = x.merge(to, on=["game_id", "team_id", "group", "stat"], how="left")
    x["teammate_out"] = np.where(x["n_out"].fillna(0) > 0, "regular teammate out (post hoc)", "no")
    x["margin_band"] = np.where(x["proj_margin"].abs() >= 14, "|margin| 14+", "|margin| < 14") if "proj_margin" in x else "n/a"
    x["week_band"] = np.where(x["week"] <= 8, "weeks 3-8", "weeks 9-16")

    res = {"props": int(len(x)), "games": int(x["game_id"].nunique())}
    for stat, g in x.groupby("stat"):
        cl = g.groupby("game_id")["d"].sum()
        r = {"props": int(len(g)), "gap_per_prop": round(float(g["d"].mean()), 5),
             "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(g)), 5),
             "brier_model": round(float(((g["ours"] - g["hit"]) ** 2).mean()), 5),
             "brier_books": round(float(((g["fair"] - g["hit"]) ** 2).mean()), 5),
             "spread_model": round(float((g["ours"] - 0.5).abs().mean()), 4),
             "spread_books": round(float((g["fair"] - 0.5).abs().mean()), 4),
             "model_mean_minus_line_pct": round(float((g["proj"] / g["line"] - 1).mean() * 100), 1),
             "model_median_minus_line_pct": round(float((g["proj"] / g["line"] - 1).median() * 100), 1),
             "corr_model_books": round(float(np.corrcoef(g["ours"], g["fair"])[0, 1]), 3)}
        for col in ("distance", "expected_by_model", "teammate_out", "group", "margin_band", "week_band"):
            r[col] = split(g, col)
        r["reliability"] = reliability(g)
        # the model's side, when it disagrees with the books by 5+ points
        dis = g[(g["ours"] - g["fair"]).abs() >= 0.05]
        side = np.where(dis["ours"] > dis["fair"], dis["hit"], 1 - dis["hit"])
        r["disagree_5pt"] = {"props": int(len(dis)), "model_side_won": round(float(side.mean()), 3) if len(dis) else None,
                             "needed_to_tie_books": round(float(np.where(dis["ours"] > dis["fair"], dis["fair"], 1 - dis["fair"]).mean()), 3) if len(dis) else None}
        res[stat] = r
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
