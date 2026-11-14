"""Where the college rushing-yards gap to the books lives, on today's model, 2024 and 2025 lines.

    python experiments/cfb_rush_gap_diag.py        (PFF data at ../pff-data, as the board)

Read-only diagnostic (Nov 13, 2026), no model change. Follows experiments/cfb_props_gap_diag.py (Nov 3,
2025 lines only, NFL widths) on the board as it stands on Nov 13: PFF inputs, college prop widths
(data/cfb/prop_params.json), two seasons of DraftKings/FanDuel main lines ten minutes before kickoff,
game-day projections (a player who appeared counts as active) and bet-time projections (active =
expected to play: a stat in the team's previous game).

Each prop's share of the gap is (model - hit)^2 - (books - hit)^2 (positive: the books did better).
Splits (post hoc = uses who actually appeared, which the books knew at kickoff and the model did not):
  group        RB, QB, other
  distance     |log(projection / line)|: < 10%, 10-30%, > 30%
  line         the line: < 40, 40-69.5, 70+ yards
  week         1-2, 3-5, 6-9, 10+
  history      the player's decayed games in the walk-forward history (one_sum): < 1.5, 1.5-3.5, 3.5+
  new_team     on a different FBS team than last season (transfer), or not
  role_change  carry share last game vs his decayed average: up 10+ points, down 10+, steady
  expected     expected by the model's rule or not (post hoc)
  teammate     an expected regular back (10%+ of carries) did not appear (post hoc)
  script       the team's projected margin: underdog 7+, within 7, favourite 7+
  bias         projection minus actual, line minus actual (yards), by group; the model's average
               chance of the over against the share that cleared
Plus the reliability of the model's chance by band, the share of props where model and books disagree
by 5+ points and how often the model's side won, and the best blend weight toward the model per
season (books + w x (model - books), w on a grid; a description, not a fit to apply).

Writes experiments/out/cfb_rush_gap_diag.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from cfbmodel import players as PL  # noqa: E402
from nflmodel import odds as O, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = HERE / "out" / "cfb_rush_gap_diag.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
BANDS = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]


def lines(season, games):
    R.set_season(season)
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    m = s[~s["market"].str.endswith("_alternate") & (s["stat"] == "rush_yds")]
    w = m.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    return w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})


def context(t: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, player_id): new team this season, carry-share change, teammate out (post hoc)."""
    t = t.sort_values(["player_id", "t"]).copy()
    last_team = t[t["played"].astype(bool) & (t["appeared"] == 1)].groupby(["player_id", "season"])["team"].agg(lambda s: s.iloc[-1]).rename("team_last").reset_index()
    last_team["season"] = last_team["season"] + 1
    c = t[["game_id", "player_id", "team", "team_id", "season", "group", "expected", "appeared"]].merge(last_team, on=["player_id", "season"], how="left")
    c["new_team"] = np.where(c["team_last"].isna(), "no prior FBS season", np.where(c["team_last"] != c["team"], "new team", "same team"))
    # carry share in his last game vs his decayed average going in
    sh = t["carries"] / t["team_car"].where(t["team_car"] > 0)
    t["share_g"] = sh.where(t["played"].astype(bool) & (t["appeared"] == 1))
    t["share_last"] = t.groupby("player_id")["share_g"].transform(lambda s: s.ffill().shift())
    c = c.merge(t[["game_id", "player_id", "share_last"]], on=["game_id", "player_id"], how="left")
    # an expected regular back (10%+ carry share coming in) who did not appear
    reg = t[(t["expected"] == 1) & (t["car_share_raw"].fillna(0) >= 0.10) & (t["group"] == "RB") & t["played"].astype(bool)]
    out = reg[reg["appeared"] == 0].groupby(["game_id", "team_id"]).size().rename("n_out").reset_index()
    return c.drop(columns=["team", "group", "expected", "appeared"]).merge(out, on=["game_id", "team_id"], how="left")


def split(x, col):
    rows, total = [], x["d"].sum()
    for v, g in x.groupby(col, observed=True, dropna=False):
        cl = g.groupby("game_id")["d"].sum()
        se = float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(g)) if len(cl) > 1 else float("nan")
        rows.append({col: str(v), "props": int(len(g)), "share_of_props": round(len(g) / len(x), 3),
                     "gap_per_prop": round(float(g["d"].mean()), 5), "se": round(se, 5),
                     "share_of_gap": round(float(g["d"].sum() / total), 3) if total else None,
                     "model_minus_line_pct": round(float((g["proj"] / g["line"] - 1).median() * 100), 1),
                     "proj_minus_actual": round(float((g["proj"] - g["y"]).mean()), 1),
                     "line_minus_actual": round(float((g["line"] - g["y"]).mean()), 1),
                     "over_rate": round(float(g["hit"].mean()), 3)})
    return rows


def describe(x):
    cl = x.groupby("game_id")["d"].sum()
    r = {"props": int(len(x)), "games": int(x["game_id"].nunique()),
         "gap_per_prop": round(float(x["d"].mean()), 5), "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(x)), 5),
         "brier_model": round(float(((x["ours"] - x["hit"]) ** 2).mean()), 5), "brier_books": round(float(((x["fair"] - x["hit"]) ** 2).mean()), 5),
         "spread_model": round(float((x["ours"] - 0.5).abs().mean()), 4), "spread_books": round(float((x["fair"] - 0.5).abs().mean()), 4),
         "corr_model_books": round(float(np.corrcoef(x["ours"], x["fair"])[0, 1]), 3),
         "mean_model_chance": round(float(x["ours"].mean()), 4), "over_rate": round(float(x["hit"].mean()), 4),
         "mean_books_chance": round(float(x["fair"].mean()), 4),
         "by_group_chance_vs_cleared": {g: {"props": int(len(v)), "model": round(float(v["ours"].mean()), 4), "cleared": round(float(v["hit"].mean()), 4)}
                                        for g, v in x.groupby("group")}}
    for col in ("group", "distance", "line_band", "week_band", "history", "new_team", "role_change", "expected_by_model", "teammate_out", "script"):
        r[col] = split(x, col)
    xb = x.assign(band=pd.cut(x["ours"], BANDS, include_lowest=True))
    r["reliability"] = [{"model_band": str(b), "props": int(len(g)), "model_mean": round(float(g["ours"].mean()), 3),
                         "books_mean": round(float(g["fair"].mean()), 3), "cleared": round(float(g["hit"].mean()), 3)}
                        for b, g in xb.groupby("band", observed=True)]
    dis = x[(x["ours"] - x["fair"]).abs() >= 0.05]
    side = np.where(dis["ours"] > dis["fair"], dis["hit"], 1 - dis["hit"])
    r["disagree_5pt"] = {"props": int(len(dis)), "share": round(len(dis) / len(x), 3),
                         "model_side_won": round(float(side.mean()), 3) if len(dis) else None,
                         "model_side_over_share": round(float((dis["ours"] > dis["fair"]).mean()), 3) if len(dis) else None}
    grid = np.round(np.arange(-0.2, 1.01, 0.05), 2)
    bri = [float(((x["fair"] + w * (x["ours"] - x["fair"]) - x["hit"]) ** 2).mean()) for w in grid]
    r["best_blend_weight"] = {"w": float(grid[int(np.argmin(bri))]), "brier_at_best": round(min(bri), 5), "brier_books": r["brier_books"]}
    return r


def frame(pproj, pinfo, ctx, season, games, model):
    w = lines(season, games)
    R.set_season(season)
    rows = R.player_frame(pproj, pinfo)
    x = R.model_chances(w.merge(rows, on=["game_id", "nname"], how="inner"), model)
    x = x[x["y"] != x["line"]].copy()
    x = x.merge(ctx, on=["game_id", "player_id"], how="left")
    x["hit"] = (x["y"] > x["line"]).astype(float)
    x["ours"] = x["our_over"] / (x["our_over"] + x["our_under"])
    x["d"] = (x["ours"] - x["hit"]) ** 2 - (x["fair"] - x["hit"]) ** 2
    x["proj"] = x["rush_yds_proj"]
    x["group"] = np.where(x["group"].isin(["RB", "QB"]), x["group"], "other")
    x["distance"] = pd.cut(np.abs(np.log(np.maximum(x["proj"], 1) / x["line"])), [0, 0.1, 0.3, 10], labels=["<10%", "10-30%", ">30%"], include_lowest=True)
    x["line_band"] = pd.cut(x["line"], [0, 39.9, 69.9, 1000], labels=["<40", "40-69.5", "70+"])
    x["week_band"] = pd.cut(x["week"], [-1, 2, 5, 9, 30], labels=["1-2", "3-5", "6-9", "10+"])
    x["history"] = pd.cut(x["one_sum"].fillna(0), [-0.1, 1.5, 3.5, 1e9], labels=["decayed games < 1.5", "1.5-3.5", "3.5+"])
    ch = x["share_last"] - x["car_share_raw"]
    x["role_change"] = np.where(ch >= 0.10, "last game up 10+ pts", np.where(ch <= -0.10, "last game down 10+ pts", "steady"))
    x["expected_by_model"] = np.where(x["expected"] == 1, "expected", "not expected (post hoc)")
    x["teammate_out"] = np.where(x["n_out"].fillna(0) > 0, "regular back out (post hoc)", "no")
    mg = x["margin"] if "margin" in x else pd.Series(0.0, index=x.index)
    x["script"] = np.where(mg <= -7, "underdog 7+", np.where(mg >= 7, "favourite 7+", "within 7"))
    x["season"] = season
    return x


def main():
    t, pinfo, games = G.table()
    if not (t.get("route_share", 0) > 0).any():
        print("WARNING: no PFF data (../pff-data): the projections are not the board's")
    ctx = context(t)
    tb = t.copy()
    m = tb["played"].astype(bool)
    tb.loc[m, "active"] = tb.loc[m, "expected"].astype(int)
    model = P.PropModel.load(PARAMS)
    res = {}
    for kind, tab in (("game_day", t), ("bet_time", tb)):
        parts = []
        for season in (2024, 2025):
            pproj = PL.walk_forward(tab, first_test=season)
            parts.append(frame(pproj, pinfo, ctx, season, games, model))
        x = pd.concat(parts, ignore_index=True)
        res[kind] = {"both": describe(x), **{str(s): describe(g) for s, g in x.groupby("season")}}
        b = res[kind]["both"]
        print(kind, "props", b["props"], "gap", b["gap_per_prop"], "±", b["se"], "best w", b["best_blend_weight"], flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
