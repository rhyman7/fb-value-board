"""College running backs: usage by down type, and player PPA, from CFBD (candidate 4).

    python experiments/cfb_usage_ppa_test.py

Nov 3, 2026. Data: data/cfb/cfbd/usage_<year>.csv (GET /player/usage: a player's share of his
team's plays overall and on standard vs passing downs, per season) and ppa_<year>.csv
(GET /ppa/players/games: per player per game average PPA, rushing and passing). Pulled by the
CFBD player data Action's `usage` mode. Player ids are the board's ESPN ids.

Two candidates, each a new column multiplied into the last calibration step (cfbmodel.players
CAL_EXTRA: the regression gets raw x column next to the chain and the recent average, fitted
walk-forward on earlier seasons, so a useless column gets a near-zero weight):

(a) Game script by role. A back whose usage comes on passing downs keeps his carries when his
    team trails; an early-down back loses them. Column `usage_script` = (passing-downs usage -
    standard-downs usage, from his PREVIOUS season, 0 if none) x the game's projected margin (`margin`, from the team's side)
    (positive = his team favored). Added to carries and rushing yards.
(b) PPA as the efficiency prior. Column `ppa_rush` = the back's decayed average rushing PPA per
    play over his earlier games (8-game half-life, last season at 60%, centred on the season's
    RB average; 0 with no history). Added to rushing yards.

Rules, written before running (memo, Nov 3):
  (a) RB carries RMSE better on tuning (2017-19) and holdout (2021+), holdout by 2+ SE; rushing-yards
      Brier at the 2025 DK/FD lines better by 2+ SE.
  (b) RB rushing-yards RMSE better on both, holdout by 2+ SE; Brier at the 2025 lines no worse.
  Rows: backs who appeared, chain or projection >= 20 yards / 4 carries. Each kept on its own.

Writes experiments/out/cfb_usage_ppa_test.json.
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
warnings.filterwarnings("ignore")
from cfbmodel import config as C, players as PL  # noqa: E402
import cfb_dvp_test as DV  # noqa: E402
import cfb_role_prior_test as RP  # noqa: E402
import cfb_rush_group_calib_test as G  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "cfb_usage_ppa_test.json"
CFBD = C.DATA / "cfbd"
TUNE = [2017, 2018, 2019]
HL, CARRY = 8.0, 0.6


def usage_prev(seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        f = CFBD / f"usage_{s - 1}.csv"
        if not f.exists():
            continue
        u = pd.read_csv(f)
        js = u["usage"].map(lambda v: json.loads(v) if isinstance(v, str) else {})
        rows.append(pd.DataFrame({"season": s, "player_id": pd.to_numeric(u["id"], errors="coerce"),
                                  "pd_diff": js.map(lambda d: (d.get("passingDowns") or 0) - (d.get("standardDowns") or 0)),
                                  "overall_prev": js.map(lambda d: d.get("overall") or 0)}))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype(np.int64)
    return out.drop_duplicates(["season", "player_id"])


def ppa_games(seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        f = CFBD / f"ppa_{s}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        js = p["averagePPA"].map(lambda v: json.loads(v) if isinstance(v, str) else {})
        rows.append(pd.DataFrame({"season": s, "week": p["week"], "player_id": pd.to_numeric(p["id"], errors="coerce"),
                                  "position": p["position"], "ppa_r": js.map(lambda d: d.get("rush")), "ppa_p": js.map(lambda d: d.get("pass"))}))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype(np.int64)
    return out


def ppa_asof(t: pd.DataFrame, pg: pd.DataFrame) -> pd.Series:
    """Decayed average of a back's earlier games' rushing PPA (centred on that season's RB mean),
    as of each row of the table, by (season, week) order."""
    pg = pg[pg["ppa_r"].notna()].copy()
    cen = pg[pg["position"] == "RB"].groupby("season")["ppa_r"].mean()
    pg["c"] = pg["ppa_r"] - pg["season"].map(cen).fillna(pg["ppa_r"].mean())
    pg["t"] = pg["season"] * PL.SEASON_LEN + pg["week"]
    pg = pg.sort_values(["player_id", "t"])
    # for each table row, decayed mean over the player's games with t < row t
    keyed = {pid: (g["t"].to_numpy(), g["c"].to_numpy(), g["season"].to_numpy()) for pid, g in pg.groupby("player_id")}
    out = np.zeros(len(t))
    tt, pid_arr, ss = t["t"].to_numpy(), t["player_id"].to_numpy(), t["season"].to_numpy()
    for i in range(len(t)):
        h = keyed.get(pid_arr[i])
        if h is None:
            continue
        ht, hc, hs = h
        m = (ht < tt[i]) & (hs >= ss[i] - 1)
        if not m.any():
            continue
        k = np.arange(m.sum())[::-1]
        w = 0.5 ** (k / HL) * np.where(hs[m] == ss[i], 1.0, CARRY)
        out[i] = float((w * hc[m]).sum() / (w.sum() + 2.0))      # shrunk by two games
    return pd.Series(out, index=t.index)


def score(cur, new, stat, floor):
    key = ["game_id", "player_id"]
    m = cur[key + ["season", "group", "appeared", "played", stat, stat + "_p", stat + "_proj"]].merge(
        new[key + [stat + "_proj"]].rename(columns={stat + "_proj": "new"}), on=key)
    d = m[(m.group == "RB") & (m.appeared == 1) & m.played.astype(bool) & m[stat].notna() & ((m[stat + "_p"] >= floor) | (m[stat + "_proj"] >= floor))]
    return {per: DV.paired_rmse(x, stat, "new") for per, x in (("tuning 2017-19", d[d.season.isin(TUNE)]), ("holdout 2021+", d[d.season >= 2021]))}


def main():
    t, pinfo, games = G.table()
    seasons = sorted(int(s) for s in t["season"].unique())
    cur_path = Path("/tmp/cfb_pproj_all.pkl")
    cur = pd.read_pickle(cur_path) if cur_path.exists() else PL.walk_forward(t)
    # (a)
    up = usage_prev(seasons)
    ta = t.merge(up[["season", "player_id", "pd_diff"]], on=["season", "player_id"], how="left")
    ta["pd_diff"] = ta["pd_diff"].fillna(0.0)
    ta["usage_script"] = ta["pd_diff"] * ta["margin"].fillna(0.0) / 10.0
    cov_a = float((ta.loc[(ta.group == "RB") & (ta.season >= 2017), "pd_diff"] != 0).mean())
    saved = dict(PL.CAL_EXTRA)
    PL.CAL_EXTRA = {**saved, "carries": ["usage_script"], "rush_yds": ["usage_script"]}
    new_a = PL.walk_forward(ta)
    PL.CAL_EXTRA = saved
    res = {"a_usage_script": {"coverage_rb_rows_with_prev_usage": cov_a, "carries": score(cur, new_a, "carries", 4.0), "rush_yds": score(cur, new_a, "rush_yds", 20.0)}}
    res["a_usage_script"]["lines_2025"] = RP.score_lines(cur, new_a, pinfo, games, "rush_yds", "carries")
    ra = res["a_usage_script"]
    ra["kept"] = bool(ra["carries"]["holdout 2021+"]["z"] <= -2 and ra["carries"]["tuning 2017-19"]["mse_diff"] <= 0 and ra["lines_2025"]["z"] <= -2)
    print("(a) carries", {k: round(v["z"], 2) for k, v in ra["carries"].items()}, "rush", {k: round(v["z"], 2) for k, v in ra["rush_yds"].items()},
          "lines z", round(ra["lines_2025"]["z"], 2), "kept", ra["kept"], flush=True)
    # (b)
    pg = ppa_games(seasons)
    tb = t.copy()
    tb["ppa_rush"] = ppa_asof(tb, pg)
    cov_b = float((tb.loc[(tb.group == "RB") & (tb.season >= 2017) & (tb.appeared == 1), "ppa_rush"] != 0).mean())
    PL.CAL_EXTRA = {**saved, "rush_yds": ["ppa_rush"]}
    new_b = PL.walk_forward(tb)
    PL.CAL_EXTRA = saved
    res["b_ppa_rush"] = {"coverage_rb_rows_with_history": cov_b, "rush_yds": score(cur, new_b, "rush_yds", 20.0)}
    res["b_ppa_rush"]["lines_2025"] = RP.score_lines(cur, new_b, pinfo, games, "rush_yds", "carries")
    rb = res["b_ppa_rush"]
    rb["kept"] = bool(rb["rush_yds"]["holdout 2021+"]["z"] <= -2 and rb["rush_yds"]["tuning 2017-19"]["mse_diff"] <= 0 and rb["lines_2025"]["new_minus_current"] <= 0)
    print("(b) rush", {k: round(v["z"], 2) for k, v in rb["rush_yds"].items()}, "lines", round(rb["lines_2025"]["new_minus_current"], 5), "z", round(rb["lines_2025"]["z"], 2), "kept", rb["kept"])
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
