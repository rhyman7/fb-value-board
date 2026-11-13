"""PFF skill and matchup inputs for the college player projections (Nov 13, 2026).

Kept under the pre-set rule (FEATURES.md "The other PFF college tables in the player projections";
experiments/cfb_pff_tables_test.py; Project docs claude/plan-cfb-pff-tables-test-2026-11-13.md and
claude/cfb-pff-tables-test-results-2026-11-13.md), provisional until the January recheck on the full
2026 season. Two pairs, each better on every measure (walk-forward RMSE 2017-19 and 2021-25, holdout
4.4-4.5 SE; Brier at the 2024 and 2025 main lines, game-day and bet-time):

  pff_yprr        receiver's yards per route run (receiving_summary; shrunk to the 2017-19 league
                  rate by 100 routes), as rate / league - 1                 -> rec_yds  (test: R1)
  pff_pass_grade  quarterback's PFF passing grade (passing_summary; dropback-weighted, shrunk by 50
                  dropbacks), (grade - league) / league spread              -> pass_yds (test: P1)
  pff_opp_cov     the opponent's PFF coverage grade (team_overview, per game; shrunk by 2 games),
                  (grade - league) / league spread                -> pass_yds, receptions, rec_yds
                                                                     (test: T5 and T6)

As of each game from earlier games only (8-game half-life, last season at 60%, nothing older), 0
with no history. League constants come from 2017-2019 only. Each enters the last calibration step
the way route and snap share do (players.CAL_EXTRA: raw projection x column), so the weights are
fitted walk-forward. PFF rows are matched to the board's games as in pff_roles.

The data is private (rhyman7/pff-data, read by cfbmodel.pff_data): these columns feed the calibration
only; never write them, or anything row by row from PFF, to docs/ or data/. Without the data (or with
stale data, the same rule as pff_roles) every column is 0 and the calibration gives it no weight: the
board is then what it was before Nov 13, 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pff_data, pff_roles

HL, CARRY = 8.0, 0.6
LEAGUE = (2017, 2018, 2019)
COLS = ("pff_yprr", "pff_pass_grade", "pff_opp_cov")


def decayed(t: pd.DataFrame, key: str, hist: pd.DataFrame, hkey: str, cols: list[str]) -> np.ndarray:
    """Decayed sums of hist[cols] over each row's key's earlier games (this season and last)."""
    h = hist.sort_values([hkey, "t"])
    keyed = {k: (g["t"].to_numpy(), g["season"].to_numpy(), g[cols].to_numpy(float)) for k, g in h.groupby(hkey)}
    out = np.zeros((len(t), len(cols)))
    tt, kk, ss = t["t"].to_numpy(), t[key].to_numpy(), t["season"].to_numpy()
    for i in range(len(t)):
        k = keyed.get(kk[i])
        if k is None:
            continue
        ht, hs, v = k
        sel = (ht < tt[i]) & (hs >= ss[i] - 1)
        if not sel.any():
            continue
        j = np.arange(sel.sum())[::-1]
        w = 0.5 ** (j / HL) * np.where(hs[sel] == ss[i], 1.0, CARRY)
        out[i] = (w[:, None] * np.nan_to_num(v[sel])).sum(axis=0)
    return out


def _on_board(table: str, t: pd.DataFrame, info: pd.DataFrame, cols: list[str], seasons) -> pd.DataFrame:
    x = pff_data.load(table, seasons)
    if x.empty:
        return x
    x = x[["season", "week", "player", "player_id", "franchise_id"] + [c for c in cols if c in x]].copy()
    x["nname"] = x["player"].map(pff_roles._norm)
    m = pff_roles.on_board(x, t, info)
    m = m.merge(t[["game_id", "t"]].drop_duplicates("game_id"), on="game_id")
    m["board_id"] = m["board_id"].astype(np.int64)
    return m


def _stale(t: pd.DataFrame, m: pd.DataFrame) -> bool:
    """pff_roles' rule: PFF missing more than the latest regular-season week the board has played."""
    done = t[t["played"].astype(bool) & t["proj_total"].notna() & (t["week"] <= 16)]
    if not len(done):
        return False
    s_last = int(done["season"].max())
    played_weeks = set(done.loc[done["season"] == s_last, "week"].astype(int))
    have = m.loc[m["season"] == s_last, "week"]
    w_pff = int(have.max()) if len(have) else -1
    return len([w for w in played_weeks if w > w_pff]) > 1


def add(t: pd.DataFrame, info: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    t = t.copy()
    for c in COLS:
        t[c] = 0.0
    if not pff_data.available():
        return t
    seasons = sorted(int(s) for s in t["season"].unique() if s >= 2017)
    try:
        rc = _on_board("receiving_summary", t, info, ["yards", "routes"], seasons)
        pa = _on_board("passing_summary", t, info, ["grades_pass", "dropbacks"], seasons)
        to = pff_data.load("team_overview", seasons)
    except Exception as e:                              # a missing table or column: no inputs, as without the data
        if verbose:
            print(f"  PFF skill inputs skipped: {e}")
        return t
    if rc.empty or pa.empty or to.empty:
        return t
    if _stale(t, rc):
        if verbose:
            print("  PFF data stale: skill inputs left at 0")
        return t
    # yards per route run
    lg = rc[rc.season.isin(LEAGUE)]
    lg_yprr = lg.yards.sum() / lg.routes.sum()
    s = decayed(t, "player_id", rc, "board_id", ["yards", "routes"])
    t["pff_yprr"] = np.where(s[:, 1] > 0, ((s[:, 0] + 100 * lg_yprr) / (s[:, 1] + 100)) / lg_yprr - 1, 0.0)
    # passing grade
    pa["g_db"] = pa["grades_pass"].fillna(0) * pa["dropbacks"].fillna(0)
    q = pa[pa.season.isin(LEAGUE) & (pa.dropbacks >= 15)]
    lg_g, sd_g = float(np.average(q.grades_pass, weights=q.dropbacks)), float(q.grades_pass.std())
    s = decayed(t, "player_id", pa, "board_id", ["g_db", "dropbacks"])
    t["pff_pass_grade"] = np.where(s[:, 1] > 0, (((s[:, 0] + 50 * lg_g) / (s[:, 1] + 50)) - lg_g) / sd_g, 0.0)
    # the opponent's coverage grade: franchise -> board team per season (the receiving mapping)
    fm = rc[["season", "franchise_id", "team"]].drop_duplicates(["season", "franchise_id"])
    to = to[["season", "week", "franchise_id", "grades_coverage_defense"]].merge(fm, on=["season", "franchise_id"])
    gm = t[t["played"].astype(bool)][["game_id", "team", "season", "week", "t"]].drop_duplicates(["team", "season", "week"])
    to = to.merge(gm, on=["team", "season", "week"])
    to["one"] = 1.0
    lg = to[to.season.isin(LEAGUE)]
    m, sd = float(lg.grades_coverage_defense.mean()), float(lg.grades_coverage_defense.std())
    tg = t[["game_id", "team", "opp", "season", "t"]].drop_duplicates(["game_id", "team"])
    so = decayed(tg.assign(team=tg["opp"]), "team", to, "team", ["grades_coverage_defense", "one"])
    tg = tg.assign(pff_opp_cov=np.where(so[:, 1] > 0, (((so[:, 0] + 2 * m) / (so[:, 1] + 2)) - m) / sd, 0.0))
    look = dict(zip(zip(tg["game_id"], tg["team"]), tg["pff_opp_cov"]))
    t["pff_opp_cov"] = [look.get(k, 0.0) for k in zip(t["game_id"], t["team"])]       # keeps t's index and order
    if verbose:
        cur = t[t["season"] == t["season"].max()]
        print(f"  PFF skill inputs: {(cur['pff_yprr'] != 0).mean():.0%} of this season's rows have yards per route run, "
              f"{(cur['pff_opp_cov'] != 0).mean():.0%} an opponent coverage grade")
    return t
