"""PFF passing inputs for the NFL passing-yards projection (Nov 13, 2026).

Kept under the pre-set rule (FEATURES.md "The college PFF pairs on the NFL projections";
experiments/nfl_pff_pairs_test.py; Project docs claude/plan-nfl-pff-pairs-test-2026-11-13.md and
claude/nfl-pff-pairs-test-results-2026-11-13.md), provisional until the January recheck on the full
2026 season: passing-yards RMSE better on 2014-18 and 2019 on, game-day (3.0 SE holdout) and
bet-time (1.6 SE), and Brier at the 2023-25 DK/FD lines better at the close and the pricing snapshot.

  pff_pass_grade  the quarterback's PFF passing grade (passing_summary), dropback-weighted, shrunk
                  to the 2014-18 league average by 50 dropbacks, (grade - league) / league spread
  pff_opp_cov     the opponent's PFF coverage grade (team_overview, per game), shrunk by 2 games,
                  (grade - league) / league spread

As of each game from the games before it (8-game half-life, last season at 60%, nothing older), 0
with no history. Both enter players.CAL_EXTRA["pass_yds"] (raw projection x column), fitted
walk-forward like the other calibration terms. Players are matched by nflverse's players file
(pff_routes.pff_ids); team_overview franchises to board teams by the majority of their receivers'
board team that season.

The data is private (rhyman7/pff-data): these columns feed the calibration only; never write them,
or anything row by row from PFF, to docs/ or data/. Without the data, or if PFF is more than a week
behind the board's latest played week, both are 0 and the calibration gives them no weight: the
board is then what it was before Nov 13, 2026. team_overview is checked on its own: if only it is
stale, pff_opp_cov alone is 0 (it must be on pff-data's daily NFL pull to stay current).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HL, CARRY = 8.0, 0.6
LEAGUE = (2014, 2015, 2016, 2017, 2018)
PLAYOFF = {28: 19, 29: 20, 30: 21, 32: 22}
COLS = ("pff_pass_grade", "pff_opp_cov")


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


def _load(table: str, last: int) -> pd.DataFrame:
    from cfbmodel import pff_data
    p = pff_data.load(table, range(2014, last + 1), league="nfl")
    if p.empty:
        return p
    p["week"] = p["week"].astype(int).map(lambda w: PLAYOFF.get(w, w))
    p = p[p["week"].between(1, 22)].copy()
    p["season"] = p["season"].astype(int)
    return p


def _compute(table: pd.DataFrame) -> pd.DataFrame | None:
    from cfbmodel import pff_data
    from .pff_routes import pff_ids
    if not pff_data.available():
        return None
    last = int(table["season"].max())
    rec, pa, to = _load("receiving_summary", last), _load("passing_summary", last), _load("team_overview", last)
    if rec.empty or pa.empty or to.empty:
        return None
    keys = table[["game_id", "player_id", "team", "opp", "season", "week", "t", "played", "active"]].copy()
    played = keys[keys["played"].astype(bool) & (keys["active"] == 1)]
    # stale data misleads (pff_roles' college rule): PFF missing more than the latest played week
    s_last = int(played["season"].max())
    weeks_played = set(played.loc[played["season"] == s_last, "week"].astype(int))

    def stale(df):
        have = df.loc[(df["season"] == s_last) & df["_rows"], "week"]
        w_pff = int(have.max()) if len(have) else -1
        return len([w for w in weeks_played if w > w_pff]) > 1, w_pff

    pa["_rows"] = pd.to_numeric(pa.get("dropbacks"), errors="coerce").fillna(0) > 0
    to["_rows"] = pd.to_numeric(to.get("grades_coverage_defense"), errors="coerce").notna()
    pa_stale, w_pa = stale(pa)
    to_stale, w_to = stale(to)
    if pa_stale:
        print(f"  PFF passing inputs: passing data stale ({s_last} through week {w_pa}); left at 0")
        return None
    ids = pff_ids()
    pw = played[["player_id", "season", "week", "t"]].drop_duplicates()
    out = keys[["game_id", "player_id"]].copy()
    pa = pa.merge(ids, left_on="player_id", right_on="pff_id", how="inner")
    for c in ("grades_pass", "dropbacks"):
        pa[c] = pd.to_numeric(pa[c], errors="coerce")
    pa = pa.dropna(subset=["grades_pass", "dropbacks"])
    pa["g_db"] = pa["grades_pass"] * pa["dropbacks"]
    q = pa[pa.season.isin(LEAGUE) & (pa.dropbacks >= 15)]
    lg_g, sd_g = float(np.average(q.grades_pass, weights=q.dropbacks)), float(q.grades_pass.std())
    p = pa.groupby(["nfl_id", "season", "week"], as_index=False)[["g_db", "dropbacks"]].sum().rename(columns={"nfl_id": "player_id"})
    p = p.merge(pw, on=["player_id", "season", "week"])
    s = decayed(keys, "player_id", p, "player_id", ["g_db", "dropbacks"])
    out["pff_pass_grade"] = np.where(s[:, 1] > 0, (((s[:, 0] + 50 * lg_g) / (s[:, 1] + 50)) - lg_g) / sd_g, 0.0)
    rec = rec.merge(ids, left_on="player_id", right_on="pff_id", how="inner")
    link = rec[["season", "week", "franchise_id", "nfl_id"]].merge(
        keys[["player_id", "season", "week", "team"]].rename(columns={"player_id": "nfl_id"}), on=["nfl_id", "season", "week"])
    fm = link.groupby(["season", "franchise_id", "team"]).size().rename("n").reset_index().sort_values("n").drop_duplicates(["season", "franchise_id"], keep="last")
    to["grades_coverage_defense"] = pd.to_numeric(to["grades_coverage_defense"], errors="coerce")
    to = to.dropna(subset=["grades_coverage_defense"]).merge(fm[["season", "franchise_id", "team"]], on=["season", "franchise_id"])
    tg = keys[["game_id", "team", "opp", "season", "week", "t", "played"]].drop_duplicates(["game_id", "team"])
    to = to.merge(tg[tg["played"].astype(bool)][["team", "season", "week", "t"]].drop_duplicates(["team", "season", "week"]), on=["team", "season", "week"])
    to["one"] = 1.0
    lg = to[to.season.isin(LEAGUE)]
    m, sd = float(lg.grades_coverage_defense.mean()), float(lg.grades_coverage_defense.std())
    so = decayed(tg.assign(team=tg["opp"]), "team", to, "team", ["grades_coverage_defense", "one"])
    tg = tg.assign(pff_opp_cov=np.where(so[:, 1] > 0, (((so[:, 0] + 2 * m) / (so[:, 1] + 2)) - m) / sd, 0.0))
    look = dict(zip(zip(tg["game_id"], tg["team"]), tg["pff_opp_cov"]))
    out["pff_opp_cov"] = [look.get(k, 0.0) for k in zip(keys["game_id"], keys["team"])]
    if to_stale:                            # team_overview is not on pff-data's daily NFL pull by default
        print(f"  PFF passing inputs: team_overview stale ({s_last} through week {w_to}); opponent coverage left at 0")
        out["pff_opp_cov"] = 0.0
    return out


def add(table: pd.DataFrame) -> pd.DataFrame:
    """table with COLS (0 without the private data, or if anything about it fails)."""
    t = table.copy()
    for c in COLS:
        t[c] = 0.0
    try:
        f = _compute(table)
    except Exception as exc:                 # never a failed board over the private data
        print(f"  PFF passing inputs skipped ({exc})")
        f = None
    if f is None:
        return t
    for c in COLS:
        t[c] = f[c].to_numpy()               # same rows, same order as table
    cur = t[t["season"] == t["season"].max()]
    print(f"  PFF passing inputs: {(cur.loc[cur['group'] == 'QB', 'pff_pass_grade'] != 0).mean():.0%} of this season's QB rows "
          f"have a passing grade, {(cur['pff_opp_cov'] != 0).mean():.0%} an opponent coverage grade")
    return t
