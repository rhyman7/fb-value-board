"""Shadow test of the defensive-player inputs (Oct 2026; FEATURES.md, "Shadow test").

The defensive candidates that passed on past seasons were all within noise, so instead of
going on the board they are run alongside it on every refresh and logged, and judged later
on games nobody had seen when the rule was set. Nothing here reaches the page, the bets or
the props.

Pre-registered (Oct 29, 2026); not to be changed while the test runs:
  margin   the current model + d_chg_pr_pfr_bk_all + d_chg_cov_db_bk_all: pass rush (PFR
           pressures) and coverage (yards per coverage snap) lost to absences beyond the last
           four games, each defender against his group's backup level (def_ratings.py)
  total    the current model + s_chg_mt_dc_all: missed-tackle rate lost, against the
           depth-chart replacement
  players  receiving yards + the opponent's missed-tackle rate and yards per target allowed;
           interceptions + the opponent's pressure rate (defense.py)
Rule: a shadow version replaces the board's only if its average miss is lower on the rest of
2026 (week 9 on) and, separately, on 2027. experiments/shadow_score.py scores it.

Logs (data/shadow/, written by the Update board Action):
  defense_games.csv    one row per game: raw model margin and total, current and shadow
  defense_players.csv  one row per player-game: receiving yards and interceptions
A game's rows are rewritten on every refresh until kickoff and never after, as the archive.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C, def_ratings, defense, model, players as PL
from .archive import kickoff

MARGIN = ["d_chg_pr_pfr_bk_all", "d_chg_cov_db_bk_all"]
TOTAL = ["s_chg_mt_dc_all"]
PLAYER = {"rec_yds": ["opp_def_mt", "opp_def_ypt"], "ints": ["opp_def_press"]}
DIR = C.DATA / "shadow"
GAMES_LOG, PLAYERS_LOG = DIR / "defense_games.csv", DIR / "defense_players.csv"


def game_features(frame: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Home minus away (d_chg_*) and home plus away (s_chg_*) of each team's change: missing
    now less what its last four games reflect. 0 where unknown."""
    T = T.copy()
    keys = sorted({c[6:] for c in MARGIN + TOTAL})        # e.g. pr_pfr_bk_all
    for k in keys:
        T["chg_" + k] = T["miss_" + k] - T["mrec_" + k]
    cols = ["chg_" + k for k in keys]
    f = frame.drop(columns=[c for c in frame.columns if c[2:] in cols and c[:2] in ("d_", "s_")])
    for side, team in (("h", "home_team"), ("a", "away_team")):
        f = f.merge(T[["game_id", "team"] + cols].rename(columns={"team": team, **{c: f"{side}_{c}" for c in cols}}),
                    on=["game_id", team], how="left")
    for c in cols:
        h, a = f.pop(f"h_{c}").fillna(0.0), f.pop(f"a_{c}").fillna(0.0)
        f["d_" + c], f["s_" + c] = h - a, h + a
    return f


def _merge_log(path, new: pd.DataFrame, frozen: set) -> int:
    """Keep every row of a game that has kicked off; replace the rest with `new`."""
    old = pd.read_csv(path, dtype={"player_id": str}) if path.exists() else pd.DataFrame(columns=new.columns)
    keep = old[old["game_id"].isin(frozen)]
    add = new[~new["game_id"].isin(frozen)]
    out = pd.concat([keep, add], ignore_index=True).sort_values([c for c in ("game_id", "player_id") if c in new.columns])
    DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return int(add["game_id"].nunique())


def run(wf_input: pd.DataFrame, frame: pd.DataFrame, games: pd.DataFrame, tg: pd.DataFrame, ptable: pd.DataFrame | None, pproj: pd.DataFrame | None,
        season: int, refresh: bool = True, verbose: bool = True, now: pd.Timestamp | None = None) -> None:
    """wf_input: the frame the board's game model is fitted on; frame: its walk-forward output."""
    now = now or pd.Timestamp(dt.datetime.now(dt.timezone.utc))
    sched = games.set_index("game_id")
    ko = {gid: kickoff(r.gameday, r.gametime) for gid, r in sched[sched.season == season].iterrows()}
    frozen = {gid for gid, t in ko.items() if t <= now}
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")

    # ---- game model --------------------------------------------------------------
    T = def_ratings.load_team_def(games, refresh_current=refresh, verbose=verbose)
    f = game_features(wf_input, T)
    sh, _ = model.walk_forward(f, margin_feats=model.MARGIN_FEATS + MARGIN, total_feats=model.TOTAL_FEATS + TOTAL)
    up = frame[(frame.season == season) & ~frame.played & ~frame.game_id.isin(frozen)]
    s = sh.set_index("game_id").loc[up.game_id]
    g = pd.DataFrame({"game_id": up.game_id.to_numpy(), "season": up.season.to_numpy(), "week": up.week.to_numpy(),
                      "kickoff": [ko[x].strftime("%Y-%m-%dT%H:%MZ") for x in up.game_id], "logged_at": stamp,
                      "spread_line": up.spread_line.to_numpy(), "total_line": up.total_line.to_numpy(),
                      "margin_current": up.model_margin.round(3).to_numpy(), "margin_shadow": s.model_margin.round(3).to_numpy(),
                      "total_current": up.model_total.round(3).to_numpy(), "total_shadow": s.model_total.round(3).to_numpy()})
    for c in MARGIN + TOTAL:
        g[c] = s[c].round(5).to_numpy()
    n = _merge_log(GAMES_LOG, g, frozen)

    # ---- player projections --------------------------------------------------------
    m = 0
    if ptable is not None and pproj is not None:
        rates = defense.team_rates(games, tg, defense.load(refresh_current=refresh, verbose=verbose))
        saved = {k: list(v) for k, v in PL.CAL_EXTRA.items()}
        try:
            PL.CAL_EXTRA.update({k: v + PLAYER.get(k, []) for k, v in saved.items()})
            sp = PL.walk_forward(defense.attach(ptable, rates), first_test=season)
        finally:
            PL.CAL_EXTRA.clear()
            PL.CAL_EXTRA.update(saved)
        key = ["game_id", "player_id"]
        cur = pproj[(pproj.season == season) & ~pproj.played.astype(bool) & (pproj.active == 1) & ~pproj.game_id.isin(frozen)]
        cur = cur[(cur["targets_proj"] >= 1.5) | ((cur["group"] == "QB") & (cur["att_share"] >= 0.5))]
        sp = sp.set_index(key)
        rows = cur[key + ["team", "opp", "group"]].copy()
        rows["logged_at"] = stamp
        for st in PLAYER:
            rows[st + "_current"] = cur[st + "_proj"].round(3).to_numpy()
            rows[st + "_shadow"] = sp.reindex(pd.MultiIndex.from_frame(cur[key]))[st + "_proj"].round(3).to_numpy()
        m = _merge_log(PLAYERS_LOG, rows, frozen)
    if verbose:
        print(f"  shadow test (defense): {n} games, {m} games of player rows logged")
