"""Team defensive rates from PFR's weekly defensive stats, for the player projections.

Tested in experiments/def_players_test.py and def_players_combined.py (FEATURES.md,
"Defensive players, re-tested and rated", Oct 2026). Three rates per defense, each decayed
over its earlier games (half-life 8 games, last season's games x 0.6), so a game only ever
sees what came before it:

  press   pressures (hurries + knockdowns + sacks) per opponent dropback
  ypt     yards allowed in coverage per target
  mt      missed tackles per tackle attempt (tackles + missed tackles)

The projections use each one against the league average of earlier seasons (in the first
season, that week's average over every team), as calibration terms: the opponent's missed-tackle rate and yards per
target in receiving yards, its pressure rate in interceptions (players.CAL_EXTRA). Seasons
before PFR has earlier games to rate from (2018 data, so 2019 on) get 0, no information.

Data: nflverse's PFR advanced stats, defense, weekly (2018 on); dropbacks faced from the
team play-by-play table. Cached per season in data/cache/def_team_<season>.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import _download, current_season

URL = "https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_week_def_{season}.parquet"
FIRST = 2018
HL, CARRY = 8.0, 0.6
RATES = {"press": ("pressures", "db_faced"), "ypt": ("yds_allowed", "targets"), "mt": ("missed", "tack_att")}
COLS = ["pressures", "yds_allowed", "targets", "missed", "tack_att"]


def _season(season: int) -> pd.DataFrame:
    raw = C.RAW / f"advstats_week_def_{season}.parquet"
    _download(URL.format(season=season), raw)
    a = pd.read_parquet(raw)
    raw.unlink(missing_ok=True)
    a["team"] = a["team"].replace(C.TEAM_ALIASES)
    a = a.rename(columns={"def_pressures": "pressures", "def_yards_allowed": "yds_allowed", "def_targets": "targets",
                          "def_missed_tackles": "missed"})
    a["tack_att"] = a["def_tackles_combined"].fillna(0) + a["missed"].fillna(0)
    t = a.groupby(["game_id", "team"], as_index=False)[COLS].sum(min_count=1)
    t.insert(0, "season", season)
    return t


def load(refresh_current: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Team-game sums of the PFR defensive stats, 2018 on."""
    now = current_season()
    out = []
    for season in range(FIRST, now + 1):
        path = C.CACHE / f"def_team_{season}.csv"
        if (season == now and refresh_current) or not path.exists():
            if verbose:
                print(f"  PFR defense {season} ...", flush=True)
            try:
                _season(season).to_csv(path, index=False)
            except Exception as exc:
                print(f"  no PFR defense for {season} ({exc})")
                continue
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["season", "game_id", "team"] + COLS)


def team_rates(games: pd.DataFrame, tg: pd.DataFrame, pfr: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team, def_press, def_ypt, def_mt, this week's league averages lg_* and the
    earlier seasons' ref_*) for every
    team-game on the schedule, played or not, from that defense's earlier games only."""
    from .players import _t, asof
    g = games[["game_id", "season", "week", "home_team", "away_team"]]
    sched = pd.concat([g.rename(columns={"home_team": "team", "away_team": "opp"}),
                       g.rename(columns={"away_team": "team", "home_team": "opp"})], ignore_index=True)
    sched["team"] = sched["team"].replace(C.TEAM_ALIASES)
    sched["opp"] = sched["opp"].replace(C.TEAM_ALIASES)
    sched = sched[sched["season"] >= FIRST - 1]
    faced = tg[["game_id", "team", "n_pass"]].rename(columns={"team": "opp", "n_pass": "db_faced"})
    t = sched.merge(faced, on=["game_id", "opp"], how="left").merge(pfr.drop(columns="season", errors="ignore"), on=["game_id", "team"], how="left")
    # a game with no PFR row (not played yet, or before 2018) adds nothing to the sums
    t.loc[t[COLS].isna().all(axis=1), "db_faced"] = np.nan
    t["t"] = _t(t["season"], t["week"])
    a = asof(t, "team", COLS + ["db_faced"], None, HL, CARRY)
    out = a[["game_id", "team", "season", "week"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (num, den) in RATES.items():
            v = a[num + "_sum"] / a[den + "_sum"]
            out["def_" + k] = v.where(a[den + "_sum"] > 0).where(a["season"] > FIRST).replace([np.inf, -np.inf], np.nan)
    for k in RATES:
        # shown on the page: this week's league average
        out["lg_" + k] = out.groupby(["season", "week"])["def_" + k].transform("mean")
        # used by the model: the league average of earlier seasons (this week's average in the
        # first season), so a league-wide change since then still counts and nothing comes
        # from later games
        by_season = out.groupby("season")["def_" + k].mean()
        earlier = {s: by_season[by_season.index < s].mean() for s in by_season.index}
        out["ref_" + k] = out["season"].map(earlier).fillna(out["lg_" + k])
    return out.drop(columns=["season", "week"])


def attach(out: pd.DataFrame, rates: pd.DataFrame | None) -> pd.DataFrame:
    """Add opp_def_<rate> (the opponent's rate minus the earlier seasons' league average, 0 where
    unknown) and, for display, opp_def_<rate>_raw and this week's league average opp_lg_<rate>,
    keyed on (game_id, opp)."""
    if rates is not None and len(rates):
        r = rates.rename(columns={"team": "opp", **{f"def_{k}": f"opp_def_{k}_raw" for k in RATES},
                                  **{f"lg_{k}": f"opp_lg_{k}" for k in RATES}, **{f"ref_{k}": f"opp_ref_{k}" for k in RATES}})
        out = out.merge(r, on=["game_id", "opp"], how="left")
    for k in RATES:
        if f"opp_def_{k}_raw" not in out:
            out[f"opp_def_{k}_raw"] = out[f"opp_lg_{k}"] = out[f"opp_ref_{k}"] = np.nan
        out[f"opp_def_{k}"] = (out[f"opp_def_{k}_raw"] - out[f"opp_ref_{k}"]).fillna(0.0)
    return out
