"""Missing starters by position group, from injury reports and snap counts (2012 on).

For each team and game:
  core players   the top-k players in each group by average snap share over the
                 team's earlier games this season (a game a player missed counts 0)
  unavailable    listed Out or Doubtful on this week's injury report, or not active
                 on this week's roster (injured reserve, game-day inactive, released).
                 For a game not yet played, the latest roster is used and only
                 longer-term lists count (inactives are named 90 minutes before kickoff)
  miss_<group>   snap-share-weighted share of the group's core that is unavailable
  mrec_<group>   the same measured on the team's last four games, i.e. what the
                 team's recent results (and so its ratings) already reflect

Groups: ol (5), pc = pass catchers WR/TE (3), rb (1), dl (4), lb (2), db (4).
Week 1 and games without snap data get zeros (no information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import _download, current_season

URL = "https://github.com/nflverse/nflverse-data/releases/download/{kind}/{kind}_{season}.parquet"
FIRST = 2012
GROUPS = {
    "ol": ({"T", "G", "C", "OL", "OT", "OG", "C/G", "G/T", "G/C", "T/G", "G/OT"}, 5, "offense_pct"),
    "pc": ({"WR", "TE", "WR/R", "TE/D"}, 3, "offense_pct"),
    "rb": ({"RB", "HB", "RB/W", "RB/F"}, 1, "offense_pct"),
    "dl": ({"DE", "DT", "DL", "NT", "DT/D", "DE/L", "DE/D"}, 4, "defense_pct"),
    "lb": ({"LB", "OLB", "ILB", "MLB", "LB/F", "LB/S"}, 2, "defense_pct"),
    "db": ({"CB", "S", "DB", "FS", "SS", "DB/L", "CB/R"}, 4, "defense_pct"),
}
RECENT = 4


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalpha())


def _season(season: int, sched: pd.DataFrame) -> pd.DataFrame:
    """sched: the nflverse games table (regular season used), for each team's game list."""
    paths = {k: C.RAW / f"{k}_{season}.parquet" for k in ("snap_counts", "injuries", "roster_weekly")}
    for k, path in paths.items():
        kind = "weekly_rosters" if k == "roster_weekly" else k
        if not path.exists():
            _download(URL.format(kind=kind, season=season).replace(f"/{kind}_{season}", f"/{k}_{season}"), path)
    s = pd.read_parquet(paths["snap_counts"])
    inj = pd.read_parquet(paths["injuries"])
    ros = pd.read_parquet(paths["roster_weekly"], columns=["team", "week", "full_name", "status", "game_type"])
    s = s[s["game_type"] == "REG"]
    ros = ros[ros["game_type"] == "REG"]
    for df in (s, inj, ros):
        df["team"] = df["team"].replace(C.TEAM_ALIASES)
    s["key"] = s["player"].map(_norm)
    inj["key"] = inj["full_name"].map(_norm)
    ros["key"] = ros["full_name"].map(_norm)
    report = inj.groupby(["team", "week", "key"])["report_status"].last().to_dict()
    roster = ros.groupby(["team", "week", "key"])["status"].first().to_dict()
    roster_weeks = ros.groupby("team")["week"].apply(lambda w: sorted(set(w))).to_dict()

    g = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    long = pd.concat([g[["game_id", "week", "home_team"]].rename(columns={"home_team": "team"}),
                      g[["game_id", "week", "away_team"]].rename(columns={"away_team": "team"})])
    played_weeks = set(s["week"])
    rows = []
    for team, tg in long.groupby("team"):
        tg = tg.sort_values("week")
        t = s[s["team"] == team]
        have = sorted(set(t["week"]))
        for gid, wk in zip(tg["game_id"], tg["week"]):
            row = {"season": season, "game_id": gid, "team": team}
            prior = [w for w in have if w < wk]
            # roster status for this week, or the latest earlier week for an upcoming game
            rw = [w for w in roster_weeks.get(team, []) if w <= wk]
            rweek = rw[-1] if rw else None
            exact = rweek == wk
            for grp, (pos, k, col) in GROUPS.items():
                row[f"miss_{grp}"] = row[f"mrec_{grp}"] = 0.0
                if not prior:
                    continue
                gg = t[t["position"].isin(pos)]
                share = gg[gg["week"].isin(prior)].pivot_table(index="key", columns="week", values=col, aggfunc="max")
                share = share.reindex(columns=prior).fillna(0.0)
                core = share.mean(axis=1).sort_values(ascending=False).head(k)
                core = core[core > 0.2]
                if core.empty:
                    continue
                out_now = []
                for key in core.index:
                    rep = report.get((team, wk, key))
                    st = roster.get((team, rweek, key)) if rweek is not None else "ACT"
                    if rep in ("Out", "Doubtful"):
                        out_now.append(key)
                    elif rweek is None:
                        continue
                    elif st is None:                       # no longer on the roster (released, traded)
                        out_now.append(key)
                    elif exact and st != "ACT":            # reserve lists, game-day inactive, suspended
                        out_now.append(key)
                    elif not exact and st not in ("ACT", "INA"):   # upcoming game: only longer-term lists
                        out_now.append(key)
                row[f"miss_{grp}"] = float(core[out_now].sum() / core.sum())
                rec = prior[-RECENT:]
                played = share.loc[core.index, rec] > 0
                row[f"mrec_{grp}"] = float(((~played).mul(core, axis=0).sum(axis=0) / core.sum()).mean())
            rows.append(row)
    for path in paths.values():
        path.unlink(missing_ok=True)
    return pd.DataFrame(rows)


def load_injuries(sched: pd.DataFrame, refresh_current: bool = False, verbose: bool = True) -> pd.DataFrame:
    now = current_season()
    out = []
    for season in range(FIRST, now + 1):
        path = C.CACHE / f"injury_groups_{season}.csv"
        if (season == now and refresh_current) or not path.exists():
            if verbose:
                print(f"  injuries and snap counts {season} ...", flush=True)
            try:
                _season(season, sched).round(4).to_csv(path, index=False)
            except Exception as exc:
                print(f"  no injury data for {season} ({exc})")
                continue
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_injury_features(df: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """d_miss_<g>: home minus away share of core missing now; d_chg_<g>: the same,
    less what each team's last four games already reflect."""
    df = df.copy()
    if inj is None or inj.empty:
        for g in GROUPS:
            df[f"d_miss_{g}"] = df[f"d_chg_{g}"] = 0.0
        return df
    cols = [c for c in inj.columns if c.startswith(("miss_", "mrec_"))]
    h = inj.rename(columns={c: "h_" + c for c in cols}).rename(columns={"team": "home_team"})
    a = inj.rename(columns={c: "a_" + c for c in cols}).rename(columns={"team": "away_team"})
    df = df.merge(h.drop(columns=["season"]), on=["game_id", "home_team"], how="left")
    df = df.merge(a.drop(columns=["season"]), on=["game_id", "away_team"], how="left")
    for g in GROUPS:
        for p in ("h_", "a_"):
            df[f"{p}miss_{g}"] = df[f"{p}miss_{g}"].fillna(0.0)
            df[f"{p}mrec_{g}"] = df[f"{p}mrec_{g}"].fillna(0.0)
        df[f"d_miss_{g}"] = df[f"h_miss_{g}"] - df[f"a_miss_{g}"]
        df[f"d_chg_{g}"] = (df[f"h_miss_{g}"] - df[f"h_mrec_{g}"]) - (df[f"a_miss_{g}"] - df[f"a_mrec_{g}"])
    return df


# --------------------------------------------------------------------------
# Pass rush, weighted by production rather than position labels
# --------------------------------------------------------------------------
# Many edge rushers are listed as linebackers, so a position-based "defensive
# line" group misses them. Here every defender is weighted by his pass-rush
# production: QB hits (which include sacks) plus one more for a sack (half for
# a half sack), from play-by-play, matched to rosters and injury reports by the
# league's player id.
PR_COLS = ["game_id", "week", "season_type", "defteam", "sack_player_id", "half_sack_1_player_id",
           "half_sack_2_player_id", "qb_hit_1_player_id", "qb_hit_2_player_id"]
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"


def _pass_rush_production(season: int) -> pd.DataFrame:
    """Per defender per game: pass-rush production."""
    path = C.RAW / f"pbp_pr_{season}.parquet"
    if not path.exists():
        _download(PBP_URL.format(season=season), path)
    p = pd.read_parquet(path, columns=PR_COLS)
    path.unlink()
    p = p[p["season_type"] == "REG"]
    p["defteam"] = p["defteam"].replace(C.TEAM_ALIASES)
    parts = [(c, 1.0) for c in ("qb_hit_1_player_id", "qb_hit_2_player_id", "sack_player_id")] + \
            [(c, 0.5) for c in ("half_sack_1_player_id", "half_sack_2_player_id")]
    long = pd.concat([p.loc[p[c].notna(), ["game_id", "week", "defteam", c]].rename(columns={c: "player", "defteam": "team"}).assign(prod=w)
                      for c, w in parts])
    return long.groupby(["game_id", "week", "team", "player"], as_index=False)["prod"].sum()


def pass_rush_season(season: int, sched: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    """miss_pr / mrec_pr per team per game (same meaning as the position groups).
    prev: last season's production, for early weeks and newly signed players (half weight)."""
    prod = _pass_rush_production(season)
    rp, ip = C.RAW / f"roster_weekly_{season}.parquet", C.RAW / f"injuries_{season}.parquet"
    if not rp.exists():
        _download(URL.format(kind="weekly_rosters", season=season).replace("/weekly_rosters_", "/roster_weekly_"), rp)
    if not ip.exists():
        _download(URL.format(kind="injuries", season=season), ip)
    ros = pd.read_parquet(rp, columns=["team", "week", "gsis_id", "status", "game_type"])
    ros = ros[(ros["game_type"] == "REG") & ros["gsis_id"].notna()]
    inj = pd.read_parquet(ip, columns=["team", "week", "gsis_id", "report_status"])
    for df in (ros, inj):
        df["team"] = df["team"].replace(C.TEAM_ALIASES)
    status = ros.groupby(["team", "week", "gsis_id"])["status"].first().to_dict()
    report = inj.groupby(["team", "week", "gsis_id"])["report_status"].last().to_dict()
    roster_weeks = ros.groupby("team")["week"].apply(lambda w: sorted(set(w))).to_dict()
    on_team = ros.groupby(["team", "week"])["gsis_id"].apply(set).to_dict()
    last_year = prev.groupby("player")["prod"].sum() * 0.5 if prev is not None and len(prev) else pd.Series(dtype=float)

    g = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    long = pd.concat([g[["game_id", "week", "home_team"]].rename(columns={"home_team": "team"}),
                      g[["game_id", "week", "away_team"]].rename(columns={"away_team": "team"})])
    rows = []
    for team, tg in long.groupby("team"):
        tp = prod[prod["team"] == team]
        weeks = sorted(tg["week"])
        for gid, wk in zip(tg["game_id"], tg["week"]):
            rw = [w for w in roster_weeks.get(team, []) if w <= wk]
            rweek = rw[-1] if rw else None
            exact = rweek == wk
            members = on_team.get((team, rweek), set()) if rweek is not None else set()
            this_year = tp[tp["week"] < wk].groupby("player")["prod"].sum()
            weight = this_year.add(last_year.reindex(list(members)).dropna(), fill_value=0.0)
            weight = weight[weight > 0]
            row = {"season": season, "game_id": gid, "team": team, "miss_pr": 0.0, "mrec_pr": 0.0}
            if weight.sum() < 3:
                rows.append(row)
                continue

            def unavailable(pid, week, exact_week):
                if report.get((team, week, pid)) in ("Out", "Doubtful"):
                    return True
                st = status.get((team, week, pid))
                if st is None:
                    return True                      # not on the roster that week
                return st != "ACT" if exact_week else st not in ("ACT", "INA")

            out = [pid for pid in weight.index if rweek is not None and unavailable(pid, rweek, exact)]
            row["miss_pr"] = float(weight[out].sum() / weight.sum())
            recent = [w for w in weeks if w < wk][-RECENT:]
            shares = [weight[[pid for pid in weight.index if unavailable(pid, w, True)]].sum() / weight.sum()
                      for w in recent if (team, w) in on_team]
            row["mrec_pr"] = float(np.mean(shares)) if shares else 0.0
            rows.append(row)
    for path in (rp, ip):
        path.unlink(missing_ok=True)
    return pd.DataFrame(rows), prod


def load_pass_rush(sched: pd.DataFrame, refresh_current: bool = False, verbose: bool = True) -> pd.DataFrame:
    now = current_season()
    out, prev = [], None
    for season in range(FIRST, now + 1):
        path, ppath = C.CACHE / f"pass_rush_{season}.csv", C.CACHE / f"pass_rush_prod_{season}.csv"
        if (season == now and refresh_current) or not (path.exists() and ppath.exists()):
            if verbose:
                print(f"  pass rush {season} ...", flush=True)
            try:
                if prev is None and season > FIRST and (C.CACHE / f"pass_rush_prod_{season - 1}.csv").exists():
                    prev = pd.read_csv(C.CACHE / f"pass_rush_prod_{season - 1}.csv")
                df, prod = pass_rush_season(season, sched, prev)
                df.round(4).to_csv(path, index=False)
                prod.to_csv(ppath, index=False)
            except Exception as exc:
                print(f"  no pass-rush data for {season} ({exc})")
                prev = None
                continue
        out.append(pd.read_csv(path))
        prev = pd.read_csv(ppath)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_pass_rush_features(df: pd.DataFrame, pr: pd.DataFrame | None) -> pd.DataFrame:
    df = df.copy()
    if pr is None or pr.empty:
        df["d_miss_pr"] = df["d_chg_pr"] = 0.0
        return df
    h = pr.rename(columns={"miss_pr": "h_miss_pr", "mrec_pr": "h_mrec_pr", "team": "home_team"}).drop(columns=["season"])
    a = pr.rename(columns={"miss_pr": "a_miss_pr", "mrec_pr": "a_mrec_pr", "team": "away_team"}).drop(columns=["season"])
    df = df.merge(h, on=["game_id", "home_team"], how="left").merge(a, on=["game_id", "away_team"], how="left")
    for c in ("h_miss_pr", "h_mrec_pr", "a_miss_pr", "a_mrec_pr"):
        df[c] = df[c].fillna(0.0)
    df["d_miss_pr"] = df["h_miss_pr"] - df["a_miss_pr"]
    df["d_chg_pr"] = (df["h_miss_pr"] - df["h_mrec_pr"]) - (df["a_miss_pr"] - df["a_mrec_pr"])
    return df


def as_of(season: int) -> str | None:
    """When the official injury report on disk was last changed (the newest date_modified in
    nflverse's injuries file for the season), for the boards' "data as of" line; None when the
    file is not on hand."""
    ip = C.RAW / f"injuries_{season}.parquet"
    if not ip.exists():
        return None
    try:
        d = pd.read_parquet(ip, columns=["date_modified"])["date_modified"].dropna()
        return pd.Timestamp(d.max()).strftime("%Y-%m-%d %H:%M") if len(d) else None
    except Exception:
        return None
