"""Part B: a rating for every defender, and team-game features built on them (Oct 2026).

    python experiments/def_ratings.py      (writes data/raw/def_experiments/team_def.csv)

Everything for a game comes from earlier games, except who is available, which uses only
what is known by Friday (as nflmodel/game_roles.py): listed Out or Doubtful on that week's
injury report, or not active on the team's roster the week before (reserve lists,
released, traded). Game-day inactives are not used.

Per defender (keyed by PFR id; nflverse ids for reports, rosters and depth charts), decayed
over earlier games (half-life 10 team games, last season's games x 0.6, as the player
projections' defense) and shrunk toward the average of his position group:
  pr_pbp   QB hits plus sacks per estimated pass-rush snap, play-by-play, 2012 on
  pr_pfr   pressures (hurries + knockdowns + sacks) per estimated pass-rush snap, PFR, 2018 on
  cov_db   yards allowed in coverage per estimated coverage snap, PFR, 2018 on
  cov_ypt  yards allowed per target, PFR, 2018 on
  mt       missed tackles / (tackles + missed tackles), PFR, 2018 on
There is no free count of how many snaps a defender rushed or covered, so both use the
same estimate: his defensive snaps x the share of the opponent's plays that were
dropbacks. Position groups: DI (DT, NT), EDGE (DE, OLB), LB, CB, S.

Per team-game (each team's defense):
  miss_<r>   for core defenders (expected snap share >= 0.2 from this season's earlier
             games) who are unavailable: expected share x (his rating - his replacement),
             signed so that a positive number is quality lost. Two replacements:
             _bk   the position group's backup level (players under 0.35 of snaps)
             _dc   the next available player at his depth-chart slot (falls back to _bk)
  mrec_<r>   the same, averaged over the team's last four games, counting core players
             who did not play them (what the team's results already reflect)
  agg_*      team-level, decayed (half-life 8 games): pressure rate, blitz rate, yards per
             target allowed and missed-tackle rate (PFR, 2018 on); FTN blitzers and pass
             rushers per dropback (2022 on)
Subsets for the Part A re-test by rating: _dl (DI + EDGE), _lb, _db (CB + S).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, players as PL  # noqa: E402
from nflmodel.data import _download, current_season  # noqa: E402

REL = "https://github.com/nflverse/nflverse-data/releases/download"
RAW = C.RAW / "def_experiments"
FIRST = 2012
GROUP = {"DT": "DI", "NT": "DI", "DL": "DI", "DE": "EDGE", "OLB": "EDGE", "EDGE": "EDGE", "LB": "LB", "ILB": "LB", "MLB": "LB",
         "CB": "CB", "S": "S", "FS": "S", "SS": "S", "DB": "S", "SAF": "S"}
COVER = {"LB", "CB", "S"}
SUBSETS = {"dl": {"DI", "EDGE"}, "lb": {"LB"}, "db": {"CB", "S"}, "all": {"DI", "EDGE", "LB", "CB", "S"}}
HL, TEAM_HL, CARRY = 10.0, 8.0, 0.6
CORE, BACKUP, RECENT = 0.2, 0.35, 4
# rating: numerator, denominator, prior opportunities, sign (+1: higher is better for the defense)
RATINGS = {"pr_pbp": ("prod", "est_db", 150.0, 1), "pr_pfr": ("pressures", "est_db", 150.0, 1),
           "cov_db": ("yds_allowed", "est_db", 150.0, -1), "cov_ypt": ("yds_allowed", "targets", 40.0, -1),
           "mt": ("missed", "tack_att", 30.0, -1)}
PFR_FROM = 2018


def _get(kind: str, name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / name
    if not p.exists():
        _download(f"{REL}/{kind}/{name}", p)
    return p


def _players() -> pd.DataFrame:
    p = pd.read_parquet(_get("players", "players.parquet"), columns=["gsis_id", "pfr_id"])
    return p.dropna().drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]


def _sched(season: int) -> pd.DataFrame:
    from nflmodel import data
    g = data.load_all(refresh=False, verbose=False)[0]
    g = g[g.season == season]
    long = pd.concat([g[["game_id", "season", "week", "gameday", "played"]].assign(team=g.home_team, opp=g.away_team),
                      g[["game_id", "season", "week", "gameday", "played"]].assign(team=g.away_team, opp=g.home_team)])
    return long


def _depth(season: int, sched: pd.DataFrame) -> pd.DataFrame:
    """(team, week, gsis, slot, rank) for defenders."""
    try:
        d = pd.read_parquet(_get("depth_charts", f"depth_charts_{season}.parquet"))
    except Exception:
        return pd.DataFrame(columns=["team", "week", "gsis", "slot", "rank"])
    if "depth_team" in d.columns:
        d = d[d["formation"] == "Defense"] if "formation" in d.columns else d
        d = d.rename(columns={"club_code": "team", "gsis_id": "gsis", "depth_position": "slot"})
        d["rank"] = pd.to_numeric(d["depth_team"], errors="coerce")
        d["week"] = pd.to_numeric(d["week"], errors="coerce")
        out = d[["team", "week", "gsis", "slot", "rank"]]
    else:   # 2025 on: dated snapshots; each game takes the last one before it
        d = d[d["pos_grp"].astype(str).str.contains(" D|Defense|Nickel|Dime", regex=True)]
        d = d.rename(columns={"gsis_id": "gsis", "pos_abb": "slot", "pos_rank": "rank"})
        d["dt"] = pd.to_datetime(d["dt"], utc=True)
        rows = []
        for r in sched.drop_duplicates(["team", "week"]).itertuples():
            cut = pd.Timestamp(r.gameday).tz_localize("UTC")
            s = d[(d["team"] == r.team) & (d["dt"] < cut)]
            if len(s):
                rows.append(s[s["dt"] == s["dt"].max()][["gsis", "slot", "rank"]].assign(team=r.team, week=r.week))
        out = pd.concat(rows) if rows else pd.DataFrame(columns=["team", "week", "gsis", "slot", "rank"])
    out = out.dropna(subset=["gsis"]).copy()
    out["team"] = out["team"].replace(C.TEAM_ALIASES)
    out["slot"] = out["slot"].astype(str).str.upper().str.replace(r"^[LR](?=[A-Z])", "", regex=True)   # LCB -> CB, RDE -> DE
    return out.sort_values("rank").drop_duplicates(["team", "week", "gsis"])


def season_rows(season: int, pfr2gsis: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dense defender-game rows (every team game from a defender's first appearance for the
    team), with his stats where he played, plus the team-game table."""
    sched = _sched(season)
    s = pd.read_parquet(_get("snap_counts", f"snap_counts_{season}.parquet"))
    s["team"] = s["team"].replace(C.TEAM_ALIASES)
    s["opp"] = s["opponent"].replace(C.TEAM_ALIASES)
    s["grp"] = s["position"].map(GROUP)
    s = s[s["grp"].notna() & (s["defense_snaps"] > 0)]
    s = s.rename(columns={"pfr_player_id": "pid"})
    s["gsis"] = s["pid"].map(pfr2gsis)
    tgames = pd.read_csv(C.CACHE / f"team_games_{season}.csv")
    off = tgames[["game_id", "team", "n_pass", "plays"]].rename(columns={"team": "opp", "n_pass": "db_faced", "plays": "plays_faced"})
    s = s.merge(off, on=["game_id", "opp"], how="left")
    s["est_db"] = s["defense_snaps"] * (s["db_faced"] / s["plays_faced"]).fillna(0.6)
    prod = pd.read_csv(C.CACHE / f"pass_rush_prod_{season}.csv").rename(columns={"player": "gsis"})
    s = s.merge(prod[["game_id", "gsis", "prod"]], on=["game_id", "gsis"], how="left")
    s["prod"] = s["prod"].fillna(0.0)
    pf = ["pressures", "blitzed", "yds_allowed", "targets", "missed", "tack_att"]
    if season >= PFR_FROM:
        a = pd.read_parquet(_get("pfr_advstats", f"advstats_week_def_{season}.parquet"))
        a = a.rename(columns={"pfr_player_id": "pid", "def_pressures": "pressures", "def_times_blitzed": "blitzed",
                              "def_yards_allowed": "yds_allowed", "def_targets": "targets", "def_missed_tackles": "missed"})
        a["tack_att"] = a["def_tackles_combined"].fillna(0) + a["missed"].fillna(0)
        s = s.merge(a[["game_id", "pid"] + pf].groupby(["game_id", "pid"], as_index=False).sum(), on=["game_id", "pid"], how="left")
        for c in pf:
            s[c] = s[c].fillna(0.0)
    else:
        for c in pf:
            s[c] = np.nan
    keep = ["game_id", "team", "pid", "gsis", "grp", "position", "defense_snaps", "defense_pct", "est_db", "prod"] + pf
    s = s[keep]

    # dense: each defender x each of his team's games from his first appearance on
    tg = sched[["game_id", "season", "week", "team", "played"]].copy()
    tg["team"] = tg["team"].replace(C.TEAM_ALIASES)
    first = s.merge(tg[["game_id", "team", "week"]], on=["game_id", "team"]).groupby(["team", "pid"])["week"].min().rename("w0").reset_index()
    dense = first.merge(tg, on="team")
    dense = dense[dense["week"] >= dense["w0"]].drop(columns="w0")
    dense = dense.merge(s, on=["game_id", "team", "pid"], how="left")
    info = s.groupby("pid")[["gsis", "grp"]].agg(lambda x: x.mode().iloc[0] if len(x.dropna()) else np.nan)
    dense["gsis"] = dense["pid"].map(info["gsis"])
    dense["grp"] = dense["pid"].map(info["grp"])
    for c in ["defense_snaps", "defense_pct", "est_db", "prod"] + pf:
        dense[c] = dense[c].fillna(0.0) if season >= PFR_FROM or c not in pf else dense[c]
    # a defender traded away keeps his old team's later games as absences; drop any of those
    # that fall in a week he played for his new team (one row per player per week)
    dense["real"] = dense["defense_snaps"] > 0
    dense = dense.sort_values(["pid", "week", "real"]).drop_duplicates(["pid", "week"], keep="last")
    return dense, tg


def availability(season: int) -> tuple[dict, dict]:
    """report[(team, week, gsis)] -> status; roster[(gsis, week)] -> (team, status)."""
    inj = pd.read_parquet(_get("injuries", f"injuries_{season}.parquet")) if season >= 2009 else pd.DataFrame()
    rep = {}
    if len(inj):
        inj["team"] = inj["team"].replace(C.TEAM_ALIASES)
        rep = inj.groupby(["team", "week", "gsis_id"])["report_status"].last().to_dict()
    ros = pd.read_parquet(_get("weekly_rosters", f"roster_weekly_{season}.parquet"), columns=["team", "week", "gsis_id", "status"])
    ros["team"] = ros["team"].replace(C.TEAM_ALIASES)
    roster = {(r.gsis_id, int(r.week)): (r.team, r.status) for r in ros.itertuples()}
    return rep, roster


def friday_out(row, rep, roster) -> bool:
    """Known by Friday: Out/Doubtful on this week's report, or the roster listing from the
    weeks before (up to three back) shows him off the team or on a reserve list."""
    if rep.get((row.team, row.week, row.gsis)) in ("Out", "Doubtful"):
        return True
    for back in (1, 2, 3):
        x = roster.get((row.gsis, row.week - back))
        if x is not None:
            team, st = x
            return team != row.team or st not in ("ACT", "INA")
    x = roster.get((row.gsis, row.week))
    return x is not None and (x[0] != row.team or x[1] not in ("ACT", "INA"))


def build(seasons=None, verbose=True) -> pd.DataFrame:
    seasons = list(seasons or range(FIRST, current_season() + 1))
    pfr2gsis = _players()
    dense, tgs, depth = [], [], []
    for season in seasons:
        if verbose:
            print(f"  defenders {season} ...", flush=True)
        d, tg = season_rows(season, pfr2gsis)
        rep, roster = availability(season)
        d["out"] = [friday_out(r, rep, roster) if isinstance(r.gsis, str) else False for r in d.itertuples()]
        dense.append(d)
        tgs.append(tg)
        depth.append(_depth(season, _sched(season)).assign(season=season))
    D = pd.concat(dense, ignore_index=True)
    TG = pd.concat(tgs, ignore_index=True)
    DC = pd.concat(depth, ignore_index=True)
    D["t"] = PL._t(D["season"], D["week"])
    TG["t"] = PL._t(TG["season"], TG["week"])

    # ratings before each game (decayed, two seasons back at most), shrunk to the group mean
    nums = sorted({v[0] for v in RATINGS.values()} | {v[1] for v in RATINGS.values()})
    R = PL.asof(D, "pid", nums, None, HL, CARRY)
    played = D[D["defense_snaps"] > 0]
    for name, (num, den, prior, sign) in RATINGS.items():
        ok = played[den] > 0
        if num in ("pressures", "yds_allowed", "missed", "targets") or den in ("targets", "tack_att"):
            ok &= played["season"] >= PFR_FROM
        grp_mean = (played[ok].groupby("grp")[num].sum() / played[ok].groupby("grp")[den].sum())
        R[name] = PL._shrink(R[num + "_sum"], R[den + "_sum"], R["grp"].map(grp_mean), prior)
        if name.startswith("cov"):
            R.loc[~R["grp"].isin(COVER), name] = np.nan
        # backup level: what low-snap players in the group produce
        sea = played.groupby(["season", "pid"])["defense_pct"].transform("mean")
        bk = played[ok & (sea < BACKUP)]
        R[name + "_bk"] = R["grp"].map(bk.groupby("grp")[num].sum() / bk.groupby("grp")[den].sum())
        if name in ("pr_pfr", "cov_db", "cov_ypt", "mt"):
            R.loc[R["season"] < PFR_FROM + 1, [name, name + "_bk"]] = np.nan        # 2018 has no earlier PFR games

    # expected share: mean defensive snap share over the team's earlier games this season
    # (games he missed count 0; his rows start at his first game for the team, so divide by
    # the number of the team's earlier games, not his own)
    R = R.sort_values(["pid", "team", "t"]).reset_index(drop=True)
    before = R.groupby(["pid", "team", "season"])["defense_pct"].transform(lambda x: x.cumsum().shift(fill_value=0.0))
    n_before = TG.sort_values("t").assign(n=lambda x: x.groupby(["team", "season"]).cumcount())[["game_id", "team", "n"]]
    R = R.merge(n_before, on=["game_id", "team"], how="left")
    R["exp_share"] = (before.to_numpy() / R["n"].replace(0, np.nan)).clip(0, 1)
    R["out"] = R["out"].fillna(False).astype(bool)
    core = R[(R["exp_share"] >= CORE)].copy()

    # depth-chart replacement for each unavailable core defender
    rate = R.set_index(["game_id", "gsis"])[[*RATINGS]].groupby(level=[0, 1]).first()
    DC = DC.merge(TG[["season", "week", "team", "game_id"]], on=["season", "week", "team"])
    outs = core[core["out"]]
    slot = DC.set_index(["game_id", "gsis"])[["slot", "rank"]]
    o = outs.join(slot, on=["game_id", "gsis"])
    o_set = set(zip(core.loc[core["out"], "game_id"], core.loc[core["out"], "gsis"]))
    repl = {}
    by_slot = DC.sort_values("rank").groupby(["game_id", "slot"])["gsis"].apply(list).to_dict()
    for r in o.itertuples():
        if not isinstance(r.slot, str):
            continue
        for j in by_slot.get((r.game_id, r.slot), []):
            if j != r.gsis and (r.game_id, j) not in o_set:
                repl[(r.game_id, r.gsis)] = j
                break
    for name, (_, _, _, sign) in RATINGS.items():
        core[f"v_{name}_bk"] = sign * (core[name] - core[name + "_bk"]) * core["exp_share"]
        rj = [rate[name].get((gid, repl.get((gid, p))), np.nan) if (gid, p) in repl else np.nan
              for gid, p in zip(core["game_id"], core["gsis"])]
        rj = pd.Series(rj, index=core.index).fillna(core[name + "_bk"])
        core[f"v_{name}_dc"] = sign * (core[name] - rj) * core["exp_share"]

    vcols = [c for c in core.columns if c.startswith("v_")]
    # share of the team's last four played games (this season) each core defender missed
    played_in = set(zip(R.loc[R["defense_snaps"] > 0, "game_id"], R.loc[R["defense_snaps"] > 0, "pid"]))
    prior_games = {}
    for (team, season), x in TG[TG["played"].astype(bool)].sort_values("t").groupby(["team", "season"]):
        ids = list(x["game_id"])
        for k, gid in enumerate(ids):
            prior_games[(gid, team)] = ids[max(0, k - RECENT):k]
    for (team, season), x in TG[~TG["played"].astype(bool)].groupby(["team", "season"]):
        done = TG[TG["played"].astype(bool) & (TG["team"] == team) & (TG["season"] == season)].sort_values("t")["game_id"].tolist()
        for gid in x["game_id"]:
            prior_games[(gid, team)] = done[-RECENT:]
    core["miss_frac"] = [np.mean([(p, pid) not in played_in for p in prior_games.get((gid, team), [])]) if prior_games.get((gid, team)) else 0.0
                         for gid, team, pid in zip(core["game_id"], core["team"], core["pid"])]
    parts = []
    for sub, grps in SUBSETS.items():
        cc = core[core["grp"].isin(grps)]
        vals = cc[vcols].fillna(0.0).clip(lower=0)                # a defender out can't make his team better
        now = vals.mul(cc["out"].astype(float), axis=0).groupby([cc["game_id"], cc["team"]]).sum()
        rec = vals.mul(cc["miss_frac"], axis=0).groupby([cc["game_id"], cc["team"]]).sum()
        now.columns = [f"miss_{v[2:]}_{sub}" for v in vcols]
        rec.columns = [f"mrec_{v[2:]}_{sub}" for v in vcols]
        parts += [now, rec]
    T = pd.concat(parts, axis=1).reset_index()
    T = TG[["game_id", "season", "week", "team"]].merge(T, on=["game_id", "team"], how="left").fillna(0.0)
    return T.merge(aggregates(D, TG), on=["game_id", "team"], how="left")


def aggregates(D: pd.DataFrame, TG: pd.DataFrame) -> pd.DataFrame:
    """Team-level, decayed, before each game: pressure rate, blitz rate, yards per target
    allowed, missed-tackle rate (PFR), FTN blitzers and pass rushers per dropback."""
    p = D[D["season"] >= PFR_FROM].groupby(["game_id", "team"])[["pressures", "blitzed", "yds_allowed", "targets", "missed", "tack_att"]].sum()
    dbf = D.groupby(["game_id", "team"])["est_db"].sum()          # snaps x dropback share, 11 per play
    t = TG[["game_id", "season", "week", "team", "t"]].merge(p.reset_index(), on=["game_id", "team"], how="left")
    t = t.merge((dbf / 11).rename("dbf").reset_index(), on=["game_id", "team"], how="left")
    t = t.merge(ftn(TG), on=["game_id", "team"], how="left")
    cols = ["pressures", "blitzed", "yds_allowed", "targets", "missed", "tack_att", "dbf", "ftn_blitzers", "ftn_rushers", "ftn_db"]
    a = PL.asof(t, "team", cols, None, TEAM_HL, CARRY)
    out = a[["game_id", "team"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["agg_press"] = a["pressures_sum"] / a["dbf_sum"]
        out["agg_blitz"] = a["blitzed_sum"] / a["dbf_sum"]
        out["agg_ypt"] = a["yds_allowed_sum"] / a["targets_sum"]
        out["agg_mt"] = a["missed_sum"] / a["tack_att_sum"]
        out["agg_ftn_blitz"] = a["ftn_blitzers_sum"] / a["ftn_db_sum"]
        out["agg_ftn_rush"] = a["ftn_rushers_sum"] / a["ftn_db_sum"]
    pre = a["season"] < PFR_FROM + 1
    out.loc[pre.to_numpy(), ["agg_press", "agg_blitz", "agg_ypt", "agg_mt"]] = np.nan
    out.loc[(a["season"] < 2023).to_numpy(), ["agg_ftn_blitz", "agg_ftn_rush"]] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def ftn(TG: pd.DataFrame) -> pd.DataFrame:
    """FTN charting, 2022 on: blitzers and pass rushers summed per defense and game."""
    out = []
    for season in sorted(TG["season"].unique()):
        if season < 2022:
            continue
        try:
            f = pd.read_parquet(_get("ftn_charting", f"ftn_charting_{season}.parquet"),
                                columns=["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers"])
            pbp = pd.read_parquet(_get("pbp", f"play_by_play_{season}.parquet"), columns=["game_id", "play_id", "defteam", "qb_dropback"])
        except Exception as exc:
            print(f"  no FTN for {season} ({exc})")
            continue
        f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        m = f.merge(pbp, on=["game_id", "play_id"])
        m = m[(m["qb_dropback"] == 1) & m["n_pass_rushers"].notna()]
        m["defteam"] = m["defteam"].replace(C.TEAM_ALIASES)
        out.append(m.groupby(["game_id", "defteam"]).agg(ftn_blitzers=("n_blitzers", "sum"), ftn_rushers=("n_pass_rushers", "sum"),
                                                         ftn_db=("play_id", "size")).reset_index().rename(columns={"defteam": "team"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["game_id", "team", "ftn_blitzers", "ftn_rushers", "ftn_db"])


def main():
    T = build()
    T.round(5).to_csv(RAW / "team_def.csv", index=False)
    print(f"wrote {RAW / 'team_def.csv'}: {len(T)} team-games")


if __name__ == "__main__":
    main()
