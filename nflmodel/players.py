"""Player projections: passing, rushing and receiving, game by game.

Opportunity times efficiency, every piece measured only from earlier games:

  team volume    pass attempts and carries per game, from the team's recent volume,
                 what the opponent has allowed, the projected margin and total from
                 the game model, and wind (a regression fitted on earlier seasons)
  player share   share of team targets / carries / pass attempts in games the
                 player was active, recent games weighted more; teammates who are
                 out leave their share to the players who are available
  efficiency     catch rate, yards per target and per carry, completion rate,
                 yards per attempt, TD and interception rates, each shrunk toward the
                 position average and multiplied by what this defense allows to that
                 position compared with the league
  touchdowns     team touchdowns from the game model's implied points, split into
                 passing and rushing by the team's tendency, then shared out by
                 red-zone targets and carries
  calibration    a last regression per stat on earlier seasons, which also takes in
                 the defense's missing backs and pass rushers and the wind

Availability comes from weekly rosters and injury reports (player ids match the
play-by-play), so a game a player missed is not counted as a zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C, overrides as overrides_mod
from .data import _download, current_season

SKILL = ("QB", "RB", "WR", "TE", "FB")
GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
SEASON_LEN = 22
TEAM_HL, PLAYER_HL, DEF_HL = 8.0, 6.0, 10.0      # half-lives in games
CARRY = 0.6                                      # weight on last season's games
PRIORS = {  # shrinkage, in opportunities, and position averages filled in from data
    "ypt": 40.0, "catch": 40.0, "ypc": 60.0, "ypa": 150.0, "comp": 150.0, "ptd": 300.0, "int": 400.0,
    "def_ypt": 80.0, "def_ypc": 120.0, "def_ypa": 250.0,
}
STATS = ["pass_att", "completions", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td",
         "targets", "receptions", "rec_yds", "rec_td"]
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet"
# Long plays and efficiency (tested Oct 2026, experiments/players_capped_test.py): when set,
# a player's yards per carry and per target are measured with each play capped at EFF_CAP
# yards, and yards beyond the cap come back as his position's average rate, since breakaway
# plays repeat little from game to game. Needs rush_yds_c<cap> and rec_yds_c<cap> in the box
# scores. CAP_RECENT also caps the recent per-game average the calibration blends in.
# None leaves everything as it was.
EFF_CAP: int | None = None
CAP_RECENT = False
INJ_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------

def _avail_season(season: int) -> pd.DataFrame:
    rp, ip = C.RAW / f"roster_weekly_{season}.parquet", C.RAW / f"injuries_{season}.parquet"
    if not rp.exists():
        _download(ROSTER_URL.format(season=season), rp)
    if not ip.exists():
        try:
            _download(INJ_URL.format(season=season), ip)
        except Exception:
            pass
    r = pd.read_parquet(rp, columns=["team", "week", "gsis_id", "status", "position", "game_type"])
    r = r[r["gsis_id"].notna() & r["position"].isin(SKILL)].copy()
    r["team"] = r["team"].replace(C.TEAM_ALIASES)
    if ip.exists():
        inj = pd.read_parquet(ip, columns=["team", "week", "gsis_id", "report_status"])
        inj["team"] = inj["team"].replace(C.TEAM_ALIASES)
        rep = inj.groupby(["team", "week", "gsis_id"])["report_status"].last()
        r = r.merge(rep.rename("report").reset_index(), on=["team", "week", "gsis_id"], how="left")
    else:
        r["report"] = np.nan
    r["active"] = ((r["status"] == "ACT") & ~r["report"].isin(["Out", "Doubtful"])).astype(int)
    r = r.rename(columns={"gsis_id": "player_id"})[["team", "week", "player_id", "position", "status", "report", "active"]]
    r.insert(0, "season", season)
    for path in (rp, ip):
        path.unlink(missing_ok=True)
    return r.drop_duplicates(["season", "week", "team", "player_id"])


def load_availability(refresh_current: bool = False, verbose: bool = True) -> pd.DataFrame:
    now = current_season()
    out = []
    for season in range(2012, now + 1):
        path = C.CACHE / f"player_avail_{season}.csv"
        if (season == now and refresh_current) or not path.exists():
            if verbose:
                print(f"  player availability {season} ...", flush=True)
            _avail_season(season).to_csv(path, index=False)
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Decayed "as of" averages
# --------------------------------------------------------------------------

def _t(season, week):
    return np.asarray(season) * SEASON_LEN + np.asarray(week)


def asof(df: pd.DataFrame, entity: str, num: list[str], den: list[str] | None, hl: float, carry: float = CARRY,
         prior_n: dict | None = None, prior_v: dict | None = None) -> pd.DataFrame:
    """For each row, decayed sums of earlier rows of the same entity (games ago half-life
    `hl`, earlier seasons times `carry`, two seasons back at most). Returns columns
    <num>_sum (and <den>_sum) so ratios can be shrunk by the caller."""
    df = df.sort_values([entity, "t"]).reset_index(drop=True)
    cols = num + (den or [])
    vals = df[cols].to_numpy(float)
    nan = np.isnan(vals)
    vals = np.where(nan, 0.0, vals)
    sums = np.zeros_like(vals)
    ngames = np.zeros(len(df))
    season = df["season"].to_numpy()
    ent = df[entity].to_numpy()
    starts = np.flatnonzero(np.r_[True, ent[1:] != ent[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s0, e0 in zip(starts, ends):
        n = e0 - s0
        if n < 2:
            continue
        idx = np.arange(n)
        age = idx[:, None] - idx[None, :]                     # rows: current game, cols: earlier game
        sd = season[s0:e0][:, None] - season[s0:e0][None, :]
        W = np.where((age > 0) & (sd <= 2), 0.5 ** (age / hl) * carry ** sd, 0.0)
        sums[s0:e0] = W @ vals[s0:e0]
        ngames[s0:e0] = W.sum(1)
    out = {c + "_sum": sums[:, k] for k, c in enumerate(cols)}
    out["n_games"] = ngames
    res = pd.concat([df, pd.DataFrame(out)], axis=1)
    return res


# --------------------------------------------------------------------------
# Building the projection table
# --------------------------------------------------------------------------

def team_games(frame: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game (played and upcoming) with game-model context and,
    for played games, team volume."""
    f = frame
    cols = ["game_id", "season", "week", "home_team", "away_team", "proj_margin", "proj_total", "wind", "indoor", "played", "neutral"]
    h = f[cols].rename(columns={"home_team": "team", "away_team": "opp"}).assign(sign=1.0, home=1 - f["neutral"])
    a = f[cols].rename(columns={"away_team": "team", "home_team": "opp"}).assign(sign=-1.0, home=0)
    tg = pd.concat([h, a], ignore_index=True)
    tg["margin"] = tg["proj_margin"] * tg["sign"]
    tg["imp_pts"] = (tg["proj_total"] + tg["margin"]) / 2
    vol = pg.groupby(["game_id", "posteam"])[["pass_att", "carries", "pass_td", "rec_td", "rush_td"]].sum().reset_index()
    vol = vol.rename(columns={"posteam": "team", "pass_att": "team_att", "carries": "team_car", "pass_td": "team_ptd", "rush_td": "team_rtd"})
    tg = tg.merge(vol.drop(columns=["rec_td"]), on=["game_id", "team"], how="left")
    tg["t"] = _t(tg["season"], tg["week"])
    off = asof(tg, "team", ["team_att", "team_car", "team_ptd", "team_rtd"], None, TEAM_HL)
    off["team_att_avg"] = off["team_att_sum"] / off["n_games"].replace(0, np.nan)
    off["team_car_avg"] = off["team_car_sum"] / off["n_games"].replace(0, np.nan)
    off["ptd_share"] = (off["team_ptd_sum"] + 6) / (off["team_ptd_sum"] + off["team_rtd_sum"] + 10)
    dfn = asof(tg.rename(columns={"team": "x", "opp": "team"}).rename(columns={"x": "opp"}), "team",
               ["team_att", "team_car"], None, TEAM_HL)
    dfn["opp_att_allowed"] = dfn["team_att_sum"] / dfn["n_games"].replace(0, np.nan)
    dfn["opp_car_allowed"] = dfn["team_car_sum"] / dfn["n_games"].replace(0, np.nan)
    # dfn is keyed by the defense: rename back so it joins on (game_id, opp)
    dfn = dfn.rename(columns={"team": "opp_key"})[["game_id", "opp_key", "opp_att_allowed", "opp_car_allowed"]]
    tg = off[["game_id", "team", "opp", "season", "week", "t", "margin", "proj_total", "imp_pts", "wind", "indoor", "played",
              "home", "team_att", "team_car", "team_ptd", "team_rtd", "team_att_avg", "team_car_avg", "ptd_share"]]
    tg = tg.merge(dfn, left_on=["game_id", "opp"], right_on=["game_id", "opp_key"], how="left").drop(columns="opp_key")
    lg_att, lg_car = tg["team_att"].mean(), tg["team_car"].mean()
    for c, lg in (("team_att_avg", lg_att), ("opp_att_allowed", lg_att), ("team_car_avg", lg_car), ("opp_car_allowed", lg_car)):
        tg[c] = tg[c].fillna(lg)
    tg["ptd_share"] = tg["ptd_share"].fillna(0.6)
    return tg


def player_rows(tg: pd.DataFrame, pg: pd.DataFrame, avail: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Every skill player on each team for each game: rostered that week (or, for an
    upcoming game, the latest roster) or credited with a stat."""
    sched = tg[["game_id", "team", "opp", "season", "week", "t"]]
    a = avail.copy()
    # an upcoming week with no roster yet uses the team's latest roster week
    latest = a.groupby(["season", "team"])["week"].max().rename("latest").reset_index()
    need = sched.merge(latest, on=["season", "team"], how="left")
    need["rweek"] = np.where(need["week"] > need["latest"], need["latest"], need["week"])
    rows = need.merge(a.rename(columns={"week": "rweek"}), on=["season", "team", "rweek"], how="inner")
    future = rows["week"] > rows["latest"]
    # for an upcoming game only longer-term absences are known
    rows.loc[future, "active"] = ((rows.loc[future, "status"].isin(["ACT", "INA"])) &
                                  ~rows.loc[future, "report"].isin(["Out", "Doubtful"])).astype(int)
    rows = rows[["game_id", "team", "opp", "season", "week", "t", "player_id", "position", "active", "status", "report"]]
    extra = [c for c in pg.columns if c.startswith(("rush_yds_c", "rec_yds_c"))]
    stats = pg.rename(columns={"posteam": "team"})[["game_id", "team", "player_id"] + STATS + ["rz_targets", "rz_carries", "gl_carries", "air_yards", "sacks"] + extra]
    rows = rows.merge(stats, on=["game_id", "team", "player_id"], how="outer", indicator=True)
    # players credited with stats but missing from the roster listing (rare): take schedule info
    miss = rows["_merge"] == "right_only"
    if miss.any():
        fill = rows.loc[miss, ["game_id", "team"]].merge(sched, on=["game_id", "team"], how="left")
        for c in ("opp", "season", "week", "t"):
            rows.loc[miss, c] = fill[c].to_numpy()
        rows.loc[miss, "position"] = rows.loc[miss, "player_id"].map(info["position"])
        rows.loc[miss, "active"] = 1
    played_any = rows["_merge"] != "left_only"
    rows.loc[played_any, "active"] = 1                 # recorded a stat, so he played
    rows = rows.drop(columns="_merge")
    rows = rows[rows["position"].isin(SKILL)].copy()
    rows["group"] = rows["position"].map(GROUP)
    rows[STATS + ["rz_targets", "rz_carries", "gl_carries", "air_yards", "sacks"] + extra] = rows[STATS + ["rz_targets", "rz_carries", "gl_carries", "air_yards", "sacks"] + extra].fillna(0.0)
    rows["season"] = rows["season"].astype(int)
    rows["week"] = rows["week"].astype(int)
    return rows.merge(tg[["game_id", "team", "team_att", "team_car", "played"]], on=["game_id", "team"], how="left")


def _shrink(num, den, prior_mean, prior_n):
    return (num + prior_mean * prior_n) / (den + prior_n)


def pregame_active(rows: pd.DataFrame, avail: pd.DataFrame) -> pd.Series:
    """Who a projection made a few days before kickoff would expect to play: his roster
    status the week before (active or a game-day inactive; injured reserve and the like
    are out), with no injury report for this week and no game-day inactive list. A player
    with no earlier listing uses this week's roster status. Used to test props against
    prices posted before the week's news (experiments/props_real_lines.py)."""
    a = avail[["season", "week", "player_id", "status"]].copy()
    a["t"] = _t(a["season"], a["week"])
    a = a.sort_values(["player_id", "t"]).drop_duplicates(["player_id", "t"], keep="last")
    a["prev_status"] = a.groupby("player_id")["status"].shift()
    a["prev_t"] = a.groupby("player_id")["t"].shift()
    a = a[["player_id", "t", "prev_status", "prev_t"]]
    r = rows[["player_id", "t", "status"]].reset_index().merge(a, on=["player_id", "t"], how="left").set_index("index")
    st = r["prev_status"].where(r["prev_status"].notna() & (r["t"] - r["prev_t"] <= 3), r["status"])
    out = st.isin(["ACT", "INA"]).astype(int)
    out[st.isna()] = 1                     # credited with stats, no roster listing at all
    return out.reindex(rows.index)


def build(frame: pd.DataFrame, pg: pd.DataFrame, info: pd.DataFrame, avail: pd.DataFrame,
          inj_groups: pd.DataFrame | None = None, pass_rush: pd.DataFrame | None = None,
          games: pd.DataFrame | None = None, overrides: pd.DataFrame | None = None,
          pregame: bool = False) -> pd.DataFrame:
    """The full player-game table with every pre-game projection input.

    overrides (optional) marks players out of upcoming games (see nflmodel/overrides.py).

    pregame: project as of a few days before kickoff (pregame_active): who plays is
    judged without this week's injury report or inactives. The board never uses this.

    games (optional) supplies each game's official starting quarterbacks, used only as an
    outcome: whether the projected starter really started (see walk_forward)."""
    tg = team_games(frame, pg)
    rows = player_rows(tg, pg, avail, info)
    if overrides is not None:
        rows = overrides_mod.apply_out(rows, overrides)
    # who the projection expects to play; history and fitting still use who did play
    rows["active_pre"] = pregame_active(rows, avail) if pregame else rows["active"]
    played_rows = rows[rows["played"] & (rows["active"] == 1)].copy()

    # ---- player usage shares, over games the player was active ---------------------
    # Every player row is kept (so players who are out still carry their history), but only
    # games a player was active for, and which have been played, feed his averages.
    act = rows.copy()
    use = (act["active"] == 1) & act["played"].astype(bool)
    act["tgt_share"] = np.where(act["team_att"] > 0, act["targets"] / act["team_att"], np.nan)
    act["car_share"] = np.where(act["team_car"] > 0, act["carries"] / act["team_car"], np.nan)
    act["att_share"] = np.where(act["team_att"] > 0, act["pass_att"] / act["team_att"], np.nan)
    act["one"] = np.where(use, 1.0, np.nan)          # upcoming games and games missed carry no outcome
    # quarterbacks: games started (half the team's passes or more), and passing stats in those games
    act["start"] = np.where(use, (act["pass_att"] >= 0.5 * act["team_att"].clip(lower=1)).astype(float), np.nan)
    for c in ("pass_att", "completions", "pass_yds", "pass_td", "ints"):
        act[c + "_st"] = np.where(use, act[c] * act["start"], np.nan)
    capc = [f"rush_yds_c{EFF_CAP}", f"rec_yds_c{EFF_CAP}"] if EFF_CAP else []
    for c in ("tgt_share", "car_share", "att_share", "targets", "receptions", "rec_yds", "rec_td", "rz_targets",
              "carries", "rush_yds", "rush_td", "rz_carries", "gl_carries", "pass_att", "completions", "pass_yds", "pass_td", "ints", *capc):
        act[c] = act[c].where(use)
    pl = asof(act, "player_id", ["tgt_share", "car_share", "att_share", "targets", "receptions", "rec_yds", "rec_td", "rz_targets",
                                 "carries", "rush_yds", "rush_td", "rz_carries", "gl_carries", "pass_att", "completions",
                                 "pass_yds", "pass_td", "ints", "pass_att_st", "completions_st", "pass_yds_st", "pass_td_st", "ints_st", *capc],
              ["one", "start"], PLAYER_HL)
    n = pl["one_sum"]
    pos_share = act[use].groupby("group")[["tgt_share", "car_share", "att_share"]].mean()
    for c in ("tgt_share", "car_share", "att_share"):
        prior = pl["group"].map(pos_share[c]).fillna(0.0) * 0.5     # an unknown player starts at half his position's average
        pl[f"{c}_raw"] = (pl[f"{c}_sum"] + prior * 1.0) / (n + 1.0)
    # efficiency, shrunk toward position averages (measured on all active games)
    tot = act[use].groupby("group")[["targets", "receptions", "rec_yds", "carries", "rush_yds", "pass_att", "completions", "pass_yds", "pass_td", "ints"]].sum()
    pm = lambda a, b: (tot[a] / tot[b].replace(0, np.nan)).fillna(0)
    g = pl["group"]
    pl["catch"] = _shrink(pl["receptions_sum"], pl["targets_sum"], g.map(pm("receptions", "targets")), PRIORS["catch"])
    pl["ypt"] = _shrink(pl["rec_yds_sum"], pl["targets_sum"], g.map(pm("rec_yds", "targets")), PRIORS["ypt"])
    pl["ypc"] = _shrink(pl["rush_yds_sum"], pl["carries_sum"], g.map(pm("rush_yds", "carries")), PRIORS["ypc"])
    if EFF_CAP:
        rc, tc = f"rush_yds_c{EFF_CAP}", f"rec_yds_c{EFF_CAP}"
        ct = act[use].groupby("group")[[rc, tc]].sum()
        tot2 = tot.join(ct)
        pc = lambda a, b: (tot2[a] / tot2[b].replace(0, np.nan)).fillna(0)
        ex_r = pc("rush_yds", "carries") - pc(rc, "carries")          # long-play yards per carry, by position
        ex_t = pc("rec_yds", "targets") - pc(tc, "targets")
        pl["ypc"] = _shrink(pl[rc + "_sum"], pl["carries_sum"], g.map(pc(rc, "carries")), PRIORS["ypc"]) + g.map(ex_r).fillna(0)
        pl["ypt"] = _shrink(pl[tc + "_sum"], pl["targets_sum"], g.map(pc(tc, "targets")), PRIORS["ypt"]) + g.map(ex_t).fillna(0)
        if CAP_RECENT:      # the recent per-game average the calibration uses, on the same basis
            pl["rush_yds_sum"] = pl[rc + "_sum"] + pl["carries_sum"] * g.map(ex_r).fillna(0)
            pl["rec_yds_sum"] = pl[tc + "_sum"] + pl["targets_sum"] * g.map(ex_t).fillna(0)
    qb = lambda a: tot.loc["QB", a] if "QB" in tot.index else np.nan
    pl["comp"] = _shrink(pl["completions_sum"], pl["pass_att_sum"], qb("completions") / qb("pass_att"), PRIORS["comp"])
    pl["ypa"] = _shrink(pl["pass_yds_sum"], pl["pass_att_sum"], qb("pass_yds") / qb("pass_att"), PRIORS["ypa"])
    pl["ptd_rate"] = _shrink(pl["pass_td_sum"], pl["pass_att_sum"], qb("pass_td") / qb("pass_att"), PRIORS["ptd"])
    pl["int_rate"] = _shrink(pl["ints_sum"], pl["pass_att_sum"], qb("ints") / qb("pass_att"), PRIORS["int"])
    pl["rz_tgt_rate"] = (pl["rz_targets_sum"] + 0.1) / (pl["targets_sum"] + 1)
    pl["rz_car_rate"] = (pl["rz_carries_sum"] + 0.1) / (pl["carries_sum"] + 1)
    pl["gl_car_rate"] = (pl["gl_carries_sum"] + 0.05) / (pl["carries_sum"] + 1)

    # ---- defense vs position --------------------------------------------------------
    # One row per defense per game (upcoming games included, with no stats), so each
    # game sees only the defense's earlier games and an upcoming game gets the latest.
    dgames = tg[["game_id", "opp", "season", "week", "t"]].drop_duplicates(["game_id", "opp"])
    played = rows[rows["played"].astype(bool)]
    rec = played[played["targets"] > 0].groupby(["game_id", "opp", "group"])[["targets", "receptions", "rec_yds"]].sum().reset_index()
    grid = dgames.merge(pd.DataFrame({"group": ["RB", "WR", "TE"]}), how="cross").merge(rec, on=["game_id", "opp", "group"], how="left")
    grid[["targets", "receptions", "rec_yds"]] = grid[["targets", "receptions", "rec_yds"]].fillna(0.0)
    grid["dkey"] = grid["opp"] + "|" + grid["group"]
    d = asof(grid, "dkey", ["targets", "receptions", "rec_yds"], None, DEF_HL)
    lg = rec.groupby("group")[["targets", "receptions", "rec_yds"]].sum()
    lg_ypt, lg_cr = d["group"].map(lg["rec_yds"] / lg["targets"]), d["group"].map(lg["receptions"] / lg["targets"])
    d["def_ypt"] = _shrink(d["rec_yds_sum"], d["targets_sum"], lg_ypt, PRIORS["def_ypt"]) / lg_ypt
    d["def_catch"] = _shrink(d["receptions_sum"], d["targets_sum"], lg_cr, PRIORS["def_ypt"]) / lg_cr
    dvp = d[["game_id", "opp", "group", "def_ypt", "def_catch"]]
    ru = played.groupby(["game_id", "opp"])[["carries", "rush_yds", "pass_att", "completions", "pass_yds"]].sum().reset_index()
    dg = dgames.merge(ru, on=["game_id", "opp"], how="left")
    dg[["carries", "rush_yds", "pass_att", "completions", "pass_yds"]] = dg[["carries", "rush_yds", "pass_att", "completions", "pass_yds"]].fillna(0.0)
    dr = asof(dg, "opp", ["carries", "rush_yds", "pass_att", "completions", "pass_yds"], None, DEF_HL)
    lgr = ru[["carries", "rush_yds", "pass_att", "completions", "pass_yds"]].sum()
    dr["def_ypc"] = _shrink(dr["rush_yds_sum"], dr["carries_sum"], lgr["rush_yds"] / lgr["carries"], PRIORS["def_ypc"]) / (lgr["rush_yds"] / lgr["carries"])
    dr["def_ypa"] = _shrink(dr["pass_yds_sum"], dr["pass_att_sum"], lgr["pass_yds"] / lgr["pass_att"], PRIORS["def_ypa"]) / (lgr["pass_yds"] / lgr["pass_att"])
    dr["def_comp"] = _shrink(dr["completions_sum"], dr["pass_att_sum"], lgr["completions"] / lgr["pass_att"], PRIORS["def_ypa"]) / (lgr["completions"] / lgr["pass_att"])
    dr_all = dr[["game_id", "opp", "def_ypc", "def_ypa", "def_comp"]]

    # last game's starting quarterback (most pass attempts) for each team-game
    top = pg[pg["pass_att"] > 0].sort_values("pass_att").groupby(["game_id", "posteam"]).tail(1)[["game_id", "posteam", "player_id"]]
    seq = tg[["game_id", "team", "t"]].merge(top.rename(columns={"posteam": "team", "player_id": "qb_this"}), on=["game_id", "team"], how="left")
    seq = seq.sort_values(["team", "t"])
    seq["prev_qb"] = seq.groupby("team")["qb_this"].transform(lambda s: s.ffill().shift())
    out = pl.merge(tg.drop(columns=["team_att", "team_car", "played"]), on=["game_id", "team", "opp", "season", "week", "t"], how="left")
    out = out.merge(seq[["game_id", "team", "prev_qb"]], on=["game_id", "team"], how="left")
    out = out.merge(dvp, on=["game_id", "opp", "group"], how="left").merge(dr_all, on=["game_id", "opp"], how="left")
    for c in ("def_ypt", "def_catch", "def_ypc", "def_ypa", "def_comp"):
        out[c] = out[c].fillna(1.0)          # QBs have no defense-vs-position row; they use def_ypa

    # did this quarterback start the game? Official starter where listed, otherwise half the passes.
    # An outcome, known only after the game: used to fit and score the passing projections.
    out["started"] = out["start"]
    if games is not None and {"home_qb_id", "away_qb_id"} <= set(games.columns):
        g = games[["game_id", "home_team", "away_team", "home_qb_id", "away_qb_id"]]
        off = pd.concat([g.rename(columns={"home_team": "team", "home_qb_id": "qb_id"})[["game_id", "team", "qb_id"]],
                         g.rename(columns={"away_team": "team", "away_qb_id": "qb_id"})[["game_id", "team", "qb_id"]]])
        off["team"] = off["team"].replace(C.TEAM_ALIASES)
        out = out.merge(off.dropna(), on=["game_id", "team"], how="left")
        known = out["qb_id"].notna() & out["played"].astype(bool)
        out.loc[known, "started"] = (out.loc[known, "player_id"] == out.loc[known, "qb_id"]).astype(float)
        out = out.drop(columns="qb_id")

    # ---- defensive injuries --------------------------------------------------------
    if inj_groups is not None and len(inj_groups):
        ig = inj_groups[["game_id", "team", "miss_db", "mrec_db", "miss_lb"]].rename(columns={"team": "opp", "miss_db": "opp_miss_db", "mrec_db": "opp_mrec_db", "miss_lb": "opp_miss_lb"})
        out = out.merge(ig, on=["game_id", "opp"], how="left")
    for c in ("opp_miss_db", "opp_mrec_db", "opp_miss_lb"):
        out[c] = out[c].fillna(0.0) if c in out else 0.0
    if pass_rush is not None and len(pass_rush):
        pr = pass_rush[["game_id", "team", "miss_pr"]].rename(columns={"team": "opp", "miss_pr": "opp_miss_pr"})
        out = out.merge(pr, on=["game_id", "opp"], how="left")
    out["opp_miss_pr"] = out["opp_miss_pr"].fillna(0.0) if "opp_miss_pr" in out else 0.0
    return out


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

class Lin:
    """Small ridge regression with an intercept."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        Z = (X - self.mu) / self.sd
        self.y0 = y.mean()
        self.b = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - self.y0))
        return self

    def predict(self, X):
        return self.y0 + ((np.asarray(X, float) - self.mu) / self.sd) @ self.b


VOL_FEATS = ["team_att_avg", "opp_att_allowed", "team_car_avg", "opp_car_allowed", "margin", "proj_total", "wind", "home"]


def _pick_starter(d: pd.DataFrame):
    """Row label of the projected starter among a team's active quarterbacks (sorted by share)."""
    top = d.iloc[0]
    prev = d[d["player_id"] == top["prev_qb"]]
    if len(prev) and not (top["att_share_raw"] >= 0.8 and top["player_id"] != top["prev_qb"]):
        return prev.index[0]
    return d.index[0]


def raw_projection(p: pd.DataFrame, att_model: Lin, car_model: Lin) -> pd.DataFrame:
    """Volume x share x efficiency, before calibration."""
    p = p.copy()
    X = p[VOL_FEATS].fillna(0).to_numpy()
    p["proj_team_att"] = att_model.predict(X).clip(15, 60)
    p["proj_team_car"] = car_model.predict(X).clip(10, 45)
    # shares among the players expected to play in this game
    act = (p["active_pre"] if "active_pre" in p else p["active"]) == 1
    for c in ("tgt_share", "car_share", "att_share"):
        raw = p[f"{c}_raw"].where(act, 0.0)
        tot = raw.groupby([p["game_id"], p["team"]]).transform("sum")
        # vacated usage goes to those available, but not all of it (some goes to players with no history)
        p[c] = np.where(tot > 0, raw / tot.clip(lower=0.85), 0.0)
    # One starting quarterback takes the passes: last game's starter if he is active, unless
    # another active quarterback is clearly the regular (80%+ of the team's recent passes), as when
    # a starter returns from injury; with neither, the active quarterback with the largest share.
    # Right on 92.7% of team-games since 2019 (91.1% for largest share alone).
    qb = p[(p["group"] == "QB") & act].copy()
    if len(qb):
        qb = qb.sort_values("att_share_raw", ascending=False)
        starters = set(qb.groupby(["game_id", "team"]).apply(_pick_starter).dropna().astype(int))
        p["att_share"] = 0.0                     # only the starter projects passes (trick plays are noise)
        p.loc[list(starters), "att_share"] = 1.0
    # QB attempts
    p["pass_att_p"] = p["att_share"] * p["proj_team_att"]
    p["targets_p"] = p["tgt_share"] * p["proj_team_att"]
    p["carries_p"] = p["car_share"] * p["proj_team_car"]
    # Defense-vs-position did not help receiving on the tuning seasons (players_ablation.py),
    # so receivers use their own catch rate and yards per target; QBs and runners keep it.
    p["receptions_p"] = p["targets_p"] * p["catch"].clip(0.2, 0.95)
    p["rec_yds_p"] = p["targets_p"] * p["ypt"]
    p["rush_yds_p"] = p["carries_p"] * p["ypc"] * p["def_ypc"]
    p["completions_p"] = p["pass_att_p"] * (p["comp"] * p["def_comp"]).clip(0.4, 0.8)
    p["pass_yds_p"] = p["pass_att_p"] * p["ypa"] * p["def_ypa"]
    p["ints_p"] = p["pass_att_p"] * p["int_rate"]
    # touchdowns from implied points: about one TD per 7.6 points, split by the team's pass share
    team_td = (p["imp_pts"].clip(3, 45) / 7.6)
    team_ptd, team_rtd = team_td * p["ptd_share"], team_td * (1 - p["ptd_share"])
    rz_t = (p["targets_p"] * p["rz_tgt_rate"])
    rz_c = (p["carries_p"] * (0.5 * p["rz_car_rate"] + 0.5 * p["gl_car_rate"] * 2))
    key = [p["game_id"], p["team"]]
    p["rec_td_p"] = team_ptd * rz_t / rz_t.groupby(key).transform("sum").replace(0, np.nan)
    p["rush_td_p"] = team_rtd * rz_c / rz_c.groupby(key).transform("sum").replace(0, np.nan)
    p["pass_td_p"] = team_ptd * p["att_share"]
    for c in STATS:
        p[c + "_p"] = p[c + "_p"].fillna(0.0)
    return p


PASS_STATS = ("pass_att", "completions", "pass_yds", "pass_td", "ints")

CAL_EXTRA = {   # extra terms in the calibration of each stat (multiplied by the raw projection).
    # Defensive injuries are kept only where they helped on 2013-2018 (receiving); see FEATURES.md.
    "pass_yds": ["wind"], "completions": ["wind"], "pass_td": [],
    "rec_yds": ["opp_miss_db", "opp_miss_pr", "wind"], "receptions": ["opp_miss_db", "wind"], "targets": ["wind"],
    "rush_yds": [], "carries": [], "rec_td": [], "rush_td": [], "ints": [], "pass_att": ["wind"],
}


def calib_X(p: pd.DataFrame, stat: str) -> np.ndarray:
    """The raw projection, its interactions with defensive injuries and wind, and the
    player's recent per-game average (form the volume x efficiency chain can miss)."""
    raw = p[stat + "_p"].to_numpy(float)
    if stat in ("pass_att", "completions", "pass_yds", "pass_td", "ints"):
        # a quarterback's form is his per-start average; with no starts, his raw projection
        per_start = p[stat + "_st_sum"] / p["start_sum"].where(p["start_sum"] > 0.3)
        recent = np.where(p["att_share"] > 0.5, per_start.fillna(p[stat + "_p"]), p[stat + "_p"])
    else:
        recent = (p[stat + "_sum"] / p["one_sum"].replace(0, np.nan)).fillna(0.0).to_numpy(float)
    cols = [raw, recent] + [raw * p[c].to_numpy(float) for c in CAL_EXTRA[stat]]
    return np.column_stack(cols)


def walk_forward(table: pd.DataFrame, first_test: int = 2013) -> pd.DataFrame:
    """Fit volume and calibration on earlier seasons, project each season in turn."""
    out = []
    seasons = sorted(table["season"].unique())
    tg = table.drop_duplicates(["game_id", "team"])
    for s in seasons:
        if s < first_test:
            continue
        tr_t = tg[(tg["season"] < s) & tg["played"].astype(bool) & tg["team_att"].notna()]
        tr_t = tr_t[tr_t["season"] >= s - 8]
        att = Lin(5.0).fit(tr_t[VOL_FEATS].fillna(0), tr_t["team_att"].to_numpy(float))
        car = Lin(5.0).fit(tr_t[VOL_FEATS].fillna(0), tr_t["team_car"].to_numpy(float))
        hist = raw_projection(table[(table["season"] < s) & (table["season"] >= max(first_test - 1, s - 6))], att, car)
        hist = hist[hist["played"].astype(bool) & (hist["active"] == 1)]
        cur = raw_projection(table[table["season"] == s], att, car)
        # Passing is fitted on projected starters who did start. A projected starter who does not
        # start (benched, or a returning starter's backup who keeps the job) throws next to nothing,
        # and fitting on those games shrank every starter's projection: starters ran about 17 yards
        # low. A passing prop is void if the quarterback does not play, so the projection is what he
        # throws if he starts.
        qb_fit = hist[(hist["att_share"] > 0.5) & (hist["started"] == 1)]
        for stat in STATS:
            h = qb_fit if stat in PASS_STATS else hist
            m = Lin(1.0).fit(calib_X(h, stat), h[stat].to_numpy(float))
            X = calib_X(cur, stat)
            # no projected volume and no history for this stat: zero, not the regression's intercept
            none = (X[:, 0] == 0) & (X[:, 1] == 0)
            cur[stat + "_proj"] = np.where(none, 0.0, np.clip(m.predict(X), 0, None))
        out.append(cur)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Board output
# --------------------------------------------------------------------------

ROSTER_LABEL = {"RES": "Injured reserve", "PUP": "PUP list", "NON": "Non-football injury", "INA": "Inactive", "DEV": "Practice squad",
                "CUT": "Released", "EXE": "Exempt", "SUS": "Suspended", "TRD": "Traded", "TRC": "Traded", "RET": "Retired", "RSN": "Reserve"}


def status_label(roster_status, report, upcoming: bool = False) -> str | None:
    """Roster status wins when the player is not active; otherwise the injury report.
    For a game not yet played, last week's game-day inactive list says nothing."""
    if upcoming and roster_status == "INA":
        roster_status = "ACT"
    if isinstance(roster_status, str) and roster_status != "ACT":
        return ROSTER_LABEL.get(roster_status, roster_status)
    return report if isinstance(report, str) else None


def board(proj: pd.DataFrame, info: pd.DataFrame, game_ids: set) -> dict:
    """Projections for the board's upcoming games, with the inputs behind them."""
    p = proj[proj["game_id"].isin(game_ids)].copy()
    if p.empty:
        return {"players": [], "teams": []}
    p["name"] = p["player_id"].map(info["name"]).fillna(p["player_id"])
    r1 = lambda x: None if pd.isna(x) else round(float(x), 1)
    r2 = lambda x: None if pd.isna(x) else round(float(x), 2)
    r3 = lambda x: None if pd.isna(x) else round(float(x), 3)

    # teammates who are out and the usage they leave behind
    teams = []
    for (gid, team), t in p.groupby(["game_id", "team"]):
        out = t[(t["active"] == 0) & ((t["tgt_share_raw"] >= 0.08) | (t["car_share_raw"] >= 0.15) | (t["att_share_raw"] >= 0.5))]
        first = t.iloc[0]
        teams.append({
            "game": gid, "team": team, "opp": first["opp"],
            "pass_att": r1(first["proj_team_att"]), "carries": r1(first["proj_team_car"]),
            "imp_pts": r1(first["imp_pts"]), "margin": r1(first["margin"]), "wind": r1(first["wind"]),
            "def_ypa": r3(first["def_ypa"]), "def_ypc": r3(first["def_ypc"]),
            "opp_miss_db": r2(first["opp_miss_db"]), "opp_miss_pr": r2(first["opp_miss_pr"]),
            "out": [{"name": info["name"].get(o.player_id, o.player_id), "pos": o.position, "status": status_label(o.status, o.report, upcoming=True) or "Out",
                     "tgt_share": r3(o.tgt_share_raw), "car_share": r3(o.car_share_raw)} for o in out.itertuples()],
        })

    a = p[(p["active"] == 1) & ((p["targets_proj"] >= 1.5) | (p["carries_proj"] >= 2) | ((p["group"] == "QB") & (p["att_share"] >= 0.5)))]
    players = []
    for r in a.itertuples():
        td = (r.rec_td_proj or 0) + (r.rush_td_proj or 0)
        players.append({
            "id": r.player_id, "name": r.name, "pos": r.position, "team": r.team, "opp": r.opp, "game": r.game_id, "week": int(r.week),
            "status": status_label(r.status, r.report, upcoming=True),
            "proj": {k: r1(getattr(r, k + "_proj")) for k in STATS},
            "td_any": r3(1 - np.exp(-max(td, 0))),
            "why": {"tgt_share": r3(r.tgt_share), "car_share": r3(r.car_share), "att_share": r3(r.att_share),
                    "tgt_share_hist": r3(r.tgt_share_raw), "car_share_hist": r3(r.car_share_raw),
                    "ypt": r2(r.ypt), "catch": r3(r.catch), "ypc": r2(r.ypc), "ypa": r2(r.ypa), "comp": r3(r.comp),
                    "def_ypt": r3(r.def_ypt), "def_ypc": r3(r.def_ypc), "def_ypa": r3(r.def_ypa),
                    "games": r1(r.one_sum)},
        })
    players.sort(key=lambda x: -(x["proj"]["rec_yds"] or 0) - (x["proj"]["rush_yds"] or 0) - (x["proj"]["pass_yds"] or 0) / 2.5)
    return {"players": players, "teams": teams}


# --------------------------------------------------------------------------
# Backtest summary for the board
# --------------------------------------------------------------------------
EVAL_SETS = {   # stat -> (label, position groups, QBs: projected starter who started)
    "pass_yds": ("Passing yards", ("QB",), True), "pass_td": ("Passing TDs", ("QB",), True),
    "rush_yds": ("Rushing yards", ("RB", "QB"), False), "carries": ("Carries", ("RB", "QB"), False),
    "receptions": ("Receptions", ("WR", "TE", "RB"), False), "rec_yds": ("Receiving yards", ("WR", "TE", "RB"), False),
}


def season_average(table: pd.DataFrame) -> pd.DataFrame:
    """Each player's average in earlier played, active games this season (last season's in week 1);
    for passing, in games he started."""
    t = table[table["active"] == 1].sort_values(["player_id", "t"]).copy()
    played = t["played"].astype(bool)
    out = t[["game_id", "player_id"]].copy()
    for st in STATS:
        v = t[st].where(played)
        if st in PASS_STATS:          # quarterbacks: per game started
            v = v.where(t["started"] == 1)
        out[st + "_savg"] = v.groupby([t["player_id"], t["season"]]).transform(lambda s: s.shift().expanding().mean()).to_numpy()
        last = v.groupby([t["player_id"], t["season"]]).mean()
        prev = pd.Series(list(zip(t["player_id"], t["season"] - 1))).map(last).to_numpy()
        out[st + "_savg"] = out[st + "_savg"].fillna(pd.Series(prev, index=out.index)).fillna(0.0)
    return out


def backtest_summary(proj: pd.DataFrame, table: pd.DataFrame, holdout_from: int) -> dict:
    p = proj.merge(season_average(table), on=["game_id", "player_id"], how="left")
    ev = p[p["played"].astype(bool) & (p["active"] == 1) & (p["season"] >= holdout_from)]
    rows = []
    for st, (label, groups, main_qb) in EVAL_SETS.items():
        x = ev[ev["group"].isin(groups)]
        if main_qb:   # the projected starter, in games he started (a passing prop is void otherwise)
            x = x[(x["att_share"] > 0.5) & (x["started"] == 1) & (x["start_sum"] > 0.3) & (x[st + "_savg"] > 0)]
        x = x[(x[st + "_savg"] > 0) | (x[st] > 0)]
        recent = (x[st + "_st_sum"] / x["start_sum"].where(x["start_sum"] > 0.3) if st in PASS_STATS
                  else x[st + "_sum"] / x["one_sum"].replace(0, np.nan))
        mae = lambda pred: round(float(np.mean(np.abs(pred.fillna(0) - x[st]))), 3)
        rows.append({"stat": label, "n": int(len(x)), "model": mae(x[st + "_proj"]), "season_avg": mae(x[st + "_savg"]), "recent": mae(recent)})
    td = ev[ev["group"].isin(["WR", "TE", "RB"]) & ((ev["rec_td_savg"] + ev["rush_td_savg"] > 0) | (ev["rec_td"] + ev["rush_td"] > 0))]
    hit = ((td["rec_td"] + td["rush_td"]) > 0).astype(float)
    brier = lambda lam: round(float(np.mean((1 - np.exp(-lam.fillna(0).clip(lower=0)) - hit) ** 2)), 4)
    recent_td = (td["rec_td_sum"] + td["rush_td_sum"]) / td["one_sum"].replace(0, np.nan)
    tdrow = {"stat": "Anytime TD (Brier score)", "n": int(len(td)), "model": brier(td["rec_td_proj"] + td["rush_td_proj"]),
             "season_avg": brier(td["rec_td_savg"] + td["rush_td_savg"]), "recent": brier(recent_td)}
    return {"from": holdout_from, "rows": rows + [tdrow]}
