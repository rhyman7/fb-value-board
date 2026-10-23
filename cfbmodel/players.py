"""College player projections: passing, rushing and receiving, game by game.

The NFL board's method (nflmodel/players.py), opportunity times efficiency, with what
public college data allows:

  team volume    pass attempts and carries per game from the team's recent volume, what
                 the opponent has allowed, the game model's projected margin and total, and
                 home field (a regression fitted on earlier seasons)
  player share   share of team carries, pass attempts and receptions in games the player
                 appeared, recent games weighted more. Targets are not used: the play-by-play
                 is missing receivers on many incompletions in 2021-2024.
  availability   there is no official college injury report or game-day inactive list. In
                 past games (the backtests), a player is expected to play if he recorded a
                 stat in his team's previous game (in a team's first game, if he played for
                 it last season). For upcoming games, a regular who sat out is expected back
                 if he played in one of the team's last three games, and the Covers.com
                 injury designations (cfbmodel/designations.py) take out anyone listed Out or
                 Doubtful. Players who were expected to play leave their share to teammates
                 if they are not.
  efficiency     completion rate, yards per attempt, TD and interception rates, yards per
                 carry and per catch, shrunk toward the position average; passing and rushing
                 multiplied by what this defense allows compared with the league
  touchdowns     team touchdowns from the game model's implied points, split into passing and
                 rushing by the team's tendency, shared out by each player's usage and his
                 touchdown rate
  calibration    a last regression per stat on earlier seasons that blends the chain with the
                 player's recent per-game average

Stats are official box scores, which in college count sacks as quarterback runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflmodel.players import Lin, _shrink, asof

from . import config as C
from . import designations as DES
from .players_data import GROUP

SEASON_LEN = 22
TEAM_HL, PLAYER_HL, DEF_HL = 6.0, 5.0, 8.0     # half-lives in games (college seasons are shorter)
PRIORS = {"ypr": 25.0, "ypc": 60.0, "ypa": 150.0, "comp": 150.0, "ptd": 300.0, "int": 400.0, "def_ypc": 120.0, "def_ypa": 250.0}
STATS = ["pass_att", "completions", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td",
         "receptions", "rec_yds", "rec_td"]
PASS_STATS = ("pass_att", "completions", "pass_yds", "pass_td", "ints")
VOL_FEATS = ["team_att_avg", "opp_att_allowed", "team_car_avg", "opp_car_allowed", "margin", "proj_total", "home", "wind"]


def _t(season, week):
    return np.asarray(season) * SEASON_LEN + np.asarray(week)


def _week(games: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(games["postseason"] == 1, C.POSTSEASON_WEEK, games["week"]), index=games.index)


def team_games(games: pd.DataFrame, frame: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """One row per FBS team per game (all games, played and upcoming): game-model context where
    the game is in the model frame, and team volume for played games."""
    g = games[["game_id", "season", "home_team", "away_team", "home_id", "away_id", "neutral", "played", "wind", "indoor"]].copy()
    g["wind"] = pd.to_numeric(g["wind"], errors="coerce").where(g["indoor"] == 0, 0.0)
    g["week"] = _week(games)
    ctx = frame[["game_id", "proj_margin", "proj_total"]] if len(frame) else pd.DataFrame(columns=["game_id", "proj_margin", "proj_total"])
    g = g.merge(ctx, on="game_id", how="left")
    # seasons before the game model's first test season: the market's numbers stand in, so the
    # player model has earlier seasons to fit on
    early = g["proj_margin"].isna() & (g["season"] < C.FIRST_TEST_SEASON)
    lines = games.set_index("game_id")
    g.loc[early, "proj_margin"] = g.loc[early, "game_id"].map(lines["spread_line"]).to_numpy()
    g.loc[early, "proj_total"] = g.loc[early, "game_id"].map(lines["total_line"]).to_numpy()
    if "wind" in frame:      # the frame's wind is cleaned (indoor games 0, blanks filled)
        fw = frame.set_index("game_id")["wind"]
        g["wind"] = g["game_id"].map(fw).fillna(g["wind"]).fillna(7.0)
    cols = ["game_id", "season", "week", "neutral", "played", "wind", "indoor", "proj_margin", "proj_total"]
    h = g[cols + ["home_team", "away_team", "home_id"]].rename(columns={"home_team": "team", "away_team": "opp", "home_id": "team_id"}).assign(sign=1.0, home=1 - g["neutral"])
    a = g[cols + ["away_team", "home_team", "away_id"]].rename(columns={"away_team": "team", "home_team": "opp", "away_id": "team_id"}).assign(sign=-1.0, home=0)
    tg = pd.concat([h, a], ignore_index=True)
    tg = tg[tg["team"] != C.POOL]
    tg["margin"] = tg["proj_margin"] * tg["sign"]
    tg["imp_pts"] = (tg["proj_total"] + tg["margin"]) / 2
    vol = pg.groupby(["game_id", "team_id"])[["pass_att", "completions", "carries", "pass_td", "rush_td"]].sum().reset_index()
    vol = vol.rename(columns={"pass_att": "team_att", "completions": "team_comp", "carries": "team_car", "pass_td": "team_ptd", "rush_td": "team_rtd"})
    tg = tg.merge(vol, on=["game_id", "team_id"], how="left")
    tg.loc[~tg["played"].astype(bool), ["team_att", "team_comp", "team_car", "team_ptd", "team_rtd"]] = np.nan
    tg["t"] = _t(tg["season"], tg["week"])
    off = asof(tg, "team", ["team_att", "team_comp", "team_car", "team_ptd", "team_rtd"], None, TEAM_HL)
    nz = off["n_games"].replace(0, np.nan)
    off["team_att_avg"], off["team_car_avg"] = off["team_att_sum"] / nz, off["team_car_sum"] / nz
    off["team_comp_rate"] = (off["team_comp_sum"] + 0.6 * 60) / (off["team_att_sum"] + 60)
    off["ptd_share"] = (off["team_ptd_sum"] + 6) / (off["team_ptd_sum"] + off["team_rtd_sum"] + 10)
    # what each defense has allowed: the opponents' volume in its earlier games
    allowed = tg[["game_id", "opp", "team_att", "team_car", "season", "week", "t"]].rename(columns={"opp": "dteam"})
    allowed = allowed[allowed["dteam"] != C.POOL]
    d = asof(allowed, "dteam", ["team_att", "team_car"], None, TEAM_HL)
    dn = d["n_games"].replace(0, np.nan)
    d["opp_att_allowed"], d["opp_car_allowed"] = d["team_att_sum"] / dn, d["team_car_sum"] / dn
    keep = ["game_id", "team", "team_id", "opp", "season", "week", "t", "margin", "proj_total", "imp_pts", "wind", "indoor", "played",
            "home", "team_att", "team_comp", "team_car", "team_ptd", "team_rtd", "team_att_avg", "team_car_avg", "team_comp_rate", "ptd_share"]
    tg = off[keep].merge(d[["game_id", "dteam", "opp_att_allowed", "opp_car_allowed"]].rename(columns={"dteam": "opp"}), on=["game_id", "opp"], how="left")
    lg_att, lg_car = tg["team_att"].mean(), tg["team_car"].mean()
    for c, lg in (("team_att_avg", lg_att), ("opp_att_allowed", lg_att), ("team_car_avg", lg_car), ("opp_car_allowed", lg_car)):
        tg[c] = tg[c].fillna(lg)
    tg["ptd_share"] = tg["ptd_share"].fillna(0.55)
    tg["team_comp_rate"] = tg["team_comp_rate"].fillna(0.6)
    return tg


def player_rows(tg: pd.DataFrame, pg: pd.DataFrame, info: pd.DataFrame, des_out: pd.DataFrame | None = None,
                starters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every offensive player of each team for each game: those with a stat in the game, plus
    those expected to play (a stat in the team's previous game; in its first game, last season;
    for upcoming games also the regulars back from a missed game). des_out: Out and Doubtful
    injury designations (designations.outs), applied to upcoming games only."""
    off = pg[(pg[["pass_att", "carries", "receptions"]].sum(axis=1) > 0)]
    stats = off[["game_id", "team_id", "player_id"] + STATS]
    tg = tg.sort_values(["team", "t"]).reset_index(drop=True)
    played = tg[tg["played"].astype(bool)]
    # who appeared in each team's previous played game
    seq = played[["game_id", "team", "team_id", "season", "t"]].copy()
    seq["prev_gid"] = seq.groupby("team")["game_id"].shift()
    seq["prev_season"] = seq.groupby("team")["season"].shift()
    nxt = tg[["game_id", "team", "season", "t"]].merge(seq[["team", "t", "game_id"]].rename(columns={"t": "pt", "game_id": "last_gid"}), on="team")
    nxt = nxt[nxt["pt"] < nxt["t"]].sort_values("pt").groupby(["game_id", "team"]).tail(1)
    nxt = nxt.merge(played[["game_id", "season"]].rename(columns={"game_id": "last_gid", "season": "last_season"}), on="last_gid")
    app = stats[["game_id", "team_id", "player_id"]].rename(columns={"game_id": "last_gid"})
    exp = nxt.merge(tg[["game_id", "team", "team_id"]], on=["game_id", "team"]).merge(app, on=["last_gid", "team_id"])
    exp = exp[["game_id", "team_id", "player_id", "season", "last_season"]]
    # first game of a season: players who appeared for the team last season
    first = exp["season"] != exp["last_season"]
    if first.any():
        ls = stats.merge(tg[["game_id", "team_id", "season"]], on=["game_id", "team_id"])
        lastseason = ls[["team_id", "season", "player_id"]].drop_duplicates()
        f0 = exp.loc[first, ["game_id", "team_id", "season", "last_season"]].drop_duplicates(["game_id", "team_id"])
        f0 = f0.merge(lastseason.rename(columns={"season": "last_season"}), on=["team_id", "last_season"])
        exp = pd.concat([exp[~first], f0[["game_id", "team_id", "player_id", "season", "last_season"]]], ignore_index=True)
    exp = pd.concat([exp[["game_id", "team_id", "player_id"]], _regulars_back(tg, stats)], ignore_index=True)
    if starters is not None and len(starters):      # a named starter plays even if he sat out last game
        st = starters[starters["qb_id"] != "replacement"].merge(tg[["game_id", "team", "team_id"]], on=["game_id", "team"])
        exp = pd.concat([exp, st[["game_id", "team_id"]].assign(player_id=st["qb_id"].astype(np.int64).to_numpy())], ignore_index=True)
    exp = exp.drop_duplicates().assign(expected=1)
    rows = stats.merge(exp, on=["game_id", "team_id", "player_id"], how="outer")
    rows["appeared"] = rows["pass_att"].notna().astype(int)
    rows["expected"] = rows["expected"].fillna(0).astype(int)
    rows[STATS] = rows[STATS].fillna(0.0)
    rows = rows.merge(tg[["game_id", "team_id", "team", "opp", "season", "week", "t", "team_att", "team_comp", "team_car", "played"]],
                      on=["game_id", "team_id"], how="inner")
    pos = rows["player_id"].map(info["position"])
    grp = pos.map(GROUP)
    # unknown or non-skill listings: classify by usage
    hist = pg.groupby("player_id")[["pass_att", "carries", "receptions"]].sum()
    guess = np.where(hist["pass_att"] > hist["carries"], "QB", np.where(hist["carries"] > hist["receptions"], "RB", "WR"))
    guess = pd.Series(guess, index=hist.index)
    rows["group"] = grp.fillna(rows["player_id"].map(guess))
    rows["position"] = pos.where(grp.notna(), rows["group"])
    # active: expected to play (for an upcoming game, the only information), or in the box score
    rows["active"] = np.where(rows["played"].astype(bool), ((rows["expected"] == 1) | (rows["appeared"] == 1)).astype(int), rows["expected"])
    rows["report"] = None
    if des_out is not None and len(des_out):
        cur = rows.loc[rows["played"].astype(bool), "season"].max()
        up = ~rows["played"].astype(bool) & (rows["season"] == cur)      # this season's games still to play
        po = DES.player_outs(rows[up], des_out, info)
        if len(po):
            hit = rows[["game_id", "player_id"]].merge(po[["game_id", "player_id", "status"]], on=["game_id", "player_id"], how="left")["status"].to_numpy()
            m = up.to_numpy() & pd.notna(hit)
            rows.loc[m, "active"] = 0
            rows.loc[m, "report"] = hit[m]
    return rows


def _regulars_back(tg: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team_id, player_id) for upcoming games: this season's regulars (two or more
    games, a large share of carries, catches or passes) who played in one of the team's last
    three games. A starter who sat out one game is expected back unless listed out."""
    cols = ["game_id", "team_id", "player_id"]
    done = tg[tg["played"].astype(bool)][["game_id", "team_id", "season", "t", "team_car", "team_comp", "team_att"]]
    if done.empty:
        return pd.DataFrame(columns=cols)
    cur = done["season"].max()
    up = tg[~tg["played"].astype(bool) & (tg["season"] == cur)][["game_id", "team_id", "season"]]
    done = done[done["season"] == cur]
    if up.empty:
        return pd.DataFrame(columns=cols)
    st = stats.merge(done, on=["game_id", "team_id"])
    if st.empty:
        return pd.DataFrame(columns=cols)
    st["car_s"] = st["carries"] / st["team_car"].replace(0, np.nan)
    st["rec_s"] = st["receptions"] / st["team_comp"].replace(0, np.nan)
    st["att_s"] = st["pass_att"] / st["team_att"].replace(0, np.nan)
    last3 = done.sort_values("t").groupby(["team_id", "season"]).tail(3)[["game_id", "team_id"]].assign(recent=1)
    st = st.merge(last3, on=["game_id", "team_id"], how="left")
    g = st.groupby(["team_id", "season", "player_id"]).agg(n=("game_id", "nunique"), car=("car_s", "mean"), rec=("rec_s", "mean"),
                                                          att=("att_s", "mean"), recent=("recent", "max")).reset_index()
    reg = g[(g["n"] >= 2) & (g["recent"] == 1) & ((g["car"] >= 0.2) | (g["rec"] >= 0.12) | (g["att"] >= 0.5))]
    return up.merge(reg[["team_id", "season", "player_id"]], on=["team_id", "season"])[cols]


def build(games: pd.DataFrame, frame: pd.DataFrame, pg: pd.DataFrame, info: pd.DataFrame,
          des_out: pd.DataFrame | None = None, starters: pd.DataFrame | None = None) -> pd.DataFrame:
    """starters (optional): this week's starting quarterbacks (starters.resolve); the named
    quarterback takes the team's passes in that game."""
    tg = team_games(games, frame, pg)
    rows = player_rows(tg, pg, info, des_out, starters)
    act = rows.copy()
    use = (act["appeared"] == 1) & act["played"].astype(bool)
    act["car_share"] = np.where(act["team_car"] > 0, act["carries"] / act["team_car"], np.nan)
    act["att_share"] = np.where(act["team_att"] > 0, act["pass_att"] / act["team_att"], np.nan)
    act["rec_share"] = np.where(act["team_comp"] > 0, act["receptions"] / act["team_comp"], np.nan)
    act["one"] = np.where(use, 1.0, np.nan)
    act["start"] = np.where(use, (act["pass_att"] >= 0.5 * act["team_att"].clip(lower=1)).astype(float), np.nan)
    for c in PASS_STATS:
        act[c + "_st"] = np.where(use, act[c] * act["start"], np.nan)
    hist_cols = ["car_share", "att_share", "rec_share"] + STATS
    for c in hist_cols:
        act[c] = act[c].where(use)
    pl = asof(act, "player_id", hist_cols + [c + "_st" for c in PASS_STATS], ["one", "start"], PLAYER_HL)
    for c in STATS:      # restore the outcomes (asof works on the masked copies)
        pl[c] = pl[c].fillna(0.0)
    n = pl["one_sum"]
    pos_share = act[use].groupby("group")[["car_share", "att_share", "rec_share"]].mean()
    for c in ("car_share", "att_share", "rec_share"):
        prior = pl["group"].map(pos_share[c]).fillna(0.0) * 0.5
        pl[f"{c}_raw"] = (pl[f"{c}_sum"] + prior * 1.0) / (n + 1.0)
    tot = act[use].groupby("group")[["receptions", "rec_yds", "rec_td", "carries", "rush_yds", "rush_td", "pass_att", "completions", "pass_yds", "pass_td", "ints"]].sum()
    pm = lambda a, b: (tot[a] / tot[b].replace(0, np.nan)).fillna(0)
    g = pl["group"]
    pl["ypr"] = _shrink(pl["rec_yds_sum"], pl["receptions_sum"], g.map(pm("rec_yds", "receptions")), PRIORS["ypr"])
    pl["ypc"] = _shrink(pl["rush_yds_sum"], pl["carries_sum"], g.map(pm("rush_yds", "carries")), PRIORS["ypc"])
    qb = lambda a: tot.loc["QB", a] if "QB" in tot.index else np.nan
    pl["comp"] = _shrink(pl["completions_sum"], pl["pass_att_sum"], qb("completions") / qb("pass_att"), PRIORS["comp"])
    pl["ypa"] = _shrink(pl["pass_yds_sum"], pl["pass_att_sum"], qb("pass_yds") / qb("pass_att"), PRIORS["ypa"])
    pl["ptd_rate"] = _shrink(pl["pass_td_sum"], pl["pass_att_sum"], qb("pass_td") / qb("pass_att"), PRIORS["ptd"])
    pl["int_rate"] = _shrink(pl["ints_sum"], pl["pass_att_sum"], qb("ints") / qb("pass_att"), PRIORS["int"])
    pl["rtd_rate"] = _shrink(pl["rec_td_sum"], pl["receptions_sum"], g.map(pm("rec_td", "receptions")), 30.0)
    pl["ctd_rate"] = _shrink(pl["rush_td_sum"], pl["carries_sum"], g.map(pm("rush_td", "carries")), 60.0)

    # defense: yards per carry and per pass attempt allowed, against the league
    dgames = tg[["game_id", "opp", "season", "week", "t", "played"]].drop_duplicates(["game_id", "opp"])
    done = rows[rows["played"].astype(bool) & (rows["appeared"] == 1)]
    ru = done.groupby(["game_id", "opp"])[["carries", "rush_yds", "pass_att", "completions", "pass_yds"]].sum().reset_index()
    dg = dgames.merge(ru, on=["game_id", "opp"], how="left")
    dg.loc[~dg["played"].astype(bool), ["carries", "rush_yds", "pass_att", "completions", "pass_yds"]] = np.nan
    dr = asof(dg, "opp", ["carries", "rush_yds", "pass_att", "completions", "pass_yds"], None, DEF_HL)
    lgr = ru[["carries", "rush_yds", "pass_att", "completions", "pass_yds"]].sum()
    dr["def_ypc"] = _shrink(dr["rush_yds_sum"], dr["carries_sum"], lgr["rush_yds"] / lgr["carries"], PRIORS["def_ypc"]) / (lgr["rush_yds"] / lgr["carries"])
    dr["def_ypa"] = _shrink(dr["pass_yds_sum"], dr["pass_att_sum"], lgr["pass_yds"] / lgr["pass_att"], PRIORS["def_ypa"]) / (lgr["pass_yds"] / lgr["pass_att"])
    dr["def_comp"] = _shrink(dr["completions_sum"], dr["pass_att_sum"], lgr["completions"] / lgr["pass_att"], PRIORS["def_ypa"]) / (lgr["completions"] / lgr["pass_att"])

    # last game's starting quarterback
    top = done[done["pass_att"] > 0].sort_values("pass_att").groupby(["game_id", "team"]).tail(1)[["game_id", "team", "player_id"]]
    seq = tg[["game_id", "team", "t"]].merge(top.rename(columns={"player_id": "qb_this"}), on=["game_id", "team"], how="left").sort_values(["team", "t"])
    seq["prev_qb"] = seq.groupby("team")["qb_this"].transform(lambda s: s.ffill().shift())
    out = pl.merge(tg.drop(columns=["team_att", "team_comp", "team_car", "played", "team_id"]), on=["game_id", "team", "opp", "season", "week", "t"], how="left")
    out = out.merge(seq[["game_id", "team", "prev_qb"]], on=["game_id", "team"], how="left")
    out = out.merge(dr[["game_id", "opp", "def_ypc", "def_ypa", "def_comp"]], on=["game_id", "opp"], how="left")
    for c in ("def_ypc", "def_ypa", "def_comp"):
        out[c] = out[c].fillna(1.0)
    out["forced_qb"] = None
    if starters is not None and len(starters):
        f = starters[starters["qb_id"] != "replacement"].set_index(["game_id", "team"])["qb_id"].astype(np.int64)
        out["forced_qb"] = [f.get((g, t)) for g, t in zip(out["game_id"], out["team"])]
    out["started"] = out["start"]      # did he start (half the team's passes)? an outcome, used to fit and score passing
    return out


def _pick_starter(d: pd.DataFrame):
    top = d.iloc[0]
    forced = top.get("forced_qb") if "forced_qb" in d else None
    if forced is not None and pd.notna(forced):     # this week's named starter (starters.py)
        hit = d[d["player_id"] == forced]
        if len(hit):
            return hit.index[0]
    prev = d[d["player_id"] == top["prev_qb"]]
    if len(prev) and not (top["att_share_raw"] >= 0.8 and top["player_id"] != top["prev_qb"]):
        return prev.index[0]
    return d.index[0]


def raw_projection(p: pd.DataFrame, att_model: Lin, car_model: Lin) -> pd.DataFrame:
    p = p.copy()
    X = p[VOL_FEATS].fillna(0).to_numpy()
    p["proj_team_att"] = att_model.predict(X).clip(12, 65)
    p["proj_team_car"] = car_model.predict(X).clip(15, 65)
    act = p["active"] == 1
    if "forced_qb" in p:                             # a named starter's teammates at QB are backups: no designed runs
        fq = pd.to_numeric(p["forced_qb"], errors="coerce")
        backup = fq.notna() & (p["group"] == "QB") & (p["player_id"] != fq)
        p["car_share_raw"] = p["car_share_raw"].where(~backup, 0.0)
    for c in ("car_share", "rec_share"):
        raw = p[f"{c}_raw"].where(act, 0.0)
        tot = raw.groupby([p["game_id"], p["team"]]).transform("sum")
        p[c] = np.where(tot > 0, raw / tot.clip(lower=0.85), 0.0)
    p["att_share"] = 0.0
    qb = p[(p["group"] == "QB") & act].sort_values("att_share_raw", ascending=False)
    if len(qb):
        starters = [ix for ix in qb.groupby(["game_id", "team"]).apply(_pick_starter).dropna().astype(int)]
        p.loc[starters, "att_share"] = 1.0
    p["pass_att_p"] = p["att_share"] * p["proj_team_att"]
    p["carries_p"] = p["car_share"] * p["proj_team_car"]
    p["receptions_p"] = p["rec_share"] * p["proj_team_att"] * p["team_comp_rate"]
    p["rec_yds_p"] = p["receptions_p"] * p["ypr"]
    p["rush_yds_p"] = p["carries_p"] * p["ypc"] * p["def_ypc"]
    p["completions_p"] = p["pass_att_p"] * (p["comp"] * p["def_comp"]).clip(0.35, 0.8)
    p["pass_yds_p"] = p["pass_att_p"] * p["ypa"] * p["def_ypa"]
    p["ints_p"] = p["pass_att_p"] * p["int_rate"]
    team_td = p["imp_pts"].fillna(28).clip(3, 60) / 7.4
    team_ptd, team_rtd = team_td * p["ptd_share"], team_td * (1 - p["ptd_share"])
    rz_r = p["receptions_p"] * p["rtd_rate"]
    rz_c = p["carries_p"] * p["ctd_rate"]
    key = [p["game_id"], p["team"]]
    p["rec_td_p"] = team_ptd * rz_r / rz_r.groupby(key).transform("sum").replace(0, np.nan)
    p["rush_td_p"] = team_rtd * rz_c / rz_c.groupby(key).transform("sum").replace(0, np.nan)
    p["pass_td_p"] = team_ptd * p["att_share"]
    for c in STATS:
        p[c + "_p"] = p[c + "_p"].fillna(0.0)
    return p


CAL_EXTRA = {s: [] for s in STATS}


def calib_X(p: pd.DataFrame, stat: str) -> np.ndarray:
    raw = p[stat + "_p"].to_numpy(float)
    if stat in PASS_STATS:
        per_start = p[stat + "_st_sum"] / p["start_sum"].where(p["start_sum"] > 0.3)
        recent = np.where(p["att_share"] > 0.5, per_start.fillna(p[stat + "_p"]), p[stat + "_p"])
    else:
        recent = (p[stat + "_sum"] / p["one_sum"].replace(0, np.nan)).fillna(0.0).to_numpy(float)
    cols = [raw, recent] + [raw * p[c].to_numpy(float) for c in CAL_EXTRA[stat]]
    return np.column_stack(cols)


def walk_forward(table: pd.DataFrame, first_test: int = C.FIRST_TEST_SEASON, lookback: int = 6) -> pd.DataFrame:
    """Fit team volume and calibration on earlier seasons and project each season in turn.
    Only games in the game-model frame (FBS against FBS) have projections."""
    out = []
    tg = table.drop_duplicates(["game_id", "team"])
    tg = tg[tg["proj_total"].notna()]
    for s in sorted(table["season"].unique()):
        if s < first_test:
            continue
        tr = tg[(tg["season"] < s) & (tg["season"] >= s - 8) & tg["played"].astype(bool) & tg["team_att"].notna()]
        att = Lin(5.0).fit(tr[VOL_FEATS].fillna(0), tr["team_att"].to_numpy(float))
        car = Lin(5.0).fit(tr[VOL_FEATS].fillna(0), tr["team_car"].to_numpy(float))
        past = table[(table["season"] < s) & (table["season"] >= max(first_test - 2, s - lookback)) & table["proj_total"].notna()]
        hist = raw_projection(past, att, car)
        hist = hist[hist["played"].astype(bool) & (hist["appeared"] == 1)]
        cur = raw_projection(table[(table["season"] == s) & table["proj_total"].notna()], att, car)
        qb_fit = hist[(hist["att_share"] > 0.5) & (hist["started"] == 1)]
        for stat in STATS:
            h = qb_fit if stat in PASS_STATS else hist
            m = Lin(1.0).fit(calib_X(h, stat), h[stat].to_numpy(float))
            X = calib_X(cur, stat)
            none = (X[:, 0] == 0) & (X[:, 1] == 0)
            cur[stat + "_proj"] = np.where(none, 0.0, np.clip(m.predict(X), 0, None))
        out.append(cur)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

EVAL_SETS = {
    "pass_yds": ("Passing yards", ("QB",), True), "pass_att": ("Pass attempts", ("QB",), True),
    "completions": ("Completions", ("QB",), True), "pass_td": ("Passing TDs", ("QB",), True),
    "rush_yds": ("Rushing yards", ("RB", "QB"), False), "carries": ("Carries", ("RB", "QB"), False),
    "receptions": ("Receptions", ("WR", "TE", "RB"), False), "rec_yds": ("Receiving yards", ("WR", "TE", "RB"), False),
}


def season_average(table: pd.DataFrame) -> pd.DataFrame:
    """Each player's average in earlier games this season in which he appeared (last season's in
    his first game); for passing, in games he started."""
    t = table[table["appeared"] == 1].sort_values(["player_id", "t"]).copy()
    played = t["played"].astype(bool)
    out = t[["game_id", "player_id"]].copy()
    for st in STATS:
        v = t[st].where(played)
        if st in PASS_STATS:
            v = v.where(t["started"] == 1)
        out[st + "_savg"] = v.groupby([t["player_id"], t["season"]]).transform(lambda s: s.shift().expanding().mean()).to_numpy()
        last = v.groupby([t["player_id"], t["season"]]).mean()
        prev = pd.Series(list(zip(t["player_id"], t["season"] - 1))).map(last).to_numpy()
        out[st + "_savg"] = out[st + "_savg"].fillna(pd.Series(prev, index=out.index)).fillna(0.0)
    return out


def backtest_summary(proj: pd.DataFrame, table: pd.DataFrame, periods: dict) -> dict:
    """Average miss per player-game for the model, the season average and the recent average.
    Scored on players who appeared in the game (a prop is void if he does not play) and, for
    passing, on the projected starter in games he started."""
    p = proj.merge(season_average(table), on=["game_id", "player_id"], how="left")
    res = {}
    for name, (lo, hi) in periods.items():
        ev = p[p["played"].astype(bool) & (p["appeared"] == 1) & p["season"].between(lo, hi)]
        rows = []
        for st, (label, groups, main_qb) in EVAL_SETS.items():
            x = ev[ev["group"].isin(groups)]
            if main_qb:
                x = x[(x["att_share"] > 0.5) & (x["started"] == 1) & (x["start_sum"] > 0.3) & (x[st + "_savg"] > 0)]
            x = x[(x[st + "_savg"] > 0) | (x[st] > 0)]
            recent = (x[st + "_st_sum"] / x["start_sum"].where(x["start_sum"] > 0.3) if st in PASS_STATS
                      else x[st + "_sum"] / x["one_sum"].replace(0, np.nan))
            mae = lambda pred: round(float(np.mean(np.abs(pred.fillna(0) - x[st]))), 3)
            rows.append({"stat": label, "n": int(len(x)), "model": mae(x[st + "_proj"]), "season_avg": mae(x[st + "_savg"]),
                         "recent": mae(recent), "bias": round(float((x[st + "_proj"] - x[st]).mean()), 2)})
        td = ev[ev["group"].isin(["WR", "TE", "RB"]) & ((ev["rec_td_savg"] + ev["rush_td_savg"] > 0) | (ev["rec_td"] + ev["rush_td"] > 0))]
        hit = ((td["rec_td"] + td["rush_td"]) > 0).astype(float)
        brier = lambda lam: round(float(np.mean((1 - np.exp(-lam.fillna(0).clip(lower=0)) - hit) ** 2)), 4)
        recent_td = (td["rec_td_sum"] + td["rush_td_sum"]) / td["one_sum"].replace(0, np.nan)
        rows.append({"stat": "Anytime TD (Brier score)", "n": int(len(td)), "model": brier(td["rec_td_proj"] + td["rush_td_proj"]),
                     "season_avg": brier(td["rec_td_savg"] + td["rush_td_savg"]), "recent": brier(recent_td), "bias": None})
        starters = ev[(ev["group"] == "QB") & (ev["att_share"] > 0.5)].drop_duplicates(["game_id", "team"])
        rows_meta = {"starter_right": round(float((starters["started"] == 1).mean()), 4) if len(starters) else None}
        res[name] = {"rows": rows, **rows_meta}
    return res


# --------------------------------------------------------------------------
# Board output
# --------------------------------------------------------------------------

def board(proj: pd.DataFrame, table: pd.DataFrame, info: pd.DataFrame, game_keys: dict, abbr: dict,
          des: pd.DataFrame | None = None) -> dict:
    """Projections for the board's upcoming games (game_keys: game_id -> board id), with the inputs
    behind them, and for each team the players listed Out or Doubtful for the game, with their
    share of the team's usage when they have one. des: the injury designations
    (designations.listed). Each one covers the team's next game (every game if out for the
    season): there, Questionable and Probable become the player's status, and a player listed
    Out or Doubtful never gets a projection."""
    p = proj[proj["game_id"].isin(game_keys) & (proj["active"] == 1)].copy()
    if p.empty:
        return {"players": [], "teams": []}
    p["name"] = p["player_id"].map(info["name"]).fillna(p["player_id"].astype(str))
    des = des if des is not None and len(des) else None
    des_out = DES.outs(des, abbr) if des is not None else None
    r1 = lambda x: None if pd.isna(x) else round(float(x), 1)
    r2 = lambda x: None if pd.isna(x) else round(float(x), 2)
    r3 = lambda x: None if pd.isna(x) else round(float(x), 3)
    hist = table[table["played"].astype(bool) & (table["appeared"] == 1)]
    up = table[~table["played"].astype(bool) & table["game_id"].isin(game_keys)][["game_id", "team", "t"]].drop_duplicates()
    cov = DES.applies(des_out, up) if des_out is not None and len(des_out) else None
    # the designation each player carries in each board game: {(game_id, team, name key): code}
    tag = {}
    if des is not None:
        for x in DES.applies(des, up).itertuples():
            tag[(x.game_id, x.team_name, x.key)] = x.d
    teams = []
    for (gid, team), t in p.groupby(["game_id", "team"]):
        first = t.iloc[0]
        season = int(first["season"])
        mine = hist[(hist["team"] == team) & (hist["season"] == season) & (hist["t"] < first["t"])]
        out = []
        if cov is not None:
            listed = cov[(cov["game_id"] == gid) & (cov["team_name"] == team)]
            if len(listed):
                g = mine.groupby("player_id").agg(car=("carries", "sum"), rec=("receptions", "sum"), tc=("team_car", "sum"),
                                                  tr=("team_comp", "sum"), pa=("pass_att", "sum"), ta=("team_att", "sum"))
                g["key"] = g.index.map(lambda i: DES.norm(info["name"].get(i)))
                g = g.reset_index().drop_duplicates("key").set_index("key")
                for x in listed.itertuples():
                    sh = g.loc[x.key] if x.key in g.index else None
                    share = (lambda a, b: r3(sh[a] / sh[b]) if sh is not None and sh[b] > 0 else None)
                    out.append({"name": x.name, "pos": str(x.pos or ""), "status": DES.OUT_CODES[x.d] + (f", {x.injury.lower()}" if isinstance(x.injury, str) and x.injury and x.injury.lower() != "undisclosed" else ""),
                                "rec_share": share("rec", "tr"), "car_share": share("car", "tc"), "att_share": share("pa", "ta")})
                # biggest roles first, then linemen and defense
                out.sort(key=lambda o: -max(o["rec_share"] or 0, o["car_share"] or 0, o["att_share"] or 0))
        teams.append({"game": game_keys[gid], "team": abbr.get(team, team), "opp": abbr.get(first["opp"], first["opp"]),
                      "pass_att": r1(first["proj_team_att"]), "carries": r1(first["proj_team_car"]),
                      "imp_pts": r1(first["imp_pts"]), "margin": r1(first["margin"]), "wind": r1(first["wind"]),
                      "def_ypa": r3(first["def_ypa"]), "def_ypc": r3(first["def_ypc"]), "out": out})
    a = p[(p["receptions_proj"] >= 0.8) | (p["carries_proj"] >= 2) | ((p["group"] == "QB") & (p["att_share"] >= 0.5))]
    # The two inputs to the last calibration step, per stat: the volume x share x efficiency
    # chain and the recent per-game average (per start, for a starter's passing). The card shows
    # both, so a projection far from the chain is explained.
    SHOWN = ("pass_yds", "carries", "rush_yds", "receptions", "rec_yds")
    def _cal_in(s):
        try:
            return calib_X(a, s)[:, :2]
        except KeyError:                   # a frame without the chain columns: nothing to show
            return np.full((len(a), 2), np.nan)
    cal_in = {s: _cal_in(s) for s in SHOWN}
    players = []
    for i, r in enumerate(a.itertuples()):
        d = tag.get((r.game_id, r.team, DES.norm(r.name)))
        if d in DES.OUT_CODES:
            continue                       # listed out for this game: no projection (player_rows already drops him)
        td = (r.rec_td_proj or 0) + (r.rush_td_proj or 0)
        players.append({
            "id": str(int(r.player_id)), "name": r.name, "pos": r.position if isinstance(r.position, str) else r.group,
            "team": abbr.get(r.team, r.team), "opp": abbr.get(r.opp, r.opp), "game": game_keys[r.game_id], "week": int(r.week),
            "status": DES.WORDS.get(d),
            "proj": {k: r1(getattr(r, k + "_proj")) for k in STATS},
            "td_any": r3(1 - np.exp(-max(td, 0))),
            "why": {"rec_share": r3(r.rec_share), "car_share": r3(r.car_share), "att_share": r3(r.att_share),
                    "rec_share_hist": r3(r.rec_share_raw), "car_share_hist": r3(r.car_share_raw),
                    "ypr": r2(r.ypr), "ypc": r2(r.ypc), "ypa": r2(r.ypa), "comp": r3(r.comp),
                    "def_ypc": r3(r.def_ypc), "def_ypa": r3(r.def_ypa), "games": r1(r.one_sum),
                    "chain": {k: r1(cal_in[k][i, 0]) for k in SHOWN},
                    "recent": {k: r1(cal_in[k][i, 1]) for k in SHOWN}},
        })
    players.sort(key=lambda x: -(x["proj"]["rec_yds"] or 0) - (x["proj"]["rush_yds"] or 0) - (x["proj"]["pass_yds"] or 0) / 2.5)
    return {"players": players, "teams": teams}
