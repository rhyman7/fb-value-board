"""Every-game floors (Nov 5, 2026): the most yards a board player has reached in every game he
played this season, shown next to his projections on both boards. Display only: nothing here
feeds the model, the bets or the parlay tools.

The floor is the lowest qualifying game. Ryan's choices (PROPS.md "Every-game floors"):
  * which stats: QB passing and rushing; RB rushing, receiving and rush + receiving; WR and TE
    receiving. At least MIN_GAMES qualifying games.
  * the adjusted floor (shown): games for his current team only; QBs only in games they started;
    early exits dropped (NFL: offensive snap share under half his season median, from nflverse
    snap counts; college, with no public snap counts: touches under 40% of his median). In the
    games left, every stat of his position counts, so a game he played with no catch is 0
    receiving yards.
  * the raw floor (kept for the test only): every game with a chance at the stat (a pass
    attempt; a carry; a target or catch), any team.
  * flags: this week's setup tougher than any game behind the floor (or, ▲, a teammate's usage
    freed up). They never move the number.

League-neutral: each board's pipeline builds a season log and a team-game context and calls
`for_board`; experiments/floor_tag_test.py calls `player_floor` and `flags` week by week.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_GAMES = 3
SPLIT = {"QB": ("pass", "rush"), "RB": ("rush", "rec", "rr"), "WR": ("rec",), "TE": ("rec",)}
YDS = {"pass": "pass_yds", "rush": "rush_yds", "rec": "rec_yds"}
EXIT_CUT = {"nfl": 0.5, "cfb": 0.4}       # share of his median below which a game is an early exit
LOG_COLS = ["player_id", "game_id", "team", "opp", "t", "week", "pass_att", "carries", "targets", "receptions",
            "pass_yds", "rush_yds", "rec_yds", "share", "qb_start"]

FLAG_TEXT = {
    "tough_d": "Toughest defense he has faced at this stat",
    "dog": "Bigger underdog than in any of these games",
    "blowout": "Big favorite: starters may sit late",
    "new_qb": "Different starting QB from these games",
    "wind": "Windier than any of these games",
    "mate_out": "Top teammate at this stat is out",
}
UP_FLAGS = {"mate_out"}
# flags the 2025 test kept (PROPS.md "Every-game floors", experiments/out/floor_tag_test.json):
# blowout risk cleared more often with the flag than without (pooled), so it stays off
SHOW_FLAGS = set(FLAG_TEXT) - {"blowout"}


def group(pos) -> str | None:
    p = str(pos or "").upper()
    return {"FB": "RB", "HB": "RB"}.get(p, p) if p in ("QB", "RB", "FB", "HB", "WR", "TE") else None


def _value(rows: pd.DataFrame, stat: str) -> pd.Series:
    if stat == "rr":
        return rows["rush_yds"] + rows["rec_yds"]
    return rows[YDS[stat]]


def _opportunity(rows: pd.DataFrame, stat: str) -> pd.Series:
    tgt = (rows["targets"] > 0) | (rows["receptions"] > 0)
    return {"pass": rows["pass_att"] > 0, "rush": rows["carries"] > 0, "rec": tgt,
            "rr": tgt | (rows["carries"] > 0)}[stat]


def kept_games(rows: pd.DataFrame, grp: str, team, league: str) -> pd.DataFrame:
    """The games behind the adjusted floor: current team, QB starts, no early exits."""
    g = rows[rows["team"] == team]
    if grp == "QB":
        g = g[g["qb_start"].astype(bool)]
    sh = g["share"]
    med = float(sh.median()) if sh.notna().any() else float("nan")
    if med > 0:
        g = g[~(sh < EXIT_CUT[league] * med)]
    return g


def player_floor(rows: pd.DataFrame, grp: str, team, league: str, adjusted: bool = True) -> dict:
    """rows: one player's games this season before the game in question. Returns
    {stat: {"floor", "g", "rows"}} for his position's stats with MIN_GAMES+ games."""
    out = {}
    base = kept_games(rows, grp, team, league) if adjusted else rows
    for st in SPLIT.get(grp, ()):
        g = base if adjusted else base[_opportunity(base, st)]
        if len(g) < MIN_GAMES:
            continue
        v = _value(g, st)
        out[st] = {"floor": float(v.min()), "g": int(len(g)), "rows": g.assign(v=v.to_numpy())}
    return out


def allowed(log: pd.DataFrame, groups: dict) -> dict:
    """(defense, position group, stat) -> yards per game allowed to that position, from the log
    given (pass only what was known at the time)."""
    x = log.assign(grp=log["player_id"].map(groups))
    x = x[x["grp"].isin(["QB", "RB", "WR", "TE"])]
    x = x.assign(rr=x["rush_yds"] + x["rec_yds"])
    games = x.groupby("opp")["game_id"].nunique()
    s = x.groupby(["opp", "grp"])[["pass_yds", "rush_yds", "rec_yds", "rr"]].sum()
    out = {}
    for (d, gp), r in s.iterrows():
        n = games.get(d, 0)
        if n:
            for st, c in (("pass", "pass_yds"), ("rush", "rush_yds"), ("rec", "rec_yds"), ("rr", "rr")):
                out[(d, gp, st)] = r[c] / n
    return out


def top_mate(team_log: pd.DataFrame, game_ids, player_id, stat: str, groups: dict):
    """The teammate with the most targets (receiving) or carries (rushing, rush + receiving;
    quarterbacks left out) over the given games."""
    t = team_log[team_log["game_id"].isin(game_ids) & (team_log["player_id"] != player_id)]
    if stat in ("rush", "rr"):
        t = t[t["player_id"].map(groups) != "QB"]
        s = t.groupby("player_id")["carries"].sum()
    else:
        s = t.groupby("player_id")["targets"].sum()
    s = s[s > 0]
    return s.idxmax() if len(s) else None


def flags(stat: str, grp: str, used: pd.DataFrame, ctx: dict, now: dict, allow: dict, mate, mates_out: set) -> list[str]:
    """used: the floor games (game_id, opp). ctx: (game_id, team) -> {"spread" (his team's,
    positive = favoured), "wind" (NaN indoors), "qb"} for those games. now: the same for this
    game plus "opp" and "team"."""
    f = []
    past = [ctx.get((g, now["team"]), {}) for g in used["game_id"]]
    a_now = allow.get((now["opp"], grp, stat))
    a_past = [allow.get((o, grp, stat)) for o in used["opp"]]
    if stat != "rr" and a_now is not None and a_past and all(a is not None for a in a_past) and a_now < min(a_past):
        f.append("tough_d")
    sp = [p.get("spread") for p in past if p.get("spread") is not None and not _nan(p.get("spread"))]
    s_now = now.get("spread")
    if s_now is not None and not _nan(s_now) and sp:
        if stat in ("rush", "rr") and s_now < 0 and s_now <= min(sp) - 3:
            f.append("dog")
        if s_now >= 14 and s_now >= max(sp) + 3:
            f.append("blowout")
    if stat in ("rec", "rr") and grp != "QB":
        q_now, q_past = now.get("qb"), {p.get("qb") for p in past}
        if q_now is not None and not _nan(q_now) and None not in q_past and str(q_now) not in {str(q) for q in q_past}:
            f.append("new_qb")
    if stat in ("pass", "rec"):
        w_now = now.get("wind")
        if w_now is not None and not _nan(w_now) and w_now >= 15:
            w_past = [0.0 if _nan(p.get("wind")) else float(p.get("wind")) for p in past]
            if w_past and w_now >= max(w_past) + 5:
                f.append("wind")
    if stat in ("rec", "rush", "rr") and mate is not None and mate in mates_out:
        f.append("mate_out")
    return f


def _nan(x) -> bool:
    try:
        return x is None or math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def for_board(players: list[dict], log: pd.DataFrame, groups: dict, ctx: dict, now: dict, outs: dict, league: str,
              opp_label=lambda o: o) -> int:
    """Add "floor" to each board player (in place). players: the board's player dicts; log: this
    season's played games (LOG_COLS); groups: player_id -> position group; ctx: (game_id, team)
    -> context for played games; now: board player index -> this game's context (team, opp,
    spread, wind, qb, game_id); outs: (game_id, team) -> set of teammate ids out. Returns players given a floor."""
    if log is None or log.empty:
        return 0
    log = log.sort_values("t")
    allow = allowed(log, groups)
    by_player = {pid: g for pid, g in log.groupby("player_id")}
    by_team = {t: g for t, g in log.groupby("team")}
    n = 0
    for i, p in enumerate(players):
        c = now.get(i)
        grp = group(p.get("pos"))
        rows = by_player.get(str(p.get("id"))) if c else None
        if c is None or grp is None or rows is None:
            continue
        fl = player_floor(rows, grp, c["team"], league)
        if not fl:
            continue
        res = {}
        for st, d in fl.items():
            g = d["rows"]
            mate = top_mate(by_team.get(c["team"], log.iloc[:0]), set(g["game_id"]), str(p.get("id")), st, groups)
            fg = [x for x in flags(st, grp, g, ctx, c, allow, mate, outs.get((c["game_id"], c["team"]), set())) if x in SHOW_FLAGS]
            vals = g["v"].to_numpy()
            lo = int(np.argmin(vals))
            srt = sorted(vals)
            res[st] = {"floor": int(round(vals[lo])), "g": d["g"],
                       "low": f"Wk {int(g['week'].iloc[lo])} vs {opp_label(g['opp'].iloc[lo])}",
                       "next": int(round(srt[1])) if len(srt) > 1 else None,
                       "avg": round(float(vals.mean()), 1),
                       "log": [[int(w), str(opp_label(o)), int(round(v))] for w, o, v in zip(g["week"], g["opp"], vals)],
                       "flags": [{"k": k, "up": k in UP_FLAGS, "text": FLAG_TEXT[k]} for k in fg]}
        p["floor"] = res
        n += 1
    return n


# ------------------------------------------------------------------------------------------
# Season logs and team-game context, per league
# ------------------------------------------------------------------------------------------
def nfl_log(season: int, pg: pd.DataFrame, games: pd.DataFrame, snaps: pd.DataFrame | None,
            depth: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """NFL: the board's play-by-play box scores (nflmodel/players_data.py, regular season) plus
    games a player took offensive snaps in without a stat (0 yards). share = offensive snap
    share. games: the board's schedule frame (home/away team, home_qb_id/away_qb_id, spread_line
    from the home side, wind, roof)."""
    g = games[games["season"] == season].copy()
    p = pg[(pg["season"] == season) & (pg["season_type"] == "REG")].copy()
    p["player_id"] = p["player_id"].astype(str)
    p = p.rename(columns={"posteam": "team", "defteam": "opp"})
    if snaps is not None and len(snaps):
        s = snaps[snaps["game_id"].isin(set(g["game_id"]))].copy()
        s["player_id"] = s["player_id"].astype(str)
        p = p.merge(s[["game_id", "player_id", "offense_pct"]], on=["game_id", "player_id"], how="outer")
        p = p[~(p["team"].isna() & (p["offense_pct"].fillna(0) <= 0))]       # special teams only: not a game played
        miss = p["team"].isna()
        if miss.any():
            gi = g.set_index("game_id")
            p.loc[miss, "week"] = p.loc[miss, "game_id"].map(gi["week"])
            team = {}
            if depth is not None and len(depth):
                d = depth[depth["season"] == season]
                team = {(str(a), int(w)): t for a, w, t in zip(d["player_id"], d["week"], d["team"])}
            known = p[~miss].sort_values("week").groupby("player_id")["team"].last().to_dict()
            for i in p.index[miss]:
                pid, gid, wk = p.at[i, "player_id"], p.at[i, "game_id"], int(p.at[i, "week"])
                t = team.get((pid, wk)) or known.get(pid)
                h, a = gi.at[gid, "home_team"], gi.at[gid, "away_team"]
                if t in (h, a):
                    p.at[i, "team"], p.at[i, "opp"] = t, a if t == h else h
        p = p[p["team"].notna()]
        p["share"] = p["offense_pct"]
    else:
        p["share"] = np.nan
    for c in ("pass_att", "carries", "targets", "receptions", "pass_yds", "rush_yds", "rec_yds"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    p["week"] = p["week"].astype(int)
    p["t"] = p["week"]
    qb = {}
    ctx = {}
    for r in g.itertuples():
        indoor = str(getattr(r, "roof", "") or "").lower() in ("dome", "closed") or bool(getattr(r, "indoor", 0) or 0)
        wind = np.nan if indoor else getattr(r, "wind", np.nan)
        for side, team, sign in (("home", r.home_team, 1), ("away", r.away_team, -1)):
            q = getattr(r, f"{side}_qb_id", None)
            qb[(r.game_id, team)] = None if _nan(q) else str(q)
            sp = getattr(r, "spread_line", np.nan)
            ctx[(r.game_id, team)] = {"spread": None if _nan(sp) else sign * float(sp), "wind": wind, "qb": qb[(r.game_id, team)]}
    p["qb_start"] = [qb.get((gid, t)) == pid for gid, t, pid in zip(p["game_id"], p["team"], p["player_id"])]
    return p[LOG_COLS].reset_index(drop=True), ctx


def cfb_log(season: int, pg: pd.DataFrame, games: pd.DataFrame, groups: dict) -> tuple[pd.DataFrame, dict]:
    """College: official box scores (cfbmodel/players_data.py; sacks count as QB runs, as props
    settle) with targets from play-by-play. share = touches (QB: pass attempts + carries; others:
    carries + targets). A QB start is the team's pass-attempt leader that game. Teams are ESPN
    team ids. games: the board's college schedule frame (home_id/away_id, spread_line from the
    home side, wind, dome, week)."""
    allg = games[games["season"] == season].copy()
    allg["t"] = pd.to_datetime(allg["kickoff"], utc=True).rank(method="dense").astype(int)
    g = allg[allg["played"].astype(bool)]
    gi = g.set_index("game_id")
    p = pg[(pg["season"] == season) & pg["game_id"].isin(set(g["game_id"]))].copy()
    p = p.rename(columns={"team_id": "team"})
    p["player_id"] = p["player_id"].astype(str)
    home = p["game_id"].map(gi["home_id"])
    p["opp"] = np.where(p["team"] == home, p["game_id"].map(gi["away_id"]), home)
    p["week"] = p["game_id"].map(gi["week"]).astype(int)
    p["t"] = p["game_id"].map(gi["t"]).astype(int)
    for c in ("pass_att", "carries", "targets", "receptions", "pass_yds", "rush_yds", "rec_yds"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    isqb = p["player_id"].map(groups) == "QB"
    p["share"] = np.where(isqb, p["pass_att"] + p["carries"], p["carries"] + p["targets"])
    lead = p.sort_values("pass_att").groupby(["game_id", "team"]).tail(1)
    lead = lead[lead["pass_att"] > 0]
    qb = {(a, b): c for a, b, c in zip(lead["game_id"], lead["team"], lead["player_id"])}
    p["qb_start"] = [qb.get((a, b)) == c for a, b, c in zip(p["game_id"], p["team"], p["player_id"])]
    ctx = {}
    for r in allg.itertuples():
        wind = np.nan if int(getattr(r, "dome", 0) or 0) else getattr(r, "wind", np.nan)
        for team, sign, side in ((r.home_id, 1, "home"), (r.away_id, -1, "away")):
            sp = getattr(r, "spread_line", np.nan)
            q = qb.get((r.game_id, team))
            if q is None and not bool(r.played):            # upcoming: the board's starting QB
                q0 = getattr(r, f"{side}_qb_id", None)
                q = None if _nan(q0) else str(int(float(q0)))
            ctx[(r.game_id, team)] = {"spread": None if _nan(sp) else sign * float(sp), "wind": wind, "qb": q}
    return p[LOG_COLS].reset_index(drop=True), ctx


# ------------------------------------------------------------------------------------------
# Board hooks (called from each pipeline; a failure there never stops the board build)
# ------------------------------------------------------------------------------------------
def _norm(n) -> str:
    return "".join(ch for ch in str(n or "").lower() if ch.isalnum())


def attach_nfl(board: dict, season: int, pg: pd.DataFrame, pinfo: pd.DataFrame, games: pd.DataFrame,
               pproj: pd.DataFrame | None, cache) -> int:
    snaps = cache / f"snaps_{season}.csv"
    depth = cache / f"depth_{season}.csv"
    log, ctx = nfl_log(season, pg, games,
                       pd.read_csv(snaps, dtype={"player_id": str}) if snaps.exists() else None,
                       pd.read_csv(depth, dtype={"player_id": str}) if depth.exists() else None)
    groups = {str(a): group(p) for a, p in pinfo["position"].items()}
    gi = games.set_index("game_id")
    now = {}
    for i, p in enumerate(board.get("players") or []):
        gid, team = p.get("game"), p.get("team")
        if gid in gi.index and (gid, team) in ctx:
            now[i] = {**ctx[(gid, team)], "team": team, "opp": p.get("opp"), "game_id": gid}
    outs = {}
    if pproj is not None and "active" in pproj:
        o = pproj[pproj["game_id"].isin({p.get("game") for p in board.get("players") or []}) & (pproj["active"] == 0)]
        for gid, team, pid in zip(o["game_id"], o["team"], o["player_id"]):
            outs.setdefault((gid, team), set()).add(str(pid))
    return for_board(board.get("players") or [], log, groups, ctx, now, outs, "nfl")


def attach_cfb(board: dict, season: int, pg: pd.DataFrame, pinfo: pd.DataFrame, games: pd.DataFrame,
               game_keys: dict, abbr: dict) -> int:
    """game_keys: game_id -> board game id; abbr: team name -> abbreviation (the board's labels)."""
    groups = {str(a): group(p) for a, p in pinfo["position"].items()}
    log, ctx = cfb_log(season, pg, games, groups)
    sg = games[games["season"] == season]
    id_abbr = {}
    for h, hn, a, an in zip(sg["home_id"], sg["home_team"], sg["away_id"], sg["away_team"]):
        id_abbr[h], id_abbr[a] = abbr.get(hn, hn), abbr.get(an, an)
    back = {v: k for k, v in game_keys.items()}
    gi = sg.set_index("game_id")
    names = pinfo["name"].to_dict()
    out_names = {}
    for t in board.get("teams") or []:
        out_names[(back.get(t.get("game")), t.get("team"))] = {_norm(o.get("name")) for o in t.get("out") or []}
    outs = {}
    now = {}
    for i, p in enumerate(board.get("players") or []):
        gid = back.get(p.get("game"))
        if gid not in gi.index:
            continue
        r = gi.loc[gid]
        team = r["home_id"] if id_abbr.get(r["home_id"]) == p.get("team") else r["away_id"] if id_abbr.get(r["away_id"]) == p.get("team") else None
        if team is None or (gid, team) not in ctx:
            continue
        opp = r["away_id"] if team == r["home_id"] else r["home_id"]
        now[i] = {**ctx[(gid, team)], "team": team, "opp": opp, "game_id": gid}
        on = out_names.get((gid, p.get("team")), set())
        if on and (gid, team) not in outs:
            mates = log.loc[log["team"] == team, "player_id"].unique()
            outs[(gid, team)] = {m for m in mates if _norm(names.get(int(m)) if str(m).lstrip("-").isdigit() else None) in on}
    return for_board(board.get("players") or [], log, groups, ctx, now, outs, "cfb", opp_label=lambda o: id_abbr.get(o, o))
