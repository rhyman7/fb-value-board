"""Extra per-game inputs for the college game model (Nov 2026, FEATURES.md "Re-judging past tests
under the keep rule"), each reduced from cfbfastR data and cached per season in data/cfb/cache/:

  play_extra_<s>.csv   one row per offense per game over the board's non-garbage scrimmage plays:
                       line yards per rush (lyd_sum / n_rush, each run clipped to -6..7), down-split
                       success, havoc and turnover-free EPA (experiments/cfb_downs_havoc_test.py)
  drive_raw_<s>.csv    one row per drive: start spot (yards to the end zone), points, whether it
                       reached the opponent's 40 (experiments/cfb_drive_test.py). per_game() turns
                       them into drive value over expected (dve) and finishing (fin)
  def_missing_<s>.csv  one row per team per game: the share of the team's core defenders (by
                       box-score production over its earlier games this season) who recorded no
                       stat in the previous game, for pass rush, coverage and tackling, and the
                       change from the three games before (experiments/cfb_def_test.py)
  data/cfb/coach_changes.csv   every FBS head-coaching change with its date (hand-kept; from
                       Wikipedia's coaching-change tables). coach_flags() marks a mid-season change.

The play and drive tables are written by data.load_season from the same play-by-play download as
team_games; the box-score table has its own download (player_box). Ratings for lyd, dve and fin
are solved by the rating engine like the base metrics (ratings.CFBRatingEngine, extra=...).
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

# ---- plays --------------------------------------------------------------------------------------
PLAY_COLS = ["standard_down", "passing_down", "line_yards", "havoc", "turnover_vec"]
PLAY_METRICS = {"sdsr": ("sd_succ", "sd_n"), "pdsr": ("pd_succ", "pd_n"), "lyd": ("lyd_sum", "n_rush"),
                "hav": ("hav_sum", "plays"), "pnt": ("pnt_epa", "pnt_n"), "rnt": ("rnt_epa", "rnt_n")}
# ---- drives -------------------------------------------------------------------------------------
DRIVE_COLS = ["drive.id", "drive.result", "start.yardsToEndzone", "start.TimeSecsRem", "drive_play_index"]
TD = {"TD", "END OF HALF TD", "END OF GAME TD"}
DEF_TD = {"INT TD", "FUMBLE RETURN TD", "FUMBLE TD"}
DROP = {"END OF HALF", "END OF GAME", "END OF QUARTER", "END OF 4TH QUARTER", "Not provided", "UNKNOWN", "KICKOFF", "2PT"}
BASE_SEASONS = (2014, 2015, 2016)      # the start-spot baseline for dve: before any scored season
DRIVE_METRICS = {"dve": ("dve", "drives"), "fin": ("fin", "trips")}
# ---- ridge penalties, chosen on the tuning seasons by the tests (experiments/out/rejudge_2026-11-04/)
EXTRA_RIDGE = {"lyd": 6.0, "dve": 4.0, "fin": 18.0, "sdsr": 18.0, "pdsr": 18.0, "hav": 54.0, "pnt": C.RIDGE["pass"], "rnt": C.RIDGE["rush"]}
# ---- defenders ----------------------------------------------------------------------------------
CORE = {"pr": 0.10, "cov": 0.10, "tk": 0.08}      # a core defender's share threshold (two appearances)
COACH_FILE = C.DATA / "coach_changes.csv"
COACH_ALIAS = {"FIU": "Florida International", "Hawaii": "Hawai'i", "Louisiana Monroe": "UL Monroe",
               "San Jose State": "San José State", "Appalachian State": "App State"}


def _flag(p: pd.DataFrame, c: str) -> pd.Series:
    return p[c].fillna(False).astype(bool) if c in p else pd.Series(False, index=p.index)


def reduce_plays(p: pd.DataFrame, season: int, games: pd.DataFrame) -> pd.DataFrame:
    """play_extra rows from a season's play-by-play frame (data.PBP_COLS + PLAY_COLS)."""
    p = p[p["game_id"].astype(int).isin(set(games.game_id))].copy()
    p["game_id"] = p["game_id"].astype(int)
    scrim = (_flag(p, "rush") | _flag(p, "pass")) & ~_flag(p, "kneel_down") & ~_flag(p, "penalty_no_play") & p["EPA"].notna()
    p = p[scrim & p["pos_team_id"].notna()].copy()
    period = pd.to_numeric(p["period"], errors="coerce").fillna(4).astype(int)
    lead = pd.to_numeric(p["pos_score_diff_start"], errors="coerce").abs().fillna(0)
    p = p[~(lead > period.map(C.GARBAGE_LEAD).fillna(999))]
    home = games.set_index("game_id")["home_id"]
    p["side"] = np.where(p["pos_team_id"].astype(int) == p["game_id"].map(home), "home", "away")
    p["epa"] = p["EPA"].clip(-C.EPA_CLIP, C.EPA_CLIP)
    p["succ"] = p["EPA_success"].fillna(False).astype(bool).astype(int)
    p["is_pass"] = _flag(p, "pass").loc[p.index].astype(int)
    p["is_rush"] = 1 - p["is_pass"]
    sd, pd_ = _flag(p, "standard_down").loc[p.index].astype(int), _flag(p, "passing_down").loc[p.index].astype(int)
    to = _flag(p, "turnover_vec").loc[p.index].astype(int)
    ly = pd.to_numeric(p["line_yards"], errors="coerce").fillna(0).clip(-6, 7) if "line_yards" in p else 0.0
    p = p.assign(sd_n=sd, sd_succ=sd * p.succ, pd_n=pd_, pd_succ=pd_ * p.succ, n_rush=p.is_rush,
                 lyd_sum=p.is_rush * ly, hav_sum=_flag(p, "havoc").loc[p.index].astype(int), plays=1,
                 pnt_n=p.is_pass * (1 - to), pnt_epa=p.epa * p.is_pass * (1 - to),
                 rnt_n=p.is_rush * (1 - to), rnt_epa=p.epa * p.is_rush * (1 - to))
    cols = sorted({c for v in PLAY_METRICS.values() for c in v})
    g = p.groupby(["game_id", "side"])[cols].sum().reset_index()
    g.insert(0, "season", season)
    return g


def reduce_drives(p: pd.DataFrame, season: int, games: pd.DataFrame) -> pd.DataFrame:
    """drive_raw rows from a season's play-by-play frame (data.PBP_COLS + DRIVE_COLS)."""
    p = p[p["game_id"].astype(int).isin(set(games.game_id))].copy()
    p["game_id"] = p["game_id"].astype(int)
    scrim = _flag(p, "rush") | _flag(p, "pass")
    p = p[scrim & p["pos_team_id"].notna() & p["drive.id"].notna()]
    p = p.sort_values(["game_id", "drive.id", "drive_play_index"])
    grp = p.groupby(["game_id", "drive.id"], sort=False)
    d = grp.agg(team_id=("pos_team_id", "first"), result=("drive.result", "first"),
                ytg=("start.yardsToEndzone", "first"), secs=("start.TimeSecsRem", "first"),
                period=("period", "first"), lead=("pos_score_diff_start", "first"),
                min_ytg=("start.yardsToEndzone", "min")).reset_index()
    for c in ("ytg", "secs", "period", "lead", "min_ytg"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    home = games.set_index("game_id")["home_id"]
    d["side"] = np.where(d.team_id.astype(int) == d.game_id.map(home), "home", "away")
    d["pts"] = np.select([d.result.isin(TD), d.result == "FG", d.result == "SF", d.result.isin(DEF_TD)], [7.0, 3.0, -2.0, -7.0], 0.0)
    garbage = d.lead.abs() > d.period.fillna(4).astype(int).map(C.GARBAGE_LEAD).fillna(999)
    d = d[~garbage & (d.secs >= 60) & ~d.result.isin(DROP) & d.ytg.between(1, 99)]
    d["trip"] = (d.min_ytg <= 40).astype(int)
    d.insert(0, "season", season)
    return d[["season", "game_id", "side", "ytg", "pts", "trip"]]


def per_game(d: pd.DataFrame) -> pd.DataFrame:
    """Drive value over expected and finishing per team-game. The expected points of a start spot
    come from BASE_SEASONS only, smoothed over 5 yards."""
    base = d[d.season.isin(BASE_SEASONS)].groupby("ytg")["pts"].mean()
    base = base.reindex(range(1, 100)).interpolate().bfill().ffill()
    base = base.rolling(5, center=True, min_periods=1).mean()
    d = d.assign(dve=d.pts - d.ytg.map(base))
    g = d.groupby(["game_id", "side"]).agg(drives=("dve", "size"), dve=("dve", "mean"))
    t = d[d.trip == 1].groupby(["game_id", "side"]).agg(trips=("pts", "size"), fin=("pts", "mean"))
    return g.join(t).fillna({"trips": 0}).reset_index()


def load_plays(seasons) -> pd.DataFrame:
    return pd.concat([pd.read_csv(C.CACHE / f"play_extra_{s}.csv") for s in seasons], ignore_index=True)


def load_drives(seasons) -> pd.DataFrame:
    return pd.concat([pd.read_csv(C.CACHE / f"drive_raw_{s}.csv") for s in seasons], ignore_index=True)


# ---- defenders missing, from the box scores ----------------------------------------------------

def _box(season: int) -> pd.DataFrame:
    url = f"{C.BASE}/{C.FILES['player_box'].format(s=season)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cfb-model"}), timeout=600) as r:
        b = pd.read_parquet(io.BytesIO(r.read()))
    b = b[b["athlete_id"].notna() & b["team_id"].notna()].copy()
    num = lambda c: pd.to_numeric(b[c], errors="coerce").fillna(0.0) if c in b else 0.0
    d = pd.DataFrame({"game_id": pd.to_numeric(b["game_id"], errors="coerce"), "team_id": pd.to_numeric(b["team_id"], errors="coerce"),
                      "player_id": pd.to_numeric(b["athlete_id"], errors="coerce"), "category": b["category"],
                      "pr": np.where(b["category"] == "defensive", num("sacks") + num("hurries"), 0.0),
                      "cov": np.where(b["category"] == "defensive", num("passesDefended"), 0.0)
                      + np.where(b["category"] == "interceptions", num("interceptions"), 0.0),
                      "tk": np.where(b["category"] == "defensive", num("totalTackles"), 0.0)})
    d = d[d["category"].isin(["defensive", "interceptions"])].drop(columns="category")
    return d.groupby(["game_id", "team_id", "player_id"], as_index=False).sum()


def defenders_missing(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Per team and game (one season): miss_def<k>, the share of core defenders (production share
    over the team's earlier games this season, games they appeared in) who recorded no stat in the
    previous game; chg_def<k>, that less its average over the three games before. k in pr, cov, tk.
    An upcoming game reads the latest played game."""
    g = games[["game_id", "season", "order", "home_id", "away_id", "home_team", "away_team", "played"]]
    keep = ["game_id", "season", "order", "team_id", "team", "played"]
    tm = pd.concat([g.rename(columns={"home_id": "team_id", "home_team": "team"})[keep],
                    g.rename(columns={"away_id": "team_id", "away_team": "team"})[keep]])
    tm = tm[tm.team != C.POOL].sort_values(["team", "order"]).reset_index(drop=True)
    p = box.merge(tm[["game_id", "team_id", "team"]], on=["game_id", "team_id"], how="inner")
    tot = p.groupby(["game_id", "team_id"])[list(CORE)].transform("sum")
    for k in CORE:
        p[k] = p[k] / tot[k].replace(0, np.nan)
    out = []
    for (team, season), t in tm.groupby(["team", "season"], sort=False):
        t = t.sort_values("order")
        gids = list(t["game_id"])
        done = dict(zip(t["game_id"], t["played"].astype(bool)))
        pt = p[(p["team"] == team) & p["game_id"].isin(gids)]
        appeared = pt.pivot_table(index="player_id", columns="game_id", values="tk", aggfunc="size").notna()
        appeared = appeared.reindex(columns=gids).fillna(False).astype(bool)
        share = {k: pt.pivot_table(index="player_id", columns="game_id", values=k, aggfunc="sum").reindex(index=appeared.index, columns=gids)
                 for k in CORE}
        hist = {k: [] for k in CORE}
        for i, gid in enumerate(gids):
            row = {"game_id": gid, "team": team}
            prev = [x for x in gids[:i] if done[x]]
            for k, thr in CORE.items():
                if not prev:
                    row[f"miss_def{k}"] = row[f"chg_def{k}"] = 0.0
                    hist[k].append(0.0)
                    continue
                n_app = appeared[prev].sum(axis=1)
                avg = share[k][prev].where(appeared[prev].to_numpy()).mean(axis=1).fillna(0.0)
                core = avg[(avg >= thr) & (n_app >= 2)]
                last = appeared[prev[-1]].reindex(core.index, fill_value=False).astype(bool).to_numpy()
                miss = float(core[~last].sum())
                base = float(np.mean(hist[k][-3:])) if hist[k] else 0.0
                row[f"miss_def{k}"], row[f"chg_def{k}"] = miss, miss - base
                hist[k].append(miss)
            out.append(row)
    return pd.DataFrame(out)


def load_defenders(games: pd.DataFrame, refresh_current: bool = False, verbose: bool = True) -> pd.DataFrame:
    """def_missing_<s>.csv for every season in `games`; the newest season re-reduced on refresh."""
    now = int(games.season.max())
    out = []
    for s in sorted(games.season.unique()):
        path = C.CACHE / f"def_missing_{s}.csv"
        if not path.exists() or (refresh_current and s == now):
            if verbose:
                print(f"  defenders missing {s} (box scores) ...", flush=True)
            defenders_missing(games[games.season == s], _box(int(s))).round(5).to_csv(path, index=False)
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True)


# ---- coaching changes ---------------------------------------------------------------------------

def coach_flags(games: pd.DataFrame, path=COACH_FILE) -> pd.DataFrame:
    """Per game: d_mid (home minus away), 1 for a team whose head coach changed during this season
    before this game (fired, interim or hired), from the dated change list."""
    ch = pd.read_csv(path, parse_dates=["date"])
    ch["team"] = ch["team"].replace(COACH_ALIAS)
    by = {t: d["date"].to_numpy() for t, d in ch.groupby("team")}
    date = pd.to_datetime(games["gameday"]).to_numpy()
    flags = {}
    for side in ("home_team", "away_team"):
        vals = []
        for t, s, dt in zip(games[side], games["season"], date):
            d = by.get(t)
            hi = np.datetime64(f"{int(s)}-08-31")
            vals.append(0.0 if d is None else float(((d > hi) & (d < dt)).any()))
        flags[side] = np.array(vals)
    return pd.DataFrame({"game_id": games["game_id"].to_numpy(), "d_mid": flags["home_team"] - flags["away_team"],
                         "s_mid": flags["home_team"] + flags["away_team"]})


def attach_game_features(frame: pd.DataFrame, games: pd.DataFrame, defenders: pd.DataFrame | None, coach: pd.DataFrame | None) -> pd.DataFrame:
    """The non-rated features: defenders missed (home minus away for the margin), coaching changes.
    0 where unknown."""
    f = frame.copy()
    if defenders is not None and len(defenders):
        cols = [c for c in defenders.columns if c.startswith(("miss_def", "chg_def"))]
        for side, team in (("h", "home_team"), ("a", "away_team")):
            f = f.merge(defenders[["game_id", "team"] + cols].rename(columns={"team": team, **{c: f"{side}_{c}" for c in cols}}),
                        on=["game_id", team], how="left")
        for c in cols:
            h, a = f.pop(f"h_{c}").fillna(0.0), f.pop(f"a_{c}").fillna(0.0)
            f["d_" + c], f["s_" + c] = h - a, h + a
    if coach is not None and len(coach):
        f = f.merge(coach, on="game_id", how="left")
        f["d_mid"], f["s_mid"] = f["d_mid"].fillna(0.0), f["s_mid"].fillna(0.0)
    return f


# ---- everything the board needs ------------------------------------------------------------------
RATED = ("lyd", "dve")      # the rated extras in the live model (Nov 4, 2026): line yards (total), drive value (margin)


def rated_inputs(plays: pd.DataFrame, drives: pd.DataFrame, which=RATED) -> dict:
    """{metric: (frame, rate column, weight column)} for ratings.CFBRatingEngine(extra=...)."""
    pl = plays.copy()
    for m, (num, den) in PLAY_METRICS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            pl[m] = np.where(pl[den] > 0, pl[num] / pl[den].replace(0, np.nan), np.nan)
    pg = per_game(drives)
    out = {}
    for m in which:
        if m in PLAY_METRICS:
            out[m] = (pl, m, PLAY_METRICS[m][1])
        elif m in DRIVE_METRICS:
            out[m] = (pg, m, DRIVE_METRICS[m][1])
        else:
            raise KeyError(m)
    return out


def load_inputs(games: pd.DataFrame, refresh_current: bool = False, verbose: bool = True, rated: bool = True) -> dict:
    """The extra game-model inputs for every season in `games`: 'rated' for the engine, 'defenders'
    and 'coach' for attach_game_features."""
    seasons = sorted(games.season.unique())
    out = {"rated": rated_inputs(load_plays(seasons), load_drives(seasons)) if rated else None}
    out["defenders"] = load_defenders(games, refresh_current=refresh_current, verbose=verbose)
    out["coach"] = coach_flags(games) if COACH_FILE.exists() else None
    return out
