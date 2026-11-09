"""Grade the conferences' availability reports against who actually played.

    python experiments/availability_grade.py [--season 2026] [--out data/cfb/availability/2026/grade.csv]

For every (team, player, game) in the vendor archive (data/cfb/availability/<season>/conf_archive.json:
Initial, Update 1, Update 2 and Game Day statuses for every power-conference conference game since
Week 0), says whether the player played, and tabulates played rates by report stage and status, per
conference. "Played" is, in order of preference:
  - PFF: offensive snaps > 0 in offense_summary for that week (private data, cfbmodel.pff_data; read
    only, nothing from it is written row by row), for players PFF has;
  - the board's box scores: a pass attempt, carry or reception in the game (players_data cache).
Players with neither source for the week (linemen and defenders without PFF, walk-ons) are "unknown"
and left out of the rates. A "Game Day: Out" player who played is the interesting kind of miss; a
"Probable" who sat is the other.

Prints the tables. With --out, also writes one row per archived designation with the outcome (never
commit that file when PFF was used: its played flags are PFF data, which stays private). Run with PFF
by the manual **Availability grade** Action (availability-grade.yml), which prints aggregates only.
Not part of any build.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import config as C, designations as DES, pff_data, players_data as PD  # noqa: E402

STAGES = ["Initial", "Update 1", "Update 2", "Game Day"]


def archive(season: int) -> pd.DataFrame:
    a = json.loads((C.DATA / "availability" / str(season) / "conf_archive.json").read_text())
    rows = []
    for org, data in a.items():
        for x in data:
            try:
                wk = dt.datetime.strptime(x.get("Week", ""), "%a, %d %b %Y %H:%M:%S %Z").date()
            except ValueError:
                wk = None
            team = x.get("TeamDisplay") or x.get("Team")
            team = {("MAC", "Miami"): "Miami (OH)"}.get((org, team), team)
            rows.append({"conf": org, "team": team, "player": x.get("Player"),
                         "number": x.get("Number"), "opponent": x.get("OpponentDisplay") or x.get("Opponent"),
                         "week_sat": wk, "week_num": x.get("WeekNum"),
                         **{s: x.get(s) or "" for s in STAGES}})
    d = pd.DataFrame(rows)
    d["key"] = d["player"].map(DES.norm)
    return d


def played_box(season: int, games: pd.DataFrame) -> pd.DataFrame:
    """(team, key, game_date) -> played (recorded a pass, carry or catch), from the box-score cache."""
    pg, info = PD.load_player_games(verbose=False, last_season=season)
    pg = pg[pg["season"] == season] if "season" in pg.columns else pg.merge(games[["game_id", "season"]], on="game_id")
    pg = pg[pg["season"] == season]
    team = pd.concat([games[["game_id", "home_id", "home_team", "gameday"]].set_axis(["game_id", "team_id", "team", "gameday"], axis=1),
                      games[["game_id", "away_id", "away_team", "gameday"]].set_axis(["game_id", "team_id", "team", "gameday"], axis=1)])
    o = pg.merge(team, on=["game_id", "team_id"])
    o["key"] = o["player_id"].map(info["name"]).map(DES.norm)
    o["pos"] = o["player_id"].map(info["position"]) if "position" in info.columns else ""
    o = o[o[["pass_att", "carries", "receptions"]].fillna(0).sum(axis=1) > 0]
    o["gameday"] = pd.to_datetime(o["gameday"]).dt.date
    stat = o.groupby(["team", "key", "gameday"]).size().reset_index()[["team", "key", "gameday"]].assign(box=1)
    # the universe: every (team, player) with a stat this season, for every game the team played
    # after that player's first stat (so a missing stat line means he did not play, not that he is
    # a lineman the box score never lists)
    first = o.groupby(["team", "key"]).agg(first=("gameday", "min"), pos=("pos", "last")).reset_index()
    tgames = team[["team", "gameday"]].drop_duplicates()
    tgames["gameday"] = pd.to_datetime(tgames["gameday"]).dt.date
    uni = first.merge(tgames, on="team")
    uni = uni[uni["gameday"] >= uni["first"]][["team", "key", "gameday", "pos"]]
    out = uni.merge(stat, on=["team", "key", "gameday"], how="left")
    out["box"] = out["box"].fillna(0).astype(int)
    return out


def played_pff(season: int, names: pd.DataFrame) -> pd.DataFrame | None:
    """(team, key, season week) -> snaps > 0 from PFF offense_summary, or None without the data.
    PFF franchises are matched to schools by shared player names (names: team, key from the box
    scores and the reports), as cfbmodel/pff_roles.py does, so a name shared across schools never
    crosses teams."""
    if not pff_data.available():
        return None
    off = pff_data.load("offense_summary", [season])
    if off.empty:
        return None
    off["key"] = off["player"].map(DES.norm)
    off["snaps"] = pd.to_numeric(off["snap_counts_total"], errors="coerce").fillna(0)
    nm = names[["team", "key"]].drop_duplicates()
    ov = off[["franchise_id", "key"]].drop_duplicates().merge(nm, on="key").groupby(["franchise_id", "team"]).size().rename("n").reset_index()
    best = ov.sort_values("n", ascending=False).drop_duplicates("franchise_id").drop_duplicates("team")
    best = best[best["n"] >= 5]
    off = off.drop(columns=[c for c in ("team",) if c in off.columns])     # PFF's own team label; ours is the school
    off = off.merge(best[["franchise_id", "team"]], on="franchise_id")
    print(f"  PFF: {best['team'].nunique()} franchises matched to schools")
    g = off.groupby(["team", "key", "week"]).agg(snaps=("snaps", "max"), ids=("player_id", "nunique")).reset_index()
    g = g[g["ids"] == 1]                         # two PFF players with one name on one team that week: no outcome
    g["pff"] = (g["snaps"] > 0).astype(int)
    # his usual snaps: the mean over his earlier weeks with snaps this season
    g = g.sort_values(["team", "key", "week"])
    played = g["snaps"].where(g["snaps"] > 0)
    g["usual"] = played.groupby([g["team"], g["key"]]).transform(lambda x: x.shift().expanding().mean())
    g["snap_ratio"] = (g["snaps"] / g["usual"]).where(g["usual"] > 0)
    return g[["team", "key", "week", "pff", "snaps", "usual", "snap_ratio"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    today = dt.date.today()
    season = a.season or (today.year if today.month >= 8 else today.year - 1)
    out = Path(a.out) if a.out else None
    d = outcomes(season)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(out, index=False)
        print(f"wrote {out}")
    report(d)


def outcomes(season: int) -> pd.DataFrame:
    """Every archived designation matched to its game (game_id, gameday, week), with played (PFF snaps
    where available, else box-score stats), source (pff | box | unknown), pos and snap_ratio."""
    today = dt.date.today()
    games = pd.read_csv(C.CACHE / f"games_{season}.csv")
    games["gameday"] = pd.to_datetime(games["kickoff"], utc=True).dt.tz_convert("America/New_York").dt.date
    arch = archive(season)
    # the archive's "Week" is the Saturday; the game may be Thursday-Saturday: match each team's game
    # in the window [Tuesday, Sunday] around it
    tg = pd.concat([games[["game_id", "home_team", "gameday", "week"]].set_axis(["game_id", "team", "gameday", "week"], axis=1),
                    games[["game_id", "away_team", "gameday", "week"]].set_axis(["game_id", "team", "gameday", "week"], axis=1)])
    tg = tg[tg["gameday"] <= today]
    rows = []
    for r in arch.to_dict("records"):
        if r["week_sat"] is None:
            continue
        mine = tg[(tg["team"] == r["team"]) & (tg["gameday"] >= r["week_sat"] - dt.timedelta(days=4)) & (tg["gameday"] <= r["week_sat"] + dt.timedelta(days=1))]
        if mine.empty:
            continue
        g = mine.iloc[0]
        rows.append({**r, "game_id": g["game_id"], "gameday": g["gameday"], "week": int(g["week"])})
    d = pd.DataFrame(rows)
    print(f"{len(arch)} archived designations, {len(d)} matched to a played game")

    box = played_box(season, games)
    d = d.merge(box, on=["team", "key", "gameday"], how="left")
    pff = played_pff(season, pd.concat([box[["team", "key"]], d[["team", "key"]]]))
    if pff is not None:
        d = d.merge(pff, on=["team", "key", "week"], how="left")
        print("  PFF snap counts used where available")
    else:
        d["pff"] = pd.NA
        d["snap_ratio"] = pd.NA
        print("  no PFF data here: box-score stats only (a player with no pass, carry or catch counts as not played; "
              "linemen and defenders are unknown)")
    d["played"] = d["pff"].where(d["pff"].notna(), d["box"])
    d["source"] = pd.Series(["pff"] * len(d)).where(d["pff"].notna(), pd.Series(["box"] * len(d)).where(d["box"].notna(), "unknown"))
    # the last status before kickoff: Game Day where the league files one, else the last update
    d["last"] = d["Game Day"].where(d["Game Day"] != "", d["Update 2"].where(d["Update 2"] != "", d["Update 1"].where(d["Update 1"] != "", d["Initial"])))
    return d


def report(d: pd.DataFrame) -> None:
    print(f"{len(d)} rows, {(d['source'] != 'unknown').sum()} with an outcome "
          f"({(d['source'] == 'pff').sum()} from PFF snaps, {(d['source'] == 'box').sum()} from box scores)\n")

    known = d[d["source"] != "unknown"].copy()
    known["played"] = known["played"].astype(int)
    for stage in STAGES:
        t = known[known[stage] != ""].groupby(stage)["played"].agg(n="size", played="mean")
        t = t.loc[[s for s in ("Out", "Doubtful", "Questionable", "Probable", "Game Time Decision", "Out (1st Half)", "Available") if s in t.index]]
        print(f"{stage}: " + "; ".join(f"{s} {int(v.n)} listed, {v.played:.0%} played" for s, v in t.iterrows()))
    print()
    gd = known[known["Game Day"] != ""]
    t = gd.groupby(["conf", "Game Day"])["played"].agg(n="size", played="mean").reset_index()
    print("Game Day report by conference:")
    for conf, g in t.groupby("conf"):
        print(f"  {conf}: " + "; ".join(f"{r['Game Day']} {int(r.n)}, {r.played:.0%} played" for _, r in g.iterrows()))
    pos_group = lambda p: {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}.get(str(p or "").upper(), "other")
    if "pos" in known.columns:
        known["grp"] = known["pos"].map(pos_group)
        print("\nGame Day by position group (QB/RB/WR/TE):")
        for grp, g in known[known["Game Day"] != ""].groupby("grp"):
            t = g.groupby("Game Day")["played"].agg(n="size", played="mean")
            print(f"  {grp}: " + "; ".join(f"{s} {int(v.n)}, {v.played:.0%}" for s, v in t.iterrows()))
    print("\nLast status before kickoff (all leagues):")
    t = known.groupby("last")["played"].agg(n="size", played="mean")
    print("  " + "; ".join(f"{s} {int(v.n)}, {v.played:.0%} played" for s, v in t.iterrows()))
    if known["snap_ratio"].notna().any():
        print("\nWhen they played, snaps as a share of their usual (PFF; median, mean, n):")
        for col in ("last", "Initial", "Update 1", "Update 2"):
            g = known[(known["played"] == 1) & known["snap_ratio"].notna() & (known[col] != "")]
            t = g.groupby(col)["snap_ratio"].agg(["median", "mean", "size"])
            print(f"  by {col}: " + "; ".join(f"{s} {v['median']:.2f}/{v['mean']:.2f}/{int(v['size'])}" for s, v in t.iterrows()))
    miss = gd[(gd["Game Day"] == "Out") & (gd["played"] == 1)]
    if len(miss):
        print(f"\n{len(miss)} listed Out on the Game Day report but recorded a stat (name matches worth checking):")
        for r in miss.head(15).to_dict("records"):
            print(f"  {r['conf']} {r['team']} {r['player']} wk {r['week']} ({r['source']})")


if __name__ == "__main__":
    main()
