"""Weekly Review: the facts behind each finished game's look back.

For one archived, graded week (NFL `docs/archive/`, college `docs/cfb/archive/`) and that
season's play-by-play (nflverse for the NFL, cfbfastR for college), writes
`review/data/<league>/<season>_w<week>.json` with, per game:

- what the board said at kickoff (the model's own number, the board's number, the market)
  and what happened, with each number's miss;
- the model's leans (whichever side of the line its own number sat on) and the staked bets;
- how the game was played (success rate and EPA per play outside garbage time, explosive
  plays, red zone, fourth downs, turnovers);
- the chance ledger: fumble bounces, defensive and return touchdowns, missed short field
  goals (points each moved the margin, at values fitted on 2023-25 by
  `experiments/review_chance_fit.py`), a starting quarterback leaving early, and the points
  defense and returns added to the total;
- the biggest player misses, early exits and players who weren't projected.

Display only (Ryan, Nov 17, 2026): nothing here feeds the model, the bets or the boards.
The written reads (`review/data/<league>/<season>_w<week>.reads.json`) are written from
this file, first by `nflmodel/reviewdraft.py`, then by the weekly review task.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review" / "data"
ARCH = {"nfl": ROOT / "docs" / "archive", "cfb": ROOT / "docs" / "cfb" / "archive"}
PBP_URL = {
    "nfl": "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.parquet",
    "cfb": "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-data/main/cfb/pbp/parquet/play_by_play_{s}.parquet",
}
# Points each chance item moved the final margin, fitted on 2023-25 (experiments/review_chance_fit.py).
# Fumble bounce: per fumble lost beyond half the team's fumbles. Non-offensive TD and short FG
# miss: per play. Refit with that script; don't hand-edit.
CHANCE_PTS = json.loads((ROOT / "data" / "review_chance.json").read_text()) if (ROOT / "data" / "review_chance.json").exists() else {
    "nfl": {"fumble": 4.0, "nonoff_td": 4.5, "fg_miss": 3.0},
    "cfb": {"fumble": 4.0, "nonoff_td": 4.5, "fg_miss": 3.0},
}
SHORT_FG = 45          # a miss from this distance or closer counts as a chance item
STAT = {"QB": ["pass_yds", "rush_yds"], "RB": ["rush_yds", "rec_yds"], "WR": ["rec_yds"], "TE": ["rec_yds"]}
MISS_MIN = 15          # a projection under this many yards is left out of the misses unless he had a game


# --------------------------------------------------------------------------
# Team box from play-by-play: one row per game and team, same fields for both leagues
# --------------------------------------------------------------------------

NFL_COLS = ["game_id", "week", "posteam", "defteam", "home_team", "away_team", "pass", "rush", "epa", "wp",
            "success", "yards_gained", "interception", "fumble", "fumble_lost", "fumble_out_of_bounds",
            "td_team", "touchdown", "play_type", "field_goal_attempt", "field_goal_result", "kick_distance",
            "down", "fourth_down_converted", "yardline_100", "drive", "qb_dropback", "passer_player_name",
            "qtr", "time", "desc", "home_score", "away_score", "spread_line", "total_line", "season_type"]


def box_nfl(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["scrim"] = ((p["pass"] == 1) | (p["rush"] == 1)) & p["epa"].notna() & p["posteam"].notna()
    p["live"] = p["scrim"] & (p["wp"].between(0.03, 0.97) | p["wp"].isna())
    p["e"] = p["epa"].clip(-4.5, 4.5)
    p["expl"] = p["scrim"] & (((p["pass"] == 1) & (p["yards_gained"] >= 20)) | ((p["rush"] == 1) & (p["yards_gained"] >= 12)))
    p["fum"] = (p["fumble"] == 1) & (p["fumble_out_of_bounds"].fillna(0) == 0)
    p["nonoff"] = (p["touchdown"] == 1) & p["td_team"].notna() & (
        (p["td_team"] != p["posteam"]) | p["play_type"].isin(["kickoff", "punt"]))
    p["fgms"] = (p["field_goal_attempt"] == 1) & (p["field_goal_result"] != "made") & (p["kick_distance"] <= SHORT_FG)
    p["fourth"] = p["scrim"] & (p["down"] == 4)
    p["rz"] = p["scrim"] & (p["yardline_100"] <= 20)
    p["rz_td"] = p["rz"] & (p["touchdown"] == 1) & (p["td_team"] == p["posteam"])
    nonoff = p[p["nonoff"]].groupby(["game_id", "td_team"]).size()
    rows = []
    for gid, g in p.groupby("game_id", sort=False):
        for team, t in g[g["posteam"].notna()].groupby("posteam", sort=False):
            live = t[t["live"]]
            db = t[(t["qb_dropback"] == 1) & t["passer_player_name"].notna()]
            rows.append(dict(
                game_id=gid, team=team,
                plays=int(t["scrim"].sum()), epa_play=float(live["e"].mean()) if len(live) else np.nan,
                success=float(live["success"].mean()) if len(live) else np.nan, explosive=int(t["expl"].sum()),
                ints=int(t["interception"].fillna(0).sum()), fumbles=int(t["fum"].sum()),
                fum_lost=int((t["fum"] & (t["fumble_lost"] == 1)).sum()),
                rz_td=int(t.loc[t["rz_td"], "drive"].nunique()), rz=int(t.loc[t["rz"], "drive"].nunique()),
                nonoff_td=int(nonoff.get((gid, team), 0)),
                fg_miss_short=int(t["fgms"].sum()),
                fourth_conv=int((t["fourth"] & (t["fourth_down_converted"] == 1)).sum()), fourth=int(t["fourth"].sum()),
                qb_exit=_qb_exit(db["passer_player_name"].tolist(), db, g, lambda r: (int(r["qtr"]), str(r["time"]))),
            ))
    return pd.DataFrame(rows)


def _qb_exit(names, db, g, when):
    """The starting QB, if he threw under 60% of the team's passes, with when he last threw."""
    if not names:
        return None
    starter = names[0]
    share = sum(n == starter for n in names) / len(names)
    if share >= 0.6:
        return None
    last = db[db["passer_player_name"] == starter].iloc[-1]
    q, clock = when(last)
    surname = re.split(r"[.\s]", starter.strip())[-1]
    text = g["desc"] if "desc" in g else g["text"]
    hurt = text.fillna("").str.contains(surname + r"\W.{0,40}injur", regex=True).any()
    return {"qb": starter, "share": round(share, 2), "qtr": q, "time": clock, "why": "injured" if hurt else "left the game"}


CFB_COLS = ["game_id", "week", "pos_team_id", "def_pos_team_id", "homeTeamId", "awayTeamId", "rush", "pass", "EPA",
            "EPA_success", "period", "pos_score_diff_start", "kneel_down", "penalty_no_play", "statYardage",
            "int", "fumble_vec", "fumble_lost", "touchdown", "type.text", "fg_attempt", "fg_made", "yds_fg",
            "down", "start.yardsToEndzone", "drive.id", "passer_player_name", "clock.displayValue", "text",
            "start.homeScore", "start.awayScore", "end.homeScore", "end.awayScore", "homeFinalScore", "awayFinalScore",
            "homeTeamSpread", "overUnder", "seasonType"]


def box_cfb(p: pd.DataFrame, id_to_abbr: dict | None = None) -> pd.DataFrame:
    from cfbmodel import config as CC
    p = p.copy()
    flag = lambda c: p[c].fillna(False).astype(bool)
    p["scrim"] = (flag("rush") | flag("pass")) & ~flag("kneel_down") & ~flag("penalty_no_play") & p["EPA"].notna() & p["pos_team_id"].notna()
    period = pd.to_numeric(p["period"], errors="coerce").fillna(4).astype(int)
    lead = pd.to_numeric(p["pos_score_diff_start"], errors="coerce").abs().fillna(0)
    p["live"] = p["scrim"] & ~(lead > period.map(CC.GARBAGE_LEAD).fillna(999))
    p["e"] = p["EPA"].clip(-CC.EPA_CLIP, CC.EPA_CLIP)
    p["succ"] = p["EPA_success"].fillna(False).astype(bool).astype(int)
    yds = pd.to_numeric(p["statYardage"], errors="coerce").fillna(0)
    p["expl"] = p["scrim"] & ((flag("pass") & (yds >= 20)) | (flag("rush") & (yds >= 12)))
    p["fum"] = flag("fumble_vec") & p["scrim"]
    p["flost"] = p["fum"] & flag("fumble_lost")
    # who scored: whichever side's score went up by 6+
    dh = pd.to_numeric(p["end.homeScore"], errors="coerce") - pd.to_numeric(p["start.homeScore"], errors="coerce")
    da = pd.to_numeric(p["end.awayScore"], errors="coerce") - pd.to_numeric(p["start.awayScore"], errors="coerce")
    p["td_side_id"] = np.where(dh >= 6, p["homeTeamId"], np.where(da >= 6, p["awayTeamId"], np.nan))
    tt = p["type.text"].fillna("")
    p["nonoff"] = flag("touchdown") & tt.str.contains("Touchdown") & ~tt.isin(["Rushing Touchdown", "Passing Touchdown"]) & p["td_side_id"].notna()
    fg = flag("fg_attempt") if "fg_attempt" in p else pd.Series(False, index=p.index)
    p["fgms"] = fg & ~flag("fg_made") & (pd.to_numeric(p["yds_fg"], errors="coerce") <= SHORT_FG)
    p["fourth"] = p["scrim"] & (pd.to_numeric(p["down"], errors="coerce") == 4)
    ytg = pd.to_numeric(p["start.yardsToEndzone"], errors="coerce")
    p["rz"] = p["scrim"] & (ytg <= 20)
    p["rz_td"] = p["rz"] & flag("touchdown") & (p["td_side_id"] == p["pos_team_id"])
    rows = []
    for gid, g in p.groupby("game_id", sort=False):
        for tid in (g["homeTeamId"].iloc[0], g["awayTeamId"].iloc[0]):
            t = g[g["pos_team_id"] == tid]
            live = t[t["live"]]
            db = t[flag("pass").loc[t.index] & t["passer_player_name"].notna()]
            rows.append(dict(
                game_id=int(gid), team_id=int(tid),
                final=float(g["homeFinalScore"].iloc[0] if tid == g["homeTeamId"].iloc[0] else g["awayFinalScore"].iloc[0]), team=(id_to_abbr or {}).get(int(tid), str(int(tid))),
                plays=int(t["scrim"].sum()), epa_play=float(live["e"].mean()) if len(live) else np.nan,
                success=float(live["succ"].mean()) if len(live) else np.nan, explosive=int(t["expl"].sum()),
                ints=int(t["int"].fillna(False).astype(bool).sum()), fumbles=int(t["fum"].sum()),
                fum_lost=int(t["flost"].sum()),
                rz_td=int(t.loc[t["rz_td"], "drive.id"].nunique()), rz=int(t.loc[t["rz"], "drive.id"].nunique()),
                nonoff_td=int((g["nonoff"] & (g["td_side_id"] == tid)).sum()),
                fg_miss_short=int(t["fgms"].sum()),
                fourth_conv=_cfb_fourth_conv(g, t), fourth=int(t["fourth"].sum()),
                qb_exit=_qb_exit(db["passer_player_name"].tolist(), db, g,
                                 lambda r: (int(r["period"]), str(r["clock.displayValue"]))),
            ))
    return pd.DataFrame(rows)


def _cfb_fourth_conv(g, t) -> int:
    """Fourth-down tries that kept the ball: the next scrimmage play is the same team's, or a TD."""
    sc = g[g["scrim"]]
    order = list(sc.index)
    n = 0
    for i in t.index[t["fourth"]]:
        if bool(t.loc[i, "touchdown"]) and t.loc[i, "td_side_id"] == t.loc[i, "pos_team_id"]:
            n += 1
            continue
        k = order.index(i)
        if k + 1 < len(order) and sc.loc[order[k + 1], "pos_team_id"] == t.loc[i, "pos_team_id"]:
            n += 1
    return n


def chance(h: dict, a: dict, pts: dict) -> dict:
    """Points the final margin moved (home minus away) on things a model can't see coming."""
    fum = (a["fum_lost"] - a["fumbles"] / 2) - (h["fum_lost"] - h["fumbles"] / 2)
    out = {
        "fumble_bounces": round(pts["fumble"] * fum, 1),
        "return_def_tds": round(pts["nonoff_td"] * (h["nonoff_td"] - a["nonoff_td"]), 1),
        "short_fg_misses": round(pts["fg_miss"] * (a["fg_miss_short"] - h["fg_miss_short"]), 1),
    }
    out["total"] = round(sum(out.values()), 1)
    out["total_pts_nonoff"] = 7 * (h["nonoff_td"] + a["nonoff_td"])
    out["qb_exits"] = [x for x in (a.get("qb_exit"), h.get("qb_exit")) if x]
    return out


# --------------------------------------------------------------------------
# One week
# --------------------------------------------------------------------------

def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if np.isnan(v) else round(v, 3)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def week_facts(league: str, season: int, week: int, box: pd.DataFrame) -> dict | None:
    path = ARCH[league] / f"{season}_w{week:02d}.json"
    A = json.loads(path.read_text())
    pts = CHANCE_PTS[league]
    games = []
    for gid, g in A["games"].items():
        res = (A.get("results") or {}).get(gid)
        if not res or res.get("home_score") is None:
            continue
        key = int(g["game_id"]) if league == "cfb" else gid
        if league == "cfb":
            bx = box[box["game_id"] == key]
            sides = {}
            if len(bx) == 2:
                # rows are ESPN home first; the board can list a neutral-site game the other way round,
                # so match by the final score the board graded
                r0, r1 = bx.iloc[0].to_dict(), bx.iloc[1].to_dict()
                flip = r0.get("final") is not None and float(r0["final"]) != float(res["home_score"]) and float(r1["final"]) == float(res["home_score"])
                sides[g["home"]], sides[g["away"]] = (r1, r0) if flip else (r0, r1)
        else:
            bx = box[box["game_id"] == key]
            sides = {r["team"]: r for r in bx.to_dict("records")}
        H, Aw = g["home"], g["away"]
        hs, as_ = float(res["home_score"]), float(res["away_score"])
        margin, total = hs - as_, hs + as_
        mk, md, mo = g["market"], g["proj"], g["model"]
        have = H in sides and Aw in sides
        h = _clean(sides[H]) if have else None
        a = _clean(sides[Aw]) if have else None
        ch = chance(h, a, pts) if have else None
        staked = []
        for b in g.get("bets", []):
            if (b.get("stake") or 0) > 0:
                r = next((x for x in res.get("bets", []) if x["market"] == b["market"] and x["side"] == b["side"]), {})
                staked.append({"pick": b["pick"], "odds": b.get("odds"), "ev": b.get("ev_rec", b.get("ev")),
                               "result": r.get("result"), "units": r.get("units"), "clv": r.get("clv")})
        leans = {}
        for k, mv, lv, act, hi in (("spread", mo["margin"], mk["spread"], margin, "home"),
                                   ("total", mo["total"], mk["total"], total, "over")):
            if lv is None or mv is None:
                continue
            side = hi if mv > lv else ("away" if hi == "home" else "under")
            d = act - lv
            leans[k] = {"side": side, "gap": round(mv - lv, 1),
                        "result": "push" if d == 0 else ("W" if (d > 0) == (side == hi) else "L")}
        act = (A.get("actuals") or {}).get(gid, {})
        misses, seen = [], set()
        for pl in (A.get("players") or {}).get(gid, []):
            x = act.get(pl["id"])
            seen.add(pl["id"])
            for s in STAT.get(pl["pos"], []):
                pj = (pl.get("proj") or {}).get(s) or 0
                if pj < MISS_MIN and not (x and (x.get(s) or 0) >= MISS_MIN):
                    continue
                av = (x or {}).get(s, 0.0) or 0.0
                misses.append({"name": pl["name"], "pos": pl["pos"], "team": pl["team"], "stat": s,
                               "proj": round(pj, 1), "act": av, "miss": round(av - pj, 1), "played": bool(x)})
        misses.sort(key=lambda m: -abs(m["miss"]))
        unproj = []
        for pid, x in act.items():
            if pid in seen or not isinstance(x, dict):
                continue
            yds = sum((x.get(k) or 0) for k in ("rush_yds", "rec_yds", "pass_yds"))
            if yds >= 40:
                unproj.append({"name": x.get("name"), "pos": x.get("pos"), "team": x.get("team"), "yds": yds,
                               "line": {k: x[k] for k in ("pass_yds", "rush_yds", "rec_yds", "receptions") if x.get(k)}})
        games.append({
            "id": gid, "slug": f"{Aw}-{H}".lower(), "date": g.get("date"), "time": g.get("time"),
            "away": Aw, "home": H, "away_name": g.get("away_name"), "home_name": g.get("home_name"),
            "away_qb": (g.get("away_qb") or {}).get("name"), "home_qb": (g.get("home_qb") or {}).get("name"),
            "neutral": g.get("neutral"), "backfilled": g.get("backfilled", False),
            "score": [as_, hs], "margin": margin, "total": total,
            "market": {"spread": mk["spread"], "total": mk["total"]},
            "board": {"margin": md["margin"], "total": md["total"], "home_win": g["win"]["adj"]},
            "model": {"margin": mo["margin"], "total": mo["total"], "home_win": g["win"]["raw"]},
            "err": {k: (round(v, 1) if v is not None else None) for k, v in {
                "model_margin": margin - mo["margin"], "market_margin": margin - mk["spread"],
                "board_margin": margin - md["margin"], "model_total": total - mo["total"],
                "market_total": total - mk["total"], "board_total": total - md["total"]}.items()},
            "leans": leans, "bets": staked,
            "box": {H: h, Aw: a} if have else None, "chance": ch,
            "neutral_margin": round(margin - ch["total"], 1) if ch else None,
            "why": [{"label": w["label"], "value": w["value"]} for w in (g.get("why") or [])][:6],
            "players": misses[:10], "unprojected": sorted(unproj, key=lambda u: -u["yds"])[:5],
        })
    if not games:
        return None
    n = len(games)
    def mean(k):
        v = [abs(x["err"][k]) for x in games if x["err"][k] is not None]
        return round(sum(v) / len(v), 2) if v else None
    summary = {
        "games": n, "graded_of": len(A["games"]),
        "lean_spread": [sum(x["leans"].get("spread", {}).get("result") == r for x in games) for r in ("W", "L", "push")],
        "lean_total": [sum(x["leans"].get("total", {}).get("result") == r for x in games) for r in ("W", "L", "push")],
        "model_closer_margin": sum(abs(x["err"]["model_margin"]) < abs(x["err"]["market_margin"]) for x in games),
        "model_closer_total": sum(abs(x["err"]["model_total"]) < abs(x["err"]["market_total"]) for x in games),
        "mae": {k: mean(k) for k in ("model_margin", "market_margin", "model_total", "market_total")},
        "chance_games": sum(1 for x in games if x["chance"] and big_chance(x)),
        "bets": [sum(b["result"] == r for x in games for b in x["bets"]) for r in ("W", "L", "P")],
        "units": round(sum((b["units"] or 0) for x in games for b in x["bets"]), 2),
    }
    colors = {}
    try:
        cm = json.loads((ROOT / "docs" / ("data.json" if league == "nfl" else "cfb/data.json")).read_text())["meta"]["colors"]
        colors = {t: cm[t] for x in games for t in (x["home"], x["away"]) if t in cm}
    except Exception:
        pass
    return {"league": league, "season": season, "week": week, "summary": summary, "colors": colors, "games": games}


def big_chance(x: dict) -> bool:
    c = x["chance"]
    return abs(c["total"]) >= 4 or c["total_pts_nonoff"] >= 14 or bool(c["qb_exits"])


def load_pbp(league: str, season: int, path: Path | None = None) -> pd.DataFrame:
    import io
    import urllib.request
    src = path or PBP_URL[league].format(s=season)
    cols = NFL_COLS if league == "nfl" else CFB_COLS
    if isinstance(src, str) and src.startswith("http"):
        with urllib.request.urlopen(src, timeout=300) as r:
            src = io.BytesIO(r.read())
    import pyarrow.parquet as pq
    have = set(pq.read_schema(src).names)
    if hasattr(src, "seek"):
        src.seek(0)
    return pd.read_parquet(src, columns=[c for c in cols if c in have])


def write_index(league: str) -> dict:
    d = OUT / league
    weeks = []
    for f in sorted(d.glob("*_w*.json")):
        if f.name.endswith(".reads.json"):
            continue
        x = json.loads(f.read_text())
        reads = f.with_name(f.stem + ".reads.json")
        r = json.loads(reads.read_text()) if reads.exists() else {}
        rg = r.get("games", {})
        verdicts = {k: {v: 0 for v in ("W", "L", "chance", "push")} for k in ("spread", "total")}
        for v in rg.values():
            for k, res in (v.get("verdict") or {}).items():
                if k in verdicts and res in verdicts[k]:
                    verdicts[k][res] += 1
        real = sorted((g for g in x["games"] if (rg.get(g["id"], {}).get("verdict") or {}).get("spread") != "chance"),
                      key=lambda g: -abs(g["err"]["model_margin"]))[:3]
        weeks.append({"file": f.name, "season": x["season"], "week": x["week"], "summary": x["summary"],
                      "reads": r.get("by", "none"), "read_games": len(rg), "verdicts": verdicts,
                      "real_misses": [{"slug": g["slug"], "away": g["away"], "home": g["home"],
                                       "miss": g["err"]["model_margin"], "mkt": g["err"]["market_margin"]} for g in real]})
    idx = {"league": league, "weeks": weeks}
    (d / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


def run(league: str, season: int, weeks: list[int], pbp_path: Path | None = None) -> list[Path]:
    p = load_pbp(league, season, pbp_path)
    p = p[p["week"].isin(weeks)]
    box = box_nfl(p) if league == "nfl" else box_cfb(p)
    out = []
    (OUT / league).mkdir(parents=True, exist_ok=True)
    for w in weeks:
        f = week_facts(league, season, w, box[box["game_id"].isin(
            p.loc[p["week"] == w, "game_id"].unique())] if len(box) else box)
        if f is None:
            print(f"  {league} {season} week {w}: no graded games")
            continue
        path = OUT / league / f"{season}_w{w:02d}.json"
        path.write_text(json.dumps(f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        miss = sum(1 for g in f["games"] if g["box"] is None)
        print(f"  {league} {season} week {w}: {len(f['games'])} games ({miss} without play-by-play)")
        out.append(path)
    write_index(league)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write the Weekly Review facts for graded weeks.")
    ap.add_argument("league", choices=["nfl", "cfb"])
    ap.add_argument("season", type=int)
    ap.add_argument("weeks", type=int, nargs="+")
    ap.add_argument("--pbp", type=Path, help="local play-by-play parquet (default: download)")
    a = ap.parse_args()
    run(a.league, a.season, a.weeks, a.pbp)
