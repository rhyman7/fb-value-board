"""Weekly Review context: the facts behind "why" (Ryan, Nov 18, 2026).

Adds to each game in a week's review facts file:

- ``flow``: score by quarter, each team's drive outcomes and points per drive, garbage-time
  points, when the total went over the market's number, and the score that settled the cover.
- ``context``: division / conference game, rest days, each team's previous game (result,
  cover, total, the model's miss) and season record against the spread, a rematch, weather,
  kickoff slot, referee and coaches where the source has them.
- ``short``: players who came up short of their projection, with what the facts say about
  why: volume against the projection (targets, carries, attempts), efficiency, snap share
  (NFL), injury report status (NFL), the previous game and season average, the game script,
  the defenders credited on his targets, and whether he faced a former team.
- ``beat``: one line per player who beat his projection (Ryan is less interested in these).
- ``defense``: per team, the defenders who made plays, defenders who played far more than
  usual (NFL snap counts) or recorded far more than usual (college box score), and regulars
  (offense and defense) who usually play and didn't.
- ``former_team``: players who played against a team they were on in the last three seasons
  (or earlier this season).

Sources. NFL: nflverse schedules, snap counts, injury reports, weekly player stats and weekly
rosters (downloaded). College: the college pipeline's committed caches in data/cfb/cache
(games, player_games, player_info), which come from the same cfbfastR files. Both leagues:
the board's own archives for previous games and projections.

Display only: nothing here feeds the model.
"""
from __future__ import annotations

import io
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NFLV = "https://github.com/nflverse/nflverse-data/releases/download/"
CFB_CACHE = ROOT / "data" / "cfb" / "cache"
GOOD_BAD = {"nfl": 10, "cfb": 14}     # a cover by this many points or more was a good / bad week
STAT_VOL = {"pass_yds": "pass_att", "rush_yds": "carries", "rec_yds": "targets"}
STAT_NAME = {"pass_yds": "passing yards", "rush_yds": "rushing yards", "rec_yds": "receiving yards"}


def _get(url: str, kind: str = "parquet"):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fb-review"}), timeout=300) as r:
        b = io.BytesIO(r.read())
    return pd.read_parquet(b) if kind == "parquet" else pd.read_csv(b, low_memory=False)


def _num(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), d)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

class Sources:
    """Everything the context needs for one league-season, loaded once."""

    def __init__(self, league: str, season: int, local: dict | None = None):
        self.league, self.season = league, season
        local = local or {}
        L = lambda key, url, kind="parquet": (pd.read_parquet(local[key]) if str(local[key]).endswith(".parquet")
                                             else pd.read_csv(local[key], low_memory=False)) if key in local else _get(url, kind)
        self.archives = self._archives()
        if league == "nfl":
            self.sched = L("games", NFLV + "schedules/games.csv", "csv")
            self.sched = self.sched[self.sched["season"] == season]
            self.snaps = L("snaps", NFLV + f"snap_counts/snap_counts_{season}.parquet")
            self.inj = L("injuries", NFLV + f"injuries/injuries_{season}.parquet")
            self.stats = L("stats", NFLV + f"stats_player/stats_player_week_{season}.parquet")
            ros = [L(f"roster_{s}", NFLV + f"weekly_rosters/roster_weekly_{s}.parquet") for s in range(season - 3, season + 1)]
            self.rosters = pd.concat([r[["season", "week", "team", "gsis_id", "pfr_id", "full_name", "position"]] for r in ros])
            self.pfr2gsis = (self.rosters.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
                             .set_index("pfr_id")["gsis_id"].to_dict())
        else:
            self.games = pd.read_csv(CFB_CACHE / f"games_{season}.csv")
            self.teams = pd.read_csv(CFB_CACHE / f"teams_{season}.csv")
            self.pg = {s: pd.read_csv(CFB_CACHE / f"player_games_{s}.csv") for s in range(season - 3, season + 1)
                       if (CFB_CACHE / f"player_games_{s}.csv").exists()}
            info = [pd.read_csv(CFB_CACHE / f"player_info_{s}.csv") for s in range(season - 3, season + 1)
                    if (CFB_CACHE / f"player_info_{s}.csv").exists()]
            self.names = pd.concat(info).drop_duplicates("player_id", keep="last").set_index("player_id")["name"].str.strip().to_dict()
            self.pos = pd.concat(info).drop_duplicates("player_id", keep="last").set_index("player_id")["position"].to_dict()
            # board abbreviation <-> ESPN team id, deduplicated the way cfbmodel.teams.load_teams does
            t = self.teams.copy()
            dup = t["abbr"].duplicated(keep=False)
            t.loc[dup, "abbr"] = t.loc[dup, "abbr"] + t.loc[dup, "team_id"].astype(str).str[-2:]
            self.abbr2id = dict(zip(t["abbr"], t["team_id"].astype(int)))
            self.id2abbr = {v: k for k, v in self.abbr2id.items()}

    def _archives(self) -> dict:
        d = ROOT / "docs" / ("archive" if self.league == "nfl" else "cfb/archive")
        out = {}
        for f in sorted(d.glob(f"{self.season}_w*.json")):
            out[int(f.stem.split("_w")[1])] = json.loads(f.read_text())
        return out


# --------------------------------------------------------------------------
# Previous games, rest, rematch, record against the spread (both leagues, from the archives)
# --------------------------------------------------------------------------

def _team_games(src: Sources, team: str, before_week: int) -> list[dict]:
    out = []
    for w in sorted(src.archives):
        if w >= before_week:
            continue
        A = src.archives[w]
        for gid, g in A["games"].items():
            if team not in (g["home"], g["away"]):
                continue
            res = (A.get("results") or {}).get(gid)
            if not res or res.get("home_score") is None:
                continue
            home = g["home"] == team
            hs, as_ = float(res["home_score"]), float(res["away_score"])
            margin = (hs - as_) * (1 if home else -1)
            line = g["market"]["spread"] * (1 if home else -1)        # team's expected margin
            mod = g["model"]["margin"] * (1 if home else -1)
            out.append({"week": w, "gid": gid, "opp": g["away"] if home else g["home"], "home": home,
                        "kickoff": g.get("kickoff"), "score": f"{int(hs if home else as_)}-{int(as_ if home else hs)}",
                        "margin": margin, "cover": round(margin - line, 1), "line": line,
                        "total": hs + as_, "total_vs_line": round(hs + as_ - g["market"]["total"], 1),
                        "model_miss": round(margin - mod, 1)})
    return out


def _days(a: str | None, b: str | None):
    try:
        f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        return round((f(b) - f(a)).total_seconds() / 86400)
    except Exception:
        return None


def team_context(src: Sources, g: dict, week: int) -> dict:
    lg = src.league
    out = {"teams": {}}
    for side in ("away", "home"):
        t = g[side]
        past = _team_games(src, t, week)
        ats = Counter("W" if x["cover"] > 0 else "L" if x["cover"] < 0 else "P" for x in past)
        prev = past[-1] if past else None
        tc = {"ats": [ats["W"], ats["L"], ats["P"]], "games": len(past)}
        if prev:
            tc["prev"] = {k: prev[k] for k in ("week", "opp", "home", "score", "margin", "cover", "total_vs_line", "model_miss")}
            tc["prev"]["label"] = ("good week" if prev["cover"] >= GOOD_BAD[lg] else
                                   "bad week" if prev["cover"] <= -GOOD_BAD[lg] else None)
            tc["rest"] = _days(prev["kickoff"], g.get("kickoff"))
            tc["after_bye"] = prev["week"] < week - 1
        # last three: covers and how far the model missed them
        tc["last3"] = [{"week": x["week"], "opp": x["opp"], "cover": x["cover"], "model_miss": x["model_miss"]} for x in past[-3:]]
        out["teams"][t] = tc
    # rematch this season
    for x in _team_games(src, g["home"], week):
        if x["opp"] == g["away"]:
            out["rematch"] = {"week": x["week"], "winner": g["home"] if x["margin"] > 0 else g["away"],
                              "score": x["score"] if x["margin"] > 0 else "-".join(reversed(x["score"].split("-"))),
                              "home_was": g["home"] if x["home"] else g["away"]}
    if lg == "nfl":
        r = src.sched[src.sched["game_id"] == g["id"]]
        if len(r):
            r = r.iloc[0]
            out["division"] = bool(r["div_game"])
            out["rest_official"] = {g["away"]: _int(r["away_rest"]), g["home"]: _int(r["home_rest"])}
            for t, k in ((g["away"], "away"), (g["home"], "home")):
                if out["rest_official"][t] is not None:
                    out["teams"][t]["rest"] = out["rest_official"][t]
            out["coaches"] = {g["away"]: r["away_coach"], g["home"]: r["home_coach"]}
            out["referee"] = r["referee"] if isinstance(r["referee"], str) else None
            out["slot"] = f"{r['weekday']} {r['gametime']}"
            out["roof"], out["surface"] = r["roof"], r["surface"]
            out["temp"], out["wind"] = _int(r["temp"]), _int(r["wind"])
            out["overtime"] = bool(r["overtime"])
    else:
        row = src.games[src.games["game_id"] == int(g.get("game_id") or 0)]
        if len(row):
            r = row.iloc[0]
            out["conference_game"] = bool(r["conference_game"])
            out["temp"], out["wind"] = _int(r.get("temp")), _int(r.get("wind"))
            tz = {g["home"]: r.get("home_tz"), g["away"]: r.get("away_tz")}
            out["time_zones"] = tz if tz[g["home"]] != tz[g["away"]] else None
            out["talent"] = {g["home"]: _int(r.get("home_talent")), g["away"]: _int(r.get("away_talent"))}
    out["neutral"] = bool(g.get("neutral"))
    return out


def _int(x):
    try:
        return None if pd.isna(x) else int(round(float(x)))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Game flow from the play-by-play
# --------------------------------------------------------------------------

def _timeline(p: pd.DataFrame, league: str):
    """Scoring changes in order: (quarter, clock, team, points, home score, away score)."""
    if league == "nfl":
        h, a = p["total_home_score"].ffill().fillna(0), p["total_away_score"].ffill().fillna(0)
        q, clock = p["qtr"], p["time"].astype(str)
    else:
        h = pd.to_numeric(p["end.homeScore"], errors="coerce").ffill().fillna(0)
        a = pd.to_numeric(p["end.awayScore"], errors="coerce").ffill().fillna(0)
        q = pd.to_numeric(p["period"], errors="coerce")
        clock = p["clock.displayValue"].astype(str)
    ev, ph, pa = [], 0.0, 0.0
    for i in range(len(p)):
        hh, aa = float(h.iloc[i]), float(a.iloc[i])
        if hh < ph or aa < pa:          # bad rows: scores never go down
            continue
        if hh > ph or aa > pa:
            ev.append({"q": int(q.iloc[i]) if not pd.isna(q.iloc[i]) else None, "clock": clock.iloc[i],
                       "side": "home" if hh > ph else "away", "pts": int((hh - ph) + (aa - pa)), "h": int(hh), "a": int(aa)})
            ph, pa = hh, aa
    return ev


def _drive_kind(s: str) -> str:
    s = str(s).lower()
    for k, keys in (("TD", ("touchdown", "td")), ("FG", ("field goal", "fg")), ("Punt", ("punt",)),
                    ("Turnover", ("interception", "int", "fumble", "turnover")), ("Downs", ("downs",)),
                    ("Half/game end", ("end of", "half", "game"))):
        if k == "FG" and ("missed" in s or "blocked" in s):
            return "Missed FG"
        if k == "Turnover" and "downs" in s:
            continue
        if any(x in s for x in keys):
            return k
    return "Other"


def game_flow(p: pd.DataFrame, g: dict, league: str) -> dict:
    H, A = g["home"], g["away"]
    ev = _timeline(p, league)
    quarters = {A: [], H: []}
    for qn in range(1, 6):
        upto = [e for e in ev if e["q"] is not None and e["q"] <= qn]
        before = [e for e in ev if e["q"] is not None and e["q"] <= qn - 1]
        if qn == 5 and not any(e["q"] and e["q"] >= 5 for e in ev):
            break
        hq = (upto[-1]["h"] if upto else 0) - (before[-1]["h"] if before else 0)
        aq = (upto[-1]["a"] if upto else 0) - (before[-1]["a"] if before else 0)
        quarters[H].append(hq)
        quarters[A].append(aq)
    # drives
    drives = {}
    if league == "nfl":
        d = p[p["posteam"].notna() & p["fixed_drive"].notna()].groupby(["posteam", "fixed_drive"])["fixed_drive_result"].first()
        for (team, _), res in d.items():
            drives.setdefault(team, Counter())[_drive_kind(res)] += 1
    else:
        tm = {}
        d = p[p["drive.id"].notna()].groupby("drive.id").agg(team=("pos_team_id", "first"), res=("drive.result", "first"))
        home_id = p["homeTeamId"].iloc[0]
        for _, r in d.iterrows():
            team = H if r["team"] == home_id else A
            drives.setdefault(team, Counter())[_drive_kind(r["res"])] += 1
    pts = {H: ev[-1]["h"] if ev else 0, A: ev[-1]["a"] if ev else 0}
    drive_out = {t: {"n": sum(c.values()), "per_drive": round(pts[t] / max(1, sum(c.values())), 2), **dict(c)} for t, c in drives.items()}
    # garbage time: scores in the 4th quarter or later when the leader was up 17+ before the score
    garbage = Counter()
    for i, e in enumerate(ev):
        prev = ev[i - 1] if i else {"h": 0, "a": 0}
        if e["q"] and e["q"] >= 4 and abs(prev["h"] - prev["a"]) >= 17:
            garbage[H if e["side"] == "home" else A] += e["pts"]
    # when the total went over, and the score that settled the cover for the model's side lean
    mt, ms = g["market"]["total"], g["market"]["spread"]
    over_at = next((f"Q{e['q']} {e['clock']}" for e in ev if e["h"] + e["a"] > mt), None) if mt is not None else None
    settle = None
    if ms is not None:
        res = lambda e: np.sign((e["h"] - e["a"]) - ms)
        final = res(ev[-1]) if ev else 0
        for i in range(len(ev) - 1, -1, -1):
            if res(ev[i]) != final:
                nxt = ev[i + 1] if i + 1 < len(ev) else None
                if nxt:
                    settle = {"q": nxt["q"], "clock": nxt["clock"], "team": H if nxt["side"] == "home" else A,
                              "pts": nxt["pts"], "score": f"{nxt['a']}-{nxt['h']}", "late": bool(nxt["q"] and nxt["q"] >= 4)}
                break
    # pace
    plays = {}
    for t in (H, A):
        if league == "nfl":
            plays[t] = int(((p["posteam"] == t) & ((p["pass"] == 1) | (p["rush"] == 1))).sum())
    lead_q3 = None
    q3 = [e for e in ev if e["q"] is not None and e["q"] <= 3]
    if q3:
        lead_q3 = {"home": q3[-1]["h"], "away": q3[-1]["a"]}
    return {"quarters": quarters, "drives": drive_out, "garbage_pts": dict(garbage), "over_at": over_at,
            "cover_settled": settle, "plays": plays or None, "after_q3": lead_q3,
            "scores": [{"q": e["q"], "clock": e["clock"], "team": H if e["side"] == "home" else A, "pts": e["pts"],
                        "score": f"{e['a']}-{e['h']}"} for e in ev]}


# --------------------------------------------------------------------------
# Players: who came up short and why, who beat it
# --------------------------------------------------------------------------

def _script(p: pd.DataFrame, team: str, league: str) -> dict:
    if league == "nfl":
        t = p[(p["posteam"] == team) & ((p["pass"] == 1) | (p["rush"] == 1))]
        d = t["score_differential"]
    else:
        t = p[(p["scrim"] if "scrim" in p else True) & (p["pos_team_id"] == team)]
        d = pd.to_numeric(t["pos_score_diff_start"], errors="coerce")
    n = max(1, len(t))
    return {"trailing9": round(float((d <= -9).sum()) / n, 2), "leading9": round(float((d >= 9).sum()) / n, 2)}


def _nfl_player_rows(src: Sources, pid: str, week: int):
    s = src.stats[(src.stats["player_id"] == pid) & (src.stats["season_type"] == "REG")]
    return s[s["week"] < week].sort_values("week"), s[s["week"] == week]


def _cfb_player_rows(src: Sources, pid: str, game_id: int, week: int):
    pg = src.pg[src.season]
    s = pg[pg["player_id"] == int(pid)].merge(src.games[["game_id", "week"]], on="game_id")
    return s[s["week"] < week].sort_values("week"), s[s["game_id"] == game_id]


def players(src: Sources, A: dict, gid: str, g: dict, week: int, p: pd.DataFrame, former: dict) -> tuple[list, list]:
    lg = src.league
    act = (A.get("actuals") or {}).get(gid, {})
    short, beat, dnp = [], [], []
    scripts = {}
    for pl in (A.get("players") or {}).get(gid, []):
        x = act.get(pl["id"])
        stats = {"QB": ["pass_yds", "rush_yds"], "RB": ["rush_yds", "rec_yds"], "WR": ["rec_yds"], "TE": ["rec_yds"]}.get(pl["pos"], [])
        for st in stats:
            pj = (pl.get("proj") or {}).get(st) or 0
            if pj < 20:
                continue
            av = (x or {}).get(st, 0.0) or 0.0
            miss = av - pj
            rel = miss / pj
            if miss >= 0:
                if miss >= max(20, 0.4 * pj):
                    beat.append({"name": pl["name"], "pos": pl["pos"], "team": pl["team"], "stat": st, "proj": round(pj), "act": round(av)})
                continue
            if miss > -max(15, 0.25 * pj):
                continue
            if not x:
                dnp.append({"name": pl["name"], "pos": pl["pos"], "team": pl["team"], "stat": st, "proj": round(pj),
                            "status": pl.get("status")})
                continue
            item = {"name": pl["name"], "pos": pl["pos"], "team": pl["team"], "stat": st, "proj": round(pj, 1),
                    "act": av, "miss": round(miss, 1), "pct": round(100 * rel), "played": bool(x)}
            # volume and efficiency against the projection
            vk = STAT_VOL[st]
            vproj = (pl.get("proj") or {}).get(vk)
            vact = (x or {}).get(vk)
            if lg == "cfb" and vk == "targets":
                vact = None
            if vproj:
                item["volume"] = {"kind": vk, "proj": round(vproj, 1), "act": vact}
            if vact and vproj:
                item["per"] = {"proj": round(pj / vproj, 1), "act": round(av / vact, 1) if vact else None}
            why = pl.get("why") or {}
            if isinstance(why, dict):
                item["recent_avg"] = _num((why.get("recent") or {}).get(st))
            # history: previous game, season average
            if lg == "nfl":
                hist, now = _nfl_player_rows(src, pl["id"], week)
                col = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards"}[st]
                if len(hist):
                    item["prev"] = {"week": int(hist.iloc[-1]["week"]), "opp": hist.iloc[-1]["opponent_team"], "val": _num(hist.iloc[-1][col], 0)}
                    item["season_avg"] = _num(hist[col].mean())
                    item["season_low"] = _num(hist[col].min(), 0)
                if len(now):
                    n0 = now.iloc[0]
                    if st == "rec_yds":
                        item["target_share"] = _num(n0.get("target_share"), 2)
                        if len(hist):
                            item["target_share_avg"] = _num(hist["target_share"].mean(), 2)
                    if st == "pass_yds":
                        item["qb"] = {"sacks": _int(n0.get("sacks_suffered")), "ints": _int(n0.get("passing_interceptions")),
                                      "cpoe": _num(n0.get("passing_cpoe")), "att": _int(n0.get("attempts"))}
                sn = src.snaps[(src.snaps["season"] == src.season)]
                name = pl["name"]
                me = sn[(sn["player"] == name) & (sn["team"] == pl["team"])]
                if len(me):
                    cur = me[me["week"] == week]
                    pre = me[me["week"] < week]
                    if len(cur):
                        item["snap_pct"] = _num(cur.iloc[0]["offense_pct"], 2)
                    if len(pre):
                        item["snap_pct_avg"] = _num(pre["offense_pct"].tail(4).mean(), 2)
                inj = src.inj[(src.inj["gsis_id"] == pl["id"]) & (src.inj["week"] == week)]
                if len(inj):
                    r = inj.iloc[-1]
                    item["injury_report"] = {"status": r["report_status"] if isinstance(r["report_status"], str) else None,
                                             "injury": r["report_primary_injury"] if isinstance(r["report_primary_injury"], str) else None,
                                             "practice": r["practice_status"] if isinstance(r["practice_status"], str) else None}
                # defenders credited on his targets / carries
                if st == "rec_yds":
                    t = p[p["receiver_player_id"] == pl["id"]]
                    cr = Counter()
                    for c in ("pass_defense_1_player_name", "pass_defense_2_player_name", "interception_player_name"):
                        cr.update(t[c].dropna())
                    tk = Counter(t.loc[t["complete_pass"] == 1, "solo_tackle_1_player_name"].dropna())
                    item["defenders"] = {"broke_up": cr.most_common(3), "tackled_after_catch": tk.most_common(2)}
                    item["targets_catches"] = [int(len(t)), int((t["complete_pass"] == 1).sum())]
                    adot = t["air_yards"].mean()
                    item["adot"] = _num(adot)
            else:
                hist, now = _cfb_player_rows(src, pl["id"], int(g.get("game_id") or 0), week)
                if len(hist):
                    item["prev"] = {"week": int(hist.iloc[-1]["week"]), "val": _num(hist.iloc[-1][st], 0)}
                    item["season_avg"] = _num(hist[st].mean())
                    item["season_low"] = _num(hist[st].min(), 0)
                if len(now) and st == "rec_yds":
                    item["volume"] = {"kind": "targets", "proj": None, "act": _int(now.iloc[0]["targets"])}
                if st == "rec_yds":
                    t = p[p["receiver_player_id"].astype(str) == str(pl["id"])]
                    cr = Counter()
                    for c in ("pass_breakup_player_name", "interception_player_name"):
                        if c in t:
                            cr.update(t[c].dropna())
                    item["defenders"] = {"broke_up": cr.most_common(3)}
            # game script for his team
            key = pl["team"] if lg == "nfl" else src.abbr2id.get(pl["team"])
            if key not in scripts:
                scripts[key] = _script(p, key, lg)
            item["script"] = scripts[key]
            if pl["name"] in former:
                item["former_team"] = former[pl["name"]]
            short.append(item)
    short.sort(key=lambda m: (m["pct"], m["miss"]))
    seen, d2 = set(), []
    for d in dnp:
        if d["name"] not in seen:
            seen.add(d["name"])
            d2.append(d)
    return short[:8], beat[:6], d2[:6]


# --------------------------------------------------------------------------
# Defense and personnel: playmakers, unusual snaps, regulars who didn't play
# --------------------------------------------------------------------------

DEF_POS = {"DE", "DT", "NT", "LB", "OLB", "ILB", "MLB", "CB", "S", "FS", "SS", "DB", "DL", "EDGE"}


def personnel_nfl(src: Sources, g: dict, week: int) -> dict:
    out = {}
    sn = src.snaps[src.snaps["season"] == src.season]
    st = src.stats[(src.stats["season"] == src.season) & (src.stats["week"] == week)]
    inj = src.inj[src.inj["week"] == week]
    for t in (g["away"], g["home"]):
        me = sn[sn["team"] == t]
        cur, pre = me[me["week"] == week], me[me["week"] < week]
        last3 = sorted(pre["week"].unique())[-3:]
        reg = pre[pre["week"].isin(last3)]
        info = {"playmakers": [], "more_than_usual": [], "regulars_out": []}
        d = st[st["team"] == t].copy()
        if len(d):
            d["impact"] = (d["def_sacks"].fillna(0) * 2 + d["def_interceptions"].fillna(0) * 3 + d["def_fumbles_forced"].fillna(0) * 2
                           + d["def_pass_defended"].fillna(0) + d["def_tackles_for_loss"].fillna(0) + d["def_qb_hits"].fillna(0) * 0.5)
            for _, r in d[d["impact"] >= 2].sort_values("impact", ascending=False).head(4).iterrows():
                bits = []
                for k, lbl in (("def_sacks", "sack"), ("def_interceptions", "INT"), ("def_fumbles_forced", "forced fumble"),
                               ("def_pass_defended", "pass defended"), ("def_tackles_for_loss", "TFL"), ("def_qb_hits", "QB hit")):
                    v = r.get(k) or 0
                    if v:
                        bits.append(f"{_f(v)} {lbl}{'s' if v > 1 and lbl not in ('INT', 'TFL') else ''}")
                info["playmakers"].append({"name": r["player_display_name"], "pos": r["position"], "line": ", ".join(bits)})
        for side, col in (("defense", "defense_pct"), ("offense", "offense_pct")):
            for _, r in cur[cur[col] >= 0.5].iterrows():
                hist = pre[pre["player"] == r["player"]][col]
                avg = float(hist.tail(4).mean()) if len(hist) else 0.0
                if avg <= 0.25:
                    info["more_than_usual"].append({"name": r["player"], "pos": r["position"], "side": side,
                                                    "pct": round(float(r[col]), 2), "usual": round(avg, 2),
                                                    "first_game": not len(hist)})
            if len(last3):
                avg = reg.groupby("player")[col].sum() / len(last3)
                for name, v in avg[avg >= 0.6].items():
                    row = cur[cur["player"] == name]
                    if not len(row) or float(row.iloc[0][col]) < 0.1:
                        pos = reg[reg["player"] == name]["position"].iloc[-1]
                        rep = inj[(inj["team"] == t) & (inj["full_name"] == name)]
                        info["regulars_out"].append({"name": name, "pos": pos, "side": side, "usual": round(float(v), 2),
                                                     "report": (rep.iloc[-1]["report_status"] if len(rep) and isinstance(rep.iloc[-1]["report_status"], str) else None)})
        out[t] = info
    return out


def personnel_cfb(src: Sources, g: dict, week: int, p: pd.DataFrame) -> dict:
    out = {}
    pg = src.pg[src.season].merge(src.games[["game_id", "week"]], on="game_id")
    gid = int(g.get("game_id") or 0)
    for t in (g["away"], g["home"]):
        tid = src.abbr2id.get(t)
        info = {"playmakers": [], "more_than_usual": [], "regulars_out": []}
        me = pg[pg["team_id"] == tid]
        cur, pre = me[me["game_id"] == gid], me[me["week"] < week]
        last3 = sorted(pre["week"].unique())[-3:]
        # playmakers: from the play-by-play credits for this team's defense
        dp = p[p["def_pos_team_id"] == tid] if "def_pos_team_id" in p else p.iloc[0:0]
        cr = defaultdict(Counter)
        for c, lbl in (("sack_player_name", "sack"), ("interception_player_name", "INT"),
                       ("pass_breakup_player_name", "pass breakup"), ("fumble_forced_player_name", "forced fumble")):
            if c in dp:
                for n in dp[c].dropna():
                    cr[n][lbl] += 1
        score = lambda c: c["sack"] * 2 + c["INT"] * 3 + c["forced fumble"] * 2 + c["pass breakup"]
        for n, c in sorted(cr.items(), key=lambda kv: -score(kv[1]))[:4]:
            if score(c) >= 2:
                info["playmakers"].append({"name": n, "pos": None, "line": ", ".join(f"{v} {k}{'s' if v > 1 and k != 'INT' else ''}" for k, v in c.items())})
        # defenders: tackles far above usual, and regular tacklers with no stat this game
        if len(last3):
            reg = pre[pre["week"].isin(last3)]
            avg = reg.groupby("player_id")["tackles"].sum() / len(last3)
            for pid, v in avg[avg >= 4].items():
                row = cur[cur["player_id"] == pid]
                if not len(row) or float(row.iloc[0]["tackles"] or 0) == 0:
                    info["regulars_out"].append({"name": src.names.get(pid, str(pid)), "pos": src.pos.get(pid), "side": "defense",
                                                 "usual": round(float(v), 1), "unit": "tackles a game", "report": None})
            for _, r in cur[cur["tackles"] >= 6].iterrows():
                h = pre[pre["player_id"] == r["player_id"]]["tackles"]
                if (h.tail(4).mean() if len(h) else 0) <= 2:
                    info["more_than_usual"].append({"name": src.names.get(r["player_id"], str(r["player_id"])), "pos": src.pos.get(r["player_id"]),
                                                    "side": "defense", "pct": None, "tackles": int(r["tackles"]),
                                                    "usual": round(float(h.tail(4).mean()), 1) if len(h) else 0.0, "first_game": not len(h)})
            # offense regulars: 5+ touches a game who had none
            pre["touch"] = pre["carries"].fillna(0) + pre["targets"].fillna(0)
            reg = pre[pre["week"].isin(last3)]
            avg = reg.groupby("player_id")["touch"].sum() / len(last3)
            for pid, v in avg[avg >= 5].items():
                if not len(cur[cur["player_id"] == pid]):
                    info["regulars_out"].append({"name": src.names.get(pid, str(pid)), "pos": src.pos.get(pid), "side": "offense",
                                                 "usual": round(float(v), 1), "unit": "carries + targets a game", "report": None})
        out[t] = info
    return out


def _f(v):
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


# --------------------------------------------------------------------------
# Former teams
# --------------------------------------------------------------------------

def former_nfl(src: Sources, g: dict, week: int) -> list[dict]:
    st = src.stats[(src.stats["season"] == src.season) & (src.stats["week"] == week) & (src.stats["team"].isin([g["home"], g["away"]]))]
    sn = src.snaps[(src.snaps["season"] == src.season) & (src.snaps["week"] == week) & (src.snaps["team"].isin([g["home"], g["away"]]))]
    played = {}
    for _, r in st.iterrows():
        played[r["player_id"]] = (r["player_display_name"], r["team"], r["position"])
    for _, r in sn.iterrows():
        gs = src.pfr2gsis.get(r["pfr_player_id"])
        if gs and gs not in played and (r["offense_pct"] >= 0.25 or r["defense_pct"] >= 0.25):
            played[gs] = (r["player"], r["team"], r["position"])
    ros = src.rosters
    out = []
    for pid, (name, team, pos) in played.items():
        opp = g["home"] if team == g["away"] else g["away"]
        past = ros[(ros["gsis_id"] == pid) & (ros["team"] == opp) & ~((ros["season"] == src.season) & (ros["week"] >= week))]
        if len(past):
            seasons = sorted(past["season"].unique())
            out.append({"name": name, "team": team, "pos": pos, "vs": opp,
                        "when": f"{seasons[0]}" + (f"-{seasons[-1]}" if len(seasons) > 1 else "")
                                + (" (this season)" if src.season in seasons else "")})
    return out


def former_cfb(src: Sources, g: dict, week: int) -> list[dict]:
    gid = int(g.get("game_id") or 0)
    cur = src.pg[src.season]
    cur = cur[cur["game_id"] == gid]
    out = []
    for _, r in cur.iterrows():
        team = src.id2abbr.get(int(r["team_id"]))
        if team not in (g["home"], g["away"]):
            continue
        opp = g["home"] if team == g["away"] else g["away"]
        oid = src.abbr2id.get(opp)
        seasons = [s for s, df in src.pg.items() if s < src.season and ((df["player_id"] == r["player_id"]) & (df["team_id"] == oid)).any()]
        if seasons:
            out.append({"name": src.names.get(r["player_id"], str(r["player_id"])), "team": team, "pos": src.pos.get(r["player_id"]),
                        "vs": opp, "when": f"{seasons[0]}" + (f"-{seasons[-1]}" if len(seasons) > 1 else "")})
    return out


# --------------------------------------------------------------------------
# Put it together
# --------------------------------------------------------------------------

def add(facts: dict, league: str, season: int, week: int, pbp_week: pd.DataFrame, src: Sources) -> dict:
    A = src.archives[week]
    for g in facts["games"]:
        ag = A["games"][g["id"]]
        key = g["id"] if league == "nfl" else int(ag.get("game_id") or 0)
        p = pbp_week[pbp_week["game_id"] == key]
        if league == "cfb" and len(p):
            p = p.copy()
            p["scrim"] = (p["rush"].fillna(False).astype(bool) | p["pass"].fillna(False).astype(bool))
        g["game_id"] = ag.get("game_id")
        try:
            g["context"] = team_context(src, g, week)
        except Exception as e:
            g["context"] = {"error": str(e)}
        if not len(p):
            continue
        try:
            g["flow"] = game_flow(p, g, league)
        except Exception as e:
            g["flow"] = {"error": str(e)}
        try:
            former = former_nfl(src, g, week) if league == "nfl" else former_cfb(src, g, week)
        except Exception as e:
            former = []
            g["former_error"] = str(e)
        g["former_team"] = former
        try:
            g["short"], g["beat"], g["dnp"] = players(src, A, g["id"], g, week, p, {f["name"]: f for f in former})
        except Exception as e:
            g["short"], g["beat"], g["dnp"] = [], [], []
            g["short_error"] = str(e)
        try:
            g["personnel"] = personnel_nfl(src, g, week) if league == "nfl" else personnel_cfb(src, g, week, p)
        except Exception as e:
            g["personnel"] = {"error": str(e)}
    return facts
