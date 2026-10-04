"""Everything a single game's page needs beyond the prices.

For each upcoming game:
  breakdown   how the model spread and total are built, piece by piece
  qbs         each starter's rating, its effect in points, and a recent game log
  ranks       season-to-date offense against defense, value and league rank
  trends      each team's last ten games straight up, against the spread and over/under
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RATING_METRICS = ("pass", "rush", "sr", "pts")

# key, label, how to compute from season sums, format, is a bigger number better for the offense?
STAT_ROWS = [
    ("ppg", "Points per game", lambda s: s["pts"] / s["g"], "{:.1f}", True),
    ("ypg", "Yards per game", lambda s: s["yards"] / s["g"], "{:.0f}", True),
    ("pass_ypg", "Pass yards per game", lambda s: s["pass_yards"] / s["g"], "{:.0f}", True),
    ("rush_ypg", "Rush yards per game", lambda s: s["rush_yards"] / s["g"], "{:.0f}", True),
    ("epa_play", "EPA per play", lambda s: s["epa"] / s["plays"], "{:+.3f}", True),
    ("epa_pass", "EPA per pass", lambda s: s["pass_epa"] / s["pass_plays"], "{:+.3f}", True),
    ("epa_rush", "EPA per rush", lambda s: s["rush_epa"] / s["rush_plays"], "{:+.3f}", True),
    ("success", "Success rate", lambda s: 100 * s["succ"] / s["plays"], "{:.1f}%", True),
    ("third", "Third down", lambda s: 100 * s["third_conv"] / s["third_att"], "{:.1f}%", True),
    ("fourth", "Fourth down", lambda s: 100 * s["fourth_conv"] / s["fourth_att"], "{:.1f}%", True),
    ("rz", "Red zone touchdowns", lambda s: 100 * s["rz_td"] / s["rz_trips"], "{:.1f}%", True),
    ("sacks", "Sacks per game", lambda s: s["sacks"] / s["g"], "{:.1f}", False),
    ("ints", "Interceptions per game", lambda s: s["ints"] / s["g"], "{:.1f}", False),
    ("fumbles", "Fumbles lost per game", lambda s: s["fumbles"] / s["g"], "{:.1f}", False),
    ("to_epa", "Turnover EPA per game", lambda s: s["to_epa"] / s["g"], "{:+.1f}", True),
]


def _f(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class MatchupBuilder:
    def __init__(self, games: pd.DataFrame, qb_games: pd.DataFrame, stats: pd.DataFrame, engine, info: dict, season: int):
        self.games, self.engine, self.info, self.season = games, engine, info, season
        self.mc, self.tc = info["margin_coefs"], info["total_coefs"]
        self._ratings: dict[int, dict] = {}
        self.played = games[games.played].sort_values(["gameday", "game_id"])
        opp = {}
        for g in games.itertuples(index=False):
            opp[(g.game_id, g.home_team)] = ("vs " + g.away_team) if not g.neutral else ("vs " + g.away_team)
            opp[(g.game_id, g.away_team)] = "at " + g.home_team
        self.opp = opp
        self.week_of = dict(zip(games.game_id, zip(games.season, games.week)))
        self.qb_games = qb_games.sort_values("order")
        self.stat_tables = self._season_stats(stats)

    # ------------------------------------------------------------------ ratings
    def ratings(self, order: int) -> dict:
        if order not in self._ratings:
            self._ratings[order] = self.engine.ratings_at(order)
        return self._ratings[order]

    def team_rating(self, r: dict, i: int) -> float:
        """Points versus an average team, before any quarterback adjustment."""
        v = sum(self.mc[f"d_{m}"] * (r[m]["off"] - r[m]["def"]) for m in RATING_METRICS) + self.mc["d_mkt"] * r["mkt"]["R"]
        return float(v[i] - v.mean())

    def total_rating(self, r: dict, i: int) -> float:
        """Points this team adds to (or takes from) a game's total."""
        v = sum(self.tc[f"s_{m}"] * (r[m]["off"] + r[m]["def"]) for m in RATING_METRICS) + self.tc["s_mkt"] * r["mkt"]["T"]
        return float(v[i])

    # ------------------------------------------------------------------ breakdown
    def breakdown(self, g) -> dict:
        r = self.ratings(int(g.order))
        h, a = self.engine.tix[g.home_team], self.engine.tix[g.away_team]
        rh, ra = self.team_rating(r, h), self.team_rating(r, a)
        qh, qa = self.mc["d_qb"] * g.home_qb_delta, -self.mc["d_qb"] * g.away_qb_delta
        margin = [
            {"key": "ratings", "label": "Team ratings", "value": _f(rh - ra), "home_rating": _f(rh), "away_rating": _f(ra)},
            {"key": "hfa", "label": "Home field" if not g.neutral else "Home field (neutral site)", "value": _f(self.mc["hfa"] * g.hfa)},
            {"key": "rest", "label": "Rest", "value": _f(self.mc["rest_diff"] * g.rest_diff)},
            {"key": "home_qb", "label": f"{g.home_team} quarterback", "value": _f(qh)},
            {"key": "away_qb", "label": f"{g.away_team} quarterback", "value": _f(qa)},
        ]
        th, ta = self.total_rating(r, h), self.total_rating(r, a)
        base = self.info["total_intercept"] + self.tc["lg_pts"] * g.lg_pts + self.tc["s_mkt"] * r["mkt"]["avg_total"]
        total = [
            {"key": "base", "label": "League baseline", "value": _f(base)},
            {"key": "home", "label": f"{g.home_team} offense and defense", "value": _f(th)},
            {"key": "away", "label": f"{g.away_team} offense and defense", "value": _f(ta)},
            {"key": "qb", "label": "Quarterbacks", "value": _f(self.tc["s_qb"] * g.s_qb)},
            {"key": "roof", "label": "Indoors" if g.indoor else "Outdoors", "value": _f(self.tc["indoor"] * g.indoor)},
        ]
        return {"margin": margin, "margin_sum": _f(sum(x["value"] for x in margin)),
                "total": total, "total_sum": _f(sum(x["value"] for x in total)),
                "weight_margin": _f(g.w_margin, 3), "weight_total": _f(g.w_total, 3)}

    # ------------------------------------------------------------------ quarterbacks
    def quarterback(self, g, side: str) -> dict:
        r = self.ratings(int(g.order))
        qid, team = getattr(g, f"{side}_qb_id"), getattr(g, f"{side}_team")
        name = getattr(g, f"{side}_qb_name")
        if not isinstance(name, str):
            name = self.engine.q_name.get(qid, "Unknown")
        delta = float(getattr(g, f"{side}_qb_delta"))
        log = []
        if isinstance(qid, str):
            rows = self.qb_games[(self.qb_games.qb_id == qid) & (self.qb_games.dropbacks >= 5) & (self.qb_games.order < g.order)].tail(8)
            for q in rows.itertuples(index=False):
                s, w = self.week_of.get(q.game_id, (0, 0))
                log.append({"when": f"W{w} '{str(s)[2:]}", "opp": self.opp.get((q.game_id, q.team), ""), "team": q.team,
                            "db": int(q.dropbacks), "epa": _f(q.epa / q.dropbacks, 3)})
        return {
            "name": name, "team": team,
            "value": _f(getattr(g, f"{side}_qb_val"), 3),                 # EPA per dropback vs league average
            "team_mix": _f(getattr(g, f"{side}_qb_val") - delta, 3),       # the quarterbacks behind the team's pass rating
            "dropbacks": int(round(r["qb_n"].get(qid, 0))) if isinstance(qid, str) else 0,
            "points": _f(self.mc["d_qb"] * delta),                         # effect on his own team's margin
            "log": log,
        }

    # ------------------------------------------------------------------ season stats and ranks
    def _season_stats(self, stats: pd.DataFrame) -> dict | None:
        if stats is None or not len(stats):
            return None
        cur = stats[stats.season == self.season]
        # Early in a season there is nothing to rank yet; fall back to last season.
        use = self.season if cur.groupby("team").size().min() >= 1 and cur.team.nunique() >= 32 else self.season - 1
        st = stats[stats.season == use]
        if not len(st):
            return None
        num = [c for c in st.columns if c not in ("season", "game_id", "team", "opp", "order")]
        pg = self.games[(self.games.season == use) & self.games.played]
        pts_for = pd.concat([pg.set_index("home_team").home_score, pg.set_index("away_team").away_score]).groupby(level=0).sum()
        pts_against = pd.concat([pg.set_index("home_team").away_score, pg.set_index("away_team").home_score]).groupby(level=0).sum()
        n_games = pd.concat([pg.home_team, pg.away_team]).value_counts()
        tables = {}
        for unit, key, pts in (("off", "team", pts_for), ("def", "opp", pts_against)):
            s = st.groupby(key)[num].sum()
            s["g"], s["pts"] = n_games.reindex(s.index), pts.reindex(s.index)
            out = pd.DataFrame(index=s.index)
            for k, _, fn, _, off_good in STAT_ROWS:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[k] = fn(s).replace([np.inf, -np.inf], np.nan)
                better_high = off_good if unit == "off" else not off_good
                out[k + "_rank"] = out[k].rank(ascending=not better_high, method="min")
            tables[unit] = out
        tables["season"] = use
        return tables

    def ranks(self, g) -> dict | None:
        t = self.stat_tables
        if t is None:
            return None

        def side(off_team, def_team):
            rows = []
            for k, label, _, fmt, _ in STAT_ROWS:
                o, d = t["off"].loc[off_team], t["def"].loc[def_team]
                cell = lambda row: {"v": fmt.format(row[k]) if np.isfinite(row[k]) else "n/a",
                                    "rank": int(row[k + "_rank"]) if np.isfinite(row[k + "_rank"]) else None}
                rows.append({"label": label, "off": cell(o), "def": cell(d)})
            return rows

        if g.home_team not in t["off"].index or g.away_team not in t["off"].index:
            return None
        return {"season": int(t["season"]), "away_off": side(g.away_team, g.home_team), "home_off": side(g.home_team, g.away_team)}

    # ------------------------------------------------------------------ recent results
    def trends(self, team: str, before) -> dict:
        p = self.played
        mine = p[((p.home_team == team) | (p.away_team == team)) & (p.gameday < before)].tail(10)
        rows, su, ats, ou = [], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        for x in mine.itertuples(index=False):
            home = x.home_team == team
            pf, pa = (x.home_score, x.away_score) if home else (x.away_score, x.home_score)
            margin = pf - pa
            line = -x.spread_line if home else x.spread_line            # this team's line; negative = favoured
            cover = margin + line if np.isfinite(line) else np.nan
            total_pts = x.home_score + x.away_score
            res = lambda v: "W" if v > 0 else "L" if v < 0 else "P"
            row = {"when": f"W{int(x.week)} '{str(int(x.season))[2:]}", "opp": ("vs " if home else "at ") + (x.away_team if home else x.home_team),
                   "pf": int(pf), "pa": int(pa), "margin": int(margin), "su": "W" if margin > 0 else "L" if margin < 0 else "T",
                   "line": _f(line, 1), "ats": res(cover) if np.isfinite(cover) else None,
                   "total_line": _f(x.total_line, 1), "points": int(total_pts),
                   "ou": ("O" if total_pts > x.total_line else "U" if total_pts < x.total_line else "P") if np.isfinite(x.total_line) else None}
            rows.append(row)
            su[0 if margin > 0 else 1 if margin < 0 else 2] += 1
            if row["ats"]:
                ats["WLP".index(row["ats"])] += 1
            if row["ou"]:
                ou["OUP".index(row["ou"])] += 1
        rec = lambda v: f"{v[0]}-{v[1]}" + (f"-{v[2]}" if v[2] else "")
        return {"team": team, "su": rec(su), "ats": rec(ats), "ou": rec(ou), "games": rows[::-1]}

    # ------------------------------------------------------------------ one game
    def detail(self, g) -> dict:
        txt = lambda v: v if isinstance(v, str) else None
        return {
            "stadium": txt(g.stadium), "surface": txt(g.surface),
            "temp": _f(g.temp, 0), "wind": _f(g.wind, 0),
            "rest": {"away": _f(g.away_rest, 0), "home": _f(g.home_rest, 0)},
            "breakdown": self.breakdown(g),
            "qbs": {"away": self.quarterback(g, "away"), "home": self.quarterback(g, "home")},
            "ranks": self.ranks(g),
            "trends": {"away": self.trends(g.away_team, g.gameday), "home": self.trends(g.home_team, g.gameday)},
        }
