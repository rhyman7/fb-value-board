"""The college board's game pages, slate and ratings table, in the NFL board's data shape
(nflmodel/slate.py and nflmodel/matchup.py) so the same page design reads them.

Team codes on the board are ESPN abbreviations; the model keys teams by school name.
"""
from __future__ import annotations

import datetime as dt
import io
import urllib.request

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from nflmodel import market as MK, odds as O
from nflmodel.slate import bet_books, cap_game, with_market

from . import bets as B
from . import config as C
from .data import _retry

LINE_COLS = ["spread_line", "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds",
             "home_moneyline", "away_moneyline"]
WIND_LO, WIND_HI = 10.0, 20.0


def _f(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def game_key(season, week, away, home) -> str:
    return f"{int(season)}_{int(week):02d}_{away}_{home}"


# --------------------------------------------------------------------------
# Season box-score stats for the matchup ranks
# --------------------------------------------------------------------------

def load_team_box(season: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"team_box_{season}.csv"
    if refresh or not path.exists():
        def go():
            url = f"{C.BASE}/team_box/parquet/team_box_{season}.parquet"
            req = urllib.request.Request(url, headers={"User-Agent": "cfb-model"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return pd.read_parquet(io.BytesIO(r.read()))
        t = _retry(go)
        split = lambda c, i: pd.to_numeric(t[c].astype(str).str.extract(r"(\d+)\s*[-/]\s*(\d+)")[i], errors="coerce")
        num = lambda c: pd.to_numeric(t[c], errors="coerce")
        out = pd.DataFrame({
            "game_id": t["game_id"].astype(int), "team_id": t["team_id"].astype(int),
            "yards": num("totalYards"), "pass_yards": num("netPassingYards"), "rush_yards": num("rushingYards"),
            "rush_att": num("rushingAttempts"), "pass_att": split("completionAttempts", 1),
            "third_conv": split("thirdDownEff", 0), "third_att": split("thirdDownEff", 1),
            "fourth_conv": split("fourthDownEff", 0), "fourth_att": split("fourthDownEff", 1),
            "first_downs": num("firstDowns"), "turnovers": num("turnovers"), "pen_yards": split("totalPenaltiesYards", 1),
        })
        out.to_csv(path, index=False)
    return pd.read_csv(path)


STAT_ROWS = [
    ("ppg", "Points per game", lambda s: s["pts"] / s["g"], "{:.1f}", True),
    ("ypg", "Yards per game", lambda s: s["yards"] / s["g"], "{:.0f}", True),
    ("pass_ypg", "Pass yards per game", lambda s: s["pass_yards"] / s["g"], "{:.0f}", True),
    ("rush_ypg", "Rush yards per game", lambda s: s["rush_yards"] / s["g"], "{:.0f}", True),
    ("ypp", "Yards per play", lambda s: s["yards"] / (s["pass_att"] + s["rush_att"]), "{:.2f}", True),
    ("epa_play", "EPA per play", lambda s: s["epa"] / s["plays"], "{:+.3f}", True),
    ("epa_pass", "EPA per pass", lambda s: s["pass_epa"] / s["n_pass"], "{:+.3f}", True),
    ("epa_rush", "EPA per rush", lambda s: s["rush_epa"] / s["n_rush"], "{:+.3f}", True),
    ("success", "Success rate", lambda s: 100 * s["succ"] / s["plays"], "{:.1f}%", True),
    ("third", "Third down", lambda s: 100 * s["third_conv"] / s["third_att"], "{:.1f}%", True),
    ("fourth", "Fourth down", lambda s: 100 * s["fourth_conv"] / s["fourth_att"], "{:.1f}%", True),
    ("first", "First downs per game", lambda s: s["first_downs"] / s["g"], "{:.1f}", True),
    ("to", "Turnovers per game", lambda s: s["turnovers"] / s["g"], "{:.1f}", False),
    ("pen", "Penalty yards per game", lambda s: s["pen_yards"] / s["g"], "{:.0f}", False),
    ("snaps", "Plays per game", lambda s: s["snaps"] / s["g"], "{:.1f}", None),
]


def season_stats(games: pd.DataFrame, tg: pd.DataFrame, season: int, refresh: bool = False) -> dict | None:
    """Season-to-date offense and defense tables with FBS ranks (last season's in the first weeks)."""
    use = season
    cur = games[(games.season == season) & games.played & (games.fbs_game == 1)]
    if pd.concat([cur.home_team, cur.away_team]).value_counts().min() < 2 if len(cur) else True:
        use = season - 1
    try:
        box = load_team_box(use, refresh=refresh and use == season)
    except Exception as exc:
        print(f"  no team box scores for {use} ({exc})")
        return None
    g = games[(games.season == use) & games.played]
    idmap = pd.concat([g[["game_id", "home_id", "home_team", "away_team", "home_score", "away_score"]].assign(team_id=g.home_id, team=g.home_team, opp=g.away_team, pf=g.home_score, pa=g.away_score),
                       g[["game_id", "away_id", "home_team", "away_team", "home_score", "away_score"]].assign(team_id=g.away_id, team=g.away_team, opp=g.home_team, pf=g.away_score, pa=g.home_score)])
    idmap = idmap[["game_id", "team_id", "team", "opp", "pf", "pa"]]
    b = box.merge(idmap, on=["game_id", "team_id"])
    e = tg[tg.season == use][["game_id", "team", "plays", "epa", "succ", "n_pass", "n_rush", "pass_epa", "rush_epa", "snaps"]]
    b = b.merge(e, on=["game_id", "team"], how="left")
    num = ["yards", "pass_yards", "rush_yards", "rush_att", "pass_att", "third_conv", "third_att", "fourth_conv", "fourth_att",
           "first_downs", "turnovers", "pen_yards", "plays", "epa", "succ", "n_pass", "n_rush", "pass_epa", "rush_epa", "snaps"]
    b = b[b.team != C.POOL]
    tables = {}
    for unit, key, pts in (("off", "team", "pf"), ("def", "opp", "pa")):
        x = b[b[key] != C.POOL]
        s = x.groupby(key)[num].sum()
        s["g"] = x.groupby(key).size()
        s["pts"] = x.groupby(key)[pts].sum()
        out = pd.DataFrame(index=s.index)
        for k, _, fn, _, off_good in STAT_ROWS:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[k] = fn(s).replace([np.inf, -np.inf], np.nan)
            if off_good is None:
                out[k + "_rank"] = out[k].rank(ascending=False, method="min")
            else:
                better_high = off_good if unit == "off" else not off_good
                out[k + "_rank"] = out[k].rank(ascending=not better_high, method="min")
        tables[unit] = out
    tables["season"] = use
    tables["n"] = len(tables["off"])
    return tables


# --------------------------------------------------------------------------
# One game's page
# --------------------------------------------------------------------------

class MatchupBuilder:
    def __init__(self, games, qb_games, stat_tables, engine, info: dict, season: int, abbr: dict):
        self.games, self.engine, self.info, self.season, self.abbr = games, engine, info, season, abbr
        self.mc = dict(zip(info["margin_feats"], info["margin_c"]))
        self.tc = dict(zip(info["total_feats"], info["total_c"]))
        self._ratings = {}
        self.played = games[games.played].sort_values(["kickoff", "game_id"])
        self.qb_games = qb_games.sort_values("order")
        self.stat_tables = stat_tables
        self.opp = {}
        for g in games.itertuples(index=False):
            ha, aa = abbr.get(g.home_team, g.home_team), abbr.get(g.away_team, g.away_team)
            self.opp[(g.game_id, g.home_team)] = "vs " + aa
            self.opp[(g.game_id, g.away_team)] = "at " + ha
        self.week_of = dict(zip(games.game_id, zip(games.season, games.week)))

    def ratings(self, order):
        if order not in self._ratings:
            self._ratings[order] = self.engine.ratings_at(order)
        return self._ratings[order]

    def team_rating(self, r, i):
        v = sum(self.mc.get(f"d_{m}", 0.0) * (r[m]["off"] - r[m]["def"]) for m in self.engine.metrics) + self.mc["d_mkt"] * r["mkt"]["R"]
        return float(v[i] - v.mean())

    def total_rating(self, r, i):
        v = sum(self.tc.get(f"s_{m}", 0.0) * (r[m]["off"] + r[m]["def"]) for m in self.engine.metrics) + self.tc["s_mkt"] * r["mkt"]["T"]
        return float(v[i])

    @staticmethod
    def uncovered(margin_feats, total_feats, metrics) -> list[str]:
        """Model features the spread and totals card has no row for (should be empty;
        tests/test_cfb.py checks). The totals baseline is what is left over, so a total
        feature missing here would be hidden inside it."""
        m_ok = {f"d_{m}" for m in metrics} | {"d_mkt", "d_talent", "d_talent_early", "d_rtprod", "hfa", "rest_diff", "d_qb", "d_qb_early"}
        t_ok = {f"s_{m}" for m in metrics} | {"s_mkt", "lg_pts", "s_qb", "s_def_rtprod", "indoor", "wind_hi"}
        return [k for k in margin_feats if k not in m_ok] + [k for k in total_feats if k not in t_ok]

    def breakdown(self, g) -> dict:
        mc, tc = self.mc, self.tc
        r = self.ratings(int(g.order))
        h, a = self.engine.tix[g.home_team], self.engine.tix[g.away_team]
        val = lambda k: float(getattr(g, k, 0.0) or 0.0)
        rat = sum(mc.get(f"d_{m}", 0.0) * val(f"d_{m}") for m in self.engine.metrics) + mc["d_mkt"] * val("d_mkt")
        pre = sum(mc.get(k, 0.0) * val(k) for k in ("d_talent", "d_talent_early", "d_rtprod"))
        early = val("early")
        qcoef = mc.get("d_qb", 0.0) + mc.get("d_qb_early", 0.0) * early
        ha, aa = self.abbr.get(g.home_team, g.home_team), self.abbr.get(g.away_team, g.away_team)
        margin = [
            {"key": "ratings", "label": "Team ratings", "value": _f(rat), "home_rating": _f(self.team_rating(r, h)), "away_rating": _f(self.team_rating(r, a))},
            {"key": "pre", "label": "Talent and returning production", "value": _f(pre)},
            {"key": "hfa", "label": "Home field" if not g.neutral else "Home field (neutral site)", "value": _f(mc["hfa"] * val("hfa"))},
            {"key": "rest", "label": "Rest", "value": _f(mc["rest_diff"] * val("rest_diff"))},
            {"key": "home_qb", "label": f"{ha} quarterback", "value": _f(qcoef * val("home_qb_delta"))},
            {"key": "away_qb", "label": f"{aa} quarterback", "value": _f(-qcoef * val("away_qb_delta"))},
        ]
        th, ta = self.total_rating(r, h), self.total_rating(r, a)
        total = [
            {"key": "base", "label": "League baseline", "value": None},
            {"key": "home", "label": f"{ha} offense and defense", "value": _f(th)},
            {"key": "away", "label": f"{aa} offense and defense", "value": _f(ta)},
            {"key": "qb", "label": "Quarterbacks", "value": _f(tc["s_qb"] * val("s_qb"))},
            {"key": "ret", "label": "Returning defensive production", "value": _f(tc.get("s_def_rtprod", 0.0) * val("s_def_rtprod"))},
            {"key": "roof", "label": "Indoors" if g.indoor else "Outdoors", "value": _f(tc["indoor"] * val("indoor"))},
        ]
        if "wind_hi" in tc:
            mph = "indoors" if g.indoor else f"{val('wind'):.0f} mph" + ("" if getattr(g, "wind_known", 0) else ", typical day assumed")
            total.append({"key": "wind", "label": f"Wind ({mph})", "value": _f(tc["wind_hi"] * val("wind_hi"))})
        # the baseline is what is left: league scoring, the market's average total and the intercept
        total[0]["value"] = _f(float(g.model_total) - sum(x["value"] or 0 for x in total[1:]))
        msum, tsum = sum(x["value"] or 0 for x in margin), sum(x["value"] or 0 for x in total)
        return {"margin": margin, "margin_sum": _f(msum), "total": total, "total_sum": _f(tsum),
                "weight_margin": _f(g.w_margin, 3), "weight_total": _f(g.w_total, 3)}

    def quarterback(self, g, side):
        r = self.ratings(int(g.order))
        qid, team = getattr(g, f"{side}_qb_id"), getattr(g, f"{side}_team")
        name = self.engine.q_name.get(qid, "Unknown") if isinstance(qid, str) else "Unknown"
        delta = float(getattr(g, f"{side}_qb_delta") or 0.0)
        early = float(getattr(g, "early", 0.0) or 0.0)
        qcoef = self.mc.get("d_qb", 0.0) + self.mc.get("d_qb_early", 0.0) * early
        log = []
        if isinstance(qid, str):
            rows = self.qb_games[(self.qb_games.qb_id == qid) & (self.qb_games.dropbacks >= 5) & (self.qb_games.order < g.order)].tail(8)
            for q in rows.itertuples(index=False):
                s, w = self.week_of.get(q.game_id, (0, 0))
                log.append({"when": f"W{w} '{str(s)[2:]}", "opp": self.opp.get((q.game_id, q.team), ""), "team": self.abbr.get(q.team, q.team),
                            "db": int(q.dropbacks), "epa": _f(q.epa / q.dropbacks, 3)})
        v = getattr(g, f"{side}_qb_val")
        return {"name": name, "team": self.abbr.get(team, team), "value": _f(v, 3), "team_mix": _f((v or 0) - delta, 3),
                "dropbacks": int(round(r["qb_n"].get(qid, 0))) if isinstance(qid, str) else 0,
                "points": _f(qcoef * delta), "log": log}

    def ranks(self, g):
        t = self.stat_tables
        if t is None or g.home_team not in t["off"].index or g.away_team not in t["off"].index:
            return None

        def side(o_team, d_team):
            rows = []
            for k, label, _, fmt, _ in STAT_ROWS:
                o, d = t["off"].loc[o_team], t["def"].loc[d_team]
                cell = lambda row: {"v": fmt.format(row[k]) if np.isfinite(row[k]) else "n/a",
                                    "rank": int(row[k + "_rank"]) if np.isfinite(row[k + "_rank"]) else None}
                rows.append({"label": label, "off": cell(o), "def": cell(d)})
            return rows
        return {"season": int(t["season"]), "teams": int(t["n"]), "away_off": side(g.away_team, g.home_team), "home_off": side(g.home_team, g.away_team)}

    def trends(self, team, before):
        p = self.played
        mine = p[((p.home_team == team) | (p.away_team == team)) & (p.kickoff < before)].tail(10)
        rows, su, ats, ou = [], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        for x in mine.itertuples(index=False):
            home = x.home_team == team
            pf, pa = (x.home_score, x.away_score) if home else (x.away_score, x.home_score)
            margin = pf - pa
            sl = float(x.spread_line) if pd.notna(x.spread_line) else np.nan
            line = B.half(-sl if home else sl) if np.isfinite(sl) else np.nan
            cover = margin + line if np.isfinite(line) else np.nan
            pts = x.home_score + x.away_score
            tl = float(B.half(x.total_line)) if pd.notna(x.total_line) else np.nan
            res = lambda v: "W" if v > 0 else "L" if v < 0 else "P"
            other = x.away_team if home else x.home_team
            row = {"when": f"W{int(x.week)} '{str(int(x.season))[2:]}", "opp": ("vs " if home else "at ") + self.abbr.get(other, other),
                   "pf": int(pf), "pa": int(pa), "margin": int(margin), "su": "W" if margin > 0 else "L" if margin < 0 else "T",
                   "line": _f(line, 1), "ats": res(cover) if np.isfinite(cover) else None, "total_line": _f(tl, 1), "points": int(pts),
                   "ou": ("O" if pts > tl else "U" if pts < tl else "P") if np.isfinite(tl) else None}
            rows.append(row)
            su[0 if margin > 0 else 1 if margin < 0 else 2] += 1
            if row["ats"]:
                ats["WLP".index(row["ats"])] += 1
            if row["ou"]:
                ou["OUP".index(row["ou"])] += 1
        rec = lambda v: f"{v[0]}-{v[1]}" + (f"-{v[2]}" if v[2] else "")
        return {"team": self.abbr.get(team, team), "su": rec(su), "ats": rec(ats), "ou": rec(ou), "games": rows[::-1]}

    def detail(self, g):
        return {"stadium": g.venue if isinstance(g.venue, str) else None, "surface": None,
                "temp": _f(g.temp, 0) if g.played else None, "wind": None,
                "rest": {"away": _f(g.away_rest, 0), "home": _f(g.home_rest, 0)},
                "breakdown": self.breakdown(g),
                "qbs": {"away": self.quarterback(g, "away"), "home": self.quarterback(g, "home")},
                "ranks": self.ranks(g),
                "trends": {"away": self.trends(g.away_team, g.kickoff), "home": self.trends(g.home_team, g.kickoff)}}


# --------------------------------------------------------------------------
# Slate
# --------------------------------------------------------------------------

HISTORY_COLS = ["game_id", "seen", "spread_line", "total_line", "home_moneyline", "away_moneyline"]


def track_lines(up: pd.DataFrame, path=None) -> dict:
    path = path or C.DATA / "line_history.csv"
    hist = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=HISTORY_COLS)
    last = hist.groupby("game_id").tail(1).set_index("game_id") if len(hist) else hist.set_index("game_id")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    new = []
    for g in up.itertuples(index=False):
        cur = [float(getattr(g, c)) if c in up and pd.notna(getattr(g, c)) else np.nan for c in HISTORY_COLS[2:]]
        if g.game_id in last.index:
            old = last.loc[g.game_id, HISTORY_COLS[2:]].astype(float).to_numpy()
            if np.allclose(old, cur, equal_nan=True):
                continue
        new.append([g.game_id, now] + cur)
    if new:
        hist = pd.concat([hist, pd.DataFrame(new, columns=HISTORY_COLS)], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(path, index=False)
    first = hist.groupby("game_id").head(1).set_index("game_id")
    moves = hist.groupby("game_id").size()
    return {gid: {"seen": str(r.seen)[:10], "spread": float(r.spread_line), "total": float(r.total_line), "moves": int(moves[gid]) - 1}
            for gid, r in first.iterrows() if np.isfinite(r.spread_line) and np.isfinite(r.total_line)}


def fair_american(p):
    if not 0 < p < 1:
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def win_probs(df: pd.DataFrame, dist: dict) -> pd.DataFrame:
    """Home win chance for the market, the raw model and the adjusted projection. Without a
    moneyline price, the market's chance comes from its spread on the win-probability curve."""
    sp = df["spread_line"].to_numpy(float)
    fair, _ = O.no_vig(df["home_moneyline"].to_numpy(float) if "home_moneyline" in df else np.full(len(df), np.nan),
                       df["away_moneyline"].to_numpy(float) if "away_moneyline" in df else np.full(len(df), np.nan))
    implied = np.where(np.isnan(fair), sp, dist["sd_win"] * ndtri(np.clip(fair, 1e-4, 1 - 1e-4)))
    out = pd.DataFrame({"game_id": df["game_id"].to_numpy()})
    out["market"] = ndtr(implied / dist["sd_win"])
    out["raw"] = ndtr((implied + df["model_margin"].to_numpy(float) - sp) / dist["sd_win"])
    out["adj"] = ndtr((implied + df["proj_margin"].to_numpy(float) - sp) / dist["sd_win"])
    return out


def build_slate(frame, dist, engine, builder, records, abbr, names, openers=None, quotes=None, haircut: float = 1.0) -> dict:
    """quotes: nflmodel/market.quotes_for(..., "cfb", names=display name -> school) or None.
    haircut: the share of claimed edge the stakes are sized from (nflmodel.backtest.edge_haircut; 1 = none)."""
    quotes = quotes or {"latest": {}, "first": {}}
    cand = frame[~frame.played].copy()
    sharp = {gid for gid, q in quotes["latest"].items() if MK.SHARP in q and "spread_line" in q[MK.SHARP] and "total_line" in q[MK.SHARP]}
    up = cand[(cand.spread_line.notna() & cand.total_line.notna()) | cand.game_id.isin(sharp)].copy()
    if up.empty:
        return {"weeks": [], "games": [], "book_lines": 0}
    opened = track_lines(up[up.spread_line.notna() & up.total_line.notna()])
    up = with_market(up, quotes["latest"])
    games = entries(up, dist, engine, builder, lambda g: records, abbr, names, opened, quotes, haircut=haircut)
    return {"weeks": sorted({x["week"] for x in games}), "games": games, "book_lines": 0}


def entries(up, dist, engine, builder, records_for, abbr, names, opened, quotes=None, haircut: float = 1.0) -> list[dict]:
    """The board's entry for each game in `up` (records_for(game) gives the records to show)."""
    quotes = quotes or {"latest": {}, "first": {}}
    coded = up.assign(home_team=up.home_team.map(lambda t: abbr.get(t, t)), away_team=up.away_team.map(lambda t: abbr.get(t, t)))
    adj = B.price_games(coded, dist, "proj", haircut=haircut)
    raw = B.price_games(coded, dist, "model")
    shopped = B.shop(coded, dist, quotes["latest"], MK.RETAIL, "proj", haircut=haircut)
    key = ["game_id", "market", "side"]
    b = adj.merge(raw[key + ["model_prob", "ev"]].rename(columns={"model_prob": "raw_prob", "ev": "raw_ev"}), on=key)
    wins = win_probs(up, dist).set_index("game_id")
    games = []
    for g in up.sort_values(["week", "kickoff", "game_id"]).itertuples(index=False):
        gb = b[b.game_id == g.game_id]
        bets = [{"market": r.market, "side": r.side, "pick": r.pick,
                 "odds": int(r.odds), "prob": round(r.model_prob, 4), "raw_prob": round(r.raw_prob, 4), "market_prob": round(r.market_prob, 4),
                 "breakeven": round(r.breakeven, 4), "edge": round(r.edge, 4), "ev": round(r.ev, 4), "raw_ev": round(r.raw_ev, 4),
                 "fair": fair_american(r.model_prob), "stake": round(r.stake, 4),
                 **bet_books(shopped, g.game_id, r.market, r.side)} for r in gb.itertuples(index=False)]
        capped = cap_game(bets, C.MAX_GAME_STAKE)
        ha, aa = abbr.get(g.home_team, g.home_team), abbr.get(g.away_team, g.away_team)
        when = pd.Timestamp(g.kickoff).tz_convert("America/New_York")
        qb = lambda side: {"name": engine.q_name.get(getattr(g, f"{side}_qb_id"), "Unknown") if isinstance(getattr(g, f"{side}_qb_id"), str) else "Unknown",
                           "value": _f(getattr(g, f"{side}_qb_val"), 3), "delta": _f(getattr(g, f"{side}_qb_delta"), 3)}
        w = wins.loc[g.game_id]
        src = getattr(g, "wind_source", "")
        games.append({
            "id": game_key(g.season, g.week, aa, ha), "game_id": int(g.game_id), "week": int(g.week),
            "date": when.strftime("%a %b %-d"), "time": when.strftime("%H:%M"),
            "away": aa, "home": ha, "neutral": bool(g.neutral),
            "away_name": names.get(g.away_team, g.away_team), "home_name": names.get(g.home_team, g.home_team),
            "away_record": records_for(g).get(g.away_team, "0-0"), "home_record": records_for(g).get(g.home_team, "0-0"),
            "early": bool(getattr(g, "early_line", False)), "weight": round(float(g.w_margin), 3),
            "open": opened.get(g.game_id),
            "opening_spread": _f(g.open_spread, 1) if pd.notna(g.open_spread) else None,
            "opening_total": _f(g.open_total, 1) if pd.notna(g.open_total) else None,
            "line_source": getattr(g, "line_source", None) if isinstance(getattr(g, "line_source", None), str) else "cfbfastR consensus",
            "anchor": getattr(g, "anchor", "consensus"),
            "sharp_open": quotes["first"].get(g.game_id),
            "books": {bk: {k: v for k, v in q.items() if k != "away_spread_point"} for bk, q in (quotes["latest"].get(g.game_id) or {}).items()},
            "detail": builder.detail(g) if builder is not None else None,
            "roof": "dome" if g.indoor else "outdoors",
            "wind": ({"mph": round(float(g.wind), 1), "source": src, "under_angle": bool(WIND_LO <= g.wind < WIND_HI)}
                     if src in ("forecast", "recorded") and getattr(g, "wind_known", 0) else None),
            "away_qb": qb("away"), "home_qb": qb("home"),
            # the line as a book posts it (the data's consensus is an average across books)
            "market": {"spread": float(B.half(g.spread_line)), "total": float(B.half(g.total_line)),
                       "consensus_spread": _f(g.spread_line, 2), "consensus_total": _f(g.total_line, 2)},
            "model": {"margin": round(float(g.model_margin), 2), "total": round(float(g.model_total), 2)},
            "proj": {"margin": round(float(g.proj_margin), 2), "total": round(float(g.proj_total), 2)},
            "win": {"adj": round(float(w["adj"]), 4), "raw": round(float(w["raw"]), 4), "market": round(float(w["market"]), 4)},
            "bets": bets, **capped,
        })
    return games


def season_records(games, season, before=None):
    rec = {}
    sel = games[(games.season == season) & games.played]
    if before is not None:
        sel = sel[sel.kickoff < before]
    for g in sel.itertuples(index=False):
        for team, diff in ((g.home_team, g.result), (g.away_team, -g.result)):
            w, l, d = rec.get(team, (0, 0, 0))
            rec[team] = (w + (diff > 0), l + (diff < 0), d + (diff == 0))
    return {t: "-".join(str(int(v)) for v in (r if r[2] else r[:2])) for t, r in rec.items()}


def ratings_table(engine, r, info, games, frame, season, teams) -> list[dict]:
    """Points per game against an average FBS team on a neutral field, with this week's quarterback."""
    mc = dict(zip(info["margin_feats"], info["margin_c"]))
    t = engine.table(r)
    t = t[t.team.isin(teams.index)].copy()
    ps = 0.5
    t["off_epa"] = ps * t.pass_off + (1 - ps) * t.rush_off
    t["def_epa"] = ps * t.pass_def + (1 - ps) * t.rush_def
    t["stats"] = sum(mc.get(f"d_{m}", 0.0) * (t[f"{m}_off"] - t[f"{m}_def"]) for m in engine.metrics)
    t["market"] = mc["d_mkt"] * t["mkt"]
    up = frame[~frame.played].sort_values("order")
    nxt = {}
    for g in up.itertuples(index=False):
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            if team not in nxt:
                qid = getattr(g, f"{side}_qb_id")
                early = float(getattr(g, "early", 0.0) or 0.0)
                nxt[team] = (engine.q_name.get(qid, "Unknown") if isinstance(qid, str) else None, float(getattr(g, f"{side}_qb_val") or 0),
                             float(getattr(g, f"{side}_qb_delta") or 0) * (mc.get("d_qb", 0) + mc.get("d_qb_early", 0) * early))
    t["qb_name"] = [nxt.get(x, (q, None, 0.0))[0] for x, q in zip(t.team, t.qb)]
    t["qb_val"] = [nxt.get(x, (None, np.nan, 0.0))[1] for x in t.team]
    t["qb_adj"] = [nxt.get(x, (None, None, 0.0))[2] for x in t.team]
    t["rating"] = t["stats"] + t["market"] + t["qb_adj"]
    t["rating"] -= t["rating"].mean()
    rec = season_records(games, season)
    t = t.sort_values("rating", ascending=False).reset_index(drop=True)
    return [{"rank": i + 1, "team": teams.loc[x.team, "abbr"], "school": x.team, "name": teams.loc[x.team, "name"],
             "conference": teams.loc[x.team, "conference"] if isinstance(teams.loc[x.team, "conference"], str) else "",
             "record": rec.get(x.team, "0-0"), "rating": round(x.rating, 2), "market_rating": round(x.mkt, 2),
             "off_epa": round(x.off_epa, 3), "def_epa": round(x.def_epa, 3),
             "pass_off": round(x.pass_off, 3), "rush_off": round(x.rush_off, 3), "pass_def": round(x.pass_def, 3), "rush_def": round(x.rush_def, 3),
             "pts_off": round(x.pts_off, 2), "pts_def": round(x.pts_def, 2),
             "qb": x.qb_name, "qb_val": None if pd.isna(x.qb_val) else round(x.qb_val, 3)} for i, x in enumerate(t.itertuples(index=False))]
