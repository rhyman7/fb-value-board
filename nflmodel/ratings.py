"""Opponent-adjusted team ratings and quarterback ratings.

Everything here is "as of" a point in time: the ratings for a given week use
only games played in earlier weeks, so the same code serves the backtest and
the live slate without leaking results.

Team ratings
    For each metric (pass EPA per dropback, rush EPA per carry, success rate,
    points) a ridge regression explains what every offense did in every game as
        league average + home edge + offense rating + defense rating
    Recent games count more (exponential decay), last season counts less again,
    and the ridge penalty pulls small samples toward league average. Solving for
    all teams at once is what makes the ratings opponent-adjusted.

Quarterback ratings
    EPA per dropback versus league average, decayed and shrunk toward a
    replacement-level prior. The feature the game model uses is the gap between
    this week's starter and the quarterbacks who produced the team's pass
    rating, so a backup starting shows up as a negative adjustment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SEASON_LEN = 22  # weeks, including playoffs
METRICS = ("pass", "rush", "sr", "pts")


def _time(order: np.ndarray | int):
    """Convert season*100+week into a running week count."""
    return (order // 100) * SEASON_LEN + (order % 100)


class RatingEngine:
    def __init__(self, team_games: pd.DataFrame, qb_games: pd.DataFrame, *,
                 half_life: float = C.HALF_LIFE_GAMES, carry: float = C.OFFSEASON_CARRY,
                 ridge: dict | None = None, window_seasons: int = 3,
                 games: pd.DataFrame | None = None, market_half_life: float = C.MARKET_HALF_LIFE):
        tg = team_games.sort_values("order").reset_index(drop=True)
        self.teams = sorted(set(tg["team"]) | set(tg["opp"]))
        self.tix = {t: i for i, t in enumerate(self.teams)}
        self.n = len(self.teams)
        self.half_life, self.carry, self.window = half_life, carry, window_seasons
        self.ridge = dict(C.RIDGE if ridge is None else ridge)

        self.order = tg["order"].to_numpy()
        self.season = self.order // 100
        self.off = tg["team"].map(self.tix).to_numpy()
        self.dfn = tg["opp"].map(self.tix).to_numpy()
        self.home = tg["is_home"].to_numpy(float)
        self.away = ((tg["side"] == "away") & (tg["neutral"] == 0)).to_numpy(float)
        self.game_id = tg["game_id"].to_numpy()
        n_pass, n_rush, plays = (tg[c].to_numpy(float) for c in ("n_pass", "n_rush", "plays"))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.y = {
                "pass": np.nan_to_num(tg["pass_epa"].to_numpy() / n_pass),
                "rush": np.nan_to_num(tg["rush_epa"].to_numpy() / n_rush),
                "sr": np.nan_to_num(tg["succ"].to_numpy() / plays),
                "pts": tg["pts"].to_numpy(float),
            }
        self.n_obs = {"pass": n_pass, "rush": n_rush, "sr": plays, "pts": np.ones(len(tg))}
        self.n_pass = n_pass

        q = qb_games.sort_values("order").reset_index(drop=True)
        self.q_order = q["order"].to_numpy()
        self.q_id = q["qb_id"].to_numpy()
        self.q_team = q["team"].map(self.tix).to_numpy()
        self.q_db = q["dropbacks"].to_numpy(float)
        self.q_epa = q["epa"].to_numpy(float)
        self.q_name = dict(zip(q["qb_id"], q["name"]))
        # Past betting lines, for market-implied ratings (what the line-makers have
        # been saying about each team, before this week's line is posted).
        self.mkt_half_life = market_half_life
        self.has_market = games is not None
        if self.has_market:
            m = games[games["spread_line"].notna() & games["total_line"].notna()
                      & games["home_team"].isin(self.tix) & games["away_team"].isin(self.tix)]
            self.m_order = m["order"].to_numpy()
            self.m_home = m["home_team"].map(self.tix).to_numpy()
            self.m_away = m["away_team"].map(self.tix).to_numpy()
            self.m_site = 1.0 - m["neutral"].to_numpy(float)
            self.m_spread = m["spread_line"].to_numpy(float)
            self.m_total = m["total_line"].to_numpy(float)

        # Each team's primary quarterback in each game, for "who started last week".
        top = q.sort_values("dropbacks").groupby(["game_id", "team"]).tail(1)
        self.last_qb = top.sort_values("order")[["order", "team", "qb_id"]]

    # ------------------------------------------------------------------
    def _weights(self, order_now: int, orders: np.ndarray) -> np.ndarray:
        age = _time(order_now) - _time(orders)
        seasons_back = order_now // 100 - orders // 100
        return 0.5 ** (age / self.half_life) * self.carry ** seasons_back

    def _solve(self, metric: str, idx: np.ndarray, decay: np.ndarray):
        """Weighted ridge: y = mu + hfa*(home - away)/2 + O[off] + D[def]."""
        n = self.n
        w = decay * self.n_obs[metric][idx]
        y = self.y[metric][idx]
        k = 2 * n + 2
        X = np.zeros((len(idx), k))
        rows = np.arange(len(idx))
        X[rows, self.off[idx]] = 1.0
        X[rows, n + self.dfn[idx]] = 1.0
        X[:, 2 * n] = 1.0
        X[:, 2 * n + 1] = (self.home[idx] - self.away[idx]) / 2.0
        Xw = X * w[:, None]
        A = X.T @ Xw
        pen = np.zeros(k)
        pen[: 2 * n] = self.ridge[metric]
        b = np.linalg.solve(A + np.diag(pen) + 1e-9 * np.eye(k), Xw.T @ y)
        return b[:n], b[n: 2 * n], b[2 * n], b[2 * n + 1]

    def ratings_at(self, order_now: int) -> dict:
        """Ratings using only games before `order_now` (season*100 + week)."""
        mask = (self.order < order_now) & (self.season >= order_now // 100 - self.window)
        idx = np.flatnonzero(mask)
        decay = self._weights(order_now, self.order[idx])
        out = {"order": order_now, "teams": self.teams}
        for m in METRICS:
            off, dfn, mu, hfa = self._solve(m, idx, decay)
            out[m] = {"off": off, "def": dfn, "mu": mu, "hfa": hfa}

        if self.has_market:
            out["mkt"] = self._market(order_now)

        # ---- quarterbacks -------------------------------------------------
        qmask = self.q_order < order_now
        qo, qid, qdb, qepa, qteam = (a[qmask] for a in (self.q_order, self.q_id, self.q_db, self.q_epa, self.q_team))
        lg = out["pass"]["mu"]
        # Decay by time (a long half-life: quarterbacks are more stable than teams).
        age = _time(order_now) - _time(qo)
        qw = 0.5 ** (age / (C.QB_HALF_LIFE_DROPBACKS / 35.0))
        df = pd.DataFrame({"id": qid, "w_db": qw * qdb, "w_epa": qw * (qepa - lg * qdb)})
        s = df.groupby("id").sum()
        P = C.QB_PRIOR_DROPBACKS
        qb_val = ((s["w_epa"] + P * C.QB_REPLACEMENT) / (s["w_db"] + P))
        out["qb"] = qb_val.to_dict()
        out["qb_n"] = s["w_db"].to_dict()

        # The quarterback mix behind each team's current pass rating.
        recent = (qo < order_now) & (qo // 100 >= order_now // 100 - self.window)
        tw = self._weights(order_now, qo[recent]) * qdb[recent]
        vals = pd.Series(qid[recent]).map(qb_val).to_numpy()
        num = np.bincount(qteam[recent], weights=tw * vals, minlength=self.n)
        den = np.bincount(qteam[recent], weights=tw, minlength=self.n)
        out["qb_base"] = np.divide(num, den, out=np.full(self.n, C.QB_REPLACEMENT), where=den > 0)

        lq = self.last_qb[self.last_qb["order"] < order_now].groupby("team").tail(1)
        out["last_qb"] = dict(zip(lq["team"], lq["qb_id"]))
        return out

    def _market(self, order_now: int) -> dict:
        """Team strength implied by earlier closing lines.

        spread = home edge + R[home] - R[away];  total = average + T[home] + T[away].
        A short half-life, because the market re-prices teams every week.
        """
        n = self.n
        idx = np.flatnonzero((self.m_order < order_now) & (self.m_order // 100 >= order_now // 100 - 1))
        if len(idx) < 16:
            return {"R": np.zeros(n), "T": np.zeros(n), "hfa": 2.0, "avg_total": 44.0}
        o = self.m_order[idx]
        w = 0.5 ** ((_time(order_now) - _time(o)) / self.mkt_half_life) * self.carry ** (order_now // 100 - o // 100)
        rows = np.arange(len(idx))
        X = np.zeros((len(idx), n + 1))
        X[rows, self.m_home[idx]] = 1.0
        X[rows, self.m_away[idx]] = -1.0
        X[:, n] = self.m_site[idx]
        Xw = X * w[:, None]
        pen = np.r_[np.full(n, 0.5), 0.0]
        b = np.linalg.solve(X.T @ Xw + np.diag(pen) + 1e-9 * np.eye(n + 1), Xw.T @ self.m_spread[idx])
        X2 = np.zeros((len(idx), n + 1))
        X2[rows, self.m_home[idx]] = 1.0
        X2[rows, self.m_away[idx]] += 1.0
        X2[:, n] = 1.0
        X2w = X2 * w[:, None]
        b2 = np.linalg.solve(X2.T @ X2w + np.diag(pen) + 1e-9 * np.eye(n + 1), X2w.T @ self.m_total[idx])
        return {"R": b[:n], "T": b2[:n], "hfa": b[n], "avg_total": b2[n]}

    def qb_value(self, r: dict, qb_id) -> float:
        if qb_id is None or (isinstance(qb_id, float) and np.isnan(qb_id)):
            return np.nan
        return r["qb"].get(qb_id, C.QB_REPLACEMENT)

    # ------------------------------------------------------------------
    def features(self, games: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """One row of pre-game features for every game in `games`."""
        rows = []
        for order_now, wk in games.groupby("order", sort=True):
            if not (self.order < order_now).any():
                continue
            r = self.ratings_at(int(order_now))
            if verbose and order_now % 100 == 1:
                print(f"  ratings through {order_now // 100 - 1}", flush=True)
            for g in wk.itertuples(index=False):
                if g.home_team not in self.tix or g.away_team not in self.tix:
                    continue
                h, a = self.tix[g.home_team], self.tix[g.away_team]
                f = {"game_id": g.game_id}
                for m in METRICS:
                    eh = r[m]["off"][h] + r[m]["def"][a]   # home offense vs away defense
                    ea = r[m]["off"][a] + r[m]["def"][h]
                    f[f"d_{m}"], f[f"s_{m}"] = eh - ea, eh + ea
                f["lg_pts"] = 2 * r["pts"]["mu"]
                if self.has_market:
                    k = r["mkt"]
                    f["d_mkt"] = k["R"][h] - k["R"][a]
                    f["s_mkt"] = k["avg_total"] + k["T"][h] + k["T"][a]
                for side, t, ti in (("home", g.home_team, h), ("away", g.away_team, a)):
                    qid = getattr(g, f"{side}_qb_id")
                    if not isinstance(qid, str):
                        qid = r["last_qb"].get(t)       # assume last week's starter
                    v = self.qb_value(r, qid)
                    f[f"{side}_qb_id"] = qid
                    f[f"{side}_qb_val"] = v
                    f[f"{side}_qb_delta"] = 0.0 if np.isnan(v) else v - r["qb_base"][ti]
                f["d_qb"] = f["home_qb_delta"] - f["away_qb_delta"]
                f["s_qb"] = f["home_qb_delta"] + f["away_qb_delta"]
                rows.append(f)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def table(self, r: dict, coefs: dict | None = None) -> pd.DataFrame:
        """A readable ratings table for one point in time."""
        t = pd.DataFrame({"team": self.teams})
        for m in METRICS:
            t[f"{m}_off"] = r[m]["off"]
            t[f"{m}_def"] = r[m]["def"]      # positive = allows more than average
        if "mkt" in r:
            t["mkt"], t["mkt_total"] = r["mkt"]["R"], r["mkt"]["T"]
        t["qb_base"] = r["qb_base"]
        t["qb_id"] = t["team"].map(r["last_qb"])
        t["qb"] = t["qb_id"].map(self.q_name)
        return t
