"""Player props: calibrated probabilities from the player projections.

players.py gives each player's expected passing, rushing and receiving line. This
module turns those means into P(over), P(push) and P(under) at any line, using the
distributions in propdist.py, with widths fitted on past misses and a monotone
correction fitted on the tuning seasons only.

The fitted parameters live in data/prop_params.json, written by
experiments/props_calibration.py (which also writes the calibration report); the
board reads them, so the weekly run never refits on its own.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config as C
from .propdist import CountFit, Isotonic, YardsFit, compound_cdf, fit_count, fit_yards, settle_probs

PARAMS_PATH = C.DATA / "prop_params.json"


@dataclass(frozen=True)
class Spec:
    label: str
    kind: str                 # "count" or "yards"
    actual: tuple             # columns summed for the outcome
    proj: tuple               # projection columns summed for the mean
    groups: tuple
    role: str | None          # usage share that sets the size of the role
    kmax: int                 # largest count tracked (counts; volume for yards)
    vol: str | None = None    # yards: the count prop that supplies the volume
    step: float = 1.0         # spacing of the synthetic test lines
    fixed_lines: tuple = ()   # standard lines, instead of lines around the projection
    min_proj: float = 0.0     # smallest projection a book would post a line for
    odds_key: str | None = None   # The Odds API market key


SPECS: dict[str, Spec] = {
    "receptions": Spec("Receptions", "count", ("receptions",), ("receptions_proj",), ("WR", "TE", "RB"), "tgt_share", 25,
                       step=1, min_proj=1.0, odds_key="player_receptions"),
    "rec_yds": Spec("Receiving yards", "yards", ("rec_yds",), ("rec_yds_proj",), ("WR", "TE", "RB"), "tgt_share", 25,
                    vol="receptions", step=10, min_proj=10.0, odds_key="player_reception_yds"),
    "carries": Spec("Carries", "count", ("carries",), ("carries_proj",), ("RB", "QB"), "car_share", 45,
                    step=2, min_proj=3.0, odds_key="player_rush_attempts"),
    "rush_yds": Spec("Rushing yards", "yards", ("rush_yds",), ("rush_yds_proj",), ("RB", "QB"), "car_share", 45,
                     vol="carries", step=10, min_proj=10.0, odds_key="player_rush_yds"),
    "pass_att": Spec("Pass attempts", "count", ("pass_att",), ("pass_att_proj",), ("QB",), None, 75,
                     step=2, min_proj=15.0, odds_key="player_pass_attempts"),
    "completions": Spec("Completions", "count", ("completions",), ("completions_proj",), ("QB",), None, 60,
                        step=2, min_proj=10.0, odds_key="player_pass_completions"),
    "pass_yds": Spec("Passing yards", "yards", ("pass_yds",), ("pass_yds_proj",), ("QB",), None, 60,
                     vol="completions", step=15, min_proj=100.0, odds_key="player_pass_yds"),
    "pass_td": Spec("Passing TDs", "count", ("pass_td",), ("pass_td_proj",), ("QB",), None, 10,
                    fixed_lines=(0.5, 1.5, 2.5), min_proj=0.5, odds_key="player_pass_tds"),
    "ints": Spec("Interceptions", "count", ("ints",), ("ints_proj",), ("QB",), None, 8,
                 fixed_lines=(0.5, 1.5), min_proj=0.3, odds_key="player_pass_interceptions"),
    "anytime_td": Spec("Anytime TD", "count", ("rec_td", "rush_td"), ("rec_td_proj", "rush_td_proj"), ("WR", "TE", "RB", "QB"),
                       "tgt_share", 8, fixed_lines=(0.5,), min_proj=0.05, odds_key="player_anytime_td"),
}
QB_STATS = {"pass_att", "completions", "pass_yds", "pass_td", "ints"}


# --------------------------------------------------------------------------
# Rows, outcomes and means
# --------------------------------------------------------------------------

def eligible(df: pd.DataFrame, stat: str, group: str, played: bool = True) -> pd.Series:
    """Rows a book would post this prop for, using only what is known before the game,
    plus the settlement rule: a prop is void if the player does not play. Quarterbacks:
    the projected starter, in games he started (official starter where listed)."""
    s = SPECS[stat]
    m = (df["group"] == group) & (df["active"] == 1) & (mean(df, stat) >= s.min_proj)
    if played:
        m &= df["played"].astype(bool)
    if group == "QB":
        m &= df["att_share"] > 0.5
        if played:
            m &= df["started"] == 1
    return m


def actual(df: pd.DataFrame, stat: str) -> np.ndarray:
    return df[list(SPECS[stat].actual)].sum(axis=1).to_numpy(float)


def mean(df: pd.DataFrame, stat: str) -> np.ndarray:
    return df[list(SPECS[stat].proj)].fillna(0).sum(axis=1).to_numpy(float)


def role(df: pd.DataFrame, stat: str, group: str) -> np.ndarray:
    s = SPECS[stat]
    if group == "QB" or s.role is None:
        return np.zeros(len(df))
    r = df[s.role] if s.role in df else pd.Series(0.0, index=df.index)
    if stat == "anytime_td":     # backs score on carries, receivers on targets
        r = df["car_share"] if group == "RB" else df["tgt_share"]
    return r.fillna(0).to_numpy(float)


def test_lines(mu: np.ndarray, stat: str) -> np.ndarray:
    """Synthetic half-point lines around each projection: the line nearest the mean and
    one and two steps either side (or the standard lines, for touchdowns and picks).
    Returns (rows, m), NaN where a line would fall below 0.5."""
    s = SPECS[stat]
    if s.fixed_lines:
        return np.tile(np.array(s.fixed_lines, float), (len(mu), 1))
    base = np.floor(mu) + 0.5
    if s.step >= 5:              # yards: round the centre to a multiple of 5, as books do
        base = np.floor(mu / 5) * 5 + 0.5
    L = base[:, None] + s.step * np.array([-2, -1, 0, 1, 2])[None, :]
    return np.where(L >= 0.5, L, np.nan)


# A ladder of lines from far below the projection to well above it, for fitting the correction
# where the books post alternate ladders (experiments/props_tail.py).
LADDER_FRACS = (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7)


def ladder_lines(mu: np.ndarray, stat: str) -> np.ndarray:
    """Half-point lines at LADDER_FRACS of each projection (yards on fives, as books post
    them); NaN below 0.5. Fixed-line stats keep their standard lines."""
    s = SPECS[stat]
    if s.fixed_lines:
        return np.tile(np.array(s.fixed_lines, float), (len(mu), 1))
    x = np.asarray(mu, float)[:, None] * np.array(LADDER_FRACS)[None, :]
    L = (np.floor(x / 5) * 5 if s.step >= 5 else np.floor(x)) + 0.5
    return np.where(L >= 0.5, L, np.nan)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

class PropModel:
    """Distribution parameters per (stat, position group) and a correction per stat."""

    def __init__(self, family: dict | None = None):
        self.family = family or {}          # stat -> "nb" | "poisson"
        self.counts: dict[tuple, CountFit] = {}
        self.yards: dict[tuple, YardsFit] = {}
        self.iso: dict[str, Isotonic] = {}
        self.meta: dict = {}

    # ---- fitting ---------------------------------------------------------
    def fit(self, df: pd.DataFrame, stats=None, fit_sample: int = 20000, seed: int = 0) -> "PropModel":
        rng = np.random.default_rng(seed)
        stats = stats or list(SPECS)
        order = [s for s in stats if SPECS[s].kind == "count"] + [s for s in stats if SPECS[s].kind == "yards"]
        for stat in order:
            s = SPECS[stat]
            for g in s.groups:
                d = df[eligible(df, stat, g)]
                if len(d) < 200:
                    continue
                if len(d) > fit_sample:
                    d = d.iloc[rng.choice(len(d), fit_sample, replace=False)]
                if s.kind == "count":
                    self.counts[(stat, g)] = fit_count(actual(d, stat), mean(d, stat), role(d, stat, g),
                                                       family=self.family.get(stat, "nb"))
                else:
                    vol = self._vol_pmf(d, stat, g)
                    if vol is None:
                        continue
                    mu_v = np.maximum(mean(d, s.vol), 0.05)
                    sub = d.index[:min(len(d), 8000)]          # the compound likelihood is slow; 8k rows is plenty
                    k = len(sub)
                    self.yards[(stat, g)] = fit_yards(actual(d, stat)[:k], vol[:k], (mean(d, stat) / mu_v)[:k],
                                                      mu_v[:k], role(d, stat, g)[:k])
        return self

    def fit_isotonic(self, preds: pd.DataFrame) -> "PropModel":
        """preds: stat, p_over (raw), hit, on half-point lines; tuning seasons only."""
        for stat, d in preds.groupby("stat"):
            if len(d) >= 500:
                self.iso[stat] = Isotonic.fit(d["p_over"], d["hit"])
        return self

    # ---- probabilities ----------------------------------------------------
    def _vol_pmf(self, d, stat, g):
        s = SPECS[stat]
        vf = self.counts.get((s.vol, g))
        if vf is None:
            return None
        return vf.pmf(np.maximum(mean(d, s.vol), 0.05), role(d, s.vol, g), SPECS[s.vol].kmax)

    def cdf_fn(self, d: pd.DataFrame, stat: str, g: str):
        """A function k -> P(X <= k) for these rows (k broadcast to (rows, m))."""
        s = SPECS[stat]
        mu = np.maximum(mean(d, stat), 0.01)
        if s.kind == "count":
            fit = self.counts[(stat, g)]
            cum = np.cumsum(fit.pmf(mu, role(d, stat, g), s.kmax), 1)

            def f(k):
                k = np.asarray(k, float).reshape(len(d), -1)
                idx = np.clip(k, -1, s.kmax).astype(int)
                return np.where(idx < 0, 0.0, np.take_along_axis(cum, np.clip(idx, 0, s.kmax), 1))
            return f
        vol = self._vol_pmf(d, stat, g)
        fit = self.yards[(stat, g)]
        mu_v = np.maximum(mean(d, s.vol), 0.05)
        ypp, k1 = mu / mu_v, fit.k1(mu_v, role(d, stat, g))

        def f(k):
            k = np.asarray(k, float).reshape(len(d), -1)
            return np.where(k < 0, 0.0, compound_cdf(vol, ypp, k1, k + 0.5))
        return f

    def probs(self, d: pd.DataFrame, stat: str, g: str, lines, calibrate: bool = True):
        """P(over), P(push), P(under) at each row's line(s); lines is (rows,) or (rows, m)."""
        lines = np.asarray(lines, float)
        shape = lines.shape
        L = lines.reshape(len(d), -1)
        f = self.cdf_fn(d, stat, g)
        Lf = np.where(np.isnan(L), 0.5, L)
        over, push, under = settle_probs(f, Lf)
        iso = self.iso.get(f"{stat}|{g}") or self.iso.get(stat)     # per position group where fitted
        if calibrate and iso is not None:
            # the correction is fitted on half-point lines; keep the push share and rescale the rest
            p_dec = over / np.maximum(over + under, 1e-9)
            p_dec = iso(p_dec)
            rest = 1 - push
            over, under = p_dec * rest, (1 - p_dec) * rest
        nan = np.isnan(L)
        over, push, under = (np.where(nan, np.nan, x).reshape(shape) for x in (over, push, under))
        return over, push, under

    # ---- saving -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "meta": self.meta, "family": self.family,
            "counts": {f"{k[0]}|{k[1]}": asdict(v) for k, v in self.counts.items()},
            "yards": {f"{k[0]}|{k[1]}": asdict(v) for k, v in self.yards.items()},
            "isotonic": {k: v.to_dict() for k, v in self.iso.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PropModel":
        m = cls(d.get("family", {}))
        m.meta = d.get("meta", {})
        m.counts = {tuple(k.split("|")): CountFit(**v) for k, v in d.get("counts", {}).items()}
        m.yards = {tuple(k.split("|")): YardsFit(**v) for k, v in d.get("yards", {}).items()}
        m.iso = {k: Isotonic(np.array(v["x"]), np.array(v["y"])) for k, v in d.get("isotonic", {}).items()}
        return m

    def save(self, path=PARAMS_PATH):
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path=PARAMS_PATH) -> "PropModel | None":
        return cls.from_dict(json.loads(path.read_text())) if path.exists() else None


# --------------------------------------------------------------------------
# Baseline: the player's recent games
# --------------------------------------------------------------------------

def recent_history(df: pd.DataFrame, stat: str, n: int = 8) -> pd.DataFrame:
    """The player's last n results for this stat in games he played (for quarterback
    passing, games he started), before each game. Columns lag1..lagn, keyed by
    game_id and player_id."""
    ok = df["played"].astype(bool) & (df["active"] == 1)
    if stat in QB_STATS:
        ok &= df["started"] == 1
    allr = df[["player_id", "game_id", "t"]].sort_values(["player_id", "t"])
    ref = df.loc[ok, ["player_id", "t"]].assign(v=actual(df.loc[ok], stat)).sort_values(["player_id", "t"])
    vals = {pid: (grp["t"].to_numpy(), grp["v"].to_numpy()) for pid, grp in ref.groupby("player_id")}
    out = np.full((len(allr), n), np.nan)
    for i, (pid, t) in enumerate(zip(allr["player_id"].to_numpy(), allr["t"].to_numpy())):
        tv = vals.get(pid)
        if tv is None:
            continue
        j = np.searchsorted(tv[0], t)          # games strictly before this one
        prev = tv[1][max(0, j - n):j][::-1]
        out[i, :len(prev)] = prev
    res = allr[["player_id", "game_id"]].reset_index(drop=True)
    for i in range(n):
        res[f"lag{i + 1}"] = out[:, i]
    return res


def empirical_over(lags: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Share of his last games over each line, shrunk to 50% by one game."""
    lags = np.asarray(lags, float)
    k = (~np.isnan(lags)).sum(axis=1)
    L = np.asarray(lines, float).reshape(len(lags), -1)
    over = (lags[:, None, :] > L[:, :, None]).sum(2)
    p = (over + 0.5) / (k[:, None] + 1.0)
    return np.where(k[:, None] >= 3, p, np.nan)
