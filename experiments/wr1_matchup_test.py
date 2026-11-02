"""Shadow corners without charting: does a defense treat the opponent's top wide receiver differently?

    python experiments/wr1_matchup_test.py

Ryan's question (Nov 2, 2026): a receiver lined up against an elite corner (Patrick Surtain)
should underperform. The board rates each defense against wide receivers as a group
(nflmodel/dvp.py: yards allowed over what the projections expected, decayed, shrunk) but has no
cornerback-by-receiver matchups (coverage charting is not public). An elite corner who follows
the other team's best receiver should show up anyway as a gap: the defense holds the WR1 further
below his projection than it holds the other receivers.

Roles, known before the game: each team's WR1 is its wide receiver with the highest projected
receiving yards; the rest are WR2+. The same dvp rating is built separately for each role.

1. Is the gap a stable trait? For each defense-season with 8+ games, the WR1 rating minus the
   WR2+ rating (the per-game log ratios, averaged) in odd weeks against even weeks: the
   correlation between the halves. Near zero means the gap is noise.
2. Does it improve projections? Chain projections (players_proj.pkl), tuning 2013-2018, holdout
   2019 on, receiving yards and receptions. Base: projection x exp(b x the WR rating), as the
   board does. Candidate: least squares on both the WR rating and the role's own rating (WR1
   rows and WR2+ rows each get their own coefficients), fitted on tuning seasons only.

Rule, set before running: kept only if the candidate lowers the average miss on the holdout by
more than two standard errors (clustered by game) and is no worse on the tuning seasons; only
then would it be checked against the books' 2025 lines.

Writes experiments/out/wr1_matchup_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import props_blend as PB  # noqa: E402
from nflmodel import config as C, dvp  # noqa: E402
from props_calibration import load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "wr1_matchup_test.json"
TUNE = list(range(2013, C.HOLDOUT_FROM))
STATS = ("rec_yds", "receptions")


def label_roles(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    wr = p[(p.group == "WR") & (p.rec_yds_proj > 0)]
    top = wr.sort_values("rec_yds_proj", ascending=False).drop_duplicates(["game_id", "team"])
    p["role"] = np.where(p.group != "WR", None, "WR2")
    p.loc[top.index, "role"] = "WR1"
    return p


def role_ratings(p: pd.DataFrame) -> pd.DataFrame:
    """dvp ratings with the WR group split by role, plus the plain WR rating."""
    old = dvp.PAIRS
    try:
        q = p[p.group == "WR"].copy()
        dvp.PAIRS = [("WR", s) for s in STATS]
        rt_wr = dvp.ratings(q)
        dvp.PAIRS = [("WR1", s) for s in STATS] + [("WR2", s) for s in STATS]
        rt_role = dvp.ratings(q.assign(group=q.role))
    finally:
        dvp.PAIRS = old
    return pd.concat([rt_wr, rt_role], ignore_index=True)


def per_game_gap(p: pd.DataFrame) -> pd.DataFrame:
    """Per defense-game log ratio for WR1 and WR2+ on receiving yards."""
    played = p.played.fillna(False).astype(bool) & (p.active == 1) & (p.group == "WR") & (p.rec_yds_proj > 0)
    d = p[played]
    a = d.groupby(["opp", "season", "week", "role"]).agg(act=("rec_yds", "sum"), exp=("rec_yds_proj", "sum")).reset_index()
    a["lr"] = np.log((a.act.clip(lower=0) + 1) / (a.exp + 1))
    w = a.pivot_table(index=["opp", "season", "week"], columns="role", values="lr").dropna().reset_index()
    w["gap"] = w["WR1"] - w["WR2"]
    return w


def stability(w: pd.DataFrame) -> dict:
    w = w.assign(half=np.where(w.week % 2 == 1, "odd", "even"))
    s = w.groupby(["opp", "season", "half"]).gap.mean().unstack()
    n = w.groupby(["opp", "season"]).size()
    s = s[n.reindex(s.index) >= 8].dropna()
    lvl = w.assign(half=w.half).groupby(["opp", "season", "half"])[["WR1", "WR2"]].mean().unstack().dropna()
    return {"defense_seasons": int(len(s)),
            "gap_split_half_corr": round(float(s["odd"].corr(s["even"])), 3),
            "wr1_level_split_half_corr": round(float(lvl[("WR1", "odd")].corr(lvl[("WR1", "even")])), 3),
            "wr2_level_split_half_corr": round(float(lvl[("WR2", "odd")].corr(lvl[("WR2", "even")])), 3)}


def attach(p, rt, grp, stat):
    return dvp.attach(p, rt, grp, stat).to_numpy(float)


def evaluate(p: pd.DataFrame, rt: pd.DataFrame) -> dict:
    played = p.played.fillna(False).astype(bool) & (p.active == 1) & (p.group == "WR")
    out = {}
    for stat in STATS:
        col = stat + "_proj"
        d = p[played & (p[col] > 0)].copy()
        y, x0 = d[stat].to_numpy(float), d[col].to_numpy(float)
        r_wr = attach(d, rt, "WR", stat)
        r_role = np.where(d.role == "WR1", attach(d, rt, "WR1", stat), attach(d, rt, "WR2", stat))
        tune = d.season.isin(TUNE).to_numpy()
        hold = (d.season >= C.HOLDOUT_FROM).to_numpy()
        # base: the board's single WR rating
        xb = x0 * r_wr
        b = float((xb[tune] * (y - x0)[tune]).sum() / (xb[tune] ** 2).sum())
        base = x0 * np.exp(b * r_wr)
        cand, split = base.copy(), base.copy()
        coefs, split_b = {}, {}
        for role in ("WR1", "WR2"):
            m = (d.role == role).to_numpy()
            X = np.column_stack([x0 * r_wr, x0 * r_role])
            beta = np.linalg.lstsq(X[m & tune], (y - x0)[m & tune], rcond=None)[0]
            coefs[role] = [round(float(v), 3) for v in beta]
            cand[m] = x0[m] * np.exp(beta[0] * r_wr[m] + beta[1] * r_role[m])
            xs = (x0 * r_wr)[m & tune]
            bs = float((xs * (y - x0)[m & tune]).sum() / (xs ** 2).sum())
            split_b[role] = round(bs, 3)
            split[m] = x0[m] * np.exp(bs * r_wr[m])
        res = {"coefs_wr_then_role": coefs, "b_base": round(b, 3), "split_b_wr_rating_only": split_b}
        # what the role's own rating adds over a role-specific strength for the plain WR rating
        d2 = np.abs(cand - y) - np.abs(split - y)
        for name, m in (("tune", tune), ("hold", hold)):
            res[f"role_rating_over_split_{name}"] = {"change": round(float(d2[m].mean()), 4),
                "se": round(float(PB._se_by_game(pd.Series(d2[m]), pd.Series(d.game_id.to_numpy()[m]))), 4),
                "change_WR1": round(float(d2[m & (d.role == "WR1").to_numpy()].mean()), 4)}
        diff = np.abs(cand - y) - np.abs(base - y)
        for name, m in (("tune", tune), ("hold", hold)):
            res[name] = {"n": int(m.sum()), "base_mae": round(float(np.abs(base - y)[m].mean()), 3),
                         "change": round(float(diff[m].mean()), 4),
                         "se": round(float(PB._se_by_game(pd.Series(diff[m]), pd.Series(d.game_id.to_numpy()[m]))), 4)}
            for role in ("WR1", "WR2"):
                mm = m & (d.role == role).to_numpy()
                res[name][f"change_{role}"] = round(float(diff[mm].mean()), 4)
        res["keep"] = bool(res["hold"]["change"] < -2 * res["hold"]["se"] and res["tune"]["change"] <= 0)
        out[stat] = res
    return out


def main():
    p = label_roles(load_proj(None))
    rt = role_ratings(p)
    w = per_game_gap(p)
    res = {"stability": stability(w)}
    den = w[(w.opp == "DEN") & (w.season >= 2023)].groupby("season")[["WR1", "WR2", "gap"]].mean().round(3)
    res["denver_2023_on"] = {int(k): v for k, v in den.T.to_dict().items()}
    res["projections"] = evaluate(p, rt)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
