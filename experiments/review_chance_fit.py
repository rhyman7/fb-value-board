"""Fit the Weekly Review's chance-ledger point values on 2023-25 (Nov 17, 2026).

For every regular-season game: how far the final margin landed from the closing spread,
regressed on the three chance items (home minus away): fumble bounces (fumbles lost beyond
half the team's fumbles), non-offensive touchdowns, and missed field goals from 45 yards or
closer. The coefficients are the points each item moved the margin past the line, which is
what the Review's ledger shows. Writes data/review_chance.json.

    python experiments/review_chance_fit.py --nfl PBP_2023 PBP_2024 PBP_2025 --cfb PBP_2023 PBP_2024 PBP_2025

Display only: the values never feed the model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import review as R

ROOT = Path(__file__).resolve().parents[1]


def frame(league: str, paths: list[str]) -> pd.DataFrame:
    out = []
    for path in paths:
        p = R.load_pbp(league, 0, Path(path))
        if league == "nfl":
            p = p[p["season_type"] == "REG"]
            box = R.box_nfl(p)
            g = p.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"),
                                         hs=("home_score", "max"), as_=("away_score", "max"), line=("spread_line", "first"))
            h = box.merge(g, left_on=["game_id", "team"], right_on=["game_id", "home"], how="inner")
            a = box.merge(g.reset_index()[["game_id", "away"]], left_on=["game_id", "team"], right_on=["game_id", "away"])
        else:
            p = p[p["seasonType"].astype(str).isin(["2", "regular"]) | (p["seasonType"] == 2)]
            box = R.box_cfb(p)
            g = p.groupby("game_id").agg(hid=("homeTeamId", "first"), hs=("homeFinalScore", "first"),
                                         as_=("awayFinalScore", "first"), line=("homeTeamSpread", "first"))
            h = box.merge(g, left_on=["game_id", "team_id"], right_on=["game_id", "hid"])
            a = box[~box.set_index(["game_id", "team_id"]).index.isin(h.set_index(["game_id", "team_id"]).index)]
        m = h.merge(a, on="game_id", suffixes=("_h", "_a"))
        m["resid"] = (m["hs_h"] if "hs_h" in m else m["hs"]) - (m["as__h"] if "as__h" in m else m["as_"]) - (m["line_h"] if "line_h" in m else m["line"])
        m["fum"] = (m["fum_lost_a"] - m["fumbles_a"] / 2) - (m["fum_lost_h"] - m["fumbles_h"] / 2)
        m["nonoff"] = m["nonoff_td_h"] - m["nonoff_td_a"]
        m["fg"] = m["fg_miss_short_a"] - m["fg_miss_short_h"]
        out.append(m[["resid", "fum", "nonoff", "fg"]].dropna())
    return pd.concat(out)


def fit(df: pd.DataFrame) -> dict:
    X = np.column_stack([np.ones(len(df)), df[["fum", "nonoff", "fg"]].to_numpy(float)])
    y = df["resid"].to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    cov = (e @ e) / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"n": len(df), "fumble": round(b[1], 2), "nonoff_td": round(b[2], 2), "fg_miss": round(b[3], 2),
            "se": {"fumble": round(se[1], 2), "nonoff_td": round(se[2], 2), "fg_miss": round(se[3], 2)},
            "resid_sd": round(float(np.std(y)), 2), "resid_sd_after": round(float(np.std(e)), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfl", nargs="+", required=True)
    ap.add_argument("--cfb", nargs="+", required=True)
    a = ap.parse_args()
    res = {"nfl": fit(frame("nfl", a.nfl)), "cfb": fit(frame("cfb", a.cfb))}
    print(json.dumps(res, indent=1))
    keep = {k: {x: v[x] for x in ("fumble", "nonoff_td", "fg_miss")} | {"fit": v} for k, v in res.items()}
    (ROOT / "data" / "review_chance.json").write_text(json.dumps(keep, indent=1))
