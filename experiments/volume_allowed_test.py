"""What a defense allows in pass attempts and carries, against what its opponents usually do.

    python experiments/volume_allowed_test.py

The team-volume regression (players.VOL_FEATS) takes `opp_att_allowed` and `opp_car_allowed`: the
pass attempts and carries the defense's opponents have had against it, recent games weighted more.
That raw count mixes three things: the defense, the offenses it happened to face, and the game
script of those games (a team that builds big leads sees more passes). Two versions that take
the other two out, each from the defense's earlier games only:

  vs usual     opponents' attempts (carries) against this defense minus those same offenses'
               own recent averages going in (team_att_avg): the schedule taken out
  vs expected  minus what the current volume model predicted for those games, which also knows
               their projected margin and total: the schedule and the game script taken out

Each is a weighted average over the defense's earlier games (TEAM_HL-game half-life, last season
at half weight, older dropped), shrunk toward 0 by SHRINK games, and replaces (variant "swap") or
joins ("add") the raw count in the regression. Walk-forward exactly as players_more_test.py:
each season's regression fitted on the eight before it, scored on team-games by average miss.
Kept only if it helps on the tuning seasons (2014-2018) and the holdout (2019 on).

Writes experiments/out/volume_allowed_test.json.
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
from nflmodel import config as C, players as PL  # noqa: E402
from props_calibration import load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "volume_allowed_test.json"
SHRINK = 4.0
LAST_SEASON = 0.5


def defense_rating(t: pd.DataFrame, resid_col: str, out_col: str) -> pd.DataFrame:
    """For each (game, defense): the weighted average of the residual its opponents had in its
    earlier games. t: one row per (game, offense team) with opp = the defense."""
    t = t.sort_values(["season", "week"])
    out = np.zeros(len(t))
    pos = {ix: i for i, ix in enumerate(t.index)}
    hl = float(PL.TEAM_HL) if hasattr(PL, "TEAM_HL") else 6.0
    for dfn, h in t.groupby("opp", sort=False):
        r = h[resid_col].to_numpy(float)
        season = h["season"].to_numpy(int)
        have = ~np.isnan(r)
        idx = np.cumsum(have) - have
        pz = np.flatnonzero(have)
        for i in range(len(h)):
            prev = pz[pz < i]
            prev = prev[season[prev] >= season[i] - 1]
            if len(prev):
                k = idx[i] - 1 - idx[prev]
                w = 0.5 ** (k / hl) * np.where(season[prev] == season[i], 1.0, LAST_SEASON)
                out[pos[h.index[i]]] = float((w * r[prev]).sum() / (w.sum() + SHRINK))
    t[out_col] = out
    return t


def walk(t, target, feats):
    pred = pd.Series(np.nan, index=t.index)
    for s in sorted(t.season.unique()):
        if s < 2013:
            continue
        tr = t[(t.season < s) & (t.season >= s - 8) & t.played.astype(bool) & t[target].notna()]
        te = t.season == s
        if len(tr) < 200:
            continue
        m = PL.Lin(5.0).fit(tr[feats].fillna(0), tr[target].to_numpy(float))
        pred[te] = m.predict(t.loc[te, feats].fillna(0))
    return pred


def main():
    proj = load_proj(None)
    t = proj.drop_duplicates(["game_id", "team"]).copy().reset_index(drop=True)
    base = list(PL.VOL_FEATS)
    rep = {"base_feats": base}
    for target, avg, raw in (("team_att", "team_att_avg", "opp_att_allowed"), ("team_car", "team_car_avg", "opp_car_allowed")):
        t[f"{target}_vs_usual"] = np.where(t.played.astype(bool), t[target] - t[avg], np.nan)
        p0 = walk(t, target, base)
        t[f"{target}_vs_exp"] = np.where(t.played.astype(bool), t[target] - p0, np.nan)
        t = defense_rating(t, f"{target}_vs_usual", f"def_{target}_usual")
        t = defense_rating(t, f"{target}_vs_exp", f"def_{target}_exp")
        variants = {"current": base,
                    "vs usual, swap": [f for f in base if f != raw] + [f"def_{target}_usual"],
                    "vs usual, add": base + [f"def_{target}_usual"],
                    "vs expected, swap": [f for f in base if f != raw] + [f"def_{target}_exp"],
                    "vs expected, add": base + [f"def_{target}_exp"]}
        ok = t.played.astype(bool) & t[target].notna() & (t.season >= 2014)
        res = {}
        preds = {name: walk(t, target, feats) for name, feats in variants.items()}
        for name, pr in preds.items():
            d = t[ok].assign(pred=pr[ok]).dropna(subset=["pred"])
            res[name] = {}
            for per, m in (("tune", d.season < C.HOLDOUT_FROM), ("hold", d.season >= C.HOLDOUT_FROM)):
                e = np.abs(d.loc[m, "pred"] - d.loc[m, target])
                e0 = np.abs(d.loc[m].assign(p0=preds["current"][d.loc[m].index])["p0"] - d.loc[m, target])
                diff = e - e0
                g = diff.groupby(d.loc[m, "game_id"]).sum()
                se = float(g.std(ddof=1) * np.sqrt(len(g)) / len(diff)) if len(g) > 1 else None
                res[name][per] = {"n": int(m.sum()), "mae": round(float(e.mean()), 4), "vs_current": round(float(diff.mean()), 4),
                                  "se": round(se, 4) if se else None}
        res["rating_sd"] = {"usual": round(float(t[f"def_{target}_usual"].std()), 3), "expected": round(float(t[f"def_{target}_exp"].std()), 3)}
        rep[target] = res
        print(f"== {target} (rating sd: vs usual {res['rating_sd']['usual']}, vs expected {res['rating_sd']['expected']})")
        for name in variants:
            r = res[name]
            print(f"  {name:20s} tune {r['tune']['mae']:.3f} ({r['tune']['vs_current']:+.4f} ± {r['tune']['se'] or 0:.4f})"
                  f"   hold {r['hold']['mae']:.3f} ({r['hold']['vs_current']:+.4f} ± {r['hold']['se'] or 0:.4f})")
    OUT.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
