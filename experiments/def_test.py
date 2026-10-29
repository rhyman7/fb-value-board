"""Part B: the rating-based defensive features in the margin and total models.

    python experiments/def_ratings.py   (once; builds data/raw/def_experiments/team_def.csv)
    python experiments/def_test.py

Each candidate is added alone to the current model (production_frame), walk-forward, the
FEATURES.md way. Margin features are home minus away (positive: the home defense is missing
more); total features are home plus away. "change" is missing now less what the team's last
four games already reflect (as d_chg_ol).

Two windows, because the inputs start in different years:
  long   play-by-play pass rush (QB hits + sacks), 2012 on: tuning 2013-2018, holdout 2019 on,
         the same as October.
  pfr    PFR pressures, coverage and tackling, 2018 on. The first season with earlier PFR games
         to rate from is 2019, and a walk-forward fit only sees a feature from the season after
         it first appears, so the first season the model can use them is 2020. Tuning
         2020-2022 (3 seasons, about 800 games), holdout 2023 on. Short: read with caution.
  ftn    FTN blitzers and pass rushers per dropback, 2022 on, usable from 2024: no tuning
         window at all, so only the 2024-2026 change is reported, and it cannot earn a place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_common as K  # noqa: E402
import def_ratings as DR  # noqa: E402
from nflmodel import model  # noqa: E402

WINDOWS = {"long": ((2013, 2018), (2019, 2100)), "pfr": ((2020, 2022), (2023, 2100)), "ftn": ((2024, 2100), (2024, 2100))}
AGG = ["agg_press", "agg_blitz", "agg_ypt", "agg_mt", "agg_ftn_blitz", "agg_ftn_rush"]


def attach(frame: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    vals = [c for c in T.columns if c.startswith(("miss_", "mrec_"))]
    T = T.copy()
    for c in [c for c in vals if c.startswith("miss_")]:
        T["chg_" + c[5:]] = T[c] - T["mrec_" + c[5:]]
    feats = [c for c in T.columns if c.startswith(("miss_", "chg_"))] + AGG
    lg = T[AGG].mean()
    f = frame
    for side, team in (("h", "home_team"), ("a", "away_team")):
        s = T[["game_id", "team"] + feats].rename(columns={"team": team, **{c: f"{side}_{c}" for c in feats}})
        f = f.merge(s, on=["game_id", team], how="left")
    new = {}
    for c in feats:
        fill = lg[c] if c in AGG else 0.0
        h, a = f[f"h_{c}"].fillna(fill), f[f"a_{c}"].fillna(fill)
        new[f"d_{c}"] = h - a
        new[f"s_{c}"] = h + a - (2 * fill if c in AGG else 0.0)
    f = pd.concat([f.drop(columns=[f"{p}_{c}" for c in feats for p in ("h", "a")]), pd.DataFrame(new, index=f.index)], axis=1)
    return f


def candidates():
    c = []
    # Part A by rating: the October groups, each defender weighted by his rating instead of counted
    for sub, lab in (("dl", "defensive line"), ("lb", "linebackers"), ("all", "all defenders")):
        for rep in ("bk", "dc"):
            for m in ("miss", "chg"):
                c.append(("A", "long", f"pass rush (pbp), {lab}, {m}, vs {rep}", [f"d_{m}_pr_pbp_{rep}_{sub}"], []))
    for rep in ("bk", "dc"):
        for m in ("miss", "chg"):
            c.append(("A", "pfr", f"pass rush (PFR), defensive line, {m}, vs {rep}", [f"d_{m}_pr_pfr_{rep}_dl"], []))
            c.append(("A", "pfr", f"coverage, linebackers, {m}, vs {rep}", [f"d_{m}_cov_db_{rep}_lb"], []))
            c.append(("A", "pfr", f"coverage, defensive backs, {m}, vs {rep}", [f"d_{m}_cov_db_{rep}_db"], []))
            c.append(("A", "pfr", f"all defenders (rush + coverage), {m}, vs {rep}", [f"d_{m}_pr_pfr_{rep}_all", f"d_{m}_cov_db_{rep}_all"], []))
            c.append(("A", "long", f"pass rush (pbp) in total, {m}, vs {rep}", [], [f"s_{m}_pr_pbp_{rep}_all"]))
            c.append(("A", "pfr", f"pass rush (PFR) in total, {m}, vs {rep}", [], [f"s_{m}_pr_pfr_{rep}_all"]))
    # Part B: the new ratings, all defenders
    for r, lab in (("pr_pfr", "pressure rate"), ("cov_db", "coverage yards per snap"), ("cov_ypt", "coverage yards per target"), ("mt", "missed-tackle rate")):
        for rep in ("bk", "dc"):
            for m in ("miss", "chg"):
                c.append(("B", "pfr", f"{lab}, {m}, vs {rep}", [f"d_{m}_{r}_{rep}_all"], []))
                c.append(("B", "pfr", f"{lab} in total, {m}, vs {rep}", [], [f"s_{m}_{r}_{rep}_all"]))
    for a, lab in (("agg_press", "team pressure rate"), ("agg_blitz", "team blitz rate"), ("agg_ypt", "team yards per target allowed"),
                   ("agg_mt", "team missed-tackle rate")):
        c.append(("B", "pfr", lab, [f"d_{a}"], []))
        c.append(("B", "pfr", lab + " in total", [], [f"s_{a}"]))
    for a, lab in (("agg_ftn_blitz", "FTN blitzers per dropback"), ("agg_ftn_rush", "FTN pass rushers per dropback")):
        c.append(("B", "ftn", lab, [f"d_{a}"], []))
        c.append(("B", "ftn", lab + " in total", [], [f"s_{a}"]))
    return c


def main():
    frame = attach(K.production_frame(), pd.read_csv(DR.RAW / "team_def.csv"))
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    base = K.per_season(frame, mf, tf)
    open_base = {w: K.against_open(frame, mf, tf, *WINDOWS[w]) for w in ("long", "pfr")}
    bm = lambda w, a, b: base.loc[a:b, (w, "sum")].sum() / base.loc[a:b, (w, "count")].sum()
    for w, (tu, ho) in WINDOWS.items():
        print(f"base, {w} window: margin tune {bm('em', *tu):.3f} hold {bm('em', *ho):.3f} | total tune {bm('et', *tu):.3f} hold {bm('et', *ho):.3f}")
    rows, seasons = [], {}
    for part, w, label, am, at in candidates():
        cand = K.per_season(frame, mf + am, tf + at)
        which = "em" if am else "et"
        d = K.compare(base, cand, *WINDOWS[w], which)
        o = None
        if am and w != "ftn":
            oc = K.against_open(frame, mf + am, tf, *WINDOWS[w])
            o = (oc["tune"][0] - open_base[w]["tune"][0], oc["hold"][0] - open_base[w]["hold"][0])
        kept = bool(d["tune"] < 0 and d["hold"] < 0 and w != "ftn")
        print(f"[{part} {w:4s}] " + K.fmt_row(label, d) + (f" | opener {o[0]:+.4f} / {o[1]:+.4f}" if o else "") + ("  <-- passes" if kept else ""), flush=True)
        rows.append({"part": part, "window": w, "candidate": label, "model": "margin" if am else "total", "features": " ".join(am + at),
                     "tune": d["tune"], "hold": d["hold"], "pre2020": d["pre2020"][0], "pre2020_better": f"{d['pre2020'][1]}/{d['pre2020'][2]}",
                     "post2020": d["post2020"][0], "post2020_better": f"{d['post2020'][1]}/{d['post2020'][2]}",
                     "open_tune": o[0] if o else None, "open_hold": o[1] if o else None, "passes": kept})
        seasons[label] = d["seasons"]
    K.OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(K.OUT / "def_test.csv", index=False)
    pd.DataFrame(seasons).round(4).to_csv(K.OUT / "def_test_seasons.csv")


if __name__ == "__main__":
    main()
