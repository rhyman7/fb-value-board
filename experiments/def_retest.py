"""Part A: re-test the defensive features rejected in October 2026 (FEATURES.md, "Injuries
beyond the quarterback" and "Pass rush, by production") against the current model, which
now also has d_depth1_out, d_snap_drop (margin) and wind, s_snaps (total).

    python experiments/def_retest.py

Same rule as October: each candidate is added alone, walk-forward; it earns a place only if
it lowers the raw model's average miss on the tuning seasons (2013-2018) and holds up on the
holdout (2019 on). Also reported: 2013-2019 against 2020 on (rosters mark game-day inactives
only from 2020; that split was chosen after seeing the October results, so it is reported,
not used to keep anything), season by season, and the model's side against the opener.
The inputs are October's: nflverse injury reports, snap counts and weekly rosters
(injuries.load_injuries, injuries.load_pass_rush), unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_common as K  # noqa: E402
from nflmodel import injuries, model  # noqa: E402

TUNE, HOLD = (2013, 2018), (2019, 2100)
DEF = ("dl", "lb", "db")


def candidates():
    m = []
    for g in DEF:
        m += [(f"{g}, missing now", [f"d_miss_{g}"], []), (f"{g}, change", [f"d_chg_{g}"], [])]
    m += [("pass rush, missing now", ["d_miss_pr"], []), ("pass rush, change", ["d_chg_pr"], []),
          ("pass rush in total, missing now", [], ["s_miss_pr"]), ("pass rush in total, change", [], ["s_chg_pr"]),
          ("all six groups, missing now", [f"d_miss_{g}" for g in injuries.GROUPS], []),
          ("all six groups, change", [f"d_chg_{g}" for g in injuries.GROUPS if g != "ol"], [])]   # d_chg_ol is already in
    return m


def main():
    frame = K.production_frame()
    g = K.games()
    pr = injuries.load_pass_rush(g, verbose=False)
    frame = injuries.add_pass_rush_features(frame, pr)
    frame["s_miss_pr"] = frame["h_miss_pr"] + frame["a_miss_pr"]
    frame["s_chg_pr"] = (frame["h_miss_pr"] - frame["h_mrec_pr"]) + (frame["a_miss_pr"] - frame["a_mrec_pr"])
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    base = K.per_season(frame, mf, tf)
    ob = K.against_open(frame, mf, tf)
    bm = lambda w, a, b: base.loc[a:b, (w, "sum")].sum() / base.loc[a:b, (w, "count")].sum()
    print(f"base: margin tune {bm('em', *TUNE):.3f} hold {bm('em', 2019, 2100):.3f} | total tune {bm('et', *TUNE):.3f} hold {bm('et', 2019, 2100):.3f}"
          f" | side vs opener tune {ob['tune'][0]:.4f} hold {ob['hold'][0]:.4f}")
    rows, seasons = [], {}
    for label, am, at in candidates():
        cand = K.per_season(frame, mf + am, tf + at)
        which = "em" if am else "et"
        d = K.compare(base, cand, TUNE, HOLD, which)
        o = K.against_open(frame, mf + am, tf + at) if am else None
        print(K.fmt_row(label, d) + (f" | opener {o['tune'][0] - ob['tune'][0]:+.4f} / {o['hold'][0] - ob['hold'][0]:+.4f}" if o else ""), flush=True)
        rows.append({"candidate": label, "model": "margin" if am else "total", "tune": d["tune"], "hold": d["hold"],
                     "pre2020": d["pre2020"][0], "pre2020_better": f"{d['pre2020'][1]}/{d['pre2020'][2]}",
                     "post2020": d["post2020"][0], "post2020_better": f"{d['post2020'][1]}/{d['post2020'][2]}",
                     "open_tune": None if o is None else o["tune"][0] - ob["tune"][0],
                     "open_hold": None if o is None else o["hold"][0] - ob["hold"][0],
                     "kept": bool(d["tune"] < 0 and d["hold"] < 0)})
        seasons[label] = d["seasons"]
    K.OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(K.OUT / "def_retest.csv", index=False)
    pd.DataFrame(seasons).round(4).to_csv(K.OUT / "def_retest_seasons.csv")


if __name__ == "__main__":
    main()
