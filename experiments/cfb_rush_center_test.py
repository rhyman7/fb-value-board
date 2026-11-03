"""College rushing-yards props: which center prices the lines best?

    python experiments/cfb_rush_center_test.py

Follow-up to cfb_rush_blend_test.py (the chain had the lowest absolute miss, but by running 9
yards low on a right-skewed stat). The question that matters for betting: used as the center of
the prop distribution, which number gives the best chance at the books' actual lines?

Rules, written before running (Nov 3, 2026):

1. Lines: the 2025 college rushing-yards main lines in data/props/history_cfb/2025/ (one
   snapshot per game about ten minutes before kickoff, DraftKings and FanDuel), the same
   frame as cfb_props_real_lines.py: both sides quoted, pushes left out, the books' chance
   the average no-vig over chance across books (power method).
2. Candidates, each written into rush_yds_proj before nflmodel.props.PropModel prices the
   line (nothing else changes; carries_proj stays the board's):
     chain, recent, average (of the two), projection (the board's; the control).
3. Score: Brier score of the over chance, and its difference from the books' and from the
   control, standard errors clustered by game. Also the nudged chance the Parlays page would
   use (books moved 20% toward the model, at most 3 points, conflicts dropped).
4. Decision: a candidate may replace the projection as the rushing prop center only if its
   Brier beats the control's by 2+ standard errors; and since this is one season of lines, only
   after the same check on 2026's lines. Otherwise nothing changes.

Writes experiments/out/cfb_rush_center_test.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import cfb_props_real_lines as R  # noqa: E402
from nflmodel import odds as O, parlays as PA, propdist as D, propmarket as PM, props as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CANDS = ("chain", "recent", "average", "projection")


def main():
    pproj, pinfo, games = R.projections()
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    rows = R.player_frame(pproj, pinfo)
    rows["recent"] = (rows["rush_yds_sum"] / rows["one_sum"].replace(0, np.nan)).fillna(0.0)
    rows["chain"] = rows["rush_yds_p"]
    rows["average"] = (rows["chain"] + rows["recent"]) / 2
    rows["projection"] = rows["rush_yds_proj"]
    model = P.PropModel.load()

    main = s[~s["alt"] & (s["stat"] == "rush_yds")]
    w = main.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index()
    w = w.dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair_b"].mean().rename(columns={"fair_b": "fair"})
    base = w.merge(rows, on=["game_id", "nname"], how="inner")

    res, ours = [], {}
    for c in CANDS:
        d = base.copy()
        d["rush_yds_proj"] = d[c]
        x = R.model_chances(d, model)
        x = x[x["y"] != x["line"]].sort_values(["game_id", "nname", "line"]).reset_index(drop=True)
        ours[c] = x
    keys = ["game_id", "nname", "line"]
    common = set.intersection(*[set(map(tuple, x[keys].to_numpy())) for x in ours.values()])
    for c in CANDS:
        x = ours[c]
        x = x[[tuple(k) in common for k in x[keys].to_numpy()]].reset_index(drop=True)
        ours[c] = x
    ctl = ours["projection"]
    hit = (ctl["y"] > ctl["line"]).astype(float).to_numpy()
    mkt = ctl["fair"].to_numpy()
    p_ctl = (ctl["our_over"] / (ctl["our_over"] + ctl["our_under"])).to_numpy()
    for c in CANDS:
        x = ours[c]
        p = (x["our_over"] / (x["our_over"] + x["our_under"])).to_numpy()
        vs_b, se_b = R.clustered_diff(p, mkt, hit, x["game_id"])
        vs_c, se_c = R.clustered_diff(p, p_ctl, hit, x["game_id"])
        gap = p - mkt
        keep = np.abs(gap) <= PA.CONFLICT
        nudged = mkt + np.clip(0.2 * gap, -PA.PULL_MAX, PA.PULL_MAX)
        n_vs, n_se = R.clustered_diff(nudged[keep], mkt[keep], hit[keep], x["game_id"].to_numpy()[keep])
        res.append({"candidate": c, "props": len(x), "games": x["game_id"].nunique(), "over_rate": hit.mean(),
                    "mean_center": float(x["rush_yds_proj"].mean()), "mean_line": float(x["line"].mean()),
                    "over_chance": float(p.mean()), "brier": D.brier(p, hit), "brier_books": D.brier(mkt, hit),
                    "vs_books": vs_b, "se_books": se_b, "vs_projection": vs_c, "se_proj": se_c,
                    "z_vs_projection": vs_c / se_c if se_c > 0 else 0.0,
                    "nudged_vs_books": n_vs, "nudged_se": n_se, "nudged_kept": int(keep.sum())})
    out = pd.DataFrame(res)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "cfb_rush_center_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
