"""T7: college opener bets at Pinnacle's Monday price, split by part of season (Nov 20, 2026).

    python experiments/cfb_opener_real_price_split.py

Addendum, written before the run: Project doc claude/plan-batch-tests-addendum-t7-2026-11-20.md.
A diagnostic: nothing changes. Committed before its first run.

experiments/pinnacle_grade.py's opener-gap bets for college 2024-25, built exactly as that script
builds them (its own raw bet-time run, Pinnacle's Monday 12:00 UTC line and price, short-week games
out), then split into weeks 1-4 and weeks 5+. It does not rewrite that script's output file.

Reading, fixed before the run: weeks 5+ above 0% at both 2+ and 3+ at Pinnacle's price and above
weeks 1-4 at both = the weeks 5+ opener record holds at a real price on the two seasons there are
(a lead for January and the 2027 prediction, not for the board). Weeks 5+ at or below 0% at either
= it does not survive a real price.

Writes experiments/out/cfb_opener_real_price_split.json.
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
warnings.filterwarnings("ignore")

import pinnacle_grade as PG  # noqa: E402

OUT = HERE / "out" / "cfb_opener_real_price_split.json"
SEASONS = (2024, 2025)


def stats(y: pd.DataFrame, res: str = "res", profit: str = "profit", clv: str = "clv_pts") -> dict:
    y = y.dropna(subset=[res])
    n, w, l = int(len(y)), int((y[res] > 0).sum()), int((y[res] < 0).sum())
    if n < 2:
        return {"n": n}
    return {"n": n, "w": w, "l": l, "win": w / (w + l) if w + l else None, "ret": float(y[profit].mean()),
            "se": float(y[profit].std(ddof=1) / np.sqrt(n)), "clv_pts": float(y[clv].mean())}


def main() -> None:
    lg = PG.League("cfb")
    games = lg.games()
    frame = PG.model_frame(lg, games)
    skip = set(games.loc[PG.early_games(lg, games), "game_id"])
    PG.FALLBACK.update(dict(zip(games["game_id"], games["spread_line"])))
    parts = []
    for s in SEASONS:
        open_q, close_q = PG.quotes("cfb", s, games, lg.team_names(s))
        parts.append(PG.opener_gap(lg, frame, s, open_q, close_q, lg.bets.fit_distribution(games, s), skip))
    G = pd.concat(parts, ignore_index=True).merge(games[["game_id", "week"]], on="game_id", how="left")
    res = {}
    for market in ("Spread", "Total"):
        x = G[G.market == market]
        res[market] = {}
        for t in ((2, 3) if market == "Spread" else (2, 3)):
            y = x[x.gap >= t]
            z = x[x.cons_gap >= t] if "cons_gap" in x else x.iloc[0:0]
            res[market][f"{t}+"] = {
                "pinnacle": {"all": stats(y), "weeks 1-4": stats(y[y.week <= 4]), "weeks 5+": stats(y[y.week > 4])},
                "consensus opener at -110": {"all": stats(z, "cons_res", "cons_profit", "cons_clv_pts"),
                                             "weeks 1-4": stats(z[z.week <= 4], "cons_res", "cons_profit", "cons_clv_pts"),
                                             "weeks 5+": stats(z[z.week > 4], "cons_res", "cons_profit", "cons_clv_pts")}}
    a, b = res["Spread"]["2+"]["pinnacle"], res["Spread"]["3+"]["pinnacle"]
    late_pos = a["weeks 5+"]["ret"] > 0 and b["weeks 5+"]["ret"] > 0
    late_ahead = a["weeks 5+"]["ret"] > a["weeks 1-4"]["ret"] and b["weeks 5+"]["ret"] > b["weeks 1-4"]["ret"]
    res["reading"] = ("Weeks 5+ are above 0% at both 2+ and 3+ at Pinnacle's Monday price and above weeks 1-4 at both: the weeks 5+ opener record holds at a real price on these two seasons. A lead only."
                      if late_pos and late_ahead else
                      "Weeks 5+ are at or below 0% at 2+ or 3+ at Pinnacle's Monday price: the weeks 5+ record at -110 does not survive a real price." if not late_pos else
                      "Weeks 5+ are above 0% at both thresholds but not above weeks 1-4 at both: mixed.")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")
    show = lambda s: "-" if s.get("ret") is None else f"{s['w']}-{s['l']} {100 * s['win']:.1f}% {100 * s['ret']:+.1f}% ± {100 * s['se']:.1f}% clv {s['clv_pts']:+.2f}"
    for market in res:
        if market == "reading":
            continue
        for t, r in res[market].items():
            for how, c in r.items():
                print(f"{market} {t} {how}: all {show(c['all'])} | wk1-4 {show(c['weeks 1-4'])} | wk5+ {show(c['weeks 5+'])}")
    print(res["reading"])


if __name__ == "__main__":
    main()
