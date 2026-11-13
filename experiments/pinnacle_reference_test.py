"""Strategy 3: a DraftKings/FanDuel price that beats Pinnacle's, split by whether the model agrees.

    python experiments/pinnacle_grade.py --league nfl --seasons 2023 2024 2025   # writes the bets
    python experiments/pinnacle_grade.py --league cfb --seasons 2024 2025
    python experiments/pinnacle_reference_test.py

The Nov 6, 2026 test (Project doc claude/betting-strategies.md section 3; its script was written in a
read-only session and never reached the repo, so this is a re-implementation of the same rule, Nov 13,
2026). Bets: the price tier at the Monday open (pinnacle_grade.price_tier: the retail price beats
Pinnacle's no-vig number by sheet.PRICE_MIN at the book's own line; one bet per game, market and side).
"Agrees": the model's line (the Sunday-night run, no week's quarterbacks or injuries:
pinnacle_grade.model_frame) is on the bet's side of Pinnacle's open line: spreads and moneylines by
the model margin against Pinnacle's spread, totals by the model total against Pinnacle's total.

Rule (Nov 6, unchanged): kept if (a) agrees beats disagrees on closing line value vs Pinnacle's close
in every league-season, (b) the pooled agrees-minus-disagrees CLV gap is 2+ SE, (c) the agrees bets'
return is no worse than the whole tier's. Nov 13: NFL 2023 is a league-season the Nov 6 test never
saw (backfilled that day); it is reported on its own and added to (a).

Writes experiments/out/pinnacle_reference_test.json.
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
from nflmodel import config as C, market as M  # noqa: E402

OUT = HERE / "out" / "pinnacle_reference_test.json"
SEASONS = {"nfl": [2023, 2024, 2025], "cfb": [2024, 2025]}


def rec(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"n": 0}
    c = b.dropna(subset=["clv"])
    return {"n": int(len(b)), "w": int((b.res > 0).sum()), "l": int((b.res < 0).sum()),
            "return": round(float(b.profit.mean()), 4), "return_se": round(PG._se(b.profit, b.game_id), 4) if len(b) > 1 else None,
            "clv": round(float(c.clv.mean()), 4) if len(c) else None, "clv_se": round(PG._se(c.clv, c.game_id), 4) if len(c) > 1 else None,
            "beat_close": round(float((c.clv > 0).mean()), 3) if len(c) else None}


def main():
    out, allb = {}, []
    for league, seasons in SEASONS.items():
        lg = PG.League(league)
        games = lg.games()
        frame = PG.model_frame(lg, games).set_index("game_id")
        bets = pd.read_csv(C.RAW / f"pinnacle_price_bets_{league}.csv")
        bets = bets[(bets["kind"] == "open") & bets["season"].isin(seasons)].copy()
        agree = []
        for s in seasons:
            open_q, _ = PG.quotes(league, s, games, lg.team_names(s))
            for r in bets[bets.season == s].itertuples():
                pin = (open_q.get(r.game_id) or {}).get(M.SHARP) or {}
                f = frame.loc[r.game_id] if r.game_id in frame.index else None
                a = np.nan
                if f is not None:
                    if r.market == "Total" and "total_line" in pin and pd.notna(f["model_total"]):
                        a = float(f["model_total"] > pin["total_line"]) if r.side == "over" else float(f["model_total"] < pin["total_line"])
                    elif r.market in ("Spread", "Moneyline") and "spread_line" in pin and pd.notna(f["model_margin"]):
                        a = float(f["model_margin"] > pin["spread_line"]) if r.side == "home" else float(f["model_margin"] < pin["spread_line"])
                agree.append((r.Index, a))
        bets["agrees"] = pd.Series(dict(agree))
        bets = bets.dropna(subset=["agrees"])
        bets["league"] = league
        allb.append(bets)
        for s in seasons:
            x = bets[bets.season == s]
            out[f"{league}_{s}"] = {"all": rec(x), "agrees": rec(x[x.agrees == 1]), "disagrees": rec(x[x.agrees == 0])}
        out[league] = {"all": rec(bets), "agrees": rec(bets[bets.agrees == 1]), "disagrees": rec(bets[bets.agrees == 0])}
    B = pd.concat(allb, ignore_index=True)
    out["pooled"] = {"all": rec(B), "agrees": rec(B[B.agrees == 1]), "disagrees": rec(B[B.agrees == 0])}
    a, d = B[B.agrees == 1].dropna(subset=["clv"]), B[B.agrees == 0].dropna(subset=["clv"])
    gap = a.clv.mean() - d.clv.mean()
    se = float(np.sqrt(PG._se(a.clv, a.game_id) ** 2 + PG._se(d.clv, d.game_id) ** 2))
    cells = {k: v for k, v in out.items() if "_" in k}
    part_a = {k: (v["agrees"].get("clv") or -9) > (v["disagrees"].get("clv") or -9) for k, v in cells.items()}
    out["rule"] = {"a_every_league_season": part_a, "a": all(part_a.values()),
                   "b_gap": round(float(gap), 4), "b_se": round(se, 4), "b": bool(gap >= 2 * se),
                   "c": bool(out["pooled"]["agrees"]["return"] >= out["pooled"]["all"]["return"])}
    out["rule"]["kept"] = bool(out["rule"]["a"] and out["rule"]["b"] and out["rule"]["c"])
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
