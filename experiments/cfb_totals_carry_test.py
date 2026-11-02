"""College totals: does the points rating lean on last season too much? (follow-up to srs_test.py)

    python experiments/cfb_totals_carry_test.py

srs_test.py found college totals did better since 2021 with this-season-only scoring ratings
(OSRS/DSRS), but not on 2017-2019. The guess: in the transfer-portal years last season says less
about this one, and the board's points rating carries it at 0.55 (config.OFFSEASON_CARRY) with a
9-game half-life (HALF_LIFE_GAMES).

Here only the points rating that feeds the total model (s_pts) is rebuilt with other settings;
everything else, the margin model included, stays as it is. Settings: carry 0.55 (current), 0.40,
0.25, 0.10, 0 x half-life 9 or 15 games.

Rule, set before running (Nov 2, 2026), with a fresh split so the seasons that suggested this
don't also decide it: choose the setting with the lowest average total miss on 2021-2023; keep it
only if on 2024-2025 it beats the current setting by at least 2 standard errors (paired, game by
game). 2017-2019 is reported for reference.

Writes experiments/out/cfb_totals_carry_test.json.
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
from cfbmodel import config as C, data, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "cfb_totals_carry_test.json"
CARRY = (0.55, 0.40, 0.25, 0.10, 0.0)
HALF = (9.0, 15.0)
CHOOSE, CHECK, OLD = (2021, 2023), (2024, 2025), (2017, 2019)


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    base_frame = model.build_frame(games, CFBRatingEngine(tg, qb, games=games))
    base_frame["act_total"] = base_frame["home_score"] + base_frame["away_score"]
    preds = {}
    for carry in CARRY:
        for hl in HALF:
            key = f"carry {carry:.2f}, half-life {hl:g}"
            if (carry, hl) == (C.OFFSEASON_CARRY, C.HALF_LIFE_GAMES):
                f = base_frame
            else:
                alt = model.build_frame(games, CFBRatingEngine(tg, qb, games=games, carry=carry, half_life=hl))
                f = base_frame.drop(columns="s_pts").merge(alt[["game_id", "s_pts"]], on="game_id", how="left")
            o, _ = model.walk_forward(f)
            o = o[o.played & o.model_total.notna()]
            preds[key] = o.set_index("game_id")[["season", "model_total", "act_total"]]
            print(key, flush=True)
    cur = f"carry {C.OFFSEASON_CARRY:.2f}, half-life {C.HALF_LIFE_GAMES:g}"
    base = preds[cur]

    def per(p, lo, hi):
        j = base.join(p[["model_total"]], rsuffix="_t", how="inner")
        j = j[(j.season >= lo) & (j.season <= hi)].dropna()
        d = (j.model_total_t - j.act_total).abs() - (j.model_total - j.act_total).abs()
        return {"n": int(len(j)), "miss": round(float((j.model_total_t - j.act_total).abs().mean()), 4),
                "change": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4) if len(d) > 1 else None}

    table = {k: {"2017-2019": per(p, *OLD), "2021-2023 (choose)": per(p, *CHOOSE), "2024-2025 (check)": per(p, *CHECK)} for k, p in preds.items()}
    pick = min(table, key=lambda k: table[k]["2021-2023 (choose)"]["miss"])
    chk = table[pick]["2024-2025 (check)"]
    res = {"current": cur, "chosen_on_2021_2023": pick, "check_2024_2025": chk,
           "keep": bool(pick != cur and chk["se"] and chk["change"] <= -2 * chk["se"]), "table": table}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1))
    for k, v in table.items():
        print(f"{k:28s} " + "  ".join(f"{p}: {x['change']:+.4f} ± {x['se']}" for p, x in v.items()))


if __name__ == "__main__":
    main()
