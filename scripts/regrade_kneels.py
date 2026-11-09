"""One-off (Nov 9, 2026): regrade the rushing props already in the live record with kneel-downs counted.

    python scripts/regrade_kneels.py          # after scripts/add_kneels.py

data/props/prop_results.csv is append-only, and this is the one sanctioned correction to it: rows
graded before Nov 9, 2026 settled rushing yards and rush attempts on the board's box score, which
leaves kneel-downs out; the books count them (nflmodel/players_data.SETTLE). For every graded
rushing-yards or rush-attempts row whose player knelt in that game, `actual` becomes the official
number and `result` / `profit_flat` follow from it. Nothing else in the file changes: other rows
keep their text exactly, and the closing-line columns are left as they were. Prints each row
whose result changes. Safe to run again: a row already at the official number is left alone.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, odds as O, players_data as PD, proplog as PL  # noqa: E402


def main():
    log = pd.read_csv(PL.LOG, low_memory=False, dtype={"player_id": str}).set_index("row_id")
    with PL.RESULTS.open(newline="") as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames, list(rd)
    box = {}
    changed = flipped = 0
    for r in rows:
        if r["result"] not in ("win", "loss", "push") or r["row_id"] not in log.index:
            continue
        g = log.loc[r["row_id"]]
        g = g.iloc[0] if isinstance(g, pd.DataFrame) else g
        extra = PD.SETTLE.get(g["stat"])
        if extra is None:
            continue
        s = int(g["season"])
        if s not in box:
            box[s] = pd.read_csv(C.CACHE / f"player_games_{s}.csv", dtype={"player_id": str}).set_index(["game_id", "player_id"])
        key = (g["game_id"], g["player_id"])
        if key not in box[s].index:
            continue
        b = box[s].loc[key]
        y = float(b[g["stat"]]) + float(b[extra])
        if y == float(r["actual"]):
            continue
        line, dec = float(g["line"]), float(O.american_to_decimal(g["price"]))
        win = y > line if g["side"] == "over" else y < line
        res = "push" if y == line else ("win" if win else "loss")
        if res != r["result"]:
            flipped += 1
            print(f"  week {int(g['week'])} {g['player']} {g['market']} {g['side']} {line}: {r['actual']} -> {y}, {r['result']} -> {res}")
        r["actual"], r["result"] = str(y), res
        r["profit_flat"] = str(0.0 if res == "push" else (round(dec - 1, 5) if win else -1.0))
        changed += 1
    with PL.RESULTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)      # the csv module's default line ending, as proplog._append writes
        w.writeheader()
        w.writerows(rows)
    print(f"{changed} rows regraded, {flipped} results changed")


if __name__ == "__main__":
    main()
