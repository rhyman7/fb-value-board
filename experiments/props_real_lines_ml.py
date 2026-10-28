"""The boosted projections against real 2025 DraftKings and FanDuel lines.

    python experiments/props_real_lines_ml.py --season 2025

Same scoring as props_real_lines.py (Brier against the books' vig-free chance, at the
pricing snapshot with pregame projections and at the close), for rushing and receiving
yards, with the current projections and with nflmodel/players_ml.py's boosted ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import data, players_ml as ML, propbackfill as B, props as P, roles  # noqa: E402
import props_real_lines as RL  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--proj", type=Path, default=None)
    args = ap.parse_args()
    games, _, _ = data.load_all(refresh=False, verbose=False)
    R = roles.load_all(games, verbose=False)
    saved = P.PropModel.load()
    rep = {}
    for name, table, kind in (("pricing", RL.load_pregame(), "flag"), ("close", load_proj(args.proj), "close")):
        table = ML.add_features(table, R)
        boosted = ML.walk_forward(table)
        for label, t in (("current", table), ("boosted", boosted)):
            m = P.PropModel(FAMILY).fit(t[(t.season < args.season) & (t.season >= 2014)])
            m.iso = saved.iso
            x = RL.priced(t, m, B.load(args.season, kind), args.season)
            for stat, groups in (("rush_yds", ("RB",)), ("rec_yds", ("WR", "RB"))):
                y = x[(x["stat"] == stat) & x["group"].isin(groups)]
                rep.setdefault(name, {}).setdefault(stat, {})[label] = RL.compare(y)
            b, s = RL.bets(x[x["stat"].isin(["rush_yds", "rec_yds"]) & x["group"].isin(["RB", "WR"])],
                           RL.priced(t, m, B.load(args.season, "close"), args.season))
            rep[name].setdefault("bets", {})[label] = {k: v for k, v in s.items() if k != "line_moves"}
        print(name, json.dumps(rep[name], indent=1), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"props_real_lines_ml_{args.season}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
