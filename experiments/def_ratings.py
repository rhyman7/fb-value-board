"""Part B: a rating for every defender, and team-game features built on them (Oct 2026).

    python experiments/def_ratings.py      (writes data/raw/def_experiments/team_def.csv)

The code lives in nflmodel/def_ratings.py (the shadow test uses it too); see its docstring.
This builds every season at once, with FTN, for the experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C  # noqa: E402
from nflmodel.def_ratings import *  # noqa: E402,F401,F403
from nflmodel.def_ratings import build  # noqa: E402

RAW = C.RAW / "def_experiments"


def main():
    T = build()
    RAW.mkdir(parents=True, exist_ok=True)
    T.round(5).to_csv(RAW / "team_def.csv", index=False)
    print(f"wrote {RAW / 'team_def.csv'}: {len(T)} team-games")


if __name__ == "__main__":
    main()
