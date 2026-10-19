"""College quarterback settings, scored on games where a team started a different
quarterback from its previous game (and on all games).

    python experiments/cfb_qb_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import data, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

mae = lambda a, b: float(np.mean(np.abs(a - b)))


def qb_changes(games):
    """game_id -> True when either team's starter differs from its previous game's."""
    p = games[games.played & games.home_qb_id.notna()].sort_values("order")
    long = []
    for side in ("home", "away"):
        long.append(p[["game_id", "season", "order"]].assign(team=p[f"{side}_team"], qb=p[f"{side}_qb_id"]))
    import pandas as pd
    long = pd.concat(long).sort_values(["team", "order"])
    long["prev"] = long.groupby(["team", "season"])["qb"].shift(1)
    long["chg"] = long["prev"].notna() & (long["prev"] != long["qb"])
    return long.groupby("game_id")["chg"].any()


def main():
    games, tg, qb = data.load_all(verbose=False)
    chg = qb_changes(games)
    settings = [("NFL-style (-0.10, 200)", {"rep": -0.10, "qb_prior": 200.0}),
                ("unknown QB at -0.20", {"rep": -0.20}),
                ("current (-0.30, 800)", {}),
                ("unknown QB at -0.35", {"rep": -0.35}),
                ("QB half-life 700", {"qb_half_life": 700.0}),
                ("QB half-life 2800", {"qb_half_life": 2800.0}),
                ("QB prior 400", {"qb_prior": 400.0}),
                ("QB prior 1600", {"qb_prior": 1600.0}),
                ("no QB features", {"no_qb": True})]
    print(f"{'setting':22s} {'tune chg':>9s} {'hold chg':>9s} {'tune all':>9s} {'hold all':>9s}")
    for name, kw in settings:
        rep = kw.pop("rep", None)
        no_qb = kw.pop("no_qb", False)
        cls = type("E", (CFBRatingEngine,), {"QB_REPLACEMENT": rep}) if rep is not None else CFBRatingEngine
        eng = cls(tg, qb, games=games, **kw)
        fr = model.build_frame(games, eng)
        mf = [c for c in model.MARGIN_FEATS if not (no_qb and "qb" in c)]
        out, _ = model.walk_forward(fr, margin_feats=mf)
        f = out[out.played & out.spread_line.notna()].copy()
        f["chg"] = f.game_id.map(chg).fillna(False).astype(bool)
        T, H = f[f.season.between(2017, 2019)], f[f.season >= 2021]
        print(f"{name:22s} {mae(T[T.chg].model_margin, T[T.chg].result):9.3f} {mae(H[H.chg].model_margin, H[H.chg].result):9.3f} "
              f"{mae(T.model_margin, T.result):9.3f} {mae(H.model_margin, H.result):9.3f}", flush=True)
    print(f"{'closing line':22s} {mae(T[T.chg].spread_line, T[T.chg].result):9.3f} {mae(H[H.chg].spread_line, H[H.chg].result):9.3f} "
          f"{mae(T.spread_line, T.result):9.3f} {mae(H.spread_line, H.result):9.3f}")
    print(f"QB-change games: {int(T.chg.sum())} tuning, {int(H.chg.sum())} holdout")


if __name__ == "__main__":
    main()
