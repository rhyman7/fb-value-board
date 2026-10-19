"""Regulars missing from the team's previous game (box scores), as game-model features.

    python experiments/cfb_missing_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import availability as A, data, model, players_data  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

mae = lambda a, b: float(np.mean(np.abs(a - b)))


def main():
    games, tg, qb = data.load_all(verbose=False)
    pg, info = players_data.load_player_games(verbose=False)
    fr = A.add_missing(model.build_frame(games, CFBRatingEngine(tg, qb, games=games)), A.team_game_missing(pg, info, games))

    def score(MF=None, TF=None):
        out, _ = model.walk_forward(fr, margin_feats=MF, total_feats=TF)
        f = out[out.played & out.spread_line.notna() & out.total_line.notna()]
        T, H = f[f.season.between(2017, 2019)], f[f.season >= 2021]
        return mae(T.model_margin, T.result), mae(H.model_margin, H.result), mae(T.model_total, T.total), mae(H.model_total, H.total)

    B = score()
    print(f"current model: margin {B[0]:.3f} / {B[1]:.3f}, total {B[2]:.3f} / {B[3]:.3f}")
    for c in ["d_miss_rb", "d_chg_rb", "d_miss_rec", "d_chg_rec", "d_now_rb", "d_now_rec"]:
        s = score(MF=model.MARGIN_FEATS + [c])
        print(f"margin + {c:12s} {s[0] - B[0]:+.3f} {s[1] - B[1]:+.3f}", flush=True)
    s = score(MF=model.MARGIN_FEATS + ["d_miss_rb", "d_miss_rec"])
    print(f"margin + both missing  {s[0] - B[0]:+.3f} {s[1] - B[1]:+.3f}")
    for c in ["s_miss_rb", "s_miss_rec", "s_chg_rec", "s_now_rec"]:
        s = score(TF=model.TOTAL_FEATS + [c])
        print(f"total + {c:12s} {s[2] - B[2]:+.3f} {s[3] - B[3]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
