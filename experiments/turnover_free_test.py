"""Do pass and rush ratings with turnovers taken out predict better?

    python experiments/turnover_free_test.py

ELWAY (Silver Bulletin) backs turnovers out of its efficiency ratings, on the view that they are
more random than the rest of a team's play. The board's pass and rush ratings count every play's
EPA, turnovers included (clipped at +/-4.5 NFL, 5 college). Here the same ratings are rebuilt
without the turnover plays (NFL: interception or fumble lost; college: cfbfastR's turnover_vec),
over the same non-garbage plays, with the same ridge penalties as the originals:

  pnt   pass EPA per non-turnover dropback
  rnt   rush EPA per non-turnover carry

Tested both ways: added to the models next to the originals, and swapped in for them
(d_pass / s_pass -> d_pnt / s_pnt, d_rush / s_rush -> d_rnt / s_rnt). The points rating and
the quarterback ratings are unchanged. The rejected fumble-luck test (FEATURES.md) is related.

Rule, set before running (Nov 2, 2026): kept only if the change lowers the raw model's average
miss in both the tuning seasons and the holdout, the holdout by at least 2 paired standard errors.

Writes experiments/out/turnover_free_test.json. The NFL counts come from data/cache/team_extra_<season>.csv
(nflmodel/extra.py `turnover_free`, Nov 4, 2026; before that the script cached nfl_turnover_free_<season>.csv).
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
from cfb_downs_havoc_test import add_metrics, load_extra, paired  # noqa: E402

OUT = HERE / "out" / "turnover_free_test.json"


def nfl_extra(tg: pd.DataFrame) -> pd.DataFrame:
    """The turnover-free play counts and EPA per team-game, keyed by side. From Nov 4, 2026 they are
    columns of the board's own extra stats (data/cache/team_extra_<season>.csv, nflmodel/extra.py
    `turnover_free`), which the Update board Action refreshes; the old nfl_turnover_free_<season>.csv
    caches were folded into them."""
    from nflmodel import extra
    te, _ = extra.load_extra(verbose=False)
    cols = ["pnt_n", "pnt_epa", "rnt_n", "rnt_epa"]
    j = tg[["game_id", "team", "side"]].merge(te[["game_id", "team"] + cols], on=["game_id", "team"], how="left")
    return j.drop(columns=["team"])


def run(league: str) -> dict:
    if league == "nfl":
        from nflmodel import config as C, data, extra, model
        from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
        games, tg, qb = data.load_all(refresh=False, verbose=False)
        games = games[games.season <= 2025]
        tg = tg[tg.season <= 2025]
        te, qe = extra.load_extra(verbose=False)
        eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
        ex = nfl_extra(tg)
        hold = C.HOLDOUT_FROM
    else:
        from cfbmodel import config as C, data, model
        from cfbmodel.ratings import CFBRatingEngine
        games, tg, qb = data.load_all(refresh=False, verbose=False)
        games = games[games.season <= 2025]
        eng = CFBRatingEngine(tg, qb, games=games)
        ex = load_extra(games)
        hold = C.HOLDOUT_FROM
    ridge = {"pnt": eng.ridge["pass"], "rnt": eng.ridge["rush"]}
    add_metrics(eng, tg, ex, ("pnt", "rnt"), ridge)
    f = model.build_frame(games, eng)
    f["act_margin"], f["act_total"] = f.home_score - f.away_score, f.home_score + f.away_score
    tune = [2017, 2018, 2019] if league == "cfb" else list(range(int(f.season.min()), hold))
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    swap = {"d_pass": "d_pnt", "d_rush": "d_rnt", "s_pass": "s_pnt", "s_rush": "s_rnt"}

    def out(mf, tf):
        o, _ = model.walk_forward(f, margin_feats=mf, total_feats=tf)
        return o[o.played & o.model_margin.notna()]

    base = out(MF, TF)
    res = {"holdout_from": hold}
    for name, mf, tf, pred, act in (
            ("margin + turnover-free added", MF + ["d_pnt", "d_rnt"], TF, "model_margin", "act_margin"),
            ("margin, swapped in", [swap.get(x, x) for x in MF], TF, "model_margin", "act_margin"),
            ("total + turnover-free added", MF, TF + ["s_pnt", "s_rnt"], "model_total", "act_total"),
            ("total, swapped in", MF, [swap.get(x, x) for x in TF], "model_total", "act_total")):
        res[name] = paired(base, out(mf, tf), pred, act, tune, hold)
        print(league, name, res[name], flush=True)
    return res


def main():
    res = {lg: run(lg) for lg in ("nfl", "cfb")}
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
