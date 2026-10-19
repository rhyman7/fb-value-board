"""Walk-forward backtest of the college model; prints the summary and saves it as JSON.

    python experiments/cfb_backtest.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import backtest, config as C, data, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402


def main():
    games, tg, qb = data.load_all(verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, info = model.walk_forward(model.build_frame(games, engine))
    graded, dists = backtest.grade_all(games, frame)
    summary = backtest.summarize(frame, graded)
    # Openers are posted before late quarterback news and before a reliable wind forecast:
    # assume last week's starters and a typical wind.
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan, wind=np.nan)
    early, _ = model.walk_forward(model.build_frame(blind, engine))
    summary["openers"] = backtest.opening_report(games, early, dists)
    summary["wind_unders"] = backtest.wind_report(frame)
    summary["coefs"] = info[max(info)]
    out = C.DATA / "backtest.json"
    out.write_text(json.dumps(summary, indent=1, default=float))
    graded.round(4).to_csv(C.DATA / "backtest_bets.csv", index=False)

    for p, a in summary["accuracy"].items():
        print(f"{p:8s} {a['n']:5d} games  margin model {a['margin_mae']['model']:.2f} blend {a['margin_mae']['blend']:.2f} "
              f"line {a['margin_mae']['market']:.2f} | total model {a['total_mae']['model']:.2f} line {a['total_mae']['market']:.2f} "
              f"| winner model {a['winner_pct']['model']:.1%} line {a['winner_pct']['market']:.1%}")
    for src in ("model", "proj"):
        for mk in ("Spread", "Total"):
            for r in summary["records"][src][mk]["rows"]:
                t, h = r["tuning"], r["holdout"]
                print(f"{src:5s} {mk:6s} {r['rule']:14s} tuning {t['w']}-{t['l']}-{t['p']} {t['win_pct'] or 0:.1%} {t['roi'] or 0:+.1%} "
                      f"| holdout {h['w']}-{h['l']}-{h['p']} {h['win_pct'] or 0:.1%} {h['roi'] or 0:+.1%} (se {h['roi_se'] or 0:.1%})")
    print(json.dumps(summary["openers"], indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
