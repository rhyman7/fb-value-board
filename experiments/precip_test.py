"""Rain and snow against the total, once data/cache/weather_history.csv is filled.

    python experiments/precip_test.py

Two checks, on outdoor games since 2014 (the cache's range; other games count as dry):

  1. As a feature of the board's total model (feature_test.py's walk-forward): rain in mm and
     snow in cm during the game window, each added alone, and a wet flag (2+ mm or any snow).
     Kept only if it lowers the raw model's average miss on the tuning seasons (to 2018) and
     holds on the holdout (2019 on).
  2. Against the closing total: unders in games with 2+ mm of rain or any snow, by period.
     The market knows the forecast; what matters is whether it priced it fully.

The backtest uses the weather that fell, not a forecast, so it flatters both checks, as the
recorded wind did for the wind under (README, "Wind and totals").

Writes experiments/out/precip_test.json.
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
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
CACHE = C.CACHE / "weather_history.csv"
OUT = HERE / "out" / "precip_test.json"


def main():
    if not CACHE.exists():
        print(f"{CACHE} is missing: run the Weather history Action first")
        return
    wx = pd.read_csv(CACHE)[["game_id", "precip_mm", "snow_cm", "temp_f"]]
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame = model.build_frame(games, eng).merge(wx, on="game_id", how="left")
    frame["rain"] = frame["precip_mm"].fillna(0.0)
    frame["snow"] = frame["snow_cm"].fillna(0.0)
    frame["wet"] = ((frame["rain"] >= 2) | (frame["snow"] > 0)).astype(float)

    def miss(feats):
        out, _ = model.walk_forward(frame, total_feats=feats)
        o = out[out.played & out.total_line.notna()]
        return {p: float(np.mean(np.abs(x.model_total - x.total)))
                for p, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM]))}, o

    base, o = miss(list(model.TOTAL_FEATS))
    rep = {"base": base, "features": {}}
    for f in ("rain", "snow", "wet"):
        r, _ = miss(list(model.TOTAL_FEATS) + [f])
        rep["features"][f] = {p: round(r[p] - base[p], 4) for p in r}
    g = frame[frame.played & frame.total_line.notna() & (frame.season >= 2014)].copy()
    g["under_won"] = np.sign(g.total_line - g.total)
    for name, m in (("wet (2+ mm or snow)", g.wet == 1), ("rain 5+ mm", g.rain >= 5), ("snow", g.snow > 0)):
        for per, pm in (("tune", g.season < C.HOLDOUT_FROM), ("hold", g.season >= C.HOLDOUT_FROM)):
            x = g[m & pm]
            w, l = int((x.under_won > 0).sum()), int((x.under_won < 0).sum())
            ret = (w * (100 / 110) - l) / max(w + l, 1)
            rep.setdefault("unders", {}).setdefault(name, {})[per] = {"w": w, "l": l, "return_at_110": round(ret, 4),
                                                                      "avg_total_minus_line": round(float((x.total - x.total_line).mean()), 2) if len(x) else None}
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
