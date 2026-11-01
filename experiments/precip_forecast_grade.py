"""Grade the wet under on the rain and snow forecasts the board actually had.

    python experiments/precip_forecast_grade.py

From Nov 1, 2026 every NFL forecast in data/wind_log.csv also carries the rain (mm) and snow (cm)
forecast for the game window (weather.GAME_HOURS hours from kickoff). For each played game this
takes the last forecast logged before kickoff and grades the under at the closing total when it
said wet (2+ mm of rain or any snow), next to the same games' recorded weather from
data/cache/weather_history.csv where the archive has them. The archive test (FEATURES.md, "Rain
and snow") used the weather that fell; this is the version that can be bet. The rule, set now:
the wet under joins the Bet sheet only if, after a full season of logged forecasts, the forecast
version is positive with closing line value above zero (as the wind under must be).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import config as C, data, weather  # noqa: E402
from wind_forecast_grade import last_before_kickoff  # noqa: E402


def main() -> None:
    log = weather.read_log()
    log = log[log["precip_mm"].notna() | log["snow_cm"].notna()]
    games = data.load_games()
    if log.empty:
        print("no rain or snow forecasts logged yet")
        return
    f = last_before_kickoff(log, games)
    f = f[~f.after_kickoff].merge(games[["game_id", "season", "week", "total", "total_line", "played"]], on="game_id")
    f = f[f.played & f.total_line.notna()]
    wet = (f.precip_mm.fillna(0) >= weather.WET_RAIN) | (f.snow_cm.fillna(0) > weather.WET_SNOW)
    x = f[wet]
    res = np.sign(x.total_line - x.total)
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    print(f"games with a forecast before kickoff: {len(f)}; forecast wet: {len(x)}")
    print(f"wet unders at the closing total: {w}-{l}-{p}, return at -110 {((w * 100 / 110 - l) / max(w + l, 1)):+.1%}")
    cache = C.CACHE / "weather_history.csv"
    if cache.exists() and len(x):
        a = pd.read_csv(cache)[["game_id", "precip_mm", "snow_cm"]].rename(columns={"precip_mm": "rain_fell", "snow_cm": "snow_fell"})
        m = x.merge(a, on="game_id", how="left")
        print(m[["game_id", "precip_mm", "rain_fell", "snow_cm", "snow_fell", "total_line", "total"]].to_string(index=False))


if __name__ == "__main__":
    main()
