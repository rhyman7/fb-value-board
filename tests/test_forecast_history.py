"""The forecast-history fetch (experiments/fetch_forecast_history.py) and the back-test that reads
its cache, run against fake archives: no network."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _games():
    return pd.DataFrame({
        "game_id": ["2023_01_A_GB", "2024_01_B_GB", "2024_02_C_GB", "2021_01_D_GB", "2024_01_E_NO"],
        "season": [2023, 2024, 2024, 2021, 2024], "week": [1, 1, 2, 1, 1],
        "gameday": pd.to_datetime(["2023-09-10", "2024-09-08", "2024-09-15", "2021-09-12", "2024-09-08"]),
        "gametime": ["13:00", "13:00", "20:20", "13:00", "13:00"],
        "home_team": ["GB", "GB", "GB", "GB", "NO"], "away_team": list("ABCDE"),
        "stadium": ["Lambeau Field"] * 4 + ["Caesars Superdome"],
        "played": [True] * 5, "neutral": [0] * 5, "indoor": [0, 0, 0, 0, 1],
    })


def _block(start, end, keys, value):
    days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")
    out = {"time": [f"{d}T{h:02d}:00" for d in days for h in range(24)]}
    for k in keys:
        out[k] = [value(k, h) for _ in days for h in range(24)]
    return out


def test_fetch_forecast_history(tmp_path):
    ff = _load("fetch_forecast_history")
    cache = tmp_path / "forecast_history.csv"
    calls = []

    def now(lat, lon, start, end):
        calls.append(("now", start, end))
        return _block(start, end, ["precipitation", "snowfall"], lambda k, h: 1.0 if k == "precipitation" else 0.0)

    def lead(lat, lon, start, end, fail=[True]):
        calls.append(("lead", start, end))
        if fail[0]:
            fail[0] = False
            raise OSError("archive down")
        return _block(start, end, [f"{v}_previous_day{n}" for n in ff.LEADS for v in ("precipitation", "snowfall")],
                      lambda k, h: (float(k[-1]) if k.startswith("precip") else 0.5) if h < 23 else None)

    out = ff.run(_games(), cache, now=now, lead=lead, pause=0, verbose=False)
    # 2023 needs only the game-morning archive; 2024 also needs the previous runs, which fail, so nothing
    # is written for it; the dome and the 2021 game are not wanted
    assert calls == [("now", "2023-09-10", "2023-09-10"), ("now", "2024-09-08", "2024-09-15"), ("lead", "2024-09-08", "2024-09-15")]
    assert list(out.game_id) == ["2023_01_A_GB"] and out.rain_d0[0] == 4.0 and pd.isna(out.rain_d1[0])
    out = ff.run(_games(), cache, now=now, lead=lead, pause=0, verbose=False).set_index("game_id")
    assert len(calls) == 5 and sorted(out.index) == ["2023_01_A_GB", "2024_01_B_GB", "2024_02_C_GB"]
    b, c = out.loc["2024_01_B_GB"], out.loc["2024_02_C_GB"]
    assert (b.rain_d1, b.rain_d2, b.rain_d3, b.snow_d1) == (4.0, 8.0, 12.0, 2.0)
    assert c.hour == 20 and c.rain_d1 == 3.0           # 20:00-22:00 counted, the missing 23:00 skipped
    assert list(out.columns) == ff.COLS[1:]
    assert ff.run(_games(), cache, now=now, lead=lead, pause=0, verbose=False).shape[0] == 3 and len(calls) == 5


def test_rain_forecast_backtest_report():
    bt = _load("rain_forecast_backtest")
    n = 8
    frame = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)], "season": [2024] * 4 + [2025] * 4,
        "rain_d0": [3.0, 0.0, 6.0, 2.5, 0.0, 9.0, 0.0, 1.0], "snow_d0": [0.0] * n,
        "rain_d1": [2.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0, 0.0], "snow_d1": [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        "rain_d2": [np.nan] * n, "snow_d2": [np.nan] * n, "rain_d3": [np.nan] * n, "snow_d3": [np.nan] * n,
        "total_line": [44.5] * n, "open_total": [46.5, 46.5, 45.5, 44.5, 44.5, np.nan, 44.5, 44.5],
        "total": [40, 50, 45, 30, 44, 41, 60, 44.5], "fell_wet": [1, 0, 1, 1, 0, 1, 1, np.nan],
    })
    rep = bt.report(frame)
    assert list(rep["leads"]) == ["game morning", "1 day ahead"]            # no 2- or 3-day forecasts in the frame
    k = rep["leads"]["1 day ahead"]["kinds"]
    wet = k["wet (2+ mm or snow)"]
    assert wet["n"] == 4 and (wet["close"]["w"], wet["close"]["l"]) == (3, 1)   # g0, g3, g5 under; g2 over
    assert (wet["open"]["w"], wet["open"]["l"]) == (3, 0)                    # g5 has no opener; g2 is under its opener
    assert wet["line_move_open_to_close"] == -1.0 and wet["share_actually_wet"] == 1.0
    assert k["rain only (2+ mm, no snow)"]["n"] == 3 and k["rain 5+ mm"]["n"] == 2
    assert rep["leads"]["1 day ahead"]["wet_games_forecast_caught"] == 0.8   # g6 fell wet with a dry forecast
    assert rep["leads"]["1 day ahead"]["fell_wet_but_not_forecast"]["l"] == 1
    assert rep["primary"]["supported"] is True and wet["by_season"][2024]["w"] == 2
