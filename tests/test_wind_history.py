"""The wind-history fetch (experiments/fetch_wind_history.py) and the scripts that read its
cache, run against a fake archive: no network."""
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
        "game_id": ["2015_01_A_GB", "2015_01_B_GB", "2015_01_C_LA", "2015_01_D_NO", "2013_01_E_GB", "2015_02_F_GB", "2015_01_G_GB"],
        "season": [2015, 2015, 2015, 2015, 2013, 2015, 2015], "week": [1, 1, 1, 1, 1, 2, 1],
        "gameday": pd.to_datetime(["2015-09-13"] * 5 + ["2015-09-20", "2015-09-13"]),
        "gametime": ["13:00", "20:20", "16:05", "13:00", "13:00", "13:00", None],
        "home_team": ["GB", "GB", "LA", "NO", "GB", "GB", "GB"], "away_team": ["A", "B", "C", "D", "E", "F", "G"],
        "stadium": ["Lambeau Field", "Lambeau Field", "Los Angeles Memorial Coliseum", "Caesars Superdome", "Lambeau Field", "Lambeau Field", "Lambeau Field"],
        "played": [True, True, True, True, True, True, False], "neutral": [0, 0, 0, 0, 0, 0, 0], "indoor": [0, 0, 0, 1, 0, 0, 0],
        "wind": [10.0, 12.0, 4.0, np.nan, 7.0, 20.0, np.nan], "total_line": [45.5] * 7, "total": [40, 50, 45.5, 30, 44, 41, np.nan],
        "under_odds": [-110] * 7,
    })


def _fake_archive(calls, fail_end="2015-09-20"):
    """A stadium-season range: the hour of day as the speed, every day the same."""
    def fetch(lat, lon, start, end):
        calls.append((lat, lon, start, end))
        if end == fail_end and start != end:
            raise OSError("archive down")
        days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")
        return {"time": [f"{d}T{h:02d}:00" for d in days for h in range(24)],
                "wind_speed_10m": [float(h) for _ in days for h in range(24)], "wind_gusts_10m": [1.5 * h for _ in days for h in range(24)],
                "wind_direction_10m": [90] * (24 * len(days))}
    return fetch


def test_fetch_wind_history_cache(tmp_path):
    fw = _load("fetch_wind_history")
    cache = tmp_path / "wind_history.csv"
    calls = []
    out = fw.run(_games(), cache, fetch=_fake_archive(calls), pause=0, verbose=False)
    # one call per stadium per season: Lambeau 2015 (9/13 to 9/20, fails), Coliseum 2015; the dome, the 2013 game
    # and the unplayed game are not wanted
    assert [(c[2], c[3]) for c in calls] == [("2015-09-13", "2015-09-13"), ("2015-09-13", "2015-09-20")]
    assert sorted(out.game_id) == ["2015_01_C_LA"]
    # the Lambeau call succeeds this time: both days come from one block
    out = fw.run(_games(), cache, fetch=_fake_archive(calls, fail_end="never"), pause=0, verbose=False)
    assert list(out.columns) == fw.COLS and sorted(out.game_id) == ["2015_01_A_GB", "2015_01_B_GB", "2015_01_C_LA", "2015_02_F_GB"]
    a = out.set_index("game_id")
    assert a.loc["2015_01_A_GB", "wind_mph"] == 13 and a.loc["2015_01_B_GB", "wind_mph"] == 20   # kickoff hours 13 and 20
    assert a.loc["2015_01_A_GB", "gust_mph"] == 19.5 and a.loc["2015_01_A_GB", "dir_deg"] == 90
    assert a.loc["2015_01_A_GB", "bearing"] == 0 and a.loc["2015_01_A_GB", "bearing_conf"] == "measured"
    assert a.loc["2015_01_A_GB", "along_mph"] == 0 and a.loc["2015_01_A_GB", "cross_mph"] == 13   # east wind across a north-south field
    assert a.loc["2015_01_C_LA", "bearing"] == 90 and a.loc["2015_01_C_LA", "along_mph"] == 16 and abs(a.loc["2015_01_C_LA", "lat"] - 34.014) < 1e-6
    # resumable: cached games are not fetched again
    calls.clear()
    fetch = _fake_archive(calls)
    out2 = fw.run(_games(), cache, fetch=fetch, pause=0, verbose=False)
    assert len(calls) == 0 and len(out2) == 4 and out2.game_id.is_unique
    assert len(pd.read_csv(cache)) == 4
    # a day's block without a "time" list still works (the forecast shape)
    assert fw.rows_for(fw.wanted(_games()).head(1), {"wind_speed_10m": [3.0] * 24}, "now")[0]["wind_mph"] == 3.0
    # --limit stops after that many calls
    cache2 = tmp_path / "w2.csv"
    calls.clear()
    fw.run(_games(), cache2, fetch=fetch, limit=1, pause=0, verbose=False)
    assert len(calls) == 1


def test_wind_direction_test_reads_cache(tmp_path):
    fw = _load("fetch_wind_history")
    wt = _load("wind_direction_test")
    cache = tmp_path / "wind_history.csv"
    fw.run(_games(), cache, fetch=_fake_archive([], fail_end="never"), pause=0, verbose=False)
    arch = wt.load_archive(cache)
    assert list(arch.columns) == ["game_id", "arch_wind", "gust", "dir", "along", "cross", "bearing_conf"]
    g = wt.attach(_games(), arch)
    assert g.set_index("game_id").loc["2015_01_A_GB", "cross"] == 13
    assert np.isnan(g.set_index("game_id").loc["2015_01_D_NO", "arch_wind"])    # indoors
    wt.compare(g)                                                                # prints, must not raise
    wt.wind_under(g)
    fx = wt.fixture(_games())
    assert set(fx.game_id) == {"2015_01_A_GB", "2015_01_B_GB", "2015_01_C_LA", "2015_02_F_GB"} and fx.gust_mph.ge(fx.wind_mph).all()
    bad = tmp_path / "bad.csv"
    bad.write_text("game_id,wind_mph\nx,1\n")
    try:
        wt.load_archive(bad)
        assert False, "a cache without the direction columns should stop the script"
    except SystemExit:
        pass


def test_wind_forecast_grade_last_before_kickoff():
    wg = _load("wind_forecast_grade")
    log = pd.DataFrame({"game_id": ["2015_01_A_GB"] * 3 + ["2015_01_B_GB"],
                        "fetched": ["2015-09-10 12:00", "2015-09-13 15:00", "2015-09-13 18:00", "2015-09-14 03:00"],   # UTC
                        "wind_mph": [5.0, 11.0, 14.0, 9.0], "gust_mph": [np.nan, 20.0, 22.0, np.nan], "dir_deg": [np.nan, 180, 190, np.nan],
                        "along_mph": [np.nan] * 4, "cross_mph": [np.nan] * 4})
    fc = wg.last_before_kickoff(log, _games()).set_index("game_id")
    assert fc.loc["2015_01_A_GB", "wind_mph"] == 11.0 and not fc.loc["2015_01_A_GB", "after_kickoff"]   # 1 p.m. ET = 17:00 UTC
    assert abs(fc.loc["2015_01_A_GB", "lead_hours"] - 2) < 1e-9
    assert fc.loc["2015_01_B_GB", "wind_mph"] == 9.0 and fc.loc["2015_01_B_GB", "after_kickoff"]         # 8:20 p.m. ET = 00:20 UTC next day
    played = fc.reset_index().merge(_games(), on="game_id")
    lines = wg.under_record(played, "wind_mph")
    assert any("10-14 mph" in ln and "1-0-0" in ln for ln in lines) and any("0-9 mph" in ln and "0-1-0" in ln for ln in lines)


def test_weather_history_rows():
    """Rain and snow are summed over the kickoff hour and the three after it."""
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    import fetch_weather_history as W
    hourly = {"time": [f"2024-12-01T{h:02d}:00" for h in range(24)], "precipitation": [0.0] * 13 + [1.0, 2.0, 0.5, 0.5, 9.0] + [0.0] * 6,
              "snowfall": [0.0] * 24, "temperature_2m": [40.0] * 24}
    g = pd.DataFrame([{"game_id": "x", "season": 2024, "gameday": "2024-12-01", "hour": 13, "home_team": "BUF"}])
    r = W.rows_for(g, hourly, "now")[0]
    assert r["precip_mm"] == 4.0 and r["snow_cm"] == 0.0 and r["temp_f"] == 40.0


def test_forecast_rain_and_snow_in_the_game_window():
    from nflmodel import weather as W
    hourly = {"wind_speed_10m": [5.0] * 24, "precipitation": [0.0] * 13 + [1.0, 0.5, 0.5, 0.25, 9.0] + [0.0] * 6,
              "snowfall": [0.0] * 24}
    r = W.at_hour(hourly, 13)
    assert r["mph"] == 5.0 and r["rain"] == 2.25 and r["snow"] == 0.0
    rows = W.log_rows({"g1": r}, now="2026-11-01 12:00")
    assert rows.loc[0, "precip_mm"] == 2.25 and list(rows.columns) == W.LOG_COLS
