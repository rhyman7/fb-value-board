"""Back-test the wet under on archived FORECASTS (not the rain that fell).

    python experiments/rain_forecast_backtest.py

Reads data/cache/forecast_history.csv (experiments/fetch_forecast_history.py) and grades the
under in outdoor NFL home games the forecast called wet, at the closing total and, from 2024,
the opening total (nfelo). Writes experiments/out/rain_forecast_backtest.json.

The rule, set Nov 23, 2026 before any forecast was fetched (Project doc
claude/plan-rain-under-forecast-backtest-2026-11-23.md):

- Primary: forecast wet one day ahead (weather.WET_RAIN mm of rain or more, or any snow, over the
  kickoff hour and the three after it: the board's own definition), bet the under. 2024 on.
- Reported beside it, not deciding: rain only, 5+ mm of rain, and the other leads (game morning
  from 2022; 2 and 3 days ahead from 2024).
- Support = the primary rule wins more than 52.4% at both the opening and the closing total and
  the final score averages below the line. About 50-60 games: enough to see the idea fail on
  forecasts, not enough to prove it. The Bet sheet rule (a full season of the board's own logged
  forecasts, positive with closing line value above zero) is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from nflmodel import config as C, data, weather  # noqa: E402

CACHE = C.CACHE / "forecast_history.csv"
FELL = C.CACHE / "weather_history.csv"
OUT = HERE / "out" / "rain_forecast_backtest.json"
LEADS = {"game morning": 0, "1 day ahead": 1, "2 days ahead": 2, "3 days ahead": 3}
BREAK_EVEN = 110 / 210


def kinds(rain: pd.Series, snow: pd.Series) -> dict:
    r, s = rain.fillna(0.0), snow.fillna(0.0)
    known = rain.notna() | snow.notna()
    return {"wet (2+ mm or snow)": known & ((r >= weather.WET_RAIN) | (s > weather.WET_SNOW)),
            "rain only (2+ mm, no snow)": known & (r >= weather.WET_RAIN) & (s <= weather.WET_SNOW),
            "rain 5+ mm": known & (r >= 5.0)}


def grade(x: pd.DataFrame, line: str) -> dict:
    x = x[x[line].notna()]
    res = np.sign(x[line] - x.total)
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = w + l
    return {"w": w, "l": l, "push": p, "win_pct": round(w / n, 4) if n else None,
            "return_at_110": round((w * 100 / 110 - l) / n, 4) if n else None,
            "se": round(float(np.sqrt(0.25 / n)), 4) if n else None,
            "avg_total_minus_line": round(float((x.total - x[line]).mean()), 2) if len(x) else None}


def report(frame: pd.DataFrame) -> dict:
    rep = {"games": int(len(frame)), "seasons": sorted(int(s) for s in frame.season.unique()), "leads": {}}
    for name, n in LEADS.items():
        rain, snow = frame.get(f"rain_d{n}"), frame.get(f"snow_d{n}")
        if rain is None or not (rain.notna() | snow.notna()).any():
            continue
        have = frame[rain.notna() | snow.notna()]
        lead = {"games_with_forecast": int(len(have)), "seasons": sorted(int(s) for s in have.season.unique()), "kinds": {}}
        for kind, m in kinds(rain, snow).items():
            x = frame[m]
            k = {"n": int(len(x)), "close": grade(x, "total_line"), "open": grade(x, "open_total"),
                 "line_move_open_to_close": round(float((x.total_line - x.open_total).mean()), 2) if x.open_total.notna().any() else None,
                 "by_season": {int(s): grade(g, "total_line") for s, g in x.groupby("season")}}
            if "fell_wet" in x and x.fell_wet.notna().any():
                k["share_actually_wet"] = round(float(x.fell_wet.dropna().mean()), 3)
            lead["kinds"][kind] = k
        if "fell_wet" in frame:
            f = frame[(rain.notna() | snow.notna()) & frame.fell_wet.notna()]
            fw = kinds(f[f"rain_d{n}"], f[f"snow_d{n}"])["wet (2+ mm or snow)"]
            lead["wet_games_forecast_caught"] = round(float(fw[f.fell_wet == 1].mean()), 3) if (f.fell_wet == 1).any() else None
            lead["fell_wet_but_not_forecast"] = grade(f[(f.fell_wet == 1) & ~fw], "total_line")
        rep["leads"][name] = lead
    p = rep["leads"].get("1 day ahead", {}).get("kinds", {}).get("wet (2+ mm or snow)")
    if p:
        c, o = p["close"], p["open"]
        ok = all(v["win_pct"] is not None and v["win_pct"] > BREAK_EVEN and v["avg_total_minus_line"] < 0 for v in (c, o))
        rep["primary"] = {"rule": "forecast wet 1 day ahead, under", "close": c, "open": o, "supported": bool(ok)}
    return rep


def main() -> None:
    fc = pd.read_csv(CACHE)
    games = data.load_games(refresh=False)
    frame = fc.merge(games[["game_id", "week", "total", "total_line", "played"]], on="game_id", how="inner")
    frame = frame.merge(data.load_openers()[["game_id", "open_total"]], on="game_id", how="left")   # nfelo, totals from 2024
    frame = frame[frame.played & frame.total_line.notna()]
    if FELL.exists():
        a = pd.read_csv(FELL)[["game_id", "precip_mm", "snow_cm"]]
        a["fell_wet"] = np.where(a.precip_mm.isna() & a.snow_cm.isna(), np.nan,
                                 ((a.precip_mm.fillna(0) >= weather.WET_RAIN) | (a.snow_cm.fillna(0) > weather.WET_SNOW)).astype(float))
        frame = frame.merge(a[["game_id", "fell_wet"]], on="game_id", how="left")
    rep = report(frame)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
