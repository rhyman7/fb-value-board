"""Grade the wind under on the forecasts the board actually had, not the wind recorded later.

    python experiments/wind_forecast_grade.py
    python experiments/wind_forecast_grade.py --log data/wind_log.csv --cache data/cache/wind_history.csv

Every forecast the NFL board fetches is appended to data/wind_log.csv. For each played game
this takes the last forecast logged before kickoff, sets it against the wind recorded in the
games file (and the archive speed from fetch_wind_history.py when the cache has the game),
and repeats the wind-under table (FEATURES.md "The wind under") by forecast band. The
sample is small for now; the script prints whatever it has.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import bets as B, config as C, data, odds as O, weather  # noqa: E402

CACHE = C.CACHE / "wind_history.csv"
BANDS = [(0, 10, "0-9 mph"), (10, 15, "10-14 mph"), (15, 20, "15-19 mph"), (20, 999, "20+ mph")]


def last_before_kickoff(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the last logged forecast before kickoff (or the last one at all,
    flagged `after_kickoff`, when every forecast came later)."""
    g = games[["game_id", "gameday", "gametime"]].copy()
    ko = pd.to_datetime(g.gameday.astype(str) + " " + g.gametime.fillna("13:00").astype(str))
    g["kickoff_utc"] = ko.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None)
    l = log.merge(g[["game_id", "kickoff_utc"]], on="game_id", how="inner")
    l["fetched_utc"] = pd.to_datetime(l.fetched)
    l = l.sort_values(["game_id", "fetched_utc"])
    before = l[l.fetched_utc <= l.kickoff_utc].groupby("game_id").tail(1).assign(after_kickoff=False)
    rest = l[~l.game_id.isin(before.game_id)].groupby("game_id").tail(1).assign(after_kickoff=True)
    out = pd.concat([before, rest], ignore_index=True)
    out["lead_hours"] = (out.kickoff_utc - out.fetched_utc).dt.total_seconds() / 3600
    return out


def under_record(f: pd.DataFrame, col: str) -> list[str]:
    """The wind-under table by band of `col`: unders at the closing total."""
    lines = []
    f = f[f.total_line.notna() & f.total.notna() & f[col].notna()]
    under = np.sign(f.total_line - f.total)
    price = f.under_odds.where(f.under_odds.notna(), B.DEFAULT_PRICE) if "under_odds" in f else pd.Series(B.DEFAULT_PRICE, index=f.index)
    dec = O.american_to_decimal(np.asarray(price, float))
    profit = np.where(under > 0, dec - 1, np.where(under < 0, -1.0, 0.0))
    for lo, hi, label in BANDS + [(10, 20, "10-19 mph (the band)")]:
        m = ((f[col] >= lo) & (f[col] < hi)).to_numpy()
        n = int(m.sum())
        if not n:
            lines.append(f"      {label:22s} no games yet")
            continue
        w, l, p = int((under[m] > 0).sum()), int((under[m] < 0).sum()), int((under[m] == 0).sum())
        pct = f"{100 * w / (w + l):.1f}%" if w + l else "n/a"
        se = f" ± {100 * profit[m].std(ddof=1) / np.sqrt(n):.1f}%" if n > 1 else ""
        lines.append(f"      {label:22s} {w}-{l}-{p}  {pct:>6}  return {100 * profit[m].mean():+.1f}%{se}  close minus final {(f.total_line - f.total)[m].mean():+.1f}")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", default=str(weather.LOG))
    ap.add_argument("--cache", default=str(CACHE), help="archive cache from fetch_wind_history.py, if any")
    a = ap.parse_args(argv)
    log = weather.read_log(a.log)
    if log.empty:
        print(f"{a.log}: no forecasts logged yet")
        return
    games = data.load_games(refresh=False)
    print(f"wind log {a.log}: {len(log)} forecasts for {log.game_id.nunique()} games, {log.fetched.min()} to {log.fetched.max()}")
    fc = last_before_kickoff(log, games)
    played = fc.merge(games[games.played][["game_id", "season", "week", "home_team", "away_team", "indoor", "wind", "total_line", "total", "under_odds"]],
                      on="game_id", how="inner")
    print(f"  played: {len(played)} games with a forecast ({int(played.after_kickoff.sum())} only logged after kickoff); "
          f"median lead {played.lead_hours.median():.0f} h before kickoff" if len(played) else "  no logged game has been played yet")
    if played.empty:
        return
    played = played.sort_values(["season", "week", "game_id"])

    # forecast against what was recorded
    rec = played[played.wind.notna()]
    print(f"\n(1) forecast against the recorded wind: {len(rec)} of {len(played)} played games have a recorded wind in the games file so far")
    if len(rec) >= 3:
        d = rec.wind_mph - rec.wind
        print(f"    correlation {np.corrcoef(rec.wind, rec.wind_mph)[0, 1]:.3f}, forecast minus recorded {d.mean():+.2f} mph, average absolute gap {d.abs().mean():.2f}")
        band = lambda w: pd.cut(w, [-1, 10, 15, 20, 999], labels=["0-9", "10-14", "15-19", "20+"])
        both = (band(rec.wind_mph).astype(str) == band(rec.wind).astype(str))
        print(f"    same band as recorded: {int(both.sum())} of {len(rec)}")
    elif len(rec):
        print("    too few to summarise; the games:")
    cache = Path(a.cache)
    if cache.exists():
        arch = pd.read_csv(cache)[["game_id", "wind_mph", "gust_mph", "dir_deg"]].rename(columns={"wind_mph": "arch_mph", "gust_mph": "arch_gust", "dir_deg": "arch_dir"})
        pa = played.merge(arch, on="game_id", how="inner")
        print(f"    archive speed (data/cache/wind_history.csv) for {len(pa)} of them" + (f": correlation {np.corrcoef(pa.arch_mph, pa.wind_mph)[0, 1]:.3f}, forecast minus archive {(pa.wind_mph - pa.arch_mph).mean():+.2f} mph" if len(pa) >= 3 else ""))
    cols = ["game_id", "fetched", "lead_hours", "wind_mph", "gust_mph", "dir_deg", "wind", "total_line", "total"]
    show = played[cols].copy()
    show["lead_hours"] = show.lead_hours.round(0)
    print("    " + show.rename(columns={"wind": "recorded"}).to_string(index=False).replace("\n", "\n    "))

    print("\n(2) unders at the closing total by the FORECAST wind (what the board's tag used):")
    for line in under_record(played, "wind_mph"):
        print(line)
    if len(rec):
        print("    by the RECORDED wind, the same games:")
        for line in under_record(rec, "wind"):
            print(line)
    gust = played[played.gust_mph.notna()]
    if len(gust):
        print("    by the forecast GUSTS:")
        for line in under_record(gust, "gust_mph"):
            print(line)
    print("\nSmall sample: the published table (FEATURES.md) is 2014 on with recorded wind. Keep logging; re-run after the season.")


if __name__ == "__main__":
    main()
