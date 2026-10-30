"""Where the college model's edge lives: the walk-forward backtest split into segments.

    python experiments/cfb_segments.py

Reuses the pipeline's walk-forward run (cfbmodel.model.walk_forward, the same frame
cfbmodel.backtest grades) and splits it, for the holdout (2021 on) and separately for the
tuning seasons (2017-2019), by:
  (a) conference tier of the two teams: P4 (SEC, Big Ten, Big 12, ACC, Notre Dame; the
      Pac-12 through 2023) against P4, G5 against G5, or mixed
  (b) day of the week of kickoff (Eastern): Saturday, midweek (Tuesday to Friday), Sunday/Monday
  (c) week of the season: 0-5, 6-10, 11+ (regular season) and bowls

Per segment: games, average miss of the raw model and of the closing line on the margin and
the total, and the record when the raw model disagrees with the line by 2+ and 3+ points
(spreads) or 3+ (totals): at the closing line, and at the opener (2021 on, the only seasons
with openers; graded the way cfbmodel.backtest.opening_report does, from a run that assumes
last week's quarterbacks and a typical wind and leaves out short-week games). Spreads and
totals are graded at -110 at the line rounded to the nearest half point, with a standard
error on the return. Conference per team per season comes from the cfbfastR teams file.

Writes experiments/out/cfb_segments.md and prints it. The walk-forward takes about 20 s on
the cached data; data/cfb/backtest_bets.csv is not needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import backtest, bets as B, config as C, data, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402
from nflmodel import odds as O  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "cfb_segments.md"
P4 = {"SEC", "Big Ten", "Big 12", "ACC"}
PAC12_P4_THROUGH = 2023
PERIODS = {"Holdout (2021 on)": lambda s: s >= C.HOLDOUT_FROM, "Tuning seasons (2017-2019)": lambda s: s.between(2017, 2019)}
DEC = O.american_to_decimal(B.DEFAULT_PRICE)


# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------

def conferences(seasons) -> dict[tuple[int, str], str]:
    """(season, school) -> conference short name, from the cfbfastR teams file (cached under data/cfb/raw)."""
    out = {}
    C.RAW.mkdir(parents=True, exist_ok=True)
    for s in sorted(set(int(x) for x in seasons)):
        path = C.RAW / f"conferences_{s}.csv"
        if not path.exists():
            t = data._get("teams", s)
            t = t[t["classification"] == "fbs"]
            pd.DataFrame({"school": t["school"], "conference": t["conference_short_name"].fillna(t["cfbd_conference"])}).to_csv(path, index=False)
        t = pd.read_csv(path)
        out.update({(s, r.school): str(r.conference) for r in t.itertuples()})
    return out


def tier(season: int, school: str, conf: str | None) -> str:
    if school == "Notre Dame":
        return "P4"
    if conf in P4 or (conf == "Pac-12" and season <= PAC12_P4_THROUGH):
        return "P4"
    return "G5"


def label(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    confs = conferences(f.season.unique())
    ht = [tier(s, h, confs.get((s, h))) for s, h in zip(f.season, f.home_team)]
    at = [tier(s, a, confs.get((s, a))) for s, a in zip(f.season, f.away_team)]
    f["seg_tier"] = ["P4 vs P4" if h == a == "P4" else "G5 vs G5" if h == a == "G5" else "Mixed (P4 vs G5)" for h, a in zip(ht, at)]
    dow = pd.to_datetime(f.kickoff, utc=True).dt.tz_convert("America/New_York").dt.dayofweek   # Monday = 0
    f["seg_day"] = np.where(dow == 5, "Saturday", np.where(dow.between(1, 4), "Midweek (Tue-Fri)", "Sunday/Monday"))
    wk = f.week.astype(int)
    f["seg_week"] = np.where(f.postseason == 1, "Bowls", np.where(wk <= 5, "Weeks 0-5", np.where(wk <= 10, "Weeks 6-10", "Weeks 11+")))
    return f


SEGMENTS = [("Conference tier", "seg_tier", ["P4 vs P4", "Mixed (P4 vs G5)", "G5 vs G5"]),
            ("Day of week", "seg_day", ["Saturday", "Midweek (Tue-Fri)", "Sunday/Monday"]),
            ("Week of season", "seg_week", ["Weeks 0-5", "Weeks 6-10", "Weeks 11+", "Bowls"])]


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def graded(f: pd.DataFrame, mcol: str, line: str, actual: str) -> pd.DataFrame:
    """One row per game with a line: the raw model's side against `line`, its cover and profit at -110."""
    x = f[f[line].notna() & f[actual].notna()].copy()
    x["d"] = x[mcol] - x[line]
    pick = np.sign(x["d"])
    x["cover"] = np.sign(x[actual] - B.half(x[line])) * pick
    x["profit"] = np.where(x.cover > 0, DEC - 1, np.where(x.cover < 0, -1.0, 0.0))
    return x[x.d != 0]


def record(y: pd.DataFrame) -> str:
    w, l = int((y.cover > 0).sum()), int((y.cover < 0).sum())
    n = len(y)
    if n < 2 or w + l == 0:
        return f"{w}-{l}" if n else "-"
    roi, se = y.profit.mean(), y.profit.std(ddof=1) / np.sqrt(n)
    return f"{w}-{l} {w / (w + l):.1%} {roi:+.1%} ± {se:.1%}"


def mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))


def tables(close: pd.DataFrame, blind: pd.DataFrame | None, period: str) -> list[str]:
    """Markdown tables for one period: spreads then totals."""
    lines = [f"### {period}", ""]
    sp_close = graded(close, "model_margin", "spread_line", "result")
    tt_close = graded(close, "model_total", "total_line", "total")
    sp_open = graded(blind, "model_margin", "open_spread", "result") if blind is not None and len(blind) else None
    tt_open = graded(blind, "model_total", "open_total", "total") if blind is not None and len(blind) else None
    has_open = sp_open is not None and len(sp_open) > 0

    head = ["Segment", "Games", "Miss: model / close", "vs close, 2+", "vs close, 3+"]
    if has_open:
        head += ["vs opener, 2+", "vs opener, 3+"]
    lines += ["**Spreads** (W-L, win rate, return per bet ± s.e., at -110)", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for title, col, order in SEGMENTS:
        for seg in order:
            x = close[close[col] == seg]
            if not len(x):
                continue
            row = [f"{title}: {seg}" if seg == order[0] else seg, str(len(x)),
                   f"{mae(x.model_margin, x.result):.2f} / {mae(x.spread_line, x.result):.2f}"]
            c = sp_close[sp_close[col] == seg]
            row += [record(c[c.d.abs() >= 2]), record(c[c.d.abs() >= 3])]
            if has_open:
                o = sp_open[sp_open[col] == seg]
                row += [record(o[o.d.abs() >= 2]), record(o[o.d.abs() >= 3])]
            lines.append("| " + " | ".join(row) + " |")
    row = ["All games", str(len(close)), f"{mae(close.model_margin, close.result):.2f} / {mae(close.spread_line, close.result):.2f}",
           record(sp_close[sp_close.d.abs() >= 2]), record(sp_close[sp_close.d.abs() >= 3])]
    if has_open:
        row += [record(sp_open[sp_open.d.abs() >= 2]), record(sp_open[sp_open.d.abs() >= 3])]
    lines += ["| " + " | ".join(f"**{v}**" for v in row) + " |", ""]

    head = ["Segment", "Games", "Miss: model / close", "vs close, 3+"]
    if has_open:
        head += ["vs opener, 3+"]
    lines += ["**Totals** (W-L, win rate, return per bet ± s.e., at -110)", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for title, col, order in SEGMENTS:
        for seg in order:
            x = close[close[col] == seg]
            if not len(x):
                continue
            row = [f"{title}: {seg}" if seg == order[0] else seg, str(len(x)),
                   f"{mae(x.model_total, x.total):.2f} / {mae(x.total_line, x.total):.2f}"]
            c = tt_close[tt_close[col] == seg]
            row += [record(c[c.d.abs() >= 3])]
            if has_open:
                o = tt_open[tt_open[col] == seg]
                row += [record(o[o.d.abs() >= 3])]
            lines.append("| " + " | ".join(row) + " |")
    row = ["All games", str(len(close)), f"{mae(close.model_total, close.total):.2f} / {mae(close.total_line, close.total):.2f}",
           record(tt_close[tt_close.d.abs() >= 3])]
    if has_open:
        row += [record(tt_open[tt_open.d.abs() >= 3])]
    lines += ["| " + " | ".join(f"**{v}**" for v in row) + " |", ""]
    if not has_open:
        lines += ["Openers are in the data from 2021 only, so this period has no opener columns.", ""]
    return lines


def main():
    games, tg, qb = data.load_all(verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, engine))
    # openers: what the model knew when they were posted (last week's starters, a typical wind)
    blind_games = games.assign(home_qb_id=np.nan, away_qb_id=np.nan, wind=np.nan)
    early, _ = model.walk_forward(model.build_frame(blind_games, engine))
    short = backtest._short_week(games)

    played = lambda f: f[f.played & f.spread_line.notna() & f.total_line.notna()]
    close = label(played(frame))
    blind = label(played(early))
    blind = blind[blind.open_spread.notna() & ~blind.game_id.map(short).fillna(False).to_numpy(bool)]

    last = int(close.season.max())
    md = ["# College board: where the edge lives", "",
          f"Walk-forward backtest, FBS-vs-FBS games with lines, 2017-{last} (`python experiments/cfb_segments.py`). "
          "Miss is the average absolute error of the raw model and of the closing line. The records bet the raw model's side "
          "wherever it is 2+ or 3+ points off the line (spreads) or 3+ off the total, at -110, at the line rounded to the "
          "half point. Closing-line records use the pipeline's run; opener records use a run that assumes last week's "
          "quarterbacks and a typical wind and leave out short-week games, as the board's opener report does. "
          "P4 is the SEC, Big Ten, Big 12, ACC and Notre Dame, plus the Pac-12 through 2023. "
          "Days are Eastern. Bowls are listed separately from weeks 11+.", ""]
    for period, sel in PERIODS.items():
        c = close[sel(close.season)]
        b = blind[sel(blind.season)]
        md += tables(c, b, period)
    text = "\n".join(md)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
