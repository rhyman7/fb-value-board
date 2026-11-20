"""College opening spreads back to 2015, from The Prediction Tracker's files (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_openers_2015_test.py

T1 of the Project doc claude/plan-batch-tests-2026-11-20.md (written before the run). A diagnostic:
nothing on the board changes. Committed before its first run.

The board has no college openers before 2021, so its opener record could not be checked on
2017-19. data/cfb/outside/ncaa<season>.csv carries an opening spread (lineopen) for every season.

1. Validation, 2021-25: the file's opener against the board's own open_spread. Needs correlation
   >= 0.97 and an average gap under 1 point, or the test stops.
2. The bet-time raw model (last week's quarterbacks, a typical wind; short-week games out) 2+ and
   3+ points off the opener, graded at the opener at -110: 2017-19, 2020, and 2021-25 from the same
   file for comparison.

Reading, fixed before the run: 2017-19 under 0% at both 2+ and 3+ = no sign of a college opener
spread edge in either period. Above 0% at both = a second period behind the opener edge, a lead for
a real-price test. Otherwise mixed.
Also reported: weeks 1-4 against 5+ at the opener for 2017-19 (the piece the early-season test of
the same day could not check; it does not change that verdict), and closing line value.

Writes experiments/out/cfb_openers_2015_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import cfb_outside_ratings_test as T  # noqa: E402  (file reading and the name match)

OUT = HERE / "out" / "cfb_openers_2015_test.json"


def pt_openers(games: pd.DataFrame) -> pd.DataFrame:
    """game_id, pt_open (home-margin terms), matched as cfb_outside_ratings_test.load_outside does."""
    from collections import Counter
    files = {s: T._read(T.OUTSIDE / f"ncaa{s}.csv") for s in sorted(games.season.unique()) if (T.OUTSIDE / f"ncaa{s}.csv").exists()}
    votes = defaultdict(Counter)
    for s, d in files.items():
        key = defaultdict(list)
        for r in games[(games.season == s) & games.played].itertuples():
            key[(int(r.home_score), int(r.away_score))].append((r.home_team, r.away_team))
        for r in d.dropna(subset=["hscore", "vscore"]).itertuples():
            for hs, vs, flip in ((r.hscore, r.vscore, False), (r.vscore, r.hscore, True)):
                c = key.get((int(hs), int(vs)), [])
                if len(c) == 1:
                    h, a = c[0] if not flip else c[0][::-1]
                    votes[r.home][h] += 1
                    votes[r.road][a] += 1
    name = {k: c.most_common(1)[0][0] for k, c in votes.items()
            if c.most_common(1)[0][1] >= 3 and c.most_common(1)[0][1] / sum(c.values()) >= 0.6}
    name.update(T.MANUAL)
    out = []
    for s, d in files.items():
        by = defaultdict(list)
        for r in games[(games.season == s) & games.played].itertuples():
            by[(r.home_team, r.away_team)].append(r)
        used = set()
        for r in d.dropna(subset=["hscore", "vscore"]).itertuples():
            h0, a0 = name.get(r.home), name.get(r.road)
            for h, a, sign in ((h0, a0, 1.0), (a0, h0, -1.0)):
                cand = [c for c in by.get((h, a), []) if c.game_id not in used
                        and {int(c.home_score), int(c.away_score)} == {int(r.hscore), int(r.vscore)}]
                if cand:
                    used.add(cand[0].game_id)
                    out.append({"game_id": cand[0].game_id, "pt_open": sign * r.lineopen})
                    break
    return pd.DataFrame(out)


def graded(f: pd.DataFrame, line: str) -> pd.DataFrame:
    from cfbmodel import bets as B
    from nflmodel import odds as O
    dec = O.american_to_decimal(B.DEFAULT_PRICE)
    x = f[f.played & f[line].notna() & f["model_margin"].notna() & f["spread_line"].notna() & (f.fbs_game == 1)].copy()
    x["d"] = x["model_margin"] - x[line]
    pick = np.sign(x["d"])
    x["cover"] = np.sign(x["result"] - B.half(x[line])) * pick
    x["profit"] = np.where(x.cover > 0, dec - 1, np.where(x.cover < 0, -1.0, 0.0))
    x["clv"] = (x["spread_line"] - x[line]) * pick
    return x[x.d != 0]


def stats(y: pd.DataFrame) -> dict:
    n, w, l = int(len(y)), int((y.cover > 0).sum()), int((y.cover < 0).sum())
    return {"n": n, "w": w, "l": l, "win": w / (w + l) if w + l else None,
            "ret": float(y.profit.mean()) if n else None,
            "se": float(y.profit.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            "clv": float(y.clv.mean()) if n else None}


def main() -> None:
    from cfbmodel import backtest, data, extra, model
    from cfbmodel.pipeline import CFBRatingEngine, PFFRET
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games["wind_source"] = np.where(games["played"], "recorded", "")
    inputs = extra.load_inputs(games, refresh_current=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games, extra=inputs["rated"])
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan, wind=np.nan)
    early, _ = model.walk_forward(PFFRET.add(model.build_frame(blind, engine)))
    short = backtest._short_week(games)
    early = early[~early["game_id"].map(short).fillna(False).to_numpy(bool)]
    early = early.merge(pt_openers(games), on="game_id", how="left")

    v = early[early.season.between(2021, 2025) & early.open_spread.notna() & early.pt_open.notna()]
    val = {"n": int(len(v)), "corr": float(v.open_spread.corr(v.pt_open)), "mean_abs_gap": float((v.open_spread - v.pt_open).abs().mean()),
           "share_within_half": float(((v.open_spread - v.pt_open).abs() <= 0.5).mean()),
           "coverage_2017_19": float(early[early.season.between(2017, 2019) & early.spread_line.notna() & early.played].pt_open.notna().mean())}
    res = {"validation": val}
    ok = val["corr"] >= 0.97 and val["mean_abs_gap"] < 1.0
    res["validation_ok"] = bool(ok)
    if ok:
        gp, gb = graded(early, "pt_open"), graded(early, "open_spread")
        res["periods"] = {}
        for name, lo, hi in (("2017-19", 2017, 2019), ("2020", 2020, 2020), ("2021-25", 2021, 2025), ("2026", 2026, 2026)):
            p = gp[gp.season.between(lo, hi)]
            r = {f"{t}+": stats(p[p.d.abs() >= t]) for t in (1, 2, 3)}
            r["2+ weeks 1-4"] = stats(p[(p.d.abs() >= 2) & (p.week <= 4)])
            r["2+ weeks 5+"] = stats(p[(p.d.abs() >= 2) & (p.week > 4)])
            r["1+ weeks 1-4"] = stats(p[(p.d.abs() >= 1) & (p.week <= 4)])
            r["1+ weeks 5+"] = stats(p[(p.d.abs() >= 1) & (p.week > 4)])
            res["periods"][name] = r
        b = gb[gb.season.between(2021, 2025)]
        res["board_openers_2021_25"] = {f"{t}+": stats(b[b.d.abs() >= t]) for t in (2, 3)}
        a, c = res["periods"]["2017-19"]["2+"], res["periods"]["2017-19"]["3+"]
        if a["ret"] < 0 and c["ret"] < 0:
            res["reading"] = "2017-19 is under 0% at both 2+ and 3+: no sign of a college opener spread edge in either period."
        elif a["ret"] > 0 and c["ret"] > 0:
            res["reading"] = "2017-19 is above 0% at both 2+ and 3+: a second period behind the opener edge; a lead for a real-price test."
        else:
            res["reading"] = "Mixed: 2017-19 is above 0% at one threshold and below at the other."
    else:
        res["reading"] = "Validation failed: the file's opener does not track the board's closely enough. Stopped."
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")

    def show(s):
        return "-" if not s["n"] else f"{s['w']}-{s['l']} {100 * s['win']:.1f}% {100 * s['ret']:+.1f}% ± {100 * s['se']:.1f}% clv {s['clv']:+.2f}"
    print("validation", {k: round(x, 3) for k, x in val.items()}, "ok" if ok else "FAILED")
    for name, r in res.get("periods", {}).items():
        for k, s in r.items():
            print(f"{name:8s} {k:14s} {show(s)}")
    if ok:
        for k, s in res["board_openers_2021_25"].items():
            print(f"board's own openers 2021-25 {k}: {show(s)}")
    print(res["reading"])


if __name__ == "__main__":
    main()
