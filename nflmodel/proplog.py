"""The prop log: what we priced, what the books offered, what happened.

  data/props/prop_log.csv      append-only. One row per run x player x market x line x
                               side: the time, every book's price, the vig-free market
                               chance (scaling and power), ours, the edge, EV, stake,
                               whether it was flagged and why not. Rows are never edited.
  data/props/prop_results.csv  append-only. One row per graded log row (by row_id): the
                               result and, for flagged bets, the closing price and the
                               closing line value.
  data/props/snapshots.csv     which events have had their pricing ("flag") and closing
                               ("close") snapshot, so credits are not spent twice.
  data/props/weekly_summary.csv  rebuilt from the two logs on every grading run.
  data/props/summary.json        the same, plus calibration of the live probabilities.

Closing line value is the early test of a real edge: wins and losses take hundreds of
bets to mean anything, but beating the closing price shows up much sooner.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from . import config as C, odds as O, propdist as D

DIR = C.DATA / "props"
LOG = DIR / "prop_log.csv"
RESULTS = DIR / "prop_results.csv"
SNAPS = DIR / "snapshots.csv"
WEEKLY = DIR / "weekly_summary.csv"
SUMMARY = DIR / "summary.json"

LOG_COLS = ["row_id", "time", "snapshot", "season", "week", "game_id", "event_id", "commence", "player_id", "player",
            "team", "pos", "market", "stat", "line", "side", "proj", "books", "n_books", "sharp", "one_sided", "hold",
            "price", "book", "fair", "fair_scaled", "our_p", "our_push", "edge", "ev", "kelly", "stake", "flagged",
            "why_not", "role_clear", "status"]
RES_COLS = ["row_id", "graded", "actual", "result", "profit_flat", "close_line", "close_price", "close_fair",
            "clv_fair", "clv_ev", "clv_note"]


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def row_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _append(path, cols, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in cols})


def _fmt(v):
    if isinstance(v, dict):
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else round(float(v), 5)
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    return "" if v is None else v


def append_priced(priced: pd.DataFrame, snapshot: str, season: int, week: int) -> int:
    if priced is None or priced.empty:
        return 0
    t = _now()
    rows = []
    for r in priced.to_dict("records"):
        r.update(time=t, snapshot=snapshot, season=season, week=week)
        r["row_id"] = row_id(t, snapshot, r["game_id"], r["player_id"], r["market"], r["line"], r["side"])
        rows.append(r)
    _append(LOG, LOG_COLS, rows)
    return len(rows)


def read_log() -> pd.DataFrame:
    return pd.read_csv(LOG, dtype={"player_id": str}) if LOG.exists() else pd.DataFrame(columns=LOG_COLS)


def read_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=RES_COLS)


def snapshots() -> pd.DataFrame:
    return pd.read_csv(SNAPS) if SNAPS.exists() else pd.DataFrame(columns=["event_id", "kind", "time", "markets", "rows"])


def record_snapshot(event_id, kind, markets, nrows):
    _append(SNAPS, ["event_id", "kind", "time", "markets", "rows"],
            [{"event_id": event_id, "kind": kind, "time": _now(), "markets": " ".join(markets), "rows": nrows}])


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

STAT_COLS = {"rec_yds": ["rec_yds"], "receptions": ["receptions"], "rush_yds": ["rush_yds"], "carries": ["carries"],
             "pass_yds": ["pass_yds"], "pass_att": ["pass_att"], "completions": ["completions"], "pass_td": ["pass_td"],
             "ints": ["ints"], "anytime_td": ["rec_td", "rush_td"]}


def outcomes(seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Box scores and game-day availability from the board's caches (refreshed daily by
    the Update board run)."""
    pg, av = [], []
    for s in seasons:
        p = C.CACHE / f"player_games_{s}.csv"
        a = C.CACHE / f"player_avail_{s}.csv"
        if p.exists():
            pg.append(pd.read_csv(p, dtype={"player_id": str}))
        if a.exists():
            av.append(pd.read_csv(a, dtype={"player_id": str}))
    return (pd.concat(pg, ignore_index=True) if pg else pd.DataFrame(),
            pd.concat(av, ignore_index=True) if av else pd.DataFrame())


def grade() -> int:
    """Grade every logged pricing row whose game is in the box scores and that has no result
    yet. A player who did not play (not active, no stats) is 'void', as the books settle it."""
    log = read_log()
    if log.empty:
        return 0
    done = set(read_results()["row_id"])
    todo = log[(log["snapshot"] == "flag") & ~log["row_id"].isin(done)]
    if todo.empty:
        return 0
    pg, av = outcomes(sorted(todo["season"].dropna().astype(int).unique()))
    if pg.empty:
        return 0
    finished = set(pg["game_id"])
    todo = todo[todo["game_id"].isin(finished)]
    close = log[log["snapshot"] == "close"]
    box = pg.set_index(["game_id", "player_id"])
    act = set()
    if len(av):
        a = av[av["active"] == 1]
        act = set(zip(a["season"], a["week"], a["team"], a["player_id"]))
    out = []
    for r in todo.to_dict("records"):
        key = (r["game_id"], r["player_id"])
        played = key in box.index or (int(r["season"]), int(r["week"]), r["team"], r["player_id"]) in act
        rec = {"row_id": r["row_id"], "graded": _now()}
        if not played:
            rec.update(result="void", profit_flat=0.0)
            out.append(rec)
            continue
        y = float(box.loc[[key], STAT_COLS[r["stat"]]].sum(axis=1).sum()) if key in box.index else 0.0
        line = float(r["line"])
        win = y > line if r["side"] == "over" else y < line
        push = y == line
        dec = float(O.american_to_decimal(r["price"]))
        rec.update(actual=y, result="push" if push else ("win" if win else "loss"),
                   profit_flat=0.0 if push else (dec - 1 if win else -1.0))
        # closing line value, for the side we priced
        c = close[(close["game_id"] == r["game_id"]) & (close["player_id"] == r["player_id"]) &
                  (close["market"] == r["market"]) & (close["side"] == r["side"])]
        if len(c):
            c = c.sort_values("time")
            same = c[np.isclose(c["line"].astype(float), line)]
            if len(same):
                cl = same.iloc[-1]
                cf = float(cl["fair"])
                note = "same line"
            else:
                cl = c.iloc[-1]
                # the closing line moved: carry its fair chance to our line with our own
                # distribution's difference between the two lines
                cf = float(cl["fair"]) + (float(r["our_p"]) - float(cl["our_p"]))
                note = f"line moved to {cl['line']}; fair price carried to {line} with the model"
            rec.update(close_line=cl["line"], close_price=cl["price"], close_fair=round(cf, 5),
                       clv_fair=round(cf - float(r["fair"]), 5),
                       clv_ev=round(cf * (dec - 1) - (1 - cf - float(r["our_push"] or 0)), 5), clv_note=note)
        out.append(rec)
    _append(RESULTS, RES_COLS, out)
    return len(out)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def summarize() -> dict:
    log, res = read_log(), read_results()
    if log.empty or res.empty:
        return {}
    x = log[log["snapshot"] == "flag"].merge(res, on="row_id", how="inner")
    x = x[x["result"] != "void"]
    # one pricing per side per prop: the first time it was priced
    x = x.sort_values("time").drop_duplicates(["game_id", "player_id", "market", "line", "side"])
    x["hit"] = (x["result"] == "win").astype(float)
    bets = x[x["flagged"] == 1].copy()
    bets["profit_stake"] = bets["profit_flat"] * bets["stake"]
    weeks = []
    for (s, w), b in bets.groupby(["season", "week"]):
        weeks.append(_line(b, {"season": int(s), "week": int(w)}))
    WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weeks).to_csv(WEEKLY, index=False)
    decided = x[x["result"] != "push"]
    p_dec = decided["our_p"] / (1 - decided["our_push"].fillna(0))      # chance of winning, given no push
    cal = {"n": int(len(decided)), "note": "every side priced at its first pricing, flagged or not; pushes left out",
           "brier": round(D.brier(p_dec, decided["hit"]), 5) if len(decided) else None,
           "brier_market": round(D.brier(decided["fair"], decided["hit"]), 5) if len(decided) else None,
           "table": D.calibration_table(p_dec, decided["hit"]) if len(decided) else []}
    out = {"updated": _now(), "all": _line(bets, {}), "weeks": weeks, "calibration_all_priced_sides": cal}
    SUMMARY.write_text(json.dumps(out, indent=1, default=float))
    return out


def _line(b: pd.DataFrame, head: dict) -> dict:
    w, l, p = int((b["result"] == "win").sum()), int((b["result"] == "loss").sum()), int((b["result"] == "push").sum())
    clv = b["clv_ev"].dropna() if "clv_ev" in b else pd.Series(dtype=float)
    return dict(head, bets=int(len(b)), record=f"{w}-{l}-{p}", units_flat=round(float(b["profit_flat"].sum()), 3),
                return_flat=round(float(b["profit_flat"].mean()), 4) if len(b) else None,
                bankroll_change=round(float(b["profit_stake"].sum()), 4) if "profit_stake" in b and len(b) else 0.0,
                avg_edge=round(float(b["edge"].mean()), 4) if len(b) else None,
                clv_bets=int(len(clv)), avg_clv_ev=round(float(clv.mean()), 4) if len(clv) else None,
                beat_close=round(float((clv > 0).mean()), 3) if len(clv) else None)
