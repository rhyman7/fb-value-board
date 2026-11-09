"""The Props section of the college board (Nov 7, 2026): the NFL page's layout on college lines.

    python -m nflmodel.cfbpropboard          # log any new lines, then rebuild the page's files

Prices: the live DraftKings and FanDuel college props nflmodel/cfbprops.py already pulls for the
Parlays page (data/props/history_cfb_live/<season>/<event_id>.csv, one file per game rewritten at
each snapshot, so after kickoff it holds the closing prices). No Odds API calls here.

The log, data/props/cfb_prop_log.csv: the first time a line (game, player, market, line) shows up
in a snapshot, one row is written with the books' prices, the vig-free market chance and the
model's chance at that moment (the college projections in docs/cfb/data.json on the college-fitted
prop distributions, data/cfb/prop_params.json via parlays.prop_params_path). Rows are never
rewritten: that is the pricing the page shows. The closing line comes from the game's file once
the game has kicked off; the result from the board's archived box scores
(docs/cfb/archive/<season>_w<WW>.json "actuals", a player missing from a fully graded game = void).

    docs/cfb/props/<season>_w<WW>.json   the same record shape as the NFL page (propboard._rec)
    docs/cfb/props/index.json            the weeks on file with a summary each, and the season
    docs/cfb/props/outlook.json          each board player's range (propoutlook, college model)

Not market-tested: the college model's chances have not been graded against real college lines,
so nothing is flagged (flags_on is false) and the page says so. Display only: nothing here feeds
the model, the bets, the Bet sheet or the Parlays page. Run by Update college board after the
build; a failure never stops it.
"""
from __future__ import annotations

import copy
import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config as C, odds as O, parlays as PL, propdist as D, propmarket as PM, propoutlook as POUT, props as P
from .cfbprops import LIVE
from .propboard import _dump, _num

OUT = C.DOCS / "cfb" / "props"
LOG = C.DATA / "props" / "cfb_prop_log.csv"
PAYLOAD = C.DOCS / "cfb" / "data.json"
ARCHIVE = C.DOCS / "cfb" / "archive"
KEY = ["game_id", "player_id", "market", "line"]
LOG_COLS = ["logged", "snapshot", "season", "week", "game_id", "event_id", "commence", "player_id", "player", "team", "pos",
            "stat", "market", "alt", "line", "proj", "p_over", "p_under", "p_push", "fair_over", "books"]
ACTUAL = {"pass_yds": ("pass_yds",), "pass_td": ("pass_td",), "rush_yds": ("rush_yds",), "rec_yds": ("rec_yds",),
          "receptions": ("receptions",), "anytime_td": ("rush_td", "rec_td")}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def _week(game_id: str) -> int:
    return int(str(game_id).split("_")[1])


def read_log() -> pd.DataFrame:
    return pd.read_csv(LOG, dtype={"player_id": str, "event_id": str}) if LOG.exists() else pd.DataFrame(columns=LOG_COLS)


def snapshots(season: int) -> pd.DataFrame:
    """Every game file this season (kicked off or not): the latest snapshot of each game."""
    parts = []
    for f in sorted((LIVE / str(season)).glob("*.csv")):
        if f.name != "fetched.csv":
            x = pd.read_csv(f)
            if len(x):
                parts.append(x)
    if not parts:
        return pd.DataFrame()
    s = pd.concat(parts, ignore_index=True)
    s["side"] = s["side"].str.lower().replace({"yes": "over", "no": "under"})
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    s.loc[s["market"] == "player_anytime_td", "line"] = 0.5
    s["nn"] = s["player"].map(PM.norm_name)
    return s.dropna(subset=["stat", "line"])


def _lines(s: pd.DataFrame, gid: dict) -> pd.DataFrame:
    """One row per event x player x market x line: each book's [over, under] and the vig-free
    chance of the over, the way the Parlays page prices college legs (parlays.cfb_book_legs)."""
    rows = []
    for (ev, nn, market, L), x in s.groupby(["event_id", "nn", "market", "line"]):
        bk, nv, one = {}, [], []
        for b in PL.BOOKS:
            d = x[x["book"] == b]
            if d.empty:
                continue
            ov = d.loc[d["side"] == "over", "price"]
            un = d.loc[d["side"] == "under", "price"]
            ov = float(ov.iloc[-1]) if len(ov) else None
            un = float(un.iloc[-1]) if len(un) else None
            if ov is None and un is None:
                continue
            bk[b] = [None if ov is None else int(ov), None if un is None else int(un)]
            if ov is not None and un is not None:
                nv.append(float(O.no_vig_power(ov, un)[0]))
            elif ov is not None:
                one.append(ov)
        stat = x["stat"].iloc[0]
        if nv:
            fair = float(np.mean(nv))
        elif one and stat == "anytime_td":
            fair = float(np.mean([O.one_sided_fair(v, PL.CFB_TD_MARGIN, "power") for v in one]))
        elif one:                                  # over-only ladder rungs: the books' rung calibration
            fair = float(np.mean([np.interp(O.implied_prob(v), *zip(*PL.CFB_RUNG_CAL)) for v in one]))
        else:
            fair = None
        rows.append({"event_id": ev, "game_id": gid.get(ev), "commence": x["commence"].iloc[0], "snapshot": x["snapshot"].max(),
                     "player": x["player"].iloc[0], "nn": nn, "stat": stat, "market": market, "alt": bool(x["alt"].iloc[0]),
                     "line": float(L), "fair_over": fair, "books": json.dumps(bk, separators=(",", ":"))})
    return pd.DataFrame(rows)


def with_target_share(payload: dict) -> dict:
    """A copy of the college payload with each player's reception share as his target share.
    The prop model sizes a receiver's range by tgt_share; the college board carries rec_share
    instead (no targets in college box scores), and propmarket.board_players fills a missing
    tgt_share with 0, which squeezes every receiver's range to almost nothing. The college widths
    were fitted with rec_share in that place (experiments/cfb_propdist_test.py)."""
    out = copy.deepcopy(payload)
    for p in (out.get("players") or {}).get("players") or []:
        why = p.get("why")
        if not isinstance(why, dict):
            continue
        if why.get("tgt_share") is None and why.get("rec_share") is not None:
            why["tgt_share"] = why["rec_share"]
        if why.get("tgt_share_hist") is None and why.get("rec_share_hist") is not None:
            why["tgt_share_hist"] = why["rec_share_hist"]
    for p in (out.get("players") or {}).get("players") or []:
        # a player the official report lists Questionable / GTD shows a projection shaded by his
        # chance to play; a prop is void if he sits, so props price his projection given he plays
        if isinstance(p.get("proj_played"), dict):
            p["proj"] = dict(p["proj_played"])
    return out


def _chances(lines: pd.DataFrame, payload: dict, model) -> pd.DataFrame:
    """The model's chances of over, push and under at each line, for players on the board
    (payload already through with_target_share)."""
    players = PM.board_players(payload)
    if players.empty or lines.empty:
        return pd.DataFrame()
    players["nn"] = players["name"].map(PM.norm_name)
    dup = players.duplicated(["game_id", "nn"], keep=False)        # a name the board knows twice: left out
    players = players[~dup]
    m = lines.merge(players[["game_id", "nn", "player_id", "name", "team", "pos", "group"]], on=["game_id", "nn"], how="inner")
    out = []
    for (stat, grp), x in m.groupby(["stat", "group"]):
        if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
            continue
        x = x.reset_index(drop=True)
        d = x[["game_id", "player_id"]].merge(players.drop(columns=["nn", "name", "team", "pos", "group"]), on=["game_id", "player_id"], how="left")
        d["group"] = grp
        for c in P.SPECS[stat].proj:
            if c not in d:
                d[c] = 0.0
        d = d.fillna({c: 0.0 for c in P.SPECS[stat].proj})
        over, push, under = model.probs(d, stat, grp, x["line"].to_numpy())
        y = x.copy()
        y["proj"] = np.round(P.mean(d, stat), 1)
        y["p_over"], y["p_push"], y["p_under"] = np.ravel(over), np.ravel(push), np.ravel(under)
        out.append(y)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def log_new(payload: dict, model, now: dt.datetime | None = None) -> int:
    """Append the first sighting of every line on an upcoming game; returns rows added."""
    now = now or dt.datetime.now(dt.timezone.utc)
    season = int((payload.get("meta") or {}).get("season") or now.year)
    s = snapshots(season)
    if s.empty:
        return 0
    s = s[pd.to_datetime(s["commence"], utc=True) > pd.Timestamp(now)]
    if s.empty:
        return 0
    gid = PL._cfb_games(payload, s)
    lines = _lines(s[s["event_id"].isin(gid)], gid)
    if lines.empty:
        return 0
    log = read_log()
    if len(log):
        seen = set(zip(log["game_id"], log["player"].map(PM.norm_name), log["market"], log["line"].astype(float)))
        lines = lines[[k not in seen for k in zip(lines["game_id"], lines["nn"], lines["market"], lines["line"])]]
    x = _chances(lines, payload, model)
    if x.empty:
        return 0
    x["logged"], x["season"], x["week"] = _now(), season, x["game_id"].map(_week)
    x["player"] = x["name"]
    x = x[LOG_COLS]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return len(x)


def _actuals(season: int, week: int) -> tuple[dict, set]:
    try:
        a = json.loads((ARCHIVE / f"{season}_w{week:02d}.json").read_text())
    except Exception:
        return {}, set()
    return a.get("actuals") or {}, set(a.get("full") or [])


def _grade(r, acts: dict, full: set) -> tuple[float | None, str | None]:
    """(actual, result of the over): None until the game is graded; void if he didn't play."""
    g = acts.get(r["game_id"])
    if not g:
        return None, None
    a = g.get(str(r["player_id"]))
    if a is None:
        return (None, "void") if r["game_id"] in full else (None, None)
    y = float(sum(a.get(k) or 0 for k in ACTUAL.get(r["stat"], ())))
    L = float(r["line"])
    return y, "win" if y > L else "push" if y == L else "loss"


def _closing(season: int, now: dt.datetime) -> tuple[dict, dict]:
    """For games that have kicked off: the main line at close (median across books) per
    game, player and stat, and the closing vig-free over at the same line."""
    s = snapshots(season)
    if s.empty:
        return {}, {}
    s = s[pd.to_datetime(s["commence"], utc=True) <= pd.Timestamp(now)]
    if s.empty:
        return {}, {}
    lines = _lines(s, {})
    main = lines[~lines["alt"]]
    cl = main.groupby(["event_id", "nn", "stat"])["line"].median().to_dict()
    cf = {(e, n, m, L): f for e, n, m, L, f in zip(lines["event_id"], lines["nn"], lines["market"], lines["line"], lines["fair_over"])}
    return cl, cf


def _rec(r) -> dict:
    po, pu, pp = r["p_over"], r["p_under"], r["p_push"]
    bk = json.loads(r["books"]) if isinstance(r["books"], str) and r["books"] else {}
    q = po / max(po + pu, 1e-9) if pd.notna(po) and pd.notna(pu) else None
    fo = r["fair_over"] if pd.notna(r["fair_over"]) else None

    def ev(side):
        xs = [v[0 if side == "o" else 1] for v in bk.values() if v[0 if side == "o" else 1] is not None]
        p = po if side == "o" else pu
        if not xs or pd.isna(p):
            return None
        best = max(xs)
        dec = 1 + best / 100 if best > 0 else 1 + 100 / -best
        return p * (dec - 1) - (1 - p - (pp or 0))
    return {
        "g": r["game_id"], "id": str(r["player_id"]), "n": r["player"], "t": r["team"], "pos": r["pos"], "s": r["stat"],
        "alt": bool(r["alt"]), "L": _num(r["line"], 1), "mu": _num(r["proj"], 1),
        "po": _num(po), "pu": _num(pu), "pp": _num(pp), "fo": _num(fo),
        "pno": None, "pnu": None, "mun": None,
        "eo": _num(q - fo) if q is not None and fo is not None else None,
        "eu": _num((1 - q) - (1 - fo)) if q is not None and fo is not None else None,
        "vo": _num(ev("o")), "vu": _num(ev("u")),
        "bk": bk, "fl": None, "wn": "not market-tested", "snap": "priced", "at": r["logged"], "ko": r["commence"],
        "cL": _num(r.get("cL"), 1), "cfo": _num(r.get("cfo")), "bg": None, "mid": False, "lg": None, "tb": None, "take": None,
        "y": _num(r.get("y"), 1), "res": r.get("res") if isinstance(r.get("res"), str) else None,
    }


def _move(r) -> float | None:
    """Points the closing main line moved toward the side the model leaned (positive = toward)."""
    if r["alt"] or pd.isna(r.get("cL")) or pd.isna(r["fair_over"]) or pd.isna(r["p_over"]):
        return None
    q = r["p_over"] / max(r["p_over"] + r["p_under"], 1e-9)
    return float(r["cL"] - r["line"]) * (1 if q > r["fair_over"] else -1)


def summary(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"props": 0}
    main = x[~x["alt"].astype(bool)]
    dec = x[x["res"].isin(["win", "loss"]) & x["fair_over"].notna() & x["p_over"].notna()]
    hit = (dec["res"] == "win").astype(float)
    ours = dec["p_over"] / (dec["p_over"] + dec["p_under"])
    mv = main.apply(_move, axis=1).dropna() if len(main) else pd.Series(dtype=float)
    bands = np.r_[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    return {
        "props": int(len(x)), "main_lines": int(len(main)), "ladder_rungs": int(len(x) - len(main)),
        "players": int(x["player_id"].nunique()), "games": int(x["game_id"].nunique()), "graded": int(len(dec)),
        "brier_ours": _num(D.brier(ours, hit), 5) if len(dec) else None,
        "brier_books": _num(D.brier(dec["fair_over"], hit), 5) if len(dec) else None,
        "moved_toward": _num((mv > 0).mean(), 3) if len(mv) else None, "moved_away": _num((mv < 0).mean(), 3) if len(mv) else None,
        "moved_points": _num(mv.mean(), 2) if len(mv) else None, "moved_n": int(len(mv)),
        "flagged": 0, "book_gaps": 0, "middles": 0, "ladder_gaps": 0, "flag_record": "0-0-0", "new": None,
        "calibration": D.calibration_table(ours, hit, bands) if len(dec) else [],
        "calibration_books": D.calibration_table(dec["fair_over"], hit, bands) if len(dec) else [],
    }


def build(now: dt.datetime | None = None, out_dir=OUT) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    log = read_log()
    out_dir.mkdir(parents=True, exist_ok=True)
    weeks, season_rows = [], []
    if len(log):
        log = log.drop_duplicates(KEY, keep="first").copy()
        log["alt"] = log["alt"].astype(str).str.lower().isin(["true", "1"])
        log["line"] = log["line"].astype(float)
        nn = log["player"].map(PM.norm_name)
        cache = {}
        for season in log["season"].unique():
            cache[int(season)] = _closing(int(season), now)
        log["cL"] = [cache[int(s)][0].get((e, n, st)) if not a else None
                     for s, e, n, st, a in zip(log["season"], log["event_id"], nn, log["stat"], log["alt"])]
        log["cfo"] = [cache[int(s)][1].get((e, n, m, L)) for s, e, n, m, L in zip(log["season"], log["event_id"], nn, log["market"], log["line"])]
        ys, rs = [], []
        acts = {}
        for r in log.to_dict("records"):
            k = (int(r["season"]), int(r["week"]))
            if k not in acts:
                acts[k] = _actuals(*k)
            y, res = _grade(r, *acts[k])
            ys.append(y); rs.append(res)
        log["y"], log["res"] = ys, rs
        log.loc[log["res"] == "void", "y"] = np.nan
        for (season, week), x in log.groupby(["season", "week"]):
            f = f"{int(season)}_w{int(week):02d}.json"
            body = {"season": int(season), "week": int(week), "summary": summary(x),
                    "props": [_rec(r) for r in x.sort_values(["game_id", "player", "stat", "market", "line"]).to_dict("records")]}
            _dump(out_dir / f, body)
            weeks.append({"file": f, "season": int(season), "week": int(week),
                          **{k: v for k, v in body["summary"].items() if not k.startswith("calibration")}})
        season_rows = log
    ix = {"updated": _now(), "league": "cfb", "flags_on": False, "market_tested": False, "weeks": weeks,
          "season": summary(season_rows) if len(season_rows) else {"props": 0}}
    _dump(out_dir / "index.json", ix)
    return ix


def write_outlook(payload: dict, model) -> int:
    return POUT.write(payload, model, path=OUT / "outlook.json")


def run() -> None:
    payload = with_target_share(json.loads(PAYLOAD.read_text()))
    model = P.PropModel.load(PL.prop_params_path("cfb"))
    n = log_new(payload, model)
    ix = build()
    k = write_outlook(payload, model)
    print(f"college props page: {n} new lines logged; {len(ix['weeks'])} week(s) on file; outlook for {k} player-stats")


if __name__ == "__main__":
    run()
