"""Parlay legs: every bet the board can price, as one list the Parlays page combines.

The page (site/parlays.html, served as docs/parlays.html) has three tools: a builder (pick the
legs, see the chance they all win and what the price has to be), a boost finder (a profit
boost and its rules in, the best legs out) and a leg finder (2-5 legs, no boost). All three
work in the browser from the file this module writes, one per board:

    docs/parlays/nfl.json   docs/parlays/cfb.json
      {league, built, generated, week, unit, games: {game_id: {...}}, legs: [...], corr: {...}}

A leg is one side at one line:

  m     market: spread, total, ml, prop
  L     the line (spreads in the picked team's terms: -3.5 lays 3.5)
  w, h  the chance it wins and the chance it pushes (a push drops the leg from a parlay)
  wv    NFL props only: the chance value is scored on, the books' own number (below)
  pm    the model's own chance where there is one; pk the market's no-vig chance
  bk    {dk, fd}: the American price at that exact line where the book posts it
  src   where w comes from (below); t: tags shown on the page
  c     what the leg is, for same-game correlations (propsim pairs and data/parlay_corr.json)

Where w comes from, and why. A parlay is only as honest as its legs' chances:

  game lines   the board's market-adjusted projection (the sharp book's no-vig line moved
               toward the model by the weight it has earned), read off the key-number curve
               at every half point within 7 of the market line (alts.py). Moved lines use the
               same curve, so a spread through 3 or 7 is priced by what those numbers are worth.
  NFL props    the books are the better forecast (PROPS.md), so the chance is the books'
               no-vig number moved toward the model: 20% of the way on two-sided main lines
               (the blend weight fitted on 2023-24 and checked on 2025 was 0.18) and half way
               on over-only ladder rungs, where the books' number is first read through how
               2025's real rungs at that price cleared (RUNG_CAL: a -1000 rung cleared about
               89%, a +1500 rung under 4%), then moved a quarter of the way to the model. A prop at a line the books
               do not post carries the same correction from the player's main line through
               the model's distribution from the nearest line the books do price, and has
               no price ("fair odds only"). Value is scored on the books' number alone (wv):
               sides the nudged chance made +EV lost about 4% to the close at every cap
               (PROPS.md, "How far the model may move a prop"), so the nudge is shown as the
               chance and never counted as an edge.
  college      no prop prices exist (CFB_PLAYERS.md), so college props are the model's chance
  props        alone, from the NFL prop distributions on the college projections, tagged
               "not market-tested".

Nothing here is a bet by itself; the page says what each combination is worth at the price
the bettor can get.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import odds as O

OUT = C.DOCS / "parlays"
CORR = C.ROOT / "data" / "parlay_corr.json"
BOOKS = {"draftkings": "dk", "fanduel": "fd"}
W_MAIN = 0.2          # model weight on two-sided prop lines (PROPS.md, "The blend, out of sample")
W_RUNG = 0.25         # model weight on over-only ladder rungs, after the calibrated price below
# Over-only ladder rungs: what a DraftKings or FanDuel rung's raw implied chance (vig in) turned
# out to be worth on 2025's 71,915 real rungs (experiments/out/props_ladders_2025.json,
# calibration.bands.book_raw: band average implied -> share that cleared). The books charge the
# most on long shots and on the shortest rungs, so a flat margin misprices both ends.
RUNG_CAL = [(0.0, 0.0), (0.0561, 0.0361), (0.1651, 0.1326), (0.3225, 0.2868), (0.4978, 0.4456), (0.6767, 0.616),
            (0.8007, 0.7314), (0.885, 0.832), (0.9396, 0.8928), (0.9812, 0.9367), (1.0, 0.955)]
W_TD = 0.2            # anytime touchdown (Yes only)
RUNG_MARGIN = 0.10    # assumed book margin on over-only rungs, as propmarket assumes
PULL_MAX = 0.03       # the model moves a prop chance at most 3 points off the books'
CONFLICT = 0.25       # model and books more than 25 points apart: different news (injury, role); left out
MOVE_LO, MOVE_HI = 0.55, 0.95     # moved prop lines kept between these chances
MOVE_MAX = 6          # at most this many moved lines per player, stat and side
CFB_ROLES = ("QB", "R1", "R2", "RB1")
CFB_MOVE_MAX = 3
CFB_STATS = ("pass_yds", "rush_yds", "rec_yds", "receptions", "anytime_td")
STAT_LABEL = {"rec_yds": "rec yds", "rush_yds": "rush yds", "pass_yds": "pass yds", "receptions": "receptions",
              "carries": "carries", "pass_att": "pass att", "completions": "completions", "pass_td": "pass TDs",
              "ints": "INTs", "anytime_td": "anytime TD"}


def american(p: float) -> int | None:
    """Fair American price for a chance (no vig)."""
    if not p or p <= 0 or p >= 1:
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def _upcoming(g: dict, now: dt.datetime) -> bool:
    try:
        return dt.datetime.fromisoformat(g["kickoff"].replace("Z", "+00:00")) > now
    except Exception:
        return False


# --------------------------------------------------------------------------
# Game lines
# --------------------------------------------------------------------------

def _shrink(proj: float, market: float, h: float) -> float:
    """The market's chance moved toward the projection by the share of claimed edge that has
    shown up in results (backtest.edge_haircut: 0 on the NFL board as of Oct 2026, 0.52 college)."""
    return market + h * (proj - market)


def _book_prices(g: dict) -> dict:
    """{dk|fd: {spread_line, home_spread_odds, ...}} for the books the bettor uses."""
    return {BOOKS[b]: q for b, q in (g.get("books") or {}).items() if b in BOOKS}


def game_legs(g: dict, league: str, haircut: float = 1.0, alt: dict | None = None) -> list[dict]:
    alts = g.get("alts") or {}
    sp = {round(float(r["line"]) * 2) / 2: r for r in alts.get("spread") or []}
    to = {round(float(r["line"]) * 2) / 2: r for r in alts.get("total") or []}
    if not sp and not to:
        return []
    bk = _book_prices(g)
    home, away = g["home"], g["away"]
    mkt = g.get("market") or {}
    legs = []
    base = {"lg": league, "g": g["id"], "ko": g["kickoff"]}

    def p_home_gt(x):           # P(home margin > x) on the projection, x a half point
        r = sp.get(x)
        return None if r is None else float(r.get("home_proj", r["home"]))

    def p_home_gt_mkt(x):
        r = sp.get(x)
        return None if r is None else float(r["home"])

    def p_over(x):
        r = to.get(x)
        return None if r is None else float(r.get("over_proj", r["over"]))

    def p_over_mkt(x):
        r = to.get(x)
        return None if r is None else float(r["over"])

    # ---- spreads: every half point on the curve, plus the books' own whole-number lines
    lines = set(sp)
    for q in bk.values():
        if q.get("spread_line") is not None:
            lines.add(float(q["spread_line"]))
    for q in (alt or {}).values():
        lines |= {x for x in q.get("spread", {}) if min(sp, default=0) - 0.5 <= x <= max(sp, default=0) + 0.5}
    for x in sorted(lines):
        # home laying x (x > 0 favourite): home wins if margin > x; push if margin == x (whole x)
        if x == int(x):
            hi, lo = p_home_gt(x + 0.5), p_home_gt(x - 0.5)
            hik, lok = p_home_gt_mkt(x + 0.5), p_home_gt_mkt(x - 0.5)
            if hi is None or lo is None:
                continue
            w_home, push = hi, max(lo - hi, 0.0)
            wk_home, pushk = hik, max(lok - hik, 0.0)
        else:
            w_home, push = p_home_gt(x), 0.0
            wk_home, pushk = p_home_gt_mkt(x), 0.0
            if w_home is None:
                continue
        if wk_home is None:
            continue
        pm_home, pm_push = w_home, push
        w_home, push = _shrink(w_home, wk_home, haircut), _shrink(push, pushk, haircut)
        for side, team, opp, w, wk, line in (("home", home, away, w_home, wk_home, -x),
                                             ("away", away, home, 1 - w_home - push, 1 - wk_home - pushk, x)):
            prices = {}
            for b, q in bk.items():
                if q.get("spread_line") is not None and abs(float(q["spread_line"]) - x) < 1e-9:
                    o = q.get(f"{side}_spread_odds")
                    if o is not None:
                        prices[b] = int(o)
            for b, q in (alt or {}).items():          # the books' alternate ladders
                o = (q.get("spread", {}).get(x) or {}).get(side)
                if o is not None and b not in prices:
                    prices[b] = int(o)
            main = abs(x - float(mkt.get("spread", 99))) < 1e-9
            legs.append({**base, "m": "spread", "lab": f"{team} {'+' if line > 0 else ''}{line:g}" if line else f"{team} PK",
                         "L": line, "w": _r(w), "h": _r(push), "pk": _r(wk) if wk is not None else None,
                         "pm": _r(pm_home if side == "home" else 1 - pm_home - pm_push),
                         "bk": prices, "src": "proj", "t": [] if prices or main else ["moved line"],
                         "c": {"k": "cover", "tm": team, "side": side}})

    # ---- totals
    lines = set(to)
    for q in bk.values():
        if q.get("total_line") is not None:
            lines.add(float(q["total_line"]))
    for q in (alt or {}).values():
        lines |= {x for x in q.get("total", {}) if min(to, default=0) - 0.5 <= x <= max(to, default=0) + 0.5}
    for x in sorted(lines):
        if x == int(x):
            hi, lo = p_over(x + 0.5), p_over(x - 0.5)
            hik, lok = p_over_mkt(x + 0.5), p_over_mkt(x - 0.5)
            if hi is None or lo is None:
                continue
            w_over, push, wk_over, pushk = hi, max(lo - hi, 0.0), hik, max(lok - hik, 0.0)
        else:
            w_over, push = p_over(x), 0.0
            wk_over, pushk = p_over_mkt(x), 0.0
            if w_over is None:
                continue
        if wk_over is None:
            continue
        pm_over = w_over
        w_over, push = _shrink(w_over, wk_over, haircut), _shrink(push, pushk, haircut)
        main = abs(x - float(mkt.get("total", -99))) < 1e-9
        pm_push = max(lo - hi, 0.0) if x == int(x) else 0.0
        for side, w, wk, pm in (("over", w_over, wk_over, pm_over), ("under", 1 - w_over - push, 1 - wk_over - pushk, 1 - pm_over - pm_push)):
            prices = {}
            for b, q in bk.items():
                if q.get("total_line") is not None and abs(float(q["total_line"]) - x) < 1e-9:
                    o = q.get(f"{side}_odds")
                    if o is not None:
                        prices[b] = int(o)
            for b, q in (alt or {}).items():
                o = (q.get("total", {}).get(x) or {}).get(side)
                if o is not None and b not in prices:
                    prices[b] = int(o)
            legs.append({**base, "m": "total", "lab": f"{side.title()} {x:g}", "L": x, "w": _r(w), "h": _r(push), "pm": _r(pm),
                         "pk": _r(wk) if wk is not None else None, "bk": prices, "src": "proj",
                         "t": [] if prices or main else ["moved line"], "c": {"k": side}})

    # ---- moneylines: P(win) is the curve's P(margin > 0) on each side (ties are rare and refunded)
    win = g.get("win") or {}
    ph = win.get("adj")
    if ph is not None:
        pk = win.get("market")
        pm_home = ph
        if pk is not None:
            ph = _shrink(ph, pk, haircut)
        for side, team, w, wk in (("home", home, ph, pk), ("away", away, 1 - ph, None if pk is None else 1 - pk)):
            prices = {b: int(q[f"{side}_moneyline"]) for b, q in bk.items() if q.get(f"{side}_moneyline") is not None}
            legs.append({**base, "m": "ml", "lab": f"{team} ML", "L": None, "w": _r(w), "h": 0.0, "pk": _r(wk),
                         "pm": _r(pm_home if side == "home" else 1 - pm_home),
                         "bk": prices, "src": "proj", "t": [], "c": {"k": "cover", "tm": team, "side": side, "ml": True}})
    return legs


# --------------------------------------------------------------------------
# NFL props
# --------------------------------------------------------------------------

def _roles(players: list[dict]) -> dict:
    """{(game, player_id): role} with propsim's roles: QB, R1-R3 by target share, RB1."""
    out = {}
    df = pd.DataFrame([{"g": p["game"], "id": p["id"], "tm": p["team"], "pos": p["pos"],
                        "att": (p.get("why") or {}).get("att_share", 0) or 0,
                        "tgt": (p.get("why") or {}).get("tgt_share", (p.get("why") or {}).get("rec_share", 0)) or 0,
                        "car": (p.get("why") or {}).get("car_share", 0) or 0,
                        "out": str(p.get("status") or "").lower().startswith(("out", "doubtful", "injured"))} for p in players])
    if df.empty:
        return out
    df = df[~df["out"]]
    for (g, tm), d in df.groupby(["g", "tm"]):
        q = d[(d.pos == "QB") & (d.att > 0.5)]
        for i in q.index[:1]:
            out[(g, d.at[i, "id"])] = "QB"
        r = d[d.pos.isin(["WR", "TE"])].sort_values("tgt", ascending=False)
        for k, i in enumerate(r.index[:3]):
            out[(g, d.at[i, "id"])] = f"R{k + 1}"
        b = d[d.pos == "RB"].sort_values("car", ascending=False)
        for i in b.index[:1]:
            out[(g, d.at[i, "id"])] = "RB1"
    return out


def _latest_props(path: Path) -> list[dict]:
    rows = json.loads(path.read_text()).get("props") or []
    best = {}
    for r in rows:
        k = (r["g"], r["id"], r["s"], r["L"], bool(r["alt"]))
        if k not in best or str(r.get("at")) > str(best[k].get("at")):
            best[k] = r
    return list(best.values())


def _pull(diff: float, weight: float) -> float:
    """How far the model moves a prop chance off the books': weight x the gap, at most PULL_MAX.
    Sides the blend flagged at 2+ points returned -1.9% +- 4.4% on 2025 (PROPS.md), so a big
    disagreement is not treated as a big edge."""
    return float(np.clip(weight * diff, -PULL_MAX, PULL_MAX))


def _nv_main(r: dict) -> float | None:
    """Books' average no-vig chance of the over on a two-sided main line (power method)."""
    ps = []
    for b in BOOKS:
        q = (r.get("bk") or {}).get(b)
        if q and q[0] is not None and q[1] is not None:
            ps.append(float(O.no_vig_power(float(q[0]), float(q[1]))[0]))
    return float(np.mean(ps)) if ps else None


def _fair_rung(odds: float) -> float:
    """A rung's chance from its price, by how 2025's rungs at that price actually cleared."""
    xs, ys = zip(*RUNG_CAL)
    return float(np.interp(O.implied_prob(float(odds)), xs, ys))


def _fair_td(odds: float) -> float:
    return float(O.one_sided_fair(float(odds), RUNG_MARGIN, "power"))


def nfl_prop_legs(props_path: Path, outlook_path: Path, roles: dict, names: dict, now: dt.datetime) -> list[dict]:
    rows = [r for r in _latest_props(props_path) if r.get("ko") and dt.datetime.fromisoformat(r["ko"].replace("Z", "+00:00")) > now]
    outlook = json.loads(outlook_path.read_text()) if outlook_path.exists() else {"players": {}, "stats": {}}
    legs, offset, seen, main_line = [], {}, set(), {}
    for r in rows:
        stat, bk = r["s"], r.get("bk") or {}
        po, pu, pp = r.get("po"), r.get("pu"), r.get("pp") or 0.0
        if po is None:
            continue
        role = roles.get((r["g"], r["id"]))
        base = {"lg": "nfl", "g": r["g"], "ko": r["ko"], "m": "prop", "pl": r["n"], "tm": r["t"], "pos": r["pos"], "s": stat}
        two_sided = any(q and q[0] is not None and q[1] is not None for q in bk.values())
        if stat == "anytime_td":
            fair = [_fair_td(q[0]) for b, q in bk.items() if b in BOOKS and q and q[0] is not None]
            if not fair:
                continue
            f = float(np.mean(fair))
            if abs(po - f) > CONFLICT:
                continue
            w = f + _pull(po - f, W_TD)
            prices = {BOOKS[b]: int(q[0]) for b, q in bk.items() if b in BOOKS and q and q[0] is not None}
            legs.append({**base, "lab": f"{r['n']} anytime TD", "L": 0.5, "d": "over", "w": _r(w), "wv": _r(f), "h": 0.0, "pm": _r(po),
                         "pk": _r(f), "bk": prices, "src": "blend", "t": [],
                         "c": {"k": "prop", "p": r["id"], "tm": r["t"], "r": role, "s": stat, "dir": "over"}})
            continue
        if two_sided and not r["alt"]:
            nv = _nv_main(r)
            if nv is None:
                continue
            nvu = 1 - nv
            if abs(po - nv * (1 - pp)) > CONFLICT:
                continue                 # the model and the books see different players (injury, role): leave it out
            w_o = nv * (1 - pp) + _pull(po - nv * (1 - pp), W_MAIN)
            w_u = nvu * (1 - pp) + _pull(pu - nvu * (1 - pp), W_MAIN)
            offset.setdefault((r["g"], r["id"], stat), []).append((float(r["L"]), w_o, nv * (1 - pp)))
            main_line[(r["g"], r["id"], stat)] = float(r["L"])
            for side, w, pm, pk, j, wv in (("over", w_o, po, nv, 0, nv * (1 - pp)), ("under", w_u, pu, nvu, 1, nvu * (1 - pp))):
                prices = {BOOKS[b]: int(q[j]) for b, q in bk.items() if b in BOOKS and q and q[j] is not None}
                legs.append({**base, "lab": f"{r['n']} {side} {r['L']:g} {STAT_LABEL.get(stat, stat)}", "L": r["L"], "d": side,
                             "w": _r(w), "wv": _r(wv), "h": _r(pp), "pm": _r(pm), "pk": _r(pk), "bk": prices, "src": "blend", "t": [],
                             "c": {"k": "prop", "p": r["id"], "tm": r["t"], "r": role, "s": stat, "dir": side}})
                seen.add((r["g"], r["id"], stat, side, float(r["L"])))
        elif r["alt"]:
            fair = [_fair_rung(q[0]) for b, q in bk.items() if b in BOOKS and q and q[0] is not None]
            if not fair:
                continue
            f = float(np.mean(fair))
            if abs(po - f) > CONFLICT:
                continue
            w = f + _pull(po - f, W_RUNG)
            prices = {BOOKS[b]: int(q[0]) for b, q in bk.items() if b in BOOKS and q and q[0] is not None}
            key = (r["g"], r["id"], stat, "over", float(r["L"]))
            if key in seen:
                continue
            seen.add(key)
            offset.setdefault((r["g"], r["id"], stat), []).append((float(r["L"]), w, f))
            legs.append({**base, "lab": f"{r['n']} {r['L']:g}+ {STAT_LABEL.get(stat, stat)}".replace(".5+", "+").replace(f"{r['L']:g}+", f"{int(r['L'] + 0.5)}+"),
                         "L": r["L"], "d": "over", "w": _r(w), "wv": _r(f), "h": 0.0, "pm": _r(po), "pk": _r(f), "bk": prices,
                         "src": "blend", "t": ["alt rung"], "c": {"k": "prop", "p": r["id"], "tm": r["t"], "r": role, "s": stat, "dir": "over"}})
    # ---- moved lines the books don't post: the model's curve plus the main line's correction
    for key, cur in (outlook.get("players") or {}).items():
        g, pid, stat = key.split("|")
        if stat == "anytime_td" or (g, pid, stat) not in main_line:
            continue
        L0, known = main_line[(g, pid, stat)], offset[(g, pid, stat)]
        ko = names.get(("ko", g))
        if not ko or dt.datetime.fromisoformat(ko.replace("Z", "+00:00")) <= now:
            continue
        step = (outlook.get("stats") or {}).get(stat, {}).get("step", 1)
        s = cur["s"]
        info = names.get((g, pid)) or {}
        for side in ("over", "under"):
            cand = []
            for k in range(1, len(s) + 1):
                L = k * step - 0.5                       # over L means at least k x step
                p_model = s[k - 1] if side == "over" else 1 - s[k - 1]
                # the correction at the nearest priced line, measured against today's curve there
                # (the prices may be from an earlier snapshot, made with an earlier projection)
                Lp, wp, wvp = min(known, key=lambda t: abs(t[0] - L))
                kp = int(round((Lp + 0.5) / step))
                now_p = s[kp - 1] if 1 <= kp <= len(s) else (1.0 if kp < 1 else 0.0)
                off, offv = wp - now_p, wvp - now_p
                w = p_model + (off if side == "over" else -off)
                wv = p_model + (offv if side == "over" else -offv)
                if not (MOVE_LO <= w <= MOVE_HI) or (g, pid, stat, side, float(L)) in seen or abs(L - L0) < 1e-9:
                    continue
                cand.append((L, w, p_model, wv))
            if len(cand) > MOVE_MAX:
                idx = np.unique(np.linspace(0, len(cand) - 1, MOVE_MAX).round().astype(int))
                cand = [cand[i] for i in idx]
            for L, w, pm, wv in cand:
                lab = (f"{info.get('n', pid)} {int(L + 0.5)}+ {STAT_LABEL.get(stat, stat)}" if side == "over"
                       else f"{info.get('n', pid)} under {L:g} {STAT_LABEL.get(stat, stat)}")
                legs.append({"lg": "nfl", "g": g, "ko": ko, "m": "prop", "pl": info.get("n", pid), "tm": info.get("tm"), "pos": info.get("pos"),
                             "s": stat, "lab": lab, "L": L, "d": side, "w": _r(w), "wv": _r(min(max(wv, 0.005), 0.995)), "h": 0.0, "pm": _r(pm), "pk": None, "bk": {},
                             "src": "carry", "t": ["moved line"],
                             "c": {"k": "prop", "p": pid, "tm": info.get("tm"), "r": roles.get((g, pid)), "s": stat, "dir": side}})
    return monotone(legs)


def _pav(y: list[float]) -> list[float]:
    """Pool-adjacent violators: the closest non-increasing sequence."""
    blocks = []
    for v in y:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, n2 = blocks.pop()
            blocks[-1] = [(blocks[-1][0] * blocks[-1][1] + v2 * n2) / (blocks[-1][1] + n2), blocks[-1][1] + n2]
    out = []
    for v, n in blocks:
        out += [v] * n
    return out


def monotone(legs: list[dict]) -> list[dict]:
    """A player's overs must get less likely as the line rises (and unders more likely). Prices
    from an earlier snapshot and today's projection can disagree by a few points between lines;
    this evens that out."""
    groups = {}
    for l in legs:
        if l["m"] == "prop" and l.get("d") in ("over", "under") and l["s"] != "anytime_td":
            groups.setdefault((l["g"], l["c"]["p"], l["s"], l["d"]), []).append(l)
    for (_, _, _, side), ls in groups.items():
        ls.sort(key=lambda l: l["L"])
        for key in ("w", "wv"):
            if any(l.get(key) is None for l in ls):
                continue
            ys = [l[key] for l in ls]
            fit = _pav(ys) if side == "over" else _pav(ys[::-1])[::-1]
            for l, w in zip(ls, fit):
                l[key] = _r(w)
    return legs


# --------------------------------------------------------------------------
# Matchup tags (NFL): what each defense has allowed to each position this season
# --------------------------------------------------------------------------

MX_KIND = {"rec_yds": ("rec_yds", "rec yds"), "receptions": ("receptions", "catches"),
           "rush_yds": ("rush_yds", "rush yds"), "carries": ("rush_yds", "rush yds"),
           "pass_yds": ("pass_yds", "pass yds"), "completions": ("pass_yds", "pass yds"),
           "pass_att": ("pass_yds", "pass yds"), "pass_td": ("pass_td", "pass TDs")}


def defense_table(season: int) -> dict:
    """{(defense, position, stat): (per game allowed, rank 1 = most allowed, teams)} from this
    season's box scores (data/cache): the raw numbers the matchup dashboards show. A tag on the
    page, not an input to any chance (experiments/def_vs_position.py decides that)."""
    try:
        g = pd.read_csv(C.DATA / "cache" / f"player_games_{season}.csv")
        info = pd.read_csv(C.DATA / "cache" / f"player_info_{season}.csv")[["player_id", "position"]]
    except FileNotFoundError:
        return {}
    g = g[g["season_type"] == "REG"].merge(info, on="player_id", how="left")
    g["position"] = g["position"].replace({"FB": "RB", "HB": "RB"})
    games = g.groupby("defteam")["game_id"].nunique()
    out = {}
    for pos in ("WR", "TE", "RB", "QB"):
        d = g[g["position"] == pos].groupby("defteam")[["rec_yds", "receptions", "rush_yds", "pass_yds", "pass_td"]].sum()
        d = d.div(games.reindex(d.index), axis=0)
        for stat in d.columns:
            r = d[stat].rank(ascending=False, method="min")
            for team, v in d[stat].items():
                out[(team, pos, stat)] = (round(float(v), 1), int(r[team]), int(len(d)))
    return out


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def tag_matchups(legs: list[dict], games: dict, table: dict) -> None:
    for l in legs:
        if l.get("m") != "prop" or l.get("s") not in MX_KIND or not table:
            continue
        g = games.get(l["g"])
        pos = l.get("pos")
        if not g or pos not in ("WR", "TE", "RB", "QB"):
            continue
        dfn = g["home"] if l.get("tm") == g["away"] else g["away"]
        key, word = MX_KIND[l["s"]]
        if pos == "QB" and key in ("rec_yds", "receptions"):
            continue
        v = table.get((dfn, pos, key))
        if v:
            n, rank, teams = v
            most = rank <= teams / 2
            place = _ordinal(rank if most else teams + 1 - rank)
            l["mx"] = f"{dfn} allows {n:g} {word}/g to {pos}s ({place} {'most' if most else 'fewest'})"


# --------------------------------------------------------------------------
# College props (model only)
# --------------------------------------------------------------------------

def cfb_prop_legs(payload: dict, model, roles: dict, now: dt.datetime) -> list[dict]:
    """The NFL prop distributions on the college projections; no prices, tagged."""
    from . import propmarket as PM, props as P
    from .propoutlook import _survival, STEP
    try:
        players = PM.board_players(payload)
    except Exception as e:                       # the college payload may lack a field the NFL one has
        print(f"  college props skipped: {e}")
        return []
    if players.empty:
        return []
    if "tgt_share" not in players or players["tgt_share"].isna().all():
        players["tgt_share"] = players.get("rec_share", 0.0)
    ko = {g["id"]: g["kickoff"] for g in payload["slate"]["games"]}
    players = players[players["game_id"].map(lambda g: g in ko and _upcoming({"kickoff": ko[g]}, now))]
    players = players[~players["status"].fillna("").str.lower().str.match(r"^(out|doubtful|injured reserve)")]
    players = players[[roles.get((g, p)) in CFB_ROLES for g, p in zip(players["game_id"], players["player_id"])]]
    legs = []
    for stat in CFB_STATS:
        for grp in players["group"].unique():
            if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
                continue
            for c in P.SPECS[stat].proj:
                if c not in players:
                    players[c] = 0.0
            ok = (players["group"] == grp) & (P.mean(players, stat) > 0)
            d = players[ok].reset_index(drop=True)
            if d.empty:
                continue
            if stat == "anytime_td":
                over, _, _ = model.probs(d, stat, grp, np.full((len(d), 1), 0.5))
                for i, row in d.iterrows():
                    w = float(over[i][0])
                    if 0.2 <= w <= 0.97:
                        legs.append(_cfb_leg(row, stat, 0.5, "over", w, ko, roles))
                continue
            step, s = _survival(model, d, stat, grp)
            for i, row in d.iterrows():
                for side in ("over", "under"):
                    cand = []
                    for k in range(1, s.shape[1] + 1):
                        L = k * step - 0.5
                        w = float(s[i, k - 1]) if side == "over" else 1 - float(s[i, k - 1])
                        if MOVE_LO <= w <= MOVE_HI:
                            cand.append((L, w))
                    if len(cand) > CFB_MOVE_MAX:
                        idx = np.unique(np.linspace(0, len(cand) - 1, CFB_MOVE_MAX).round().astype(int))
                        cand = [cand[j] for j in idx]
                    for L, w in cand:
                        legs.append(_cfb_leg(row, stat, L, side, w, ko, roles))
    return legs


def _cfb_leg(row, stat, L, side, w, ko, roles):
    n = row.get("name", row["player_id"])
    if stat == "anytime_td":
        lab = f"{n} anytime TD"
    elif side == "over":
        lab = f"{n} {int(L + 0.5)}+ {STAT_LABEL.get(stat, stat)}"
    else:
        lab = f"{n} under {L:g} {STAT_LABEL.get(stat, stat)}"
    return {"lg": "cfb", "g": row["game_id"], "ko": ko[row["game_id"]], "m": "prop", "pl": n, "tm": row["team"], "pos": row.get("pos"),
            "s": stat, "lab": lab, "L": float(L), "d": side, "w": _r(w), "h": 0.0, "pm": _r(w), "pk": None, "bk": {},
            "src": "model", "t": ["not market-tested"],
            "c": {"k": "prop", "p": str(row["player_id"]), "tm": row["team"], "r": roles.get((row["game_id"], row["player_id"])), "s": stat, "dir": side}}


# --------------------------------------------------------------------------
# Correlations
# --------------------------------------------------------------------------

def correlations(prop_model) -> dict:
    pairs = dict(((prop_model.meta or {}).get("correlation") or {}).get("pairs") or {}) if prop_model else {}
    gl = json.loads(CORR.read_text()).get("pairs", {}) if CORR.exists() else {}
    return {"props": {k: v[0] for k, v in pairs.items()}, "game": {k: v[0] for k, v in gl.items()}}


GAME_LO, GAME_HI = 0.15, 0.97     # legs outside these chances are left out (nobody parlays them)


def compact(l: dict) -> dict:
    """Drop what the page can rebuild (league, kickoff and fair price come from elsewhere)."""
    c = dict(l["c"])
    if c.get("k") == "prop":
        c = {"k": "prop", "p": c["p"], "r": c.get("r")}
    elif c.get("k") == "cover":
        c = {"k": "cover", "tm": c["tm"]}
    out = {k: l[k] for k in ("g", "m", "lab", "L", "w", "h", "src") if k in l}
    for k in ("d", "wv", "pm", "pk", "tm", "s", "mx"):
        if l.get(k) is not None:
            out[k] = l[k]
    if l.get("bk"):
        out["bk"] = l["bk"]
    if l.get("t"):
        out["t"] = l["t"]
    if l["m"] == "prop":
        out["pl"] = l.get("pl")
    out["c"] = c
    return out


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def build(payload: dict, league: str, props_dir: Path | None = None, prop_model=None, now: dt.datetime | None = None,
          alt: dict | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    games = [g for g in payload["slate"]["games"] if _upcoming(g, now)]
    legs, gmeta = [], {}
    h = (payload.get("meta") or {}).get("haircut_factor")
    h = 1.0 if h is None or not (payload.get("meta") or {}).get("haircut_on", True) else float(h)
    for g in games:
        a = alt or {}
        ga = a.get(g["id"]) or a.get(g.get("game_id")) or a.get(str(g.get("game_id"))) or {}
        gl = game_legs(g, league, h, ga)
        if not gl and league == "cfb":
            continue
        legs += gl
        mkt = g.get("market") or {}
        gmeta[g["id"]] = {"away": g["away"], "home": g["home"], "ko": g["kickoff"], "date": g.get("date"), "time": g.get("time"),
                          "spread": mkt.get("spread"), "total": mkt.get("total"), "early": bool(g.get("early")),
                          "dk": "dk" in _book_prices(g), "fd": "fd" in _book_prices(g)}
    players = (payload.get("players") or {}).get("players") or []
    roles = _roles(players)
    names = {(p["game"], p["id"]): {"n": p["name"], "tm": p["team"], "pos": p["pos"]} for p in players}
    for g in payload["slate"]["games"]:
        names[("ko", g["id"])] = g["kickoff"]
    if league == "nfl" and props_dir is not None:
        files = sorted(props_dir.glob(f"{payload['meta']['season']}_w*.json"))
        if files:
            legs += nfl_prop_legs(files[-1], props_dir / "outlook.json", roles, names, now)
        try:
            tag_matchups(legs, {k: {"home": v["home"], "away": v["away"]} for k, v in gmeta.items()}, defense_table(int(payload["meta"]["season"])))
        except Exception as exc:
            print(f"  parlays: matchup tags skipped ({exc})")
    if league == "cfb" and prop_model is not None:
        legs += [l for l in cfb_prop_legs(payload, prop_model, roles, now) if l["g"] in gmeta]
    legs = [compact(l) for l in legs if l["g"] in gmeta and l.get("w") is not None and GAME_LO <= l["w"] <= GAME_HI]
    for i, l in enumerate(legs):
        l["i"] = i
    meta = payload.get("meta") or {}
    return {"league": league, "built": now.strftime("%Y-%m-%d %H:%M UTC"), "generated": meta.get("generated"),
            "season": meta.get("season"), "week": meta.get("week"), "market_as_of": (payload.get("slate") or {}).get("market_as_of"),
            "games": gmeta, "legs": legs,
            "corr": correlations(prop_model)}


def _nfl_alt(payload: dict) -> dict:
    """The NFL board's games matched to the stored alternate ladders (names are the teams' own)."""
    from . import market as M
    gs = pd.DataFrame([{"game_id": g["id"], "home_team": g["home"], "away_team": g["away"],
                        "kickoff": pd.Timestamp(g["kickoff"])} for g in payload["slate"]["games"]])
    return M.alt_quotes(gs, "nfl") if len(gs) else {}


def write(league: str, payload_path: Path | None = None, alt: dict | None = None) -> Path:
    from . import props as P
    payload_path = payload_path or (C.DOCS / ("data.json" if league == "nfl" else "cfb/data.json"))
    payload = json.loads(Path(payload_path).read_text())
    try:
        model = P.PropModel.load()
    except Exception:
        model = None
    if alt is None and league == "nfl":
        try:
            alt = _nfl_alt(payload)
        except Exception as exc:
            print(f"  parlays: alternate ladders skipped ({exc})")
    out = build(payload, league, props_dir=C.DOCS / "props", prop_model=model, alt=alt)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{league}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    n = len(out["legs"])
    print(f"parlays: {league} {n} legs in {len(out['games'])} games -> {path} ({path.stat().st_size // 1024} KB)")
    return path


def copy_page() -> Path:
    """site/parlays.html -> docs/parlays.html (the page reads docs/parlays/*.json)."""
    dst = C.DOCS / "parlays.html"
    dst.write_text((C.SITE / "parlays.html").read_text())
    return dst
