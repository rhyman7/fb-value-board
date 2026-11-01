"""The bet sheet: what each flagged bet is made of, which bucket it belongs to, and how the
sharp line has moved since it opened. Used by both boards (slate.build, cfbmodel/board.entries).

Every bet on the board carries:

  tier     price    a retail price beats the sharp book's own no-vig number at this line
                    with no model at all (bets.shop: beats_sharp) by PRICE_MIN or more. The
                    only bucket that makes money with a zero-skill model, so it is listed first.
           opener   the model is OPENER_GAP points or more off the opening line on this side
                    (3+ on totals), and the market has not yet taken it: the line is within
                    MOVED points of the opener, or the gap at the current line still clears
                    the threshold. The only edge the backtests have found (openers in
                    data.json): the line then tends to move toward the model, and once it has,
                    the same sides win below break-even at the close (close_win_pct). A bet
                    that was off the opener and has been priced out carries missed: True.
                    College spreads are never here: off the opener they have lost on the
                    holdout seasons at every threshold below 5; college totals are.
           wind     an under in an outdoor game with 10-19 mph of wind (backtest.wind_report),
                    while the total has not already dropped WIND_MOVED points since the open
                    (the market has priced the wind; missed: True).
           model    everything else the market-adjusted projection makes +EV on a spread or
                    total. Moneylines are never here: on the graded bets since 2019 they have
                    lost at every claimed edge, so a moneyline is a bet only in the price tier.
           None     not +EV on the market-adjusted projection, or not priced
  gap      model minus the current anchor line, signed toward this side (points; None for
           moneylines, where it is the spread gap)
  opener_gap  the same against the opener the bucket was backtested on (nfelo for the NFL,
              cfbfastR for college), else the sharp book's first stored line, else the
              board's first line
  move     how far the sharp line has moved from that opener toward this side (points;
           positive means the market has come the model's way)
  why      the rows of the model's number for this market (matchup.breakdown), largest
           first, each {label, value, news}: news marks the inputs that are this week's
           information (quarterbacks, injuries, snap shares, wind) rather than the
           ratings themselves

A game's derivative sides (`g["derivs"]`, nflmodel/derivs.py: team totals and first-half
lines, each with deriv: True) get the same keys: tier `price` when a retail price beats the
sharp book's own derivative quote by PRICE_MIN, else None (no opener, wind or model tier:
the curve's opinion on derivatives is not backtested against derivative lines yet), and
empty gap, move and why.
"""
from __future__ import annotations

OPENER_GAP = {"nfl": 2.0, "cfb": 3.0}       # points off the opener (spread) that count as the opener bucket
PRICE_MIN = 0.01                             # a price must beat the sharp no-vig number by this much per dollar to be a bet
TOTAL_GAP = 3.0
WIND_LO, WIND_HI = 10.0, 20.0
# breakdown keys that carry this week's news rather than the standing ratings
NEWS_KEYS = {"home_qb", "away_qb", "qb", "ol", "depth1", "snaps", "wind", "rest", "pre", "ret"}
TIERS = ["price", "opener", "wind", "model"]
# Staking rules from the audit (Nov 1, 2026, Ryan approved):
WIND_STAKE_SCALE = 0.5     # wind-under stakes at half until the forecast grade has a season (FEATURES.md, "The wind under")
CLV_GUARD_N = 50           # a kind of bet whose live closing line value, over at least this many graded bets,
CLV_GUARD_Z = 2.0          # is below zero by more than this many standard errors carries no stake until it recovers
MOVED = 0.5                # points toward the model after which an opener bet counts as taken by the market
WIND_MOVED = 1.5           # points the total has dropped after which the wind is priced in
# line movement (line_moves): the gap off the opener that makes a side worth watching, (spread, total)
MOVE_GAP = {"nfl": (2.0, 3.0), "cfb": (3.0, 3.0)}
TEASER_CLEAR = 0.7385      # a 6-point leg's chance that makes a two-team teaser at -120 break even (sqrt(120/220))


def _sgn(side: str) -> int:
    """+1 when more home margin / more points helps this side."""
    return 1 if side in ("home", "over") else -1


def _open_line(g: dict, market: str):
    """The opening line to measure against: the data's opener first (nfelo, cfbfastR), since
    that is the number the opener bucket was backtested on; else the sharp book's first stored
    quote when the log caught it at the open (its first snapshot no later than the day this
    board first saw the game); else the first line this board saw. Spreads home-positive."""
    key = "spread" if market in ("Spread", "Moneyline") else "total"
    v = g.get("opening_spread") if key == "spread" else g.get("opening_total")
    if v is not None:
        return float(v), "data"
    so, op = g.get("sharp_open") or {}, g.get("open") or {}
    if so.get(key) is not None and (op.get("seen") is None or str(so.get("seen", ""))[:10] <= str(op["seen"])[:10]):
        return float(so[key]), "sharp"
    if op.get(key) is not None:
        return float(op[key]), "board"
    return None, None


def why_rows(detail: dict | None, market: str) -> list[dict]:
    """The model's number for this market as its rows, largest first."""
    bd = (detail or {}).get("breakdown") or {}
    rows = bd.get("total" if market == "Total" else "margin") or []
    out = []
    for r in rows:
        if r.get("value") is None:
            continue
        if market == "Total" and r.get("key") == "base":
            continue                              # the league baseline is not a reason
        out.append({"label": r.get("label"), "value": round(float(r["value"]), 2), "news": r.get("key") in NEWS_KEYS})
    return sorted(out, key=lambda r: -abs(r["value"]))


def annotate(g: dict, league: str = "nfl") -> None:
    """Add tier, gap, opener_gap, move and why to each bet of a board game entry, in place."""
    thresh = OPENER_GAP.get(league, 2.0)
    for b in g.get("bets") or []:
        mk, side = b["market"], b["side"]
        s = _sgn(side)
        is_total = mk == "Total"
        model = g["model"]["total" if is_total else "margin"]
        now = g["market"]["total" if is_total else "spread"]
        b["gap"] = round(s * (model - now), 2)
        opened, src = _open_line(g, mk)
        if opened is not None:
            b["opener_gap"] = round(s * (model - opened), 2)
            b["opener_src"] = src
            b["move"] = round(s * (now - opened), 2)
        else:
            b["opener_gap"] = b["opener_src"] = b["move"] = None
        b["why"] = why_rows(g.get("detail"), mk)
        b["missed"] = False
        ev = b.get("ev") or 0.0
        best_ev = b.get("best_ev")
        plus = max(ev, best_ev if best_ev is not None else ev) > 0
        need = TOTAL_GAP if is_total else thresh
        off = b["opener_gap"] is not None and b["opener_gap"] >= need and (is_total or league != "cfb")
        wind = g.get("wind") or {}
        windy = bool(wind.get("under_angle")) and (g.get("roof") in (None, "outdoors", "open", "unknown"))
        wind_under = is_total and side == "under" and windy and league == "nfl"
        taken = off and b["move"] > MOVED and b["gap"] < need
        priced = wind_under and b["move"] is not None and b["move"] >= WIND_MOVED
        final_side = g.get("final_week") and not is_total      # config.FINAL_WEEK_MARKET: the market knows the final week better
        if b.get("beats_sharp") and (b.get("sharp_ev") or 0.0) >= PRICE_MIN:
            b["tier"] = "price"
        elif final_side:
            b["tier"] = None
        elif mk == "Moneyline":
            b["tier"] = None                      # a moneyline is only a bet when its price beats the sharp line
        elif off and plus and not taken:
            b["tier"] = "opener"                  # off the opener and the market has not taken it yet
        elif wind_under and not priced:
            b["tier"] = "wind"
        else:
            b["tier"] = "model" if plus else None
            b["missed"] = taken or priced         # the market has moved to the model, or priced the wind
    # derivatives (nflmodel/derivs.py): a price bet when a retail price beats the sharp book's
    # own derivative quote by PRICE_MIN; never an opener, wind or model bet, since the curve's
    # opinion on derivatives has no graded history yet (FEATURES.md, "Derivative markets")
    for b in g.get("derivs") or []:
        b["gap"] = b["opener_gap"] = b["opener_src"] = b["move"] = None
        b["missed"] = False
        b["why"] = []
        b["tier"] = "price" if b.get("beats_sharp") and (b.get("sharp_ev") or 0.0) >= PRICE_MIN else None


def inputs_stamp(generated: str, **as_of) -> dict:
    """When each input was last read, for the sheet's "data as of" line: generated plus any
    of market, injuries, lines, designations (strings or None)."""
    return {"generated": generated, **{k: v for k, v in as_of.items() if v}}


BREAKEVEN = 0.5238          # at -110


def _rate(openers: dict | None, wind: dict | None, b: dict) -> float | None:
    """The measured win rate of the bet's bucket: the opener report's 2+ (3+ totals) row on the
    holdout seasons when it has 200 bets, else all seasons unless the holdout lost; the wind
    report's latest era for wind unders. Once the line has moved toward an opener bet, the
    rate is the bucket's win rate at the closing line (close_win_pct), which is what is left
    of the edge after the market has moved."""
    t = b.get("tier")
    if t == "wind" and wind and wind.get("eras"):
        return (wind["eras"][-1] or {}).get("win_pct")
    if t == "opener" and openers:
        src = openers.get("total") if b["market"] == "Total" and openers.get("total") else openers
        thr = 3 if b["market"] == "Total" else 2
        row = next((r for r in src.get("rows") or [] if r.get("threshold") == thr), None)
        if not row:
            return None
        h = row.get("holdout") or {}
        key = "close_win_pct" if (b.get("move") or 0) > 0 else "win_pct"
        if h.get("n", 0) >= 200:
            return h.get(key)
        return None if (h.get("roi") or 0) <= 0 else row.get(key)
    return None


def bucket_stakes(games: list[dict], openers: dict | None, wind: dict | None, kelly: float, max_stake: float) -> int:
    """For opener and wind bets the model sizes no stake for, a stake from the bucket's own
    record: quarter (kelly) Kelly on the sharp line's chance for the side at that book's line
    plus the bucket's edge over a fair line (its win rate shrunk halfway to break-even, less
    0.5), at the
    net price of the book with the best price against the sharp line (ev_line; the bucket's
    rate was measured at about -110 at the opener, so the book is chosen on price, never on
    the model's EV). Written as `bucket_stake` and `bucket_book` on the bet (0 and None when
    there is none) so the page and the ledger size it the same way. Returns the bets given one."""
    from . import odds as O
    n = 0
    for g in games:
        for b in g.get("bets") or []:
            b["bucket_stake"], b["bucket_book"] = 0.0, None
            if b.get("tier") not in ("opener", "wind"):
                continue
            best = next((x for x in b.get("books") or [] if x["book"] == b.get("best")), None)
            sized = (best["stake"] if best else b.get("stake", 0.0)) or 0.0
            if sized > 0:
                continue
            priced = [x for x in b.get("books") or [] if x.get("ev_line") is not None]
            q = max(priced, key=lambda x: x["ev_line"]) if priced else best
            w = _rate(openers, wind, b)
            if w is None or w <= BREAKEVEN:
                continue
            # the bucket's edge is an edge over a fair -110 line: half of (w - 0.5) after the shrink to
            # break-even, added to what the sharp line says this side's chance is at this book's line
            # and price (from ev_line), so a plus-money price on a worse number is not sized as an edge
            lift = (w + BREAKEVEN) / 2 - 0.5
            price = q["odds_net"] if q else b["odds"]
            push = (q.get("p_push") or 0.0) if q else 0.0
            if q is not None and q.get("ev_line") is not None:
                p = (q["ev_line"] + 1 - push) / float(O.american_to_decimal(price)) + lift
            else:
                p = 0.5 + lift
            k = float(O.kelly(p, price, push))
            scale = WIND_STAKE_SCALE if b.get("tier") == "wind" else 1.0
            b["bucket_stake"] = round(min(k * kelly * scale, max_stake), 4)
            b["bucket_book"] = q["book"] if q and b["bucket_stake"] > 0 else None
            n += int(b["bucket_stake"] > 0)
    return n


def clv_guard(games: list[dict], ledger_summary: dict | None, n_min: int = CLV_GUARD_N, z: float = CLV_GUARD_Z) -> dict:
    """Turn off the stake of every bet of a kind the live ledger says is losing to the close:
    over at least n_min bets with closing line value, the average below zero by more than z
    standard errors. The bets stay on the sheet, marked, at no stake. Returns {tier: reason} for
    the kinds turned off (meta.muted_tiers), {} when none is."""
    muted = {}
    for tier, r in ((ledger_summary or {}).get("by_tier") or {}).items():
        n, clv, se = r.get("clv_n") or 0, r.get("clv"), r.get("clv_se")
        if n >= n_min and clv is not None and se and clv + z * se < 0:
            muted[tier] = f"closing line value {clv:+.1%} ± {se:.1%} over {n} graded bets"
    if not muted:
        return {}
    for g in games:
        for b in list(g.get("bets") or []) + list(g.get("derivs") or []):
            if (b.get("tier") or "model") not in muted:
                continue
            b["muted"] = True
            for key in ("stake", "bucket_stake"):
                if b.get(key):
                    b[key] = 0.0
            for x in b.get("books") or []:
                if x.get("stake"):
                    x["stake"] = 0.0
    return muted


# ---------------- line movement ----------------
# The opener edge is the line moving to the model: these are the sides where the model
# disagrees with the opener by MOVE_GAP or more, with how far the sharp line has come since.
# Written into data.json by the board builds (slate["moves"]) and, after every Market lines
# snapshot, to docs/moves.json and docs/cfb/moves.json by `run_market.py alerts` with
# Pinnacle's newest line, which is also what the push alerts are made from.

def _teaser_leg(g: dict, market: str, side: str, now: float) -> dict | None:
    """The 6-point teaser leg on this side at the line `now`: the teased line (from this
    side's view), its chance from the board's alternate-line ladder (market-anchored curve),
    whether it clears a two-team teaser at -120, and whether it is a Wong leg (NFL)."""
    from . import alts as A
    key = "total" if market == "Total" else "spread"
    teased = now - 6 if side in ("home", "over") else now + 6
    row = next((r for r in ((g.get("alts") or {}).get(key) or []) if abs(float(r["line"]) - teased) < 1e-6), None)
    if row is None or row.get(side) is None:
        return None
    p = float(row[side])
    shown = teased if key == "total" else (-teased if side == "home" else teased)
    return {"line": round(shown, 1), "prob": round(p, 4), "clears": p >= TEASER_CLEAR,
            "wong": key == "spread" and A.wong(now, side)}


def _pick(g: dict, market: str, side: str, line: float) -> str:
    if market == "Total":
        return f"{'Over' if side == 'over' else 'Under'} {line:g}"
    team = g["home"] if side == "home" else g["away"]
    v = -line if side == "home" else line
    return f"{team} {'PK' if v == 0 else f'{v:+g}'}"


def move_status(left: float, move: float, need: float) -> str:
    """Live: the gap at the current line still clears the threshold, or the line has moved no
    more than MOVED toward the model (the sheet's opener tier, annotate). Partly gone: the line
    has come part of the way. Missed: it has reached or passed the model."""
    if left <= 0:
        return "Missed"
    return "Live" if left >= need or move <= MOVED else "Partly gone"


def bucket_record(openers: dict | None, market: str, gap: float) -> dict | None:
    """The opener report's row for a side this far off the opener: the highest threshold the
    gap clears, when that row made money on the holdout seasons (else None: no backtested
    edge). {threshold, win_pct, roi, n} on the holdout seasons."""
    src = (openers or {}).get("total") if market == "Total" else openers
    rows = [r for r in (src or {}).get("rows") or [] if (r.get("threshold") or 0) > 0 and gap >= r["threshold"]]
    if not rows:
        return None
    r = max(rows, key=lambda r: r["threshold"])
    h = r.get("holdout") or r
    if (h.get("roi") or 0) <= 0:
        return None
    return {"threshold": r["threshold"], "win_pct": h.get("win_pct"), "roi": h.get("roi"), "n": h.get("n")}


def line_moves(games: list[dict], league: str, openers: dict | None = None, now: dict | None = None) -> list[dict]:
    """One row per game and market (spread, total) where the model is MOVE_GAP or more off the
    opener (or off the current line): the model's side, opener -> now, how much of the gap
    the market has closed, the status, and the model's side as a 6-point teaser leg. `now`
    overrides the board's anchor line per game ({id: {"spread": .., "total": ..}}, home-positive)
    with a fresher sharp quote (`fresh` on the row says which). `bucket` is the opener report's
    record for a side this far off (bucket_record), None when no backtested edge covers it:
    those rows are information, and the push alerts leave them out."""
    rows = []
    for g in games:
        model = g.get("model") or {}
        for market, mkey, key in (("Spread", "margin", "spread"), ("Total", "total", "total")):
            m = model.get(mkey)
            if market == "Spread" and g.get("final_week"):
                continue                                  # config.FINAL_WEEK_MARKET: no model sides in the final week
            fresh = ((now or {}).get(g["id"]) or {}).get(key)
            cur = fresh if fresh is not None else (g.get("market") or {}).get(key)
            opened, src = _open_line(g, market)
            if m is None or cur is None or opened is None:
                continue
            need = MOVE_GAP.get(league, (2.0, 3.0))[1 if market == "Total" else 0]
            side = ("home" if m > opened else "away") if market == "Spread" else ("over" if m > opened else "under")
            s = _sgn(side)
            gap_open, left, move = s * (m - opened), s * (m - float(cur)), s * (float(cur) - opened)
            if gap_open < need and left < need:
                continue
            if gap_open < need:                    # only off the current line: the line moved away from the model
                side = ("home" if m > cur else "away") if market == "Spread" else ("over" if m > cur else "under")
                s = _sgn(side)
                gap_open, left, move = s * (m - opened), s * (m - float(cur)), s * (float(cur) - opened)
            rows.append({
                "game": g["id"], "away": g["away"], "home": g["home"], "kickoff": g.get("kickoff"),
                "market": market, "side": side, "pick": _pick(g, market, side, float(cur)),
                "model": round(float(m), 1), "opener": float(opened), "opener_src": src, "now": float(cur),
                "fresh": fresh is not None, "gap_open": round(gap_open, 1), "left": round(left, 1), "move": round(move, 1),
                "closed": round(min(max(move / gap_open, 0.0), 1.0), 2) if gap_open > 0 else None,
                "status": move_status(left, move, need),
                # the bucket is judged on the gap at the opener, which is what was backtested
                "bucket": bucket_record(openers, market, gap_open) if gap_open >= need else None,
                "tease": _teaser_leg(g, market, side, float(cur)),
            })
    order = {"Live": 0, "Partly gone": 1, "Missed": 2}
    return sorted(rows, key=lambda r: (order[r["status"]], -r["left"], r["kickoff"] or ""))


# ---------------- weekly exposure cap ----------------

def taken_stake(b: dict) -> float:
    """The stake the sheet puts on a bet: at the book it is taken at (the one beating the sharp
    line for a price bet, else the bucket's book or the best book), else the side's own; the
    bucket stake when the model sizes nothing."""
    if not b.get("tier"):
        return 0.0
    want = b.get("beats_sharp") if b.get("tier") == "price" and b.get("beats_sharp") else (b.get("bucket_book") or b.get("best"))
    q = next((x for x in b.get("books") or [] if x["book"] == want), None)
    s = float((q["stake"] if q else b.get("stake")) or 0.0)
    return s if s > 0 else float(b.get("bucket_stake") or 0.0)


def cap_week(games: list[dict], cap: float) -> dict:
    """Scale every stake of a week down together when the bets the sheet flags that week add up
    to more than `cap` of bankroll (on top of the per-bet and per-game caps; slate.cap_game is
    the same rule per game). Every stake in the week, at the anchor, at each book and the bucket
    stake, is multiplied by one factor, written as `week_scale` on each game. Returns
    {week: {"stake": the sum before the cap, "scale": the factor}} for the weeks on the board."""
    out = {}
    weeks: dict = {}
    for g in games:
        weeks.setdefault(g.get("week"), []).append(g)
    for wk, gs in weeks.items():
        bets = [b for g in gs for b in list(g.get("bets") or []) + list(g.get("derivs") or [])]
        total = sum(taken_stake(b) for b in bets)
        scale = 1.0 if total <= cap or total <= 0 else cap / total
        if scale < 1.0:
            for b in bets:
                b["stake"] = round(float(b.get("stake") or 0.0) * scale, 4)
                if b.get("bucket_stake"):
                    b["bucket_stake"] = round(b["bucket_stake"] * scale, 4)
                for q in b.get("books") or []:
                    q["stake"] = round(float(q.get("stake") or 0.0) * scale, 4)
        for g in gs:
            g["week_scale"] = round(scale, 4)
        out[str(wk)] = {"stake": round(total, 4), "scale": round(scale, 4)}
    return out
