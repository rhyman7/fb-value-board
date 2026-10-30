"""The bet sheet: what each flagged bet is made of, which bucket it belongs to, and how the
sharp line has moved since it opened. Used by both boards (slate.build, cfbmodel/board.entries).

Every bet on the board carries:

  tier     price    a retail price beats the sharp book's own no-vig number at this line
                    with no model at all (bets.shop: beats_sharp) by PRICE_MIN or more. The
                    only bucket that makes money with a zero-skill model, so it is listed first.
           opener   the model is OPENER_GAP points or more off the sharp book's opening line
                    on this side (3+ on totals). The only edge the backtests have found
                    (openers in data.json): the line then tends to move toward the model.
           wind     an under in an outdoor game with 10-19 mph of wind (backtest.wind_report).
           model    everything else the market-adjusted projection makes +EV on a spread or
                    total. Moneylines are never here: on the graded bets since 2019 they have
                    lost at every claimed edge, so a moneyline is a bet only in the price tier.
           None     not +EV on the market-adjusted projection, or not priced
  gap      model minus the current anchor line, signed toward this side (points; None for
           moneylines, where it is the spread gap)
  opener_gap  the same against the sharp book's first stored line, else the board's first
              line, else the nfelo/cfbfastR opener
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


def _sgn(side: str) -> int:
    """+1 when more home margin / more points helps this side."""
    return 1 if side in ("home", "over") else -1


def _open_line(g: dict, market: str):
    """The opening line to measure against: the sharp book's first stored quote when the log
    caught it at the open (its first snapshot no later than the day this board first saw the
    game), else the first line this board saw, else the data's opener. Spreads home-positive."""
    key = "spread" if market in ("Spread", "Moneyline") else "total"
    so, op = g.get("sharp_open") or {}, g.get("open") or {}
    if so.get(key) is not None and (op.get("seen") is None or str(so.get("seen", ""))[:10] <= str(op["seen"])[:10]):
        return float(so[key]), "sharp"
    if op.get(key) is not None:
        return float(op[key]), "board"
    v = g.get("opening_spread") if key == "spread" else g.get("opening_total")
    return (float(v), "data") if v is not None else (None, None)


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
        ev = b.get("ev") or 0.0
        best_ev = b.get("best_ev")
        plus = max(ev, best_ev if best_ev is not None else ev) > 0
        wind = g.get("wind") or {}
        windy = bool(wind.get("under_angle")) and (g.get("roof") in (None, "outdoors", "open", "unknown"))
        if b.get("beats_sharp") and (b.get("sharp_ev") or 0.0) >= PRICE_MIN:
            b["tier"] = "price"
        elif mk == "Moneyline":
            b["tier"] = None                      # a moneyline is only a bet when its price beats the sharp line
        elif plus and b["opener_gap"] is not None and b["opener_gap"] >= (TOTAL_GAP if is_total else thresh):
            b["tier"] = "opener"
        elif is_total and side == "under" and windy and league == "nfl":
            b["tier"] = "wind"
        elif plus:
            b["tier"] = "model"
        else:
            b["tier"] = None
    # derivatives (nflmodel/derivs.py): a price bet when a retail price beats the sharp book's
    # own derivative quote by PRICE_MIN; never an opener, wind or model bet, since the curve's
    # opinion on derivatives has no graded history yet (FEATURES.md, "Derivative markets")
    for b in g.get("derivs") or []:
        b["gap"] = b["opener_gap"] = b["opener_src"] = b["move"] = None
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
    report's latest era for wind unders."""
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
        if h.get("n", 0) >= 200:
            return h.get("win_pct")
        return None if (h.get("roi") or 0) <= 0 else row.get("win_pct")
    return None


def bucket_stakes(games: list[dict], openers: dict | None, wind: dict | None, kelly: float, max_stake: float) -> int:
    """For opener and wind bets the model sizes no stake for, a stake from the bucket's own
    record: quarter (kelly) Kelly on the bucket's win rate shrunk halfway to break-even, at the
    net price of the book taken. Written as `bucket_stake` on the bet (0 when there is none)
    so the page and the ledger size it the same way. Returns the bets given one."""
    from . import odds as O
    n = 0
    for g in games:
        for b in g.get("bets") or []:
            b["bucket_stake"] = 0.0
            if b.get("tier") not in ("opener", "wind"):
                continue
            q = next((x for x in b.get("books") or [] if x["book"] == b.get("best")), None)
            sized = (q["stake"] if q else b.get("stake", 0.0)) or 0.0
            if sized > 0:
                continue
            w = _rate(openers, wind, b)
            if w is None or w <= BREAKEVEN:
                continue
            p = (w + BREAKEVEN) / 2
            price = q["odds_net"] if q else b["odds"]
            k = float(O.kelly(p, price))
            b["bucket_stake"] = round(min(k * kelly, max_stake), 4)
            n += int(b["bucket_stake"] > 0)
    return n
