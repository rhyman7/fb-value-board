"""Player props against the market: vig-free prices, edge, expected value and stakes.

For each player prop the books post, this works out

  market   each book's no-vig chance of the over, by simple scaling and by the power
           method, and a weighted consensus across books (sharper books count more,
           oddsapi.BOOK_WEIGHT). Anytime-TD markets are often posted one side only; those
           assume a two-way overround (ONE_SIDED_MARGIN, refitted from any two-sided
           touchdown markets in the log).
  ours     P(over), P(push), P(under) at that line from the projection (props.PropModel)
  edge     ours minus the consensus, on the side being priced
  EV       per unit staked at the best price posted: p_win x payout - p_lose
  stake    a quarter of the Kelly stake at our probability, capped per bet and per game

A side is flagged only when the edge clears MIN_EDGE, the line is liquid (two or more
books at that line, or a sharp book, and a hold that is not extreme) and the player's
role is clear (see role_clear).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config as C, odds as O
from .oddsapi import BOOK_WEIGHT
from .props import SPECS, PropModel

MARKET_STAT = {s.odds_key: k for k, s in SPECS.items() if s.odds_key}
# Alternate yardage ladders (200+, 225+, 250+ ...): priced off the same distribution as the
# main line. Ryan bets lower rungs more than the main line (Oct 2026).
ALT_MARKETS = ["player_pass_yds_alternate", "player_rush_yds_alternate", "player_reception_yds_alternate"]
MARKET_STAT.update({m: MARKET_STAT[m.removesuffix("_alternate")] for m in ALT_MARKETS})
# The markets fetched: every one the model prices (ten). About 15 games x 10 markets x 3
# snapshots (open, pricing, close) = 450 credits a week. Change with PROPS_MARKETS.
DEFAULT_MARKETS = [s.odds_key for s in SPECS.values() if s.odds_key] + ALT_MARKETS

MIN_EDGE = 0.05            # in probability; see PROPS.md for how it was chosen
# Flags are paused (Oct 2026): against real 2025 DraftKings and FanDuel lines the model's
# chances were worse than the books' vig-free ones, and the sides these rules would have
# flagged broke even (PROPS.md). Every side is still priced and logged; nothing is flagged
# until a market-anchored version beats the books out of sample.
FLAGS_ON = False
MIN_BOOKS = 2              # books posting this player's market (at any line): with DraftKings and
                           # FanDuel only, both; their lines often differ by a half point
SHARP: set = set()         # no sharp book in the feed (DraftKings and FanDuel only)
MAX_HOLD = 0.10            # a two-way market holding more than this is not treated as liquid
MAX_ODDS = 400             # longshots beyond +400 / -400 are not flagged
ONE_SIDED_MARGIN = 0.08    # assumed two-way overround for one-sided markets (anytime TD)
ALT_MARGIN = 0.10          # the same for a ladder rung posted over-only: an assumption (ladders
                           # usually carry more margin than the main line), to be checked
                           # against rungs posted with both sides once the log has them
KELLY_FRACTION = min(C.KELLY_FRACTION, 0.25)
MAX_STAKE_BET = 0.01       # share of bankroll on any one prop
MAX_STAKE_GAME = 0.03      # across all props in one game
ROLE_SHIFT = 0.25          # usage moved this much (relative) by teammates' absences = role unclear
MIN_GAMES = 4.0            # weighted games of history needed for a clear role

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
NAME_TO_ABBR = {v: k for k, v in C.TEAM_NAMES.items()}


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`]", "", s)
    s = _SUFFIX.sub("", s)
    return re.sub(r"[^a-z]+", " ", s).strip()


# --------------------------------------------------------------------------
# The board's players as a projection table
# --------------------------------------------------------------------------

GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}


def board_players(payload: dict) -> pd.DataFrame:
    """Upcoming players from docs/data.json, in the column layout props.PropModel reads."""
    rows = []
    for p in (payload.get("players") or {}).get("players", []):
        r = {"player_id": p["id"], "name": p["name"], "pos": p["pos"], "group": GROUP.get(p["pos"], p["pos"]),
             "team": p["team"], "opp": p["opp"], "game_id": p["game"], "week": p["week"], "status": p.get("status"),
             "active": 1, "played": False, "started": np.nan}
        for k, v in (p.get("proj") or {}).items():
            r[f"{k}_proj"] = v if v is not None else 0.0
        why = p.get("why") or {}
        for k in ("tgt_share", "car_share", "att_share", "tgt_share_hist", "car_share_hist", "games"):
            r[k] = why.get(k)
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        for c in ("tgt_share", "car_share", "att_share"):
            df[c] = df[c].fillna(0.0)
    return df


def role_clear(df: pd.DataFrame) -> pd.Series:
    """No injury designation beyond Probable, an established role (enough games), and usage
    not much moved by teammates being out. Quarterbacks: the projected starter."""
    status_ok = df["status"].isna() | df["status"].isin(["Probable"])
    games_ok = df["games"].fillna(0) >= MIN_GAMES
    shift = pd.Series(0.0, index=df.index)
    for now, hist in (("tgt_share", "tgt_share_hist"), ("car_share", "car_share_hist")):
        h = df[hist].fillna(0)
        rel = (df[now] - h).abs() / h.where(h > 0.05)
        shift = np.maximum(shift, rel.fillna(0))
    qb_ok = (df["group"] != "QB") | (df["att_share"] >= 0.99)
    return status_ok & games_ok & (shift <= ROLE_SHIFT) & qb_ok


# --------------------------------------------------------------------------
# Matching books' rows to the board
# --------------------------------------------------------------------------

def match(odds_rows: pd.DataFrame, players: pd.DataFrame, slate_games: list[dict]) -> pd.DataFrame:
    """Attach game_id and player_id to each odds row (by teams, date and player name)."""
    if odds_rows.empty:
        return odds_rows.assign(game_id=pd.Series(dtype=object), player_id=pd.Series(dtype=object))
    o = odds_rows.copy()
    o["home"] = o["home_name"].map(NAME_TO_ABBR)
    o["away"] = o["away_name"].map(NAME_TO_ABBR)
    gid = {(g["home"], g["away"]): g["id"] for g in slate_games}
    o["game_id"] = [gid.get((h, a)) for h, a in zip(o["home"], o["away"])]
    o["nname"] = o["player"].map(norm_name)
    pl = players.assign(nname=players["name"].map(norm_name))
    # a name used by two players in the same game is ambiguous: leave both out
    pl = pl[~pl.duplicated(["game_id", "nname"], keep=False)]
    o = o.merge(pl[["game_id", "nname", "player_id"]], on=["game_id", "nname"], how="left")
    return o


# --------------------------------------------------------------------------
# Market prices
# --------------------------------------------------------------------------

def consensus(o: pd.DataFrame, one_sided_margin: float = ONE_SIDED_MARGIN) -> pd.DataFrame:
    """One row per (game, player, market, line): every book's over/under prices, each
    book's no-vig chance of the over (scaling and power), and the weighted consensus."""
    o = o[o["player_id"].notna()].copy()
    o["line"] = np.where(o["market"] == "player_anytime_td", 0.5, o["line"])
    o["side"] = o["side"].replace({"yes": "over", "no": "under"})
    o = o[o["side"].isin(["over", "under"]) & o["line"].notna()]
    key = ["event_id", "commence", "game_id", "player_id", "player", "market", "line"]
    wide = o.pivot_table(index=key + ["book"], columns="side", values="price", aggfunc="last").reset_index()
    for c in ("over", "under"):
        if c not in wide:
            wide[c] = np.nan
    two = wide["over"].notna() & wide["under"].notna()
    wide["hold"] = np.where(two, O.hold(wide["over"].fillna(-110), wide["under"].fillna(-110)), np.nan)
    pa, _ = O.no_vig(wide["over"].fillna(-110), wide["under"].fillna(-110))
    pw, _ = O.no_vig_power(wide["over"].fillna(-110), wide["under"].fillna(-110))
    one = wide["over"].notna() & wide["under"].isna()
    marg = np.where(wide["market"].str.endswith("_alternate"), ALT_MARGIN, one_sided_margin)
    wide["fair_prop"] = np.where(two, pa, np.where(one, O.one_sided_fair(wide["over"].fillna(100), marg, "proportional"), np.nan))
    wide["fair_power"] = np.where(two, pw, np.where(one, O.one_sided_fair(wide["over"].fillna(100), marg, "power"), np.nan))
    # an under posted alone (rare) gives a fair over too
    lone_u = (wide["under"].notna() & wide["over"].isna()).to_numpy()
    wide.loc[lone_u, "fair_prop"] = 1 - O.one_sided_fair(wide.loc[lone_u, "under"], marg[lone_u], "proportional")
    wide.loc[lone_u, "fair_power"] = 1 - O.one_sided_fair(wide.loc[lone_u, "under"], marg[lone_u], "power")
    wide["w"] = wide["book"].map(BOOK_WEIGHT).fillna(0.5)
    wide["one_sided"] = ~two
    books_mkt = wide.groupby(["game_id", "player_id", "market"])["book"].nunique().rename("books_market")

    out = []
    for k, g in wide.groupby(key, sort=False):
        w = g["w"].to_numpy()
        rec = dict(zip(key, k))
        rec.update({
            "books": {b: [None if pd.isna(x) else int(x) for x in (ov, un)] for b, ov, un in zip(g["book"], g["over"], g["under"])},
            "n_books": int(len(g)), "sharp": bool(g["book"].isin(SHARP).any()),
            "one_sided": bool(g["one_sided"].all()),
            "hold": float(np.nanmedian(g["hold"])) if g["hold"].notna().any() else np.nan,
            "fair_prop": float(np.average(g["fair_prop"], weights=w)),
            "fair_power": float(np.average(g["fair_power"], weights=w)),
        })
        for side in ("over", "under"):
            s = g[g[side].notna()]
            if len(s):
                best = s.iloc[int(np.argmax(O.american_to_decimal(s[side])))]
                rec[f"best_{side}"], rec[f"best_{side}_book"] = int(best[side]), best["book"]
            else:
                rec[f"best_{side}"], rec[f"best_{side}_book"] = np.nan, None
        out.append(rec)
    out = pd.DataFrame(out)
    return out.merge(books_mkt.reset_index(), on=["game_id", "player_id", "market"], how="left")


def fit_one_sided_margin(log: pd.DataFrame, default: float = ONE_SIDED_MARGIN) -> float:
    """The median two-way overround of anytime-TD markets where a book posted both sides."""
    if log is None or log.empty or "hold" not in log:
        return default
    h = log[(log["market"] == "player_anytime_td") & (~log["one_sided"].astype(bool)) & log["hold"].notna()]["hold"]
    return float(h.median()) if len(h) >= 30 else default


# --------------------------------------------------------------------------
# Our side of it
# --------------------------------------------------------------------------

def price(cons: pd.DataFrame, players: pd.DataFrame, model: PropModel) -> pd.DataFrame:
    """Our probabilities, edge, EV and stake for each side of each consensus row."""
    if cons.empty:
        return cons
    cons = cons.copy()
    cons["stat"] = cons["market"].map(MARKET_STAT)
    pl = players.set_index("player_id")
    rows = []
    for (stat, ), g in cons.groupby(["stat"]):
        if stat not in SPECS:
            continue
        d = pl.loc[g["player_id"]].reset_index()
        for grp in d["group"].unique():
            m = (d["group"] == grp).to_numpy()
            if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
                continue
            gg, dd = g[m], d[m].reset_index(drop=True)
            over, push, under = model.probs(dd, stat, grp, gg["line"].to_numpy(float))
            for i, r in enumerate(gg.to_dict("records")):
                base = dict(r, team=dd.loc[i, "team"], pos=dd.loc[i, "pos"], status=dd.loc[i, "status"],
                            proj=float(sum(dd.loc[i, c] for c in SPECS[stat].proj)), role_clear=bool(role_clear(dd.iloc[[i]]).iloc[0]))
                for side, p_win, p_lose, fair in (("over", over[i], under[i], r["fair_power"]),
                                                  ("under", under[i], over[i], 1 - r["fair_power"])):
                    best = r.get(f"best_{side}")
                    if best is None or pd.isna(best):
                        continue
                    # the market's fair chance of winning, given no push, set against ours on the same basis
                    ours_dec = p_win / max(p_win + p_lose, 1e-9)
                    rows.append(dict(base, side=side, our_p=float(p_win), our_push=float(push[i]),
                                     fair=float(fair), fair_scaled=float(r["fair_prop"] if side == "over" else 1 - r["fair_prop"]),
                                     edge=float(ours_dec - fair), price=int(best), book=r[f"best_{side}_book"],
                                     ev=float(O.expected_value(p_win, best, push[i])),
                                     kelly=float(O.kelly(p_win, best, push[i]))))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return flag(out)


def flag(b: pd.DataFrame) -> pd.DataFrame:
    """Apply the flag rules and size the stakes."""
    b = b.copy()
    posted = b["books_market"] if "books_market" in b else b["n_books"]
    liquid = ((posted >= MIN_BOOKS) | b["sharp"]) & ((b["hold"].isna()) | (b["hold"] <= MAX_HOLD))
    in_range = b["price"].abs() <= MAX_ODDS
    reasons = []
    for e, lq, rc, ir, ev in zip(b["edge"], liquid, b["role_clear"], in_range, b["ev"]):
        r = []
        if e < MIN_EDGE:
            r.append("edge")
        if ev <= 0:
            r.append("ev")
        if not lq:
            r.append("liquidity")
        if not rc:
            r.append("role")
        if not ir:
            r.append("price")
        if not FLAGS_ON:
            r.append("paused")
        reasons.append(",".join(r))
    b["why_not"] = reasons
    b["flagged"] = b["why_not"] == ""
    stake = np.where(b["flagged"], np.minimum(KELLY_FRACTION * b["kelly"], MAX_STAKE_BET), 0.0)
    b["stake"] = stake
    # cap the game: scale everything in a game down together
    tot = b.groupby("game_id")["stake"].transform("sum")
    b["stake"] = np.where(tot > MAX_STAKE_GAME, b["stake"] * MAX_STAKE_GAME / tot, b["stake"]).round(4)
    # one side per prop at most (both sides can't both clear the edge, but guard anyway)
    return b
