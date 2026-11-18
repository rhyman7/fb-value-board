"""Weekly Review: first-draft reads written from the facts file (nflmodel/review.py).

Writes `review/data/<league>/<season>_w<week>.reads.json`:

    {"by": "draft" | "claude" | "mixed", "week_notes": [...],
     "games": {<game id>: {"by": "draft" | "claude", "headline": str,
                           "verdict": {"spread": "W"|"L"|"chance"|"push", "total": ...},
                           "right": [...], "wrong": [...], "chance": [...], "improve": [...]}}}

A draft says only what the numbers show. The weekly review task rewrites each game's read
in its own words and marks it "claude"; a game already marked "claude" is never overwritten
here (pass --force to redraft everything).
"""
from __future__ import annotations

import json
from pathlib import Path

from nflmodel import review as R

STATN = {"pass_yds": "passing yards", "rush_yds": "rushing yards", "rec_yds": "receiving yards"}


def _f(x, d=1):
    return f"{x:.{d}f}".rstrip("0").rstrip(".") if d else f"{x:.0f}"


def fav(g, m):
    if abs(m) < 0.05:
        return "even"
    return f"{g['home'] if m > 0 else g['away']} by {_f(abs(m))}"


def line_txt(g, spread):
    if abs(spread) < 0.05:
        return "a pick'em"
    t = g["home"] if spread > 0 else g["away"]
    return f"{t} −{_f(abs(spread))}"


def lean_txt(g, k):
    L = g["leans"][k]
    if k == "spread":
        side = g[L["side"]]
        num = g["market"]["spread"] * (1 if L["side"] == "home" else -1)
        return f"{side} {'−' if num > 0 else '+'}{_f(abs(num))}" if abs(num) >= 0.05 else f"{side} pick'em"
    return f"{L['side']} {_f(g['market']['total'])}"


def verdicts(g) -> dict:
    """W/L for each lean, or 'chance' when the chance items alone flipped it."""
    out = {}
    c = g["chance"]
    exit_ = bool(c and c["qb_exits"])
    for k in ("spread", "total"):
        L = g["leans"].get(k)
        if not L:
            continue
        res = L["result"]
        if c:
            if k == "spread":
                d = g["neutral_margin"] - g["market"]["spread"]
                hi = L["side"] == "home"
            else:
                d = g["total"] - c["total_pts_nonoff"] - g["market"]["total"]
                hi = L["side"] == "over"
            neutral = "push" if d == 0 else ("W" if (d > 0) == hi else "L")
            if neutral != res and res != "push":
                res = "chance"
        if exit_ and res in ("W", "L"):
            res = "chance"
        out[k] = res
    return out


def draft_game(g: dict, league: str) -> dict:
    v = verdicts(g)
    c, box, H, A = g["chance"], g["box"], g["home"], g["away"]
    mo, mk = g["model"], g["market"]
    right, wrong, chance, improve = [], [], [], []

    # the two leans
    for k, nm in (("spread", "side"), ("total", "total")):
        L = g["leans"].get(k)
        if not L:
            continue
        mine = fav(g, mo["margin"]) if k == "spread" else _f(mo["total"])
        res = f"{fav(g, g['margin'])}" if k == "spread" else f"{_f(g['total'], 0)} points"
        s = f"The {nm} lean, {lean_txt(g, k)} (model {mine}), {'won' if L['result'] == 'W' else 'lost' if L['result'] == 'L' else 'pushed'}: {res}."
        if v.get(k) == "chance":
            (right if L["result"] == "W" else wrong).append(s + " Chance decided it (below), so it doesn't count for or against the read.")
        elif L["result"] == "W":
            right.append(s)
        elif L["result"] == "L":
            wrong.append(s + (f" The model was {_f(abs(L['gap']))} points off the market here, one of its bigger gaps." if abs(L["gap"]) >= (3 if k == "spread" and league == "nfl" else 5) else ""))

    # who was closer
    for k, nm in (("margin", "margin"), ("total", "total")):
        em, ek = g["err"][f"model_{k}"], g["err"][f"market_{k}"]
        if em is None or ek is None:
            continue
        if abs(em) + 1 < abs(ek):
            right.append(f"On the {nm} the model was closer than the market (missed by {_f(abs(em))} against {_f(abs(ek))}).")
        elif abs(ek) + 1 < abs(em):
            wrong.append(f"On the {nm} the market was closer than the model (missed by {_f(abs(ek))} against {_f(abs(em))}).")

    # how it was played: did the box agree with the scoreboard?
    if box and box[H] and box[A]:
        h, a = box[H], box[A]
        if h["success"] is not None and a["success"] is not None:
            better = H if (h["epa_play"] or 0) > (a["epa_play"] or 0) else A
            winner = H if g["margin"] > 0 else A if g["margin"] < 0 else None
            sh, sa = round(100 * h["success"]), round(100 * a["success"])
            if winner and better != winner and abs((h["epa_play"] or 0) - (a["epa_play"] or 0)) >= 0.05:
                chance.append(f"{better} played the better game play for play (success rate {sh if better == H else sa}% to "
                              f"{sa if better == H else sh}%, EPA per play {R_sgn(h['epa_play'] if better == H else a['epa_play'])} to "
                              f"{R_sgn(a['epa_play'] if better == H else h['epa_play'])}) and still lost.")

    # chance items
    if c:
        for q in c["qb_exits"]:
            chance.append(f"{q['qb']} {('got hurt and ' if q['why'] == 'injured' else '')}left in quarter {q['qtr']} "
                          f"({q['time']}); he threw {round(100 * q['share'])}% of the team's passes. Nothing before kickoff could price that in.")
            improve.append(f"{q['qb']}'s line and his receivers' are counted in this week's projection misses; "
                           "an early exit isn't a projection miss.")
        if abs(c["fumble_bounces"]) >= 2:
            t = H if c["fumble_bounces"] > 0 else A
            chance.append(f"Fumble bounces went {t}'s way, worth about {_f(abs(c['fumble_bounces']))} points "
                          f"({A} {_n(box[A]['fumbles'], 'fumble')}, {box[A]['fum_lost']} lost; {H} {_n(box[H]['fumbles'], 'fumble')}, {box[H]['fum_lost']} lost).")
        nh, na = box[H]["nonoff_td"], box[A]["nonoff_td"]
        if nh or na:
            who = " and ".join(f"{t} {n}" for t, n in ((A, na), (H, nh)) if n)
            chance.append(f"Defensive or return touchdowns: {who}. They put {c['total_pts_nonoff']} points on the total"
                          + (f" and moved the margin about {_f(abs(c['return_def_tds']))}." if c["return_def_tds"] else "."))
        if c["short_fg_misses"]:
            chance.append(f"Missed field goals from 45 yards or closer cost about {_f(abs(c['short_fg_misses']))} points of margin.")
        if abs(c["total"]) >= 2:
            chance.append(f"With even luck the margin would have been {fav(g, g['neutral_margin'])} (actual {fav(g, g['margin'])}).")
    if not chance:
        chance.append("Nothing big: no early exit at quarterback, no defensive or return touchdowns, and the fumble bounces were close to even. The result mostly reflects how the teams played.")

    # players
    exits = {q["qb"].split(".")[-1].strip() for q in (c["qb_exits"] if c else [])}
    real = [p for p in g["players"] if p["played"] and not any(p["name"].endswith(e) for e in exits)]
    good = sorted([p for p in real if p["proj"] >= 40 and abs(p["miss"]) <= 0.12 * p["proj"]], key=lambda p: -p["proj"])[:2]
    if good:
        right.append("Projections that held: " + "; ".join(f"{p['name']} {_f(p['act'], 0)} {STATN[p['stat']]} (projected {_f(p['proj'], 0)})" for p in good) + ".")
    big, named = [], set()
    for p in real:
        if abs(p["miss"]) >= max(25, 0.5 * p["proj"]) and p["name"] not in named:
            big.append(p)
            named.add(p["name"])
    big = big[:3]
    if big:
        wrong.append("Biggest player misses: " + "; ".join(
            f"{p['name']} {_f(p['act'], 0)} {STATN[p['stat']] if p['act'] != 1 else STATN[p['stat']][:-1]} against {_f(p['proj'], 0)} projected" for p in big) + ".")
    dnp = [p for p in g["players"] if not p["played"]][:2]
    if dnp:
        improve.append("Projected but didn't play: " + ", ".join(p["name"] for p in dnp) + ". Worth checking how late " + ("his" if len(dnp) == 1 else "their") + " status was known.")
    if g["unprojected"]:
        u = g["unprojected"][0]
        improve.append(f"{u['name']} ({u['pos']}, {u['team']}) had {_f(u['yds'], 0)} yards and wasn't on the board's projections. "
                       "A role the board didn't see coming.")
    if not improve:
        improve.append("Nothing specific to fix from this game. One game isn't evidence; it goes into the season patterns.")

    # headline
    if c and c["qb_exits"]:
        q = c["qb_exits"][0]
        head = f"{q['qb']} left in quarter {q['qtr']}, and that shaped the result more than anything the model or market could see."
    elif any(x == "chance" for x in v.values()):
        k = next(k for k, x in v.items() if x == "chance")
        head = f"The {'side' if k == 'spread' else 'total'} came down to chance: {('fumble bounces and return scores' if k == 'spread' else 'defensive and return touchdowns')} flipped it."
    else:
        wins = [k for k, x in v.items() if x == "W"]
        loss = [k for k, x in v.items() if x == "L"]
        closer = abs(g["err"]["model_margin"]) < abs(g["err"]["market_margin"])
        if len(wins) == 2:
            head = "Both leans won on the merits."
        elif len(loss) == 2:
            head = "Both leans lost, and the luck was close to even, so these were real misses."
        else:
            head = f"Split: the {'side' if 'spread' in wins else 'total'} lean won and the {'side' if 'spread' in loss else 'total'} lost."
        head += f" Final {fav(g, g['margin'])}, {_f(g['total'], 0)} points; the model had {fav(g, mo['margin'])}, the market {fav(g, mk['spread'])}."
        if not closer and abs(g["err"]["model_margin"]) > 14:
            head += " Neither number was close."
    return {"by": "draft", "headline": head, "verdict": v, "right": right, "wrong": wrong, "chance": chance, "improve": improve}


def _n(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def R_sgn(x):
    return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x):.2f}"


def week_notes(f: dict) -> list[str]:
    s, n = f["summary"], f["summary"]["games"]
    ls, lt, m = s["lean_spread"], s["lean_total"], s["mae"]
    notes = [f"Side leans {ls[0]}-{ls[1]}" + (f"-{ls[2]}" if ls[2] else "") + f", total leans {lt[0]}-{lt[1]}" + (f"-{lt[2]}" if lt[2] else "") + ".",
             f"Average miss on the margin: model {m['model_margin']}, market {m['market_margin']}; on the total: model {m['model_total']}, market {m['market_total']}. "
             + ("The model beat the market on margins this week." if m["model_margin"] + 0.2 < m["market_margin"] else
                "The market beat the model on margins this week." if m["market_margin"] + 0.2 < m["model_margin"] else
                "Model and market were level on margins.")]
    big = sorted(f["games"], key=lambda g: -abs(g["err"]["model_margin"]))[:4]
    notes.append("Biggest margin misses: " + ", ".join(f"{g['away']} @ {g['home']} ({_f(abs(g['err']['model_margin']))})" for g in big) + ".")
    notes.append(f"Chance played a big part in {s['chance_games']} of {n} games.")
    if sum(s["bets"]):
        notes.append(f"Staked bets: {s['bets'][0]}-{s['bets'][1]}" + (f"-{s['bets'][2]}" if s["bets"][2] else "") + f", {s['units']:+.2f} units.")
    else:
        notes.append("No bets were staked; the leans are the model's opinions, not money.")
    return notes


def run(league: str, season: int, week: int, force: bool = False) -> Path:
    fp = R.OUT / league / f"{season}_w{week:02d}.json"
    f = json.loads(fp.read_text())
    rp = fp.with_name(fp.stem + ".reads.json")
    old = json.loads(rp.read_text()) if rp.exists() else {"games": {}}
    games = {}
    for g in f["games"]:
        prev = old["games"].get(g["id"])
        games[g["id"]] = prev if prev and prev.get("by") == "claude" and not force else draft_game(g, league)
    bys = {x["by"] for x in games.values()}
    out = {"by": bys.pop() if len(bys) == 1 else "mixed",
           "week_by": old.get("week_by", "draft") if not force else "draft",
           "week_notes": old["week_notes"] if old.get("week_by") == "claude" and not force else week_notes(f),
           "games": games}
    rp.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    R.write_index(league)
    return rp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Draft the Weekly Review reads from the facts files.")
    ap.add_argument("league", choices=["nfl", "cfb"])
    ap.add_argument("season", type=int)
    ap.add_argument("weeks", type=int, nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.weeks:
        print(run(a.league, a.season, w, a.force))
