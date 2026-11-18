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


# ---------------------------------------------------------------- context (Nov 18, 2026)

VOL = {"pass_att": "attempts", "carries": "carries", "targets": "targets"}
UNIT = {"pass_att": "attempt", "carries": "carry", "targets": "target"}


def why_short(m: dict) -> str:
    """One player who came up short, and what the facts say about why."""
    s = f"{m['name']} ({m['pos']}, {m['team']}): {_f(m['act'], 0)} {STATN[m['stat']]} against {_f(m['proj'], 0)} projected."
    bits = []
    v = m.get("volume") or {}
    if v.get("act") is not None and v.get("proj"):
        ratio = v["act"] / v["proj"]
        word = VOL[v["kind"]]
        if ratio < 0.7:
            bits.append(f"usage fell: {_f(v['act'])} {word} against {_f(v['proj'])} projected")
        elif ratio > 1.1:
            bits.append(f"usage was there ({_f(v['act'])} {word}, {_f(v['proj'])} projected) but the yards weren't")
        per = m.get("per") or {}
        if per.get("act") is not None and ratio >= 0.7 and per["act"] < 0.7 * per["proj"]:
            bits.append(f"{_f(per['act'])} yards per {UNIT[v['kind']]} against {_f(per['proj'])} expected")
    if m.get("target_share") is not None and m.get("target_share_avg"):
        if m["target_share"] < m["target_share_avg"] - 0.06:
            bits.append(f"target share {round(100 * m['target_share'])}% against his usual {round(100 * m['target_share_avg'])}%")
    if m.get("snap_pct") is not None and m.get("snap_pct_avg") is not None and m["snap_pct"] < m["snap_pct_avg"] - 0.15:
        bits.append(f"played {round(100 * m['snap_pct'])}% of snaps (usually {round(100 * m['snap_pct_avg'])}%)")
    rep = m.get("injury_report") or {}
    if rep.get("status"):
        bits.append(f"listed {rep['status']}" + (f" ({rep['injury']})" if rep.get("injury") else "") + " on the injury report")
    elif rep.get("practice") and "Full" not in rep["practice"]:
        bits.append(f"practice: {rep['practice'].lower()}" + (f" ({rep['injury']})" if rep.get("injury") else ""))
    sc = m.get("script") or {}
    if m["stat"] == "rush_yds" and sc.get("trailing9", 0) >= 0.4:
        bits.append(f"his team trailed by 9+ on {round(100 * sc['trailing9'])}% of its plays")
    if m["stat"] in ("pass_yds", "rec_yds") and sc.get("leading9", 0) >= 0.4:
        bits.append(f"his team led by 9+ on {round(100 * sc['leading9'])}% of its plays, so less need to throw")
    q = m.get("qb") or {}
    if q.get("ints") or (q.get("sacks") or 0) >= 4 or (q.get("cpoe") is not None and q["cpoe"] <= -8):
        bits.append(", ".join(x for x in (f"{q['ints']} INT" if q.get("ints") else "", f"{q['sacks']} sacks" if q.get("sacks") else "",
                                          f"completion % {_f(q['cpoe'])} below expected" if q.get("cpoe") is not None and q["cpoe"] < 0 else "") if x))
    d = (m.get("defenders") or {}).get("broke_up") or []
    if d:
        bits.append("passes broken up / picked by " + ", ".join(f"{n}" + (f" ({c})" if c > 1 else "") for n, c in d))
    if m.get("former_team"):
        bits.append(f"facing {m['former_team']['vs']}, his team in {m['former_team']['when']}")
    hist = []
    if m.get("prev"):
        hist.append(f"previous game {_f(m['prev']['val'], 0)}")
    if m.get("season_avg") is not None:
        hist.append(f"season average {_f(m['season_avg'])}")
    if m.get("season_low") is not None and m["act"] <= m["season_low"]:
        hist.append("a season low")
    b = "; ".join(bits)
    return s + (" " + b[:1].upper() + b[1:] + "." if bits else "") + (f" ({', '.join(hist)})." if hist else "")


def why_total(g: dict) -> str | None:
    f = g.get("flow") or {}
    if not f.get("drives"):
        return None
    H, A = g["home"], g["away"]
    side = (g["leans"].get("total") or {}).get("side")
    over = g["total"] > g["market"]["total"]
    parts = []
    for t in (A, H):
        d = f["drives"].get(t) or {}
        lbl = {"TD": ("TD", "TDs"), "FG": ("FG", "FGs"), "Turnover": ("turnover", "turnovers"),
               "Downs": ("turnover on downs", "turnovers on downs"), "Missed FG": ("missed FG", "missed FGs"), "Punt": ("punt", "punts")}
        stops = [f"{d[k]} {lbl[k][d[k] != 1]}" for k in lbl if d.get(k)]
        parts.append(f"{t} {d.get('n', 0)} drives, {', '.join(stops)} ({_f(d.get('per_drive', 0))} points a drive)")
    q = f.get("quarters") or {}
    qs = " / ".join(f"{t} {'-'.join(str(x) for x in q.get(t, []))}" for t in (A, H)) if q else ""
    s = f"Why the total went {'over' if over else 'under'} {_f(g['market']['total'])}: " + "; ".join(parts) + "."
    if qs:
        s += f" By quarter: {qs}."
    if f.get("over_at"):
        s += f" It went over at {f['over_at']}."
    gp = sum((f.get("garbage_pts") or {}).values())
    if gp:
        s += f" {gp} points came with one team up 17+ in the fourth quarter."
    return s


def why_cover(g: dict) -> str | None:
    f = g.get("flow") or {}
    st = f.get("cover_settled")
    if not st:
        return None
    return (f"The cover was settled by {st['team']}'s {st['pts']}-point score in quarter {st['q']} ({st['clock']}), making it {st['score']}"
            + (" — a late swing." if st.get("late") else "."))


def possibilities(g: dict, league: str) -> list[str]:
    """Context worth a look. Possibilities, not conclusions."""
    out = []
    c = g.get("context") or {}
    tms = c.get("teams") or {}
    for t, x in tms.items():
        pv = x.get("prev")
        if pv and pv.get("label"):
            out.append(f"{t} came off a {pv['label']}: {'beat' if pv['margin'] > 0 else 'lost to'} {pv['opp']} {pv['score']} "
                       f"and {'covered' if pv['cover'] > 0 else 'missed the cover'} by {_f(abs(pv['cover']))}. A letdown or bounce-back is possible.")
        if pv and abs(pv.get("model_miss") or 0) >= 14:
            # Nov 18, 2026 check (Project doc claude/team-rating-bias-check-2026-11-18.md): on the tuning seasons a
            # team's recent misses did not carry into its next NFL game (0.03 +/- 0.04 per point) and carried
            # about a tenth in college (0.10 +/- 0.03), with the closing line already holding it in both.
            tail = ("In the NFL a run of misses has not carried into the next game, so read this as noise unless something changed (a quarterback, injuries)."
                    if league == "nfl" else
                    "In college about a tenth of a team's recent average miss has carried into its next game, and the line already has it.")
            out.append(f"The model also missed {t}'s previous game by {_f(abs(pv['model_miss']))} ({'better' if pv['model_miss'] > 0 else 'worse'} than it expected). {tail}")
        if x.get("after_bye"):
            out.append(f"{t} was coming off a bye.")
        if x.get("rest") is not None and x["rest"] <= 5:
            out.append(f"{t} was on a short week ({x['rest']} days' rest).")
    if c.get("division"):
        out.append("A division game: familiarity can narrow margins and slow offenses.")
    if c.get("conference_game") is False:
        out.append("A non-conference game.")
    if c.get("rematch"):
        r = c["rematch"]
        out.append(f"A rematch: {r['winner']} won the first meeting {r['score']} in week {r['week']}.")
    if c.get("time_zones"):
        out.append("The road team crossed time zones (" + " vs ".join(f"{t} {z.split('/')[-1].replace('_', ' ')}" for t, z in c["time_zones"].items()) + ").")
    if c.get("wind") and c["wind"] >= 15:
        out.append(f"Wind {c['wind']} mph.")
    for fx in g.get("former_team") or []:
        out.append(f"{fx['name']} ({fx['pos']}, {fx['team']}) faced {fx['vs']}, his team in {fx['when']}: each side knows the other's habits.")
        if len(out) > 9:
            break
    for t, pe in (g.get("personnel") or {}).items():
        if not isinstance(pe, dict):
            continue
        outs = [r for r in pe.get("regulars_out", [])][:4]
        if outs:
            out.append(f"{t} played without regulars: " + ", ".join(
                f"{r['name']} ({r.get('pos') or r['side']}" + (f", {r['report']}" if r.get("report") else "") + ")" for r in outs)
                + (". The board's projections may not have known." if league == "cfb" else "."))
        more = [r for r in pe.get("more_than_usual", [])][:3]
        if more:
            out.append(f"{t} leaned on players who usually play little: " + ", ".join(
                f"{r['name']} ({r.get('pos') or r['side']}, " + (f"{round(100 * r['pct'])}% of snaps, usually {round(100 * r['usual'])}%" if r.get("pct") is not None
                                                             else f"{r.get('tackles')} tackles, usually {r['usual']}") + ")" for r in more) + ".")
        pm = pe.get("playmakers") or []
        if pm:
            out.append(f"{t}'s defensive playmakers: " + "; ".join(f"{r['name']} {r['line']}" for r in pm[:3]) + ".")
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

    # players: who came up short and why (Ryan cares about these more than the overs)
    shorts = g.get("short") or []
    for m in shorts[:4]:
        wrong.append(why_short(m))
    if g.get("beat"):
        right.append("Beat their projections: " + "; ".join(f"{b['name']} {b['act']} ({b['proj']} projected)" for b in g["beat"][:3]) + ".")
    if g.get("dnp"):
        improve.append("Projected but didn't play: " + ", ".join(f"{d['name']} ({d['pos']})" for d in g["dnp"][:4])
                       + ". Check how late that was known.")
    # totals and covers: why
    wt, wc = why_total(g), why_cover(g)
    if wt:
        (right if v.get("total") == "W" else wrong if v.get("total") == "L" else chance).append(wt)
    if wc:
        (right if v.get("spread") == "W" else wrong if v.get("spread") == "L" else chance).append(wc)
    if c:
        for q in c["qb_exits"]:
            improve.append(f"{q['qb']}'s line and his receivers' count in this week's misses; an early exit isn't a projection miss.")
    if g["unprojected"]:
        u = g["unprojected"][0]
        improve.append(f"{u['name']} ({u['pos']}, {u['team']}) had {_f(u['yds'], 0)} yards and wasn't on the board's projections: a role it didn't see coming.")
    pos = possibilities(g, league)
    improve.extend(f"Possibility: {x}" for x in pos[:6])
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
