"""The books' lean to the over on player props, and unders the model agrees with.

    python experiments/props_over_lean_test.py

Nov 9, 2026 (PROPS.md "The over lean and model-agreed unders"). An exploratory pass the same day,
outside the repo and without the PFF route inputs (Project doc claude/props-gap-2026-11-09.md),
found two things on the 2023-2025 DraftKings/FanDuel main lines: overs clear less often than the
books' vig-free chance says, and unders the projections strongly agree with returned more than
zero. This script re-runs both on the board's own projections (the New projections with PFF route
share where the board uses them, the current ones elsewhere; pregame projections at the pricing
snapshot, game-day at the close), with the rules below written before it was run.

Because the exploratory pass came first and chose the cuts (35%, quarterback markets apart,
passing yards out), this is NOT an out-of-sample test of them: it checks that the result holds on
the board's exact projections. The out-of-sample record is the live log from here on (part C).

Rows: two-sided main lines, each book's line counted separately, pushes left out, players who
played (a prop is void otherwise). Standard errors clustered by game. Widths fitted on seasons
before the one scored; boosted step walk-forward; nothing fitted on the season scored.

A. THE OVER LEAN (no model). fair = the books' vig-free over chance (power method, as the board).
   Shade = fair + s, where s is the average of (went over - fair) on every earlier season with
   lines, fitted separately for two groups: quarterback markets (passing yards, and a
   quarterback's rushing yards) and everything else. A stat with no earlier season of lines
   (receptions, carries: 2025 only) takes its group's s.
   RULE A (the keep rule: better on every measure, worse on none): Brier(shade) < Brier(fair) on
   2024 (s from 2023) and on 2025 (s from 2023-2024), at the pricing snapshot and at the close.
   Four measures. Passing only says the fair number can be improved; putting it on the Props
   section or the Parlays page needs Ryan's go-ahead, a plan and a mockup.

B. MODEL-AGREED UNDERS. Props where the board's over chance, our_over / (our_over + our_under),
   is below 0.35, passing yards left out. The bet: the under at that book's posted price.
   RULE B: the return is above zero in each of 2023, 2024 and 2025, at the pricing snapshot and
   at the close. Six measures. Reported alongside and required before any stake, but not part of
   the rule here: closing line value (did the line or the under's vig-free chance move the bet's
   way between pricing and close). The playbook's standard stands: a record without closing line
   value is unproven.
   Also reported: the return by band of the model's over chance, and by stat and position.

C. THE 2026 LIVE LOG (data/props/prop_log.csv with prop_results.csv), reported only: the same two
   measures on the props logged and graded so far. Too few to judge yet.

Writes experiments/out/props_over_lean_test.json.

ADDED AFTER THE FIRST RUN (Nov 9, 2026), and disclosed as such: the first run showed quarterback
rush-attempt overs clearing 32% of the time. The cause is grading, not the market: the board's box
scores (players_data.py) leave kneel-downs out, and the books count a kneel as a rushing attempt
for about -1 yard (2025: 0.79 kneels per team-game; 1.8 kneels and -1.9 yards in the 44% of
team-games with any). So a quarterback's carries are graded about 0.8 low and his rushing yards
about 0.9 high. Both rules are therefore reported twice: "as_written" (every row, the version the
rules were written for) and "qb_rushing_out" (quarterback carries and quarterback rushing yards
dropped), which is the one to read. Nothing else was changed.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nflmodel import config as C, data, players_ml as ML, propbackfill as B, props as P, roles  # noqa: E402
import props_real_lines as RL  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "props_over_lean_test.json"
SEASONS = (2023, 2024, 2025)
AGREE = 0.35                     # rule B: the board's over chance below this
BANDS = [0, 0.35, 0.42, 0.48, 0.54, 1.0]
ADOPTED = {(s, g) for s, gs in ML.ADOPT.items() for g in gs}


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def se(v: pd.Series, games: pd.Series) -> float:
    s = v.groupby(games).sum()
    return float(s.std(ddof=1) * np.sqrt(len(s)) / len(v)) if len(s) > 1 else float("nan")


def qb_market(x: pd.DataFrame) -> pd.Series:
    return (x["stat"] == "pass_yds") | ((x["stat"] == "rush_yds") & (x["group"] == "QB"))


def board_rows(table: pd.DataFrame, boosted: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Every two-sided main line of 2023-2025 at one snapshot, priced with the projections the board
    shows: boosted where players_ml.ADOPT says so, the current ones elsewhere."""
    saved = P.PropModel.load()
    parts = []
    for season in SEASONS:
        snaps = B.load(season, kind)
        for label, t in (("cur", table), ("new", boosted)):
            m = P.PropModel(FAMILY).fit(t[(t.season < season) & (t.season >= 2014)])
            m.iso = saved.iso
            x = RL.priced(t, m, snaps, season)
            if len(x):
                adopted = pd.Series([(s, g) in ADOPTED for s, g in zip(x["stat"], x["group"])], index=x.index)
                parts.append(x[adopted if label == "new" else ~adopted])
    x = pd.concat(parts, ignore_index=True)
    x = x[x["y"] != x["line"]].copy()
    x["hit"] = (x["y"] > x["line"]).astype(float)
    x["ours"] = x["our_over"] / (x["our_over"] + x["our_under"])
    x["u_ret"] = np.where(x["hit"] == 0, dec(x["under"]) - 1, -1.0)
    x["o_ret"] = np.where(x["hit"] == 1, dec(x["over"]) - 1, -1.0)
    x["qbm"] = qb_market(x)
    return x


def lean_table(x: pd.DataFrame, by) -> list[dict]:
    rows = []
    for key, g in x.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({"key": [str(k) for k in key], "n": int(len(g)), "over_rate": round(float(g.hit.mean()), 4),
                     "fair_over": round(float(g.fair_over.mean()), 4),
                     "under_return": round(float(g.u_ret.mean()), 4), "under_se": round(se(g.u_ret, g.game_id), 4),
                     "over_return": round(float(g.o_ret.mean()), 4), "over_se": round(se(g.o_ret, g.game_id), 4)})
    return rows


def rule_a(x: pd.DataFrame) -> dict:
    out = {}
    for test in (2024, 2025):
        tr, te = x[x.season < test], x[x.season == test]
        s = {bool(q): float((g.hit - g.fair_over).mean()) for q, g in tr.groupby("qbm")}
        shade = (te.fair_over + te.qbm.map(s)).clip(0.02, 0.98)
        d = (shade - te.hit) ** 2 - (te.fair_over - te.hit) ** 2
        # the nudge toward the model on the same props, for comparison (weight fitted on earlier seasons)
        w = float(((tr.ours - tr.fair_over) * (tr.hit - tr.fair_over)).sum() / ((tr.ours - tr.fair_over) ** 2).sum())
        dn = ((te.fair_over + w * (te.ours - te.fair_over)).clip(0.02, 0.98) - te.hit) ** 2 - (te.fair_over - te.hit) ** 2
        tr_sh = (tr.fair_over + tr.qbm.map(s))
        w2 = float(((tr.ours - tr_sh) * (tr.hit - tr_sh)).sum() / ((tr.ours - tr_sh) ** 2).sum())
        db = ((shade + w2 * (te.ours - shade)).clip(0.02, 0.98) - te.hit) ** 2 - (te.fair_over - te.hit) ** 2
        out[str(test)] = {"n": int(len(te)), "shift_qb_markets": round(s.get(True, float("nan")), 4), "shift_other": round(s.get(False, float("nan")), 4),
                          "brier_fair": round(float(((te.fair_over - te.hit) ** 2).mean()), 5),
                          "shade_minus_fair": round(float(d.mean()), 5), "se": round(se(d, te.game_id), 5),
                          "nudge_weight": round(w, 3), "nudge_minus_fair": round(float(dn.mean()), 5), "nudge_se": round(se(dn, te.game_id), 5),
                          "shade_plus_nudge_weight": round(w2, 3), "shade_plus_nudge_minus_fair": round(float(db.mean()), 5),
                          "shade_plus_nudge_se": round(se(db, te.game_id), 5),
                          "better": bool(d.mean() < 0)}
    return out


def rule_b(x: pd.DataFrame) -> dict:
    y = x[x["stat"] != "pass_yds"]
    a = y[y.ours < AGREE]
    out = {"by_season": {}, "bands": [], "by_stat": []}
    for season in SEASONS:
        g = a[a.season == season]
        out["by_season"][str(season)] = {"n": int(len(g)), "under_wins": round(float(1 - g.hit.mean()), 4),
                                         "under_return": round(float(g.u_ret.mean()), 4), "se": round(se(g.u_ret, g.game_id), 4),
                                         "above_zero": bool(g.u_ret.mean() > 0)}
    out["all"] = {"n": int(len(a)), "under_wins": round(float(1 - a.hit.mean()), 4), "under_return": round(float(a.u_ret.mean()), 4),
                  "se": round(se(a.u_ret, a.game_id), 4), "avg_price_decimal": round(float(dec(a.under).mean()), 3)}
    band = pd.cut(y.ours, BANDS)
    for b, g in y.groupby(band):
        out["bands"].append({"band": str(b), "n": int(len(g)), "under_return": round(float(g.u_ret.mean()), 4), "under_se": round(se(g.u_ret, g.game_id), 4),
                             "over_return": round(float(g.o_ret.mean()), 4), "over_se": round(se(g.o_ret, g.game_id), 4),
                             "under_return_by_season": {str(s): round(float(g[g.season == s].u_ret.mean()), 4) for s in SEASONS}})
    for (st, grp), g in a.groupby(["stat", "group"]):
        out["by_stat"].append({"stat": st, "group": grp, "n": int(len(g)), "under_return": round(float(g.u_ret.mean()), 4), "se": round(se(g.u_ret, g.game_id), 4)})
    return out


def clv(fl: pd.DataFrame, cl: pd.DataFrame) -> dict:
    """For the model-agreed unders taken at the pricing snapshot: where the same book stood at the close."""
    a = fl[(fl["stat"] != "pass_yds") & (fl.ours < AGREE)]
    c = cl[["game_id", "player_id", "market", "book", "line", "fair_over", "under"]].rename(
        columns={"line": "c_line", "fair_over": "c_fair", "under": "c_under"})
    c = c.sort_values("c_fair").drop_duplicates(["game_id", "player_id", "market", "book"])
    m = a.merge(c, on=["game_id", "player_id", "market", "book"], how="inner")
    same = m[m.c_line == m.line]
    # an under bettor wants the line to fall (his number is then the higher one) or the under's chance to rise
    gain = (1 - same.c_fair) - (1 - same.fair_over)
    price_clv = (1 - same.c_fair) * dec(same.under) - 1      # the bet's EV at the closing vig-free chance
    return {"bets": int(len(a)), "still_posted_at_close": int(len(m)),
            "line_fell": round(float((m.c_line < m.line).mean()), 3), "line_rose": round(float((m.c_line > m.line).mean()), 3),
            "line_same": round(float((m.c_line == m.line).mean()), 3), "avg_line_move": round(float((m.c_line - m.line).mean()), 3),
            "same_line_n": int(len(same)), "same_line_fair_under_pricing": round(float((1 - same.fair_over).mean()), 4),
            "same_line_fair_under_close": round(float((1 - same.c_fair).mean()), 4),
            "same_line_fair_gain": round(float(gain.mean()), 4), "same_line_fair_gain_se": round(se(gain, same.game_id), 4),
            "same_line_ev_at_closing_fair": round(float(price_clv.mean()), 4), "same_line_ev_se": round(se(price_clv, same.game_id), 4)}


def live_2026() -> dict:
    """The live log so far: two-sided main lines, graded. our_p is the logged side's chance; the New
    projections' chance (prop_log_new.csv) where the board shows it."""
    L = pd.read_csv(C.DATA / "props" / "prop_log.csv", low_memory=False)
    R = pd.read_csv(C.DATA / "props" / "prop_results.csv")
    N = pd.read_csv(C.DATA / "props" / "prop_log_new.csv")
    d = L.merge(R, on="row_id").merge(N[["row_id", "our_p_new"]], on="row_id", how="left")
    d = d[d.result.isin(["win", "loss"]) & (d.one_sided == 0) & ~d.market.str.contains("alternate")]
    d["ret"] = np.where(d.result == "win", dec(d.price) - 1, -1.0)
    d["win"] = (d.result == "win").astype(float)
    d["p"] = d.our_p_new.fillna(d.our_p)
    out = {}
    for snap, x in d.groupby("snapshot"):
        r = {"weeks": sorted(int(w) for w in x.week.unique())}
        for side, g in x.groupby("side"):
            r[side] = {"n": int(len(g)), "win_rate": round(float(g.win.mean()), 4), "fair": round(float(g.fair.mean()), 4),
                       "return": round(float(g.ret.mean()), 4), "se": round(se(g.ret, g.game_id), 4)}
        a = x[(x.side == "under") & (x.stat != "pass_yds") & (x.p > 1 - AGREE) & ~((x.pos == "QB") & x.stat.isin(["carries", "rush_yds"]))]
        r["model_agreed_unders"] = {"n": int(len(a)), "win_rate": round(float(a.win.mean()), 4) if len(a) else None,
                                    "return": round(float(a.ret.mean()), 4) if len(a) else None,
                                    "se": round(se(a.ret, a.game_id), 4) if len(a) > 1 else None}
        out[snap] = r
    return out


def qb_rushing(x: pd.DataFrame) -> pd.Series:
    """Quarterback carries and rushing yards: graded without kneel-downs, so not comparable with the books."""
    return (x["group"] == "QB") & x["stat"].isin(["carries", "rush_yds"])


def report(x: pd.DataFrame) -> dict:
    return {"n": int(len(x)), "games": int(x.game_id.nunique()),
            "ours_minus_books_brier": round(float(((x.ours - x.hit) ** 2 - (x.fair_over - x.hit) ** 2).mean()), 5),
            "lean_by_season": lean_table(x, "season"), "lean_by_stat": lean_table(x, ["stat", "group"]),
            "lean_by_book": lean_table(x, "book"), "lean_by_group": lean_table(x, "qbm"),
            "rule_a": rule_a(x), "rule_b": rule_b(x)}


def passes(v: dict) -> dict:
    return {"rule_a_passes": all(v[n]["rule_a"][s]["better"] for n in ("pricing", "close") for s in ("2024", "2025")),
            "rule_b_passes": all(v[n]["rule_b"]["by_season"][str(s)]["above_zero"] for n in ("pricing", "close") for s in SEASONS)}


ROWS_CACHE = C.RAW / "props_over_lean_rows.pkl"     # data/raw is git-ignored


def main():
    if ROWS_CACHE.exists():
        rows, pff = pd.read_pickle(ROWS_CACHE)
    else:
        games, _, _ = data.load_all(refresh=False, verbose=False)
        R = roles.load_all(games, verbose=False)
        rows, pff = {}, {}
        for name, table, kind in (("pricing", RL.load_pregame(), "flag"), ("close", load_proj(None), "close")):
            table = ML.add_features(table, R)
            pff[name] = bool(set(ML.ROUTE_F) <= set(table.columns) and table[ML.ROUTE_F[0]].notna().any())
            boosted = ML.walk_forward(table, first=min(SEASONS))
            rows[name] = board_rows(table, boosted, kind)
        pd.to_pickle((rows, pff), ROWS_CACHE)
    rep = {"agree_below": AGREE, "seasons": list(SEASONS), "pff_route_inputs": pff}
    for version, keep in (("as_written", lambda x: x), ("qb_rushing_out", lambda x: x[~qb_rushing(x)])):
        v = {name: report(keep(rows[name])) for name in ("pricing", "close")}
        v["clv_pricing_to_close"] = clv(keep(rows["pricing"]), keep(rows["close"]))
        v.update(passes(v))
        rep[version] = v
        print(version, json.dumps({n: {"rule_a": v[n]["rule_a"], "rule_b_all": v[n]["rule_b"]["all"], "rule_b_by_season": v[n]["rule_b"]["by_season"]}
                                   for n in ("pricing", "close")}, indent=1))
        print(json.dumps({k: v[k] for k in ("clv_pricing_to_close", "rule_a_passes", "rule_b_passes")}, indent=1), flush=True)
    rep["live_2026"] = live_2026()
    print(json.dumps({"pff_route_inputs": pff, "live_2026": rep["live_2026"]}, indent=1))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
