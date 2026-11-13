"""Pre-set test: the model nudge on college prop main lines (Nov 13, 2026).

    python experiments/cfb_props_nudge_test.py

Rule written before running (Project doc claude/plan-cfb-props-nudge-test-2026-11-13.md):

Candidate: on two-sided college main lines (passing yards, passing TDs, rushing yards, receiving
yards) the books' no-vig over chance (power method, averaged over DraftKings and FanDuel) moved 20%
of the way to the model's chance, at most 3 points; legs more than 25 points apart stay at the
books' chance. Model: the college board's walk-forward projections (PFF inputs when the private
data is present) on the college prop widths, data/cfb/prop_params.json. Ladders and anytime TD are
not part of it.

Data: 2025 lines (tuning: the college widths were chosen on them) and 2024 lines (holdout), ten
minutes before kickoff, pushes out; each with game-day projections (a player who appeared is
active, as in every earlier college prop test) and bet-time projections (active = expected to
play: a stat in the team's previous game; no designations). 2026 live: the chance the board
logged when it first saw each line (docs/cfb/props/2026_w*.json), graded games only.

Rule: Brier, nudged minus books, clustered by game, for each stat and all four together, on 2025
and 2024, game-day and bet-time (20 measures). Kept only if nudged is below the books on all 20,
and on the 2024 all-stats measure by at least 1 standard error for both kinds. 2026 live is
reported; worse than the books by more than 1 SE overall goes to Ryan with the result. Props
check, reported: the legs the nudge makes +EV at the better DK/FD price, record and return. Mixed
results go to Ryan. Reported only, not adoptable from this run: the same nudge at the NFL's
current 7.5-point cap.

Writes experiments/out/cfb_props_nudge_test.json. Changes nothing on the boards.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import odds as O, propdist as D, propmarket as PM, props as P  # noqa: E402
import cfb_props_real_lines as R  # noqa: E402

OUT = HERE / "out" / "cfb_props_nudge_test.json"
PARAMS = ROOT / "data" / "cfb" / "prop_params.json"
W, CAP, CONFLICT = 0.2, 0.03, 0.25
CAP_NFL = 0.075                       # reported only
STATS = ["pass_yds", "pass_td", "rush_yds", "rec_yds"]
BOOKS = ["draftkings", "fanduel"]


def nudge(ours, fair, cap=CAP):
    gap = ours - fair
    return np.where(np.abs(gap) <= CONFLICT, fair + np.clip(W * gap, -cap, cap), fair)


def measure(ours, fair, hit, game) -> dict:
    out = {}
    for cap, key in ((CAP, "nudge"), (CAP_NFL, "nudge_cap_7_5_reported_only")):
        d, se = R.clustered_diff(nudge(ours, fair, cap), fair, hit, game)
        out[key] = {"minus_books": round(d, 6), "se": round(se, 6), "z": round(d / se, 2) if se else None}
    m, mse = R.clustered_diff(ours, fair, hit, game)
    out["model_alone"] = {"minus_books": round(m, 6), "se": round(mse, 6)}
    out["props"], out["games"] = int(len(hit)), int(pd.Series(game).nunique())
    out["moved"] = round(float(np.mean(np.abs(nudge(ours, fair) - fair) > 1e-9)), 3)
    return out


def ev_legs(x: pd.DataFrame, p_over: np.ndarray) -> dict:
    """Bet a side where the nudged chance is +EV at the better DK/FD price; one side per prop."""
    rows = []
    for i, r in enumerate(x.itertuples()):
        best = None
        for side, p in (("over", p_over[i]), ("under", 1 - p_over[i])):
            price = getattr(r, f"best_{side}")
            if pd.isna(price):
                continue
            ev = p * float(O.american_to_decimal(price)) - 1
            if ev > 0 and (best is None or ev > best[2]):
                best = (side, price, ev)
        if best is None:
            continue
        won = (r.y > r.line) if best[0] == "over" else (r.y < r.line)
        rows.append({"game": r.game_id, "won": bool(won), "ev": best[2],
                     "profit": float(O.american_to_decimal(best[1])) - 1 if won else -1.0})
    if not rows:
        return {"bets": 0}
    b = pd.DataFrame(rows)
    cl = b.groupby("game")["profit"].sum()
    se = float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(b)) if len(cl) > 1 else None
    return {"bets": int(len(b)), "won": int(b.won.sum()), "lost": int((~b.won).sum()), "return": round(float(b.profit.mean()), 4),
            "return_se": round(se, 4) if se else None, "ev_claimed": round(float(b.ev.mean()), 4)}


def score(x: pd.DataFrame) -> dict:
    hit = (x["y"] > x["line"]).astype(float).to_numpy()
    ours = (x["our_over"] / (x["our_over"] + x["our_under"])).to_numpy()
    fair = x["fair"].to_numpy()
    res = {"all": measure(ours, fair, hit, x["game_id"].to_numpy())}
    for st in STATS:
        m = (x["stat"] == st).to_numpy()
        res[st] = measure(ours[m], fair[m], hit[m], x["game_id"].to_numpy()[m])
    res["ev_legs"] = {"nudged": ev_legs(x, nudge(ours, fair)), "books_alone": ev_legs(x, fair)}
    return res


def table():
    from cfbmodel import data, model, players, players_data
    from cfbmodel.ratings import CFBRatingEngine
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, engine))
    pg, pinfo = players_data.load_player_games(verbose=False)
    return players.build(games, frame, pg, pinfo, None, None), pinfo, games


def main_lines(season: int, games: pd.DataFrame) -> pd.DataFrame:
    R.set_season(season)
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s = s[~s["market"].str.endswith("_alternate") & s["stat"].isin(STATS) & s["book"].isin(BOOKS)]
    w = s.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index()
    w = w.dropna(subset=["over", "under"])
    w["fair_b"] = O.no_vig_power(w["over"], w["under"])[0]
    dec = lambda a: O.american_to_decimal(a)
    w["dover"], w["dunder"] = dec(w["over"]), dec(w["under"])
    key = ["game_id", "nname", "stat", "line"]
    g = w.groupby(key)
    out = g["fair_b"].mean().rename("fair").to_frame()
    out["best_over"] = w.loc[g["dover"].idxmax(), key + ["over"]].set_index(key)["over"]
    out["best_under"] = w.loc[g["dunder"].idxmax(), key + ["under"]].set_index(key)["under"]
    return out.reset_index()


def live_2026() -> dict:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "docs" / "cfb" / "props" / "2026_w*.json"))):
        for p in json.load(open(f))["props"]:
            if p["alt"] or p["s"] not in STATS or p.get("y") is None or p["y"] == p["L"] or p.get("fo") is None:
                continue
            if p.get("po") is None or p.get("pu") is None or (p["po"] + p["pu"]) <= 0:
                continue
            bk = p.get("bk") or {}
            def best(i):
                prices = [v[i] for b, v in bk.items() if b in BOOKS and v and len(v) > i and v[i] is not None]
                return max(prices, key=lambda a: float(O.american_to_decimal(a))) if prices else np.nan
            rows.append({"game_id": p["g"], "stat": p["s"], "line": p["L"], "y": p["y"], "fair": p["fo"],
                         "our_over": p["po"], "our_under": p["pu"], "best_over": best(0), "best_under": best(1)})
    x = pd.DataFrame(rows)
    return score(x) if len(x) else {"props": 0}


def main():
    from cfbmodel import players
    t, pinfo, games = table()
    model = P.PropModel.load(PARAMS)
    res = {"rule": __doc__.split("Rule:")[1].split("Writes")[0].strip(), "params": str(PARAMS.relative_to(ROOT))}
    for kind in ("game_day", "bet_time"):
        tk = t.copy()
        if kind == "bet_time":
            m = tk["played"].astype(bool)
            tk.loc[m, "active"] = tk.loc[m, "expected"].astype(int)
        for season in (2025, 2024):
            pproj = players.walk_forward(tk, first_test=season)
            lines = main_lines(season, games)
            rows = R.player_frame(pproj, pinfo)
            x = R.model_chances(lines.merge(rows, on=["game_id", "nname"], how="inner"), model)
            x = x[x["y"] != x["line"]]
            res.setdefault(kind, {})[str(season)] = score(x)
            a = res[kind][str(season)]["all"]["nudge"]
            print(kind, season, "props", res[kind][str(season)]["all"]["props"], "nudge-books", a, flush=True)
    res["live_2026"] = live_2026()
    checks = []
    for kind in ("game_day", "bet_time"):
        for season in ("2025", "2024"):
            for st in ["all"] + STATS:
                checks.append(res[kind][season][st]["nudge"]["minus_books"] < 0)
    hold = [res[k]["2024"]["all"]["nudge"] for k in ("game_day", "bet_time")]
    res["verdict"] = {"measures_better": int(sum(checks)), "measures": len(checks),
                      "holdout_by_1se": all(h["minus_books"] <= -h["se"] for h in hold),
                      "kept": bool(all(checks) and all(h["minus_books"] <= -h["se"] for h in hold)),
                      "live_2026_worse_by_1se": bool(res["live_2026"].get("all", {}).get("nudge", {}).get("minus_books", 0)
                                                     > res["live_2026"].get("all", {}).get("nudge", {}).get("se", 1))}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
