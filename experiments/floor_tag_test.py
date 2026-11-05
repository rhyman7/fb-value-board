"""Every-game floors on the 2025 ladder rungs: adjusted vs raw floor, and the setup flags.

    python experiments/floor_tag_test.py

Rule set before running (PROPS.md "Every-game floors on the boards", Nov 5, 2026). For every
2025 over-only alternate rung (passing, rushing, receiving; DraftKings and FanDuel, the pricing
snapshot, the better of the two prices), the player's floor from his earlier 2025 games
(nflmodel/floors.py), raw and adjusted. A version "tags" a rung when its floor is above the line.

1. Adjusted vs raw: the highest tagged rung per player, game and stat, return per $1 at the better
   price (standard error clustered by game); NFL, college and pooled. Also every tagged rung's
   clear rate against the books' raw implied chance and margin-free chance (10% rung margin).
2. Flags, on adjusted-tagged rungs: share cleared with each flag vs without (30+ flagged rungs to
   judge). Flags use what was known going in, except the starting QB and teammates out, which are
   taken from who played (nearly always known by kickoff).

Writes experiments/out/floor_tag_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import config as C, floors as F, odds as O, propbackfill as B, propmarket as PM  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "floor_tag_test.json"
SEASON = 2025
STAT = {"player_pass_yds_alternate": "pass", "player_rush_yds_alternate": "rush", "player_reception_yds_alternate": "rec"}


def _se(x: pd.Series, cl: pd.Series) -> float:
    x = pd.Series(np.asarray(x, float))
    cl = pd.Series(np.asarray(cl))
    if len(x) < 2:
        return float("nan")
    d = (x - x.mean()).groupby(cl.to_numpy()).sum()
    return float(np.sqrt((d ** 2).sum()) / len(x))


# ------------------------------------------------------------------------------------------
def nfl_inputs():
    from nflmodel import data
    games = data.load_games(refresh=False)
    pg = pd.read_csv(C.CACHE / f"player_games_{SEASON}.csv")
    info = pd.read_csv(C.CACHE / f"player_info_{SEASON}.csv", dtype={"player_id": str})
    snaps = pd.read_csv(C.CACHE / f"snaps_{SEASON}.csv", dtype={"player_id": str})
    depth = pd.read_csv(C.CACHE / f"depth_{SEASON}.csv", dtype={"player_id": str})
    log, ctx = F.nfl_log(SEASON, pg, games, snaps, depth)
    groups = {str(a): F.group(p) for a, p in zip(info["player_id"], info["position"])}
    names = dict(zip(info["player_id"].astype(str), info["name"]))
    s = B.load(SEASON, "flag")
    s = s[s["market"].isin(STAT) & (s["side"].str.lower() == "over")].copy()
    s = s.sort_values("snapshot").drop_duplicates(["game_id", "player", "market", "line", "book"], keep="last")
    return log, ctx, groups, names, s[["game_id", "player", "market", "line", "book", "price"]]


def cfb_inputs():
    from cfbmodel import data as CD, config as CC
    import cfb_props_real_lines as R
    games, _, _ = CD.load_all(refresh=False, verbose=False)
    pg = pd.read_csv(CC.CACHE / f"player_games_{SEASON}.csv")
    info = pd.read_csv(CC.CACHE / f"player_info_{SEASON}.csv")
    groups = {str(a): F.group(p) for a, p in zip(info["player_id"], info["position"])}
    names = dict(zip(info["player_id"].astype(str), info["name"]))
    log, ctx = F.cfb_log(SEASON, pg, games, groups)
    s = R.snapshots()
    s = s[pd.to_datetime(s["snapshot"], utc=True) < pd.to_datetime(s["commence"], utc=True)]
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s = s[s["market"].isin(STAT) & (s["side"] == "over")]
    s = s.sort_values("snapshot").drop_duplicates(["game_id", "player", "market", "line", "book"], keep="last")
    return log, ctx, groups, names, s[["game_id", "player", "market", "line", "book", "price"]]


# ------------------------------------------------------------------------------------------
def rung_table(league, log, ctx, groups, names, s) -> pd.DataFrame:
    """One row per rung with both floors, the actual, the flags."""
    s = s.copy()
    s["stat"] = s["market"].map(STAT)
    s["nname"] = s["player"].map(PM.norm_name)
    s = s.sort_values("price", ascending=False).drop_duplicates(["game_id", "nname", "stat", "line"])  # better book
    played = log.copy()
    played["nname"] = played["player_id"].map(names).map(lambda n: PM.norm_name(n) if isinstance(n, str) else None)
    played = played[~played.duplicated(["game_id", "nname"], keep=False)]
    x = s.merge(played, on=["game_id", "nname"], how="inner")
    x["grp"] = x["player_id"].map(groups)
    x = x[[st in F.SPLIT.get(g, ()) for st, g in zip(x["stat"], x["grp"])]]
    by_player = {p: g.sort_values("t") for p, g in log.groupby("player_id")}
    by_game_team = {k: set(g["player_id"]) for k, g in log.groupby(["game_id", "team"])}
    rows = []
    cache = {}
    allow_at = {}
    for r in x.itertuples():
        key = (r.player_id, r.game_id)
        if key not in cache:
            hist = by_player[r.player_id]
            prior = hist[hist["t"] < r.t]
            cache[key] = (F.player_floor(prior, r.grp, r.team, league, adjusted=False),
                          F.player_floor(prior, r.grp, r.team, league, adjusted=True))
        raw, adj = cache[key]
        if r.stat not in raw and r.stat not in adj:
            continue
        y = {"pass": r.pass_yds, "rush": r.rush_yds, "rec": r.rec_yds}[r.stat]
        fl = []
        if r.stat in adj:
            if r.t not in allow_at:
                allow_at[r.t] = F.allowed(log[log["t"] < r.t], groups)
            g = adj[r.stat]["rows"]
            team_log = log[(log["team"] == r.team) & (log["t"] < r.t)]
            mate = F.top_mate(team_log, set(g["game_id"]), r.player_id, r.stat, groups)
            in_game = by_game_team.get((r.game_id, r.team), set())
            out = {mate} if mate is not None and mate not in in_game else set()
            now = {**ctx.get((r.game_id, r.team), {}), "team": r.team, "opp": r.opp}
            fl = F.flags(r.stat, r.grp, g, ctx, now, allow_at[r.t], mate, out)
        rows.append({"league": league, "game_id": r.game_id, "player_id": r.player_id, "stat": r.stat, "grp": r.grp,
                     "line": float(r.line), "price": float(r.price), "y": float(y),
                     "raw": raw.get(r.stat, {}).get("floor"), "raw_g": raw.get(r.stat, {}).get("g"),
                     "adj": adj.get(r.stat, {}).get("floor"), "adj_g": adj.get(r.stat, {}).get("g"),
                     "flags": fl})
    d = pd.DataFrame(rows)
    d["hit"] = (d["y"] > d["line"]).astype(float)
    d["dec"] = O.american_to_decimal(d["price"])
    d["implied"] = O.implied_prob(d["price"])
    d["fair"] = O.one_sided_fair(d["price"], PM.ALT_MARGIN, "power")
    return d


def grade_best(d: pd.DataFrame, col: str) -> dict:
    t = d[d[col].notna() & (d[col] > d["line"])]
    b = t.sort_values("line", ascending=False).drop_duplicates(["game_id", "player_id", "stat"])
    if b.empty:
        return {"bets": 0}
    prof = np.where(b["hit"] > 0, b["dec"] - 1, -1.0)
    return {"bets": int(len(b)), "games": int(b["game_id"].nunique()), "won": round(float(b["hit"].mean()), 4),
            "implied": round(float(b["implied"].mean()), 4), "fair": round(float(b["fair"].mean()), 4),
            "median_price": float(b["price"].median()), "avg_line_minus_floor": round(float((b["line"] - b[col]).mean()), 1),
            "return": round(float(prof.mean()), 4), "return_se": round(_se(prof, b["game_id"]), 4)}


def tagged(d: pd.DataFrame, col: str) -> dict:
    t = d[d[col].notna() & (d[col] > d["line"])]
    if t.empty:
        return {"rungs": 0}
    return {"rungs": int(len(t)), "cleared": round(float(t["hit"].mean()), 4),
            "cleared_se": round(_se(t["hit"], t["game_id"]), 4),
            "implied": round(float(t["implied"].mean()), 4), "fair": round(float(t["fair"].mean()), 4)}


def flag_report(d: pd.DataFrame) -> dict:
    t = d[d["adj"].notna() & (d["adj"] > d["line"])]
    out = {}
    for k in F.FLAG_TEXT:
        has = t["flags"].map(lambda f: k in f)
        a, b = t[has], t[~has]
        r = {"flagged": int(len(a)), "cleared_flagged": round(float(a["hit"].mean()), 4) if len(a) else None,
             "cleared_unflagged": round(float(b["hit"].mean()), 4) if len(b) else None,
             "fair_flagged": round(float(a["fair"].mean()), 4) if len(a) else None,
             "fair_unflagged": round(float(b["fair"].mean()), 4) if len(b) else None}
        if len(a) >= 2:
            diff = a["hit"].mean() - b["hit"].mean()
            r["diff"] = round(float(diff), 4)
            r["diff_se"] = round(float(np.hypot(_se(a["hit"], a["game_id"]), _se(b["hit"], b["game_id"]))), 4)
        if len(a) < 30:
            r["verdict"] = "not enough to judge (under 30)"
        else:
            good = r["diff"] > 0 if k in F.UP_FLAGS else r["diff"] < 0
            r["verdict"] = "show" if good else "off"
        out[k] = r
    return out


def main():
    parts = []
    for league, fn in (("nfl", nfl_inputs), ("cfb", cfb_inputs)):
        print(f"{league} ...", flush=True)
        parts.append(rung_table(league, *fn()))
    d = pd.concat(parts, ignore_index=True)
    res = {"season": SEASON, "rungs": int(len(d))}
    for name, sub in (("nfl", d[d.league == "nfl"]), ("cfb", d[d.league == "cfb"]), ("pooled", d)):
        res[name] = {"best_rung_raw": grade_best(sub, "raw"), "best_rung_adjusted": grade_best(sub, "adj"),
                     "tagged_raw": tagged(sub, "raw"), "tagged_adjusted": tagged(sub, "adj"),
                     "by_stat_adjusted": {st: grade_best(g, "adj") for st, g in sub.groupby("stat")}}
    res["flags"] = flag_report(d)
    res["flags_by_league"] = {lg: flag_report(g) for lg, g in d.groupby("league")}
    pr, pa = res["pooled"]["best_rung_raw"], res["pooled"]["best_rung_adjusted"]
    res["verdict_adjusted"] = "adjusted stays" if pa.get("return", -9) >= pr.get("return", -9) else "adjusted worse: ask Ryan"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    d.drop(columns="flags").assign(flags=d["flags"].map(",".join)).to_csv(HERE / "out" / "floor_tag_rungs.csv", index=False)
    print(json.dumps({k: res[k] for k in ("nfl", "cfb", "pooled")}, indent=1, default=float))
    print(json.dumps(res["flags"], indent=1))
    print(res["verdict_adjusted"])


if __name__ == "__main__":
    main()
