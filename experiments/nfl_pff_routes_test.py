"""PFF route share in the NFL New projections (WR and RB receptions and receiving yards).

    PFF_DATA_DIR=../pff-data python experiments/nfl_pff_routes_test.py [--skip-props]

Nov 5, 2026. The Oct 31 routes test (experiments/routes_test.py, PLAYERS.md) used nflverse's
after-the-season participation data as a stand-in for routes and found they added about 0.85%
to the target-share fit beyond snap share, so PFF was put off for the NFL. Ryan now has PFF Pro,
and the real weekly PFF numbers passed for college on Nov 4 (cfb_pff_role_test.py). This tests
them where the NFL board would use them: the boosted New projections (nflmodel/players_ml.py),
which the board shows for WR and RB receptions and receiving yards (players_ml.ADOPT).

Data: PFF receiving_summary, NFL, 2014-2026 weeks 1-18 (private repo rhyman7/pff-data, read
through cfbmodel.pff_data with league="nfl"; never committed here). PFF players are matched to
the board's (nflverse gsis) ids by nflverse's players file `pff_id`; player-weeks to the board's
games by season and week (regular season only: playoff weeks carry no PFF routes here).

Features, as of each game (earlier games he played only; the board's player decay, players.asof
with PLAYER_HL, last season carried at players.CARRY):
  route_avg    decayed routes / decayed team pass plays (team pass plays: the most `pass_plays`
               any of the team's players logged that week in the table)
  route_last   route share in the last game he played with PFF data
  route_trend  route_last - route_avg
  tprr_avg     decayed targets / decayed routes (missing below 20 decayed routes)
Missing before 2014 and with no history; the boosted model handles missing values.

Variants, same walk-forward (each season predicted by a model trained on earlier seasons,
players_ml.walk_forward, first test season 2015 for both), Questionable step applied to both:
  current   players_ml.FEATS (the board today)
  routes    players_ml.FEATS + the four features above

Rule, written before running (Nov 5, 2026, CONVENTIONS.md "Model tests: the keep rule"): kept
only if "routes" is better than "current" on every measure below and worse on none; any measure
worse makes it a trade-off for Ryan. Standard errors (clustered by game) reported, no threshold.
  1. Projection RMSE on player-games where he was active and played, for each of WR receptions,
     WR receiving yards, RB receptions, RB receiving yards: tuning 2015-2018 and holdout
     2019-2025 (8 measures).
  2. Props check: Brier at the real 2023-2025 DraftKings/FanDuel main lines
     (experiments/props_real_lines_ml.py's scoring), receptions and receiving yards for WR and RB
     together, pooled over 2023-2025, at the pricing snapshot (pregame projections) and at the
     close (4 measures). Per-season numbers are reported too but not judged.
Nothing goes on the board from this script; a kept result comes back to Ryan with a plan first.

Writes experiments/out/nfl_pff_routes_test.json (scores only, no PFF rows).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
from cfbmodel import pff_data  # noqa: E402
from nflmodel import config as C, data, players as PL, players_ml as ML, propbackfill as B, props as P, roles  # noqa: E402
import props_real_lines as RL  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

OUT = HERE / "out" / "nfl_pff_routes_test.json"
FIRST, HOLD_FROM, LAST = 2015, 2019, 2025
ROUTE_F = ["route_avg", "route_last", "route_trend", "tprr_avg"]
JUDGED = {"receptions": ("WR", "RB"), "rec_yds": ("WR", "RB")}
PROP_SEASONS = (2023, 2024, 2025)


# ------------------------------------------------------------------ PFF routes onto the board's ids

def pff_routes() -> pd.DataFrame:
    rec = pff_data.load("receiving_summary", range(2014, LAST + 2), league="nfl")
    rec = rec[rec["week"].between(1, 18)]
    rec = rec[["season", "week", "player_id", "player", "team_name", "routes", "pass_plays", "targets"]].copy()
    team = rec.groupby(["season", "week", "team_name"])["pass_plays"].max().rename("team_pass").reset_index()
    rec = rec.merge(team, on=["season", "week", "team_name"], how="left")
    pl = pd.read_parquet(C.RAW / "players.parquet", columns=["gsis_id", "pff_id"]).dropna()
    pl["pff_id"] = pd.to_numeric(pl["pff_id"], errors="coerce")
    pl = pl.dropna().drop_duplicates("pff_id")
    rec = rec.merge(pl.rename(columns={"gsis_id": "nfl_id"}), left_on="player_id", right_on="pff_id", how="left")
    return rec


def route_features(table: pd.DataFrame, rec: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t = table[["game_id", "player_id", "team", "season", "week", "t", "active", "played", "group"]].copy()
    r = rec.dropna(subset=["nfl_id"]).drop_duplicates(["nfl_id", "season", "week"])
    t = t.merge(r[["nfl_id", "season", "week", "routes", "team_pass", "targets"]].rename(columns={"nfl_id": "player_id", "targets": "pff_tgt"}),
                on=["player_id", "season", "week"], how="left")
    played = t["played"].fillna(False).astype(bool) & (t["active"] == 1)
    # the team's pass plays that game (the most any of its players logged); a player who played
    # but has no PFF row that week ran no routes, as long as his team has PFF data
    t["team_pass"] = t.groupby(["game_id", "team"])["team_pass"].transform("max")
    fill = played & t["routes"].isna() & t["team_pass"].gt(0)
    t.loc[fill, ["routes", "pff_tgt"]] = 0.0
    use = played & t["routes"].notna() & t["team_pass"].gt(0)
    t["rt"] = t["routes"].where(use)
    t["tp"] = t["team_pass"].where(use)
    t["tg"] = t["pff_tgt"].where(use)
    f = PL.asof(t, "player_id", ["rt", "tp", "tg"], None, PL.PLAYER_HL)     # sorted by player, t; index 0..n-1
    out = f[["game_id", "player_id"]].copy()
    out["route_avg"] = (f["rt_sum"] / f["tp_sum"].replace(0, np.nan)).clip(0, 1)
    out["tprr_avg"] = (f["tg_sum"] / f["rt_sum"].where(f["rt_sum"] >= 20)).clip(0, 1)
    share = (f["rt"] / f["tp"]).clip(0, 1)
    out["route_last"] = share.groupby(f["player_id"]).transform(lambda s: s.ffill().shift())
    out["route_trend"] = out["route_last"] - out["route_avg"]
    pw = use & t["group"].isin(["WR", "TE", "RB"]) & (t["season"] >= 2014)
    cov = {"pff_rows": int(len(rec)), "pff_matched_share": round(float(rec["nfl_id"].notna().mean()), 4),
           "board_receiver_games_2014on": int((played & t["group"].isin(["WR", "TE", "RB"]) & (t["season"] >= 2014) & (t["week"] <= 18)).sum()),
           "with_routes": int(pw.sum())}
    return out, cov


# ------------------------------------------------------------------------------- the two variants

def run_variant(table: pd.DataFrame, feats: list[str], first: int) -> pd.DataFrame:
    return ML.questionable(ML.walk_forward(table, first=first, feats=feats), first=C.FIRST_TEST_SEASON)


def rmse(cur: pd.DataFrame, new: pd.DataFrame) -> dict:
    key = ["game_id", "player_id"]
    b = cur[cur.played.fillna(False).astype(bool) & (cur.active == 1) & cur.season.between(FIRST, LAST)].set_index(key)
    n = new.set_index(key).loc[b.index]
    out = {}
    for stat, groups in JUDGED.items():
        for grp in groups:
            row = {}
            for per, pm in (("tune", b.season < HOLD_FROM), ("hold", b.season >= HOLD_FROM)):
                x = b[(b.group == grp) & pm & b[stat].notna()]
                y = x[stat].to_numpy(float)
                e0 = (x[stat + "_proj"].to_numpy(float) - y) ** 2
                e1 = (n.loc[x.index, stat + "_proj"].to_numpy(float) - y) ** 2
                d = pd.Series(e1 - e0).groupby(x.index.get_level_values("game_id").to_numpy()).sum()
                r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
                se = float(d.std(ddof=1) * np.sqrt(len(d)) / len(e0) / (2 * r0))
                row[per] = {"n": int(len(x)), "rmse_current": round(r0, 4), "rmse_routes": round(r1, 4),
                            "change": round(r1 - r0, 4), "se": round(se, 4), "better": bool(r1 < r0)}
            out[f"{grp} {stat}"] = row
            print(f"  {grp} {stat:10s} tune {row['tune']['rmse_current']:.4f} -> {row['tune']['rmse_routes']:.4f} ({row['tune']['change']:+.4f} ± {row['tune']['se']:.4f})"
                  f"   hold {row['hold']['rmse_current']:.4f} -> {row['hold']['rmse_routes']:.4f} ({row['hold']['change']:+.4f} ± {row['hold']['se']:.4f})", flush=True)
    return out


def props_check(R: dict, rec: pd.DataFrame) -> dict:
    saved = P.PropModel.load()
    rows = {}
    for snap, base, kind in (("pricing", RL.load_pregame(), "flag"), ("close", load_proj(None), "close")):
        tab = ML.add_features(base, R, routes=False)
        rf, _ = route_features(tab, rec)
        tab = tab.merge(rf, on=["game_id", "player_id"], how="left")
        for season in PROP_SEASONS:
            for label, feats in (("current", list(ML.FEATS)), ("routes", list(ML.FEATS) + ROUTE_F)):
                t = run_variant(tab, feats, season)
                m = P.PropModel(FAMILY).fit(t[(t.season < season) & (t.season >= 2014)])
                m.iso = saved.iso
                x = RL.priced(t, m, B.load(season, kind), season)
                if x.empty:
                    continue
                x = x[x["stat"].isin(list(JUDGED)) & x["group"].isin(["WR", "RB"])].copy()
                x["label"], x["season_"], x["snap"] = label, season, snap
                rows.setdefault((snap, label), []).append(x)
            print(f"  priced {snap} {season}", flush=True)
    out = {}
    for snap in ("pricing", "close"):
        a = pd.concat(rows[(snap, "current")], ignore_index=True)
        b = pd.concat(rows[(snap, "routes")], ignore_index=True)
        res = {}
        for stat in list(JUDGED) + ["both"]:
            for season in list(PROP_SEASONS) + ["pooled"]:
                xa = a if stat == "both" else a[a.stat == stat]
                xb = b if stat == "both" else b[b.stat == stat]
                if season != "pooled":
                    xa, xb = xa[xa.season_ == season], xb[xb.season_ == season]
                key = ["game_id", "player_id", "stat", "line", "book"] if "book" in xa else ["game_id", "player_id", "stat", "line"]
                xa = xa[xa["y"] != xa["line"]].drop_duplicates(key)
                xb = xb[xb["y"] != xb["line"]].drop_duplicates(key)
                j = xa[key + ["y", "our_over", "our_under", "fair_over"]].merge(
                    xb[key + ["our_over", "our_under"]], on=key, suffixes=("_c", "_r"))
                if j.empty:
                    continue
                hit = (j["y"] > j["line"]).astype(float)
                pc = j["our_over_c"] / (j["our_over_c"] + j["our_under_c"])
                pr = j["our_over_r"] / (j["our_over_r"] + j["our_under_r"])
                bc, br, bm = (pc - hit) ** 2, (pr - hit) ** 2, (j["fair_over"] - hit) ** 2
                d = (br - bc).groupby(j["game_id"]).sum()
                res[f"{stat} {season}"] = {"n": int(len(j)), "brier_current": round(float(bc.mean()), 5), "brier_routes": round(float(br.mean()), 5),
                                           "brier_books": round(float(bm.mean()), 5), "change": round(float((br - bc).mean()), 6),
                                           "se": round(float(d.std(ddof=1) * np.sqrt(len(d)) / len(j)), 6), "better": bool(br.mean() < bc.mean())}
                r = res[f"{stat} {season}"]
                print(f"  {snap:8s} {stat:10s} {season!s:7s} n {r['n']:5d}  books {r['brier_books']:.5f}  current {r['brier_current']:.5f}"
                      f"  routes {r['brier_routes']:.5f}  ({r['change']:+.6f} ± {r['se']:.6f})", flush=True)
        out[snap] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-props", action="store_true")
    args = ap.parse_args()
    games = data.load_games()
    R = roles.load_all(games, verbose=False)
    rec = pff_routes()
    proj = ML.add_features(load_proj(None), R, routes=False)
    rf, cov = route_features(proj, rec)
    print("coverage", cov, flush=True)
    proj = proj.merge(rf, on=["game_id", "player_id"], how="left")
    res = {"coverage": cov, "rule": "routes kept only if better on all 8 RMSE and 4 pooled Brier measures"}
    w = proj[(proj.season.between(2015, 2025)) & proj.group.isin(["WR", "RB"]) & proj.played.fillna(False).astype(bool)]
    res["feature_present_share"] = {c: round(float(w[c].notna().mean()), 4) for c in ROUTE_F}
    print("present", res["feature_present_share"], flush=True)
    print("walk-forward RMSE, current -> with PFF routes:", flush=True)
    cur = run_variant(proj, list(ML.FEATS), FIRST)
    new = run_variant(proj, list(ML.FEATS) + ROUTE_F, FIRST)
    res["rmse"] = rmse(cur, new)
    if not args.skip_props:
        print("\nprops check (Brier at the real DK/FD lines):", flush=True)
        res["props"] = props_check(R, rec)
    # judged: the 8 RMSE measures and the 4 pooled Brier measures (receptions, receiving yards x pricing, close)
    measures = [v[p]["better"] for v in res["rmse"].values() for p in ("tune", "hold")]
    if "props" in res:
        measures += [res["props"][s][f"{st} pooled"]["better"] for s in ("pricing", "close") for st in JUDGED]
    res["better_on"] = f"{sum(measures)} of {len(measures)}"
    res["kept"] = bool(all(measures)) and "props" in res
    print("better on", res["better_on"], "-> kept" if res["kept"] else "-> not kept", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
