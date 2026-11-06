"""PFF prop inputs for the NFL New projections, judged where the floors strategy bets (Nov 6, 2026).

    python experiments/nfl_pff_prop_inputs_test.py [--skip-props]

THE RULE (written before the first run, Nov 6, 2026)
-----------------------------------------------------
The board's New projections (nflmodel/players_ml.py: WR and RB receptions and receiving yards, RB
carries and rushing yards) already use PFF route share. The floors strategy (claude/betting-
strategies.md, strategy 1) bets alternate rungs the model expects a player to clear. This test adds
the PFF inputs that describe the bottom of a player's range, as of each game from earlier games
only (the board's player decay, players.asof with PLAYER_HL, last season carried):

  receiving chain (receptions, receiving yards):
    adot_avg       targets-weighted average depth of target (receiving_summary)
    slot_avg       slot snaps / (slot + wide + inline snaps)
    contested_avg  contested targets / targets
    yac_avg        yards after catch / receptions
    opp_cov        the opponent's coverage grade: snap-weighted grades_coverage_defense of its
                   defenders (defense_summary), decayed over its previous games (half-life 6
                   games, last season carried at half weight)
  rushing chain (carries, rushing yards):
    ybc_avg        (yards - yards after contact) / attempts (rushing_summary)
    break_avg      breakaway attempts / attempts
    zone_avg       zone attempts / (zone + gap attempts)
    opp_rdef       the opponent's run-defense grade (grades_run_defense), as opp_cov

Variants, same walk-forward as the board (players_ml.walk_forward, first test season 2015,
Questionable step applied), PFF route share in all of them:
  current   players_ml.FEATS + route share (the board today)
  receiving current + the receiving inputs     judged on the receiving chain
  rushing   current + the rushing inputs       judged on the rushing chain

Measures (keep rule, CONVENTIONS.md: a chain is kept only if better on every measure of its chain
and worse on none; any measure worse makes it a trade-off for Ryan; standard errors clustered by
game, reported, no threshold):
  receiving chain (10): projection RMSE for WR receptions, WR receiving yards, RB receptions, RB
    receiving yards, tuning 2015-2018 and holdout 2019-2025 (8); Brier at the real 2023-2025
    DraftKings/FanDuel main lines, receptions and receiving yards, WR and RB, pricing snapshot,
    pooled (1); Brier on the floor rungs: 2023-2025 receiving-yards alternate rungs (Over, pricing
    snapshot) whose books' reading (parlays.RUNG_CAL on the price) is 65-95%, WR and RB, pooled (1).
  rushing chain (6): RMSE for RB carries and RB rushing yards, tuning and holdout (4); Brier at the
    main lines, rushing yards and rush attempts, RB (1); Brier on the floor rungs, rushing-yards
    alternates, RB, 65-95% (1).
The closing snapshot is not scored (the floors strategy bets the pricing snapshot).
Nothing goes on the board from this script; a kept chain comes back to Ryan with a plan first.

Writes experiments/out/nfl_pff_prop_inputs_test.json (scores only, no PFF rows).
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
from nflmodel import config as C, data, odds as O, parlays, players as PL, players_ml as ML, propbackfill as B, props as P, roles  # noqa: E402
from nflmodel import propmarket as PM  # noqa: E402
import props_real_lines as RL  # noqa: E402
from props_calibration import FAMILY, load_proj  # noqa: E402

OUT = HERE / "out" / "nfl_pff_prop_inputs_test.json"
FIRST, HOLD_FROM, LAST = 2015, 2019, 2025
PROP_SEASONS = (2023, 2024, 2025)
REC_F = ["adot_avg", "slot_avg", "contested_avg", "yac_avg", "opp_cov"]
RUSH_F = ["ybc_avg", "break_avg", "zone_avg", "opp_rdef"]
CHAINS = {
    "receiving": {"feats": REC_F, "rmse": {"receptions": ("WR", "RB"), "rec_yds": ("WR", "RB")},
                  "lines": {"receptions", "rec_yds"}, "groups": ("WR", "RB"), "alt": "player_reception_yds_alternate"},
    "rushing": {"feats": RUSH_F, "rmse": {"carries": ("RB",), "rush_yds": ("RB",)},
                "lines": {"rush_yds", "carries"}, "groups": ("RB",), "alt": "player_rush_yds_alternate"},
}
TEAM_HL, TEAM_CARRY = 6.0, 0.5
PLAYOFF = {28: 19, 29: 20, 30: 21, 32: 22}


def _ids() -> pd.DataFrame:
    pl = pd.read_parquet(C.RAW / "players.parquet", columns=["gsis_id", "pff_id"]).dropna()
    pl["pff_id"] = pd.to_numeric(pl["pff_id"], errors="coerce")
    return pl.dropna().drop_duplicates("pff_id").rename(columns={"gsis_id": "nfl_id"})


def _load(table: str) -> pd.DataFrame:
    p = pff_data.load(table, range(2014, LAST + 2), league="nfl")
    p["week"] = p["week"].astype(int).map(lambda w: PLAYOFF.get(w, w))
    p = p[p["week"].between(1, 22)].copy()
    p["season"] = p["season"].astype(int)
    return p.merge(_ids(), left_on="player_id", right_on="pff_id", how="left")


def player_inputs(table: pd.DataFrame) -> pd.DataFrame:
    """REC_F (but opp_cov) and RUSH_F (but opp_rdef) for every row of the player table."""
    rec = _load("receiving_summary")
    for c in ("avg_depth_of_target", "targets", "slot_snaps", "wide_snaps", "inline_snaps", "contested_targets",
              "yards_after_catch", "receptions"):
        rec[c] = pd.to_numeric(rec[c], errors="coerce").fillna(0.0)
    rec["adot_x"] = rec["avg_depth_of_target"] * rec["targets"]
    rec["align"] = rec["slot_snaps"] + rec["wide_snaps"] + rec["inline_snaps"]
    rush = _load("rushing_summary")
    for c in ("attempts", "yards", "yards_after_contact", "breakaway_attempts", "zone_attempts", "gap_attempts"):
        rush[c] = pd.to_numeric(rush[c], errors="coerce").fillna(0.0)
    rush["ybc"] = rush["yards"] - rush["yards_after_contact"]
    rush["zg"] = rush["zone_attempts"] + rush["gap_attempts"]
    keep_r = ["nfl_id", "season", "week", "adot_x", "targets", "slot_snaps", "align", "contested_targets", "yards_after_catch", "receptions"]
    keep_u = ["nfl_id", "season", "week", "ybc", "attempts", "breakaway_attempts", "zone_attempts", "zg"]
    r = rec.dropna(subset=["nfl_id"])[keep_r].groupby(["nfl_id", "season", "week"], as_index=False).sum()
    u = rush.dropna(subset=["nfl_id"])[keep_u].groupby(["nfl_id", "season", "week"], as_index=False).sum()
    r = r.rename(columns={"targets": "p_tgt", "receptions": "p_rec"})
    t = table[["game_id", "player_id", "team", "season", "week", "t", "active", "played"]].copy()
    t = t.merge(r.rename(columns={"nfl_id": "player_id"}), on=["player_id", "season", "week"], how="left")
    t = t.merge(u.rename(columns={"nfl_id": "player_id"}), on=["player_id", "season", "week"], how="left")
    played = t["played"].fillna(False).astype(bool) & (t["active"] == 1)
    cols = ["adot_x", "p_tgt", "slot_snaps", "align", "contested_targets", "yards_after_catch", "p_rec",
            "ybc", "attempts", "breakaway_attempts", "zone_attempts", "zg"]
    for c in cols:
        t[c] = t[c].where(played)              # games he didn't play add nothing
    f = PL.asof(t, "player_id", cols, None, PL.PLAYER_HL)
    out = f[["game_id", "player_id"]].copy()
    div = lambda a, b, m=0.0: (f[a + "_sum"] / f[b + "_sum"].where(f[b + "_sum"] > m))
    out["adot_avg"] = div("adot_x", "p_tgt", 2)
    out["slot_avg"] = div("slot_snaps", "align", 10).clip(0, 1)
    out["contested_avg"] = div("contested_targets", "p_tgt", 2).clip(0, 1)
    out["yac_avg"] = div("yards_after_catch", "p_rec", 2)
    out["ybc_avg"] = div("ybc", "attempts", 5)
    out["break_avg"] = div("breakaway_attempts", "attempts", 5).clip(0, 1)
    out["zone_avg"] = div("zone_attempts", "zg", 5).clip(0, 1)
    return out


def team_defense(table: pd.DataFrame) -> pd.DataFrame:
    """opp_cov and opp_rdef per (game_id, team): the opponent's decayed unit grades before the game."""
    d = _load("defense_summary").drop(columns=["team"], errors="ignore")      # PFF's own team field
    for c in ("grades_coverage_defense", "grades_run_defense", "snap_counts_coverage", "snap_counts_run_defense"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # PFF team_name -> board team, by the majority of linked offensive players' board team that week
    # (the player table holds offensive players only; first run's sanity check had whole teams
    # missing when linked through defenders)
    r = _load("receiving_summary")
    link = r.dropna(subset=["nfl_id"])[["season", "week", "team_name", "nfl_id"]].merge(
        table[["player_id", "season", "week", "team"]].rename(columns={"player_id": "nfl_id"}), on=["nfl_id", "season", "week"])
    tm = link.groupby(["season", "team_name", "team"]).size().rename("n").reset_index().sort_values("n").drop_duplicates(["season", "team_name"], keep="last")
    d = d.merge(tm[["season", "team_name", "team"]], on=["season", "team_name"], how="inner")
    agg = []
    for g, sc, name in (("grades_coverage_defense", "snap_counts_coverage", "cov"), ("grades_run_defense", "snap_counts_run_defense", "rdef")):
        x = d.dropna(subset=[g, sc])
        x = x[x[sc] > 0]
        a = x.assign(w=x[sc], wg=x[sc] * x[g]).groupby(["team", "season", "week"], as_index=False)[["w", "wg"]].sum()
        a[name] = a["wg"] / a["w"]
        agg.append(a[["team", "season", "week", name]])
    tw = agg[0].merge(agg[1], on=["team", "season", "week"], how="outer").sort_values(["team", "season", "week"])
    # as of a game: decayed mean over the team's earlier PFF games
    lut = {}
    for team, g in tw.groupby("team"):
        g = g.reset_index(drop=True)
        lut[team] = g
    games = table[["game_id", "team", "opp", "season", "week"]].drop_duplicates(["game_id", "team"])
    out = []
    for r in games.itertuples(index=False):
        g = lut.get(r.opp)
        val = [np.nan, np.nan]
        if g is not None:
            prev = g[(g.season < r.season) | ((g.season == r.season) & (g.week < r.week))]
            prev = prev[prev.season >= r.season - 1].iloc[::-1]
            if len(prev):
                k = np.arange(len(prev))
                wts = 0.5 ** (k / TEAM_HL) * np.where(prev["season"].to_numpy() < r.season, TEAM_CARRY, 1.0)
                for j, c in enumerate(("cov", "rdef")):
                    x = prev[c].to_numpy(float)
                    ok = ~np.isnan(x)
                    val[j] = float((wts[ok] * x[ok]).sum() / wts[ok].sum()) if ok.any() else np.nan
        out.append((r.game_id, r.team, val[0], val[1]))
    return pd.DataFrame(out, columns=["game_id", "team", "opp_cov", "opp_rdef"])


def with_inputs(base: pd.DataFrame) -> pd.DataFrame:
    pi = player_inputs(base)
    td = team_defense(base)
    return base.merge(pi, on=["game_id", "player_id"], how="left").merge(td, on=["game_id", "team"], how="left")


def run_variant(table: pd.DataFrame, feats: list[str], first: int) -> pd.DataFrame:
    return ML.questionable(ML.walk_forward(table, first=first, feats=feats), first=C.FIRST_TEST_SEASON)


def rmse(cur: pd.DataFrame, new: pd.DataFrame, judged: dict) -> dict:
    key = ["game_id", "player_id"]
    b = cur[cur.played.fillna(False).astype(bool) & (cur.active == 1) & cur.season.between(FIRST, LAST)].set_index(key)
    n = new.set_index(key).loc[b.index]
    out = {}
    for stat, groups in judged.items():
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
                row[per] = {"n": int(len(x)), "current": round(r0, 4), "new": round(r1, 4), "change": round(r1 - r0, 4),
                            "se": round(se, 4), "better": bool(r1 < r0)}
            out[f"{grp} {stat}"] = row
            print(f"  {grp} {stat:10s} tune {row['tune']['current']:.4f} -> {row['tune']['new']:.4f} ({row['tune']['change']:+.4f} ± {row['tune']['se']:.4f})"
                  f"   hold {row['hold']['current']:.4f} -> {row['hold']['new']:.4f} ({row['hold']['change']:+.4f} ± {row['hold']['se']:.4f})", flush=True)
    return out


def rungs(proj: pd.DataFrame, model: P.PropModel, snaps: pd.DataFrame, season: int, market: str) -> pd.DataFrame:
    """Over-only alternate rungs with the model's chance and the books' reading (RUNG_CAL)."""
    s = snaps[(snaps["market"] == market) & (snaps["side"].str.lower() == "over")].copy()
    if s.empty:
        return pd.DataFrame()
    info = pd.read_csv(C.CACHE / f"player_info_{season}.csv", dtype={"player_id": str})
    rows = proj[(proj["season"] == season) & proj["played"].astype(bool)].merge(info[["player_id", "name"]], on="player_id", how="left")
    rows["nname"] = rows["name"].map(PM.norm_name)
    rows = rows[~rows.duplicated(["game_id", "nname"], keep=False)]
    s["nname"] = s["player"].map(PM.norm_name)
    s = s.groupby(["game_id", "nname", "line", "book"], as_index=False)["price"].last()
    xs, ys = zip(*parlays.RUNG_CAL)
    s["books"] = np.interp(O.implied_prob(s["price"].to_numpy(float)), xs, ys)
    s = s[s["books"].between(0.65, 0.95)]
    w = s.merge(rows, on=["game_id", "nname"], how="inner")
    stat = PM.MARKET_STAT[market]
    out = []
    for grp, g in w.groupby("group"):
        if (stat, grp) not in model.counts and (stat, grp) not in model.yards:
            continue
        g = g[P.eligible(g, stat, grp)].reset_index(drop=True)
        if g.empty:
            continue
        over, push, under = model.probs(g, stat, grp, g["line"].to_numpy(float))
        g["our_over"] = over / (over + under)
        g["y"] = P.actual(g, stat)
        g["stat"] = stat
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def props_check(base: pd.DataFrame, chain: str) -> dict:
    cfg = CHAINS[chain]
    saved = P.PropModel.load()
    feats_c = list(ML.FEATS) + list(ML.ROUTE_F)
    feats_n = feats_c + cfg["feats"]
    lines, rr = {"current": [], "new": []}, {"current": [], "new": []}
    for season in PROP_SEASONS:
        snaps = B.load(season, "flag")
        for label, feats in (("current", feats_c), ("new", feats_n)):
            t = run_variant(base, feats, season)
            m = P.PropModel(FAMILY).fit(t[(t.season < season) & (t.season >= 2014)])
            m.iso = saved.iso
            x = RL.priced(t, m, snaps, season)
            if not x.empty:
                x = x[x["stat"].isin(cfg["lines"]) & x["group"].isin(cfg["groups"])]
                lines[label].append(x)
            r = rungs(t, m, snaps, season, cfg["alt"])
            if not r.empty:
                rr[label].append(r[r["group"].isin(cfg["groups"])])
        print(f"  priced {chain} {season}", flush=True)
    res = {}
    for name, store, key in (("main lines", lines, ["game_id", "player_id", "stat", "line", "book"]),
                             ("floor rungs 65-95%", rr, ["game_id", "player_id", "stat", "line", "book"])):
        a = pd.concat(store["current"], ignore_index=True)
        b = pd.concat(store["new"], ignore_index=True)
        a = a[a["y"] != a["line"]].drop_duplicates(key)
        b = b[b["y"] != b["line"]].drop_duplicates(key)
        if name == "main lines":
            a["p"] = a["our_over"] / (a["our_over"] + a["our_under"])
            b["p"] = b["our_over"] / (b["our_over"] + b["our_under"])
            a["bk"] = a["fair_over"]
        else:
            a["p"], b["p"], a["bk"] = a["our_over"], b["our_over"], a["books"]
        j = a[key + ["y", "p", "bk"]].merge(b[key + ["p"]], on=key, suffixes=("_c", "_n"))
        hit = (j["y"] > j["line"]).astype(float)
        bc, bn, bm = (j["p_c"] - hit) ** 2, (j["p_n"] - hit) ** 2, (j["bk"] - hit) ** 2
        d = (bn - bc).groupby(j["game_id"]).sum()
        res[name] = {"n": int(len(j)), "games": int(j["game_id"].nunique()), "brier_current": round(float(bc.mean()), 5),
                     "brier_new": round(float(bn.mean()), 5), "brier_books": round(float(bm.mean()), 5),
                     "change": round(float((bn - bc).mean()), 6), "se": round(float(d.std(ddof=1) * np.sqrt(len(d)) / len(j)), 6),
                     "better": bool(bn.mean() < bc.mean()), "hit_rate": round(float(hit.mean()), 4)}
        r = res[name]
        print(f"  {chain} {name:18s} n {r['n']:6d}  books {r['brier_books']:.5f}  current {r['brier_current']:.5f}  new {r['brier_new']:.5f}"
              f"  ({r['change']:+.6f} ± {r['se']:.6f})  cleared {r['hit_rate']:.1%}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-props", action="store_true")
    args = ap.parse_args()
    if not pff_data.available():
        sys.exit("PFF data not found (../pff-data or PFF_DATA_DIR)")
    games = data.load_games()
    R = roles.load_all(games, verbose=False)
    base = with_inputs(ML.add_features(load_proj(None), R))
    w = base[base.season.between(2015, LAST) & base.played.fillna(False).astype(bool)]
    present = {c: round(float(w[w.group.isin(["WR", "RB"])][c].notna().mean()), 4) for c in REC_F + RUSH_F}
    print("present (WR/RB played 2015-2025):", present, flush=True)
    res = {"present": present}
    cur = run_variant(base, list(ML.FEATS) + list(ML.ROUTE_F), FIRST)
    for chain, cfg in CHAINS.items():
        print(f"\n{chain} chain, walk-forward RMSE, current -> with PFF inputs:", flush=True)
        new = run_variant(base, list(ML.FEATS) + list(ML.ROUTE_F) + cfg["feats"], FIRST)
        out = {"rmse": rmse(cur, new, cfg["rmse"])}
        if not args.skip_props:
            out["props"] = props_check(base, chain)
        m = [v[p]["better"] for v in out["rmse"].values() for p in ("tune", "hold")]
        if "props" in out:
            m += [out["props"][k]["better"] for k in ("main lines", "floor rungs 65-95%")]
        out["better_on"] = f"{sum(m)} of {len(m)}"
        out["kept"] = bool(all(m)) and "props" in out
        print(f"{chain}: better on {out['better_on']} -> {'kept' if out['kept'] else 'not kept' if not any(m) else 'not kept (mixed: a trade-off for Ryan)' if 'props' in out else 'props not run'}", flush=True)
        res[chain] = out
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
