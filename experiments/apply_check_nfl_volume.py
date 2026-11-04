"""Combined check for the NFL team-volume changes kept under the keep rule, and the carries head-to-head.

    python experiments/apply_check_nfl_volume.py [--skip-props]

Nov 4, 2026 (FEATURES.md "Re-judging past tests under the keep rule"). Re-judged as better on every
measure: pass rate over expected (PROE) in the team pass-attempt regression, and three overlapping
versions of the team carries regression (PROE; play-calling = pace and neutral pass rate, team and
opponent; the defense's carries allowed "vs usual" in place of the raw count). Here they are measured
together on today's model (the turnover-free spread model of PR #198), with the production code
(players.team_games tendencies, players.ATT_FEATS / CAR_FEATS).

Rule, written before the run:
  1. Team pass attempts: PROE is kept if the regression's average miss on team-games is lower on the
     tuning seasons (2014-2018) and on the holdout (2019 on). Paired by team-game, standard errors
     clustered by game.
  2. Team carries: every version and every combination of the three is scored the same way. The
     kept version is the one that beats the current regression in both periods and is not beaten in
     both periods by any other version; if several are unbeaten, the one with the lowest holdout
     miss. If none beats the current regression in both periods, carries stay as they are.
  3. The chosen set then runs through the player chain (players.walk_forward): RMSE of the
     projections for the stats the volume feeds (QB pass attempts, completions, passing yards; RB
     carries and rushing yards; WR/TE targets and receiving yards), tuning and holdout, on players who
     were active and played. And the props check: Brier against the books at the real 2023-2025
     DraftKings/FanDuel lines (experiments/props_real_lines.py, pricing and closing snapshots), for the
     current chain and the new one. The set goes on the board only if no reported measure is worse;
     anything worse is a trade-off and goes to Ryan.

Writes experiments/out/apply_check_nfl_volume.json; the projection pickles the props check reads go
to data/raw/ (not committed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import config as C, data, extra, injuries, model, players as PL, players_data as PD, roles  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "apply_check_nfl_volume.json"
OUT_PIECES = HERE / "out" / "apply_check_nfl_volume_pieces.json"
BASE = list(PL.VOL_FEATS)
PROE = ["proe_off", "opp_proe_def"]
PLAY = ["pace", "neutral_pass", "opp_pace", "opp_np"]
USUAL = lambda feats: [f for f in feats if f != "opp_car_allowed"] + ["opp_car_usual"]
CAR_VERSIONS = {
    "current": BASE,
    "PROE": BASE + PROE,
    "play-calling": BASE + PLAY,
    "vs usual (swap)": USUAL(BASE),
    "PROE + play-calling": BASE + PROE + PLAY,
    "PROE + vs usual": USUAL(BASE) + PROE,
    "play-calling + vs usual": USUAL(BASE) + PLAY,
    "all three": USUAL(BASE) + PROE + PLAY,
}
PROJ_STATS = {"QB": ["pass_att", "completions", "pass_yds"], "RB": ["carries", "rush_yds"], "WR": ["targets", "rec_yds"], "TE": ["targets", "rec_yds"]}


def tables():
    """The board's player table and the pregame one (props_real_lines.load_pregame), plus play-calling."""
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    av = PL.load_availability(verbose=False)
    table = PL.build(frame, pg, info, av, inj, pr, games=games)
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan)
    pframe, _ = model.walk_forward(model.build_frame(blind, eng))
    pre = PL.build(pframe, pg, info, av, None, None, games=games, pregame=True)
    R = roles.load_all(games, verbose=False)
    return [add_play_calling(x, R["ctx_team"]) for x in (table, pre)]


def add_play_calling(table: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame:
    """Team pace and neutral pass rate as of each game (roles.features' method), the team's and the
    opponent's, on every row of the table."""
    tt = table.drop_duplicates(["game_id", "team"])[["game_id", "team", "opp", "season", "t", "played"]].merge(
        ct[["game_id", "team", "plays", "n_neutral", "pass_neutral"]], on=["game_id", "team"], how="left")
    pl = tt["played"].astype(bool)
    tt["plays_v"] = tt["plays"].where(pl)
    tt["pn"] = tt["pass_neutral"].where(pl)
    tt["nn"] = tt["n_neutral"].where(pl)
    tt["one_t"] = np.where(pl & tt["plays"].notna(), 1.0, np.nan)
    a = PL.asof(tt, "team", ["plays_v", "pn", "nn"], ["one_t"], PL.TEAM_HL)
    a["pace"] = a["plays_v_sum"] / a["one_t_sum"].replace(0, np.nan)
    a["neutral_pass"] = (a["pn_sum"] + 20 * 0.55) / (a["nn_sum"] + 20)
    f = a[["game_id", "team", "pace", "neutral_pass"]]
    o = f.rename(columns={"team": "opp", "pace": "opp_pace", "neutral_pass": "opp_np"})
    out = table.merge(f, on=["game_id", "team"], how="left").merge(o, on=["game_id", "opp"], how="left")
    for c in PLAY:
        out[c] = out[c].fillna(out[c].mean())
    return out


def walk(t: pd.DataFrame, target: str, feats: list[str]) -> pd.Series:
    """players.walk_forward's team regression, season by season (eight earlier seasons)."""
    pred = pd.Series(np.nan, index=t.index)
    for s in sorted(t.season.unique()):
        if s < 2013:
            continue
        tr = t[(t.season < s) & (t.season >= s - 8) & t.played.astype(bool) & t[target].notna()]
        if len(tr) < 200:
            continue
        m = PL.Lin(5.0).fit(tr[feats].fillna(0), tr[target].to_numpy(float))
        te = t.season == s
        pred[te] = m.predict(t.loc[te, feats].fillna(0))
    return pred


def paired_team(t, target, p0, p1):
    ok = t.played.astype(bool) & t[target].notna() & (t.season >= C.FIRST_TEST_SEASON) & p0.notna() & p1.notna()
    d = t[ok]
    res = {}
    for per, m in (("tune", d.season < C.HOLDOUT_FROM), ("hold", d.season >= C.HOLDOUT_FROM)):
        e0, e1 = (p0[d.index][m] - d.loc[m, target]).abs(), (p1[d.index][m] - d.loc[m, target]).abs()
        diff = e1 - e0
        g = diff.groupby(d.loc[m, "game_id"]).sum()
        res[per] = {"n": int(m.sum()), "mae": round(float(e1.mean()), 4), "change": round(float(diff.mean()), 4),
                    "se": round(float(g.std(ddof=1) * np.sqrt(len(g)) / len(diff)), 4)}
    return res


def head_to_head(t: pd.DataFrame) -> dict:
    tg = t.drop_duplicates(["game_id", "team"]).copy().reset_index(drop=True)
    res = {"attempts": {}, "carries": {}}
    p0 = walk(tg, "team_att", BASE)
    r = paired_team(tg, "team_att", p0, walk(tg, "team_att", BASE + PROE))
    r["keep"] = bool(r["tune"]["change"] < 0 and r["hold"]["change"] < 0)
    res["attempts"]["+ PROE"] = r
    print(f"attempts + PROE   tune {r['tune']['change']:+.4f} ± {r['tune']['se']:.4f}   hold {r['hold']['change']:+.4f} ± {r['hold']['se']:.4f}   {'keep' if r['keep'] else 'no'}", flush=True)
    preds = {k: walk(tg, "team_car", v) for k, v in CAR_VERSIONS.items()}
    for k in CAR_VERSIONS:
        if k == "current":
            continue
        r = paired_team(tg, "team_car", preds["current"], preds[k])
        r["beats_current"] = bool(r["tune"]["change"] < 0 and r["hold"]["change"] < 0)
        res["carries"][k] = r
        print(f"carries {k:26s} tune {r['tune']['mae']:.4f} ({r['tune']['change']:+.4f} ± {r['tune']['se']:.4f})   hold {r['hold']['mae']:.4f} ({r['hold']['change']:+.4f} ± {r['hold']['se']:.4f})", flush=True)
    cands = [k for k, r in res["carries"].items() if r["beats_current"]]
    unbeaten = [k for k in cands if not any(res["carries"][o]["tune"]["mae"] < res["carries"][k]["tune"]["mae"]
                                             and res["carries"][o]["hold"]["mae"] < res["carries"][k]["hold"]["mae"] for o in cands if o != k)]
    pick = min(unbeaten, key=lambda k: res["carries"][k]["hold"]["mae"]) if unbeaten else None
    res["carries_pick"] = pick
    res["carries_unbeaten"] = unbeaten
    print(f"carries: beat current {cands}; unbeaten {unbeaten}; pick: {pick}", flush=True)
    return res


def projections(t: pd.DataFrame, att_feats, car_feats) -> pd.DataFrame:
    PL.ATT_FEATS, PL.CAR_FEATS = list(att_feats), list(car_feats)
    return PL.walk_forward(t, first_test=C.FIRST_TEST_SEASON)


def proj_rmse(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    key = ["game_id", "player_id"]
    b = base[base.played.astype(bool) & (base.active == 1)].set_index(key)
    n = new.set_index(key).loc[b.index]
    out = {}
    for grp, stats in PROJ_STATS.items():
        m = b.group == grp
        if grp == "QB":
            m &= (b.att_share > 0.5) & (b.started == 1)
        for st in stats:
            row = {}
            for per, pm in (("tune", (b.season < C.HOLDOUT_FROM)), ("hold", (b.season >= C.HOLDOUT_FROM))):
                x = b[m & pm]
                y = x[st].to_numpy(float)
                e0 = (x[st + "_proj"].to_numpy(float) - y) ** 2
                e1 = (n.loc[x.index, st + "_proj"].to_numpy(float) - y) ** 2
                d = pd.Series(e1 - e0).groupby(x.index.get_level_values("game_id").to_numpy()).sum()
                r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
                # standard error of the RMSE change, from the clustered squared-error differences
                se = float(d.std(ddof=1) * np.sqrt(len(d)) / len(e0) / (2 * r0))
                row[per] = {"n": int(len(x)), "rmse_current": round(r0, 4), "rmse_new": round(r1, 4), "change": round(r1 - r0, 4), "se": round(se, 4)}
            out[f"{grp} {st}"] = row
            print(f"  {grp} {st:12s} tune {row['tune']['rmse_current']:.3f} -> {row['tune']['rmse_new']:.3f} ({row['tune']['change']:+.4f} ± {row['tune']['se']:.4f})"
                  f"   hold {row['hold']['rmse_current']:.3f} -> {row['hold']['rmse_new']:.3f} ({row['hold']['change']:+.4f} ± {row['hold']['se']:.4f})", flush=True)
    return out


def props_check(tag: str, proj: pd.DataFrame, pre: pd.DataFrame) -> dict:
    """experiments/props_real_lines.py on these projections, 2023-2025: Brier vs the books, pricing and close."""
    pp, qp = C.RAW / f"apply_vol_{tag}_proj.pkl", C.RAW / f"apply_vol_{tag}_pre.pkl"
    proj.merge(PL.season_average(proj), on=["game_id", "player_id"], how="left").to_pickle(pp)
    pre.merge(PL.season_average(pre), on=["game_id", "player_id"], how="left").to_pickle(qp)
    out = {}
    for season in (2023, 2024, 2025):
        subprocess.run([sys.executable, str(HERE / "props_real_lines.py"), "--season", str(season), "--proj", str(pp), "--pregame-proj", str(qp)],
                       check=True, capture_output=True)
        rep = json.loads((HERE / "out" / f"props_real_lines_{season}.json").read_text())
        out[season] = {snap: {st: {"n": v["n"], "brier_ours": v.get("brier_ours", v.get("ours")), "brier_market": v.get("brier_market", v.get("market"))}
                              for st, v in rep[snap].items()} for snap in ("pricing", "close")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-props", action="store_true")
    ap.add_argument("--rmse-only", action="store_true", help="only the head-to-head and the projection RMSE by piece")
    args = ap.parse_args()
    table, pre = tables()
    res = {"head_to_head": head_to_head(table)}
    pick = res["head_to_head"]["carries_pick"]
    att_new = BASE + PROE if res["head_to_head"]["attempts"]["+ PROE"]["keep"] else BASE
    car_new = CAR_VERSIONS[pick] if pick else BASE
    res["chosen"] = {"ATT_FEATS": att_new, "CAR_FEATS": car_new}
    print(f"\nchosen: attempts {att_new}\n        carries  {car_new}", flush=True)
    print("\nprojection RMSE, current chain -> chosen chain:", flush=True)
    p_cur = projections(table, BASE, BASE)
    p_new = projections(table, att_new, car_new)
    res["projection_rmse"] = proj_rmse(p_cur, p_new)
    # each piece on its own through the chain, so a stat that gets worse can be laid at one door
    res["projection_rmse_by_piece"] = {}
    for name, (af, cf) in (("attempts only", (att_new, BASE)), ("carries only", (BASE, car_new))):
        if (af, cf) != (att_new, car_new):
            print(f"\nprojection RMSE, {name}:", flush=True)
            res["projection_rmse_by_piece"][name] = proj_rmse(p_cur, projections(table, af, cf))
    if not args.skip_props and not args.rmse_only:
        print("\nprops check (Brier vs the books at the 2023-2025 lines) ...", flush=True)
        q_cur = projections(pre, BASE, BASE)
        q_new = projections(pre, att_new, car_new)
        res["props"] = {"current": props_check("cur", p_cur, q_cur), "new": props_check("new", p_new, q_new)}
        for season in (2023, 2024, 2025):
            for snap in ("pricing", "close"):
                a, b = res["props"]["current"][season][snap], res["props"]["new"][season][snap]
                line = "  ".join(f"{st} {a[st]['brier_ours']:.4f}->{b[st]['brier_ours']:.4f}" for st in a if st != "all")
                print(f"  {season} {snap:8s} all {a['all']['brier_ours']:.4f} -> {b['all']['brier_ours']:.4f} (books {a['all']['brier_market']:.4f})   {line}", flush=True)
    out = OUT_PIECES if args.rmse_only else OUT
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
