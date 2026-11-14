"""College-fitted prop distributions, scored at the 2025 DraftKings and FanDuel main lines.

    python experiments/cfb_propdist_test.py

The college legs on the Parlays page use the NFL prop distributions (data/prop_params.json:
negative-binomial counts, gamma yards per play, early-exit component, isotonic correction),
fitted on NFL projection misses. College projections miss by more (RB rushing yards RMSE about
40 against the NFL's 32), so the same widths make the college chances too confident:
experiments/cfb_props_gap_diag.py found the model's chance at a 2025 line carried about a fifth
of the confidence it claimed (best single shrink toward 50% was 0.2 on both yardage stats).

This test refits the same family on college data. Rule, written before the run (Nov 3, 2026):

  fit      PropModel widths per (stat, position group) on the college walk-forward player table,
           seasons 2017-2024 (every season fitted only on earlier ones, so nothing in it has seen
           2025), the mean held equal to the projection as the NFL fit does; the isotonic
           correction fitted on synthetic lines around the projection on the same seasons.
  score    Brier at the 2025 main lines (pushes out), model minus books, standard errors
           clustered by game, for the production-equivalent (widths + correction).
  keep     only if rushing yards AND receiving yards each improve on the current NFL-fitted
           chances by 2+ standard errors, and no stat (passing yards, passing TDs included) is
           worse by 2 standard errors.
  report   the widths-only version and a 2021-2024 fit as information, not for the decision.

Writes experiments/out/cfb_propdist_test.json.

Nov 4, 2026, the keep rule (FEATURES.md "Re-judging past tests under the keep rule"): the widths-only
version (near-miss #6) is applied when it is better than the current NFL-fitted chances on every stat
with 2025 lines (Brier, however small) and worse on none. `--save` writes it to data/cfb/prop_params.json
(the NFL parameters with the college widths swapped in; the NFL correction and correlations kept),
which nflmodel/parlays.py reads for the college legs; it refuses to write when any stat is worse.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.cfb_props_gap_diag as G  # noqa: E402
import experiments.cfb_props_real_lines as R  # noqa: E402
from nflmodel import propmarket as PM, props as P  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "cfb_propdist_test.json"
CACHE = Path("/tmp/cfb_pproj_2017.pkl")   # delete it after a model change; the table is rebuilt from the current code
FIT_FROM, FIT_TO = 2017, 2024
STATS = ["receptions", "rec_yds", "carries", "rush_yds", "pass_att", "completions", "pass_yds", "pass_td", "ints", "anytime_td"]


def projections_2017():
    """The college walk-forward player table from 2017 (cached)."""
    if CACHE.exists():
        return pd.read_pickle(CACHE), pd.read_pickle(CACHE.with_name("cfb_pinfo.pkl")), pd.read_pickle(CACHE.with_name("cfb_games.pkl"))
    from cfbmodel import data, model, players, players_data
    from cfbmodel.ratings import CFBRatingEngine
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, engine))
    pg, pinfo = players_data.load_player_games(verbose=False)
    pproj = players.walk_forward(players.build(games, frame, pg, pinfo, None, None), first_test=FIT_FROM)
    pproj.to_pickle(CACHE)
    pinfo.to_pickle(CACHE.with_name("cfb_pinfo.pkl"))
    games.to_pickle(CACHE.with_name("cfb_games.pkl"))
    return pproj, pinfo, games


def prep(pproj: pd.DataFrame) -> pd.DataFrame:
    d = pproj[pproj["played"].astype(bool)].copy()
    d["tgt_share"] = d.get("rec_share", 0.0)
    return d


def synthetic_preds(df: pd.DataFrame, m: P.PropModel) -> pd.DataFrame:
    out = []
    for stat in STATS:
        s = P.SPECS[stat]
        for g in s.groups:
            d = df[P.eligible(df, stat, g)].reset_index(drop=True)
            if d.empty or ((stat, g) not in m.counts and (stat, g) not in m.yards):
                continue
            L = P.test_lines(P.mean(d, stat), stat)
            raw, _, _ = m.probs(d, stat, g, L, calibrate=False)
            y = P.actual(d, stat)
            k = L.shape[1]
            out.append(pd.DataFrame({"stat": stat, "group": g, "line": L.ravel(), "p_over": raw.ravel(),
                                     "hit": (np.repeat(y, k) > L.ravel()).astype(float)}))
    r = pd.concat(out, ignore_index=True)
    return r[r["line"].notna()]


def fit_college(df: pd.DataFrame, since: int) -> tuple[P.PropModel, P.PropModel]:
    tr = df[(df.season >= since) & (df.season <= FIT_TO)]
    saved = P.PropModel.load()
    m = P.PropModel(saved.family if saved else {}).fit(tr, stats=STATS)
    widths_only = P.PropModel(m.family)
    widths_only.counts, widths_only.yards = dict(m.counts), dict(m.yards)
    m.fit_isotonic(synthetic_preds(tr, widths_only))
    return widths_only, m


def score(x: pd.DataFrame, m: P.PropModel, label: str) -> pd.DataFrame:
    z = R.model_chances(x, m)
    z = z[z["y"] != z["line"]].copy()
    z["hit"] = (z["y"] > z["line"]).astype(float)
    z[label] = z["our_over"] / (z["our_over"] + z["our_under"])
    return z[["game_id", "nname", "stat", "line", "fair", "hit", label]]


def clustered(d, game):
    cl = pd.Series(np.asarray(d)).groupby(np.asarray(game)).sum()
    return float(np.mean(d)), float(cl.std(ddof=1) * np.sqrt(len(cl)) / len(d))


CFB_PARAMS = Path(__file__).resolve().parent.parent / "data" / "cfb" / "prop_params.json"


def main():
    save = "--save" in sys.argv
    pproj, pinfo, games = projections_2017()
    df = prep(pproj)
    fits = {}
    for since in (FIT_FROM, 2021):
        w, m = fit_college(df, since)
        fits[f"widths_{since}"], fits[f"full_{since}"] = w, m
    # 2025 lines, the same frame as the Nov 1 test
    s = R.snapshots()
    gid = R.match_events(s, games)
    s = s.merge(gid, left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s["alt"] = s["market"].str.endswith("_alternate")
    rows = R.player_frame(pproj, pinfo)
    x = G.main_lines(s).merge(rows, on=["game_id", "nname"], how="inner")
    base = score(x, P.PropModel.load(), "current")
    keys = ["game_id", "nname", "stat", "line"]
    for k, m in fits.items():
        base = base.merge(score(x, m, k)[keys + [k]], on=keys, how="inner")
    res = {"fit_seasons": [FIT_FROM, FIT_TO], "props": int(len(base)), "games": int(base.game_id.nunique()),
           "params_2017": {f"{k[0]}|{k[1]}": v.__dict__ for k, v in fits["full_2017"].yards.items()} |
                          {f"{k[0]}|{k[1]}": v.__dict__ for k, v in fits["full_2017"].counts.items()},
           "params_current": {k: v for k, v in json.loads(P.PARAMS_PATH.read_text()).items() if k in ("counts", "yards")},
           "by_stat": {}}
    decision = {}
    for stat, g in base.groupby("stat"):
        r = {"props": int(len(g)), "brier_books": round(float(((g.fair - g.hit) ** 2).mean()), 5),
             "brier_current": round(float(((g.current - g.hit) ** 2).mean()), 5)}
        cur_vs, cur_se = clustered((g.current - g.hit) ** 2 - (g.fair - g.hit) ** 2, g.game_id)
        r["current_minus_books"], r["current_se"] = round(cur_vs, 5), round(cur_se, 5)
        for k in fits:
            d_cur, se_cur = clustered((g[k] - g.hit) ** 2 - (g.current - g.hit) ** 2, g.game_id)
            d_bk, se_bk = clustered((g[k] - g.hit) ** 2 - (g.fair - g.hit) ** 2, g.game_id)
            r[k] = {"brier": round(float(((g[k] - g.hit) ** 2).mean()), 5),
                    "minus_current": round(d_cur, 5), "se": round(se_cur, 5), "z": round(d_cur / se_cur, 2) if se_cur else None,
                    "minus_books": round(d_bk, 5), "books_se": round(se_bk, 5),
                    "spread": round(float((g[k] - 0.5).abs().mean()), 4)}
        r["spread_current"] = round(float((g.current - 0.5).abs().mean()), 4)
        res["by_stat"][stat] = r
        decision[stat] = r["full_2017"]
    better = all(decision[s]["minus_current"] <= -2 * decision[s]["se"] for s in ("rush_yds", "rec_yds") if s in decision)
    worse = any(decision[s]["minus_current"] >= 2 * decision[s]["se"] for s in decision)
    res["rule"] = {"rush_and_rec_2se_better": bool(better), "any_stat_2se_worse": bool(worse), "keep": bool(better and not worse)}
    # the keep rule (Nov 4, 2026): widths only, better on every stat with lines
    w = {s: res["by_stat"][s]["widths_2017"]["minus_current"] for s in res["by_stat"]}
    res["keep_rule_widths_only"] = {"minus_current": w, "better_on_all": bool(all(v < 0 for v in w.values()))}
    if save:
        if not res["keep_rule_widths_only"]["better_on_all"]:
            print("widths-only is worse on a stat; not saved:", w)
        else:
            saved = P.PropModel.load()
            out = P.PropModel(saved.family)
            out.counts, out.yards, out.iso = dict(fits["widths_2017"].counts), dict(fits["widths_2017"].yards), dict(saved.iso)
            out.meta = dict(saved.meta) | {"college_widths": {"fit_seasons": [FIT_FROM, FIT_TO], "written": "2026-11-04",
                                                              "note": "widths fitted on the college walk-forward; correction and the rest from data/prop_params.json"}}
            prev = P.PropModel.load(CFB_PARAMS) if CFB_PARAMS.exists() else None
            if prev is not None and prev.shift:      # the college QB rushing shift (Nov 13, 2026) is kept across refits
                out.shift = dict(prev.shift)
                out.meta["chance_shift"] = prev.meta.get("chance_shift")
            out.save(CFB_PARAMS)
            print(f"wrote {CFB_PARAMS}")
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for stat, r in res["by_stat"].items():
        f = r["full_2017"]
        print(f"{stat:9s} n={r['props']:4d} books {r['brier_books']:.5f} current {r['brier_current']:.5f} (vs books {r['current_minus_books']:+.5f}) "
              f"college {f['brier']:.5f} (vs current {f['minus_current']:+.5f} ± {f['se']:.5f}, z {f['z']}; vs books {f['minus_books']:+.5f} ± {f['books_se']:.5f}) "
              f"spread {r['spread_current']:.3f} -> {f['spread']:.3f}")
    print("rule:", res["rule"])


if __name__ == "__main__":
    main()
