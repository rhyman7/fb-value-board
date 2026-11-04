"""Combined check for NFL game-model changes kept under the keep rule, on the board's own frame.

    python experiments/apply_check_nfl_game.py

Nov 4, 2026 (FEATURES.md "Re-judging past tests under the keep rule"). The re-judge scored each
change on its own; changes interact, so before a set goes on the board the set is checked together,
against the model `main` had before it (MAIN_MARGIN / MAIN_TOTAL below), on the frame the Update
board Action builds: the board's rating engine (features_ext.PRODUCTION_METRICS), injuries, wind,
and the role signals (game_roles.py) when the player table loads.

Measures, every one of which must be better for the set to be kept (the keep rule): the raw
model's average margin miss on the tuning seasons (2014-2018) and the holdout (2019 on), and the
blended projection's average miss on the same, all paired by game with clustered standard errors;
the walk-forward spread record at the closing line (market-adjusted sides with EV above 0; return
on the holdout) is reported alongside. Each piece is also shown on its own and with the others
removed, so the piece that spoils a set can be named.

Writes experiments/out/apply_check_nfl_game.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import backtest, config as C, data, extra, injuries, model, overrides, weather  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "apply_check_nfl_game.json"

# The margin and total models as they were on main before this set (Nov 4, 2026).
MAIN_MARGIN = ["d_pass", "d_rush", "d_sr", "d_pts", "d_mkt", "d_qb", "hfa", "rest_diff",
               "d_chg_ol", "d_depth1_out", "d_snap_drop"]
MAIN_TOTAL = list(model.TOTAL_FEATS)

PIECES = {   # name -> function(margin feats) -> margin feats
    "special teams (d_st)": lambda mf: mf + ["d_st"],
    "turnover-free swap (d_pnt/d_rnt for d_pass/d_rush)": lambda mf: [{"d_pass": "d_pnt", "d_rush": "d_rnt"}.get(x, x) for x in mf],
}
# What went on the board after the run (Nov 4, 2026): the swap alone. Special teams was a trade-off on
# this frame (the re-judge frame had no role signals) and made the set worse than the swap alone.
KEPT = ["turnover-free swap (d_pnt/d_rnt for d_pass/d_rush)"]
CHECK_METRICS = ("st",)   # rated metrics the pieces need beyond the board's PRODUCTION_METRICS


def board_frame(verbose: bool = True):
    """The frame the Update board Action fits the game model on (nflmodel/pipeline.py), offline."""
    say = print if verbose else (lambda *a, **k: None)
    games, tg, qb, stats = data.load_all(refresh=False, verbose=False, with_stats=True)
    ov = overrides.load()
    games = overrides.apply_starters(games, ov, verbose=False)
    team_extra, qb_extra = extra.load_extra(verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    metrics = tuple(PRODUCTION_METRICS) + tuple(m for m in CHECK_METRICS if m not in PRODUCTION_METRICS)
    engine = ExtRatingEngine(tg, qb, team_extra, qb_extra, extra_metrics=metrics, games=games)
    base = model.build_frame(games, engine, injuries=inj)
    frame, _ = model.walk_forward(base, margin_feats=MAIN_MARGIN, total_feats=MAIN_TOTAL)
    try:
        from nflmodel import game_roles, players, players_data, roles
        pg, pinfo = players_data.load_player_games(verbose=False)
        avail = players.load_availability(verbose=False)
        R = roles.load_all(games, verbose=False)
        rows = players.player_rows(players.team_games(frame, pg), pg, avail, pinfo)
        rows = overrides.apply_out(rows, ov) if ov is not None else rows
        base = game_roles.attach(base, game_roles.team_features(rows, R, avail))
        say("  role signals attached")
    except Exception as exc:
        say(f"  role signals skipped ({exc}); the check runs without them")
    return games, base


def run(frame, mf, tf):
    out, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    return out


def paired(base, new, col, actual):
    b = base[base.played & base.spread_line.notna() & (base.season >= C.FIRST_TEST_SEASON)].set_index("game_id")
    n = new.set_index("game_id").loc[b.index]
    e0, e1 = (b[col] - b[actual]).abs(), (n[col] - b[actual]).abs()
    res = {}
    for name, m in (("tune", b.season < C.HOLDOUT_FROM), ("hold", b.season >= C.HOLDOUT_FROM)):
        d = (e1 - e0)[m]
        res[name] = {"n": int(m.sum()), "base": round(float(e0[m].mean()), 4), "change": round(float(d.mean()), 4),
                     "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)}
    return res


def spread_record(games, out):
    graded, _ = backtest.grade_all(games, out)
    g = graded[(graded.source == "proj") & (graded.market == "Spread") & (graded.ev > 0)]
    r = {}
    for name, x in (("all", g), ("hold", g[g.season >= C.HOLDOUT_FROM])):
        rec = backtest._record(x)
        r[name] = {k: rec[k] for k in ("n", "w", "l", "p", "roi", "roi_se", "units")}
    return r


def judge(res):
    keys = [(k, p) for k in ("model_margin", "proj_margin") for p in ("tune", "hold")]
    better = all(res[k][p]["change"] < 0 for k, p in keys)
    worse = all(res[k][p]["change"] >= 0 for k, p in keys)
    return "KEEP (better on every measure)" if better else "NO (worse on every measure)" if worse else "TRADE-OFF"


def main():
    games, frame = board_frame()
    base = run(frame, MAIN_MARGIN, MAIN_TOTAL)
    results = {"main_margin": MAIN_MARGIN, "spread_record_main": spread_record(games, base), "variants": {}}
    print(f"main spread record {results['spread_record_main']}", flush=True)

    def score(name, mf):
        out = run(frame, mf, MAIN_TOTAL)
        r = {"margin_feats": mf,
             "model_margin": paired(base, out, "model_margin", "result"),
             "proj_margin": paired(base, out, "proj_margin", "result"),
             "spread_record": spread_record(games, out)}
        r["verdict"] = judge(r)
        results["variants"][name] = r
        mm, pm = r["model_margin"], r["proj_margin"]
        print(f"{name}\n   raw   tune {mm['tune']['change']:+.4f} ± {mm['tune']['se']:.4f}   hold {mm['hold']['change']:+.4f} ± {mm['hold']['se']:.4f}"
              f"\n   blend tune {pm['tune']['change']:+.4f} ± {pm['tune']['se']:.4f}   hold {pm['hold']['change']:+.4f} ± {pm['hold']['se']:.4f}"
              f"\n   spread EV>0 holdout roi {r['spread_record']['hold']['roi']} (n {r['spread_record']['hold']['n']})   {r['verdict']}", flush=True)
        return out

    names = list(PIECES)
    for n in names:                                    # each piece alone
        score(n, PIECES[n](list(MAIN_MARGIN)))
    mf_all = list(MAIN_MARGIN)
    for n in names:
        mf_all = PIECES[n](mf_all)
    score("ALL together", mf_all)
    if len(names) > 2:                                 # all but one, to name a spoiler
        for skip in names:
            mf = list(MAIN_MARGIN)
            for n in names:
                if n != skip:
                    mf = PIECES[n](mf)
            score(f"all but: {skip}", mf)
    kept = list(MAIN_MARGIN)
    for n in KEPT:
        kept = PIECES[n](kept)
    assert sorted(kept) == sorted(model.MARGIN_FEATS), (kept, model.MARGIN_FEATS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
