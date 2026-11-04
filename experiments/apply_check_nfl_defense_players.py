"""Check before applying the shadow test's player version: defensive rates in the player projections.

    python experiments/apply_check_nfl_defense_players.py [--volume current|chosen] [--skip-props]

Nov 4, 2026. Ryan's call under the keep rule (Project doc claude/keep-rule-rejudge-2026-11-04.md):
the player part of the shadow test (nflmodel/shadow.py PLAYER, pre-registered Oct 29, 2026) goes
live now instead of waiting for its 2027 judgment. The inputs (nflmodel/defense.py): the opponent's
missed-tackle rate and yards per target allowed as calibration terms for receiving yards, its
pressure rate for interceptions, each against the league average of earlier seasons; 0 (no
information) before 2019.

Rule, written before the run: the version is applied if, on today's chain, it is better on every
measure and worse on none:
  * RMSE of receiving yards for WR, TE and RB and of interceptions for starting quarterbacks, on
    players who were active and played, tuning 2020-2022 and holdout 2023 on (the shadow test's
    periods; before 2019 the inputs are 0 and the projections identical). Paired, standard errors
    clustered by game.
  * Brier against the books at the real 2023-2025 DraftKings/FanDuel lines for receiving yards and
    receptions (experiments/props_real_lines.py, pricing and closing snapshots). Interceptions have
    no logged lines.
Anything worse is a trade-off and goes to Ryan.

--volume chosen runs on the team-volume features experiments/apply_check_nfl_volume.py picked
(its JSON's "chosen"); current runs on players.VOL_FEATS for both regressions.
Writes experiments/out/apply_check_nfl_defense_players.json.
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
import apply_check_nfl_volume as V  # noqa: E402
from nflmodel import config as C, data, defense, players as PL, shadow  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "apply_check_nfl_defense_players.json"
TUNE_FROM, HOLD_FROM = 2020, 2023
STATS = {"WR": ["rec_yds"], "TE": ["rec_yds"], "RB": ["rec_yds"], "QB": ["ints"]}


def with_defense(table: pd.DataFrame, rates: pd.DataFrame, att_feats, car_feats) -> pd.DataFrame:
    saved = {k: list(v) for k, v in PL.CAL_EXTRA.items()}
    try:
        PL.CAL_EXTRA.update({k: v + shadow.PLAYER.get(k, []) for k, v in saved.items()})
        return V.projections(defense.attach(table, rates), att_feats, car_feats)
    finally:
        PL.CAL_EXTRA.clear()
        PL.CAL_EXTRA.update(saved)


def rmse(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    key = ["game_id", "player_id"]
    b = base[base.played.astype(bool) & (base.active == 1) & (base.season >= TUNE_FROM)].set_index(key)
    n = new.set_index(key).loc[b.index]
    out = {}
    for grp, stats in STATS.items():
        m = b.group == grp
        if grp == "QB":
            m &= (b.att_share > 0.5) & (b.started == 1)
        for st in stats:
            row = {}
            for per, pm in (("tune", b.season < HOLD_FROM), ("hold", b.season >= HOLD_FROM)):
                x = b[m & pm]
                y = x[st].to_numpy(float)
                e0 = (x[st + "_proj"].to_numpy(float) - y) ** 2
                e1 = (n.loc[x.index, st + "_proj"].to_numpy(float) - y) ** 2
                d = pd.Series(e1 - e0).groupby(x.index.get_level_values("game_id").to_numpy()).sum()
                r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
                se = float(d.std(ddof=1) * np.sqrt(len(d)) / len(e0) / (2 * r0))
                row[per] = {"n": int(len(x)), "rmse_current": round(r0, 4), "rmse_new": round(r1, 4), "change": round(r1 - r0, 4), "se": round(se, 4)}
            out[f"{grp} {st}"] = row
            print(f"  {grp} {st:8s} tune {row['tune']['rmse_current']:.4f} -> {row['tune']['rmse_new']:.4f} ({row['tune']['change']:+.4f} ± {row['tune']['se']:.4f})"
                  f"   hold {row['hold']['rmse_current']:.4f} -> {row['hold']['rmse_new']:.4f} ({row['hold']['change']:+.4f} ± {row['hold']['se']:.4f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", choices=["current", "chosen"], default="chosen")
    ap.add_argument("--skip-props", action="store_true")
    args = ap.parse_args()
    if args.volume == "chosen":
        ch = json.loads(V.OUT.read_text())["chosen"]
        att, car = ch["ATT_FEATS"], ch["CAR_FEATS"]
    else:
        att, car = V.BASE, V.BASE
    print(f"volume features: attempts {att}\n                 carries  {car}", flush=True)
    table, pre = V.tables()
    games, tg, _ = data.load_all(refresh=False, verbose=False)
    rates = defense.team_rates(games, tg, defense.load(verbose=False))
    p_cur, p_new = V.projections(table, att, car), with_defense(table, rates, att, car)
    res = {"volume": {"ATT_FEATS": att, "CAR_FEATS": car}, "cal_extra_added": shadow.PLAYER}
    print("projection RMSE, current chain -> with the defensive rates:", flush=True)
    res["rmse"] = rmse(p_cur, p_new)
    moved = p_new[(p_new.season == p_new.season.max()) & ~p_new.played.astype(bool)].set_index(["game_id", "player_id"])
    cur = p_cur.set_index(["game_id", "player_id"]).reindex(moved.index)
    d = (moved["rec_yds_proj"] - cur["rec_yds_proj"]).abs()
    d = d[moved["targets_proj"] >= 1.5]
    res["upcoming_move"] = {"rec_yds median": round(float(d.median()), 3), "95th": round(float(d.quantile(0.95)), 3), "max": round(float(d.max()), 3), "n": int(len(d))}
    print(f"  upcoming receiving projections move: median {d.median():.2f}, 95th {d.quantile(0.95):.2f}, max {d.max():.2f} yards ({len(d)} players)", flush=True)
    if not args.skip_props:
        print("\nprops check (Brier vs the books at the 2023-2025 lines) ...", flush=True)
        q_cur, q_new = V.projections(pre, att, car), with_defense(pre, rates, att, car)
        res["props"] = {"current": V.props_check("defcur", p_cur, q_cur), "new": V.props_check("defnew", p_new, q_new)}
        for season in (2023, 2024, 2025):
            for snap in ("pricing", "close"):
                a, b = res["props"]["current"][season][snap], res["props"]["new"][season][snap]
                line = "  ".join(f"{st} {a[st]['brier_ours']:.4f}->{b[st]['brier_ours']:.4f}" for st in a if st in ("rec_yds", "receptions"))
                print(f"  {season} {snap:8s} all {a['all']['brier_ours']:.4f} -> {b['all']['brier_ours']:.4f} (books {a['all']['brier_market']:.4f})   {line}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
