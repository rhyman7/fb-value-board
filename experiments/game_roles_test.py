"""Player-level role data in the game model: does it improve spreads and totals?

    python experiments/game_roles_test.py [--proj path]

Candidates, each added to the production margin or total model on its own (walk-forward,
same as feature_test.py; tuning 2011-2018 out of sample, holdout 2019 on; a candidate
earns a place only if it helps on the tuning seasons, the holdout is the check):

  s_neutral_pass / d_neutral_pass   each team's pass rate on early downs in neutral game
                                    situations, from its earlier games (roles.py)
  d_miss_use / s_miss_use           the usage missing for this game: the target and carry
                                    shares (recent games) of skill players who are not
                                    active, summed per team
  d_depth1_out / s_depth1_out       depth-chart starters at QB, RB, WR and TE not active
  d_snap_drop / s_snap_drop         how far the team's regular skill players' snap share
                                    fell in their last two games against their average

Also scored: change in the average miss on the margin and the total, and against the
closing line (how often the raw model's side covered), on the same games.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model, players as PL, players_data as PD, roles  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")


def team_features(proj: pd.DataFrame, R: dict) -> pd.DataFrame:
    """One row per team-game with the candidate team-level features."""
    f = roles.features(proj, R)
    x = proj[["game_id", "team", "player_id", "group", "active", "tgt_share_raw", "car_share_raw"]].merge(f, on=["game_id", "player_id"], how="left")
    x["use"] = x["tgt_share_raw"].fillna(0) + x["car_share_raw"].fillna(0)
    out = x["active"] == 0
    regular = (x["snap_avg"].fillna(0) >= 0.4) & (x["active"] == 1)
    g = x.groupby(["game_id", "team"])
    t = pd.DataFrame({
        "miss_use": x["use"].where(out, 0).groupby([x["game_id"], x["team"]]).sum(),
        "depth1_out": ((x["depth"] == 1) & out).groupby([x["game_id"], x["team"]]).sum(),
        "snap_drop": (-(x["snap_trend"].fillna(0)).clip(upper=0)).where(regular, 0).groupby([x["game_id"], x["team"]]).sum(),
        "neutral_pass": g["neutral_pass"].first(),
    }).reset_index()
    return t


def attach(frame: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    h = t.rename(columns={c: "h_" + c for c in t.columns if c not in ("game_id", "team")})
    a = t.rename(columns={c: "a_" + c for c in t.columns if c not in ("game_id", "team")})
    f = f.merge(h, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left").drop(columns="team")
    f = f.merge(a, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left").drop(columns="team")
    lg = t["neutral_pass"].mean()
    for c in ("miss_use", "depth1_out", "snap_drop", "neutral_pass"):
        hc, ac = f["h_" + c], f["a_" + c]
        fill = lg if c == "neutral_pass" else 0.0
        hc, ac = hc.fillna(fill), ac.fillna(fill)
        f["d_" + c] = hc - ac
        f["s_" + c] = hc + ac - (2 * lg if c == "neutral_pass" else 0)
    return f


def score(frame, mf, tf):
    out, _ = model.walk_forward(frame, margin_feats=mf, total_feats=tf)
    o = out[out.played & out.spread_line.notna() & (out.season >= 2014)]
    res = {}
    for name, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        side = np.sign(x.model_margin - x.spread_line)
        cover = np.sign(x.result - x.spread_line)
        res[name] = {"margin": np.mean(np.abs(x.model_margin - x.result)), "total": np.mean(np.abs(x.model_total - x.total)),
                     "ats": float(np.mean((side == cover)[cover != 0])), "n": len(x)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=None)
    args = ap.parse_args()
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame = model.build_frame(games, eng, injuries=inj)
    if args.proj and Path(args.proj).exists():
        proj = pd.read_pickle(args.proj)
    else:
        f0, _ = model.walk_forward(frame)
        pg, info = PD.load_player_games(verbose=False)
        av = PL.load_availability(verbose=False)
        proj = PL.build(f0, pg, info, av, inj, injuries.load_pass_rush(games, verbose=False), games=games)
    R = roles.load_all(games, verbose=False)
    frame = attach(frame, team_features(proj, R))
    mf, tf = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    base = score(frame, mf, tf)
    print(f"base (2014 on)  margin tune {base['tune']['margin']:.3f} hold {base['hold']['margin']:.3f} | total tune {base['tune']['total']:.3f} "
          f"hold {base['hold']['total']:.3f} | raw side vs close tune {base['tune']['ats']:.3f} hold {base['hold']['ats']:.3f}")
    print("change when added (miss: negative = better; side vs close: positive = better)")
    rows = []
    for c in ("d_miss_use", "d_depth1_out", "d_snap_drop", "d_neutral_pass"):
        r = score(frame, mf + [c], tf)
        rows.append(("margin", c, r["tune"]["margin"] - base["tune"]["margin"], r["hold"]["margin"] - base["hold"]["margin"],
                     r["tune"]["ats"] - base["tune"]["ats"], r["hold"]["ats"] - base["hold"]["ats"]))
        print(f"  margin {c:15s} tune {rows[-1][2]:+.4f} hold {rows[-1][3]:+.4f} | side tune {rows[-1][4]:+.4f} hold {rows[-1][5]:+.4f}", flush=True)
    for c in ("s_neutral_pass", "s_miss_use", "s_depth1_out", "s_snap_drop"):
        r = score(frame, mf, tf + [c])
        rows.append(("total", c, r["tune"]["total"] - base["tune"]["total"], r["hold"]["total"] - base["hold"]["total"], np.nan, np.nan))
        print(f"  total  {c:15s} tune {rows[-1][2]:+.4f} hold {rows[-1][3]:+.4f}", flush=True)
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["model", "feature", "miss_tune", "miss_hold", "side_tune", "side_hold"]).to_csv(out / "game_roles_test.csv", index=False)


if __name__ == "__main__":
    main()
