"""Does capping long plays make rushing and receiving projections more accurate?

    python experiments/players_capped_test.py

Breakaway runs and catches repeat little from game to game, so a player's yards per carry
and per target may be measured better with each play capped (yards beyond the cap come back
at his position's average rate). The outcomes predicted are never capped: a projection is
scored against the real box score, long plays included.

Variants: no cap (the board today), caps at 15, 20 and 30 yards on efficiency, and the same
caps on the recent per-game average the calibration blends in. Chosen on 2013-2018 (the
player model's tuning seasons); 2019 on is reported once.

Scored on the same player-games for every variant (picked on the uncapped projection and
what was known before the game): running backs and quarterbacks for rushing yards, wide
receivers, tight ends and backs for receiving yards. Average miss, root-mean-square miss,
and the Brier score of P(over) at half-point lines either side of the uncapped projection,
with distribution widths fitted on the tuning seasons for each variant.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model, players as PL, players_data as PD, propdist as D, props as P  # noqa: E402
from nflmodel.data import _download  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
CAPS = (15, 20, 30)
OUT = Path(__file__).resolve().parent / "out"


def capped(season: int) -> pd.DataFrame:
    """Per player-game rushing and receiving yards with each play capped (cached in data/raw)."""
    path = C.RAW / f"capped_{season}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"player_id": str})
    raw = C.RAW / f"pbp_cap_{season}.parquet"
    _download(C.PBP_URL.format(season=season), raw)
    cols = ["game_id", "posteam", "play_type", "pass_attempt", "rush_attempt", "sack", "qb_kneel", "qb_spike", "two_point_attempt",
            "receiver_player_id", "rusher_player_id", "receiving_yards", "rushing_yards"]
    p = pd.read_parquet(raw, columns=cols)
    raw.unlink()
    p = p[(p["two_point_attempt"].fillna(0) == 0) & p["posteam"].notna()]
    ru = p[(p["rush_attempt"] == 1) & (p["qb_kneel"].fillna(0) == 0) & p["rusher_player_id"].notna()]
    rc = p[(p["pass_attempt"] == 1) & (p["sack"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0) & p["receiver_player_id"].notna()]
    out = []
    for c in CAPS:
        a = ru.assign(v=ru["rushing_yards"].fillna(0).clip(upper=c)).groupby(["game_id", "rusher_player_id"])["v"].sum().rename(f"rush_yds_c{c}")
        b = rc.assign(v=rc["receiving_yards"].fillna(0).clip(upper=c)).groupby(["game_id", "receiver_player_id"])["v"].sum().rename(f"rec_yds_c{c}")
        a.index.names = b.index.names = ["game_id", "player_id"]
        out += [a, b]
    d = pd.concat(out, axis=1).fillna(0).reset_index()
    d.to_csv(path, index=False)
    return d


def main():
    print("Game model ...", flush=True)
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    print("Capped box scores ...", flush=True)
    cp = pd.concat([capped(s) for s in range(PD.FIRST, int(pg["season"].max()) + 1)], ignore_index=True)
    pg = pg.merge(cp, on=["game_id", "player_id"], how="left")
    for c in cp.columns[2:]:
        pg[c] = pg[c].fillna(0)
    av = PL.load_availability(verbose=False)

    variants = [("no cap", None, False)] + [(f"cap {c}", c, False) for c in CAPS] + [(f"cap {c} + recent", c, True) for c in CAPS]
    proj = {}
    for name, cap, rec in variants:
        print(f"  {name} ...", flush=True)
        PL.EFF_CAP, PL.CAP_RECENT = cap, rec
        table = PL.build(frame, pg, info, av, inj, pr, games=games)
        proj[name] = PL.walk_forward(table)
    PL.EFF_CAP, PL.CAP_RECENT = None, False

    base = proj["no cap"]
    rows = []
    for stat, groups in (("rush_yds", ("RB", "QB")), ("rec_yds", ("WR", "TE", "RB"))):
        for g in groups:
            sel = P.eligible(base, stat, g)
            keys = base.loc[sel, ["game_id", "player_id"]]
            mu0 = base.loc[sel, stat + "_proj"].to_numpy()
            L = (np.floor(mu0 / 5) * 5 + 0.5)[:, None] + np.array([-10, 0, 10])[None, :]
            for name, _, _ in variants:
                x = keys.merge(proj[name], on=["game_id", "player_id"], how="left")
                tune = x["season"] < C.HOLDOUT_FROM
                m = P.PropModel({}).fit(x[tune], stats=[P.SPECS[stat].vol, stat])
                for period, mask in (("2013-18", tune), ("2019-26", ~tune)):
                    d = x[mask].reset_index(drop=True)
                    Lp = L[mask.to_numpy()]
                    y = d[stat].to_numpy()
                    over, _, _ = m.probs(d, stat, g, Lp, calibrate=False)
                    ok = Lp >= 0.5
                    hit = (y[:, None] > Lp)[ok].astype(float)
                    rows.append({"stat": stat, "group": g, "variant": name, "period": period, "n": len(d),
                                 "mae": round(float(np.mean(np.abs(d[stat + "_proj"] - y))), 3),
                                 "rmse": round(float(np.sqrt(np.mean((d[stat + "_proj"] - y) ** 2))), 3),
                                 "bias": round(float(np.mean(d[stat + "_proj"] - y)), 3),
                                 "brier": round(D.brier(over[ok], hit), 5)})
    r = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    r.to_csv(OUT / "players_capped_test.csv", index=False)
    pd.set_option("display.width", 200)
    for period in ("2013-18", "2019-26"):
        print(f"\n== {period}")
        print(r[r.period == period].drop(columns="period").to_string(index=False))


if __name__ == "__main__":
    main()
