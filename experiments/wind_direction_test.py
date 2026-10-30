"""Does the wind's direction (and its gusts) tell the totals model anything the speed does not?

    python experiments/wind_direction_test.py              # data/cache/wind_history.csv (fetch_wind_history.py)
    python experiments/wind_direction_test.py --no-model   # parts (a) and (c) only: no ratings needed
    python experiments/wind_direction_test.py --fixture    # made-up archive numbers, to check the script runs

Given the archive wind at kickoff for every outdoor game since 2014 (speed, gusts, direction,
and the along-field and crosswind parts from nflmodel/stadiums.py), this script:

  (a) compares the wind recorded in the games file with the archive's speed (correlation,
      mean difference), so we know how much to trust each;
  (b) adds gusts, the along-field part and the crosswind part to the TOTAL model one at a
      time and scores them walk-forward, with FEATURES.md's rule: tuning seasons 2014-2018,
      holdout 2019 on; a feature is kept only if it helps on tuning and is not worse on the
      holdout (the holdout is the check, not the selector);
  (c) repeats the wind-under table (FEATURES.md "The wind under": unders at the closing
      total, by band) with the along-field part and then the crosswind part in place of the
      recorded speed. If the along-field bands carry the effect and the crosswind bands
      don't, direction matters; if both look like the speed table, it doesn't.

Games the archive lacks keep the model's usual wind handling (blank = league median).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import backtest, config as C, data, model, stadiums  # noqa: E402
from nflmodel.features_ext import add_context  # noqa: E402

warnings.filterwarnings("ignore")
CACHE = C.CACHE / "wind_history.csv"
TUNE_FROM = 2014
CANDIDATES = [("gust", "gusts (mph)"), ("along", "along-field wind (mph)"), ("cross", "crosswind (mph)"),
              ("arch_wind", "archive speed instead of recorded")]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_archive(path) -> pd.DataFrame:
    """The archive cache with the columns this script uses, numeric, keyed by game_id."""
    a = pd.read_csv(path)
    need = ["game_id", "wind_mph", "gust_mph", "dir_deg", "along_mph", "cross_mph"]
    for c in need:
        if c not in a.columns:
            raise SystemExit(f"{path} lacks column {c}: run experiments/fetch_wind_history.py")
    a = a.drop_duplicates("game_id", keep="last")
    out = pd.DataFrame({"game_id": a.game_id, "arch_wind": pd.to_numeric(a.wind_mph, errors="coerce"),
                        "gust": pd.to_numeric(a.gust_mph, errors="coerce"), "dir": pd.to_numeric(a.dir_deg, errors="coerce"),
                        "along": pd.to_numeric(a.along_mph, errors="coerce"), "cross": pd.to_numeric(a.cross_mph, errors="coerce")})
    if "bearing_conf" in a.columns:
        out["bearing_conf"] = a.bearing_conf.values
    return out


def fixture(games: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """A made-up archive: speed near the recorded wind, random direction, gusts above speed.
    Only for checking the script runs; the numbers mean nothing."""
    rng = np.random.default_rng(seed)
    g = games[(games.season >= TUNE_FROM) & games.played & (games.neutral == 0) & (games.indoor == 0)]
    rec = pd.to_numeric(g.wind, errors="coerce").fillna(8.0).to_numpy()
    speed = np.clip(rec + rng.normal(0, 3, len(g)), 0, None)
    direction = rng.uniform(0, 360, len(g))
    rows = []
    for gm, w, d in zip(g.itertuples(index=False), speed, direction):
        site = stadiums.site(gm.home_team, gm.stadium if isinstance(gm.stadium, str) else None)
        along = cross = conf = None
        if site:
            along, cross = stadiums.components(w, d, site[2])
            conf = site[3]
        rows.append({"game_id": gm.game_id, "wind_mph": round(w, 1), "gust_mph": round(w * rng.uniform(1.2, 1.8), 1),
                     "dir_deg": int(d), "along_mph": None if along is None else round(along, 1),
                     "cross_mph": None if cross is None else round(cross, 1), "bearing_conf": conf})
    return pd.DataFrame(rows)


def attach(games: pd.DataFrame, arch: pd.DataFrame) -> pd.DataFrame:
    g = games.merge(arch, on="game_id", how="left")
    outdoor = g.indoor == 0
    for c in ("arch_wind", "gust", "along", "cross"):
        g[c] = g[c].where(outdoor)                # indoors nothing blows
    return g


# --------------------------------------------------------------------------
# (a) recorded wind against the archive
# --------------------------------------------------------------------------

def compare(g: pd.DataFrame) -> None:
    f = g[(g.indoor == 0) & g.played & (g.season >= TUNE_FROM)]
    both = f[f.wind.notna() & f.arch_wind.notna()]
    print(f"(a) recorded wind vs archive speed, outdoor games since {TUNE_FROM}")
    print(f"    outdoor games {len(f)}, with recorded wind {int(f.wind.notna().sum())}, in the archive {int(f.arch_wind.notna().sum())}, both {len(both)}")
    if len(both) < 3:
        print("    too few games with both to compare")
        return
    d = both.arch_wind - both.wind
    print(f"    correlation {np.corrcoef(both.wind, both.arch_wind)[0, 1]:.3f}, archive minus recorded {d.mean():+.2f} mph "
          f"(s.d. {d.std():.2f}), average absolute gap {d.abs().mean():.2f}")
    band = lambda w: pd.cut(w, [-1, 10, 15, 20, 999], labels=["0-9", "10-14", "15-19", "20+"])
    ct = pd.crosstab(band(both.wind), band(both.arch_wind))
    ct.index.name, ct.columns.name = "recorded", "archive"
    print("    games by band, recorded (rows) against archive (columns):")
    print("    " + ct.to_string().replace("\n", "\n    "))
    for lbl, x in (("2014-2018", both[both.season < C.HOLDOUT_FROM]), (f"{C.HOLDOUT_FROM} on", both[both.season >= C.HOLDOUT_FROM])):
        if len(x) > 2:
            print(f"    {lbl}: n {len(x)}, correlation {np.corrcoef(x.wind, x.arch_wind)[0, 1]:.3f}, archive minus recorded {(x.arch_wind - x.wind).mean():+.2f}")
    if "bearing_conf" in g:
        n = g[g.along.notna()].bearing_conf.value_counts()
        print(f"    games with a field bearing: {', '.join(f'{k} {v}' for k, v in n.items())}")


# --------------------------------------------------------------------------
# (b) the totals model
# --------------------------------------------------------------------------

def score(frame: pd.DataFrame, tf: list[str]) -> dict:
    out, _ = model.walk_forward(frame, total_feats=tf)
    o = out[out.played & out.total_line.notna() & (out.season >= TUNE_FROM)]
    return {name: float(np.mean(np.abs(x.model_total - x.total)))
            for name, x in (("tune", o[o.season < C.HOLDOUT_FROM]), ("hold", o[o.season >= C.HOLDOUT_FROM]))}


def feature_test(games: pd.DataFrame, arch: pd.DataFrame) -> None:
    from nflmodel import extra
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    print(f"(b) the TOTAL model with each candidate added, walk-forward; tuning {TUNE_FROM}-{C.HOLDOUT_FROM - 1}, holdout {C.HOLDOUT_FROM} on")
    games_all, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games_all)
    frame = model.build_frame(games_all, engine)
    frame = attach(frame, arch)
    outdoor = frame.indoor == 0
    for c in ("arch_wind", "gust", "along", "cross"):
        known = outdoor & frame[c].notna() & frame.played
        fill = frame.loc[known, c].median() if known.any() else 0.0
        frame[c] = frame[c].where(outdoor, 0.0).fillna(fill)     # blank = a typical day, as the model treats wind
    base = score(frame, list(model.TOTAL_FEATS))
    print(f"    base total miss: tuning {base['tune']:.3f}, holdout {base['hold']:.3f}")
    print("    change in average miss when added (negative = better); kept if it helps on tuning and is not worse on the holdout")
    for col, label in CANDIDATES:
        tf = [f for f in model.TOTAL_FEATS if not (col == "arch_wind" and f == "wind")] + [col]
        r = score(frame, tf)
        dt, dh = r["tune"] - base["tune"], r["hold"] - base["hold"]
        keep = "KEEP" if dt < 0 and dh <= 0 else "no"
        print(f"    {label:36s} tuning {dt:+.4f}  holdout {dh:+.4f}  {keep}")
    r = score(frame, list(model.TOTAL_FEATS) + ["along", "cross"])
    dt, dh = r["tune"] - base["tune"], r["hold"] - base["hold"]
    print(f"    {'along-field and crosswind together':36s} tuning {dt:+.4f}  holdout {dh:+.4f}  {'KEEP' if dt < 0 and dh <= 0 else 'no'}")


# --------------------------------------------------------------------------
# (c) the wind under, by component
# --------------------------------------------------------------------------

def _rec(r: dict) -> str:
    if not r["n"]:
        return "no games"
    pct = "n/a" if r["win_pct"] is None else f"{100 * r['win_pct']:.1f}%"
    roi = "n/a" if r["roi"] is None else f"{100 * r['roi']:+.1f}%"
    se = "" if r["roi_se"] is None else f" ± {100 * r['roi_se']:.1f}%"
    return f"{r['w']}-{r['l']}-{r['p']}  {pct:>6}  return {roi}{se}  close minus final {r['close_minus_actual']:+.1f}"


def wind_under(games: pd.DataFrame) -> None:
    print("(c) unders at the closing total in outdoor games, by band (FEATURES.md \"The wind under\")")
    g = games[games.season >= TUNE_FROM].copy()
    for col, label in (("wind", "recorded wind (the published table, 2014 on)"), ("arch_wind", "archive speed"),
                       ("along", "along-field part"), ("cross", "crosswind part"), ("gust", "gusts")):
        f = g.assign(wind=pd.to_numeric(g[col], errors="coerce"))
        f["wind_known"] = (f.wind.notna() & (f.indoor == 0)).astype(int)
        f["wind"] = f.wind.fillna(0.0)
        rep = backtest.wind_report(f)
        print(f"    by {label}:")
        for t in rep["tiers"]:
            print(f"      {t['label']:10s} {_rec(t)}")
        print(f"      10-19 mph by era: " + "; ".join(f"{e['label']} {_rec(e)}" for e in rep["eras"] if e["n"]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=str(CACHE), help="archive cache from fetch_wind_history.py")
    ap.add_argument("--fixture", action="store_true", help="made-up archive numbers, to check the script runs")
    ap.add_argument("--no-model", action="store_true", help="skip (b), the walk-forward model test")
    a = ap.parse_args(argv)
    games = data.load_games(refresh=False)
    if a.fixture:
        print("FIXTURE: made-up archive numbers; the results below mean nothing")
        arch = load_archive_frame(fixture(games))
    else:
        if not Path(a.cache).exists():
            raise SystemExit(f"{a.cache} not found: run experiments/fetch_wind_history.py (or the Wind history Action) first")
        arch = load_archive(a.cache)
    g = attach(games, arch)
    compare(g)
    print()
    wind_under(g)
    print()
    if not a.no_model:
        feature_test(games, arch)


def load_archive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """load_archive for a frame already in memory (the fixture)."""
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return load_archive(buf)


if __name__ == "__main__":
    main()
