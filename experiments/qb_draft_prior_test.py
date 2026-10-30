"""A backup quarterback's prior from his draft capital, tested walk-forward (Oct 2026).

    python experiments/qb_draft_prior_test.py

config.QB_REPLACEMENT gives every unknown quarterback the same prior (-0.13 EPA per dropback
against the league). Here the prior depends on where he was drafted (nflverse draft picks:
round 1, rounds 2-3, later or undrafted). Two versions:

  fitted    for each season S, the prior of a bucket is the EPA per dropback, against the
            league, of earlier quarterbacks in that bucket over their first 300 dropbacks
            (careers starting 2007 or later, so the start is in the data; drafted 2006 on),
            shrunk toward -0.13 with the weight of 5 quarterbacks: a walk-forward estimate
  fixed     -0.13 plus a set offset per bucket (round 1 higher, late or undrafted lower),
            the same every season, a grid of offsets

Scored as experiments/qb_test.py: the raw model's average miss on games where either team
started a different quarterback from its previous game (QB-change games) and on all games,
tuning 2011-2018 and holdout 2019 on. A version earns a place only if it lowers the miss on
the tuning seasons' QB-change games and holds on the holdout.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.data import _download  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402
from experiments.qb_test import qb_change_games, score  # noqa: E402

warnings.filterwarnings("ignore")
DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
BUCKETS = ("round 1", "rounds 2-3", "later or undrafted")
FIRST_DROPBACKS = 300
SHRINK_QBS = 5.0


def load_draft() -> pd.DataFrame:
    path = C.RAW / "draft_picks.csv"
    if not path.exists():
        _download(DRAFT_URL, path)
    d = pd.read_csv(path, usecols=["season", "round", "gsis_id", "position"])
    d = d[(d.position == "QB") & d.gsis_id.notna()].rename(columns={"season": "draft_season"})
    return d[["gsis_id", "draft_season", "round"]].drop_duplicates("gsis_id")


def bucket_of(rnd) -> str:
    if pd.isna(rnd):
        return BUCKETS[2]
    return BUCKETS[0] if rnd == 1 else BUCKETS[1] if rnd <= 3 else BUCKETS[2]


def early_careers(qb: pd.DataFrame, tg: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """One row per quarterback whose career starts in the data: his bucket, the season of his
    first dropback and his EPA per dropback against the league over his first 300."""
    lg = tg.groupby("season").apply(lambda t: t["pass_epa"].sum() / t["n_pass"].sum()).rename("lg")
    q = qb.sort_values("order").merge(lg, on="season", how="left")
    q["rel"] = q["epa"] - q["lg"] * q["dropbacks"]
    q["cum"] = q.groupby("qb_id")["dropbacks"].cumsum()
    first = q.groupby("qb_id")["season"].min().rename("first_season")
    early = q[q["cum"] - q["dropbacks"] < FIRST_DROPBACKS].groupby("qb_id").agg(db=("dropbacks", "sum"), rel=("rel", "sum"))
    early = early.join(first).join(draft.set_index("gsis_id"), how="left")
    early = early[(early.first_season >= C.FIRST_SEASON + 1) & (early.draft_season.fillna(early.first_season) >= C.FIRST_SEASON)]
    early["bucket"] = early["round"].map(bucket_of)
    early["epa_db"] = early["rel"] / early["db"]
    return early.reset_index()


def fitted_priors(early: pd.DataFrame, draft: pd.DataFrame, seasons) -> tuple[dict, pd.DataFrame]:
    """{season: {qb_id: prior}} from quarterbacks whose first 300 dropbacks were complete before
    the season, and the bucket values by season."""
    bucket = draft.set_index("gsis_id")["round"].map(bucket_of).to_dict()
    out, rows = {}, []
    for s in seasons:
        past = early[(early.first_season < s - 1) & (early.db >= FIRST_DROPBACKS * 0.8)]
        vals = {}
        for b in BUCKETS:
            x = past[past.bucket == b]
            n = len(x)
            vals[b] = (x.epa_db.sum() + SHRINK_QBS * C.QB_REPLACEMENT) / (n + SHRINK_QBS) if n else C.QB_REPLACEMENT
            rows.append({"season": s, "bucket": b, "n": n, "prior": round(vals[b], 3)})
        table = {qid: vals[b] for qid, b in bucket.items()}
        out[int(s)] = {"__default__": vals[BUCKETS[2]], **table}
    return out, pd.DataFrame(rows)


class _Engine(ExtRatingEngine):
    """Undrafted quarterbacks (no draft row) take the late-or-undrafted bucket."""

    def _qb_replacement(self, order_now, ids):
        if not self.qb_priors:
            return np.full(len(ids), self.QB_REPLACEMENT)
        table = self.qb_priors.get(int(order_now) // 100, {})
        d = table.get("__default__", self.QB_REPLACEMENT)
        return np.array([table.get(i, d) for i in ids], float)


def fixed_priors(draft: pd.DataFrame, seasons, offsets: dict) -> dict:
    bucket = draft.set_index("gsis_id")["round"].map(bucket_of).to_dict()
    table = {qid: C.QB_REPLACEMENT + offsets[b] for qid, b in bucket.items()}
    return {int(s): {"__default__": C.QB_REPLACEMENT + offsets[BUCKETS[2]], **table} for s in seasons}


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    draft = load_draft()
    changed = qb_change_games(games)
    seasons = sorted(games.season.unique())
    early = early_careers(qb, tg, draft)
    print("quarterbacks whose first 300 dropbacks are in the data, by draft bucket (EPA/dropback vs league, first 300):")
    for b in BUCKETS:
        x = early[(early.bucket == b) & (early.db >= FIRST_DROPBACKS * 0.8)]
        print(f"  {b:20s} n {len(x):3d}  mean {x.epa_db.mean():+.3f}  median {x.epa_db.median():+.3f}")
    fit, table = fitted_priors(early, draft, seasons)
    print("fitted priors by season (walk-forward):")
    print(table[table.season.isin([2011, 2014, 2019, 2023, 2026])].pivot(index="season", columns="bucket", values="prior").to_string())

    mf = list(model.MARGIN_FEATS)
    print("\nversion                       | tune all  chg (mkt)  | hold all  chg (mkt)   n_chg")
    rows = []

    def run(name, priors):
        eng = _Engine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_priors=priors)
        r = score(model.build_frame(games, eng), changed, mf)
        t, h = r["tune"], r["hold"]
        print(f"{name:30s}| {t['all']:.3f}  {t['chg']:.3f} ({t['chg_mkt']:.3f}) | {h['all']:.3f}  {h['chg']:.3f} ({h['chg_mkt']:.3f})  {t['n_chg']}/{h['n_chg']}", flush=True)
        rows.append({"version": name, **{f"tune_{k}": v for k, v in t.items()}, **{f"hold_{k}": v for k, v in h.items()}})

    run("flat -0.13 (current)", None)
    run("fitted by draft bucket", fit)
    for o1, o3 in ((0.04, -0.04), (0.06, -0.06), (0.08, -0.04), (0.04, -0.08), (0.10, -0.10)):
        run(f"fixed r1 {o1:+.2f}, late {o3:+.2f}", fixed_priors(draft, seasons, {BUCKETS[0]: o1, BUCKETS[1]: 0.0, BUCKETS[2]: o3}))
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / "qb_draft_prior.csv", index=False)


if __name__ == "__main__":
    main()
