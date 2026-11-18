"""Re-reading the model's chances on NFL floor rungs: a pre-set test.

    PFF_DATA_DIR=<pff-data clone> python experiments/rung_recalibration_test.py freeze
    PFF_DATA_DIR=<pff-data clone> python experiments/rung_recalibration_test.py score   # after week 13 is graded

Plan written before any run (Project doc claude/plan-rung-recalibration-test-2026-11-18.md).

Rungs: DK/FD alternate passing, rushing and receiving yards, over only, the books' reading
(parlays.RUNG_CAL) 65-95%, built exactly as experiments/floor_rung_diagnosis.py builds them.
  2025: pricing snapshot, the model's chance from bet-time projections (the Nov 6 definition).
  2026: the live prop log, the chance the board logged. Weeks 8-10 were seen by the diagnosis.

Versions, each fitted per stat (pass_yds, rush_yds, rec_yds):
  A  isotonic map from the model's chance to the share that cleared, ends held flat.
  B  logistic: logit(p_new) = a + b * logit(p_model).
  C  B plus the rung's share of the main line and position (RB, WR, TE, QB); rungs without a main
     line use the 2025 median share for their stat.

freeze (tuning only, run before any week 11 rung is scored):
  2025 cut into five blocks of consecutive weeks; each block scored with a map fitted on the other
  four. The version with the lowest cross-fitted Brier (all stats) is chosen; none better than the
  raw chance -> nothing goes to the holdout. The chosen version is refitted on 2025 plus 2026 weeks
  8-10 (no data after week 10) and written to experiments/frozen/rung_recalibration_map.json.
score (holdout 2026 weeks 11-13; week 14 added once if under 1,200 rungs):
  Measures, chosen version against the raw chance on the same rungs: Brier and log loss on 2025
  cross-fitted (tuning) and on the holdout. Kept only if better on all four with the holdout Brier
  gain at least 1 paired standard error (clustered by game); otherwise mixed -> a trade-off for Ryan.
  Reported, not used: calibration bands, the books' Brier, and the money view for rungs the re-read
  chance puts 5+ points above the books.
"""
from __future__ import annotations

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

FROZEN = HERE / "frozen" / "rung_recalibration_map.json"
OUT = HERE / "out" / "rung_recalibration_test.json"
STATS = ("pass_yds", "rush_yds", "rec_yds")
POS = ("QB", "RB", "WR", "TE")
EPS = 1e-4
LAST_FIT_WEEK_2026 = 10
HOLDOUT, EXTRA_WEEK, MIN_HOLDOUT = (11, 12, 13), 14, 1200


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def brier(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def logloss(p, y):
    p, y = np.clip(np.asarray(p, float), EPS, 1 - EPS), np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------- the three versions: fit(rows of one stat) -> params; apply(params, rows) -> chance ----------

def _share(x: pd.DataFrame, med: float) -> np.ndarray:
    s = (x["line"] / x["main_line"]).astype(float)
    return s.fillna(med).clip(0.0, 1.5).to_numpy()


def _design(x: pd.DataFrame, version: str, med: float) -> np.ndarray:
    cols = [logit(x["model"])]
    if version == "C":
        cols.append(_share(x, med))
        g = x["group"].astype(str).to_numpy()
        cols += [(g == p).astype(float) for p in POS[1:]]          # QB is the base level
    return np.column_stack(cols)


def fit(x: pd.DataFrame, version: str) -> dict:
    if version == "A":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x["model"], x["hit"])
        return {"version": "A", "x": [round(float(v), 6) for v in iso.X_thresholds_], "y": [round(float(v), 6) for v in iso.y_thresholds_]}
    from sklearn.linear_model import LogisticRegression
    med = float((x["line"] / x["main_line"]).median()) if x["main_line"].notna().any() else 0.6
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_design(x, version, med), x["hit"].astype(int))
    return {"version": version, "intercept": round(float(lr.intercept_[0]), 6), "coef": [round(float(c), 6) for c in lr.coef_[0]],
            "terms": ["logit(model)"] + (["rung/main line", *[f"pos={p}" for p in POS[1:]]] if version == "C" else []), "median_share": round(med, 6)}


def apply(prm: dict, x: pd.DataFrame) -> np.ndarray:
    if prm["version"] == "A":
        return np.interp(x["model"].astype(float), prm["x"], prm["y"])
    z = prm["intercept"] + _design(x, prm["version"], prm["median_share"]) @ np.asarray(prm["coef"])
    return 1 / (1 + np.exp(-z))


def apply_all(m: dict, x: pd.DataFrame) -> np.ndarray:
    out = x["model"].astype(float).to_numpy().copy()
    for st, prm in m.items():
        k = (x["stat"] == st).to_numpy()
        if k.any():
            out[k] = apply(prm, x[k])
    return out


# ---------- tuning ----------

def blocks(a: pd.DataFrame) -> pd.Series:
    weeks = sorted(a["week"].unique())
    cut = np.array_split(np.array(weeks), 5)
    lab = {int(w): i for i, c in enumerate(cut) for w in c}
    return a["week"].map(lab)


def crossfit(a: pd.DataFrame, version: str) -> np.ndarray:
    blk, out = blocks(a), np.full(len(a), np.nan)
    for b in sorted(blk.unique()):
        tr, te = (blk != b).to_numpy(), (blk == b).to_numpy()
        for st in STATS:
            ktr, kte = tr & (a["stat"] == st).to_numpy(), te & (a["stat"] == st).to_numpy()
            if kte.any():
                out[kte] = apply(fit(a[ktr], version), a[kte])
    return out


def paired(new, old, y, games) -> dict:
    """Mean gain (old minus new; positive = new better) with an SE clustered by game, for Brier and log loss."""
    r = {}
    for name, f in (("brier", brier), ("log_loss", logloss)):
        d = pd.Series(f(old, y) - f(new, y)).groupby(np.asarray(games)).sum()
        n = len(y)
        r[name] = {"raw": round(float(f(old, y).mean()), 5), "re_read": round(float(f(new, y).mean()), 5),
                   "gain": round(float(d.sum() / n), 5), "se": round(float(d.std(ddof=1) * np.sqrt(len(d)) / n), 5)}
    return r


def load_rungs():
    import floor_rung_diagnosis as D
    p = D.projections()
    a = D.rungs_2025(p)
    b = D.rungs_2026(p)
    b["proj"] = pd.to_numeric(b["proj"], errors="coerce")
    for x in (a, b):
        x["books"] = D.reading(x["price"]) if "books" not in x or x["books"].isna().all() else x["books"]
    a = a[a["books"].between(0.65, 0.95)].copy() if "books" in a else a
    return a.reset_index(drop=True), b.reset_index(drop=True)


def freeze(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} already exists: the map is frozen. Delete it only with Ryan's say-so.")
    b_fit = b[b["week"] <= LAST_FIT_WEEK_2026]
    assert b_fit["week"].max() <= LAST_FIT_WEEK_2026
    y, g = a["hit"].to_numpy(), a["game_id"].to_numpy()
    tune = {v: paired(crossfit(a, v), a["model"].to_numpy(), y, g) for v in ("A", "B", "C")}
    best = min(tune, key=lambda v: tune[v]["brier"]["re_read"])
    rec = {"written": pd.Timestamp.now(tz="America/New_York").isoformat(timespec="minutes"),
           "plan": "claude/plan-rung-recalibration-test-2026-11-18.md",
           "fit_rows": {"2025": int(len(a)), "2026 weeks 8-10": int(len(b_fit))}, "last_2026_week_in_fit": int(b_fit["week"].max()),
           "tuning_crossfit_2025": tune}
    if tune[best]["brier"]["gain"] <= 0:
        rec.update(chosen=None, note="No version beat the raw model chance on tuning: nothing goes to the holdout, nothing is kept.")
    else:
        both = pd.concat([a, b_fit], ignore_index=True)
        rec.update(chosen=best, map={st: fit(both[both["stat"] == st], best) for st in STATS})
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps(rec, indent=1))
    return rec


def bands(p, y) -> list:
    out = []
    for lo in np.arange(0.4, 1.0, 0.05):
        k = (p >= lo) & (p < lo + 0.05)
        if k.sum() >= 50:
            out.append({"says": f"{lo:.0%}-{lo + 0.05:.0%}", "n": int(k.sum()), "said": round(float(p[k].mean()), 3), "cleared": round(float(y[k].mean()), 3)})
    return out


def score(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    import floor_rung_diagnosis as D
    rec = json.loads(FROZEN.read_text())
    res = {"frozen": {k: rec[k] for k in ("written", "chosen", "last_2026_week_in_fit")}}
    if not rec.get("chosen"):
        res["result"] = "Nothing went to the holdout (no version beat the raw chance on tuning)."
        return res
    weeks = list(HOLDOUT)
    h = b[b["week"].isin(weeks)]
    if len(h) < MIN_HOLDOUT:
        weeks.append(EXTRA_WEEK)
        h = b[b["week"].isin(weeks)]
    h = h.copy()
    h["re_read"] = apply_all(rec["map"], h)
    t = rec["tuning_crossfit_2025"][rec["chosen"]]
    ho = paired(h["re_read"].to_numpy(), h["model"].to_numpy(), h["hit"].to_numpy(), h["game_id"].to_numpy())
    better = [t["brier"]["gain"] > 0, t["log_loss"]["gain"] > 0, ho["brier"]["gain"] > 0, ho["log_loss"]["gain"] > 0]
    kept = all(better) and ho["brier"]["gain"] >= ho["brier"]["se"]
    res.update(holdout_weeks=weeks, holdout_rungs=int(len(h)), holdout_games=int(h["game_id"].nunique()), tuning=t, holdout=ho,
               better_on=better, verdict="kept (provisional)" if kept else ("trade-off for Ryan" if any(better) else "not kept"))
    bb = brier(h["books"], h["hit"])
    res["reported"] = {
        "books_brier_holdout": round(float(bb.mean()), 5),
        "bands_raw": bands(h["model"].to_numpy(), h["hit"].to_numpy()),
        "bands_re_read": bands(h["re_read"].to_numpy(), h["hit"].to_numpy()),
        "by_stat": {st: paired(g["re_read"].to_numpy(), g["model"].to_numpy(), g["hit"].to_numpy(), g["game_id"].to_numpy()) for st, g in h.groupby("stat")},
        "money_re_read_5plus_above_books": D.money(h[h["re_read"] - h["books"] >= 0.05]),
    }
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("freeze", "score"):
        raise SystemExit(__doc__)
    a, b = load_rungs()
    out = freeze(a, b) if mode == "freeze" else score(a, b)
    if mode == "score":
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
