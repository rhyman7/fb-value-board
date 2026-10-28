"""Player-prop probabilities: choose the distributions, prove the calibration, fit the
parameters the board uses.

    python experiments/props_calibration.py --select   # candidate distributions, tuning seasons only
    python experiments/props_calibration.py            # the report, holdout run once; writes
                                                       # data/prop_params.json and experiments/out/

Seasons: methods are chosen on 2013-2018 (the player model's tuning seasons); 2019 on is
the holdout. Inside the tuning seasons every season is scored with parameters fitted on
the other tuning seasons. Each holdout season is scored with distribution widths fitted
on the seasons before it (as the live board would have had) and with the monotone
correction fitted on the tuning seasons only.

There are no historical prop prices, so every line here is synthetic: half-point lines
around each projection, the standard lines for touchdowns and interceptions, and a line
at the player's recent median (the closest public stand-in for a book's line). Real
lines enter only through the live log (data/props/prop_log.csv).

Rows: players a book would post the prop for, chosen only on what was known before the
game (an active player whose projection clears the stat's minimum; quarterbacks, the
projected starter), and the settlement rule that a prop is void if he does not play.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, propdist as D, props as P, propsim as S  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
CACHE = C.RAW / "players_proj.pkl"
TUNE = list(range(2013, C.HOLDOUT_FROM))


def load_proj(path: Path | None) -> pd.DataFrame:
    """The player walk-forward (players_test.py's table), cached."""
    path = path or CACHE
    if path.exists():
        return pd.read_pickle(path)
    from nflmodel import data, extra, injuries, model, players as PL, players_data as PD
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    av = PL.load_availability(verbose=False)
    table = PL.build(frame, pg, info, av, inj, pr, games=games)
    proj = PL.walk_forward(table).merge(PL.season_average(table), on=["game_id", "player_id"], how="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    proj.to_pickle(path)
    return proj


def as_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """The same rows with each projection replaced by the season average."""
    b = df.copy()
    for st in ("pass_att", "completions", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td",
               "targets", "receptions", "rec_yds", "rec_td"):
        b[st + "_proj"] = b[st + "_savg"].fillna(0)
    return b


# --------------------------------------------------------------------------
# Predictions at synthetic lines
# --------------------------------------------------------------------------

def predict(df, models, base_models, stat, hist):
    """Long table: one row per player-game x synthetic line, with the model's raw and
    corrected P(over), the season-average baseline and his recent-games share."""
    s = P.SPECS[stat]
    out = []
    for g in s.groups:
        m = P.eligible(df, stat, g)
        d = df[m].reset_index(drop=True)
        if d.empty or (stat, g) not in models.counts and (stat, g) not in models.yards:
            continue
        L = P.test_lines(P.mean(d, stat), stat)
        raw, _, _ = models.probs(d, stat, g, L, calibrate=False)
        cal, _, _ = models.probs(d, stat, g, L, calibrate=True)
        bd = as_baseline(d)
        if base_models is not None and ((stat, g) in base_models.counts or (stat, g) in base_models.yards):
            base, _, _ = base_models.probs(bd, stat, g, L, calibrate=False)
        else:
            base = np.full_like(raw, np.nan)
        h = d[["game_id", "player_id"]].merge(hist, on=["game_id", "player_id"], how="left")
        lags = h[[f"lag{i}" for i in range(1, 9)]].to_numpy(float)
        emp = P.empirical_over(lags, L)
        y = P.actual(d, stat)
        k = L.shape[1]
        rep = lambda a: np.repeat(np.asarray(a), k)
        out.append(pd.DataFrame({
            "stat": stat, "group": g, "season": rep(d["season"]), "game_id": rep(d["game_id"]), "player_id": rep(d["player_id"]),
            "line_no": np.tile(np.arange(k), len(d)), "line": L.ravel(), "mu": rep(P.mean(d, stat)), "y": rep(y),
            "p_raw": raw.ravel(), "p_cal": cal.ravel(), "p_base": base.ravel(), "p_emp": emp.ravel()}))
    if not out:
        return pd.DataFrame()
    r = pd.concat(out, ignore_index=True)
    r = r[r["line"].notna()]
    r["hit"] = (r["y"] > r["line"]).astype(float)
    return r


def median_line_test(df, models, stat, hist):
    """At a line set at the player's median of his last 8 games (at least 4): the model's
    P(over) against a coin flip and against his recent-games share."""
    s = P.SPECS[stat]
    out = []
    for g in s.groups:
        d = df[P.eligible(df, stat, g)].reset_index(drop=True)
        if d.empty or ((stat, g) not in models.counts and (stat, g) not in models.yards):
            continue
        h = d[["game_id", "player_id"]].merge(hist, on=["game_id", "player_id"], how="left")
        lags = h[[f"lag{i}" for i in range(1, 9)]].to_numpy(float)
        n = (~np.isnan(lags)).sum(1)
        med = np.nanmedian(np.where(n[:, None] >= 4, lags, np.nan), 1)
        L = np.floor(med) + 0.5
        if s.fixed_lines:
            L = np.full(len(d), s.fixed_lines[0])
        ok = ~np.isnan(L)
        if not ok.any():
            continue
        d, L, lags = d[ok].reset_index(drop=True), L[ok], lags[ok]
        cal, _, _ = models.probs(d, stat, g, L, calibrate=True)
        emp = P.empirical_over(lags, L[:, None])[:, 0]
        out.append(pd.DataFrame({"stat": stat, "group": g, "season": d["season"].to_numpy(), "line": L,
                                 "p_cal": cal, "p_emp": emp, "hit": (P.actual(d, stat) > L).astype(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def push_test(df, models, stat):
    """On a whole-number line (the integer nearest the projection): predicted push rate
    against how often the result landed exactly on it."""
    out = []
    for g in P.SPECS[stat].groups:
        d = df[P.eligible(df, stat, g)].reset_index(drop=True)
        if d.empty or ((stat, g) not in models.counts and (stat, g) not in models.yards):
            continue
        L = np.maximum(np.round(P.mean(d, stat)), 1.0)
        _, push, _ = models.probs(d, stat, g, L, calibrate=False)
        out.append({"stat": stat, "group": g, "n": len(d), "pred_push": float(np.mean(push)),
                    "actual_push": float(np.mean(P.actual(d, stat) == L))})
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def scores(r: pd.DataFrame, cols=("p_raw", "p_cal", "p_base", "p_emp")) -> dict:
    out = {"n": int(len(r))}
    common = r[list(cols)].notna().all(1)       # every method scored on the same rows
    out["n_common"] = int(common.sum())
    for c in cols:
        x = r[common]
        if len(x):
            out[c] = {"brier": D.brier(x[c], x["hit"]), "log_loss": D.log_loss(x[c], x["hit"])}
    return out


def by(r, keys, cols=("p_raw", "p_cal", "p_base", "p_emp")):
    rows = []
    for k, g in r.groupby(keys):
        sc = scores(g, cols)
        rec = dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else (k,)))
        rec["n"] = sc["n_common"]
        for c in cols:
            if c in sc:
                rec[f"{c}_brier"] = round(sc[c]["brier"], 5)
                rec[f"{c}_ll"] = round(sc[c]["log_loss"], 5)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

FAMILY = {k: "nb" for k, s in P.SPECS.items() if s.kind == "count"}   # chosen with --select, see PROPS.md


def run_select(proj):
    """Candidate distributions on the tuning seasons, each season scored with parameters
    fitted on the other tuning seasons."""
    tune = proj[proj["season"].isin(TUNE)]
    res = []
    for stat, s in P.SPECS.items():
        for g in s.groups:
            acc = {}
            for hold in TUNE:
                tr, te = tune[tune.season != hold], tune[tune.season == hold]
                dte = te[P.eligible(te, stat, g)].reset_index(drop=True)
                if dte.empty:
                    continue
                L = P.test_lines(P.mean(dte, stat), stat)
                ok = ~np.isnan(L)
                hit = (P.actual(dte, stat)[:, None] > L)[ok].astype(float)
                cands = {}
                if s.kind == "count":
                    for fam in ("poisson", "nb"):
                        for ex in (False, True):
                            dtr = tr[P.eligible(tr, stat, g)]
                            if len(dtr) < 200:
                                continue
                            dtr = dtr.sample(min(len(dtr), 20000), random_state=0)
                            fit = D.fit_count(P.actual(dtr, stat), P.mean(dtr, stat), P.role(dtr, stat, g), fam, ex)
                            m = P.PropModel(); m.counts[(stat, g)] = fit
                            cands[f"{fam}{'+exit' if ex else ''}"] = m.probs(dte, stat, g, L, calibrate=False)[0]
                else:
                    m = P.PropModel(FAMILY).fit(tr, stats=[s.vol, stat])
                    if (stat, g) not in m.yards:
                        continue
                    cands["compound"] = m.probs(dte, stat, g, L, calibrate=False)[0]
                    dtr = tr[P.eligible(tr, stat, g)].sample(frac=1, random_state=0).iloc[:8000]
                    hf = D.fit_hurdle(P.actual(dtr, stat), P.mean(dtr, stat), P.role(dtr, stat, g))
                    p0, k = hf.params(P.mean(dte, stat), P.role(dte, stat, g))
                    cands["hurdle gamma"] = 1 - D.hurdle_cdf(P.mean(dte, stat), p0, k, np.floor(np.where(ok, L, 0.5)) + 0.5)
                for name, pr in cands.items():
                    acc.setdefault(name, []).append((D.brier(pr[ok], hit), D.log_loss(pr[ok], hit), ok.sum()))
            for name, v in acc.items():
                a = np.array(v)
                res.append({"stat": stat, "group": g, "candidate": name, "n_lines": int(a[:, 2].sum()),
                            "brier": round(float(np.average(a[:, 0], weights=a[:, 2])), 5),
                            "log_loss": round(float(np.average(a[:, 1], weights=a[:, 2])), 5)})
                print(res[-1], flush=True)
    t = pd.DataFrame(res)
    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / "props_select.csv", index=False)
    print(t.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--proj", type=Path, default=None)
    args = ap.parse_args()
    proj = load_proj(args.proj)
    if args.select:
        return run_select(proj)

    stats = list(P.SPECS)
    hist = {st: P.recent_history(proj, st) for st in stats}
    tune = proj[proj["season"].isin(TUNE)]

    # ---- tuning seasons: leave one season out ------------------------------------
    print("Tuning seasons, leave one season out ...", flush=True)
    tparts, tmodels = [], {}
    for hold in TUNE:
        tr = tune[tune.season != hold]
        te = tune[tune.season == hold]
        m = P.PropModel(FAMILY).fit(tr)
        tmodels[hold] = m
        bm = P.PropModel(FAMILY).fit(as_baseline(tr))
        for st in stats:
            tparts.append(predict(te, m, bm, st, hist[st]))
        print(f"  {hold}", flush=True)
    tpred = pd.concat(tparts, ignore_index=True)

    # the correction, fitted on the tuning seasons; on the tuning seasons it is scored
    # leave-one-season-out as well, so "after" there is not fitted on the rows it scores
    iso_all = P.PropModel(FAMILY)
    iso_all.fit_isotonic(tpred.rename(columns={"p_raw": "p_over"}))
    tpred["p_cal"] = np.nan
    for hold in TUNE:
        m = tmodels[hold].fit_isotonic(tpred[tpred.season != hold].rename(columns={"p_raw": "p_over"}))
        sel = tpred.season == hold
        for st, g in tpred[sel].groupby("stat"):
            tpred.loc[g.index, "p_cal"] = m.iso[st](g["p_raw"]) if st in m.iso else g["p_raw"]

    # ---- holdout: widths walk-forward, correction from tuning only ------------------
    print("Holdout seasons, run once ...", flush=True)
    hparts, hmed, pushes = [], [], []
    for s in sorted(proj.season.unique()):
        if s < C.HOLDOUT_FROM:
            continue
        tr = proj[(proj.season < s) & (proj.season >= 2013)]
        te = proj[proj.season == s]
        m = P.PropModel(FAMILY).fit(tr)
        m.iso = iso_all.iso
        bm = P.PropModel(FAMILY).fit(as_baseline(tr))
        for st in stats:
            hparts.append(predict(te, m, bm, st, hist[st]))
            hmed.append(median_line_test(te, m, st, hist[st]))
            if s == max(proj.season.unique()) - 1 or s == C.HOLDOUT_FROM:
                pushes += [dict(x, season=int(s)) for x in push_test(te, m, st)]
        print(f"  {s}", flush=True)
    hpred = pd.concat(hparts, ignore_index=True)
    hmed = pd.concat(hmed, ignore_index=True)

    # ---- same-game simulation ----------------------------------------------------
    print("Game simulation ...", flush=True)
    sim_report = S.evaluate(proj, FAMILY, TUNE, C.HOLDOUT_FROM)

    # ---- production parameters: every season to date, correction from tuning --------
    print("Fitting production parameters ...", flush=True)
    prod = P.PropModel(FAMILY).fit(proj[proj.season >= 2013])
    prod.iso = iso_all.iso
    prod.meta = {"fitted_through": int(proj[proj.played.astype(bool)].season.max()),
                 "tuning": [TUNE[0], TUNE[-1]], "holdout_from": C.HOLDOUT_FROM,
                 "correlation": sim_report["loadings"]}
    prod.save()

    # ---- report ------------------------------------------------------------------
    report = {
        "tuning": {"by_stat": by(tpred, ["stat"]).to_dict("records"), "by_stat_group": by(tpred, ["stat", "group"]).to_dict("records")},
        "holdout": {"by_stat": by(hpred, ["stat"]).to_dict("records"), "by_stat_group": by(hpred, ["stat", "group"]).to_dict("records")},
        "calibration": {},
        "median_line": by(hmed, ["stat"], cols=("p_cal", "p_emp")).to_dict("records"),
        "median_line_tuning": None,
        "pushes": pushes,
        "sim": sim_report,
        "isotonic": {k: v.to_dict() for k, v in iso_all.iso.items()},
    }
    # coin flip at the median line
    for rec in report["median_line"]:
        x = hmed[hmed.stat == rec["stat"]]
        rec["coin_brier"] = 0.25
        rec["over_rate"] = round(float(x["hit"].mean()), 4)
        rec["model_side_hit"] = round(float(np.mean(np.where(x["p_cal"] >= 0.5, x["hit"], 1 - x["hit"]))), 4)
        strong = x[(x["p_cal"] - 0.5).abs() >= 0.05]
        rec["model_side_hit_5pt"] = round(float(np.mean(np.where(strong["p_cal"] >= 0.5, strong["hit"], 1 - strong["hit"]))), 4) if len(strong) else None
        rec["n_5pt"] = int(len(strong))
    for name, r in (("tuning", tpred), ("holdout", hpred)):
        for st, g in r.groupby("stat"):
            report["calibration"].setdefault(name, {})[st] = {
                "raw": D.calibration_table(g["p_raw"], g["hit"]), "corrected": D.calibration_table(g["p_cal"], g["hit"])}
        for (st, grp), g in r.groupby(["stat", "group"]):
            report["calibration"].setdefault(name + "_group", {})[f"{st}|{grp}"] = D.calibration_table(g["p_cal"], g["hit"])
    # edge threshold, set on the tuning seasons: how far from 50% does the model have to be
    # at the median line before its side wins often enough to beat -110?
    # (each tuning season with its leave-one-out widths and correction)
    tmed = pd.concat([median_line_test(tune[tune.season == h], tmodels[h], st, hist[st])
                      for h in TUNE for st in stats], ignore_index=True)
    thr = []
    for t in (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        x = tmed[(tmed["p_cal"] - 0.5).abs() >= t]
        if len(x):
            win = np.where(x["p_cal"] >= 0.5, x["hit"], 1 - x["hit"])
            pred = np.where(x["p_cal"] >= 0.5, x["p_cal"], 1 - x["p_cal"])
            thr.append({"min_gap": t, "n": int(len(x)), "pred_win": round(float(pred.mean()), 4), "actual_win": round(float(win.mean()), 4)})
    report["median_line_tuning"] = thr

    OUT.mkdir(exist_ok=True)
    (OUT / "props_calibration.json").write_text(json.dumps(report, indent=1, default=float))
    hpred.to_csv(C.RAW / "props_holdout_preds.csv", index=False)
    reliability_chart(tpred, hpred, OUT / "props_reliability.png")
    print_report(report)


def reliability_chart(tpred, hpred, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed: no reliability chart")
        return
    stats = list(P.SPECS)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4), sharex=True, sharey=True)
    edges = np.linspace(0, 1, 11)
    for ax, st in zip(axes.ravel(), stats):
        for r, col, lab, sty in ((tpred, "p_raw", "tuning, raw", ":"), (hpred, "p_raw", "holdout, raw", "--"), (hpred, "p_cal", "holdout, corrected", "-")):
            g = r[r.stat == st]
            idx = np.clip(np.digitize(g[col], edges) - 1, 0, 9)
            xs, ys, ns = [], [], []
            for b in range(10):
                m = idx == b
                if m.sum() >= 50:
                    xs.append(g[col][m].mean()); ys.append(g["hit"][m].mean()); ns.append(m.sum())
            ax.plot(xs, ys, sty, marker="o", ms=3, label=lab)
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8)
        ax.set_title(P.SPECS[st].label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.supxlabel("Predicted chance of the over (synthetic lines)")
    fig.supylabel("How often it went over")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  wrote {path}")


def print_report(rep):
    pd.set_option("display.width", 200)
    for k in ("tuning", "holdout"):
        print(f"\n== {k}: Brier / log loss, lower is better (raw model, corrected model, season average, last 8 games)")
        print(pd.DataFrame(rep[k]["by_stat"]).to_string(index=False))
    print("\n== holdout by stat and position")
    print(pd.DataFrame(rep["holdout"]["by_stat_group"]).to_string(index=False))
    print("\n== holdout, line at the player's recent median")
    print(pd.DataFrame(rep["median_line"]).to_string(index=False))
    print("\n== tuning, side the model favours at the median line, by size of the gap from 50%")
    print(pd.DataFrame(rep["median_line_tuning"]).to_string(index=False))
    print("\n== pushes on whole-number lines")
    print(pd.DataFrame(rep["pushes"]).round(4).to_string(index=False))
    print("\n== game simulation")
    print(json.dumps(rep["sim"], indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
