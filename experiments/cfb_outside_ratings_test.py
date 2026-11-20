"""Outside computer ratings as an input to the college margin model (Nov 20, 2026).

    PFF_DATA_DIR=<pff-data clone> python experiments/cfb_outside_ratings_test.py

Plan, written before any run: Project doc claude/plan-cfb-outside-ratings-test-2026-11-20.md.
Data checks (step 0): claude/cfb-outside-ratings-step0-data-checks-2026-11-20.md. This script was
committed before its first run.

CONDITIONAL RUN (Ryan's choice, Nov 20): step 0's check 4, that the source does not revise its
predictions after games, cannot be done until the Nov 21 games are in the archive. The fits below
are run before that check and are void if it fails.

The change: one new input in the market-residual margin fit (cfbmodel.model.market_residual):
  d_out = outside predicted home margin - the closing spread   (0 when a game has none)
It earns its own weight against the line, fitted for season S on earlier seasons only, with the
board's recency weights and ridge setting. The game model underneath, the totals and everything
else are untouched.

Versions:
  A  outside = the file's own average of all systems for the game (lineavg)
  B  outside = the mean of the systems present in every season 2015-2026 (SYSTEMS_B, fixed in
     step 0 by coverage alone)
  C  as A, plus d_out_early = d_out in weeks 1-4 and 0 afterwards

Rule, set before running:
  - Twelve measures: average miss, RMS miss and side-of-closing-line win rate, on the board's
    frame and on the bet-time frame (last week's quarterbacks, no wind, no injury inputs), each on
    tuning 2017-19 and holdout 2021-25 (2020 in neither). The plan's text named four kinds (it
    listed only the average miss for bet-time) while saying twelve; this is the stricter reading,
    fixed here before the run. Against today's board on the same games: played FBS games with a
    closing spread and a walk-forward number. Win rate counts games where both the board and the
    candidate take a side and the game did not land on the line.
  - Chosen on tuning alone: among versions better than the board on all six tuning measures, the
    lowest tuning average miss.
  - Kept only if the chosen version is better on every measure on both periods and the holdout
    average miss improves by at least 1 paired standard error. Provisional, rechecked on the full
    2026 season in January.
  - Mixed, or better everywhere but under 1 SE: a trade-off for Ryan.
  - No version better on all tuning measures: nothing goes to the holdout, nothing is kept.
  - Reported, not used: 2026 so far, the weight on d_out by season, results by part of season,
    how the other inputs' weights move.

Timing limit, true even on a pass: the source posts a week's predictions midweek, so the bet-time
measure here flatters the idea (it uses numbers that do not exist when early lines are priced).

Writes experiments/out/cfb_outside_ratings_test.json.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
OUT = HERE / "out" / "cfb_outside_ratings_test.json"
OUTSIDE = HERE.parent / "data" / "cfb" / "outside"
TUNE, HOLD = (2017, 2019), (2021, 2025)
EARLY_WEEKS = 4

# Step 0, check 5: present for at least 80% of lined FBS games in every season 2015-2026.
SYSTEMS_B = ["lineca", "lineclean", "linecong", "linecurry", "linedokter", "linedonchess", "linedunk", "lineelo",
             "linefpi", "linehow", "linekam", "linelaz", "lineloud", "linemassey", "linemoore", "linepfz",
             "linepibias", "linepimean", "linepiratings", "linepugh", "linesag", "linesaggm", "linesagpred"]
ALIAS = {"linemass": "linemassey", "lineespn": "linefpi", "linebillings": "linebill"}   # one system, two spellings across seasons
MANUAL = {"Connecticut": "UConn", "Florida St.": "Florida State", "Rice": "Rice"}        # the three the score match left ambiguous


# ---------------------------------------------------------------------------------------------
# The outside files
# ---------------------------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(path.read_bytes().decode("latin-1"))))
    head = [ALIAS.get(h.strip().lower(), h.strip().lower()) for h in rows[0]]
    d = pd.DataFrame([r for r in rows[1:] if len(r) == len(head)], columns=head)
    d = d.loc[:, ~d.columns.duplicated()]
    for c in d.columns:
        if c not in ("home", "road"):
            d[c] = pd.to_numeric(d[c].str.strip().replace({"": np.nan, ".": np.nan}), errors="coerce")
    d["home"], d["road"] = d["home"].str.strip(), d["road"].str.strip()
    return d


def load_outside(games: pd.DataFrame) -> pd.DataFrame:
    """game_id, out_a (the file's average), out_b (mean of SYSTEMS_B), in the board's home-margin
    terms. Teams are matched by a name map inferred from unique final scores within a season."""
    files = {s: _read(OUTSIDE / f"ncaa{s}.csv") for s in sorted(games.season.unique()) if (OUTSIDE / f"ncaa{s}.csv").exists()}
    live = sorted((OUTSIDE / "live").glob("*.csv"))
    if live:                                   # this week's games are not in the season archive yet
        last = max(files)
        lv = _read(live[-1])
        files[last] = pd.concat([files[last], lv[[c for c in lv.columns if c in files[last].columns]]], ignore_index=True)
    votes = defaultdict(Counter)
    for s, d in files.items():
        key = defaultdict(list)
        for r in games[(games.season == s) & games.played].itertuples():
            key[(int(r.home_score), int(r.away_score))].append((r.home_team, r.away_team))
        for r in d.dropna(subset=["hscore", "vscore"]).itertuples():
            for hs, vs, flip in ((r.hscore, r.vscore, False), (r.vscore, r.hscore, True)):
                c = key.get((int(hs), int(vs)), [])
                if len(c) == 1:
                    h, a = c[0] if not flip else c[0][::-1]
                    votes[r.home][h] += 1
                    votes[r.road][a] += 1
    name = {k: c.most_common(1)[0][0] for k, c in votes.items()
            if c.most_common(1)[0][1] >= 3 and c.most_common(1)[0][1] / sum(c.values()) >= 0.6}
    name.update(MANUAL)
    out = []
    for s, d in files.items():
        by = defaultdict(list)
        for r in games[games.season == s].itertuples():
            by[(r.home_team, r.away_team)].append(r)
        b_cols = [c for c in SYSTEMS_B if c in d.columns]
        d = d.assign(h=d.home.map(name), a=d.road.map(name), out_b=d[b_cols].mean(axis=1))
        used = set()
        for r in d.itertuples():
            for h, a, sign in ((r.h, r.a, 1.0), (r.a, r.h, -1.0)):     # neutral sites are sometimes listed the other way round
                cand = [c for c in by.get((h, a), []) if c.game_id not in used]
                if len(cand) > 1 and not np.isnan(getattr(r, "hscore", np.nan)):
                    cand = [c for c in cand if c.played and {int(c.home_score), int(c.away_score)} == {int(r.hscore), int(r.vscore)}] or cand
                if cand:
                    used.add(cand[0].game_id)
                    out.append({"game_id": cand[0].game_id, "out_a": sign * r.lineavg, "out_b": sign * r.out_b})
                    break
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# The board's model, and the candidate stack
# ---------------------------------------------------------------------------------------------

def frames():
    """(built close frame, built bet-time frame): every input the board's game model uses."""
    from cfbmodel import data, extra, model
    from cfbmodel.pipeline import CFBRatingEngine, PFFRET
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games["wind_source"] = np.where(games["played"], "recorded", "")
    inputs = extra.load_inputs(games, refresh_current=False, verbose=False)
    engine = CFBRatingEngine(tg, qb, games=games, extra=inputs["rated"])
    close = PFFRET.add(model.build_frame(games, engine, inputs=inputs))
    blind = games.assign(home_qb_id=np.nan, away_qb_id=np.nan, wind=np.nan)
    early = PFFRET.add(model.build_frame(blind, engine))
    return close, early


def add_outside(df: pd.DataFrame, outside: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(outside, on="game_id", how="left")
    for v, col in (("a", "out_a"), ("b", "out_b")):
        gap = df[col] - df["spread_line"]
        df[f"d_out_{v}"] = gap.where(gap.notna(), 0.0)
    df["d_out_a_early"] = np.where(df["week"] <= EARLY_WEEKS, df["d_out_a"], 0.0)
    return df


VERSIONS = {"A": ["d_out_a"], "B": ["d_out_b"], "C": ["d_out_a", "d_out_a_early"]}


def stacked(df: pd.DataFrame, extra_feats: list[str]):
    """The board's walk-forward, then the market-residual step refitted with extra inputs. With
    no extras this reproduces the board's proj_margin exactly (asserted in main)."""
    from cfbmodel import config as C, model
    wf, info = model.walk_forward(df)
    full = df.copy()
    MF = list(model.MARGIN_FEATS)
    for col in MF:
        if col not in full:
            full[col] = 0.0
    proj = wf["proj_margin"].copy()
    coefs = {}
    for s in sorted(wf.season.unique()):
        hl = (wf.season == s) & wf["spread_line"].notna()
        train = full[(full.season < s) & full.played & full["spread_line"].notna()]
        st = model.market_residual(train, MF + extra_feats, s)
        if st is not None and hl.any():
            X = full.loc[wf.index[hl], MF + extra_feats + ["spread_line"]].to_numpy(float)
            proj.loc[hl] = wf.loc[hl, "spread_line"] + st.predict(X)
            coefs[int(s)] = st.raw_coefs(MF + extra_feats + ["spread_line"])
    return wf.assign(cand=proj), coefs


# ---------------------------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------------------------

def scored(wf: pd.DataFrame) -> pd.DataFrame:
    return wf[wf.played & wf["spread_line"].notna() & wf["proj_margin"].notna() & wf["cand"].notna() & (wf.fbs_game == 1)]


def measures(g: pd.DataFrame) -> dict:
    eb, ec = (g.proj_margin - g.result).abs(), (g.cand - g.result).abs()
    d = ec - eb
    res = g.result - g.spread_line
    sb, sc = np.sign(g.proj_margin - g.spread_line), np.sign(g.cand - g.spread_line)
    both = (sb != 0) & (sc != 0) & (res != 0)
    return {"n": int(len(g)),
            "mae_board": float(eb.mean()), "mae_cand": float(ec.mean()), "d_mae": float(d.mean()),
            "se_d_mae": float(d.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else None,
            "rmse_board": float(np.sqrt(((g.proj_margin - g.result) ** 2).mean())),
            "rmse_cand": float(np.sqrt(((g.cand - g.result) ** 2).mean())),
            "n_side": int(both.sum()),
            "win_board": float((np.sign(res[both]) == sb[both]).mean()) if both.any() else None,
            "win_cand": float((np.sign(res[both]) == sc[both]).mean()) if both.any() else None}


def period(g: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return g[(g.season >= lo) & (g.season <= hi)]


def better(m_close: dict, m_early: dict) -> dict:
    """The six comparisons for one period (True = candidate better)."""
    return {"mae": m_close["mae_cand"] < m_close["mae_board"],
            "rmse": m_close["rmse_cand"] < m_close["rmse_board"],
            "win": (m_close["win_cand"] or 0) > (m_close["win_board"] or 0),
            "bet_mae": m_early["mae_cand"] < m_early["mae_board"],
            "bet_rmse": m_early["rmse_cand"] < m_early["rmse_board"],
            "bet_win": (m_early["win_cand"] or 0) > (m_early["win_board"] or 0)}


def main() -> None:
    close, early = frames()
    outside = load_outside(close)
    close, early = add_outside(close, outside), add_outside(early, outside)

    base, _ = stacked(close, [])
    assert float((base.cand - base.proj_margin).abs().max()) < 1e-9, "the re-implemented stack does not reproduce the board"

    res = {"note": "CONDITIONAL: run before step 0's check 4 (no after-the-fact revisions). Void if that check fails.",
           "versions": {}, "systems_b": SYSTEMS_B}
    tune_ok = {}
    for v, feats in VERSIONS.items():
        wc, coefs = stacked(close, feats)
        we, _ = stacked(early, feats)
        gc, ge = scored(wc), scored(we)
        mt, mte = measures(period(gc, *TUNE)), measures(period(ge, *TUNE))
        bt = better(mt, mte)
        res["versions"][v] = {"feats": feats, "tune": mt, "tune_bet": mte, "tune_better": bt,
                              "coef_by_season": {s: {k: round(float(c[k]), 4) for k in feats + ["spread_line"]} for s, c in coefs.items()}}
        tune_ok[v] = all(bt.values())
        res["versions"][v]["_frames"] = (gc, ge, coefs)

    passing = [v for v in VERSIONS if tune_ok[v]]
    res["tune_passing"] = passing
    if not passing:
        res["verdict"] = "No version is better than the board on all six tuning measures. Nothing is carried to the holdout; nothing is kept."
    else:
        chosen = min(passing, key=lambda v: res["versions"][v]["tune"]["mae_cand"])
        gc, ge, coefs = res["versions"][chosen]["_frames"]
        mh, mhe = measures(period(gc, *HOLD)), measures(period(ge, *HOLD))
        bh = better(mh, mhe)
        se_ok = mh["d_mae"] < 0 and abs(mh["d_mae"]) >= mh["se_d_mae"]
        res["chosen"] = chosen
        res["holdout"] = {"close": mh, "bet": mhe, "better": bh, "one_se": bool(se_ok)}
        n_better = sum(res["versions"][chosen]["tune_better"].values()) + sum(bh.values())
        if all(bh.values()) and se_ok:
            res["verdict"] = f"Version {chosen}: better on all 12 measures and the holdout average miss by at least 1 SE. Passes the keep rule (provisional)."
        else:
            res["verdict"] = f"Version {chosen}: a trade-off ({n_better} of 12 measures better; holdout 1 SE: {bool(se_ok)}). Ryan's call."
        # reported, not used
        rep = {"2026": {"close": measures(period(gc, 2026, 2026)), "bet": measures(period(ge, 2026, 2026))},
               "by_season": {int(s): measures(d) for s, d in gc.groupby("season")},
               "by_part": {}}
        for label, lo, hi in (("weeks 1-4", 1, 4), ("weeks 5-9", 5, 9), ("weeks 10+", 10, 99)):
            part = gc[(gc.week >= lo) & (gc.week <= hi)]
            rep["by_part"][label] = {"tune": measures(period(part, *TUNE)), "holdout": measures(period(part, *HOLD)),
                                     "2026": measures(period(part, 2026, 2026))}
        _, base_coefs = stacked(close, [])
        last = max(coefs)
        rep["weights_latest_season"] = {"season": last,
                                        "board": {k: round(float(x), 4) for k, x in base_coefs[last].items()},
                                        "candidate": {k: round(float(x), 4) for k, x in coefs[last].items()}}
        res["reported"] = rep
    for v in res["versions"]:
        res["versions"][v].pop("_frames")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")

    print(res["note"])
    for v, r in res["versions"].items():
        t, tb = r["tune"], r["tune_bet"]
        print(f"{v} tuning: miss {t['mae_board']:.3f} -> {t['mae_cand']:.3f}, rms {t['rmse_board']:.3f} -> {t['rmse_cand']:.3f}, "
              f"side {100 * t['win_board']:.1f}% -> {100 * t['win_cand']:.1f}% ({t['n_side']}); "
              f"bet-time miss {tb['mae_board']:.3f} -> {tb['mae_cand']:.3f}; better on {sum(r['tune_better'].values())} of 6")
    print(res["verdict"])


if __name__ == "__main__":
    main()
