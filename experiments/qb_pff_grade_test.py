"""Quarterback value from PFF grades, tested walk-forward (Nov 5, 2026).

    python experiments/qb_pff_grade_test.py

THE RULE (written before the first run, Nov 5, 2026)
-----------------------------------------------------
Today a quarterback's value is his decayed EPA per dropback against the league, shrunk with
200 dropbacks toward a flat prior of -0.13 (config.QB_REPLACEMENT). Every unknown backup starts
at -0.13. This test adds PFF's NFL passing_summary (rhyman7/pff-data, 2014 on, one row per
quarterback per game) in two ways:

  PFF score   each PFF game is turned into an EPA-scale score with a linear map fitted
              walk-forward: for season S, PFF games of seasons before S (2014 on, 10+ dropbacks,
              weighted by dropbacks) regress the game's EPA per dropback (against that season's
              PFF average) on the PFF inputs. Two maps:
                grade      grades_pass
                grade+bt   grades_pass, btt_rate, twp_rate
              A quarterback's PFF value g as of a game is his decayed (the engine's quarterback
              half-life), dropback-weighted PFF score over earlier games (regular season and
              playoffs), shrunk toward -0.13 with Pg dropbacks. No PFF games: g = -0.13.
  prior       the EPA rating shrinks toward g instead of -0.13 (so a backup with PFF snaps
              anywhere, any team, starts from his own grades). Pg in 100, 200, 400; both maps.
  blend       value = (1 - a) * EPA rating + a * g, a in 0.25, 0.5 (grade+bt map, Pg 200).
  prior+blend prior (grade+bt, Pg 200) and blend a = 0.25.

PFF players are matched to nflverse ids by nflverse's players file (pff_id), then by unique
name within the team-season. The match rate (share of PFF dropbacks matched) is printed.

Scored as experiments/qb_test.py (the raw model, walk-forward refits): average miss of the
margin on all games and on QB-change games (either team starting a different quarterback from
its previous game), and of the total on all games, on the tuning seasons 2015-2018 (the PFF
era before the holdout; 2014 is PFF's burn-in) and the holdout 2019 on. The current model is
scored on the same games.

KEEP RULE (CONVENTIONS.md, Nov 4, 2026): a version is kept only if it is better on every
measure (margin all / QB-change and total, tuning and holdout: six numbers) and worse on none.
The version is chosen on the tuning seasons alone (best tuning QB-change margin miss among those
better than the current model on all three tuning measures); its holdout numbers decide. A
mixed result goes to Ryan as a trade-off. Paired standard errors are reported, not used.

Props check: not applicable. The player projections (players.py, players_ml.py) don't read
the game model's quarterback rating; nothing on the props side changes with it.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, model  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402
from nflmodel.ratings import _time  # noqa: E402
from experiments.qb_test import qb_change_games  # noqa: E402

warnings.filterwarnings("ignore")
TUNE = (2015, 2018)
MAPS = {"grade": ["grades_pass"], "grade+bt": ["grades_pass", "btt_rate", "twp_rate"]}
PLAYOFF_WEEK = {28: 19, 29: 20, 30: 21, 32: 22}


def pff_qb_games(qb: pd.DataFrame) -> pd.DataFrame:
    """PFF passing_summary rows with the nflverse qb id and the engine's order."""
    from cfbmodel import pff_data
    from nflmodel.pff_routes import pff_ids, _players
    from nflmodel.propmarket import norm_name
    seasons = range(2014, int(qb.season.max()) + 1)
    p = pff_data.load("passing_summary", seasons, league="nfl")
    p = p[p["dropbacks"].fillna(0) > 0].copy()
    p["week"] = p["week"].astype(int)
    p["week"] = p["week"].map(lambda w: PLAYOFF_WEEK.get(w, w))
    p = p[p["week"].between(1, 22)]
    p["order"] = p["season"].astype(int) * 100 + p["week"]
    p = p.merge(pff_ids(), left_on="player_id", right_on="pff_id", how="left")
    # name fallback: unique normalised name among nflverse quarterbacks of that team-season
    names = _players().dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]
    q = qb[["season", "team", "qb_id"]].drop_duplicates()
    q["nn"] = q["qb_id"].map(names).map(norm_name)
    q = q.dropna(subset=["nn"]).drop_duplicates(["season", "team", "nn"], keep=False)
    p["nn"] = p["player"].map(norm_name)
    miss = p["nfl_id"].isna()
    fb = p.loc[miss, ["season", "team_name", "nn"]].reset_index().merge(
        q.rename(columns={"team": "team_name"}), on=["season", "team_name", "nn"], how="inner").set_index("index")["qb_id"]
    p.loc[fb.index, "nfl_id"] = fb
    rate = p.loc[p.nfl_id.notna(), "dropbacks"].sum() / p["dropbacks"].sum()
    print(f"PFF QB games {len(p)}, dropbacks matched to nflverse ids: {rate:.1%}")
    p = p[p.nfl_id.notna()].copy()
    for c in ("grades_pass", "btt_rate", "twp_rate", "epa"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["epa_db"] = p["epa"] / p["dropbacks"]
    return p[["nfl_id", "season", "order", "dropbacks", "epa_db", "grades_pass", "btt_rate", "twp_rate"]].dropna(subset=["grades_pass"])


def fit_maps(p: pd.DataFrame, seasons) -> dict:
    """{map: {season: (cols, coef)}} fitted on PFF games of earlier seasons (walk-forward)."""
    p = p.copy()
    lg = p.groupby("season").apply(lambda x: np.average(x.epa_db.fillna(0), weights=x.dropbacks))
    p["rel"] = p["epa_db"] - p["season"].map(lg)
    out = {}
    for name, cols in MAPS.items():
        out[name] = {}
        for s in seasons:
            tr = p[(p.season < s) & (p.dropbacks >= 10)].dropna(subset=cols + ["rel"])
            if len(tr) < 200:
                continue
            X = np.column_stack([np.ones(len(tr))] + [tr[c].fillna(0).to_numpy() for c in cols])
            w = tr.dropbacks.to_numpy(float)
            beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], tr.rel.to_numpy() * np.sqrt(w), rcond=None)[0]
            out[name][int(s)] = beta
    return out


class PffEngine(ExtRatingEngine):
    """The production engine with a PFF-informed quarterback value (see the rule above)."""

    def __init__(self, *a, pff=None, coefs=None, cols=None, pg=200.0, mode="prior", alpha=0.0, **k):
        self.pff, self.coefs, self.cols, self.pg, self.mode, self.alpha = pff, coefs, cols, pg, mode, alpha
        super().__init__(*a, **k)

    def _g(self, order_now, ids):
        s = order_now // 100
        beta = self.coefs.get(int(s))
        if beta is None:
            return np.full(len(ids), self.QB_REPLACEMENT)
        p = self.pff[self.pff.order < order_now]
        X = np.column_stack([np.ones(len(p))] + [p[c].fillna(0).to_numpy() for c in self.cols])
        score = X @ beta
        w = 0.5 ** ((_time(order_now) - _time(p.order.to_numpy())) / (self.qb_half_life / 35.0)) * p.dropbacks.to_numpy(float)
        df = pd.DataFrame({"id": p.nfl_id.to_numpy(), "w": w, "ws": w * score}).groupby("id").sum()
        g = (df["ws"] + self.pg * self.QB_REPLACEMENT) / (df["w"] + self.pg)
        return pd.Series(ids).map(g).fillna(self.QB_REPLACEMENT).to_numpy()

    def _qb_replacement(self, order_now, ids):
        if self.mode in ("prior", "both"):
            return self._g(order_now, list(ids))
        return np.full(len(ids), self.QB_REPLACEMENT)

    def ratings_at(self, order_now):
        out = super().ratings_at(order_now)
        if self.mode in ("blend", "both") and self.alpha:
            ids = list(out["qb"].keys())
            g = dict(zip(ids, self._g(order_now, ids)))
            a = self.alpha
            new = {i: (1 - a) * v + a * g[i] for i, v in out["qb"].items()}
            out["qb"] = new
            self._rebuild_bases(out, order_now, new)
        return out

    def _rebuild_bases(self, out, order_now, qb_val):
        """qb_base and qb_mkt_base from the blended values, with the engine's own weights."""
        qmask = self.q_order < order_now
        qo, qid, qdb, qteam = (x[qmask] for x in (self.q_order, self.q_id, self.q_db, self.q_team))
        recent = (qo // 100 >= order_now // 100 - self.window)
        tw = self._weights(order_now, qo[recent]) * qdb[recent]
        vals = pd.Series(qid[recent]).map(qb_val).to_numpy()
        num = np.bincount(qteam[recent], weights=tw * vals, minlength=self.n)
        den = np.bincount(qteam[recent], weights=tw, minlength=self.n)
        out["qb_base"] = np.divide(num, den, out=np.full(self.n, self.QB_REPLACEMENT), where=den > 0)
        mrec = qo // 100 >= order_now // 100 - 1
        mw = 0.5 ** ((_time(order_now) - _time(qo[mrec])) / self.mkt_half_life) * self.carry ** (order_now // 100 - qo[mrec] // 100) * qdb[mrec]
        mv = pd.Series(qid[mrec]).map(qb_val).to_numpy()
        mnum = np.bincount(qteam[mrec], weights=mw * mv, minlength=self.n)
        mden = np.bincount(qteam[mrec], weights=mw, minlength=self.n)
        out["qb_mkt_base"] = np.divide(mnum, mden, out=np.full(self.n, self.QB_REPLACEMENT), where=mden > 0)

    def qb_value(self, r, qb_id):
        v = super().qb_value(r, qb_id)
        if self.mode in ("blend", "both") and self.alpha and qb_id is not None and not (isinstance(qb_id, float) and np.isnan(qb_id)) \
                and qb_id not in r["qb"]:
            g = self._g(r["order"], [qb_id])[0]
            v = (1 - self.alpha) * v + self.alpha * g
        return v


def measures(out, changed):
    o = out[out.played & out.spread_line.notna()].copy()
    o["em"] = np.abs(o.model_margin - o.result)
    o["et"] = np.abs(o.model_total - o.total)
    res = {}
    for name, x in (("tune", o[o.season.between(*TUNE)]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        c = x[x.game_id.isin(changed)]
        res[name] = {"all": x.em.mean(), "chg": c.em.mean(), "tot": x.et.mean(), "n": len(x), "n_chg": len(c)}
    return res, o[["game_id", "season", "em", "et"]]


def paired_se(a, b, col, mask):
    d = (a.loc[mask, col].to_numpy() - b.loc[mask, col].to_numpy())
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    changed = qb_change_games(games)
    pff = pff_qb_games(qb)
    seasons = sorted(games.season.unique())
    coefs = fit_maps(pff, seasons)
    for name in MAPS:
        b = coefs[name].get(2019)
        print(f"map {name} fitted for 2019 (on 2014-2018): {np.round(b, 4)}")

    base_eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games)
    base_out, _ = model.walk_forward(model.build_frame(games, base_eng))
    base, base_rows = measures(base_out, changed)

    versions = [("current model", None)]
    for m in MAPS:
        for pg in (100, 200, 400):
            versions.append((f"prior {m} Pg{pg:.0f}", dict(mode="prior", cols=MAPS[m], coefs=coefs[m], pg=pg)))
    for a in (0.25, 0.5):
        versions.append((f"blend grade+bt a{a}", dict(mode="blend", cols=MAPS["grade+bt"], coefs=coefs["grade+bt"], pg=200.0, alpha=a)))
    versions.append(("prior+blend a0.25", dict(mode="both", cols=MAPS["grade+bt"], coefs=coefs["grade+bt"], pg=200.0, alpha=0.25)))

    print("\nversion                  | tune margin all  chg   total | hold margin all  chg   total | n tune/hold (chg)")
    rows, outs = [], {}
    for name, kw in versions:
        if kw is None:
            r, rr = base, base_rows
        else:
            eng = PffEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, pff=pff, **kw)
            out, _ = model.walk_forward(model.build_frame(games, eng))
            r, rr = measures(out, changed)
        outs[name] = rr
        t, h = r["tune"], r["hold"]
        print(f"{name:25s}| {t['all']:.3f} {t['chg']:.3f} {t['tot']:.3f}  | {h['all']:.3f} {h['chg']:.3f} {h['tot']:.3f}  "
              f"| {t['n']}/{h['n']} ({t['n_chg']}/{h['n_chg']})", flush=True)
        rows.append({"version": name, **{f"tune_{k}": v for k, v in t.items()}, **{f"hold_{k}": v for k, v in h.items()}})

    df = pd.DataFrame(rows)
    cur = df.iloc[0]
    cand = df.iloc[1:]
    ok = cand[(cand.tune_all < cur.tune_all) & (cand.tune_chg < cur.tune_chg) & (cand.tune_tot < cur.tune_tot)]
    print()
    if ok.empty:
        print("No version beats the current model on all three tuning measures: nothing to carry to the holdout.")
        pick = cand.sort_values("tune_chg").iloc[0]
        print(f"(best tuning QB-change miss: {pick.version})")
    else:
        pick = ok.sort_values("tune_chg").iloc[0]
        print(f"Chosen on tuning: {pick.version}")
    a, b = outs[pick.version], outs["current model"]
    for per, mask in (("tune", a.season.between(*TUNE)), ("hold", a.season >= C.HOLDOUT_FROM)):
        for col, lab in (("em", "margin all"), ("et", "total")):
            d, se = paired_se(a, b, col, mask.to_numpy())
            print(f"  {per} {lab:11s} {d:+.4f} (SE {se:.4f})")
        cm = (mask & a.game_id.isin(changed)).to_numpy()
        d, se = paired_se(a, b, "em", cm)
        print(f"  {per} margin chg  {d:+.4f} (SE {se:.4f})")
    better = all(pick[f"{p}_{m}"] < cur[f"{p}_{m}"] for p in ("tune", "hold") for m in ("all", "chg", "tot"))
    print(f"\nKEEP: {'yes' if (better and not ok.empty) else 'no'}")
    outdir = Path(__file__).parent / "out"
    outdir.mkdir(exist_ok=True)
    df.round(4).to_csv(outdir / "qb_pff_grade.csv", index=False)


if __name__ == "__main__":
    main()
