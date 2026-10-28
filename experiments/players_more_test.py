"""Three more player-projection tests (Oct 2026).

    python experiments/players_more_test.py [--proj path]

1. Team volume with play-calling: the team pass-attempt and carry regressions with pace
   (plays per game) and neutral pass rate, the team's and the opponent's, added. Scored on
   team-games: average miss on pass attempts and carries.
2. Quarterback passing through the boosted last step (players_ml.py's approach, with
   quarterback inputs): passing yards, attempts, completions and touchdowns, for the
   projected starter in games he started. Average miss and the Brier score at lines.
3. Questionable players: how much of his usual usage a player listed Questionable gets
   when he plays, by position and practice status (full, limited, did not practice),
   fitted on the tuning seasons and applied as a multiplier. Scored on Questionable
   players who played.

Tuning 2014-2018, holdout 2019 on; each change is kept only if it helps on both.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, players as PL, propdist as D, props as P, roles  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"


def periods(d):
    return (("2014-18", d[(d.season >= 2014) & (d.season < C.HOLDOUT_FROM)]), ("2019-26", d[d.season >= C.HOLDOUT_FROM]))


# ---------------------------------------------------------------------- 1. team volume
def test_volume(proj, F):
    t = proj.drop_duplicates(["game_id", "team"]).copy()      # pace and neutral_pass are already on every row
    o = t[["game_id", "team", "pace", "neutral_pass"]].rename(columns={"team": "opp", "pace": "opp_pace", "neutral_pass": "opp_np"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    for c in ("pace", "neutral_pass", "opp_pace", "opp_np"):
        t[c] = t[c].fillna(t[c].median())
    base = PL.VOL_FEATS
    more = base + ["pace", "neutral_pass", "opp_pace", "opp_np"]
    rows = []
    for target in ("team_att", "team_car"):
        preds = {"current": pd.Series(np.nan, index=t.index), "with play-calling": pd.Series(np.nan, index=t.index)}
        for s in sorted(t.season.unique()):
            if s < 2014:
                continue
            tr = t[(t.season < s) & (t.season >= s - 8) & t.played.astype(bool) & t[target].notna()]
            te = t.season == s
            for name, feats in (("current", base), ("with play-calling", more)):
                m = PL.Lin(5.0).fit(tr[feats].fillna(0), tr[target].to_numpy(float))
                preds[name][te] = m.predict(t.loc[te, feats].fillna(0))
        ok = t.played.astype(bool) & t[target].notna()
        for name, pr in preds.items():
            for per, d in periods(t[ok].assign(pred=pr[ok])):
                rows.append({"test": "team volume", "stat": target, "variant": name, "period": per, "n": len(d),
                             "mae": round(float(np.mean(np.abs(d["pred"] - d[target]))), 3)})
    return rows


# ---------------------------------------------------------------------- 2. quarterback passing
QB_F = ["{s}_p", "{s}_proj", "{s}_recent", "proj_team_att", "imp_pts", "margin", "proj_total", "wind", "home", "def_ypa", "def_comp",
        "ypa", "comp", "ptd_rate", "int_rate", "start_sum", "one_sum", "age", "draft_pick", "exp", "new_team", "pace", "neutral_pass", "opp_miss_pr", "opp_miss_db"]


def test_qb(proj):
    p = proj.copy()
    qb = (p.group == "QB") & (p.att_share > 0.5)
    for st in ("pass_yds", "pass_att", "completions", "pass_td"):
        p[st + "_recent"] = p[st + "_st_sum"] / p["start_sum"].where(p["start_sum"] > 0.3)
    rows = []
    for st in ("pass_yds", "pass_att", "completions", "pass_td"):
        cols = [f.format(s=st) for f in QB_F]
        new = p[st + "_proj"].copy()
        for s in sorted(p.season.unique()):
            if s < 2014:
                continue
            tr = p[qb & (p.season < s) & p.played.astype(bool) & (p.started == 1)]
            te = p[qb & (p.season == s)]
            if te.empty:
                continue
            m = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                              min_samples_leaf=100, l2_regularization=1.0, random_state=0)
            m.fit(tr[cols].to_numpy(float), tr[st].clip(lower=0).to_numpy(float))
            new.loc[te.index] = m.predict(te[cols].to_numpy(float))
        for name, col in (("current", p[st + "_proj"]), ("boosted", new)):
            x = p.assign(**{st + "_proj": col})
            sel = P.eligible(x, st, "QB") & (x.season >= 2014)
            d0 = x[sel]
            spec = P.SPECS[st]
            m = P.PropModel({}).fit(d0[d0.season < C.HOLDOUT_FROM], stats=[spec.vol, st] if spec.vol else [st])
            for per, d in periods(d0):
                base_mu = p.loc[d.index, st + "_proj"].to_numpy(float)     # lines around the current projection, same for both
                if spec.fixed_lines:
                    L = np.tile(np.array(spec.fixed_lines, float), (len(d), 1))
                elif st == "pass_yds":
                    L = (np.floor(base_mu / 5) * 5 + 0.5)[:, None] + np.array([-15, 0, 15])[None, :]
                else:
                    L = (np.floor(base_mu) + 0.5)[:, None] + spec.step * np.array([-1, 0, 1])[None, :]
                d = d.reset_index(drop=True)
                rec = {"test": "QB passing", "stat": st, "variant": name, "period": per, "n": len(d),
                       "mae": round(float(np.mean(np.abs(d[st + "_proj"] - d[st]))), 3)}
                if (st, "QB") in m.counts or (st, "QB") in m.yards:
                    over, _, _ = m.probs(d, st, "QB", L, calibrate=False)
                    ok = L >= 0.5
                    rec["brier"] = round(D.brier(over[ok], (d[st].to_numpy()[:, None] > L)[ok].astype(float)), 5)
                rows.append(rec)
    return rows


# ---------------------------------------------------------------------- 3. questionable players
def test_questionable(proj, R):
    pr = R["practice"]
    p = proj.merge(pr.rename(columns={"practice": "practice_lvl", "report": "report_lvl"}), on=["season", "week", "team", "player_id"], how="left")
    q = p[(p.report_lvl == 1) & p.played.astype(bool) & (p.active == 1) & p.group.isin(["RB", "WR", "TE"])].copy()
    q["practice_lvl"] = q["practice_lvl"].fillna(0)
    rows = []
    for st in ("carries", "rush_yds", "targets", "receptions", "rec_yds"):
        tune = q[(q.season >= 2014) & (q.season < C.HOLDOUT_FROM) & (q[st + "_proj"] > 0.5)]
        mult = (tune.groupby(["group", "practice_lvl"])[st].sum() / tune.groupby(["group", "practice_lvl"])[st + "_proj"].sum()).clip(0.5, 1.2)
        k = pd.MultiIndex.from_frame(q[["group", "practice_lvl"]])
        q["m"] = pd.Series(mult.reindex(k).to_numpy(), index=q.index).fillna(1.0)
        for name, col in (("current", q[st + "_proj"]), ("Questionable multiplier", q[st + "_proj"] * q["m"])):
            for per, d in periods(q.assign(pred=col)):
                d = d[d[st + "_proj"] > 0.5]
                rows.append({"test": "Questionable", "stat": st, "variant": name, "period": per, "n": len(d),
                             "mae": round(float(np.mean(np.abs(d["pred"] - d[st]))), 3), "bias": round(float(np.mean(d["pred"] - d[st])), 3)})
        rows.append({"test": "Questionable", "stat": st, "variant": "multipliers (tuning)", "period": "", "n": len(tune),
                     "mae": None, "bias": None, "note": "; ".join(f"{g} practice {int(l)}: {v:.2f}" for (g, l), v in mult.items())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=None)
    args = ap.parse_args()
    games, _, _ = data.load_all(refresh=False, verbose=False)
    proj = pd.read_pickle(args.proj)
    R = roles.load_all(games, verbose=False)
    F = roles.features(proj, R)
    proj = proj.merge(F.drop(columns=[c for c in ("depth",) if c in F]), on=["game_id", "player_id"], how="left")
    rows = []
    print("1. team volume ...", flush=True)
    rows += test_volume(proj, proj[["game_id", "team", "pace", "neutral_pass"]])
    print("2. quarterback passing ...", flush=True)
    rows += test_qb(proj)
    print("3. questionable players ...", flush=True)
    rows += test_questionable(proj, R)
    r = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    r.to_csv(OUT / "players_more_test.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 200)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
