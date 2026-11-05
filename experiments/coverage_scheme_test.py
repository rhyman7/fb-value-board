"""Man / zone coverage in the NFL receiving projections (PFF coverage-scheme tables).

    python experiments/coverage_scheme_test.py      # needs rhyman7/pff-data (nfl/) next to this repo

Ryan's question (Nov 5, 2026): can the model tell which cornerback a receiver will face? Coverage
charting by receiver isn't sold, but PFF's Receiving vs Scheme and Defense Coverage Scheme tables
split every receiver's routes and every defender's coverage snaps into man and zone, by game week,
2014 on (pulled into rhyman7/pff-data on Nov 5, 2026). Quick checks the same day: a defense's man
rate is a steady trait (odd vs even weeks 0.82, season to season 0.56); a receiver's man-minus-zone
yards per route is mostly noise (odd vs even 0.03-0.05; season to season 0.19 for WRs, 0.11 TEs).

Inputs, known before each game (PFF weeks through the week before; lag 2 reported too, since
PFF posts a week's games a few days late):
  D  the defense's man rate (coverage snaps in man / all coverage snaps), this season to date plus
     K_DEF snaps of last season's rate (league average when there is none), minus the average D
     of all defenses that week.
  G  the receiver's man-vs-zone edge: log(man rate per route / zone rate per route), each shrunk
     toward his overall rate by K_REC routes, capped at +/- G_CAP (0 with no PFF history). Rate = yards per route for receiving yards,
     receptions per route for receptions. Routes from earlier seasons weigh PRIOR_W per season back.
Candidates, each a log adjustment on top of the board's defense-vs-position rating r (dvp.py):
  base  projection x exp(b r)                      (the board)
  A     projection x exp(b r + a D)                (man rate alone)
  B     projection x exp(b r + a D + c D G)        (man rate x the receiver's edge)
Coefficients by position group (WR, TE, RB) and stat, least squares on (actual - projection).

Measures, every one of them, set before running (Nov 5, 2026), for receiving yards and receptions:
  1. Chain projections (players_proj.pkl), fitted on 2014-2018: RMSE on 2014-2018 (tuning).
  2. Same fit: RMSE on 2019 on (holdout).
  3. The projections the board shows (New projections for WR and RB, players_ml; chain for TE),
     fitted on 2019-2021: RMSE on 2022 on.
  4. Against the books: 2023-2025 DraftKings / FanDuel main lines (props_blend's priced frames),
     WR / TE / RB. logit(chance) = logit(book's vig-free chance) + k x signal, k fitted on
     2023-2024, Brier on 2025; signal = the candidate's log adjustment (chain fit) vs the base's.
Keep rule (CONVENTIONS.md, Nov 4): a candidate is kept if it is better than the base on all
eight numbers (seven: the books have receptions lines for 2025 only, so measure 4 can't be fitted for
receptions; found on the first run, Nov 5, and marked not scored; it decided nothing) (4 measures x 2 stats) and worse on none. B is kept over A only if it is also better
than A on all eight. Anything mixed goes to Ryan as a trade-off. Standard errors (clustered by
game) are reported but don't decide. Kept means it goes to Ryan with a plan before the board.

Writes experiments/out/coverage_scheme_test.json (model results only; no PFF rows).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import props_blend as PB  # noqa: E402
from nflmodel import config as C, data, dvp, players_ml, roles  # noqa: E402
from props_calibration import load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "coverage_scheme_test.json"
K_DEF = 750.0          # about three games of a defense's coverage snaps
K_REC = 150.0          # routes of shrinkage toward the receiver's overall rate, each side
PRIOR_W = 0.6          # weight of each earlier season's routes
G_CAP = 0.5            # |G| capped: a receiver with near-zero or negative yards per route gave |G| > 10
GROUPS = ("WR", "TE", "RB")
STATS = ("rec_yds", "receptions")
TUNE = list(range(2014, C.HOLDOUT_FROM))
TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "OAK": "LV", "SD": "LAC", "SL": "LA"}
WEEK = {28: 19, 29: 20, 30: 21, 32: 22}


def pff_dir() -> Path:
    for p in (os.environ.get("PFF_DATA_DIR"), HERE.parent / ".pff-data", HERE.parent.parent / "pff-data"):
        if p and (Path(p) / "nfl").exists():
            return Path(p) / "nfl"
    sys.exit("rhyman7/pff-data with nfl/ not found (PFF_DATA_DIR, .pff-data or ../pff-data)")


def load(table: str, cols) -> pd.DataFrame:
    fs = sorted(glob.glob(str(pff_dir() / "*" / f"w*_{table}.csv.gz")))
    d = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in fs], ignore_index=True)
    d["team"] = d.team_name.replace(TEAM)
    d["week"] = d.week.replace(WEEK)
    return d


def defense_rate(lag: int) -> pd.DataFrame:
    """(season, week, opp, D): the defense's man rate before the game, centred on the week."""
    d = load("defense_coverage_scheme", {"season", "week", "team_name", "man_snap_counts_coverage",
                                         "zone_snap_counts_coverage"})
    w = d.groupby(["season", "week", "team"])[["man_snap_counts_coverage", "zone_snap_counts_coverage"]].sum()
    w.columns = ["man", "zone"]
    w = w.reset_index()
    seas = w.groupby(["season", "team"])[["man", "zone"]].sum()
    prev = (seas.man / (seas.man + seas.zone)).rename("prev").reset_index()
    prev["season"] += 1
    lg = w.groupby("season")[["man", "zone"]].sum()
    lg = (lg.man / (lg.man + lg.zone)).rename("lg").reset_index()
    lg["season"] += 1
    rows = []
    for season in range(2014, int(w.season.max()) + 1):
        ws = w[w.season == season]
        for week in range(1, 23):
            past = ws[ws.week <= week - lag].groupby("team")[["man", "zone"]].sum()
            rows.append(pd.DataFrame({"season": season, "week": week, "team": past.index,
                                      "man": past.man.to_numpy(), "zone": past.zone.to_numpy()}))
    cum = pd.concat(rows, ignore_index=True)
    teams = pd.MultiIndex.from_product([range(2014, int(w.season.max()) + 1), range(1, 23), sorted(w.team.unique())],
                                       names=["season", "week", "team"]).to_frame(index=False)
    x = teams.merge(cum, on=["season", "week", "team"], how="left").fillna({"man": 0, "zone": 0})
    x = x.merge(prev, on=["season", "team"], how="left").merge(lg, on="season", how="left")
    x["prior"] = x.prev.fillna(x.lg).fillna(w.man.sum() / (w.man + w.zone).sum())
    x["rate"] = (x.man + K_DEF * x.prior) / (x.man + x.zone + K_DEF)
    x["D"] = x.rate - x.groupby(["season", "week"]).rate.transform("mean")
    return x.rename(columns={"team": "opp"})[["season", "week", "opp", "D"]]


def receiver_edge(lag: int) -> pd.DataFrame:
    """(season, week, pff_id, G_rec_yds, G_receptions) before each week."""
    r = load("receiving_scheme", {"season", "week", "team_name", "player_id", "man_routes", "zone_routes",
                                  "man_yards", "zone_yards", "man_receptions", "zone_receptions"})
    v = ["man_routes", "zone_routes", "man_yards", "zone_yards", "man_receptions", "zone_receptions"]
    w = r.groupby(["player_id", "season", "week"])[v].sum().reset_index()
    seas = w.groupby(["player_id", "season"])[v].sum()
    out = []
    S = int(w.season.max())
    for season in range(2014, S + 1):
        pri = None
        for k in range(1, season - 2014 + 1):
            if season - k not in seas.index.get_level_values(1):
                continue
            s = seas.xs(season - k, level=1) * PRIOR_W ** k
            pri = s if pri is None else pri.add(s, fill_value=0)
        ws = w[w.season == season]
        for week in range(1, 23):
            cur = ws[ws.week <= week - lag].groupby("player_id")[v].sum()
            tot = cur if pri is None else cur.add(pri, fill_value=0)
            if not len(tot):
                continue
            g = pd.DataFrame({"pff_id": tot.index, "season": season, "week": week})
            routes = (tot.man_routes + tot.zone_routes).to_numpy()
            for stat, a, b in (("rec_yds", "man_yards", "zone_yards"), ("receptions", "man_receptions", "zone_receptions")):
                ov = ((tot[a] + tot[b]).to_numpy() + 1e-9) / (routes + 1e-9)
                m = (tot[a].to_numpy() + K_REC * ov) / (tot.man_routes.to_numpy() + K_REC)
                z = (tot[b].to_numpy() + K_REC * ov) / (tot.zone_routes.to_numpy() + K_REC)
                ok = (ov > 0) & (routes > 0) & (m > 0) & (z > 0)
                g["G_" + stat] = np.clip(np.where(ok, np.log(np.where(ok, m / np.where(z > 0, z, 1), 1)), 0.0), -G_CAP, G_CAP)
            out.append(g)
    return pd.concat(out, ignore_index=True)


def ids() -> pd.DataFrame:
    pl = pd.read_parquet(roles._get(f"{roles.REL}/players/players.parquet", C.RAW / "players.parquet"),
                         columns=["gsis_id", "pff_id"]).dropna()
    pl["pff_id"] = pd.to_numeric(pl.pff_id, errors="coerce")
    return pl.dropna().drop_duplicates("gsis_id").rename(columns={"gsis_id": "player_id"})


def attach_cov(p: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    p = p.drop(columns=[c for c in p.columns if c in ("D", "G_rec_yds", "G_receptions", "pff_id")])
    p = p.merge(ids(), on="player_id", how="left")
    p = p.merge(defense_rate(lag), on=["season", "week", "opp"], how="left")
    p = p.merge(receiver_edge(lag), on=["season", "week", "pff_id"], how="left")
    for c in ("D", "G_rec_yds", "G_receptions"):
        p[c] = p[c].fillna(0.0)
    return p


def design(d, rt, grp, stat):
    r = dvp.attach(d, rt, grp, stat).to_numpy(float)
    D = d.D.to_numpy(float)
    return {"base": np.column_stack([r]), "A": np.column_stack([r, D]),
            "B": np.column_stack([r, D, D * d["G_" + stat].to_numpy(float)])}


def fit_apply(d, rt, stat, fit_mask):
    """Each candidate's log adjustment per row, coefficients fitted per group on fit_mask."""
    col = stat + "_proj"
    y, x0 = d[stat].to_numpy(float), d[col].to_numpy(float)
    adj = {k: np.zeros(len(d)) for k in ("base", "A", "B")}
    coefs = {}
    for grp in GROUPS:
        g = (d.group == grp).to_numpy()
        X = design(d[g], rt, grp, stat)
        for k, M in X.items():
            m = fit_mask[g]
            beta = np.linalg.lstsq(x0[g][m, None] * M[m], (y - x0)[g][m], rcond=None)[0]
            adj[k][g] = M @ beta
            coefs[f"{grp}_{k}"] = [round(float(v), 4) for v in beta]
    return adj, coefs


def rmse_block(d, adj, stat, mask):
    y, x0 = d[stat].to_numpy(float), d[stat + "_proj"].to_numpy(float)
    gid = pd.Series(d.game_id.to_numpy()[mask])
    sq = {k: (x0 * np.exp(a) - y)[mask] ** 2 for k, a in adj.items()}
    res = {"n": int(mask.sum()), "rmse_base": round(float(np.sqrt(sq["base"].mean())), 4)}
    for k in ("A", "B"):
        dd = pd.Series(sq[k] - sq["base"])
        res[f"rmse_{k}"] = round(float(np.sqrt(sq[k].mean())), 4)
        res[f"{k}_minus_base_mse"] = round(float(dd.mean()), 4)
        res[f"{k}_se"] = round(float(PB._se_by_game(dd, gid)), 4)
    dd = pd.Series(sq["B"] - sq["A"])
    res["B_minus_A_mse"], res["B_minus_A_se"] = round(float(dd.mean()), 4), round(float(PB._se_by_game(dd, gid)), 4)
    for grp in GROUPS:
        m = (d.group == grp).to_numpy()[mask]
        res[f"{grp}_rmse"] = {k: round(float(np.sqrt(v[m].mean())), 4) for k, v in sq.items()}
    return res


def played(p):
    return p.played.fillna(False).astype(bool) & (p.active == 1) & p.group.isin(GROUPS)


def chain(pall, rt, lag=1):
    p = attach_cov(pall, lag)
    out, fits = {}, {}
    for stat in STATS:
        d = p[played(p) & (p[stat + "_proj"] > 0) & (p.season >= 2014)].reset_index(drop=True)
        tune = d.season.isin(TUNE).to_numpy()
        adj, coefs = fit_apply(d, rt, stat, tune)
        out[stat] = {"coefs": coefs, "tuning": rmse_block(d, adj, stat, tune),
                     "holdout": rmse_block(d, adj, stat, (d.season >= C.HOLDOUT_FROM).to_numpy())}
        fits[stat] = coefs
        print("chain lag", lag, stat, json.dumps(out[stat]["tuning"]), json.dumps(out[stat]["holdout"]), flush=True)
    return out, fits


def shown(pall):
    games = data.load_games()
    R = roles.load_all(games, verbose=False)
    ml = players_ml.questionable(players_ml.walk_forward(players_ml.add_features(pall, R), first=C.HOLDOUT_FROM),
                                 first=C.FIRST_TEST_SEASON)
    ml = ml[ml.season >= C.HOLDOUT_FROM]
    # the board: New projections for WR and RB receiving, chain for TE
    ch = pall[(pall.season >= C.HOLDOUT_FROM) & (pall.group == "TE")]
    p = pd.concat([ml[ml.group.isin(["WR", "RB"])], ch], ignore_index=True)
    rt = dvp.ratings(p)
    p = attach_cov(p, 1)
    out = {}
    for stat in STATS:
        d = p[played(p) & (p[stat + "_proj"] > 0)].reset_index(drop=True)
        adj, coefs = fit_apply(d, rt, stat, d.season.between(C.HOLDOUT_FROM, 2021).to_numpy())
        out[stat] = {"coefs": coefs, "2022_on": rmse_block(d, adj, stat, (d.season >= 2022).to_numpy())}
        print("shown", stat, json.dumps(out[stat]["2022_on"]), flush=True)
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_k(z0, s, y):
    k = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-(z0 + k * s)))
        g, h = float((s * (y - p)).sum()), float((s * s * p * (1 - p)).sum())
        if h <= 0:
            break
        k += g / h
    return k


def books(pall, rt, fits):
    priced = pd.read_pickle(C.RAW / "props_blend_priced.pkl")
    p = attach_cov(pall, 1)
    keys = p[["game_id", "player_id", "opp", "season", "week", "D", "G_rec_yds", "G_receptions"]]
    out = {}
    for stat in STATS:
        def frame(seasons):
            fl = pd.concat([priced[s][0] for s in seasons], ignore_index=True)
            fl = fl[(fl.stat == stat) & fl.group.isin(GROUPS)]
            fl = fl.drop(columns=[c for c in ("opp", "D", "G_rec_yds", "G_receptions") if c in fl.columns])
            x = PB.frame(fl.merge(keys.drop(columns=["season", "week"]), on=["game_id", "player_id"], how="inner"))
            x = x.reset_index(drop=True)
            for k in ("base", "A", "B"):
                x["s_" + k] = 0.0
            for grp in GROUPS:
                g = (x.group == grp).to_numpy()
                for k, M in design(x[g], rt, grp, stat).items():
                    x.loc[g, "s_" + k] = M @ np.array(fits[stat][f"{grp}_{k}"])
            return x

        tr, te = frame((2023, 2024)), frame((2025,))
        if not len(tr):
            out[stat] = {"not_scored": "no 2023-2024 lines to fit k on (the priced frames have receptions for 2025 only)"}
            print("books", stat, out[stat], flush=True)
            continue
        y = te.hit.to_numpy()
        res = {"n_2025": int(len(te)), "games_2025": int(te.game_id.nunique())}
        br = {}
        for k in ("base", "A", "B"):
            kk = fit_k(logit(tr.mkt.to_numpy()), tr["s_" + k].to_numpy(), tr.hit.to_numpy())
            pr = 1 / (1 + np.exp(-(logit(te.mkt.to_numpy()) + kk * te["s_" + k].to_numpy())))
            br[k] = (pr - y) ** 2
            res[f"k_{k}"] = round(kk, 3)
            res[f"brier_{k}"] = round(float(br[k].mean()), 6)
        res["brier_book"] = round(float(((te.mkt.to_numpy() - y) ** 2).mean()), 6)
        gid = te.game_id.reset_index(drop=True)
        for a, b in (("A", "base"), ("B", "base"), ("B", "A")):
            dd = pd.Series(br[a] - br[b])
            res[f"{a}_minus_{b}"] = {"brier": round(float(dd.mean()), 6), "se": round(PB._se_by_game(dd, gid), 6)}
        out[stat] = res
        print("books", stat, json.dumps(res), flush=True)
    return out


def verdict(res):
    v = {}
    for cand, ref in (("A", "base"), ("B", "base"), ("B", "A")):
        signs = []
        for stat in STATS:
            c = res["chain"][stat]
            if ref == "base":
                signs += [c["tuning"][f"{cand}_minus_base_mse"], c["holdout"][f"{cand}_minus_base_mse"],
                          res["shown"][stat]["2022_on"][f"{cand}_minus_base_mse"]]
            else:
                signs += [c["tuning"]["B_minus_A_mse"], c["holdout"]["B_minus_A_mse"],
                          res["shown"][stat]["2022_on"]["B_minus_A_mse"]]
            if "not_scored" not in res["books"][stat]:
                signs.append(res["books"][stat][f"{cand}_minus_{ref}"]["brier"])
        v[f"{cand}_vs_{ref}"] = {"better": sum(s < 0 for s in signs), "worse": sum(s > 0 for s in signs),
                                 "of": len(signs), "kept": all(s < 0 for s in signs)}
    return v


def main():
    pall = load_proj(None)
    rt = dvp.ratings(pall)
    res = {}
    res["chain"], fits = chain(pall, rt, 1)
    res["chain_lag2"], _ = chain(pall, rt, 2)
    res["books"] = books(pall, rt, fits)
    res["shown"] = shown(pall)
    res["verdict"] = verdict(res)
    print(json.dumps(res["verdict"], indent=1))
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
