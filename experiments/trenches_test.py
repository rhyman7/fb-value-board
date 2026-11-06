"""The trenches matchup from PFF: pass pressure allowed against pass-rush win rate (Nov 6, 2026).

    python experiments/trenches_test.py [--skip-props]

THE RULE (written before the first run, Nov 6, 2026)
-----------------------------------------------------
Sack rate was tested and dropped (FEATURES.md), but sacks are the unstable end of a stable process:
in PFF's data pressure rate and pass-rush win rate hold year to year far better than sack rate.
This test builds, per team and as of each game from its earlier games (decayed, half-life 8
games, last season at half weight), from PFF NFL tables (rhyman7/pff-data, 2014 on):

  OP  offense pressure allowed: pressured dropbacks / dropbacks (passing_pressure, QB rows)
  DW  defense pass-rush win rate: pass-rush wins / pass-rush opportunities (defense_pass_rush)
  PS  offense pressure-to-sack: sacks when pressured / pressured dropbacks (passing_pressure)
Each centred on its league average; teams with no history 0.

Matchup for each offense: E (expected pressure) = OP(offense) + DW(opposing defense);
S (expected sacks) = E x (1 + PS(offense) / league PS).
Features: margin d_E = E(away offense) - E(home offense), total s_E = E(home) + E(away);
likewise d_S, s_S.

Versions, each added to the production NFL model (QB prior and role signals included):
  A  pressure: margin + d_E, total + s_E
  B  sacks:    margin + d_S, total + s_S
  C  A + B

Measures: raw model average miss, margin and total, tuning 2015-2018 (2014 is the PFF burn-in)
and holdout 2019 on (4), plus a props check (1): the QB passing-yards projection adjusted by
beta x (the QB's offense's matchup term for the chosen version: E for A, S for B, both for C),
beta fitted walk-forward on earlier seasons' residuals of starting QBs, scored by Brier at the
real 2023-2025 DraftKings/FanDuel passing-yards main lines (pricing snapshot), pooled.

KEEP RULE (CONVENTIONS.md): the version is chosen on the tuning seasons alone (among versions
better than the current model on both tuning measures, the lowest tuning total miss, since the
trenches bear on totals first); kept only if better on all five measures and worse on none; a
mixed result goes to Ryan. Paired standard errors and season splits reported, not used.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
from cfbmodel import pff_data  # noqa: E402
from nflmodel import config as C, data, extra, injuries, model, pff_qb  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

TUNE = (2015, 2018)
HL, CARRY = 8.0, 0.5
PLAYOFF = {28: 19, 29: 20, 30: 21, 32: 22}


def _load(table):
    p = pff_data.load(table, range(2014, 2027), league="nfl").drop(columns=["team"], errors="ignore")
    p["week"] = p["week"].astype(int).map(lambda w: PLAYOFF.get(w, w))
    p = p[p["week"].between(1, 22)].copy()
    p["season"] = p["season"].astype(int)
    return p


def team_map(qb: pd.DataFrame) -> dict:
    """(season, PFF team_name) -> nflverse team, through quarterbacks' ids."""
    pp = _load("passing_pressure")
    ids = pd.read_parquet(C.RAW / "players.parquet", columns=["gsis_id", "pff_id"]).dropna()
    ids["pff_id"] = pd.to_numeric(ids["pff_id"], errors="coerce")
    pp = pp.merge(ids.dropna().drop_duplicates("pff_id"), left_on="player_id", right_on="pff_id")
    q = qb[["qb_id", "team", "order"]].assign(season=lambda x: x.order // 100, week=lambda x: x.order % 100)
    m = pp.merge(q, left_on=["gsis_id", "season", "week"], right_on=["qb_id", "season", "week"])
    v = m.groupby(["season", "team_name", "team"]).size().rename("n").reset_index().sort_values("n")
    return {(s, t): nv for s, t, nv in v.drop_duplicates(["season", "team_name"], keep="last")[["season", "team_name", "team"]].itertuples(index=False)}


def weekly(qb: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season, week): pressured dropbacks, dropbacks, pressured sacks; pass-rush wins, opps."""
    tm = team_map(qb)
    pp = _load("passing_pressure")
    for c in ("base_dropbacks", "pressure_dropbacks", "pressure_sacks"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce").fillna(0.0)
    pp["team"] = [tm.get((s, t)) for s, t in zip(pp.season, pp.team_name)]
    o = pp.dropna(subset=["team"]).groupby(["team", "season", "week"], as_index=False)[["base_dropbacks", "pressure_dropbacks", "pressure_sacks"]].sum()
    dr = _load("defense_pass_rush")
    for c in ("pass_rush_wins", "pass_rush_opp"):
        dr[c] = pd.to_numeric(dr[c], errors="coerce").fillna(0.0)
    dr["team"] = [tm.get((s, t)) for s, t in zip(dr.season, dr.team_name)]
    d = dr.dropna(subset=["team"]).groupby(["team", "season", "week"], as_index=False)[["pass_rush_wins", "pass_rush_opp"]].sum()
    return o.merge(d, on=["team", "season", "week"], how="outer").fillna(0.0)


def asof(w: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """OP, DW, PS per (game_id, team) from the team's earlier games."""
    lg_op = w.pressure_dropbacks.sum() / w.base_dropbacks.sum()
    lg_dw = w.pass_rush_wins.sum() / w.pass_rush_opp.sum()
    lg_ps = w.pressure_sacks.sum() / w.pressure_dropbacks.sum()
    by = {t: g.sort_values(["season", "week"]).reset_index(drop=True) for t, g in w.groupby("team")}
    long = pd.concat([games[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"}),
                      games[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"})])
    rows = []
    for r in long.itertuples(index=False):
        g = by.get(r.team)
        op = dw = ps = 0.0
        if g is not None:
            prev = g[((g.season < r.season) | ((g.season == r.season) & (g.week < r.week))) & (g.season >= r.season - 1)].iloc[::-1]
            if len(prev):
                k = np.arange(len(prev))
                wt = 0.5 ** (k / HL) * np.where(prev.season.to_numpy() < r.season, CARRY, 1.0)
                sm = lambda c: float((wt * prev[c].to_numpy(float)).sum())
                if sm("base_dropbacks") > 50:
                    op = sm("pressure_dropbacks") / sm("base_dropbacks") - lg_op
                if sm("pass_rush_opp") > 50:
                    dw = sm("pass_rush_wins") / sm("pass_rush_opp") - lg_dw
                if sm("pressure_dropbacks") > 20:
                    ps = sm("pressure_sacks") / sm("pressure_dropbacks") - lg_ps
        rows.append((r.game_id, r.team, op, dw, ps))
    a = pd.DataFrame(rows, columns=["game_id", "team", "OP", "DW", "PS"])
    a.attrs["lg_ps"] = lg_ps
    return a


def add_features(frame: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    lg_ps = a.attrs.get("lg_ps", 0.2)
    for side in ("home", "away"):
        x = a.rename(columns={"team": f"{side}_team", "OP": f"{side}_OP", "DW": f"{side}_DW", "PS": f"{side}_PS"})
        f = f.merge(x, on=["game_id", f"{side}_team"], how="left")
    for c in ("OP", "DW", "PS"):
        f[f"home_{c}"] = f[f"home_{c}"].fillna(0.0)
        f[f"away_{c}"] = f[f"away_{c}"].fillna(0.0)
    f["E_home"] = f["home_OP"] + f["away_DW"]           # pressure on the home offense
    f["E_away"] = f["away_OP"] + f["home_DW"]
    f["S_home"] = f["E_home"] * (1 + f["home_PS"] / lg_ps)
    f["S_away"] = f["E_away"] * (1 + f["away_PS"] / lg_ps)
    f["d_E"], f["s_E"] = f["E_away"] - f["E_home"], f["E_home"] + f["E_away"]
    f["d_S"], f["s_S"] = f["S_away"] - f["S_home"], f["S_home"] + f["S_away"]
    return f


def measures(out):
    o = out[out.played & out.spread_line.notna()].copy()
    o["em"] = np.abs(o.model_margin - o.result)
    o["et"] = np.abs(o.model_total - o.total)
    res = {}
    for name, x in (("tune", o[o.season.between(*TUNE)]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        res[name] = {"margin": x.em.mean(), "total": x.et.mean(), "n": len(x)}
    return res, o[["game_id", "season", "em", "et"]].reset_index(drop=True)


def props_check(fr: pd.DataFrame, terms: list[str]) -> dict:
    """QB passing yards: projection + beta . matchup terms (walk-forward), Brier at the real lines."""
    import props_real_lines as RL
    from nflmodel import propbackfill as B, props as P
    from props_calibration import FAMILY, load_proj
    base = load_proj(None)
    side = fr[["game_id", "home_team", "away_team"] + [f"{t}_home" for t in terms] + [f"{t}_away" for t in terms]]
    h = side.rename(columns={"home_team": "team", **{f"{t}_home": t for t in terms}})[["game_id", "team"] + terms]
    w = side.rename(columns={"away_team": "team", **{f"{t}_away": t for t in terms}})[["game_id", "team"] + terms]
    base = base.merge(pd.concat([h, w]), on=["game_id", "team"], how="left")
    for t in terms:
        base[t] = base[t].fillna(0.0)
    saved = P.PropModel.load()
    out = {"current": [], "new": []}
    for season in (2023, 2024, 2025):
        tr = base[(base.season < season) & (base.season >= 2015) & (base.group == "QB") & base.played.fillna(False).astype(bool)
                  & (base.att_share >= 0.99) & base.pass_yds.notna()]
        X = tr[terms].to_numpy(float)
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), (tr.pass_yds - tr.pass_yds_proj).to_numpy(float), rcond=None)[0]
        adj = base.copy()
        qbm = adj.group == "QB"
        adj.loc[qbm, "pass_yds_proj"] = adj.loc[qbm, "pass_yds_proj"] + adj.loc[qbm, terms].to_numpy(float) @ beta[1:]
        for label, tab in (("current", base), ("new", adj)):
            m = P.PropModel(FAMILY).fit(tab[(tab.season < season) & (tab.season >= 2014)])
            m.iso = saved.iso
            x = RL.priced(tab, m, B.load(season, "flag"), season)
            if not x.empty:
                out[label].append(x[(x.stat == "pass_yds") & (x.group == "QB")])
        print(f"  priced {season}: beta {np.round(beta[1:], 1)}", flush=True)
    key = ["game_id", "player_id", "stat", "line", "book"]
    a = pd.concat(out["current"]).drop_duplicates(key)
    b = pd.concat(out["new"]).drop_duplicates(key)
    a, b = a[a.y != a.line], b[b.y != b.line]
    j = a[key + ["y", "our_over", "our_under", "fair_over"]].merge(b[key + ["our_over", "our_under"]], on=key, suffixes=("_c", "_n"))
    hit = (j.y > j.line).astype(float)
    pc = j.our_over_c / (j.our_over_c + j.our_under_c)
    pn = j.our_over_n / (j.our_over_n + j.our_under_n)
    bc, bn, bm = (pc - hit) ** 2, (pn - hit) ** 2, (j.fair_over - hit) ** 2
    d = (bn - bc).groupby(j.game_id).sum()
    r = {"n": len(j), "books": round(float(bm.mean()), 5), "current": round(float(bc.mean()), 5), "new": round(float(bn.mean()), 5),
         "change": round(float((bn - bc).mean()), 6), "se": round(float(d.std(ddof=1) * np.sqrt(len(d)) / len(j)), 6)}
    print(f"  QB passing yards at the lines: n {r['n']}  books {r['books']:.5f}  current {r['current']:.5f}  new {r['new']:.5f}"
          f"  ({r['change']:+.6f} ± {r['se']:.6f})", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-props", action="store_true")
    args = ap.parse_args()
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    pr = pff_qb.load(qb, sorted(games.season.unique()))
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=pr)
    base = model.build_frame(games, eng, injuries=injuries.load_injuries(games, verbose=False))
    from nflmodel import game_roles, players, players_data, roles
    first, _ = model.walk_forward(base)
    pg, pinfo = players_data.load_player_games(verbose=False)
    pav = players.load_availability(verbose=False)
    R = roles.load_all(games, verbose=False)
    base = game_roles.attach(base, game_roles.team_features(players.player_rows(players.team_games(first, pg), pg, pav, pinfo), R, pav))

    w = weekly(qb)
    a = asof(w, games)
    print(f"team-weeks {len(w)}; as-of rows {len(a)}, with history {(a[['OP','DW','PS']].abs().sum(axis=1) > 0).mean():.1%}")
    fr = add_features(base, a)
    print(fr[fr.season >= 2015][["d_E", "s_E", "d_S", "s_S"]].describe().round(3).loc[["mean", "std"]].to_string())
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    versions = {"A pressure": (MF + ["d_E"], TF + ["s_E"], ["E"]),
                "B sacks": (MF + ["d_S"], TF + ["s_S"], ["S"]),
                "C both": (MF + ["d_E", "d_S"], TF + ["s_E", "s_S"], ["E", "S"])}
    out0, _ = model.walk_forward(fr)
    r0, rows0 = measures(out0)
    print(f"\n{'version':14s}| tune margin  total | hold margin  total")
    print(f"{'current model':14s}| {r0['tune']['margin']:.3f}  {r0['tune']['total']:.3f} | {r0['hold']['margin']:.3f}  {r0['hold']['total']:.3f}")
    res, rows = {}, {}
    for name, (mf, tf, _) in versions.items():
        out, _ = model.walk_forward(fr, margin_feats=mf, total_feats=tf)
        res[name], rows[name] = measures(out)
        r = res[name]
        print(f"{name:14s}| {r['tune']['margin']:.3f}  {r['tune']['total']:.3f} | {r['hold']['margin']:.3f}  {r['hold']['total']:.3f}", flush=True)
    ok = [n for n, r in res.items() if r["tune"]["margin"] < r0["tune"]["margin"] and r["tune"]["total"] < r0["tune"]["total"]]
    pick = min(ok, key=lambda n: res[n]["tune"]["total"]) if ok else min(res, key=lambda n: res[n]["tune"]["total"])
    print(("\nChosen on tuning: " if ok else "\nNo version beats the current model on both tuning measures; closest on total: ") + pick)
    m = rows[pick].merge(rows0, on=["game_id", "season"], suffixes=("_n", "_b"))
    for per, mask in (("tune", m.season.between(*TUNE)), ("hold", m.season >= C.HOLDOUT_FROM)):
        for col in ("em", "et"):
            d = (m.loc[mask, f"{col}_n"] - m.loc[mask, f"{col}_b"]).to_numpy()
            print(f"  {per} {'margin' if col == 'em' else 'total':6s} {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
    by = m.assign(dm=m.em_n - m.em_b, dt=m.et_n - m.et_b).groupby("season")[["dm", "dt"]].mean()
    print(by[by.index >= 2015].round(4).to_string())
    measures_ok = [res[pick][p][k] < r0[p][k] for p in ("tune", "hold") for k in ("margin", "total")]
    if ok and not args.skip_props:
        pc = props_check(fr, versions[pick][2])
        measures_ok.append(pc["change"] < 0)
    verdict = "KEEP" if (ok and all(measures_ok) and len(measures_ok) == 5) else ("not kept" if not ok else "mixed: Ryan's call" if any(measures_ok) else "not kept")
    print(f"\nbetter on {sum(measures_ok)} of {len(measures_ok)} -> {verdict}")


if __name__ == "__main__":
    main()
