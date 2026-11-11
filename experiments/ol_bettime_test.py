"""Two or more offensive-line regulars out, known at bet time (NFL game model). Nov 11, 2026.

    PFF_DATA_DIR=../pff-data python experiments/ol_bettime_test.py

THE RULE (written before the first run, Nov 11, 2026; Project doc
claude/plan-ol-bettime-test-2026-11-11.md)
-------------------------------------------------------------------
Lead: offenses missing two or more line regulars scored 1.4 points under expected and 0.9 under the
closing implied team total (2015-2025, who actually played; one missing lineman did nothing).
d_chg_ol is roughly linear in absences. Does a threshold at two, from pregame knowledge, add to it?

  usual five   the team's five linemen (PFF T, G, C) with the largest usual snap share over its
               previous 8 PFF games (weights 0.5^(k/4); a missed game counts 0)
  out          Out or Doubtful on that week's injury report, or on a reserve list on that week's
               roster (RES, PUP, NWT, SUS, RSN, RSR, EXE). Game-day inactives, releases and trades
               are not counted.
  n_out        how many of the usual five are out

Versions (added to the production margin and total features):
  A  flag n_out >= 2: margin d_ol2 (home minus away), total s_ol2 (sum)
  B  count min(n_out, 3): d_oln, s_oln
  C  as A, with d_chg_ol removed from the margin model

Measures: raw model average miss (walk-forward refits), margin and total, tuning 2015-2018 and
holdout 2019 on; four measures, the current model on the same games.
Chosen on tuning alone: among versions better on both tuning measures, the lowest tuning margin.
KEEP (CONVENTIONS.md, Nov 11, 2026): better on all four and the holdout improvement at least 1
paired standard error on both margin and total; provisional, rechecked on 2026 in January. Mixed,
or better everywhere but under 1 SE: a trade-off for Ryan. No version better on both tuning
measures: nothing carried to the holdout, nothing kept.
Reported, not used: season split, how often the flag fires.

Props check (reported; the projections don't read game-model features): QB passing-yards main
lines at DK/FD 2023-2025, pricing snapshot, current projections, for quarterbacks whose line has
the flag on: lines, over rate, Brier books and model, average actual minus line. Judged only
with 30+ lines; a bias of 2+ SE becomes its own pre-set test. No sacks prop on the board.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pff_availability_test as AV  # noqa: E402
from nflmodel import config as C, data, extra, injuries, model, pff_qb  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
TUNE = AV.TUNE
OUT_STATUS = AV.OUT_STATUS


def ol_out(games, p, ros, inj) -> pd.DataFrame:
    """n_out per team-game: usual five linemen who are out before kickoff."""
    tm = AV.team_map(p, ros)
    p = p[p["unit"] == "ol"].copy()
    p["team"] = [tm.get((s, t)) for s, t in zip(p["season"], p["team_name"])]
    p = p.dropna(subset=["team"])
    p["share"] = p["snaps"] / p.groupby(["team", "order"])["snaps"].transform("max")
    seq = {t: np.sort(g["order"].unique()) for t, g in p.groupby("team")}
    share = {t: g.pivot_table(index="pid", columns="order", values="share", aggfunc="max").fillna(0.0) for t, g in p.groupby("team")}
    out = {}
    rs = ros[ros["status"].isin(OUT_STATUS)].dropna(subset=["gsis_id"])
    for k, g in rs.groupby(["team", "season", "week"]):
        out.setdefault(k, set()).update(g["gsis_id"])
    ij = inj[inj["report_status"].isin(["Out", "Doubtful"])].dropna(subset=["gsis_id"])
    for k, g in ij.groupby(["team", "season", "week"]):
        out.setdefault(k, set()).update(g["gsis_id"])
    sched = games[games["season"] >= AV.FIRST][["game_id", "season", "week", "home_team", "away_team"]]
    long = pd.concat([sched.rename(columns={"home_team": "team"})[["game_id", "season", "week", "team"]],
                      sched.rename(columns={"away_team": "team"})[["game_id", "season", "week", "team"]]])
    rows = []
    for r in long.itertuples(index=False):
        n = 0
        o_all = seq.get(r.team)
        ids = out.get((r.team, r.season, r.week), set())
        if o_all is not None and ids:
            prev = o_all[o_all < r.season * 100 + r.week][-AV.N_GAMES:][::-1]
            if len(prev):
                w = 0.5 ** (np.arange(len(prev)) / AV.SHARE_HL)
                us = (share[r.team][list(prev)] * w).sum(axis=1)
                five = us[us > 0].sort_values(ascending=False).head(5).index
                n = sum(1 for x in five if x in ids)
        rows.append({"game_id": r.game_id, "team": r.team, "n_out": n})
    return pd.DataFrame(rows)


def add(frame, a):
    f = frame.copy()
    a = a.assign(ol2=(a.n_out >= 2).astype(float), oln=a.n_out.clip(upper=3).astype(float))
    for side, col in (("h", "home_team"), ("a", "away_team")):
        f = f.merge(a.rename(columns={"team": col, "ol2": f"{side}_ol2", "oln": f"{side}_oln"})[["game_id", col, f"{side}_ol2", f"{side}_oln"]],
                    on=["game_id", col], how="left")
    for c in ("ol2", "oln"):
        f[f"h_{c}"], f[f"a_{c}"] = f[f"h_{c}"].fillna(0.0), f[f"a_{c}"].fillna(0.0)
        f[f"d_{c}"], f[f"s_{c}"] = f[f"h_{c}"] - f[f"a_{c}"], f[f"h_{c}"] + f[f"a_{c}"]
    return f


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    last = int(games.season.max())
    inj_old = injuries.load_injuries(games, verbose=False)
    pr = pff_qb.load(qb, sorted(games.season.unique()))
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=pr)
    base = model.build_frame(games, eng, injuries=inj_old)
    from nflmodel import game_roles, players, players_data, roles
    first, _ = model.walk_forward(base)
    pg, pinfo = players_data.load_player_games(verbose=False)
    pav = players.load_availability(verbose=False)
    R = roles.load_all(games, verbose=False)
    prow = players.player_rows(players.team_games(first, pg), pg, pav, pinfo)
    base = game_roles.attach(base, game_roles.team_features(prow, R, pav))
    p = AV.pff_player_games(last)
    ros, inj = AV.rosters_and_injuries(last)
    a = ol_out(games, p, ros, inj)
    a.to_csv(Path(__file__).parent / "out" / "ol_bettime_flags.csv", index=False)
    x = a.merge(games[["game_id", "season"]], on="game_id")
    x = x[x.season >= 2015]
    print(f"team-games 2015 on: {len(x)}; n_out 0/1/2/3+: {[(int((x.n_out == k).sum()) if k < 3 else int((x.n_out >= 3).sum())) for k in range(4)]}; flag rate {(x.n_out >= 2).mean():.1%}")

    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    versions = {"A flag 2+": (MF + ["d_ol2"], TF + ["s_ol2"]),
                "B count to 3": (MF + ["d_oln"], TF + ["s_oln"]),
                "C flag 2+, replaces d_chg_ol": ([c for c in MF if c != "d_chg_ol"] + ["d_ol2"], TF + ["s_ol2"])}
    fr = add(base, a)
    out0, _ = model.walk_forward(base)
    r0, rows0 = AV.measures(out0)
    print("\nversion                       | tune margin  total | hold margin  total")
    print(f"{'current model':30s}| {r0['tune']['margin']:.3f}  {r0['tune']['total']:.3f} | {r0['hold']['margin']:.3f}  {r0['hold']['total']:.3f}   n {r0['tune']['n']}/{r0['hold']['n']}")
    res, rows = {}, {}
    for name, (mf, tf) in versions.items():
        out, _ = model.walk_forward(fr, margin_feats=mf, total_feats=tf)
        res[name], rows[name] = AV.measures(out)
        r = res[name]
        print(f"{name:30s}| {r['tune']['margin']:.3f}  {r['tune']['total']:.3f} | {r['hold']['margin']:.3f}  {r['hold']['total']:.3f}", flush=True)
    for name in versions:
        m = rows[name].merge(rows0, on=["game_id", "season"], suffixes=("_n", "_b"))
        s = []
        for per, mask in (("tune", m.season.between(*TUNE)), ("hold", m.season >= C.HOLDOUT_FROM)):
            for col in ("em", "et"):
                d = (m.loc[mask, f"{col}_n"] - m.loc[mask, f"{col}_b"]).to_numpy()
                s.append(f"{per} {'margin' if col == 'em' else 'total'} {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
        by = m.assign(dm=m.em_n - m.em_b, dt=m.et_n - m.et_b).groupby("season")[["dm", "dt"]].mean()
        print(f"{name}: " + "; ".join(s) + f"; seasons better margin/total {int((by.dm < 0).sum())}/{int((by.dt < 0).sum())} of {len(by)}")
    ok = [n for n, r in res.items() if r["tune"]["margin"] < r0["tune"]["margin"] and r["tune"]["total"] < r0["tune"]["total"]]
    if not ok:
        print("\nNo version beats the current model on both tuning measures. KEEP: no")
    else:
        pick = min(ok, key=lambda n: res[n]["tune"]["margin"])
        print(f"\nChosen on tuning: {pick} (judge the holdout lines above against the 1 SE bar)")
    props(a)


def props(a):
    try:
        import props_real_lines as RL
        from backup_qb_pick_test import table
        from nflmodel import players as PL, propbackfill as B, props as P
        from props_calibration import FAMILY
        t = table()
        p = PL.walk_forward(t)
        saved = P.PropModel.load()
        parts = []
        for season in (2023, 2024, 2025):
            m = P.PropModel(FAMILY).fit(p[(p.season < season) & (p.season >= 2014)])
            m.iso = saved.iso
            x = RL.priced(p, m, B.load(season, "flag"), season)
            if not x.empty:
                parts.append(x[(x.stat == "pass_yds") & (x.group == "QB")])
        j = pd.concat(parts, ignore_index=True).merge(a, on=["game_id", "team"], how="left")
        j = j[j.y != j.line]
        for lab, d in (("flag on (2+ out)", j[j.n_out >= 2]), ("1 out", j[j.n_out == 1]), ("0 out", j[j.n_out == 0])):
            if d.empty:
                print(f"props {lab}: no lines")
                continue
            hit = (d.y > d.line).astype(float)
            bm = ((d.fair_over - hit) ** 2).mean()
            bo = ((d.our_over / (d.our_over + d.our_under) - hit) ** 2).mean()
            g = d.drop_duplicates(["game_id", "player_id"])
            bias = f"actual - line {np.mean(g.y - g.line):+.1f} (SE {np.std(g.y - g.line, ddof=1) / np.sqrt(len(g)):.1f})"
            print(f"props {lab}: lines {len(d)} ({len(g)} QB-games) over rate {hit.mean():.3f} Brier books {bm:.4f} model {bo:.4f}; {bias}")
    except Exception as exc:  # noqa: BLE001
        print(f"props check failed: {exc!r}")


if __name__ == "__main__":
    main()
