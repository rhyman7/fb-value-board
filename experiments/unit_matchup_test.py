"""Projected unit matchup gaps from PFF grades, known at bet time (NFL game model). Nov 13, 2026.

    PFF_DATA_DIR=../pff-data python experiments/unit_matchup_test.py

THE RULE (written before the first run, Nov 13, 2026; Project doc
claude/plan-unit-matchup-test-2026-11-13.md)
-------------------------------------------------------------------
Ryan's question: the private unit tracker now sets each offensive unit against the defensive
unit it faces this week. Should the game model use that? Earlier tests added how much value was
missing (FEATURES.md "Whole-roster availability weighted by PFF grades", Nov 5: no) and a flag
for two linemen out (Nov 11: no). Neither set a unit against the unit across from it, and
neither used the unit's level, only what was missing. This test does both.

  player rating  as the tracker (pff-data unit_tracker.py): snap-weighted PFF grade over his
                 earlier games on any team, 20-week half-life, shrunk toward 55 with 200 snaps.
                 OL (T, G, C) and backs (HB, FB): offense grade; WR/TE: route grade; DL (DI, ED)
                 and LB: defense grade; CB/S: coverage grade.
  usual lineup   snap shares within the unit over the team's previous 8 PFF games, any season
                 (weights 0.5^(k/4), k = 0 the last game; a missed game counts 0), summing to 1.
  out            known before kickoff, by nflverse id: Out or Doubtful on that week's injury
                 report, or on a reserve list on that week's roster (RES, PUP, NWT, SUS, RSN,
                 RSR, EXE). Game-day inactives, releases and trades are not counted.
  projected      the usual lineup with the players who are out removed; their share goes to the
                 unit's players outside its regulars (usual top N: OL 5, WR/TE 4, backs 1, DL 4,
                 LB 2, DB 4) in proportion to their usual share, or to everyone left if there
                 are none.
  unit rating    share-weighted player ratings at that week, minus the mean of the same unit
                 over every team playing that season-week (so 0 is a league-average unit, and a
                 team with no PFF history, 2014 week 1 and before, gets 0).
  gap            offense unit minus the unit it faces: line v defensive line, receivers v
                 secondary, backs v linebackers. Per game: home offense's gap gh and away
                 offense's gap ga. Margin feature gh - ga; total feature gh + ga.

Versions (added to the production margin and total features):
  A  three gaps at projected ratings: margin d_mg_ol, d_mg_rec, d_mg_rb; total s_mg_ol, s_mg_rec, s_mg_rb
  B  one gap, the mean of the three, at projected ratings: d_mg, s_mg
  C  the shift only (projected gap minus usual gap, what this week's absences move): three pairs d_ms_*, s_ms_*
  D  three gaps at usual ratings (no injury information): d_mu_*, s_mu_*

Measures: raw model average miss (walk-forward refits), margin and total, tuning 2015-2018
(2014 is the PFF burn-in) and holdout 2019 on; four measures, the current model on the same games.
Chosen on tuning alone: among versions better on both tuning measures, the lowest tuning margin.
KEEP (CONVENTIONS.md, Nov 11, 2026): better on all four and the holdout improvement at least 1
paired standard error on both margin and total; provisional, rechecked on 2026 in January. Mixed,
or better everywhere but under 1 SE: a trade-off for Ryan. No version better on both tuning
measures: nothing carried to the holdout, nothing kept.
Reported, not used: paired differences with standard errors, the season split, each feature's
spread, and how much of version B's margin feature the closing spread already explains.

Props check (reported; the projections don't read game-model features): QB passing-yards main
lines at DK/FD 2023-2025, pricing snapshot, current projections, split into thirds by the
offense's projected receivers-v-secondary gap: lines, over rate, Brier books and model, average
actual minus line. A bias of 2+ SE in the top or bottom third becomes its own pre-set test.

No per-game file is written: the unit ratings are built from PFF rows and stay out of this repo.
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
UNITS = {"OL": ({"T", "G", "C"}, "offense_summary", "grades_offense", "snap_counts_total"),
         "REC": ({"WR", "TE"}, "offense_summary", "grades_pass_route", "snap_counts_pass_route"),
         "RB": ({"HB", "FB"}, "offense_summary", "grades_offense", "snap_counts_total"),
         "DL": ({"DI", "ED"}, "defense_summary", "grades_defense", "snap_counts_defense"),
         "LB": ({"LB"}, "defense_summary", "grades_defense", "snap_counts_defense"),
         "DB": ({"CB", "S"}, "defense_summary", "grades_coverage_defense", "snap_counts_coverage")}
NREG = {"OL": 5, "REC": 4, "RB": 1, "DL": 4, "LB": 2, "DB": 4}
PAIRS = {"ol": ("OL", "DL"), "rec": ("REC", "DB"), "rb": ("RB", "LB")}
HL, K, BASE = 20.0, 200.0, 55.0
N_GAMES, SHARE_HL = 8, 4.0


def player_games(last: int) -> pd.DataFrame:
    """One row per player-game and unit: season, week, order, t, team_name, pid (nflverse id), grade, snaps."""
    from cfbmodel import pff_data
    from nflmodel.pff_routes import pff_ids
    src = {t: pff_data.load(t, range(AV.FIRST, last + 1), league="nfl") for t in ("offense_summary", "defense_summary")}
    parts = []
    for u, (pos, table, g, s) in UNITS.items():
        d = src[table]
        d = d[d["position"].isin(pos)][["season", "week", "player_id", "team_name", g, s]].copy()
        d.columns = ["season", "week", "player_id", "team_name", "grade", "snaps"]
        d["unit"] = u
        parts.append(d)
    p = pd.concat(parts, ignore_index=True)
    for c in ("grade", "snaps"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["grade", "snaps"])
    p = p[p["snaps"] > 0]
    p["week"] = p["week"].astype(int).map(lambda w: AV.PLAYOFF_WEEK.get(w, w))
    p = p[p["week"].between(1, 22)]
    p["season"] = p["season"].astype(int)
    p["order"] = p["season"] * 100 + p["week"]
    p["t"] = p["season"] * 40 + p["week"]
    p = p.merge(pff_ids(), left_on="player_id", right_on="pff_id", how="inner").drop(columns=["pff_id"])
    p["pid"] = p["nfl_id"]
    return p.reset_index(drop=True)


def unit_ratings(games: pd.DataFrame, p: pd.DataFrame, ros: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """Per team-game and unit: usual and projected rating (raw), and regulars out."""
    tm = AV.team_map(p, ros)
    p = p.copy()
    p["team"] = [tm.get((s, t)) for s, t in zip(p["season"], p["team_name"])]
    p = p.dropna(subset=["team"])
    hist = {k: (d["t"].to_numpy(), d["grade"].to_numpy(), d["snaps"].to_numpy()) for k, d in p.sort_values("t").groupby(["unit", "pid"])}

    def rating(u, pid, t):
        tt, g, s = hist[(u, pid)]
        m = tt < t
        if not m.any():
            return BASE
        w = s[m] * 0.5 ** ((t - tt[m]) / HL)
        return float((np.sum(w * g[m]) + K * BASE) / (np.sum(w) + K))

    share, seq = {}, {}
    for (team, u), d in p.groupby(["team", "unit"]):
        piv = d.pivot_table(index="pid", columns="order", values="snaps", aggfunc="sum").fillna(0.0)
        share[(team, u)] = piv / piv.sum(axis=0)
        seq[(team, u)] = np.array(sorted(piv.columns))
    out = {}
    rs = ros[ros["status"].isin(AV.OUT_STATUS)].dropna(subset=["gsis_id"])
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
        ids = out.get((r.team, r.season, r.week), set())
        order, t = r.season * 100 + r.week, r.season * 40 + r.week
        for u in UNITS:
            o_all = seq.get((r.team, u))
            if o_all is None:
                continue
            prev = o_all[o_all < order][-N_GAMES:][::-1]
            if len(prev) == 0:
                continue
            w = 0.5 ** (np.arange(len(prev)) / SHARE_HL)
            usual = (share[(r.team, u)][list(prev)] * w).sum(axis=1)
            usual = usual[usual > 0]
            usual = usual / usual.sum()
            rt = {q: rating(u, q, t) for q in usual.index}
            ur = sum(usual[q] * rt[q] for q in usual.index)
            reg = list(usual.sort_values(ascending=False).head(NREG[u]).index)
            gone = [q for q in usual.index if q in ids]
            proj = usual
            if gone:
                proj = usual.copy()
                lost = float(proj[gone].sum())
                proj[gone] = 0.0
                depth = [q for q in proj.index if q not in reg and q not in ids]
                pool = depth if depth and proj[depth].sum() > 0 else [q for q in proj.index if q not in ids]
                if pool and proj[pool].sum() > 0:
                    proj[pool] += lost * proj[pool] / proj[pool].sum()
                    proj = proj / proj.sum()
                else:
                    proj = usual
            pr = sum(proj[q] * rt[q] for q in proj.index)
            rows.append({"game_id": r.game_id, "season": r.season, "week": r.week, "team": r.team, "unit": u,
                         "usual": ur, "proj": pr, "n_out": sum(1 for q in reg if q in ids)})
    a = pd.DataFrame(rows)
    for c in ("usual", "proj"):                            # 0 = a league-average unit that week
        a[c] = a[c] - a.groupby(["season", "week", "unit"])[c].transform("mean")
    return a


def add(frame: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    z = {c: a.pivot_table(index=["game_id", "team"], columns="unit", values=c) for c in ("usual", "proj")}

    def val(c, gid, team, u):
        try:
            v = z[c].at[(gid, team), u]
        except KeyError:
            return 0.0
        return 0.0 if pd.isna(v) else float(v)

    new = {}
    for key, (off, de) in PAIRS.items():
        for tag, c in (("mg", "proj"), ("mu", "usual")):
            gh = np.array([val(c, g, h, off) - val(c, g, w, de) for g, h, w in zip(f.game_id, f.home_team, f.away_team)])
            ga = np.array([val(c, g, w, off) - val(c, g, h, de) for g, h, w in zip(f.game_id, f.home_team, f.away_team)])
            new[f"h_{tag}_{key}"], new[f"a_{tag}_{key}"] = gh, ga
            new[f"d_{tag}_{key}"], new[f"s_{tag}_{key}"] = gh - ga, gh + ga
        new[f"h_ms_{key}"], new[f"a_ms_{key}"] = new[f"h_mg_{key}"] - new[f"h_mu_{key}"], new[f"a_mg_{key}"] - new[f"a_mu_{key}"]
        new[f"d_ms_{key}"], new[f"s_ms_{key}"] = new[f"d_mg_{key}"] - new[f"d_mu_{key}"], new[f"s_mg_{key}"] - new[f"s_mu_{key}"]
    new["d_mg"] = np.mean([new[f"d_mg_{k}"] for k in PAIRS], axis=0)
    new["s_mg"] = np.mean([new[f"s_mg_{k}"] for k in PAIRS], axis=0)
    return pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)


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
    p = player_games(last)
    ros, inj = AV.rosters_and_injuries(last)
    a = unit_ratings(games, p, ros, inj)
    x = a[a.season >= 2015]
    print(f"team-game units 2015 on: {len(x)}; spread of the unit rating (projected) by unit: "
          + ", ".join(f"{u} {x[x.unit == u].proj.std():.2f}" for u in UNITS))
    print("regulars out per team-game unit (0/1/2+): " + ", ".join(
        f"{u} {int((x[x.unit == u].n_out == 0).sum())}/{int((x[x.unit == u].n_out == 1).sum())}/{int((x[x.unit == u].n_out >= 2).sum())}" for u in UNITS))
    fr = add(base, a)
    j = fr[(fr.season >= 2015) & fr.spread_line.notna()]
    feats = [f"d_{t}_{k}" for t in ("mg", "ms", "mu") for k in PAIRS] + ["d_mg"]
    print("feature spread (SD, 2015 on): " + ", ".join(f"{c} {j[c].std():.2f}" for c in feats))
    r2 = np.corrcoef(j["d_mg"], j["spread_line"])[0, 1] ** 2
    print(f"closing spread explains {r2:.1%} of version B's margin feature (d_mg), 2015 on")

    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    three = lambda tag: ([f"d_{tag}_{k}" for k in PAIRS], [f"s_{tag}_{k}" for k in PAIRS])  # noqa: E731
    versions = {"A three gaps, projected": (MF + three("mg")[0], TF + three("mg")[1]),
                "B one gap, projected": (MF + ["d_mg"], TF + ["s_mg"]),
                "C shift only": (MF + three("ms")[0], TF + three("ms")[1]),
                "D three gaps, usual": (MF + three("mu")[0], TF + three("mu")[1])}
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
    g = pd.concat([fr[["game_id", "home_team", "h_mg_rec"]].rename(columns={"home_team": "team", "h_mg_rec": "gap"}),
                   fr[["game_id", "away_team", "a_mg_rec"]].rename(columns={"away_team": "team", "a_mg_rec": "gap"})])
    props(g)


def props(g: pd.DataFrame):
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
        j = pd.concat(parts, ignore_index=True).merge(g, on=["game_id", "team"], how="inner")
        j = j[j.y != j.line]
        qg = j.drop_duplicates(["game_id", "player_id"])
        lo, hi = qg["gap"].quantile([1 / 3, 2 / 3])
        for lab, d in ((f"bottom third (gap under {lo:+.2f})", j[j.gap < lo]), ("middle third", j[(j.gap >= lo) & (j.gap <= hi)]),
                       (f"top third (gap over {hi:+.2f})", j[j.gap > hi])):
            if d.empty:
                print(f"props {lab}: no lines")
                continue
            hit = (d.y > d.line).astype(float)
            bm = ((d.fair_over - hit) ** 2).mean()
            bo = ((d.our_over / (d.our_over + d.our_under) - hit) ** 2).mean()
            q = d.drop_duplicates(["game_id", "player_id"])
            bias = f"actual - line {np.mean(q.y - q.line):+.1f} (SE {np.std(q.y - q.line, ddof=1) / np.sqrt(len(q)):.1f})"
            print(f"props {lab}: lines {len(d)} ({len(q)} QB-games) over rate {hit.mean():.3f} Brier books {bm:.4f} model {bo:.4f}; {bias}")
    except Exception as exc:  # noqa: BLE001
        print(f"props check failed: {exc!r}")


if __name__ == "__main__":
    main()
