"""Whole-roster availability weighted by PFF grades, tested walk-forward (Nov 5, 2026).

    python experiments/pff_availability_test.py

THE RULE (written before the first run, Nov 5, 2026)
-----------------------------------------------------
The game model prices the starting quarterback (EPA rating with the PFF prior), offensive-line
starters missing beyond recent games (d_chg_ol) and skill-position depth-chart starters out
(d_depth1_out). Everyone else's absence reaches it only through later lines. Earlier tests
counted missing players by position group without weighing how good they are (FEATURES.md
"Injuries beyond the quarterback"). This test weighs each missing player by his PFF grade.

  player value   PFF overall grade (offense_summary grades_offense for OL and skill players;
                 defense_summary grades_defense for the front and the secondary), decayed by
                 time (half-life 20 weeks) and weighted by snaps over his earlier games on any
                 team, shrunk toward the replacement level REP with 300 snaps; value = max(grade
                 - REP, 0) / 10. Quarterbacks, specialists and players out of their side's table
                 are left out.
  usual share    his share of his team's snaps on his side of the ball, decayed over the team's
                 previous 8 PFF games (weights 0.5^(k/4), k = 0 the last game; a game he missed
                 counts 0, so a long absence the ratings have already seen fades out).
  out            known before kickoff: Out or Doubtful on that week's injury report for the team,
                 or on the team's roster that week on a reserve list (RES, PUP, NWT, SUS, RSN, RSR,
                 EXE). Game-day inactives (INA) and released or traded players are not counted.
  missing value  per team-game and unit: sum over the usual players who are out of usual share x
                 value. Units: OL (T, G, C), skill (WR, TE, HB, FB), front (DI, ED, LB), secondary
                 (CB, S). Offense = OL + skill, defense = front + secondary.
  features       margin: home minus away missing value; total: both teams' sum.

Versions (all added to the production margin and total features):
  A  offense/defense split, REP 60: margin d_av_off, d_av_def; total s_av_off, s_av_def
  B  four units, REP 60: margin d_av_ol, d_av_skill, d_av_front, d_av_sec; total s_av_off, s_av_def
  C  as A, REP 50
  D  as A, REP 70
  E  as A, replacing d_chg_ol and d_depth1_out in the margin model

Data: PFF NFL offense_summary and defense_summary (rhyman7/pff-data, 2014 on; players linked to
rosters by nflverse's weekly-roster pff_id), nflverse injury reports and weekly rosters 2014 on.
Games before a team has PFF history (2014 week 1) get 0.

Scored on the raw model's average miss (walk-forward refits, as experiments/qb_test.py): margin
and total on the tuning seasons 2015-2018 (2014 is the PFF burn-in) and the holdout 2019 on,
four measures, the current model scored on the same games.

KEEP RULE (CONVENTIONS.md, Nov 4, 2026): kept only if better on all four measures and worse on
none. The version is chosen on the tuning seasons alone: among versions better than the current
model on both tuning measures, the lowest tuning margin miss; its holdout numbers decide. A mixed
result goes to Ryan as a trade-off. Paired standard errors and the season-by-season split are
reported, not used.
Also reported, not used: the spread record at the opening line when the model disagrees by 2+
(backtest.opening_report on this frame). Caveat: availability here is Friday's knowledge, which
the opener doesn't have, so that record flatters any availability input.

Props check: not applicable. The player projections don't read the game model's features.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, data, extra, injuries, model, pff_qb  # noqa: E402
from nflmodel.data import _download  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402
from nflmodel.ratings import _time  # noqa: E402

warnings.filterwarnings("ignore")
TUNE = (2015, 2018)
FIRST = 2014
PLAYOFF_WEEK = {28: 19, 29: 20, 30: 21, 32: 22}
UNITS = {"ol": {"T", "G", "C"}, "skill": {"WR", "TE", "HB", "FB"}, "front": {"DI", "ED", "LB"}, "sec": {"CB", "S"}}
SIDE = {"ol": "off", "skill": "off", "front": "def", "sec": "def"}
OUT_STATUS = {"RES", "PUP", "NWT", "SUS", "RSN", "RSR", "EXE"}
GRADE_HL_WEEKS = 20.0
SHRINK_SNAPS = 300.0
N_GAMES, SHARE_HL = 8, 4.0
NV = "https://github.com/nflverse/nflverse-data/releases/download/{kind}/{f}"
CACHE = C.RAW / "availability"


def _nv(kind: str, f: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f
    if not p.exists():
        _download(NV.format(kind=kind, f=f), p)
    return pd.read_parquet(p)


def pff_player_games(last: int) -> pd.DataFrame:
    """One row per player-game: season, order, pff team_name, unit, snaps, grade."""
    from cfbmodel import pff_data
    parts = []
    for table, side, gcol, scol in (("offense_summary", "off", "grades_offense", "snap_counts_total"),
                                     ("defense_summary", "def", "grades_defense", "snap_counts_defense")):
        p = pff_data.load(table, range(FIRST, last + 1), league="nfl")
        p = p[["season", "week", "player_id", "team_name", "position", gcol, scol]].rename(columns={gcol: "grade", scol: "snaps"})
        p["unit"] = None
        for u, pos in UNITS.items():
            if SIDE[u] == side:
                p.loc[p["position"].isin(pos), "unit"] = u
        parts.append(p[p["unit"].notna()])
    p = pd.concat(parts, ignore_index=True)
    for c in ("grade", "snaps"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["snaps"])
    p = p[p["snaps"] > 0]
    p["week"] = p["week"].astype(int).map(lambda w: PLAYOFF_WEEK.get(w, w))
    p = p[p["week"].between(1, 22)]
    p["season"] = p["season"].astype(int)
    p["order"] = p["season"] * 100 + p["week"]
    p["side"] = p["unit"].map(SIDE)
    return p.reset_index(drop=True)


def rosters_and_injuries(last: int):
    ros, inj = [], []
    for s in range(FIRST, last + 1):
        r = _nv("weekly_rosters", f"roster_weekly_{s}.parquet")[["season", "week", "team", "status", "gsis_id", "pff_id"]]
        ros.append(r)
        i = _nv("injuries", f"injuries_{s}.parquet")[["season", "week", "team", "gsis_id", "report_status"]]
        inj.append(i)
    ros = pd.concat(ros, ignore_index=True)
    inj = pd.concat(inj, ignore_index=True)
    for d in (ros, inj):
        d["team"] = d["team"].replace(C.TEAM_ALIASES)
        d["season"] = d["season"].astype(int)
        d["week"] = d["week"].astype(int)
    ros["pff_id"] = pd.to_numeric(ros["pff_id"], errors="coerce")
    return ros, inj


def team_map(p: pd.DataFrame, ros: pd.DataFrame) -> dict:
    """(season, PFF team_name) -> nflverse team, by majority of linked roster rows."""
    r = ros.dropna(subset=["pff_id"])[["season", "week", "team", "pff_id"]].drop_duplicates()
    m = p[["season", "week", "team_name", "player_id"]].merge(r, left_on=["season", "week", "player_id"],
                                                             right_on=["season", "week", "pff_id"], how="inner")
    v = m.groupby(["season", "team_name", "team"]).size().reset_index(name="n").sort_values("n")
    return {(s, t): nv for s, t, nv in v.drop_duplicates(["season", "team_name"], keep="last")[["season", "team_name", "team"]].itertuples(index=False)}


def availability(games: pd.DataFrame, p: pd.DataFrame, ros: pd.DataFrame, inj: pd.DataFrame, rep: float) -> pd.DataFrame:
    """Missing value per team-game and unit."""
    tm = team_map(p, ros)
    p = p.copy()
    p["team"] = [tm.get((s, t)) for s, t in zip(p["season"], p["team_name"])]
    p = p.dropna(subset=["team"])
    side_snaps = p.groupby(["team", "order", "side"])["snaps"].transform("max")
    p["share"] = p["snaps"] / side_snaps
    # player grade as of any order: the latest state before it
    pv = p.dropna(subset=["grade"]).sort_values(["player_id", "order"])
    pv["t"] = _time(pv["order"].to_numpy()).astype(float)
    # running decayed sums per player (inclusive), to look up "before order X"
    pv["ws"] = 0.0
    pv["wg"] = 0.0
    out_ws, out_wg = np.empty(len(pv)), np.empty(len(pv))
    pid, t, sn, gr = pv["player_id"].to_numpy(), pv["t"].to_numpy(), pv["snaps"].to_numpy(float), pv["grade"].to_numpy(float)
    ws = wg = 0.0
    for k in range(len(pv)):
        if k == 0 or pid[k] != pid[k - 1]:
            ws = wg = 0.0
        else:
            dec = 0.5 ** ((t[k] - t[k - 1]) / GRADE_HL_WEEKS)
            ws, wg = ws * dec, wg * dec
        ws, wg = ws + sn[k], wg + sn[k] * gr[k]
        out_ws[k], out_wg[k] = ws, wg
    pv["ws"], pv["wg"] = out_ws, out_wg
    state = pv[["player_id", "order", "t", "ws", "wg", "unit"]]

    # team game sequences (PFF games), per side
    tg = p.groupby(["team", "order"]).size().reset_index()[["team", "order"]].sort_values(["team", "order"])
    seq = {tm_: g["order"].to_numpy() for tm_, g in tg.groupby("team")}
    share = p.set_index(["team", "order", "player_id"])["share"]
    unit_of = p.drop_duplicates(["team", "player_id"], keep="last").set_index(["team", "player_id"])["unit"]

    # out sets: (team, season, week) -> set of pff ids
    gs2pff = ros.dropna(subset=["pff_id"]).drop_duplicates(["season", "gsis_id"]).set_index(["season", "gsis_id"])["pff_id"]
    rs = ros[ros["status"].isin(OUT_STATUS)].dropna(subset=["pff_id"])
    out = {}
    for (tm_, s, w), g in rs.groupby(["team", "season", "week"]):
        out.setdefault((tm_, s, w), set()).update(g["pff_id"].astype(int))
    ij = inj[inj["report_status"].isin(["Out", "Doubtful"])]
    for (tm_, s, w), g in ij.groupby(["team", "season", "week"]):
        ids = [gs2pff.get((s, x)) for x in g["gsis_id"]]
        out.setdefault((tm_, s, w), set()).update(int(x) for x in ids if x is not None and not pd.isna(x))

    sched = games[games["season"] >= FIRST][["game_id", "season", "week", "home_team", "away_team"]]
    long = pd.concat([sched.rename(columns={"home_team": "team"})[["game_id", "season", "week", "team"]],
                      sched.rename(columns={"away_team": "team"})[["game_id", "season", "week", "team"]]])
    long["order"] = long["season"] * 100 + long["week"]
    st_by_pid = {k: g for k, g in state.groupby("player_id")}
    rows = []
    for r in long.itertuples(index=False):
        row = {"game_id": r.game_id, "team": r.team, **{f"av_{u}": 0.0 for u in UNITS}}
        o_all = seq.get(r.team)
        if o_all is None:
            rows.append(row)
            continue
        prev = o_all[o_all < r.order][-N_GAMES:][::-1]
        ids_out = out.get((r.team, r.season, r.week), set())
        if len(prev) == 0 or not ids_out:
            rows.append(row)
            continue
        wts = 0.5 ** (np.arange(len(prev)) / SHARE_HL)
        for pid_ in ids_out:
            u = unit_of.get((r.team, pid_))
            if u is None:
                continue
            sh = np.array([share.get((r.team, o, pid_), 0.0) for o in prev])
            us = float((wts * sh).sum() / wts.sum())
            if us <= 0:
                continue
            stt = st_by_pid.get(pid_)
            if stt is None:
                continue
            b = stt[stt["order"] < r.order]
            if b.empty:
                g_ = rep
            else:
                last = b.iloc[-1]
                dec = 0.5 ** ((_time(r.order) - last["t"]) / GRADE_HL_WEEKS)
                g_ = (last["wg"] * dec + SHRINK_SNAPS * rep) / (last["ws"] * dec + SHRINK_SNAPS)
            row[f"av_{u}"] += us * max(g_ - rep, 0.0) / 10.0
        rows.append(row)
    a = pd.DataFrame(rows)
    a["av_off"] = a["av_ol"] + a["av_skill"]
    a["av_def"] = a["av_front"] + a["av_sec"]
    return a


def add_features(frame: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    cols = ["av_ol", "av_skill", "av_front", "av_sec", "av_off", "av_def"]
    h = a.rename(columns={"team": "home_team", **{c: f"h_{c}" for c in cols}})
    w = a.rename(columns={"team": "away_team", **{c: f"a_{c}" for c in cols}})
    f = f.merge(h, on=["game_id", "home_team"], how="left").merge(w, on=["game_id", "away_team"], how="left")
    for c in cols:
        f[f"h_{c}"] = f[f"h_{c}"].fillna(0.0)
        f[f"a_{c}"] = f[f"a_{c}"].fillna(0.0)
        f[f"d_{c}"] = f[f"h_{c}"] - f[f"a_{c}"]
        f[f"s_{c}"] = f[f"h_{c}"] + f[f"a_{c}"]
    return f


def measures(out):
    o = out[out.played & out.spread_line.notna()].copy()
    o["em"] = np.abs(o.model_margin - o.result)
    o["et"] = np.abs(o.model_total - o.total)
    res = {}
    for name, x in (("tune", o[o.season.between(*TUNE)]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        res[name] = {"margin": x.em.mean(), "total": x.et.mean(), "n": len(x)}
    return res, o[["game_id", "season", "em", "et"]].reset_index(drop=True)


def main():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    last = int(games.season.max())
    inj_old = injuries.load_injuries(games, verbose=False)
    pr = pff_qb.load(qb, sorted(games.season.unique()))
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=pr)
    base = model.build_frame(games, eng, injuries=inj_old)
    # the production frame: role signals attached as pipeline.py does (d_depth1_out, d_snap_drop)
    from nflmodel import game_roles, players, players_data, roles
    first, _ = model.walk_forward(base)
    pg, pinfo = players_data.load_player_games(verbose=False)
    pav = players.load_availability(verbose=False)
    R = roles.load_all(games, verbose=False)
    prow = players.player_rows(players.team_games(first, pg), pg, pav, pinfo)
    base = game_roles.attach(base, game_roles.team_features(prow, R, pav))
    p = pff_player_games(last)
    ros, inj = rosters_and_injuries(last)
    print(f"PFF player-games {len(p)}; out-of-game ids from rosters/reports")

    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    versions = {
        "A off/def REP60": (60, MF + ["d_av_off", "d_av_def"], TF + ["s_av_off", "s_av_def"]),
        "B four units REP60": (60, MF + ["d_av_ol", "d_av_skill", "d_av_front", "d_av_sec"], TF + ["s_av_off", "s_av_def"]),
        "C off/def REP50": (50, MF + ["d_av_off", "d_av_def"], TF + ["s_av_off", "s_av_def"]),
        "D off/def REP70": (70, MF + ["d_av_off", "d_av_def"], TF + ["s_av_off", "s_av_def"]),
        "E off/def REP60, replaces OL+depth": (60, [c for c in MF if c not in ("d_chg_ol", "d_depth1_out")] + ["d_av_off", "d_av_def"],
                                               TF + ["s_av_off", "s_av_def"]),
    }
    avail = {}
    for rep in sorted({v[0] for v in versions.values()}):
        avail[rep] = availability(games, p, ros, inj, rep)
        a = avail[rep]
        nz = (a[["av_off", "av_def"]].sum(axis=1) > 0).mean()
        print(f"REP {rep}: team-games with some missing value {nz:.1%}; mean off {a.av_off.mean():.3f} def {a.av_def.mean():.3f}", flush=True)

    out0, _ = model.walk_forward(base)
    r0, rows0 = measures(out0)
    print("\nversion                              | tune margin  total | hold margin  total")
    print(f"{'current model':37s}| {r0['tune']['margin']:.3f}  {r0['tune']['total']:.3f} | {r0['hold']['margin']:.3f}  {r0['hold']['total']:.3f}"
          f"   n {r0['tune']['n']}/{r0['hold']['n']}")
    res, rows, frames = {}, {}, {}
    for name, (rep, mf, tf) in versions.items():
        fr = add_features(base, avail[rep])
        out, info = model.walk_forward(fr, margin_feats=mf, total_feats=tf)
        r, rr = measures(out)
        res[name], rows[name], frames[name] = r, rr, out
        print(f"{name:37s}| {r['tune']['margin']:.3f}  {r['tune']['total']:.3f} | {r['hold']['margin']:.3f}  {r['hold']['total']:.3f}", flush=True)

    ok = [n for n, r in res.items() if r["tune"]["margin"] < r0["tune"]["margin"] and r["tune"]["total"] < r0["tune"]["total"]]
    print()
    if not ok:
        best = min(res, key=lambda n: res[n]["tune"]["margin"])
        print(f"No version beats the current model on both tuning measures (best tuning margin: {best}). KEEP: no")
        pick = best
    else:
        pick = min(ok, key=lambda n: res[n]["tune"]["margin"])
        print(f"Chosen on tuning: {pick}")
    a, b = rows[pick], rows0
    m = a.merge(b, on=["game_id", "season"], suffixes=("_n", "_b"))
    for per, mask in (("tune", m.season.between(*TUNE)), ("hold", m.season >= C.HOLDOUT_FROM)):
        for col in ("em", "et"):
            d = (m.loc[mask, f"{col}_n"] - m.loc[mask, f"{col}_b"]).to_numpy()
            print(f"  {per} {'margin' if col == 'em' else 'total':6s} {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
    by = m.assign(dm=m.em_n - m.em_b, dt=m.et_n - m.et_b).groupby("season")[["dm", "dt"]].mean()
    print(by.round(4).to_string())
    print("seasons better, margin / total:", int((by.dm < 0).sum()), "/", int((by.dt < 0).sum()), "of", len(by))
    if ok:
        keep = all(res[pick][p_][k] < r0[p_][k] for p_ in ("tune", "hold") for k in ("margin", "total"))
        print(f"\nKEEP: {'yes' if keep else 'no (mixed: goes to Ryan as a trade-off)' if res[pick]['hold']['margin'] < r0['hold']['margin'] or res[pick]['hold']['total'] < r0['hold']['total'] else 'no'}")

    # informational: opener record at 2+ disagreement (Friday availability: flatters the input)
    try:
        from nflmodel import backtest
        op = data.load_openers(refresh=False)
        for name, fr in (("current model", out0), (pick, frames[pick])):
            rep_ = backtest.opening_report(games, fr, op)
            row = next(x for x in rep_["rows"] if x["threshold"] == 2)
            h = row["holdout"]
            print(f"opener 2+ ({name}): all {row['w']}-{row['l']} roi {row['roi']:+.3f}; holdout {h['w']}-{h['l']} roi {h['roi']:+.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"opener report skipped ({exc})")

    outdir = Path(__file__).parent / "out"
    outdir.mkdir(exist_ok=True)
    pd.DataFrame([{"version": "current model", **{f"{p_}_{k}": v for p_ in r0 for k, v in r0[p_].items()}}] +
                 [{"version": n, **{f"{p_}_{k}": v for p_ in r for k, v in r[p_].items()}} for n, r in res.items()]).round(4).to_csv(outdir / "pff_availability.csv", index=False)


if __name__ == "__main__":
    main()
