"""College: returning production from PFF that counts incoming transfers, tested walk-forward (Nov 5, 2026).

    python experiments/cfb_pff_returning_test.py

THE RULE (written before the first run, Nov 5, 2026)
-----------------------------------------------------
The college game model's returning production (CFBD's share of last season's production back,
d_rtprod; s_off_rtprod and s_def_rtprod in the total) counts only players who stayed. Portal
additions are invisible. This test builds a transfer-aware version from PFF:

  player value   last season's PFF snaps x (snap-weighted grade - 60), clipped at 0, from
                 offense_summary (grades_offense) and defense_summary (grades_defense), every
                 position, on whatever team he played for. Divided by 1,000.
  on the roster  this season's CFBD roster (data/cfb/cfbd/roster_{S}.csv). Roster players are
                 matched to last season's PFF players by normalised name; a name shared by several
                 PFF players is resolved by the same team, then by the transfer portal's origin
                 (CFBD portal file, 2021 on), else left out.
  team value     sum over the roster's matched players, offense and defense separately
                 (ret_off, ret_def) and together (ret).
PFF franchises are mapped to the board's team names by shared player names with the CFBD roster
of the same season. Teams without a mapped franchise, or seasons without last season's PFF data
(2017: PFF starts in 2017), take the median.

Features: d_pffret = home minus away (ret); d_pffret_early = d_pffret x early (the board's early
weight, 1 in weeks 0-1 to 0 from week 6); totals s_pffret_off, s_pffret_def (both teams' sums).

Versions (each on top of the production college model, inputs and rated extras included):
  A  margin + d_pffret; total + s_pffret_off, s_pffret_def
  B  margin + d_pffret_early (total unchanged)
  C  A + d_pffret_early
  D  A replacing CFBD returning production (d_rtprod; s_off_rtprod, s_def_rtprod)

Scored as the other college tests: raw model average miss, margin and total, tuning 2017-2019
and holdout 2021 on (FBS games with lines; 2020 in neither). Note: the feature exists from 2018
and the walk-forward model first fits it with 2018 in its training seasons, so on the tuning
seasons only 2019 can move. Weeks 0-5 are reported separately, not used.

KEEP RULE (CONVENTIONS.md, Nov 4, 2026): kept only if better on all four measures (margin and
total, tuning and holdout) and worse on none. The version is chosen on the tuning seasons alone
(among those better than the current model on both tuning measures, the lowest tuning margin
miss); its holdout decides. A mixed result goes to Ryan as a trade-off. Paired standard errors
reported, not used. Props: not applicable (the player projections don't read these features).
Correction after the first run (Nov 5, 2026): the total features were 2 x median in seasons
without data, which kept a rounding-size spread that blew up the 2018 totals fit (+1.11 points);
they are now centred so those rows are exactly 0. Nothing else changed; every version re-run.
Caveat: CFBD rosters are the season's roster as posted, which can include players added after
week 0; that is a small look-ahead, noted, not corrected.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import config as C, data, extra, model, pff_data  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
TUNE = (2017, 2019)
REP = 60.0
CFBD = C.DATA / "cfbd"


def nn(s) -> str:
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower())
    return re.sub(r"[^a-z]", "", s)


def pff_seasons(seasons) -> pd.DataFrame:
    """Player-season: player_id, season, franchise_id, side, snaps, grade (snap-weighted), name."""
    parts = []
    for table, side, g, sc in (("offense_summary", "off", "grades_offense", "snap_counts_total"),
                               ("defense_summary", "def", "grades_defense", "snap_counts_defense")):
        p = pff_data.load(table, seasons, league="ncaa")
        p = p[["season", "player", "player_id", "franchise_id", g, sc]].rename(columns={g: "grade", sc: "snaps"})
        for c in ("grade", "snaps"):
            p[c] = pd.to_numeric(p[c], errors="coerce")
        p = p.dropna(subset=["grade", "snaps"])
        p = p[p.snaps > 0]
        p["gs"] = p["grade"] * p["snaps"]
        a = p.groupby(["season", "player_id", "franchise_id", "player"], as_index=False)[["snaps", "gs"]].sum()
        a["side"] = side
        parts.append(a)
    a = pd.concat(parts, ignore_index=True)
    a["grade"] = a["gs"] / a["snaps"]
    a["value"] = (a["snaps"] * (a["grade"] - REP)).clip(lower=0) / 1000.0
    a["nn"] = a["player"].map(nn)
    return a


def rosters(seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        f = CFBD / f"roster_{s}.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["firstName", "lastName", "team"])
            r["season"] = s
            out.append(r)
    r = pd.concat(out, ignore_index=True)
    r["nn"] = (r["firstName"].astype(str) + " " + r["lastName"].astype(str)).map(nn)
    return r


def portal(seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        f = CFBD / f"portal_{s}.csv"
        if f.exists():
            p = pd.read_csv(f, usecols=["season", "firstName", "lastName", "origin", "destination"])
            out.append(p)
    if not out:
        return pd.DataFrame(columns=["season", "nn", "origin"])
    p = pd.concat(out, ignore_index=True)
    p["nn"] = (p["firstName"].astype(str) + " " + p["lastName"].astype(str)).map(nn)
    return p[["season", "nn", "origin", "destination"]]


def franchise_map(pf: pd.DataFrame, ros: pd.DataFrame) -> pd.DataFrame:
    """(season, franchise_id) -> board team, by shared normalised names in the same season."""
    a = pf[["season", "franchise_id", "nn"]].drop_duplicates().merge(ros[["season", "team", "nn"]].drop_duplicates(), on=["season", "nn"])
    v = a.groupby(["season", "franchise_id", "team"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    v = v[v.n >= 8].drop_duplicates(["season", "franchise_id"]).drop_duplicates(["season", "team"])
    return v[["season", "franchise_id", "team"]]


def team_values(seasons) -> tuple[pd.DataFrame, dict]:
    pf = pff_seasons(range(2017, max(seasons) + 1))
    ros = rosters(range(2017, max(seasons) + 1))
    fm = franchise_map(pf, ros)
    pf = pf.merge(fm, on=["season", "franchise_id"], how="left")       # team = last season's team (may be NaN)
    por = portal(range(2021, max(seasons) + 1))
    rows, stats = [], {"rostered": 0, "matched": 0, "transfers_in": 0}
    for s in seasons:
        prev = pf[pf.season == s - 1]
        if prev.empty:
            continue
        # one row per PFF player last season: total value by side, last team
        pv = prev.groupby(["player_id", "nn", "side"], as_index=False)["value"].sum().pivot_table(
            index=["player_id", "nn"], columns="side", values="value", aggfunc="sum").fillna(0.0).reset_index()
        team_of = prev.sort_values("snaps").drop_duplicates("player_id", keep="last").set_index("player_id")["team"]
        pv["prev_team"] = pv["player_id"].map(team_of)
        r = ros[ros.season == s][["team", "nn"]].drop_duplicates()
        stats["rostered"] += len(r)
        m = r.merge(pv, on="nn", how="inner")
        # resolve shared names: the same team, then the portal origin, else drop
        cnt = m.groupby(["team", "nn"])["player_id"].transform("size")
        uniq = m[cnt == 1]
        amb = m[cnt > 1]
        same = amb[amb.prev_team == amb.team]
        same = same[same.groupby(["team", "nn"])["player_id"].transform("size") == 1]
        pr = por[por.season == s]
        if len(pr):
            amb2 = amb[~amb.set_index(["team", "nn"]).index.isin(same.set_index(["team", "nn"]).index)]
            o = amb2.merge(pr[["nn", "origin", "destination"]].rename(columns={"destination": "team"}), on=["team", "nn"])
            o = o[o.prev_team == o.origin]
            o = o[o.groupby(["team", "nn"])["player_id"].transform("size") == 1].drop(columns=["origin"])
        else:
            o = amb.iloc[:0]
        mm = pd.concat([uniq, same, o], ignore_index=True)
        # a PFF player can sit on only one roster: drop players matched to several teams
        mm = mm[mm.groupby("player_id")["team"].transform("nunique") == 1]
        stats["matched"] += len(mm)
        stats["transfers_in"] += int((mm.prev_team.notna() & (mm.prev_team != mm.team)).sum())
        for c in ("off", "def"):
            if c not in mm:
                mm[c] = 0.0
        t = mm.groupby("team", as_index=False)[["off", "def"]].sum()
        t["season"] = s
        rows.append(t)
    tv = pd.concat(rows, ignore_index=True).rename(columns={"off": "ret_off", "def": "ret_def"})
    tv["ret"] = tv["ret_off"] + tv["ret_def"]
    return tv, stats


def add_features(frame: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    for side in ("home", "away"):
        x = tv.rename(columns={"team": f"{side}_team", "ret": f"{side}_ret", "ret_off": f"{side}_ret_off", "ret_def": f"{side}_ret_def"})
        f = f.merge(x, on=["season", f"{side}_team"], how="left")
    for c in ("ret", "ret_off", "ret_def"):
        fill = pd.concat([f[f"home_{c}"], f[f"away_{c}"]]).median()
        f[f"home_{c}"] = f[f"home_{c}"].fillna(fill)
        f[f"away_{c}"] = f[f"away_{c}"].fillna(fill)
    f["d_pffret"] = f["home_ret"] - f["away_ret"]
    f["d_pffret_early"] = f["d_pffret"] * f["early"]
    # centred on the fill, so rows without data are exactly 0: a filled constant like 2 x median
    # kept a rounding-size spread that the ridge's standardisation blew up (first run, 2018 totals
    # +1.11; fixed and every version re-run, Nov 5, 2026)
    for c, k in (("s_pffret_off", "ret_off"), ("s_pffret_def", "ret_def")):
        fill = pd.concat([f[f"home_{k}"], f[f"away_{k}"]]).median()
        f[c] = (f[f"home_{k}"] - fill) + (f[f"away_{k}"] - fill)
        f.loc[f[c].abs() < 1e-9, c] = 0.0
    return f


def measures(out):
    o = out[out.played & out.spread_line.notna() & out.total_line.notna()].copy()
    o["em"] = np.abs(o.model_margin - o.result)
    o["et"] = np.abs(o.model_total - o.total)
    o["wk05"] = (o["week"] <= 5) & (o["postseason"] == 0)
    res = {}
    for name, x in (("tune", o[o.season.between(*TUNE)]), ("hold", o[o.season >= C.HOLDOUT_FROM])):
        e = x[x.wk05]
        res[name] = {"margin": x.em.mean(), "total": x.et.mean(), "m05": e.em.mean(), "t05": e.et.mean(), "n": len(x)}
    return res, o[["game_id", "season", "week", "em", "et"]].reset_index(drop=True)


def main():
    games, tg, qb = data.load_all(verbose=False)
    inputs = extra.load_inputs(games, verbose=False)
    eng = CFBRatingEngine(tg, qb, games=games, extra=inputs["rated"])
    base = model.build_frame(games, eng, inputs=inputs)
    seasons = sorted(int(s) for s in games.season.unique())
    tv, st = team_values([s for s in seasons if s >= 2018])
    print(f"roster rows {st['rostered']}, matched to last season's PFF {st['matched']}, of them transfers in {st['transfers_in']}")
    print(tv.groupby("season")[["ret", "ret_off", "ret_def"]].agg(["mean", "std"]).round(2).to_string())
    fr = add_features(base, tv)
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    versions = {
        "A ret + totals": (MF + ["d_pffret"], TF + ["s_pffret_off", "s_pffret_def"]),
        "B ret early only": (MF + ["d_pffret_early"], TF),
        "C A + early": (MF + ["d_pffret", "d_pffret_early"], TF + ["s_pffret_off", "s_pffret_def"]),
        "D A replacing CFBD returning": ([c for c in MF if c != "d_rtprod"] + ["d_pffret"],
                                         [c for c in TF if c not in ("s_off_rtprod", "s_def_rtprod")] + ["s_pffret_off", "s_pffret_def"]),
    }
    out0, _ = model.walk_forward(fr)
    r0, rows0 = measures(out0)
    hdr = "version                        | tune margin total (wk0-5 m / t) | hold margin total (wk0-5 m / t)"
    print("\n" + hdr)
    fmt = lambda r: (f"{r['tune']['margin']:.3f} {r['tune']['total']:.3f} ({r['tune']['m05']:.3f} / {r['tune']['t05']:.3f}) | "
                     f"{r['hold']['margin']:.3f} {r['hold']['total']:.3f} ({r['hold']['m05']:.3f} / {r['hold']['t05']:.3f})")
    print(f"{'current model':31s}| {fmt(r0)}   n {r0['tune']['n']}/{r0['hold']['n']}")
    res, rows = {}, {}
    for name, (mf, tf) in versions.items():
        out, _ = model.walk_forward(fr, margin_feats=mf, total_feats=tf)
        res[name], rows[name] = measures(out)
        print(f"{name:31s}| {fmt(res[name])}", flush=True)
    ok = [n for n, r in res.items() if r["tune"]["margin"] < r0["tune"]["margin"] and r["tune"]["total"] < r0["tune"]["total"]]
    print()
    pick = min(ok, key=lambda n: res[n]["tune"]["margin"]) if ok else min(res, key=lambda n: res[n]["tune"]["margin"])
    print(("Chosen on tuning: " if ok else "No version beats the current model on both tuning measures; closest: ") + pick)
    m = rows[pick].merge(rows0, on=["game_id", "season", "week"], suffixes=("_n", "_b"))
    for per, mask in (("tune", m.season.between(*TUNE)), ("hold", m.season >= C.HOLDOUT_FROM)):
        for col in ("em", "et"):
            d = (m.loc[mask, f"{col}_n"] - m.loc[mask, f"{col}_b"]).to_numpy()
            print(f"  {per} {'margin' if col == 'em' else 'total':6s} {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
    by = m.assign(dm=m.em_n - m.em_b, dt=m.et_n - m.et_b).groupby("season")[["dm", "dt"]].mean()
    print(by[by.index >= 2018].round(4).to_string())
    keep = bool(ok) and all(res[pick][p][k] < r0[p][k] for p in ("tune", "hold") for k in ("margin", "total"))
    print(f"\nKEEP: {'yes' if keep else 'no'}")
    outdir = Path(__file__).parent / "out"
    outdir.mkdir(exist_ok=True)
    pd.DataFrame([{"version": "current model", **{f"{p}_{k}": v for p in r0 for k, v in r0[p].items()}}] +
                 [{"version": n, **{f"{p}_{k}": v for p in r for k, v in r[p].items()}} for n, r in res.items()]).round(4).to_csv(outdir / "cfb_pff_returning.csv", index=False)


if __name__ == "__main__":
    main()
