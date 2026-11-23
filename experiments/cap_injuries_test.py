"""Pre-set test 3 of Nov 22, 2026: cap-weighted non-QB injuries in the NFL game model.

    PFF_DATA_DIR=<pff-data clone> python experiments/cap_injuries_test.py

Rule written before running (Project doc claude/plan-three-untracked-stats-tests-2026-11-22.md).
Today the game model has one group injury input, d_chg_ol (offensive-line starters missing beyond
recent games, weighted by snap share), plus the skill-starter role inputs. ELWAY's idea: weight a
missing player by his share of the salary cap.

Input, per team and game, with injuries.py's own core players (its six non-QB groups, top players by
snap share over the team's earlier games) and its own definition of unavailable:
  weight     the cap share (percent of the cap) of the player's contract in force that season: the
             latest contract signed that season or earlier (OverTheCap via nflverse
             historical_contracts, matched by league player id); 0.3 if none is matched
  cap_miss   summed weight of the core players unavailable now
  cap_mrec   the same over the team's last four games (a core player with no snaps counts), averaged
  chg        cap_miss - cap_mrec, for offense groups (ol, pc, rb) and defense groups (dl, lb, db)
Versions:
  A  margin + d_cap (home - away of offense + defense); total + s_cap_off, s_cap_def (both teams' sums)
  B  margin + d_cap_off, d_cap_def; total as A
  C  A with d_chg_ol taken out of the margin
Measures (6): the game model's average miss on the margin and on the total, tuning 2014-18 and
  holdout 2019-25, game-day inputs (4); the same two on 2019-25 with Friday information (the model
  fitted on game-day inputs as now, predicting from nflmodel/bettime.py's Friday quarterbacks and
  injuries._season(bettime=True)-style inputs) (2).
Rule: the version is chosen on tuning alone (better than the current model on both tuning measures,
  then the lowest tuning margin miss). Kept only if better on all six and each holdout gain >= 1
  paired SE. Mixed, or under 1 SE: a trade-off for Ryan. No version better on both tuning measures:
  nothing is carried to the holdout.
Props check, only if a version passes: the player walk-forward with the candidate's margin and total.
Limit written before the run: players on rookie contracts are underweighted by cap share.

Writes experiments/out/cap_injuries_test.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import bettime, config as C, injuries as IJ, model, players as PL  # noqa: E402
from nflmodel.data import _download  # noqa: E402

OUT = HERE / "out" / "cap_injuries_test.json"
CONTRACTS = "https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.parquet"
TUNE, HOLD = (2014, 2018), (2019, 2025)
OFF, DEF = ("ol", "pc", "rb"), ("dl", "lb", "db")
NO_CONTRACT = 0.3


def contracts() -> pd.DataFrame:
    f = C.RAW / "historical_contracts.parquet"
    if not f.exists():
        _download(CONTRACTS, f)
    c = pd.read_parquet(f, columns=["gsis_id", "year_signed", "apy_cap_pct"]).dropna()
    c = c[c["year_signed"] > 0]
    c["pct"] = 100 * c["apy_cap_pct"]
    return c.sort_values(["gsis_id", "year_signed", "pct"])


def cap_lookup(c: pd.DataFrame, season: int) -> dict:
    x = c[c["year_signed"] <= season].drop_duplicates("gsis_id", keep="last")
    return dict(zip(x["gsis_id"], x["pct"]))


def cap_season(season: int, sched: pd.DataFrame, cap: dict, bt: bool = False) -> pd.DataFrame:
    """injuries._season with each core player weighted by cap share (no overrides: played games only)."""
    f = C.RAW / f"cap_groups{'_bt' if bt else ''}_{season}.csv"
    if f.exists():
        return pd.read_csv(f)
    paths = {k: C.RAW / f"{k}_{season}.parquet" for k in ("snap_counts", "injuries", "roster_weekly")}
    for k, path in paths.items():
        kind = "weekly_rosters" if k == "roster_weekly" else k
        if not path.exists():
            _download(IJ.URL.format(kind=kind, season=season).replace(f"/{kind}_{season}", f"/{k}_{season}"), path)
    s = pd.read_parquet(paths["snap_counts"])
    inj = pd.read_parquet(paths["injuries"])
    ros = pd.read_parquet(paths["roster_weekly"], columns=["team", "week", "full_name", "status", "game_type", "gsis_id"])
    s, ros = s[s["game_type"] == "REG"], ros[ros["game_type"] == "REG"]
    for df in (s, inj, ros):
        df["team"] = df["team"].replace(C.TEAM_ALIASES)
    s["key"], inj["key"], ros["key"] = s["player"].map(IJ._norm), inj["full_name"].map(IJ._norm), ros["full_name"].map(IJ._norm)
    report = inj.groupby(["team", "week", "key"])["report_status"].last().to_dict()
    roster = ros.groupby(["team", "week", "key"])["status"].first().to_dict()
    roster_weeks = ros.groupby("team")["week"].apply(lambda w: sorted(set(w))).to_dict()
    ids = ros.dropna(subset=["gsis_id"]).groupby(["team", "key"])["gsis_id"].last().to_dict()
    pos = set().union(*(IJ.GROUPS[g][0] for g in IJ.ALIAS_GROUPS))
    sn = s.loc[s["position"].isin(pos), ["team", "player"]].rename(columns={"player": "name"})
    al_ros = IJ.alias_map(sn, ros[["team", "full_name"]].rename(columns={"full_name": "name"}))
    al_rep = IJ.alias_map(sn, inj[["team", "full_name"]].rename(columns={"full_name": "name"}))
    for d, al in ((roster, al_ros), (report, al_rep)):
        rev = {}
        for (team, k), to in al.items():
            rev.setdefault((team, to), []).append(k)
        for (team, wk_, key_), v in list(d.items()):
            for k in rev.get((team, key_), ()):
                d.setdefault((team, wk_, k), v)
    for (team, k), to in al_ros.items():
        if (team, to) in ids:
            ids.setdefault((team, k), ids[(team, to)])
    g = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    long = pd.concat([g[["game_id", "week", "home_team"]].rename(columns={"home_team": "team"}),
                      g[["game_id", "week", "away_team"]].rename(columns={"away_team": "team"})])
    rows = []
    for team, tg in long.groupby("team"):
        tg = tg.sort_values("week")
        t = s[s["team"] == team]
        have = sorted(set(t["week"]))
        for gid, wk in zip(tg["game_id"], tg["week"]):
            row = {"season": season, "game_id": gid, "team": team, "core": 0, "matched": 0}
            for side in ("off", "def"):
                row[f"cap_miss_{side}"] = row[f"cap_mrec_{side}"] = 0.0
            prior = [w for w in have if w < wk]
            rw = [w for w in roster_weeks.get(team, []) if (w < wk if bt else w <= wk)]
            rweek = rw[-1] if rw else None
            exact = rweek == wk
            if prior and (wk in have):
                for grp, (pos, k, col) in IJ.GROUPS.items():
                    gg = t[t["position"].isin(pos)]
                    share = gg[gg["week"].isin(prior)].pivot_table(index="key", columns="week", values=col, aggfunc="max")
                    share = share.reindex(columns=prior).fillna(0.0)
                    core = share.mean(axis=1).sort_values(ascending=False).head(k)
                    core = core[core > 0.2]
                    if core.empty:
                        continue
                    w = pd.Series({key: cap.get(ids.get((team, key)), np.nan) for key in core.index})
                    row["core"] += len(w)
                    row["matched"] += int(w.notna().sum())
                    w = w.fillna(NO_CONTRACT)
                    out_now = []
                    for key in core.index:
                        rep = report.get((team, wk, key))
                        st = roster.get((team, rweek, key)) if rweek is not None else "ACT"
                        if rep in ("Out", "Doubtful"):
                            out_now.append(key)
                        elif rweek is None:
                            continue
                        elif st is None:
                            out_now.append(key)
                        elif exact and st != "ACT":
                            out_now.append(key)
                        elif not exact and st not in ("ACT", "INA"):
                            out_now.append(key)
                    side = "off" if grp in OFF else "def"
                    row[f"cap_miss_{side}"] += float(w[out_now].sum())
                    rec = prior[-IJ.RECENT:]
                    played = share.loc[core.index, rec] > 0
                    row[f"cap_mrec_{side}"] += float((~played).mul(w, axis=0).sum(axis=0).mean())
            rows.append(row)
    for path in paths.values():
        path.unlink(missing_ok=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(f, index=False)
    print("  cap groups", season, "bet-time" if bt else "", flush=True)
    return out


def attach(frame: pd.DataFrame, cg: pd.DataFrame) -> pd.DataFrame:
    cg = cg.copy()
    cg["off"], cg["dfn"] = cg["cap_miss_off"] - cg["cap_mrec_off"], cg["cap_miss_def"] - cg["cap_mrec_def"]
    h = cg[["game_id", "team", "off", "dfn"]].rename(columns={"team": "home_team", "off": "h_off", "dfn": "h_def"})
    a = cg[["game_id", "team", "off", "dfn"]].rename(columns={"team": "away_team", "off": "a_off", "dfn": "a_def"})
    f = frame.drop(columns=[c for c in ("h_off", "h_def", "a_off", "a_def") if c in frame]).merge(h, on=["game_id", "home_team"], how="left") \
             .merge(a, on=["game_id", "away_team"], how="left")
    f[["h_off", "h_def", "a_off", "a_def"]] = f[["h_off", "h_def", "a_off", "a_def"]].fillna(0.0)
    f["d_cap_off"], f["d_cap_def"] = f["h_off"] - f["a_off"], f["h_def"] - f["a_def"]
    f["d_cap"] = f["d_cap_off"] + f["d_cap_def"]
    f["s_cap_off"], f["s_cap_def"] = f["h_off"] + f["a_off"], f["h_def"] + f["a_def"]
    return f


def frames():
    from nflmodel import data, extra, game_roles, pff_qb, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    te, qe = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    inj = IJ.load_injuries(games, refresh_current=False, verbose=False)
    prior = pff_qb.load(qb, sorted(games.season.unique()))
    print("quarterback prior:", "PFF grades" if prior is not None else "flat (NOT the production model)", flush=True)
    engine = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    base = model.build_frame(games, engine, injuries=inj)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = PL.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)
    f0, _ = model.walk_forward(base)
    rows = PL.player_rows(PL.team_games(f0, pg), pg, avail, pinfo)
    rf = game_roles.team_features(rows, R, avail)
    f = game_roles.attach(base, rf)
    hs = list(range(HOLD[0], HOLD[1] + 1))
    fr = game_roles.attach(model.build_frame(bettime.friday_games(games, hs), engine, injuries=bettime.friday_injuries(games, hs)), rf)
    c = contracts()
    cg = pd.concat([cap_season(s, games, cap_lookup(c, s)) for s in range(IJ.FIRST, HOLD[1] + 1)], ignore_index=True)
    cb = pd.concat([cap_season(s, games, cap_lookup(c, s), bt=True) for s in hs], ignore_index=True)
    return attach(f, cg), attach(fr, cb), cg


def main():
    f, fr, cg = frames()
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    versions = {"A (one number)": (MF + ["d_cap"], TF + ["s_cap_off", "s_cap_def"]),
                "B (offense and defense apart)": (MF + ["d_cap_off", "d_cap_def"], TF + ["s_cap_off", "s_cap_def"]),
                "C (A in place of d_chg_ol)": ([c for c in MF if c != "d_chg_ol"] + ["d_cap"], TF + ["s_cap_off", "s_cap_def"])}
    k = cg[cg.season.between(2014, 2025)]
    hf = f[f.season.between(*HOLD) & f.played]
    res = {"inputs": {"core_players_matched_to_a_contract": round(float(k.matched.sum() / k.core.sum()), 3),
                      "team_games_with_any_cap_missing": round(float(((k.cap_miss_off + k.cap_miss_def) > 0).mean()), 3),
                      "avg_cap_pct_missing_per_team_game": round(float((k.cap_miss_off + k.cap_miss_def).mean()), 2),
                      "spread_2019_25": {c: round(float(hf[c].std()), 3) for c in ("d_cap", "d_cap_off", "d_cap_def", "d_chg_ol")},
                      "corr_d_cap_d_chg_ol_2019_25": round(float(np.corrcoef(hf["d_cap"], hf["d_chg_ol"])[0, 1]), 3)}}
    print(json.dumps(res), flush=True)

    def out(mf, tf):
        o, _ = model.walk_forward(f, margin_feats=mf, total_feats=tf)
        o = o[o.played & o.model_margin.notna() & (o.season <= HOLD[1])].copy()
        o["em"], o["et"] = (o.model_margin - o.result).abs(), (o.model_total - o.total).abs()
        return o.set_index("game_id")

    def friday(mf, tf):
        parts = []
        for d in (f, fr):
            for c in mf + tf:
                if c not in d:
                    d[c] = 0.0
        for s in range(HOLD[0], HOLD[1] + 1):
            tr = f[(f.season < s) & f.played]
            m = model.Linear(20.0, intercept=False).fit(tr[mf].to_numpy(float), tr["result"].to_numpy(float))
            t = model.Linear(20.0).fit(tr[tf].to_numpy(float), tr["total"].to_numpy(float))
            x = fr[(fr.season == s) & fr.played].copy()
            x["em"] = np.abs(m.predict(x[mf].to_numpy(float)) - x["result"].to_numpy(float))
            x["et"] = np.abs(t.predict(x[tf].to_numpy(float)) - x["total"].to_numpy(float))
            parts.append(x[["game_id", "season", "em", "et"]])
        return pd.concat(parts).set_index("game_id")

    def cmp(b, o, per):
        k = b.season.between(*per)
        r = {}
        for col, lab in (("em", "margin"), ("et", "total")):
            a_, b_ = b.loc[k, col].to_numpy(float), o.loc[b.index[k], col].to_numpy(float)
            dd = b_ - a_
            r[lab] = {"n": int(k.sum()), "now": round(float(a_.mean()), 4), "candidate": round(float(b_.mean()), 4),
                      "diff": round(float(dd.mean()), 5), "se": round(float(dd.std(ddof=1) / np.sqrt(len(dd))), 5)}
        return r

    base = out(MF, TF)
    outs = {v: out(*fs) for v, fs in versions.items()}
    res["tuning"] = {v: cmp(base, o, TUNE) for v, o in outs.items()}
    print(json.dumps(res["tuning"]), flush=True)
    ok = [v for v in versions if all(m["diff"] < 0 for m in res["tuning"][v].values())]
    res["better_on_both_tuning_measures"] = ok
    if not ok:
        res["verdict"] = "No version better on both tuning measures: nothing carried to the holdout, nothing kept."
    else:
        pick = min(ok, key=lambda v: res["tuning"][v]["margin"]["candidate"])
        res["chosen"] = pick
        res["holdout"] = cmp(base, outs[pick], HOLD)
        res["friday_information_2019_25"] = cmp(friday(MF, TF), friday(*versions[pick]), HOLD)
        ms = list(res["tuning"][pick].values()) + list(res["holdout"].values()) + list(res["friday_information_2019_25"].values())
        better = [m["diff"] < 0 for m in ms]
        se_ok = all(-m["diff"] >= m["se"] for m in res["holdout"].values())
        res["better_on"] = f"{sum(better)} of {len(better)}"
        res["passes_before_props_check"] = bool(all(better) and se_ok)
        res["verdict"] = ("Passes the six measures: the props check decides." if res["passes_before_props_check"] else
                          "Better on every measure but a holdout gain is under 1 SE: a trade-off for Ryan." if all(better)
                          else "Mixed: a trade-off for Ryan, not kept by the rule.")
        o, info = model.walk_forward(f, margin_feats=versions[pick][0], total_feats=versions[pick][1])
        res["coefficients_2025_fit"] = {c: info[2025]["margin_coefs"].get(c) for c in versions[pick][0] if "cap" in c or c == "d_chg_ol"} | \
                                       {c: info[2025]["total_coefs"].get(c) for c in versions[pick][1] if "cap" in c}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
