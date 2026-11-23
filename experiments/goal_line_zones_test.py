"""Pre-set test 1 of Nov 22, 2026: finer goal-line zones in the NFL touchdown projections.

    PFF_DATA_DIR=<pff-data clone> python experiments/goal_line_zones_test.py

Rule written before running (Project doc claude/plan-three-untracked-stats-tests-2026-11-22.md).
Today (players.raw_projection) a team's rushing touchdowns are shared out by carries x (half his
red-zone carry rate + half his inside-the-5 carry rate, doubled) and its receiving touchdowns by
targets x his red-zone target rate. Not used: end-zone targets, finer target or carry zones, league
touchdown rates by zone.

Zone counts per player-game from play-by-play (the plays players_data.py counts):
  targets  ez (air yards reach the goal line) | other inside the 10 | other 11-20 | elsewhere
  carries  inside the 2 | 3-5 | 6-10 | 11-20 | elsewhere
League touchdowns per target and per carry in each zone: 2014-18 only. Player sums as of each game
from the games he played (players.asof, the board's player decay).
Versions (only rec_td and rush_td projections are taken from the candidate):
  A  receiving weight = targets x (0.5 x rz_tgt_rate + 0.5 x k x ez_tgt_rate), k = league red-zone
     targets / end-zone targets; ez_tgt_rate = (ez + 0.05) / (targets + 1). Rushing unchanged.
  B  receiving weight = targets x xtd_tgt, rushing weight = carries x xtd_car, where
     xtd = (sum of zone count x league zone rate + 1 x his position's league rate) / (count + 1)
  C  B's rushing half only.
Measures (14):
  anytime-TD Brier (1 - exp(-(rec + rush TD projection))), WR/TE/RB who played with a current-model
    chance of 5%+: game-day and bet-time, tuning 2014-18 and holdout 2019-25 (4)
  receiving-TD RMSE, WR/TE/RB with 1+ projected reception (4); rushing-TD RMSE, RB/QB with 3+
    projected carries (4); a measure a version does not touch counts as not worse
  props check: the same Brier on player-games with a DraftKings/FanDuel anytime-TD price in 2025,
    closing prices with game-day projections and the pricing snapshot with bet-time (2)
Rule: the version is chosen on tuning alone (better than the current model on every tuning measure
  it changes, then the lowest game-day tuning anytime-TD Brier). Kept only if better on every measure
  and each holdout gain >= 1 paired SE (clustered by game). Mixed, or under 1 SE: a trade-off for
  Ryan. No version better on every tuning measure: nothing is carried to the holdout.
Reported, not used: the weekly Likeliest TD Scorers list rebuilt from bet-time projections (top 15 a
  week by anytime-TD chance), how many scored, now against each version, 2019-25 and 2026.
Projections are players.walk_forward (touchdowns do not pass through the boosted step).

Writes experiments/out/goal_line_zones_test.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from nflmodel import config as C, odds as O, players as PL, propbackfill as B, propmarket as PM  # noqa: E402

OUT = HERE / "out" / "goal_line_zones_test.json"
TABLES = C.RAW / "air_yards_wopr_tables.pkl"     # the board's player tables (air_yards_wopr_test.tables())
LEAGUE = (2014, 2018)
TUNE, HOLD = (2014, 2018), (2019, 2025)
TZ = ["t_ez", "t_10", "t_20", "t_out"]
CZ = ["c_2", "c_5", "c_10", "c_20", "c_out"]
MODE = {"v": None}


def zones(seasons) -> pd.DataFrame:
    from nflmodel.data import _download
    got = []
    for s in seasons:
        f = C.RAW / f"td_zones_{s}.csv"
        if not f.exists():
            raw = C.RAW / f"pbp_zone_{s}.parquet"
            _download(C.PBP_URL.format(season=s), raw)
            p = pd.read_parquet(raw, columns=["game_id", "posteam", "pass_attempt", "rush_attempt", "sack", "qb_spike", "qb_kneel",
                                              "two_point_attempt", "receiver_player_id", "rusher_player_id", "air_yards", "yardline_100",
                                              "pass_touchdown", "rush_touchdown", "td_player_id"])
            raw.unlink()
            p = p[(p["two_point_attempt"].fillna(0) == 0) & p["posteam"].notna() & p["yardline_100"].notna()]
            y = p["yardline_100"]
            tg = p[(p["pass_attempt"] == 1) & (p["sack"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0) & p["receiver_player_id"].notna()].copy()
            ez = tg["air_yards"].notna() & (tg["air_yards"] >= tg["yardline_100"])
            tg["zone"] = np.where(ez, "t_ez", np.where(tg["yardline_100"] <= 10, "t_10", np.where(tg["yardline_100"] <= 20, "t_20", "t_out")))
            tg["td"] = ((tg["pass_touchdown"] == 1) & (tg["td_player_id"] == tg["receiver_player_id"])).astype(int)
            tg = tg.rename(columns={"receiver_player_id": "player_id"})
            ru = p[(p["rush_attempt"] == 1) & (p["qb_kneel"].fillna(0) == 0) & p["rusher_player_id"].notna()].copy()
            yr = ru["yardline_100"]
            ru["zone"] = np.where(yr <= 2, "c_2", np.where(yr <= 5, "c_5", np.where(yr <= 10, "c_10", np.where(yr <= 20, "c_20", "c_out"))))
            ru["td"] = ((ru["rush_touchdown"] == 1) & (ru["td_player_id"] == ru["rusher_player_id"])).astype(int)
            ru = ru.rename(columns={"rusher_player_id": "player_id"})
            x = pd.concat([tg[["game_id", "player_id", "zone", "td"]], ru[["game_id", "player_id", "zone", "td"]]])
            n = x.groupby(["game_id", "player_id", "zone"]).size().unstack("zone").reindex(columns=TZ + CZ).fillna(0)
            td = x.groupby(["game_id", "player_id", "zone"])["td"].sum().unstack("zone").reindex(columns=TZ + CZ).fillna(0).add_suffix("_td")
            n.join(td).reset_index().assign(season=s).to_csv(f, index=False)
            print("  zones", s, flush=True)
        got.append(pd.read_csv(f, dtype={"player_id": str}))
    return pd.concat(got, ignore_index=True)


def inputs(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a = t[["game_id", "player_id", "season", "week", "t", "played", "active", "group"]].copy()
    a["player_id"] = a["player_id"].astype(str)
    z = zones(sorted(int(s) for s in a["season"].unique()))
    a = a.merge(z.drop(columns="season"), on=["game_id", "player_id"], how="left")
    use = (a["active"] == 1) & a["played"].astype(bool)
    cols = TZ + CZ
    a[cols + [c + "_td" for c in cols]] = a[cols + [c + "_td" for c in cols]].fillna(0.0)
    lg = a[use & a["season"].between(*LEAGUE)]
    rate = {c: float(lg[c + "_td"].sum() / lg[c].sum()) for c in cols}
    pos_t = (lg.groupby("group")[[c + "_td" for c in TZ]].sum().sum(axis=1) / lg.groupby("group")[TZ].sum().sum(axis=1))
    pos_c = (lg.groupby("group")[[c + "_td" for c in CZ]].sum().sum(axis=1) / lg.groupby("group")[CZ].sum().sum(axis=1))
    for c in cols:
        a[c] = a[c].where(use)
    f = PL.asof(a, "player_id", cols, None, PL.PLAYER_HL)
    tsum = sum(f[c + "_sum"] for c in TZ).to_numpy(float)
    csum = sum(f[c + "_sum"] for c in CZ).to_numpy(float)
    xt = sum(f[c + "_sum"].to_numpy(float) * rate[c] for c in TZ)
    xc = sum(f[c + "_sum"].to_numpy(float) * rate[c] for c in CZ)
    out = f[["game_id", "player_id"]].copy()
    out["ez_tgt_rate"] = (f["t_ez_sum"].to_numpy(float) + 0.05) / (tsum + 1)
    out["xtd_tgt"] = (xt + f["group"].map(pos_t).fillna(pos_t.mean()).to_numpy(float)) / (tsum + 1)
    out["xtd_car"] = (xc + f["group"].map(pos_c).fillna(pos_c.mean()).to_numpy(float)) / (csum + 1)
    info = {"league_td_per_opportunity_2014_18": {c: round(v, 4) for c, v in rate.items()},
            "position_td_per_target": pos_t.round(4).to_dict(), "position_td_per_carry": pos_c.round(4).to_dict()}
    return out, info


_RAW = PL.raw_projection


def raw_projection(p, att_model, car_model):
    p = _RAW(p, att_model, car_model)
    v = MODE["v"]
    if v is None:
        return p
    key = [p["game_id"], p["team"]]
    team_td = p["imp_pts"].clip(3, 45) / 7.6
    team_ptd, team_rtd = team_td * p["ptd_share"], team_td * (1 - p["ptd_share"])
    if v in ("A", "B"):
        w = p["targets_p"] * ((0.5 * p["rz_tgt_rate"] + 0.5 * MODE["k"] * p["ez_tgt_rate"]) if v == "A" else p["xtd_tgt"])
        p["rec_td_p"] = (team_ptd * w / w.groupby(key).transform("sum").replace(0, np.nan)).fillna(0.0)
    if v in ("B", "C"):
        w = p["carries_p"] * p["xtd_car"]
        p["rush_td_p"] = (team_rtd * w / w.groupby(key).transform("sum").replace(0, np.nan)).fillna(0.0)
    return p


PL.raw_projection = raw_projection


def project(table, v):
    MODE["v"] = v
    try:
        return PL.walk_forward(table, first_test=C.FIRST_TEST_SEASON).reset_index(drop=True)
    finally:
        MODE["v"] = None


def _cmp(e0, e1, gid, root=False):
    d = e1 - e0
    cl = pd.Series(d).groupby(gid).sum()
    n = len(d)
    if root:
        r0, r1 = float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean()))
        return {"n": n, "now": round(r0, 5), "candidate": round(r1, 5), "diff": round(r1 - r0, 6),
                "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / n / (2 * r0)), 6)}
    return {"n": n, "now": round(float(e0.mean()), 5), "candidate": round(float(e1.mean()), 5), "diff": round(float(d.mean()), 6),
            "se": round(float(cl.std(ddof=1) * np.sqrt(len(cl)) / n), 6)}


def chance(x):
    return 1 - np.exp(-(x["rec_td_proj"].fillna(0) + x["rush_td_proj"].fillna(0)).clip(lower=0).to_numpy(float))


def measures(base, cand, per) -> dict:
    ok = (base["played"].astype(bool) & (base["active"] == 1) & base["season"].between(*per)).to_numpy()
    gid = base["game_id"].to_numpy()
    hit = ((base["rec_td"].fillna(0) + base["rush_td"].fillna(0)) > 0).to_numpy(float)
    c0, c1 = chance(base), chance(cand)
    k = ok & base["group"].isin(["WR", "TE", "RB"]).to_numpy() & (c0 >= 0.05)
    out = {"anytime Brier": _cmp((c0[k] - hit[k]) ** 2, (c1[k] - hit[k]) ** 2, gid[k])}
    for st, filt in (("rec_td", base["group"].isin(["WR", "TE", "RB"]) & (base["receptions_proj"] >= 1)),
                     ("rush_td", base["group"].isin(["RB", "QB"]) & (base["carries_proj"] >= 3))):
        k = ok & filt.to_numpy()
        y = base[st].fillna(0).to_numpy(float)[k]
        out[st + " RMSE"] = _cmp((base[st + "_proj"].to_numpy(float)[k] - y) ** 2, (cand[st + "_proj"].to_numpy(float)[k] - y) ** 2, gid[k], root=True)
    return out


def td_prices(base, cand, kind) -> dict:
    s = B.load(2025, kind)
    s = s[(s["market"] == "player_anytime_td") & s["side"].str.lower().isin(["yes", "over"]) & s["book"].isin(["draftkings", "fanduel"])].copy()
    s["nname"] = s["player"].map(PM.norm_name)
    s["imp"] = 1 / O.american_to_decimal(s["price"])
    s = s.groupby(["game_id", "nname"], as_index=False)["imp"].mean()
    info = pd.read_csv(C.CACHE / "player_info_2025.csv", dtype={"player_id": str})
    b = base[(base["season"] == 2025) & base["played"].astype(bool)].copy()
    b["c1"] = chance(cand.loc[b.index])
    b["c0"] = chance(b)
    b["player_id"] = b["player_id"].astype(str)
    b = b.merge(info[["player_id", "name"]], on="player_id", how="left")
    b["nname"] = b["name"].map(PM.norm_name)
    b = b[~b.duplicated(["game_id", "nname"], keep=False)]
    x = s.merge(b, on=["game_id", "nname"], how="inner")
    hit = ((x["rec_td"].fillna(0) + x["rush_td"].fillna(0)) > 0).to_numpy(float)
    r = _cmp((x["c0"].to_numpy() - hit) ** 2, (x["c1"].to_numpy() - hit) ** 2, x["game_id"].to_numpy())
    fair = (x["imp"] / (1 + PM.ONE_SIDED_MARGIN)).to_numpy()
    r["books"] = round(float(((fair - hit) ** 2).mean()), 5)
    r["scored"] = round(float(hit.mean()), 4)
    return r


def likeliest(base, cands, per) -> dict:
    done = set(base.loc[base["played"].astype(bool), "game_id"])        # games that have been played
    b = base[base["season"].between(*per) & base["group"].isin(["WR", "TE", "RB", "QB"]) & base["game_id"].isin(done)]
    hit = ((b["rec_td"].fillna(0) + b["rush_td"].fillna(0)) > 0).astype(float)
    hit = hit.where(b["played"].astype(bool), 0.0)
    out = {}
    for name, t in [("now", base)] + list(cands.items()):
        c = pd.Series(chance(t.loc[b.index]), index=b.index)
        top = c.groupby([b["season"], b["week"]]).rank(ascending=False, method="first") <= 15
        out[name] = {"picks": int(top.sum()), "scored": int(hit[top].sum()), "rate": round(float(hit[top].mean()), 4),
                     "expected": round(float(c[top].mean()), 4)}
    return out


def main():
    T = pickle.load(open(TABLES, "rb"))
    feats, info = inputs(T["gameday"])
    g0 = T["gameday"]
    lg = g0[g0["played"].astype(bool) & (g0["active"] == 1) & g0["season"].between(*LEAGUE)]
    zz = zones(range(LEAGUE[0], LEAGUE[1] + 1))
    MODE["k"] = float(lg["rz_targets"].sum() / zz["t_ez"].sum())
    info["k_red_zone_targets_per_end_zone_target"] = round(MODE["k"], 3)
    for k in ("gameday", "bettime"):
        T[k] = T[k].assign(player_id=T[k]["player_id"].astype(str)).merge(feats, on=["game_id", "player_id"], how="left")
        T[k]["ez_tgt_rate"] = T[k]["ez_tgt_rate"].fillna(0.05)
        for c in ("xtd_tgt", "xtd_car"):
            T[k][c] = T[k][c].fillna(T[k][c].median())
    res = {"inputs": info}
    print(json.dumps(info), flush=True)
    now = {k: project(T[k], None) for k in ("gameday", "bettime")}
    print("current done", flush=True)
    cands, tun = {}, {}
    for v in ("A", "B", "C"):
        cands[v] = {k: project(T[k], v) for k in ("gameday", "bettime")}
        for k in ("gameday", "bettime"):
            assert (now[k][["game_id", "player_id"]].to_numpy() == cands[v][k][["game_id", "player_id"]].to_numpy()).all()
        tun[v] = {f"{m} {k}": r for k in ("gameday", "bettime") for m, r in measures(now[k], cands[v][k], TUNE).items()}
        print(v, json.dumps(tun[v]), flush=True)
    res["tuning"] = tun
    changed = {v: [m for m in tun[v].values() if m["diff"] != 0] for v in tun}
    ok = [v for v in tun if changed[v] and all(m["diff"] < 0 for m in changed[v])]
    res["better_on_every_tuning_measure"] = ok
    if not ok:
        res["verdict"] = "No version better on every tuning measure: nothing carried to the holdout, nothing kept."
    else:
        pick = min(ok, key=lambda v: tun[v]["anytime Brier gameday"]["candidate"])
        c = cands[pick]
        res["chosen"] = pick
        res["holdout"] = {f"{m} {k}": r for k in ("gameday", "bettime") for m, r in measures(now[k], c[k], HOLD).items()}
        res["td_prices_2025"] = {"closing, game-day": td_prices(now["gameday"], c["gameday"], "close"),
                                 "pricing snapshot, bet-time": td_prices(now["bettime"], c["bettime"], "flag")}
        ms = [m for m in list(tun[pick].values()) + list(res["holdout"].values()) + list(res["td_prices_2025"].values()) if m["diff"] != 0]
        better = [m["diff"] < 0 for m in ms]
        se_ok = all(-m["diff"] >= m["se"] for m in res["holdout"].values() if m["diff"] != 0)
        res["better_on"] = f"{sum(better)} of {len(better)} measures it changes"
        res["kept"] = bool(all(better) and se_ok)
        res["verdict"] = "Kept (provisional)." if res["kept"] else ("Better on every measure but a holdout gain is under 1 SE: a trade-off for Ryan."
                                                                     if all(better) else "Mixed: a trade-off for Ryan, not kept by the rule.")
    bt = {v: cands[v]["bettime"] for v in cands}
    res["reported_likeliest_list"] = {"2019-25": likeliest(now["bettime"], bt, HOLD), "2026": likeliest(now["bettime"], bt, (2026, 2026))}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
