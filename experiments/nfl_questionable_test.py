"""Questionable players on the NFL board: the chance he sits (pre-set rule, Nov 9, 2026).

    python experiments/nfl_questionable_test.py [--table path] [--out path]

The rule was approved before this script was written (Project doc
claude/plan-nfl-questionable-test-2026-11-09.md). In short:

B0   a live-like backtest. Every game is projected the way the board projects an upcoming
     game: expected to play = rostered (active or game-day inactive) and not listed Out or
     Doubtful; the inactive list is not used. History and fitting still use who did play,
     as on the board. New projections (players_ml) and the Questionable multiplier as live.
V1   a Questionable RB/WR/TE keeps c of his projected receptions, receiving yards, carries
     and rushing yards, c = the share of earlier seasons' Questionable RB/WR/TE who played;
     the rest goes to his teammates in the game who are not on the report, in proportion to
     their own projection (as cfbmodel.players.shade_for_availability).
V2   as V1, c by position x last practice status, shrunk toward V1's rate by n / (n + 200).
V3   the Questionable multiplier refit with players who sat removed (played = an offensive
     snap or a stat), scored on Questionable players who played.

Played (the outcome and the rates) = an offensive snap or any stat. Tuning 2014-2018,
holdout 2019 on. With the private PFF data checked out (.pff-data/, as in the Update board
Action) the PFF quarterback prior and the route inputs are on, as live; without it both are
off. The defense-vs-position step is not applied and the game model runs without role
signals; all of that is the same for B0 and every variant. Prints aggregates only.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "nfl_questionable_test.json"
GROUPS = ("RB", "WR", "TE")
SHADE_STATS = ("receptions", "rec_yds", "carries", "rush_yds")
MEASURES = (("rec_yds", ("WR", "TE", "RB")), ("rush_yds", ("RB",)), ("receptions", ("WR", "TE", "RB")), ("carries", ("RB",)))
SHRINK = 200
PERIODS = (("tuning 2014-18", 2014, C.HOLDOUT_FROM - 1), ("holdout 2019-26", C.HOLDOUT_FROM, 2100))
SHADE_PERIODS = (("tuning 2015-18", 2015, C.HOLDOUT_FROM - 1), ("holdout 2019-26", C.HOLDOUT_FROM, 2100))
PROP_MARKETS = {"player_receptions": "receptions", "player_reception_yds": "rec_yds", "player_rush_yds": "rush_yds"}


# ---------------------------------------------------------------------- pure pieces (tested)
def listed_active(status: pd.Series, report: pd.Series) -> pd.Series:
    """Who the board would expect to play before the inactive list: on the roster as active or
    a game-day inactive (or credited with a stat and on no roster), and not Out or Doubtful."""
    ok = status.isin(["ACT", "INA"]) | status.isna()
    return (ok & ~report.isin(["Out", "Doubtful"])).astype(int)


def questionable_mask(p: pd.DataFrame) -> pd.Series:
    return (p["report"] == "Questionable") & p["group"].isin(GROUPS) & (p["listed"] == 1)


def play_rates(hist: pd.DataFrame, shrink: float = SHRINK) -> tuple[float, dict]:
    """V1's one rate and V2's {(group, practice_lvl): rate}, from earlier Questionable players
    (rows with played_snap). Practice levels 0 full, 1 limited, 2 did not practice."""
    if hist.empty:
        return float("nan"), {}
    v1 = float(hist["played_snap"].mean())
    g = hist[hist["practice_lvl"].isin([0, 1, 2])].groupby(["group", "practice_lvl"])["played_snap"].agg(["mean", "size"])
    v2 = {k: float(v1 + r["size"] / (r["size"] + shrink) * (r["mean"] - v1)) for k, r in g.iterrows()}
    return v1, v2


def shade(p: pd.DataFrame, chance: pd.Series, share: bool = True) -> pd.DataFrame:
    """Each listed player keeps chance of the four stats; with share, the rest goes to his
    teammates in the game with no listing who are expected to play, in proportion to their own
    projection (without it, the rest is dropped)."""
    p = p.copy()
    c = chance.reindex(p.index)
    free = p["report"].isna() & (p["listed"] == 1)
    grp = [p["game_id"], p["team"]]
    for s in SHADE_STATS:
        col = s + "_proj"
        src = p[col].astype(float).fillna(0.0)
        new = src * c.fillna(1.0)
        lost = (src - new).groupby(grp).transform("sum")
        base = src.where(free, 0.0)
        tot = base.groupby(grp).transform("sum")
        p[col] = new + ((lost * base / tot.replace(0, np.nan)).fillna(0.0) if share else 0.0)
    return p


def fit_lambda(proj, chance, y) -> float:
    """S3's partial shade: the lambda in proj x (1 - lambda (1 - chance)) with the least squared
    error against y (least squares, clipped to 0-1). nan with nothing to fit."""
    proj, chance, y = (np.asarray(v, float) for v in (proj, chance, y))
    a = proj * (1 - chance)
    den = float((a * a).sum())
    if den <= 0:
        return float("nan")
    return float(np.clip((a * (proj - y)).sum() / den, 0.0, 1.0))


def partial(p: pd.DataFrame, chance: pd.Series, lam: dict) -> pd.DataFrame:
    """Each listed player's four stats x (1 - lambda (1 - chance)), lambda per stat; teammates
    unchanged."""
    p = p.copy()
    c = chance.reindex(p.index)
    for s in SHADE_STATS:
        col = s + "_proj"
        k = 1 - lam.get(s, 0.0) * (1 - c.fillna(1.0))
        p[col] = p[col].astype(float).fillna(0.0) * k
    return p


def paired(e_new, e_old, groups) -> tuple[float, float]:
    """Mean of e_new - e_old and its SE clustered by group."""
    diff = pd.Series(np.asarray(e_new, float) - np.asarray(e_old, float))
    sums = diff.groupby(np.asarray(groups)).sum()
    n, k = len(diff), len(sums)
    se = float(np.sqrt(k / (k - 1) * (sums ** 2).sum()) / n) if k > 1 else float("nan")
    return float(diff.mean()), se


def rmse_block(b0: pd.DataFrame, new: pd.DataFrame, mask: pd.Series) -> dict:
    out = {}
    for stat, pos in MEASURES:
        m = (mask & b0["group"].isin(pos)).to_numpy()
        if not m.any():
            out[stat] = {"n": 0}
            continue
        y = b0[stat].to_numpy(float)[m]
        e0 = (b0[stat + "_proj"].to_numpy(float)[m] - y) ** 2
        e1 = (new[stat + "_proj"].to_numpy(float)[m] - y) ** 2
        d, se = paired(e1, e0, b0["game_id"].to_numpy()[m])
        out[stat] = {"n": int(m.sum()), "rmse_b0": round(float(np.sqrt(e0.mean())), 4), "rmse_new": round(float(np.sqrt(e1.mean())), 4),
                     "mse_diff": round(d, 4), "se": round(se, 4), "z": round(d / se, 2) if se and se > 0 else 0.0}
    return out


def better_all(block: dict) -> bool:
    v = [r["mse_diff"] < 0 for r in block.values() if r.get("n")]
    return bool(v) and all(v)


def worse_any(block: dict) -> bool:
    return any(r["mse_diff"] > 0 for r in block.values() if r.get("n"))


def verdict(res: dict) -> str:
    """res: {period: {"all": block, "listed": block}}. Kept: better on all four over every
    player-game in both periods and not worse on the listed players alone in either.
    Not kept: worse on all four in both periods. Otherwise mixed."""
    allp = [res[p]["all"] for p in res]
    if all(better_all(b) for b in allp) and not any(worse_any(res[p]["listed"]) for p in res):
        return "kept"
    if all(not any(r["mse_diff"] < 0 for r in b.values() if r.get("n")) for b in allp):
        return "not kept"
    return "mixed"


# ---------------------------------------------------------------------- building the projections
def load_table(path: Path | None) -> pd.DataFrame:
    path = path or C.RAW / "nflq_table.pkl"
    if path.exists():
        return pd.read_pickle(path)
    from nflmodel import data, extra, injuries, model, players as PL, players_data as PD
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    prior = None
    try:                                     # PFF quarterback prior, as the board (None without PFF data)
        from nflmodel import pff_qb
        prior = pff_qb.load(qb, sorted(games.season.unique()))
    except Exception as exc:  # noqa: BLE001
        print(f"  PFF quarterback prior off ({exc})")
    print(f"  quarterback prior: {'PFF' if prior is not None else 'flat'}", flush=True)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    table = PL.build(frame, pg, info, PL.load_availability(verbose=False), inj, injuries.load_pass_rush(games, verbose=False), games=games)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(path)
    return table


def snap_played(table: pd.DataFrame) -> pd.Series:
    """An offensive snap or any stat in the game."""
    on = set()
    for s in sorted(table["season"].unique()):
        f = C.CACHE / f"snaps_{int(s)}.csv"
        if f.exists():
            sn = pd.read_csv(f, usecols=["game_id", "player_id", "offense_snaps"])
            sn = sn[sn["offense_snaps"] > 0]
            on |= set(zip(sn["game_id"], sn["player_id"]))
    stat = table[["targets", "carries", "pass_att", "receptions"]].fillna(0).sum(axis=1) > 0
    snap = pd.Series([(g, p) in on for g, p in zip(table["game_id"], table["player_id"])], index=table.index)
    return (snap | stat).astype(int)


def projections(table: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """B0 and V3 for each season: the season's rows projected live-like, everything fitted on
    earlier seasons as the board does (who did play)."""
    from nflmodel import data, players as PL, players_ml as ML, roles
    games = data.load_games()
    R = roles.load_all(games, verbose=False)
    truth = PL.walk_forward(table, first_test=C.FIRST_TEST_SEASON)
    truth_f = ML.add_features(truth, R)
    feats = ML.feats_for(truth_f)
    print(f"  route inputs: {'on (PFF)' if feats != list(ML.FEATS) else 'off'}", flush=True)
    feat_cols = [c for c in truth_f.columns if c not in truth.columns]
    key = ["game_id", "player_id"]
    fmap = truth_f[key + feat_cols].drop_duplicates(key)
    out = []
    for s in seasons:
        t = table.copy()
        t["active_pre"] = np.where(t["season"] == s, t["listed"], t["active"])
        cur = PL.walk_forward(t, first_test=C.FIRST_TEST_SEASON, only={s})
        cur = cur.merge(fmap, on=key, how="left")
        both = pd.concat([truth_f[truth_f["season"] < s], cur], ignore_index=True)
        ml = ML.walk_forward(both, first=s, feats=feats)
        b0 = ML.questionable(ml, first=s)
        clean = ml.copy()
        clean["active"] = np.where(clean["season"] < s, clean["played_snap"], clean["active"])
        v3 = ML.questionable(clean, first=s)
        keep = b0["season"] == s
        x = b0[keep].copy()
        for st in ML.Q_STATS:
            x[st + "_v3"] = v3.loc[keep, st + "_proj"].to_numpy()
        out.append(x)
        print(f"  {s}: {int(keep.sum())} player-games", flush=True)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------- props check
def props_check(p: pd.DataFrame, variants: dict) -> dict:
    """Share of correct over/under calls at the pricing-snapshot main line (median across
    DraftKings and FanDuel), for players who played. variants: name -> frame like p."""
    from nflmodel import propbackfill as B, propmarket as PM
    info = pd.concat([pd.read_csv(f, dtype={"player_id": str}) for f in sorted(C.CACHE.glob("player_info_*.csv"))]).drop_duplicates("player_id", keep="last")
    names = info.set_index("player_id")["name"].map(PM.norm_name)
    res = {}
    for season in range(2023, 2027):
        try:
            snaps = B.load(season, "flag")
        except Exception:
            continue
        if snaps is None or snaps.empty:
            continue
        s = snaps[snaps["market"].isin(PROP_MARKETS) & snaps["book"].isin(["draftkings", "fanduel"])].copy()
        s["nname"] = s["player"].map(PM.norm_name)
        lines = s.groupby(["game_id", "nname", "market"])["line"].median().reset_index()
        res[season] = lines
    if not res:
        return {"note": "no historical prop snapshots found"}
    lines = pd.concat(res.values(), ignore_index=True)
    lines["stat"] = lines["market"].map(PROP_MARKETS)
    base = p.copy()
    base["nname"] = base["player_id"].map(names)
    q = questionable_mask(base)
    team_has_q = q.groupby([base["game_id"], base["team"]]).transform("any")
    base["who"] = np.where(q, "questionable", np.where(team_has_q & base["report"].isna(), "teammate", "other"))
    base = base[(base["who"] != "other") & (base["played_snap"] == 1) & base["played"].astype(bool)]
    base = base[~base.duplicated(["game_id", "nname"], keep=False)]
    m = lines.merge(base, on=["game_id", "nname"], how="inner")
    out = {}
    for name, frame in variants.items():
        f = frame.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])
        rows = []
        for st in set(PROP_MARKETS.values()):
            x = m[m["stat"] == st]
            if x.empty:
                continue
            col = st + "_v3" if name == "V3" else st + "_proj"
            proj = f[col].reindex(pd.MultiIndex.from_frame(x[["game_id", "player_id"]])).to_numpy(float)
            y = x[st].to_numpy(float)
            ln = x["line"].to_numpy(float)
            keep = (y != ln) & ~np.isnan(proj)
            rows.append(pd.DataFrame({"who": x["who"].to_numpy()[keep], "stat": st,
                                      "correct": ((proj > ln) == (y > ln))[keep]}))
        r = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["who", "stat", "correct"])
        out[name] = {w: {"n": int(len(g)), "correct": round(float(g["correct"].mean()), 4) if len(g) else None}
                     for w, g in r.groupby("who")}
    return out


def rate_history(table: pd.DataFrame) -> pd.DataFrame:
    """Every Questionable RB/WR/TE expected to play, with his last practice status (the board's
    practice caches) and whether he played; all seasons, for the walk-forward rates."""
    h = table[questionable_mask(table) & table["played"].astype(bool)][["season", "week", "team", "player_id", "group", "played_snap"]].copy()
    pr = []
    for s in sorted(h["season"].unique()):
        f = C.CACHE / f"practice_{int(s)}.csv"
        if f.exists():
            pr.append(pd.read_csv(f, usecols=["season", "week", "team", "player_id", "practice"]))
    if pr:
        pr = pd.concat(pr).drop_duplicates(["season", "week", "team", "player_id"], keep="last")
        pr[["season", "week"]] = pr[["season", "week"]].astype(int)
        h = h.merge(pr.rename(columns={"practice": "practice_lvl"}), on=["season", "week", "team", "player_id"], how="left")
    else:
        h["practice_lvl"] = np.nan
    return h


# ---------------------------------------------------------------------- main
def shade_test(p: pd.DataFrame, q: pd.Series, c2: pd.Series) -> dict:
    """The second pre-set test (Project doc claude/plan-nfl-questionable-shade-test-2026-11-09.md):
    S1-S3 against B1, the board as live since Nov 9 (the clean multiplier)."""
    b1 = p.copy()
    for st in SHADE_STATS:
        b1[st + "_proj"] = p[st + "_v3"]
    s1, s2 = shade(b1, c2), shade(b1, c2, share=False)
    s3 = b1.copy()
    lams = {}
    for s in sorted(b1["season"].unique()):
        te = q & (b1["season"] == s)
        tr = q & (b1["season"] < s) & (b1["season"] >= C.FIRST_TEST_SEASON) & c2.notna()
        if not te.any() or not tr.any():
            continue
        lam = {}
        for st, pos in MEASURES:
            m = tr & b1["group"].isin(pos)
            lam[st] = fit_lambda(b1.loc[m, st + "_proj"], c2[m], b1.loc[m, st])
        lam = {k: v for k, v in lam.items() if v == v}
        lams[int(s)] = {k: round(v, 3) for k, v in lam.items()}
        part = partial(b1[te], c2[te], lam)
        for st in SHADE_STATS:
            s3.loc[te, st + "_proj"] = part[st + "_proj"]
    s3_ok = b1["season"].isin(list(lams))
    team_q = q.groupby([b1["game_id"], b1["team"]]).transform("any")
    res = {"lambda": lams, "n_questionable": {}, "S1": {}, "S2": {}, "S3": {}}
    for label, lo, hi in SHADE_PERIODS:
        per = (b1["season"] >= lo) & (b1["season"] <= hi) & s3_ok
        univ = per & team_q
        res["n_questionable"][label] = {"listed": int((q & per).sum()), "played": int((q & per & (b1["played_snap"] == 1)).sum())}
        for name, new in (("S1", s1), ("S2", s2), ("S3", s3)):
            res[name][label] = {"all": rmse_block(b1, new, univ), "listed": rmse_block(b1, new, univ & q)}
    for name in ("S1", "S2", "S3"):
        res[name]["verdict"] = verdict({k: v for k, v in res[name].items() if k != "verdict"})
    kept = [n for n in ("S1", "S2", "S3") if res[n]["verdict"] == "kept"]
    if len(kept) > 1:
        hold = SHADE_PERIODS[-1][0]
        res["winner"] = min(kept, key=lambda n: res[n][hold]["all"]["rec_yds"]["rmse_new"])
    print("Props check ...", flush=True)
    pc = props_check(b1, {"B1": b1, "S1": s1})
    if "note" in pc:
        res["props"] = pc
    else:
        res["props"] = {"teammates": {n: pc[n].get("teammate") for n in ("B1", "S1")}, "S2": "no prop change", "S3": "no prop change"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--shade", action="store_true", help="the second test: S1-S3 on the live (clean) multiplier")
    args = ap.parse_args()
    table = load_table(args.table)
    table["listed"] = listed_active(table["status"], table["report"])
    table["played_snap"] = snap_played(table)
    seasons = sorted(int(s) for s in table["season"].unique() if s >= C.FIRST_TEST_SEASON)
    print("Projecting live-like, season by season ...", flush=True)
    p = projections(table, seasons)
    p = p[p["played"].astype(bool)].reset_index(drop=True)
    for st in SHADE_STATS + ("targets",):
        p[st] = p[st].fillna(0.0)

    # the chance to play, walk-forward
    q = questionable_mask(p)
    hist_all = rate_history(table)
    c1 = pd.Series(np.nan, index=p.index)
    c2 = pd.Series(np.nan, index=p.index)
    rates = {}
    for s in seasons:
        v1, v2 = play_rates(hist_all[hist_all["season"] < s])
        if v1 != v1:
            continue
        te = q & (p["season"] == s)
        c1[te] = v1
        c2[te] = [v2.get((g, l), v1) for g, l in zip(p.loc[te, "group"], p.loc[te, "practice_lvl"])]
        rates[s] = {"V1": round(v1, 3), "V2": {f"{g} {['full', 'limited', 'DNP'][int(l)]}": round(v, 3) for (g, l), v in v2.items()}}
    if args.shade:
        res = shade_test(p, q, c2)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res, indent=1, default=float))
        return
    p1, p2 = shade(p, c1), shade(p, c2)

    team_q = q.groupby([p["game_id"], p["team"]]).transform("any")
    res = {"rates": rates, "n_questionable": {}, "V1": {}, "V2": {}, "V3": {}}
    for label, lo, hi in PERIODS:
        per = (p["season"] >= lo) & (p["season"] <= hi) & p["season"].isin(list(rates))
        univ = per & team_q
        res["n_questionable"][label] = {"listed": int((q & per).sum()), "played": int((q & per & (p["played_snap"] == 1)).sum())}
        for name, new in (("V1", p1), ("V2", p2)):
            res[name][label] = {"all": rmse_block(p, new, univ), "listed": rmse_block(p, new, univ & q)}
        v3 = p.copy()
        for st in SHADE_STATS:
            v3[st + "_proj"] = p[st + "_v3"]
        res["V3"][label] = rmse_block(p, v3, per & q & (p["played_snap"] == 1))
    for name in ("V1", "V2"):
        res[name]["verdict"] = verdict({k: v for k, v in res[name].items() if k != "verdict"})
    v3ok = [better_all(res["V3"][lbl]) for lbl, _, _ in PERIODS]
    res["V3"]["verdict"] = "kept" if all(v3ok) else ("not kept" if not any(v3ok) else "mixed")
    print("Props check ...", flush=True)
    res["props"] = props_check(p, {"B0": p, "V1": p1, "V2": p2, "V3": p})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
