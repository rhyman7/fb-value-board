"""Pre-set test: shade Questionable / Game Time Decision / Probable players' college projections by their
chance to play (Project doc claude/plan-questionable-gtd-test-2026-11-09.md, approved Nov 9, 2026 with
V2b as QB vs non-QB; the rule was locked before this script was written).

    python experiments/cfb_availability_test.py [--season 2026]

Baseline B0: the board's archived projections (docs/cfb/archive, frozen at kickoff) with the official
Out / Doubtful designations applied. Variants (each on top of B0):
  V1   conference definitions: Probable .75, Questionable .50, Game Time Decision .50, Out (1st Half) .50
  V2   played rate by last status, fitted on weeks 0-6 (PFF snaps where available, else box scores)
  V2b  as V2 by status x {QB, non-QB}; a cell under 10 cases uses the V2 rate
  V3   props only: a GTD player's projection given he plays x the weeks 0-6 median snaps-when-played
A shaded player's lost receptions / receiving yards / carries / rushing yards go to his undesignated
teammates in proportion to their own projection of that stat.

Prints aggregates only (the rates come from PFF, which stays private): run by the Availability grade
Action with run=test. Writes nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from cfbmodel import config as C, designations as DES  # noqa: E402
import availability_grade as AG  # noqa: E402

STATS = ["receptions", "rec_yds", "carries", "rush_yds"]
V1 = {"Probable": 0.75, "Questionable": 0.50, "Game Time Decision": 0.50, "Out (1st Half)": 0.50}
SHADED = set(V1)
OUT = {"Out", "Doubtful"}
FIT_MAX_WEEK, TEST_MIN_WEEK, MIN_CELL = 6, 7, 10
MEASURES = [("rec_yds", ("WR", "TE", "RB")), ("rush_yds", ("RB",)), ("receptions", ("WR", "TE", "RB")), ("carries", ("RB",))]


# ---- data ------------------------------------------------------------------------------------------
def archived(season: int) -> pd.DataFrame:
    """One row per archived player-game in a played game: board ids, school, pos, projection, actual
    (0 for a projected player with no box line)."""
    teams = pd.read_csv(C.CACHE / f"teams_{season}.csv")
    school = dict(zip(teams["abbr"], teams["school"]))
    rows = []
    for f in sorted((C.DOCS / "archive").glob(f"{season}_w*.json")):
        a = json.loads(f.read_text())
        gs = a.get("games") or {}
        espn = {g["id"]: g.get("game_id") for g in (gs.values() if isinstance(gs, dict) else gs)}
        results, actuals = a.get("results") or {}, a.get("actuals") or {}
        for gid, plist in (a.get("players") or {}).items():
            if gid not in results:
                continue                          # not played (or not graded yet)
            act = actuals.get(gid) or {}
            for p in plist:
                x = act.get(str(p["id"])) or {}
                rows.append({"board_game": gid, "game_id": espn.get(gid), "week": int(p["week"]), "abbr": p["team"],
                             "team": school.get(p["team"], p["team"]), "player_id": str(p["id"]), "name": p["name"],
                             "pos": str(p.get("pos") or "").upper(), "key": DES.norm(p["name"]),
                             **{s + "_proj": float((p.get("proj") or {}).get(s) or 0.0) for s in STATS},
                             **{s: float(x.get(s) or 0.0) for s in STATS}, "dnp": not x})
    d = pd.DataFrame(rows)
    d["game_id"] = pd.to_numeric(d["game_id"], errors="coerce")
    return d


# ---- mechanics ---------------------------------------------------------------------------------------
def shade(d: pd.DataFrame, chance: pd.Series, col: str = "proj") -> pd.DataFrame:
    """Multiply each player's projected stats by chance (1 for undesignated players), redistributing the
    loss to undesignated teammates in the same team-game in proportion to their own projection."""
    o = d.copy()
    c = chance.reindex(o.index).fillna(1.0)
    free = o["status"].isna()                     # teammates with no designation take the share
    for s in STATS:
        src = o[f"{s}_{col}"] if f"{s}_{col}" in o.columns else o[s + "_proj"]
        new = src * c
        lost = (src - new).groupby([o["board_game"], o["abbr"]]).transform("sum")
        base = src.where(free, 0.0)
        tot = base.groupby([o["board_game"], o["abbr"]]).transform("sum")
        add = np.where(tot > 0, lost * base / tot.replace(0, np.nan), 0.0)
        o[f"{s}_new"] = new + np.nan_to_num(add)
    return o


def paired(e_new, e_old, groups) -> tuple[float, float]:
    """Mean of e_new - e_old and its SE clustered by group."""
    diff = pd.Series(e_new - e_old)
    g = pd.Series(groups)
    n, sums = len(diff), diff.groupby(g.values).sum()
    k = len(sums)
    se = float(np.sqrt(k / (k - 1) * (sums ** 2).sum()) / n) if k > 1 else float("nan")
    return float(diff.mean()), se


def rmse_block(base: pd.DataFrame, new: pd.DataFrame, mask: pd.Series) -> dict:
    out = {}
    for stat, pos in MEASURES:
        m = mask & base["pos"].isin(pos)
        if m.sum() == 0:
            out[stat] = {"n": 0}
            continue
        e0 = (base.loc[m, f"{stat}_new"] - base.loc[m, stat]).to_numpy() ** 2
        e1 = (new.loc[m, f"{stat}_new"] - new.loc[m, stat]).to_numpy() ** 2
        diff, se = paired(e1, e0, base.loc[m, "board_game"].to_numpy())
        out[stat] = {"n": int(m.sum()), "rmse_b0": float(np.sqrt(e0.mean())), "rmse_new": float(np.sqrt(e1.mean())),
                     "mse_diff": diff, "se": se, "z": diff / se if se and se == se and se > 0 else 0.0}
    return out


def verdict(primary: dict, desig: dict) -> str:
    better = [primary[s]["mse_diff"] < 0 for s, _ in MEASURES if primary[s].get("n")]
    worse3 = [desig[s]["mse_diff"] > 0 for s, _ in MEASURES if desig[s].get("n")]
    if better and all(better) and not any(worse3):
        return "KEPT"
    if better and not any(better):
        return "NOT KEPT"
    return "MIXED"


# ---- main --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    a = ap.parse_args()
    today = dt.date.today()
    season = a.season or (today.year if today.month >= 8 else today.year - 1)

    arch = archived(season)
    print(f"archive: {len(arch)} player-games in played games, weeks {arch['week'].min()}-{arch['week'].max()}")
    des = AG.outcomes(season)
    des = des[des["last"] != ""]
    # one designation per (game, school, name): the matched game from the grade
    des = des.drop_duplicates(["game_id", "team", "key"])
    m = arch.merge(des[["game_id", "team", "key", "last", "played", "source", "snap_ratio"]], on=["game_id", "team", "key"], how="left")
    m["status"] = m["last"].where(m["last"].isin(SHADED | OUT))
    teams_reporting = des[["game_id", "team"]].drop_duplicates().assign(reporting=True)
    m = m.merge(teams_reporting, on=["game_id", "team"], how="left")
    m["reporting"] = m["reporting"].fillna(False).astype(bool)
    print(f"matched designations on archived players: " + ", ".join(f"{k} {v}" for k, v in m["status"].value_counts().items()))
    print(f"universe (team-games with a report): {int(m['reporting'].sum())} player-games, "
          f"{m.loc[m['reporting'], 'board_game'].nunique()} games")

    # B0: the official Out / Doubtful applied (what the live board does since Nov 9)
    raw = m.assign(**{f"{s}_new": m[s + "_proj"] for s in STATS})
    outc = pd.Series(np.where(m["status"].isin(OUT), 0.0, 1.0), index=m.index)
    b0 = shade(m.assign(status=m["status"].where(m["status"].isin(OUT))), outc)
    b0["status"] = m["status"]
    for s in STATS:                                # B0 is the base the variants shade
        b0[s + "_b0"] = b0[s + "_new"]
    n_out = int((m["status"].isin(OUT) & (m[[s + "_proj" for s in STATS]].sum(axis=1) > 0)).sum())
    U = b0["reporting"]
    info = rmse_block(raw, b0, U)
    print(f"\n[info] official Out/Doubtful players the archived board still projected: {n_out}")
    print("[info] B0 vs the raw archive (all player-games in the universe): " +
          "; ".join(f"{s} {v['rmse_b0']:.2f}->{v['rmse_new']:.2f} (z {v['z']:+.2f})" for s, v in info.items() if v.get("n")))

    # fitted rates (weeks 0-6)
    known = des[(des["source"] != "unknown") & (des["week"] <= FIT_MAX_WEEK) & des["last"].isin(SHADED)].copy()
    known["played"] = known["played"].astype(float)
    known["qb"] = known["pos"].fillna("").str.upper().eq("QB")
    v2 = known.groupby("last")["played"].mean().to_dict()
    v2n = known.groupby("last").size().to_dict()
    cell = known.groupby(["last", "qb"])["played"].agg(["mean", "size"])
    v2b = {}
    for st in SHADED:
        for q in (True, False):
            if (st, q) in cell.index and cell.loc[(st, q), "size"] >= MIN_CELL:
                v2b[(st, q)] = float(cell.loc[(st, q), "mean"])
            elif st in v2:
                v2b[(st, q)] = float(v2[st])
    gtd_ratio = known[(known["last"] == "Game Time Decision") & (known["played"] == 1)]["snap_ratio"].dropna()
    v3 = float(gtd_ratio.median()) if len(gtd_ratio) else None
    print("\nfitted on weeks 0-%d:" % FIT_MAX_WEEK)
    print("  V2 : " + "; ".join(f"{k} {v:.2f} (n {v2n[k]})" for k, v in sorted(v2.items())))
    print("  V2b: " + "; ".join(f"{k[0]}/{'QB' if k[1] else 'non-QB'} {v:.2f}" for k, v in sorted(v2b.items())))
    print(f"  V3 : GTD snaps-when-played median {v3 if v3 is None else round(v3, 3)} (n {len(gtd_ratio)})")

    def chance(rates: dict, by_qb: bool = False) -> pd.Series:
        st = b0["status"]
        if by_qb:
            q = b0["pos"].eq("QB")
            vals = [rates.get((s, bool(x)), np.nan) if s in SHADED else np.nan for s, x in zip(st, q)]
        else:
            vals = [rates.get(s, np.nan) if s in SHADED else np.nan for s in st]
        return pd.Series(vals, index=b0.index)

    base_for_variants = b0.assign(status=b0["status"].where(b0["status"].isin(SHADED)))
    variants = {"V1": chance(V1), "V2": chance(v2), "V2b": chance(v2b, by_qb=True)}
    results = {}
    late = b0["week"] >= TEST_MIN_WEEK
    designated = b0["status"].isin(SHADED)
    for name, c in variants.items():
        new = shade(base_for_variants, c, col="b0")
        new["status"] = b0["status"]
        b0n = b0.copy()                               # B0 compared in the same columns
        r = {}
        for period, mask in (("all", U), ("w7+", U & late)):
            if name != "V1" and period == "all":
                continue
            prim = rmse_block(b0n, new, mask)
            des3 = rmse_block(b0n, new, mask & designated)
            r[period] = {"primary": prim, "designated": des3, "verdict": verdict(prim, des3)}
        results[name] = r

    print("\n== Results (paired against B0; z < 0 = better; SE clustered by game)")
    for name, r in results.items():
        for period, x in r.items():
            print(f"\n{name} [{period}] -> {x['verdict']}")
            for grp in ("primary", "designated"):
                print(f"  {grp:10s} " + "; ".join(
                    f"{s} n{v['n']} {v['rmse_b0']:.3f}->{v['rmse_new']:.3f} (z {v['z']:+.2f})" for s, v in x[grp].items() if v.get("n")))

    # V3: GTD players who played, weeks 7+, projection given he plays
    g = b0[(b0["status"] == "Game Time Decision") & late & ~b0["dnp"]].copy()
    print(f"\n== V3 (props, given played): GTD players who played in weeks {TEST_MIN_WEEK}+: {len(g)}")
    if v3 is not None and len(g):
        for stat, _ in MEASURES:
            e0 = (g[stat + "_b0"] - g[stat]) ** 2
            e1 = (g[stat + "_b0"] * v3 - g[stat]) ** 2
            diff, se = paired(e1.to_numpy(), e0.to_numpy(), g["board_game"].to_numpy())
            print(f"  {stat}: rmse {np.sqrt(e0.mean()):.3f} -> {np.sqrt(e1.mean()):.3f} (z {diff / se if se and se > 0 else 0:+.2f})")
    # props check
    try:
        pl = pd.read_csv(C.DATA.parent / "props" / "cfb_prop_log.csv")
        pl = pl[(~pl["alt"].astype(str).str.lower().eq("true")) & pl["stat"].isin(["receptions", "rec_yds", "rush_yds"])]
        pl = pl.sort_values("logged").drop_duplicates(["game_id", "player_id", "stat"], keep="last")
        pl["player_id"] = pl["player_id"].astype(str)
        j = pl.merge(b0[designated & late & ~b0["dnp"]], left_on=["game_id", "player_id"], right_on=["board_game", "player_id"])
        print(f"  props check: {len(j)} main-line props for designated players who played in weeks {TEST_MIN_WEEK}+"
              + (" (too few to judge; the projection result decides)" if len(j) < 30 else ""))
        if len(j):
            act = np.array([r[r_stat] for r, r_stat in zip(j.to_dict("records"), j["stat"])])
            proj = np.array([r[s + "_b0"] for r, s in zip(j.to_dict("records"), j["stat"])])
            side0 = np.sign(proj - j["line"].to_numpy())
            truth = np.sign(act - j["line"].to_numpy())
            print(f"  B0 correct side: {(side0 == truth).mean():.0%}")
            gtd = j["status"].eq("Game Time Decision").to_numpy()
            if v3 is not None and gtd.any():
                side3 = np.sign(np.where(gtd, proj * v3, proj) - j["line"].to_numpy())
                print(f"  V3 correct side: {(side3 == truth).mean():.0%} ({int(gtd.sum())} GTD props)")
    except Exception as exc:  # noqa: BLE001
        print(f"  props check skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
