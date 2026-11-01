"""Defense against each position: does it make player projections better, and does it beat the books?

    python experiments/def_vs_position.py

The player model uses one number per defense for receiving (it was dropped: it did not help
receivers) and team-level yards per carry and per attempt for rushing and passing (PLAYERS.md).
The matchup dashboards split pass defense by position: yards allowed to wide receivers, tight
ends and running backs. This builds that split and tests it twice.

Rating. For every defense-game, the yards it allowed to a position (receiving yards to WRs, to
TEs and to RBs; rushing yards to RBs; passing yards to QBs; receptions to each) over the yards
the board's own projections expected those players to get (players_proj.pkl, walk-forward).
The projections already know each offense's usage and quality, so the ratio is opponent-adjusted.
A defense's rating before a game is the log of that ratio averaged over its earlier games
(6-game half-life, last season's games at half weight), shrunk toward 0 with SHRINK pseudo-games.
Only games before the one rated are used.

Test 1, accuracy (tuning 2013-2018, holdout 2019 on). projection x exp(b x rating), one b per
position and stat fitted by least squares on the tuning seasons; the holdout's average miss
against the projection alone, with a standard error clustered by game.

Test 2, against the books (2023-2025 DraftKings and FanDuel main lines, props_blend's priced
frames). logit(chance) = logit(book's vig-free chance) + c x rating (signed for the over), one
c per stat fitted on 2023-24, scored on 2025: Brier against the book's, and the sides it moves
2+ points over the book with positive EV at the book's price, graded with closing line value
carried to the bet's line (props_blend.flagged).

Rule, set before running:
  into the projections   test 1 holdout gain > 2 standard errors for that position and stat
                         (and no worse on the tuning seasons, where it was fitted)
  counted as an edge     test 2 on 2025: Brier better than the book's by > 2 SE AND flagged
                         sides' carried CLV above 0 by > 2 SE
  shown on the page      always (it is a fact about the matchup, not a bet)

Writes experiments/out/def_vs_position.json.
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
import props_blend as PB  # noqa: E402
from nflmodel import config as C  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "def_vs_position.json"
HALF_LIFE = 6.0
LAST_SEASON = 0.5
SHRINK = 4.0
TUNE = list(range(2013, C.HOLDOUT_FROM))
# (position group, stat): the props a matchup could move
PAIRS = [("WR", "rec_yds"), ("TE", "rec_yds"), ("RB", "rec_yds"), ("WR", "receptions"), ("TE", "receptions"),
         ("RB", "receptions"), ("RB", "rush_yds"), ("QB", "pass_yds"), ("RB", "carries")]


def _se(v: pd.Series, g: pd.Series) -> float:
    return PB._se_by_game(v, g)


def ratings(proj: pd.DataFrame) -> pd.DataFrame:
    """rating per (game_id, defense=opp, group, stat), from games before it."""
    p = proj[(proj["played"].astype(bool)) & (proj["active"] == 1)].copy()
    out = []
    for grp, stat in PAIRS:
        col = f"{stat}_proj"
        d = p[(p["group"] == grp) & p[col].notna()]
        if grp == "QB":
            d = d[d["att_share"] > 0.5]
        g = d.groupby(["game_id", "season", "week", "opp"]).agg(act=(stat, "sum"), exp=(col, "sum")).reset_index()
        g = g[g["exp"] > 1]
        g["lr"] = np.log((g["act"].clip(lower=0) + 1) / (g["exp"] + 1))      # a team can lose yards; floor at 0
        g = g.sort_values(["season", "week"])
        rows = []
        for dfn, h in g.groupby("opp"):
            h = h.reset_index(drop=True)
            for i in range(len(h)):
                prev = h.iloc[:i]
                cur = h.iloc[i]
                prev = prev[prev["season"] >= cur["season"] - 1]
                if len(prev):
                    same = prev["season"] == cur["season"]
                    k = np.arange(len(prev))[::-1]               # games ago
                    w = 0.5 ** (k / HALF_LIFE) * np.where(same, 1.0, LAST_SEASON)
                    r = float((w * prev["lr"]).sum() / (w.sum() + SHRINK))
                else:
                    r = 0.0
                rows.append((cur["game_id"], dfn, grp, stat, r))
        out.append(pd.DataFrame(rows, columns=["game_id", "opp", "group", "stat", "rating"]))
    return pd.concat(out, ignore_index=True)


def accuracy(proj: pd.DataFrame, rt: pd.DataFrame) -> list[dict]:
    p = proj[(proj["played"].astype(bool)) & (proj["active"] == 1)]
    res = []
    for grp, stat in PAIRS:
        col = f"{stat}_proj"
        d = p[(p["group"] == grp) & p[col].notna() & (p[col] > 0)]
        if grp == "QB":
            d = d[d["att_share"] > 0.5]
        d = d.merge(rt[(rt.group == grp) & (rt.stat == stat)][["game_id", "opp", "rating"]], on=["game_id", "opp"], how="left")
        d["rating"] = d["rating"].fillna(0.0)
        tr, ho = d[d.season.isin(TUNE)], d[d.season >= C.HOLDOUT_FROM]
        # least squares for b in y = proj x exp(b r) ~ proj x (1 + b r)
        x = tr[col] * tr["rating"]
        b = float((x * (tr[stat] - tr[col])).sum() / (x ** 2).sum()) if (x ** 2).sum() > 0 else 0.0
        def gain(t):
            adj = t[col] * np.exp(b * t["rating"])
            diff = (adj - t[stat]).abs() - (t[col] - t[stat]).abs()
            return float(diff.mean()), _se(diff, t["game_id"]), float((t[col] - t[stat]).abs().mean())
        gt, st, mt = gain(tr)
        gh, sh, mh = gain(ho)
        res.append({"group": grp, "stat": stat, "b": round(b, 3), "n_holdout": int(len(ho)),
                    "rating_sd": round(float(ho["rating"].std()), 3),
                    "tune_mae": round(mt, 3), "tune_gain": round(gt, 4), "tune_se": round(st, 4),
                    "holdout_mae": round(mh, 3), "holdout_gain": round(gh, 4), "holdout_se": round(sh, 4),
                    "into_projections": bool(gh < -2 * sh and gt <= 0)})
    return res


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def vs_books(rt: pd.DataFrame, proj: pd.DataFrame) -> dict:
    cache = C.RAW / "props_blend_priced.pkl"
    priced = pd.read_pickle(cache)
    stats = sorted({s for _, s in PAIRS})
    def attach(fl):
        fl = fl[fl["stat"].isin(stats) & fl["group"].isin(["WR", "TE", "RB", "QB"])].copy()
        return fl.merge(rt, on=["game_id", "opp", "group", "stat"], how="left").assign(rating=lambda x: x["rating"].fillna(0.0))
    tr = PB.frame(attach(pd.concat([priced[s][0] for s in (2023, 2024)], ignore_index=True)))
    te_fl = attach(priced[2025][0])
    te = PB.frame(te_fl)
    cl = priced[2025][1]
    coefs, rows = {}, []
    for st in stats:
        a = tr[tr["stat"] == st]
        if len(a) < 300:
            continue
        # one-parameter logistic with the book as offset: Newton steps on c
        c = 0.0
        z0, r, y = logit(a["mkt"].to_numpy()), a["rating"].to_numpy(), a["hit"].to_numpy()
        for _ in range(25):
            p = 1 / (1 + np.exp(-(z0 + c * r)))
            g, h = float((r * (y - p)).sum()), float((r * r * p * (1 - p)).sum())
            if h <= 0:
                break
            c += g / h
        coefs[st] = c
    def adjusted(x):
        c = x["stat"].map(coefs).fillna(0.0)
        return 1 / (1 + np.exp(-(logit(x["mkt"]) + c * x["rating"])))
    for st, g in te.groupby("stat"):
        p = adjusted(g)
        d = (p - g["hit"]) ** 2 - (g["mkt"] - g["hit"]) ** 2
        rows.append({"stat": st, "n": int(len(g)), "c": round(coefs.get(st, 0.0), 3), "brier_minus_book": round(float(d.mean()), 6),
                     "se": round(_se(d, g["game_id"]), 6)})
    p = adjusted(te)
    d = (p - te["hit"]) ** 2 - (te["mkt"] - te["hit"]) ** 2
    pooled = {"n": int(len(te)), "brier_minus_book": round(float(d.mean()), 6), "se": round(_se(d, te["game_id"]), 6)}
    # flagged sides: the adjusted chance in place of ours
    fl = te_fl.copy()
    rest = 1 - fl["our_push"]
    pa = 1 / (1 + np.exp(-(logit(fl["fair_over"]) + fl["stat"].map(coefs).fillna(0.0) * fl["rating"])))
    fl["our_over"], fl["our_under"] = pa * rest, (1 - pa) * rest
    from props_calibration import FAMILY
    from nflmodel import props as P
    m = P.PropModel(FAMILY).fit(proj[(proj.season < 2025) & (proj.season >= 2013)])
    saved = P.PropModel.load()
    if saved is not None:
        m.iso = saved.iso
    flagged = PB.flagged(fl, cl, 1.0, 0.02, m)
    passed = pooled["brier_minus_book"] < -2 * pooled["se"] and bool(flagged.get("clv_carried_se")) and \
        (flagged.get("clv_carried") or 0) > 2 * (flagged.get("clv_carried_se") or 1)
    return {"coefs_2023_24": {k: round(v, 3) for k, v in coefs.items()}, "by_stat_2025": rows, "pooled_2025": pooled,
            "flagged_2025": flagged, "counted_as_edge": bool(passed)}


def main():
    from props_calibration import load_proj
    proj = load_proj(None)
    print("ratings ...", flush=True)
    rt = ratings(proj)
    assert rt["rating"].notna().all()
    rt.to_pickle(C.RAW / "def_vs_position.pkl")
    print("test 1: accuracy", flush=True)
    acc = accuracy(proj, rt)
    print(pd.DataFrame(acc).to_string(index=False))
    print("test 2: against the books", flush=True)
    vb = vs_books(rt, proj)
    print(json.dumps(vb, indent=1, default=float))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"settings": {"half_life": HALF_LIFE, "last_season": LAST_SEASON, "shrink": SHRINK},
                               "accuracy": acc, "vs_books": vb}, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
