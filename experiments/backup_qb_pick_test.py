"""Which quarterback the player projections start when last game's starter is out (Nov 6, 2026).

    python experiments/backup_qb_pick_test.py

THE RULE (written before the first run, Nov 6, 2026)
-----------------------------------------------------
players._pick_starter: last game's starter if he is active (unless another active QB is clearly
the regular, 80%+ share); otherwise the active QB with the largest att_share_raw. A quarterback
with no NFL games gets a default share of half the average QB's (about 0.3), while a backup who has
been active all season with a handful of passes gets close to 0, so an unknown rookie outranks a
veteran backup (Nov 5, 2026: Carson Beck over Gardner Minshew for ARI). The candidates change only
the "otherwise" branch, the fallback games:
  current   largest att_share_raw (the board today)
  volume    largest decayed pass attempts (pass_att_sum), then att_share_raw
  starts    largest decayed games started (start_sum), then pass_att_sum, then att_share_raw
  known     largest att_share_raw, quarterbacks with no games (one_sum 0) ranked last

Fallback games: team-games where last game's starter is not among the team's pregame-active
quarterbacks. Seasons as the player walk-forward (players.walk_forward, first test 2013):
tuning 2013-2018, holdout 2019 on. Measures (four):
  1-2  pick accuracy in fallback games (the projected starter is the QB who started), tuning, holdout
  3-4  RMSE of the actual starter's passing-yards projection in fallback games, tuning, holdout
       (a starter the projection didn't pick is projected near 0)
The version is chosen on the tuning seasons (best tuning pick accuracy); KEEP RULE (CONVENTIONS.md):
kept only if better on all four and worse on none; a mixed result goes to Ryan.
Props check: QB passing-yards main lines at 2023-2025 DK/FD in fallback games are reported (the
books post them for the starter they expect, so the sample is small); judged only with 30+ lines.
Also reported: pick accuracy over all team-games (it can only change in fallback games).

Applied Nov 6, 2026: players._pick_starter now uses "volume", so on a rerun "current" and "volume"
are the same (fallback right 79.9% / 73.4%).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
from nflmodel import config as C, data, extra, injuries, model, players as PL, players_data as PD, pff_qb  # noqa: E402
from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine  # noqa: E402

TUNE, HOLD = (2013, 2018), 2019
ORIG = PL._pick_starter


def make_pick(key):
    def pick(d: pd.DataFrame):
        top = d.iloc[0]
        prev = d[d["player_id"] == top["prev_qb"]]
        if len(prev) and not (top["att_share_raw"] >= 0.8 and top["player_id"] != top["prev_qb"]):
            return prev.index[0]
        return key(d)
    return pick


VERSIONS = {
    "current": None,
    "volume": make_pick(lambda d: d.sort_values(["pass_att_sum", "att_share_raw"], ascending=False, na_position="last").index[0]),
    "starts": make_pick(lambda d: d.sort_values(["start_sum", "pass_att_sum", "att_share_raw"], ascending=False, na_position="last").index[0]),
    "known": make_pick(lambda d: d.assign(_k=(d["one_sum"].fillna(0) > 0).astype(int)).sort_values(["_k", "att_share_raw"], ascending=False).index[0]),
}


def table():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    inj = injuries.load_injuries(games, verbose=False)
    pr = injuries.load_pass_rush(games, verbose=False)
    eng = ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games,
                          qb_prior_model=pff_qb.load(qb, sorted(games.season.unique())))
    frame, _ = model.walk_forward(model.build_frame(games, eng, injuries=inj))
    pg, info = PD.load_player_games(verbose=False)
    av = PL.load_availability(verbose=False)
    return PL.build(frame, pg, info, av, inj, pr, games=games)


def fallback_keys(t: pd.DataFrame) -> set:
    act = (t["active_pre"] if "active_pre" in t else t["active"]) == 1
    q = t[(t["group"] == "QB") & act]
    have = q.groupby(["game_id", "team"]).apply(lambda d: d["prev_qb"].iloc[0] in set(d["player_id"]))
    return set(have[~have].index)


def score(p: pd.DataFrame, fb: set) -> dict:
    q = p[(p["group"] == "QB") & p["played"].astype(bool)].copy()
    q["key"] = list(zip(q["game_id"], q["team"]))
    st = q[q["started"] == 1].drop_duplicates(["game_id", "team"])
    pick = q[q["att_share"] >= 0.99][["game_id", "team", "player_id"]].rename(columns={"player_id": "picked"})
    st = st.merge(pick, on=["game_id", "team"], how="left")
    st["right"] = (st["picked"] == st["player_id"]).astype(float)
    st["fb"] = st["key"].isin(fb)
    st["err2"] = (st["pass_yds_proj"] - st["pass_yds"]) ** 2
    out = {}
    for name, m in (("tune", st.season.between(*TUNE)), ("hold", st.season >= HOLD)):
        f = st[m & st.fb]
        out[name] = {"n_fb": len(f), "acc_fb": round(float(f.right.mean()), 4), "rmse_fb": round(float(np.sqrt(f.err2.mean())), 3),
                     "acc_all": round(float(st[m].right.mean()), 4)}
    return out, st


def main():
    t = table()
    fb = fallback_keys(t)
    print(f"fallback team-games: {len(fb)}", flush=True)
    res, rows = {}, {}
    for name, fn in VERSIONS.items():
        PL._pick_starter = fn or ORIG
        p = PL.walk_forward(t)
        res[name], rows[name] = score(p, fb)
        r = res[name]
        print(f"{name:8s} tune: fallback right {r['tune']['acc_fb']:.1%} (n {r['tune']['n_fb']}), starter pass-yds RMSE {r['tune']['rmse_fb']:.1f}, all games {r['tune']['acc_all']:.2%}"
              f" | hold: {r['hold']['acc_fb']:.1%} (n {r['hold']['n_fb']}), RMSE {r['hold']['rmse_fb']:.1f}, all {r['hold']['acc_all']:.2%}", flush=True)
    PL._pick_starter = ORIG
    cand = {k: v for k, v in res.items() if k != "current"}
    pick = max(cand, key=lambda k: cand[k]["tune"]["acc_fb"])
    c, n = res["current"], res[pick]
    ms = [n["tune"]["acc_fb"] > c["tune"]["acc_fb"], n["hold"]["acc_fb"] > c["hold"]["acc_fb"],
          n["tune"]["rmse_fb"] < c["tune"]["rmse_fb"], n["hold"]["rmse_fb"] < c["hold"]["rmse_fb"]]
    print(f"\nchosen on tuning: {pick}; better on {sum(ms)} of 4 -> {'KEEP' if all(ms) else 'not kept' if not any(ms) else 'mixed'}")
    # changed picks, for reading
    a, b = rows["current"], rows[pick]
    j = a[["game_id", "team", "player_id", "picked", "right", "fb"]].merge(b[["game_id", "team", "picked", "right"]], on=["game_id", "team"], suffixes=("_c", "_n"))
    ch = j[j.picked_c != j.picked_n]
    print(f"picks changed: {len(ch)}; right before {int(ch.right_c.sum())}, after {int(ch.right_n.sum())}")
    print(ch[ch.game_id.str.startswith("2026")].to_string(index=False))
    props(t, fb, pick)


def props(t: pd.DataFrame, fb: set, pick: str) -> None:
    """Passing-yards main lines in fallback games, 2023-2025, pricing snapshot: Brier, current vs chosen."""
    import props_real_lines as RL
    from nflmodel import propbackfill as B, props as P
    from props_calibration import FAMILY
    saved = P.PropModel.load()
    got = {}
    for name in ("current", pick):
        PL._pick_starter = VERSIONS[name] or ORIG
        p = PL.walk_forward(t)
        parts = []
        for season in (2023, 2024, 2025):
            m = P.PropModel(FAMILY).fit(p[(p.season < season) & (p.season >= 2014)])
            m.iso = saved.iso
            x = RL.priced(p, m, B.load(season, "flag"), season)
            if not x.empty:
                x = x[(x.stat == "pass_yds") & (x.group == "QB")]
                parts.append(x[[k in fb for k in zip(x.game_id, x.team)]])
        got[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    PL._pick_starter = ORIG
    key = ["game_id", "player_id", "line", "book"]
    a, b = got["current"], got[pick]
    if a.empty or b.empty:
        print("props: no fallback-game passing lines")
        return
    j = a[key + ["y", "our_over", "our_under", "fair_over"]].merge(b[key + ["our_over", "our_under"]], on=key, suffixes=("_c", "_n"))
    j = j[j.y != j.line]
    hit = (j.y > j.line).astype(float)
    bc = (j.our_over_c / (j.our_over_c + j.our_under_c) - hit) ** 2
    bn = (j.our_over_n / (j.our_over_n + j.our_under_n) - hit) ** 2
    bm = (j.fair_over - hit) ** 2
    print(f"props, fallback-game QB passing-yards lines 2023-25: n {len(j)} ({j.game_id.nunique()} games)  books {bm.mean():.4f}"
          f"  current {bc.mean():.4f}  {pick} {bn.mean():.4f}  -> {'judged' if len(j) >= 30 else 'reported only (under 30 lines)'}")


if __name__ == "__main__":
    main()
