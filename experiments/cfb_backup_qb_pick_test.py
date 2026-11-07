"""Which quarterback the college player projections start when last game's starter is out (Nov 6, 2026).

    python experiments/cfb_backup_qb_pick_test.py

The college version of experiments/backup_qb_pick_test.py (the NFL test, kept and applied Nov 6, 2026).

THE RULE (written before the first run, Nov 6, 2026)
-----------------------------------------------------
cfbmodel.players._pick_starter: this week's named starter (the daily starting-QB list, starters.py)
if there is one; else last game's starter if he is active (unless another active QB is clearly the
regular, 80%+ share); otherwise the active QB with the largest att_share_raw. As in the NFL, a
quarterback with no games gets a default share of half the position average, so he can outrank a
backup who has thrown a handful of passes. The candidates change only the "otherwise" branch, the
fallback games (the same four versions as the NFL test):
  current   largest att_share_raw (the board today)
  volume    largest decayed pass attempts (pass_att_sum), then att_share_raw
  starts    largest decayed games started (start_sum), then pass_att_sum, then att_share_raw
  known     largest att_share_raw, quarterbacks with no games (one_sum 0) ranked last

College has no game-day inactive list, and in the backtest table "active" for a past game includes
anyone who appeared, so the actual starter is always in the pool and last game's starter is always
"expected". The fallback is simulated as it happens live: a team the starting-QB list misses whose
last starter is listed Out (Covers). Fallback games: played team-games, not the team's first game of
the season, where last game's starter recorded no stat. In those team-games the quarterbacks' pool
is set to what is known before kickoff: the quarterbacks expected to play (a stat in the team's
previous game), last game's starter taken out. The actual starter can be missing from that pool
(no stat last game); such games count as wrong for every version. All versions run on the same
simulated table, and the backtest has no named starters (starters=None), as in the other tests.

Seasons as the college walk-forward: tuning 2017-2019, holdout 2021 on (config.HOLDOUT_FROM; 2020
left out, as in the other college tests). Measures (four):
  1-2  pick accuracy in fallback games (the projected starter is the QB who started), tuning, holdout
  3-4  RMSE of the actual starter's passing-yards projection in fallback games, tuning, holdout
The version is chosen on the tuning seasons (best tuning pick accuracy); KEEP RULE (CONVENTIONS.md):
kept only if better on all four and worse on none; a mixed result goes to Ryan.
Props check: QB passing-yards main lines at 2025 DK/FD in fallback games (college prop history is
2025 only), Brier current vs chosen; judged only with 30+ lines, otherwise reported.
Also reported: the number of first-game-of-season team-games where last season's final starter had
no stat (not simulated: live, he stays "expected" unless the list or Covers says otherwise).
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
from cfbmodel import config as C, players as PL  # noqa: E402

TUNE, HOLD = (2017, 2019), C.HOLDOUT_FROM
ORIG = PL._pick_starter


def make_pick(key):
    def pick(d: pd.DataFrame):
        top = d.iloc[0]
        forced = top.get("forced_qb") if "forced_qb" in d else None
        if forced is not None and pd.notna(forced):
            hit = d[d["player_id"] == forced]
            if len(hit):
                return hit.index[0]
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


def base_table() -> pd.DataFrame:
    import cfb_rush_group_calib_test as X          # the cached college player table
    return X.table()


def fallback(t: pd.DataFrame) -> tuple[pd.DataFrame, set, int]:
    """The simulated table, the fallback (game_id, team) keys, and the first-game count."""
    tg = t.drop_duplicates(["game_id", "team"])[["game_id", "team", "season", "t", "played", "prev_qb"]].sort_values(["team", "t"])
    tg["first"] = tg.groupby(["team", "season"]).cumcount() == 0
    app = set(zip(t.loc[t["appeared"] == 1, "game_id"], t.loc[t["appeared"] == 1, "team"], t.loc[t["appeared"] == 1, "player_id"]))
    tg["prev_in"] = [(g, tm, q) in app for g, tm, q in zip(tg["game_id"], tg["team"], tg["prev_qb"])]
    out = tg["played"].astype(bool) & tg["prev_qb"].notna() & ~tg["prev_in"]
    fb = set(zip(tg.loc[out & ~tg["first"], "game_id"], tg.loc[out & ~tg["first"], "team"]))
    n_first = int((out & tg["first"]).sum())
    s = t.copy()
    key = pd.Series(list(zip(s["game_id"], s["team"])), index=s.index)
    m = key.isin(fb) & (s["group"] == "QB")
    s.loc[m, "active"] = ((s.loc[m, "expected"] == 1) & (s.loc[m, "player_id"] != s.loc[m, "prev_qb"])).astype(int)
    return s, fb, n_first


def score(p: pd.DataFrame, fb: set) -> tuple[dict, pd.DataFrame]:
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
    t, pinfo, games = base_table()
    s, fb, n_first = fallback(t)
    in_pool = s[(s["group"] == "QB") & (s["active"] == 1)]
    print(f"fallback team-games: {len(fb)}; first games of a season with last season's starter gone: {n_first} (not simulated)", flush=True)
    res, rows, proj = {}, {}, {}
    for name, fn in VERSIONS.items():
        PL._pick_starter = fn or ORIG
        p = PL.walk_forward(s)
        proj[name] = p
        res[name], rows[name] = score(p, fb)
        r = res[name]
        print(f"{name:8s} tune: fallback right {r['tune']['acc_fb']:.1%} (n {r['tune']['n_fb']}), starter pass-yds RMSE {r['tune']['rmse_fb']:.1f}, all games {r['tune']['acc_all']:.2%}"
              f" | hold: {r['hold']['acc_fb']:.1%} (n {r['hold']['n_fb']}), RMSE {r['hold']['rmse_fb']:.1f}, all {r['hold']['acc_all']:.2%}", flush=True)
    PL._pick_starter = ORIG
    # how often the actual starter was in the pregame pool at all (the ceiling for any version)
    st = rows["current"]
    pool = set(zip(in_pool["game_id"], in_pool["team"], in_pool["player_id"]))
    f = st[st.fb]
    f = f.assign(reach=[(g, tm, q) in pool for g, tm, q in zip(f.game_id, f.team, f.player_id)])
    print(f"actual starter in the pregame pool (ceiling): tuning {f[f.season.between(*TUNE)].reach.mean():.1%}, holdout {f[f.season >= HOLD].reach.mean():.1%}")
    cand = {k: v for k, v in res.items() if k != "current"}
    pick = max(cand, key=lambda k: cand[k]["tune"]["acc_fb"])
    c, n = res["current"], res[pick]
    ms = [n["tune"]["acc_fb"] > c["tune"]["acc_fb"], n["hold"]["acc_fb"] > c["hold"]["acc_fb"],
          n["tune"]["rmse_fb"] < c["tune"]["rmse_fb"], n["hold"]["rmse_fb"] < c["hold"]["rmse_fb"]]
    worse = [n["tune"]["acc_fb"] < c["tune"]["acc_fb"], n["hold"]["acc_fb"] < c["hold"]["acc_fb"],
             n["tune"]["rmse_fb"] > c["tune"]["rmse_fb"], n["hold"]["rmse_fb"] > c["hold"]["rmse_fb"]]
    verdict = "KEEP" if all(ms) else ("not kept" if not any(ms) else "mixed")
    print(f"\nchosen on tuning: {pick}; better on {sum(ms)} of 4, worse on {sum(worse)} -> {verdict}")
    a, b = rows["current"], rows[pick]
    j = a[["game_id", "team", "season", "player_id", "picked", "right", "fb"]].merge(b[["game_id", "team", "picked", "right"]], on=["game_id", "team"], suffixes=("_c", "_n"))
    ch = j[j.picked_c != j.picked_n]
    print(f"picks changed: {len(ch)} (in fallback games {int(ch.fb.sum())}); right before {int(ch.right_c.sum())}, after {int(ch.right_n.sum())}")
    props(proj["current"], proj[pick], pinfo, games, fb, pick)


def props(pc: pd.DataFrame, pn: pd.DataFrame, pinfo, games, fb: set, pick: str) -> None:
    """Passing-yards main lines in fallback games, 2025 DK/FD: Brier, current vs chosen."""
    import cfb_props_real_lines as R
    from nflmodel import odds as O, parlays as PA, propdist as D, propmarket as PM, props as P
    s = R.snapshots()
    s = s.merge(R.match_events(s, games), left_on="event_id", right_index=True, how="inner")
    s["nname"] = s["player"].map(PM.norm_name)
    s["stat"] = s["market"].map(PM.MARKET_STAT)
    s = s[~s["market"].str.endswith("_alternate") & (s["stat"] == "pass_yds")]
    if "book" in s:
        s = s[s["book"].isin(["draftkings", "fanduel"])]
    w = s.pivot_table(index=["game_id", "nname", "stat", "line", "book"], columns="side", values="price", aggfunc="last").reset_index().dropna(subset=["over", "under"])
    w["fair"] = O.no_vig_power(w["over"], w["under"])[0]
    w = w.groupby(["game_id", "nname", "stat", "line"], as_index=False)["fair"].mean()
    model = P.PropModel.load(PA.prop_params_path("cfb"))
    got = {}
    for name, p in (("current", pc), (pick, pn)):
        rows = R.player_frame(p, pinfo)
        rows = rows[np.array([k in fb for k in zip(rows.game_id, rows.team)]) & (rows.group == "QB").to_numpy()]
        x = w.merge(rows, on=["game_id", "nname"], how="inner")
        x = R.model_chances(x, model) if len(x) else pd.DataFrame()
        got[name] = x[x["y"] != x["line"]] if len(x) else x
    a, b = got["current"], got[pick]
    if a.empty or b.empty:
        print("props: no fallback-game passing-yards lines in 2025")
        return
    key = ["game_id", "nname", "line"]
    j = a[key + ["y", "fair", "our_over", "our_under"]].merge(b[key + ["our_over", "our_under"]], on=key, suffixes=("_c", "_n"))
    hit = (j.y > j.line).astype(float)
    bc = (j.our_over_c / (j.our_over_c + j.our_under_c) - hit) ** 2
    bn = (j.our_over_n / (j.our_over_n + j.our_under_n) - hit) ** 2
    bm = (j.fair - hit) ** 2
    print(f"props, fallback-game QB passing-yards lines 2025: n {len(j)} ({j.game_id.nunique()} games)  books {bm.mean():.4f}"
          f"  current {bc.mean():.4f}  {pick} {bn.mean():.4f}  -> {'judged' if len(j) >= 30 else 'reported only (under 30 lines)'}")


if __name__ == "__main__":
    main()
