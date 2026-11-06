"""College returning production from PFF that counts incoming transfers (margin model only).

Applied Nov 6, 2026 (Ryan's choice, option 1 of the trade-off; CFB_FEATURES.md "Returning
production from PFF that counts incoming transfers", experiments/cfb_pff_returning_test.py):
the margin half of version A, holdout margin miss -0.022 (SE 0.009), tuning -0.005, totals
unchanged (the total half made the holdout total worse and is not used).

For each team-season, the players on this season's CFBD roster (data/cfb/cfbd/roster_{S}.csv,
pulled by the CFBD player data Action) are matched by name to last season's PFF college players
on any team, and valued at last season's PFF snaps x (snap-weighted grade - 60), clipped at 0,
divided by 1,000 (offense_summary grades_offense, defense_summary grades_defense). A name shared
by several PFF players is resolved by the same team, then by the transfer portal's origin, else
left out. The feature d_pffret is home minus away; a team without a value takes the median.
Without the PFF data (no .pff-data checkout) every game gets 0, which the model fits as no
information: the board then runs as before.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C, pff_data

REP = 60.0
CFBD = C.DATA / "cfbd"
FIRST_PFF = 2017


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
    pf = pff_seasons(range(FIRST_PFF, max(seasons) + 1))
    ros = rosters(range(FIRST_PFF, max(seasons) + 1))
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


def add(frame: pd.DataFrame, seasons=None, verbose: bool = False) -> pd.DataFrame:
    """frame with d_pffret (home minus away transfer-aware returning value); 0 without PFF data."""
    f = frame.copy()
    f["d_pffret"] = 0.0
    if not pff_data.available():
        if verbose:
            print("  PFF returning production: no PFF data, d_pffret = 0")
        return f
    seasons = [int(s) for s in (seasons if seasons is not None else sorted(f.season.unique())) if s > FIRST_PFF]
    try:
        tv, st = team_values(seasons)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  PFF returning production skipped ({exc})")
        return f
    x = tv[["season", "team", "ret"]]
    h = f[["season", "home_team"]].merge(x.rename(columns={"team": "home_team", "ret": "h"}), on=["season", "home_team"], how="left")["h"].to_numpy()
    a = f[["season", "away_team"]].merge(x.rename(columns={"team": "away_team", "ret": "a"}), on=["season", "away_team"], how="left")["a"].to_numpy()
    fill = float(np.nanmedian(np.concatenate([h, a]))) if np.isfinite(np.concatenate([h, a])).any() else 0.0
    f["d_pffret"] = np.nan_to_num(h, nan=fill) - np.nan_to_num(a, nan=fill)
    if verbose:
        print(f"  PFF returning production: {len(tv)} team-seasons, {st['matched']} roster matches ({st['transfers_in']} transfers in)")
    return f
