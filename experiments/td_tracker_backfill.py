"""Rebuild the Likeliest TD Scorers list for past 2026 NFL weeks, ranked by the board's chance.

    python experiments/td_tracker_backfill.py <2026 pbp parquet>

The report (rhyman7/nfl scripts/build_report.py, td_rows) lists the 20 players most likely to
score a rushing or receiving touchdown. Since the November feed it ranks by the value board's
anytime-TD chance; before that it ranked by its own rougher estimate. This rebuilds every past
week as if the board's chance had been used, for a tracker.

Who is eligible (the report's own rules, from season-to-date stats before the week): played 2+
games; not ruled out before kickoff; at least one rushing or receiving TD so far; RB/FB 10+
carries plus catches a game, WR 3.5+ catches a game, TE 3+; a QB only if he is the team's
projected starter (his rushing TDs only).

The chance: the board's td_any as shown before kickoff where it was saved (docs/archive, weeks 7
on); earlier weeks rebuilt the board's way, 1 - exp(-(projected rushing + receiving TDs)), from
the walk-forward projections (players_proj.pkl), which are what the board would have shown.
Result: did he score a rushing or receiving TD (nflverse play-by-play).

Writes experiments/out/td_tracker_2026.csv (every week's top 20) and td_tracker_2026.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import config as C  # noqa: E402
from props_calibration import load_proj  # noqa: E402

SEASON, TOP = 2026, 20
OUT = HERE / "out"


def scorers(pbp: Path) -> set:
    p = pd.read_parquet(pbp, columns=["game_id", "td_player_id", "rush_touchdown", "pass_touchdown"])
    t = p[((p.rush_touchdown == 1) | (p.pass_touchdown == 1)) & p.td_player_id.notna()]
    return set(zip(t.game_id, t.td_player_id))


def shown() -> dict:
    """(game_id, player_id) -> td_any as shown on the board before kickoff, from the archive."""
    out = {}
    for f in sorted((C.DOCS / "archive").glob(f"{SEASON}_w*.json")):
        d = json.loads(f.read_text())
        for g, ps in (d.get("players") or {}).items():
            for x in ps:
                if x.get("td_any") is not None:
                    out[(g, x["id"])] = float(x["td_any"])
    return out


def main():
    pbp = Path(sys.argv[1])
    proj = load_proj(None)
    p = proj[proj.season == SEASON].copy()
    played_games = set(pd.read_parquet(pbp, columns=["game_id"]).game_id.unique())
    td = scorers(pbp)
    sh = shown()
    p["lam"] = p.rec_td_proj.fillna(0) + p.rush_td_proj.fillna(0)
    p["rebuilt"] = 1 - np.exp(-p.lam.clip(lower=0))
    # season-to-date actuals from the games already played (pbp-confirmed)
    hist = p[p.played.astype(bool) & p.game_id.isin(played_games)]
    rows = []
    for w in sorted(p.week.unique()):
        wk = p[(p.week == w) & p.game_id.isin(played_games)]
        if wk.empty:
            continue
        h = hist[hist.week < w].groupby("player_id").agg(gp=("one", "size"), car=("carries", "mean"), rec=("receptions", "mean"),
                                                          rtd=("rush_td", "sum"), ctd=("rec_td", "sum"))
        x = wk.merge(h, left_on="player_id", right_index=True, how="inner")
        x = x[(x.active_pre == 1) & (x.gp >= 2)]
        starter = x[x.position == "QB"].sort_values("pass_att_proj", ascending=False).drop_duplicates(["game_id", "team"])
        is_qb = x.index.isin(starter.index)
        rb = x.position.isin(["RB", "FB"]) & (x.car + x.rec >= 10) & (x.rtd + x.ctd > 0)
        wr = (x.position == "WR") & (x.rec >= 3.5) & (x.rtd + x.ctd > 0)
        te = (x.position == "TE") & (x.rec >= 3) & (x.rtd + x.ctd > 0)
        qb = is_qb & (x.rtd > 0)
        x = x[rb | wr | te | qb].copy()
        x["chance_shown"] = [sh.get((g, pid)) for g, pid in zip(x.game_id, x.player_id)]
        x["chance"] = x.chance_shown.fillna(x.rebuilt)
        x["source"] = np.where(x.chance_shown.notna(), "shown", "rebuilt")
        x = x.sort_values("chance", ascending=False).head(TOP)
        x["rank"] = range(1, len(x) + 1)
        x["scored"] = [int((g, pid) in td) for g, pid in zip(x.game_id, x.player_id)]
        rows.append(x)
    t = pd.concat(rows, ignore_index=True)
    info = pd.read_csv(C.CACHE / f"player_info_{SEASON}.csv", dtype={"player_id": str})[["player_id", "name"]]
    t = t.merge(info, on="player_id", how="left")
    cols = ["week", "rank", "name", "position", "team", "opp", "game_id", "chance", "source", "rebuilt", "scored"]
    t = t[cols].sort_values(["week", "rank"])
    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / "td_tracker_2026.csv", index=False)
    weeks = t.groupby("week").agg(players=("scored", "size"), scored=("scored", "sum"), expected=("chance", "sum"),
                                  source=("source", lambda s: ",".join(sorted(set(s))))).round(2)
    band = t.groupby(pd.cut(t.chance, [0, 0.3, 0.4, 0.5, 0.6, 1])).agg(n=("scored", "size"), chance=("chance", "mean"), hit=("scored", "mean")).round(3)
    both = t[t.source == "shown"]
    res = {"weeks": {int(k): v for k, v in weeks.T.to_dict().items()},
           "total": {"players": int(len(t)), "scored": int(t.scored.sum()), "expected": round(float(t.chance.sum()), 1)},
           "top5": {"players": int((t["rank"] <= 5).sum()), "scored": int(t[t["rank"] <= 5].scored.sum()),
                    "expected": round(float(t[t["rank"] <= 5].chance.sum()), 1)},
           "by_chance": {str(k): v for k, v in band.T.to_dict().items()},
           "shown_vs_rebuilt_corr": round(float(both.chance.corr(both.rebuilt)), 3) if len(both) > 5 else None}
    (OUT / "td_tracker_2026.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
