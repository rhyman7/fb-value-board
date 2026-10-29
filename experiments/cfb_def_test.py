"""College board: missing defenders, weighted by production, in the game model (Oct 2026).

    python experiments/cfb_def_test.py

College football has no snap counts, no coverage charting and no public injury history, so
the NFL ratings (def_ratings.py) can't be built. What the ESPN box scores (cfbfastR player
box, 2014 on) do have for every defender: sacks, QB hurries, tackles for loss, passes
defended, interceptions and tackles. Per team and game:

  share      each defender's share of his team's production over its earlier games this
             season (games he appeared in): pass rush = sacks + hurries, coverage = passes
             defended + interceptions, tackling = tackles
  core       a share of at least 10% (pass rush, coverage) or 8% (tackles), two appearances
  miss_*     the share of core defenders who recorded no stat in the team's previous game
             (the box-score signal the board used for offense before Covers.com)
  chg_*      that, less its average over the three games before
  now_*      the share of core defenders who recorded no stat in this game. Known only after
             the game: a ceiling on what perfect injury news could add, never usable

The upcoming-game version would come from Covers.com Out/Doubtful designations, which have no
history, so only the box-score versions can be tested. Same windows as the other college
tests: tuning 2017-2019, holdout 2021 on (2020 is left out); kept only if it helps on both.
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import config as C, data, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

RAW = C.RAW / "def"
OUT = Path(__file__).resolve().parent / "out"
CORE = {"pr": 0.10, "cov": 0.10, "tk": 0.08}
mae = lambda a, b: float(np.mean(np.abs(a - b)))


def box(season: int) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"def_{season}.csv"
    if path.exists():
        return pd.read_csv(path)
    url = f"{C.BASE}/player_box/parquet/player_box_{season}.parquet"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cfb-model"}), timeout=600) as r:
        b = pd.read_parquet(io.BytesIO(r.read()))
    b = b[b["athlete_id"].notna() & b["team_id"].notna()].copy()
    num = lambda c: pd.to_numeric(b[c], errors="coerce").fillna(0.0) if c in b else 0.0
    d = pd.DataFrame({"game_id": pd.to_numeric(b["game_id"], errors="coerce"), "team_id": pd.to_numeric(b["team_id"], errors="coerce"),
                      "player_id": pd.to_numeric(b["athlete_id"], errors="coerce"), "category": b["category"],
                      "pr": np.where(b["category"] == "defensive", num("sacks") + num("hurries"), 0.0),
                      "cov": np.where(b["category"] == "defensive", num("passesDefended"), 0.0)
                      + np.where(b["category"] == "interceptions", num("interceptions"), 0.0),
                      "tk": np.where(b["category"] == "defensive", num("totalTackles"), 0.0)})
    d = d[d["category"].isin(["defensive", "interceptions"])].drop(columns="category")
    d = d.groupby(["game_id", "team_id", "player_id"], as_index=False).sum()
    d.to_csv(path, index=False)
    return d


def team_game(games: pd.DataFrame, pdf: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "season", "order", "home_id", "away_id", "home_team", "away_team", "played"]]
    keep = ["game_id", "season", "order", "team_id", "team", "played"]
    tm = pd.concat([g.rename(columns={"home_id": "team_id", "home_team": "team"})[keep],
                    g.rename(columns={"away_id": "team_id", "away_team": "team"})[keep]])
    tm = tm[tm.team != "FCS"].sort_values(["team", "order"]).reset_index(drop=True)
    p = pdf.merge(tm[["game_id", "team_id", "team"]], on=["game_id", "team_id"], how="inner")
    tot = p.groupby(["game_id", "team_id"])[list(CORE)].transform("sum")
    for k in CORE:
        p[k] = p[k] / tot[k].replace(0, np.nan)
    out = []
    for (team, season), t in tm.groupby(["team", "season"], sort=False):
        t = t.sort_values("order")
        gids = list(t["game_id"])
        done = dict(zip(t["game_id"], t["played"].astype(bool)))
        pt = p[(p["team"] == team) & p["game_id"].isin(gids)]
        appeared = pt.pivot_table(index="player_id", columns="game_id", values="tk", aggfunc="size").notna()
        appeared = appeared.reindex(columns=gids).fillna(False).astype(bool)
        share = {k: pt.pivot_table(index="player_id", columns="game_id", values=k, aggfunc="sum").reindex(index=appeared.index, columns=gids)
                 for k in CORE}
        hist = {k: [] for k in CORE}
        for i, gid in enumerate(gids):
            row = {"game_id": gid, "team": team}
            prev = [x for x in gids[:i] if done[x]]
            for k, thr in CORE.items():
                if not prev:
                    row[f"miss_def{k}"] = row[f"chg_def{k}"] = row[f"now_def{k}"] = 0.0
                    hist[k].append(0.0)
                    continue
                n_app = appeared[prev].sum(axis=1)
                avg = share[k][prev].where(appeared[prev].to_numpy()).mean(axis=1).fillna(0.0)
                core = avg[(avg >= thr) & (n_app >= 2)]
                last = appeared[prev[-1]].reindex(core.index, fill_value=False).astype(bool).to_numpy()
                miss = float(core[~last].sum())
                now = float(core[~appeared[gid].reindex(core.index, fill_value=False).astype(bool).to_numpy()].sum()) if done[gid] else 0.0
                base = float(np.mean(hist[k][-3:])) if hist[k] else 0.0
                row[f"miss_def{k}"], row[f"chg_def{k}"], row[f"now_def{k}"] = miss, miss - base, now
                hist[k].append(miss)
            out.append(row)
    return pd.DataFrame(out)


def main():
    games, tg, qb = data.load_all(verbose=False)
    pdf = pd.concat([box(s) for s in sorted(games.season.unique()) if s >= 2014], ignore_index=True)
    miss = team_game(games, pdf)
    fr = model.build_frame(games, CFBRatingEngine(tg, qb, games=games))
    cols = [c for c in miss.columns if c not in ("game_id", "team")]
    for side in ("home", "away"):
        fr = fr.merge(miss.rename(columns={c: f"{side}_{c}" for c in cols}).rename(columns={"team": f"{side}_team"}), on=["game_id", f"{side}_team"], how="left")
    for c in cols:
        h, a = fr[f"home_{c}"].fillna(0.0), fr[f"away_{c}"].fillna(0.0)
        fr[f"d_{c}"], fr[f"s_{c}"] = h - a, h + a

    def score(MF=None, TF=None):
        out, _ = model.walk_forward(fr, margin_feats=MF, total_feats=TF)
        f = out[out.played & out.spread_line.notna() & out.total_line.notna()]
        T, H = f[f.season.between(2017, 2019)], f[f.season >= 2021]
        return mae(T.model_margin, T.result), mae(H.model_margin, H.result), mae(T.model_total, T.total), mae(H.model_total, H.total)

    B = score()
    print(f"current model: margin {B[0]:.3f} / {B[1]:.3f}, total {B[2]:.3f} / {B[3]:.3f}")
    rows = []
    for k, lab in (("pr", "pass rush"), ("cov", "coverage"), ("tk", "tackling")):
        for m, mlab in (("miss", "missed last game"), ("chg", "change"), ("now", "out this game (ceiling, not usable)")):
            s = score(MF=model.MARGIN_FEATS + [f"d_{m}_def{k}"])
            t = score(TF=model.TOTAL_FEATS + [f"s_{m}_def{k}"])
            r = {"candidate": f"{lab}, {mlab}", "margin_tune": s[0] - B[0], "margin_hold": s[1] - B[1],
                 "total_tune": t[2] - B[2], "total_hold": t[3] - B[3]}
            rows.append(r)
            print(f"{r['candidate']:48s} margin {r['margin_tune']:+.4f} / {r['margin_hold']:+.4f}   total {r['total_tune']:+.4f} / {r['total_hold']:+.4f}", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "cfb_def_test.csv", index=False)


if __name__ == "__main__":
    main()
