"""Pass rate over expected (PROE): does a team's play-calling tendency help the NFL models?

    python experiments/proe_test.py [pbp_dir]

PROE is how often a team passes compared with what the down, distance, field position, score and
clock say an average team would (nflfastR's xpass). nfelo shows it on its team pages; the board
does not use it. The pace test (seconds per play) was rejected, but PROE measures play-calling,
not speed: a pass-first team throws more and stops the clock more.

Per team-game, from nflverse play-by-play (2012 on), neutral situations only (downs 1-3, win
probability 0.20-0.80, more than two minutes left in the half): offense PROE = mean(pass - xpass)
over its plays, and the same for the opponents of each defense ("PROE allowed"). As of each game:
the weighted average of the team's earlier games (TEAM_HL-game half-life, last season at half
weight, shrunk toward 0 by 4 games), as experiments/volume_allowed_test.py builds its ratings.

1. Team pass attempts and carries (players.VOL_FEATS regression, walk-forward on the 8 seasons
   before each, as volume_allowed_test.py): add the offense's PROE and the opponent's PROE allowed.
2. Game totals (the board's walk-forward total model): add s_proe = both offenses' PROE plus both
   defenses' PROE allowed.

Rule, set before running (Nov 2, 2026): kept only if the average miss drops on the tuning seasons
(2014-2018) and on the holdout (2019 on), the holdout by more than two standard errors clustered by
game. Writes experiments/out/proe_test.json.
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
import volume_allowed_test as V  # noqa: E402
from nflmodel import config as C, players as PL  # noqa: E402
from props_calibration import load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "proe_test.json"
COLS = ["game_id", "season", "posteam", "defteam", "pass", "rush", "down", "wp", "half_seconds_remaining", "xpass"]


def team_proe(pbp_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(pbp_dir.glob("p20*.parquet")):
        p = pd.read_parquet(f, columns=COLS)
        p = p[((p["pass"] == 1) | (p["rush"] == 1)) & p.xpass.notna() & p.down.isin([1, 2, 3])
              & p.wp.between(0.2, 0.8) & (p.half_seconds_remaining > 120) & p.posteam.notna()]
        p = p.assign(oe=p["pass"] - p.xpass,
                     team=p.posteam.replace(C.TEAM_ALIASES), opp=p.defteam.replace(C.TEAM_ALIASES))
        rows.append(p.groupby(["game_id", "team", "opp"]).agg(proe=("oe", "mean"), n=("oe", "size")).reset_index())
    return pd.concat(rows, ignore_index=True)


def asof(t: pd.DataFrame, key: str, val: str, out: str) -> pd.DataFrame:
    """Weighted average of `val` over each `key` team's earlier games (V.defense_rating's method)."""
    tmp = t.rename(columns={"opp": "_opp"}).assign(opp=t[key])
    tmp = V.defense_rating(tmp, val, out)
    return t.assign(**{out: tmp[out].reindex(t.index).to_numpy()})


def volume(t: pd.DataFrame) -> dict:
    base = list(PL.VOL_FEATS)
    res = {}
    for target in ("team_att", "team_car"):
        variants = {"current": base, "+ PROE (offense and opponent allowed)": base + ["proe_off", "opp_proe_def"]}
        preds = {k: V.walk(t, target, v) for k, v in variants.items()}
        ok = t.played.astype(bool) & t[target].notna() & (t.season >= 2014)
        d = t[ok].assign(p0=preds["current"][ok], p1=preds["+ PROE (offense and opponent allowed)"][ok]).dropna(subset=["p0", "p1"])
        r = {}
        for per, m in (("tune", d.season < C.HOLDOUT_FROM), ("hold", d.season >= C.HOLDOUT_FROM)):
            e0, e1 = (d.loc[m, "p0"] - d.loc[m, target]).abs(), (d.loc[m, "p1"] - d.loc[m, target]).abs()
            diff = e1 - e0
            g = diff.groupby(d.loc[m, "game_id"]).sum()
            r[per] = {"n": int(m.sum()), "mae": round(float(e0.mean()), 4), "change": round(float(diff.mean()), 4),
                      "se": round(float(g.std(ddof=1) * np.sqrt(len(g)) / len(diff)), 4)}
        r["keep"] = bool(r["tune"]["change"] < 0 and r["hold"]["change"] < -2 * r["hold"]["se"])
        res[target] = r
        print(target, r, flush=True)
    return res


def totals(t: pd.DataFrame) -> dict:
    from nflmodel import data, extra, model
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    te, qe = extra.load_extra(verbose=False)
    frame = model.build_frame(games, ExtRatingEngine(tg, qb, te, qe, extra_metrics=PRODUCTION_METRICS, games=games))
    side = t[["game_id", "team", "proe_off", "proe_def"]]
    fr = frame.merge(side.rename(columns={"team": "home_team", "proe_off": "h_off", "proe_def": "h_def"}), on=["game_id", "home_team"], how="left")
    fr = fr.merge(side.rename(columns={"team": "away_team", "proe_off": "a_off", "proe_def": "a_def"}), on=["game_id", "away_team"], how="left")
    fr["s_proe"] = fr[["h_off", "a_off", "h_def", "a_def"]].fillna(0).sum(axis=1)
    fr["act_total"] = fr.home_score + fr.away_score
    TF = list(model.TOTAL_FEATS)
    o0, _ = model.walk_forward(fr)
    o1, _ = model.walk_forward(fr, total_feats=TF + ["s_proe"])
    j = o0[o0.played & o0.model_total.notna()][["game_id", "season", "model_total", "act_total"]].merge(
        o1[["game_id", "model_total"]], on="game_id", suffixes=("", "_t"))
    j["d"] = (j.model_total_t - j.act_total).abs() - (j.model_total - j.act_total).abs()
    r = {}
    for per, m in (("tune", j.season < C.HOLDOUT_FROM), ("hold", j.season >= C.HOLDOUT_FROM)):
        x = j[m]
        r[per] = {"n": int(len(x)), "mae": round(float((x.model_total - x.act_total).abs().mean()), 4),
                  "change": round(float(x.d.mean()), 4), "se": round(float(x.d.std(ddof=1) / np.sqrt(len(x))), 4)}
    r["keep"] = bool(r["tune"]["change"] < 0 and r["hold"]["change"] < -2 * r["hold"]["se"])
    print("total", r, flush=True)
    return r


def main():
    pbp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else C.RAW / "pbp_proe"
    g = team_proe(pbp_dir)
    proj = load_proj(None)
    t = proj.drop_duplicates(["game_id", "team"]).copy().reset_index(drop=True)
    t = t.merge(g[["game_id", "team", "proe"]], on=["game_id", "team"], how="left")
    t["proe_v"] = np.where(t.played.astype(bool), t.proe, np.nan)
    t = asof(t, "team", "proe_v", "proe_off")          # the offense's own tendency
    t = V.defense_rating(t, "proe_v", "proe_def")      # what this team's opponents did against it... per row's opp
    # proe_def above is the rating of the row's opponent (V.defense_rating keys on `opp`): rename
    t = t.rename(columns={"proe_def": "opp_proe_def"})
    own_def = t[["game_id", "opp", "opp_proe_def"]].rename(columns={"opp": "team", "opp_proe_def": "proe_def"})
    t = t.merge(own_def, on=["game_id", "team"], how="left")
    sd = {"proe_off_sd": round(float(t.proe_off.std()), 4), "opp_proe_def_sd": round(float(t.opp_proe_def.std()), 4),
          "team_games_with_proe": int(t.proe.notna().sum())}
    print(sd)
    res = {"ratings": sd, "volume": volume(t), "totals": totals(t)}
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
