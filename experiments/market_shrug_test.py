"""Pre-set test 2 of Nov 26, 2026: injury news the line ignores (the "market shrug"), NFL game model.

    PFF_DATA_DIR=<pff-data clone> python experiments/market_shrug_test.py

Rule written before running (Project doc claude/plan-injury-news-tests-2026-11-26.md). Lead: on the
Nov 26, 2026 board the starters-out input moved the model from Buffalo by 3.2 to 6.2 over Kansas
City during the week while the sharp line stayed at 2.5.

Definitions (spreads and margins are home minus away, positive = home):
  news count    depth-chart starters at QB, RB, WR and TE out because of this week's report (Out or
                Doubtful) whose roster status the week before had them available; home minus away.
                Starters already off the roster the week before are not news: the opener knew them.
  news points   the season's fitted weight on d_depth1_out times the news count
  news game     news points of 1.0 or more either way, with an opening and a closing spread
  line move     closing spread minus nfelo's opening spread, signed toward the side the news favors
  shrug         a news game where the line moved zero or the other way
  followed      a news game where it moved half a point or more toward the news
Part 1 (reported, tuning seasons 2014-18 only, not a rule): the number of news games, the share
  that are shrugs, and in each group how much of the news points showed in the final margin beyond
  the closing line (slope through the origin of result minus closing spread on news points).
Part 2 versions, each changing only news games (the model is fitted as now; only this week's
  prediction changes):
  A  drop on a shrug               in shrug games the news points are removed
  B  count what the line confirms  news points are capped at the line's move toward them (zero on a
                                   shrug)
  C  half on a shrug               in shrug games the news points are halved
Measures (6), on news games only, against the current model:
  1-2  raw model margin miss, tuning 2014-18 and holdout 2019-25
  3-4  board (market-adjusted) margin miss, tuning and holdout
  5-6  Brier of the home cover chance at the closing spread, tuning and holdout
Details fixed here before the first run (they fill gaps in the plan's wording):
  - The board margin is rebuilt for each version with model.py's own blend (weights from that
    version's earlier seasons; the final week goes to the line). The current model rebuilt this way
    must match model.walk_forward exactly, and the script stops if it does not.
  - Cover chance and standard errors as experiments/new_absences_test.py.
Rule: as test 1. A version is chosen on tuning alone (better on all three tuning measures, then the
  lowest tuning raw margin miss); kept only if better on all six and each holdout gain >= 1 paired
  SE; otherwise a trade-off or nothing. If no version passes tuning the holdout is not read.
2026 is reported for the chosen version only, never judged.
Limit written before the run: history has the opener and the close only. The close also holds news
  from after Friday and some openers are posted after an injury is known, so a pass here would
  still be graded forward on the forecast log.

Writes experiments/out/market_shrug_test.json.
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
warnings.filterwarnings("ignore")

from nflmodel import config as C, model  # noqa: E402
import new_absences_test as N  # noqa: E402

OUT = HERE / "out" / "market_shrug_test.json"
TUNE, HOLD = N.TUNE, N.HOLD
NEWS_MIN = 1.0


def raw_model(f: pd.DataFrame, MF: list[str]) -> pd.DataFrame:
    """model.walk_forward's raw margin, plus each season's points per starter out."""
    df = f.copy()
    df["model_margin"], df["per_starter"] = np.nan, np.nan
    j = MF.index("d_depth1_out")
    for s in sorted(x for x in df.season.unique() if x >= model.FIRST_OOS):
        train, test = df[(df.season < s) & df.played], df.season == s
        m = model.Linear(20.0, intercept=False).fit(train[MF].to_numpy(), train["result"].to_numpy())
        X = df.loc[test, MF].to_numpy()
        df.loc[test, "model_margin"] = m.predict(X)
        one = np.zeros((2, len(MF)))
        one[1, j] = 1.0
        p = m.predict(one)
        df.loc[test, "per_starter"] = p[1] - p[0]
    return df[df.season >= model.FIRST_OOS]


def blend(df: pd.DataFrame, col: str) -> pd.Series:
    """model.walk_forward's market-adjusted margin for the raw margin in `col`."""
    proj = df[col].copy()
    fw = model.final_week(df)
    for s in sorted(df.season.unique()):
        past = df[(df.season < s) & (df.season >= max(model.FIRST_OOS, s - C.BLEND_LOOKBACK)) & df.played]
        wm = model._blend_weight(*(past[c].to_numpy() for c in (col, "spread_line", "result")))
        k = (df.season == s) & df.spread_line.notna()
        proj[k] = df.loc[k, "spread_line"] + wm * (df.loc[k, col] - df.loc[k, "spread_line"])
        if C.FINAL_WEEK_MARKET:
            k2 = k & fw
            proj[k2] = df.loc[k2, "spread_line"]
    return proj


def main():
    base, rows, R, avail, op, pff = N.frames()
    t, _ = N.team_counts(rows, R, avail)
    f = N.attach(base, t).merge(op[["game_id", "open_spread"]], on="game_id", how="left")
    MF, TF = list(model.MARGIN_FEATS), list(model.TOTAL_FEATS)
    ref, _ = model.walk_forward(f, margin_feats=MF, total_feats=TF)
    df = raw_model(f, MF)
    chk = ref.set_index("game_id")
    d0 = df.set_index("game_id")
    assert np.allclose(d0["model_margin"], chk["model_margin"], equal_nan=True), "raw margin differs from model.walk_forward"
    assert np.allclose(blend(df, "model_margin").to_numpy(), ref["proj_margin"].to_numpy(), equal_nan=True), "blend differs from model.walk_forward"

    df["news"] = df["per_starter"] * df["d_n_news"]
    df["move"] = (df["spread_line"] - df["open_spread"]) * np.sign(df["news"])
    df["news_game"] = (df["news"].abs() >= NEWS_MIN) & df["open_spread"].notna() & df["spread_line"].notna() & df["played"]
    df["shrug"] = df["news_game"] & (df["move"] <= 0)
    df["followed"] = df["news_game"] & (df["move"] >= 0.5)
    conf = np.sign(df["news"]) * np.minimum(df["news"].abs(), df["move"].clip(lower=0).fillna(0))
    df["m_A"] = df["model_margin"] - np.where(df["shrug"], df["news"], 0.0)
    df["m_B"] = df["model_margin"] - np.where(df["news_game"], df["news"] - conf, 0.0)
    df["m_C"] = df["model_margin"] - np.where(df["shrug"], df["news"] / 2, 0.0)

    def scored(col):
        o = df.copy()
        o["proj_margin"] = blend(df, col)
        o["model_margin"] = df[col]
        return N.add_scores(o)

    b = scored("model_margin")
    news_ids = set(df.loc[df["news_game"], "game_id"])
    tn = df[df.season.between(*TUNE)]

    def group(g):
        if len(g) == 0:
            return {"n": 0}
        x, y = g["news"].to_numpy(float), (g["result"] - g["spread_line"]).to_numpy(float)
        slope = float((x * y).sum() / (x * x).sum())
        se = float(np.sqrt(((y - slope * x) ** 2 * x * x).sum()) / (x * x).sum())
        toward = np.sign(x) * y
        return {"n": int(len(g)), "avg_news_points": round(float(np.abs(x).mean()), 2), "avg_line_move_toward_news": round(float(g["move"].mean()), 2),
                "share_of_news_points_seen_beyond_the_close": round(slope, 3), "se": round(se, 3),
                "avg_margin_beyond_close_toward_news": round(float(toward.mean()), 2),
                "se_avg": round(float(toward.std(ddof=1) / np.sqrt(len(toward))), 2) if len(toward) > 1 else None}

    res = {"quarterback_prior_pff": pff,
           "points_per_starter_out_by_season": {int(s): round(float(v), 3) for s, v in df.groupby("season")["per_starter"].first().items() if s >= TUNE[0]},
           "news_games": {"tuning_2014_18": int(tn["news_game"].sum()), "holdout_2019_25": int(df[df.season.between(*HOLD)]["news_game"].sum()),
                          "tuning_games_with_an_opener": int((tn["open_spread"].notna() & tn["played"]).sum())},
           "part1_tuning_only": {"all news games": group(tn[tn["news_game"]]), "shrug": group(tn[tn["shrug"]]),
                                 "followed": group(tn[tn["followed"]]),
                                 "shrug_share": round(float(tn["shrug"].sum() / max(tn["news_game"].sum(), 1)), 3)}}
    print(json.dumps(res), flush=True)
    outs = {"A (drop on a shrug)": scored("m_A"), "B (count what the line confirms)": scored("m_B"), "C (half on a shrug)": scored("m_C")}
    res = N.judge(res, b, outs, only=news_ids)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
