"""What the board knows before kickoff, and the weight the model earns with it (Nov 19, 2026).

The backtest knows each game's actual starting quarterbacks and game-day inactives; the board, pricing
a game at the current line a few days out, knows only what was public by about Friday. On 2019-25 the
model earned a weight of 0.22 against the closing line with game-day information and 0.14 with Friday
information (Project docs claude/bettime-information-results-2026-11-19.md, test W in
claude/nfl-friday-weight-test-results-2026-11-19.md). From Nov 19, 2026 (Ryan's call, provisional,
January recheck; config.FRIDAY_WEIGHT) the board prices current-line games (not early lines, not the
final week) at the Friday weight.

The Friday reconstruction of a past game:
  quarterbacks  last week's starter, unless the Friday report has him Out or Doubtful or the previous
                week's roster has him on a reserve list or off it, or the actual starter already started
                for the team earlier in the season (a starter coming back, announced in the week): then
                the actual starter. Unknown: first-time starters not replacing a listed injury.
  injuries      injuries._season(bettime=True): the final report's Out/Doubtful and the previous
                week's longer-term lists, no game-day inactives
  roles         as the board's (game_roles.friday_active is already Friday information)

friday_weight(): the board's model (fitted on game-day inputs, as now) predicts each of the previous
BLEND_LOOKBACK seasons from the Friday inputs; the weight is least squares of (result - line) on
(prediction - line), clipped to [0, 1], exactly as model._blend_weight. Past seasons only, so the
reconstruction is cached per season in data/cache/ and the current season is never needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C, injuries, model
from .data import _download

URL = "https://github.com/nflverse/nflverse-data/releases/download/{kind}/{name}_{season}.parquet"
FIRST_QB = 2009


def _parquet(kind: str, name: str, season: int, cols: list[str]) -> pd.DataFrame:
    path = C.RAW / f"{name}_{season}.parquet"
    if not path.exists():
        _download(URL.format(kind=kind, name=name, season=season), path)
    try:
        d = pd.read_parquet(path)
        return d[[c for c in cols if c in d]]
    finally:
        path.unlink(missing_ok=True)


def _qb_season(games: pd.DataFrame, season: int) -> pd.DataFrame:
    rep = _parquet("injuries", "injuries", season, ["season", "week", "team", "gsis_id", "report_status"])
    ros = _parquet("weekly_rosters", "roster_weekly", season, ["season", "week", "team", "gsis_id", "status"])
    out_set = set(map(tuple, rep.loc[rep.report_status.isin(["Out", "Doubtful"]), ["week", "gsis_id"]].to_numpy()))
    ros_st = ros.drop_duplicates(["week", "gsis_id"]).set_index(["week", "gsis_id"])["status"].to_dict()
    g = games[games.season == season]
    long = pd.concat([g[["game_id", "week", "home_team", "home_qb_id"]].set_axis(["game_id", "week", "team", "qb"], axis=1).assign(side="home"),
                      g[["game_id", "week", "away_team", "away_qb_id"]].set_axis(["game_id", "week", "team", "qb"], axis=1).assign(side="away")])
    rows = []
    for team, x in long.sort_values("week").groupby("team"):
        seen, prev_qb, prev_wk = set(), None, None
        for r in x.itertuples(index=False):
            q = r.qb if isinstance(r.qb, str) else None
            if prev_qb is None or q == prev_qb:
                pick = q                                       # season opener, or the same starter
            elif (r.week, prev_qb) in out_set or ros_st.get((prev_wk, prev_qb)) not in (None, "ACT", "INA"):
                pick = q                                       # last week's starter listed Out/Doubtful or on a reserve list
            elif q in seen and (r.week, q) not in out_set:
                pick = q                                       # an earlier starter coming back
            else:
                pick = None                                    # the board would assume last week's starter
            rows.append({"game_id": r.game_id, "side": r.side, "qb": pick})
            if q:
                seen.add(q)
            prev_qb, prev_wk = (q or prev_qb), r.week
    d = pd.DataFrame(rows).pivot(index="game_id", columns="side", values="qb")
    return d.rename(columns={"home": "home_qb_id", "away": "away_qb_id"}).reset_index()


def _cached(name: str, season: int, build) -> pd.DataFrame:
    path = C.CACHE / f"{name}_{season}.csv"
    if not path.exists():
        build().to_csv(path, index=False)
    return pd.read_csv(path)


def friday_games(games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """games with each listed season's quarterbacks replaced by the Friday rule (blank = last week's)."""
    g = games.copy()
    for s in seasons:
        q = _cached("bettime_qb", s, lambda: _qb_season(games, s)).set_index("game_id")
        m = g.season == s
        g.loc[m, "home_qb_id"] = g.loc[m, "game_id"].map(q["home_qb_id"])
        g.loc[m, "away_qb_id"] = g.loc[m, "game_id"].map(q["away_qb_id"])
    return g


def friday_injuries(games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    parts = [_cached("injury_groups_bt", s, lambda: injuries._season(s, games, bettime=True))
             for s in seasons if s >= injuries.FIRST]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def friday_weight(games: pd.DataFrame, engine, wf_input: pd.DataFrame, season: int, roles: pd.DataFrame | None = None) -> dict:
    """The current-line weight for `season` from the Friday reconstruction of the previous
    BLEND_LOOKBACK seasons. wf_input: the frame the board's model is fitted on (game-day inputs);
    roles: game_roles.team_features() output, attached to the Friday frame as to the board's."""
    seasons = [s for s in range(max(model.FIRST_OOS, season - C.BLEND_LOOKBACK), season)]
    gf = friday_games(games, [s for s in seasons if s >= FIRST_QB])
    fr = model.build_frame(gf, engine, injuries=friday_injuries(games, seasons))
    if roles is not None:
        from . import game_roles
        fr = game_roles.attach(fr, roles)
    MF = model.MARGIN_FEATS
    for d in (wf_input, fr):
        for c in MF:
            if c not in d:
                d[c] = 0.0
    fr = fr.set_index("game_id")
    pred, line, res, by = [], [], [], {}
    for s in seasons:
        tr = wf_input[(wf_input.season < s) & wf_input.played]
        m = model.Linear(20.0, intercept=False).fit(tr[MF].to_numpy(float), tr["result"].to_numpy(float))
        x = fr[(fr.season == s) & fr.played & fr.spread_line.notna()]   # every game, as walk_forward measures w_margin
        p = m.predict(x[MF].to_numpy(float))
        by[int(s)] = round(model._blend_weight(p, x.spread_line.to_numpy(float), x.result.to_numpy(float)), 3)
        pred.append(p); line.append(x.spread_line.to_numpy(float)); res.append(x.result.to_numpy(float))
    w = model._blend_weight(np.concatenate(pred), np.concatenate(line), np.concatenate(res))
    return {"weight": round(float(w), 3), "seasons": [int(seasons[0]), int(seasons[-1])], "by_season": by}
