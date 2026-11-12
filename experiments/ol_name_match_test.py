"""Offensive-line input: linemen the name match loses (Nov 11, 2026).

    python experiments/ol_name_match_test.py          (PFF_DATA_DIR=../pff-data for the production model)

Found while wiring late injury news into the game model. `injuries._season` matches a team's core
players (named as the snap counts spell them) to the injury report and the weekly roster by
normalised full name. Where the two files spell a player differently ("Paris Johnson" against
"Paris Johnson Jr.", "Delmar Glaze" against "DJ Glaze", "Samuel Cosmi" against "Sam Cosmi") the
roster lookup finds nothing and the player is counted as off the roster, so unavailable, in every
game, played or upcoming. On the Nov 11 board that is 9 of the 14 core linemen flagged as out.

Candidate (C): the same input with a looser match, tried only when the exact one fails, within the
team: (1) the names agree once a Jr./Sr./II/III/IV suffix is dropped; (2) same last name and first
initial, and only one such player on the team's roster (or report) that season. Nothing else in
the input changes, and no other group is used by the game model.

Rule, written before the first run (the Nov 11 keep rule, CONVENTIONS.md "Model tests"):
  Model: the production walk-forward (model.walk_forward on the frame with the role signals), once
  with d_chg_ol as it is (B0) and once with the candidate's (C), each fitted and predicting on its
  own input.
  Measures, played games with a closing spread, tuning 2014-18 and holdout 2019 on, paired by game:
    1. average miss of the raw model's margin (model_margin)
    2. average miss of the market-adjusted margin the board shows (proj_margin)
  Kept only if C is lower on both measures in both periods and the holdout difference is at least
  1 standard error on each. Kept means provisional (rechecked on the next full season). Anything
  else is reported as it is; because B0 is a known defect, a mixed or flat result goes to Ryan as
  a trade-off (fix the input for correctness, or leave the fitted model alone).
  Totals: no injury input in the totals model, so no total moves.
  Props check: the player projections do not read the offensive-line group (they read opp_miss_db,
  opp_miss_lb and the pass rush). Reported as "none changed" for this test. The same name match
  feeds those defensive groups; how often it fails there is counted below and is not fixed or
  tested here.
Reported, not in the rule: how often the two inputs differ, by season; the candidate's coefficient;
the 2026 season alone; the share of core player-games each group loses to the name match.

Writes experiments/out/ol_name_match_test.json.
"""
from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def _tokens(name: str) -> list[str]:
    t = ["".join(ch for ch in w if ch.isalpha()) for w in str(name).lower().replace("-", " ").split()]
    t = [w for w in t if w]
    while len(t) > 1 and t[-1] in SUFFIX:
        t.pop()
    return t


def alias_map(src: pd.DataFrame, dst: pd.DataFrame, norm) -> dict:
    """{(team, src key): dst key} for src names with no exact match in dst on the same team.
    src, dst: frames with team and name columns (raw names). See the module docstring."""
    out = {}
    dst = dst.drop_duplicates(["team", "name"])
    by_team = {t: list(g["name"]) for t, g in dst.groupby("team")}
    for team, g in src.drop_duplicates(["team", "name"]).groupby("team"):
        names = by_team.get(team, [])
        keys = {norm(n) for n in names}
        base, fl = {}, {}
        for n in names:
            tk = _tokens(n)
            if not tk:
                continue
            base.setdefault("".join(tk), set()).add(norm(n))
            fl.setdefault((tk[0][0], tk[-1]), set()).add(norm(n))
        for n in g["name"]:
            k, tk = norm(n), _tokens(n)
            if k in keys or not tk:
                continue
            hit = base.get("".join(tk)) or fl.get((tk[0][0], tk[-1])) or set()
            if len(hit) == 1:
                out[(team, k)] = next(iter(hit))
    return out


def candidate_season(injuries):
    """injuries._season with the looser match (source patch, as review_fix_tests.py T1 does)."""
    src = inspect.getsource(injuries._season)
    a = '    roster_weeks = ros.groupby("team")["week"].apply(lambda w: sorted(set(w))).to_dict()\n'
    b = "rep = report.get((team, wk, key))"
    c = 'st = roster.get((team, rweek, key)) if rweek is not None else "ACT"'
    assert all(x in src for x in (a, b, c)), "injuries._season changed: update the patch"
    src = src.replace(a, a + '    _sn = s[["team", "player"]].rename(columns={"player": "name"})\n'
                             '    al_ros = alias_map(_sn, ros[["team", "full_name"]].rename(columns={"full_name": "name"}), _norm)\n'
                             '    al_rep = alias_map(_sn, inj[["team", "full_name"]].rename(columns={"full_name": "name"}), _norm)\n'
                             '    ALIASES.append((season, len(al_ros), len(al_rep)))\n')
    src = src.replace(b, "rep = report.get((team, wk, al_rep.get((team, key), key)))")
    src = src.replace(c, 'st = roster.get((team, rweek, al_ros.get((team, key), key))) if rweek is not None else "ACT"')
    ns = dict(vars(injuries))
    ns.update(alias_map=alias_map, ALIASES=[])
    exec(src.replace("def _season(", "def _season_loose("), ns)
    return ns["_season_loose"], ns["ALIASES"]


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    return {"diff": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4), "n": int(len(d))}


def main() -> None:
    from nflmodel import data, extra, game_roles, injuries, model, players, players_data, roles, weather
    from nflmodel.features_ext import PRODUCTION_METRICS, ExtRatingEngine
    games, tg, qb, _ = data.load_all(refresh=False, verbose=False, with_stats=True)
    team_extra, qb_extra = extra.load_extra(refresh_current=False, verbose=False)
    games = weather.fill_wind(games, verbose=False)
    prior = None
    try:
        from nflmodel import pff_qb
        prior = pff_qb.load(qb, sorted(games.season.unique()))
    except Exception as exc:  # noqa: BLE001
        print(f"PFF quarterback prior skipped ({exc})")
    print("quarterback prior:", "PFF grades" if prior is not None else "flat (no PFF data: NOT the production model)")
    last = int(games[games.played].season.max())

    loose, aliases = candidate_season(injuries)
    now_parts, cand_parts = [], []
    for s in range(injuries.FIRST, last + 1):           # both rebuilt from the same downloads' logic
        try:
            now_parts.append(injuries._season(s, games))
            cand_parts.append(loose(s, games))
        except Exception as exc:  # noqa: BLE001
            print(f"  {s}: skipped ({exc})")
    inj_now, inj_c = pd.concat(now_parts, ignore_index=True), pd.concat(cand_parts, ignore_index=True)

    engine = ExtRatingEngine(tg, qb, team_extra, qb_extra, extra_metrics=PRODUCTION_METRICS, games=games, qb_prior_model=prior)
    pg, pinfo = players_data.load_player_games(refresh_current=False, verbose=False)
    avail = players.load_availability(refresh_current=False, verbose=False)
    R = roles.load_all(games, refresh_current=False, verbose=False)

    def walk(inj):
        base = model.build_frame(games, engine, injuries=inj)
        f0, _ = model.walk_forward(base)
        rows = players.player_rows(players.team_games(f0, pg), pg, avail, pinfo)
        f, info = model.walk_forward(game_roles.attach(base, game_roles.team_features(rows, R, avail)))
        return f, info

    f_now, _ = walk(inj_now)
    f_c, _ = walk(inj_c)
    x = f_now.merge(f_c[["game_id", "model_margin", "proj_margin", "d_chg_ol"]], on="game_id", suffixes=("", "_c"))
    p = x[x.played & x.spread_line.notna() & (x.season >= 2014)]
    per = {"tuning 2014-18": p[p.season < 2019], "holdout 2019 on": p[p.season >= 2019], f"{last} alone": p[p.season == last]}
    out = {"qb_prior": "pff" if prior is not None else "flat", "measures": {}}
    for name, col in (("raw model margin", "model_margin"), ("market-adjusted margin", "proj_margin")):
        out["measures"][name] = {}
        for k, d in per.items():
            mn, mc = (d[col] - d.result).abs(), (d[col + "_c"] - d.result).abs()
            out["measures"][name][k] = {"now": round(float(mn.mean()), 4), "candidate": round(float(mc.mean()), 4), **paired(mc, mn)}
    keep = all(out["measures"][m][k]["diff"] < 0 for m in out["measures"] for k in list(per)[:2]) and \
        all(-out["measures"][m]["holdout 2019 on"]["diff"] >= out["measures"][m]["holdout 2019 on"]["se"] for m in out["measures"])
    out["kept"] = bool(keep)

    # how big the defect is
    m = inj_now.merge(inj_c, on=["season", "game_id", "team"], suffixes=("", "_c"))
    played_ids = set(games.loc[games.played, "game_id"])
    mp = m[m.game_id.isin(played_ids)]
    out["defect"] = {"aliases_by_season (season, roster, report)": aliases,
                     "by_group": {g: {"team_games_differ": round(float((~np.isclose(mp[f"miss_{g}"], mp[f"miss_{g}_c"])).mean()), 4),
                                      "mean_missing_now": round(float(mp[f"miss_{g}"].mean()), 4),
                                      "mean_missing_candidate": round(float(mp[f"miss_{g}_c"].mean()), 4)} for g in injuries.GROUPS},
                     "ol_by_season": {int(s): {"team_games_differ": round(float((~np.isclose(d.miss_ol, d.miss_ol_c)).mean()), 4),
                                               "mean_missing_now": round(float(d.miss_ol.mean()), 4),
                                               "mean_missing_candidate": round(float(d.miss_ol_c.mean()), 4)} for s, d in mp.groupby("season")}}
    out["games_where_d_chg_ol_differs"] = round(float((~np.isclose(p.d_chg_ol, p.d_chg_ol_c)).mean()), 4)
    MF = model.MARGIN_FEATS
    tr_now, tr_c = f_now[f_now.played], f_c[f_c.played]
    out["coef_d_chg_ol (all played seasons)"] = {
        "now": model.Linear(20.0, intercept=False).fit(tr_now[MF].to_numpy(float), tr_now["result"].to_numpy(float)).raw_coefs(MF)["d_chg_ol"],
        "candidate": model.Linear(20.0, intercept=False).fit(tr_c[MF].to_numpy(float), tr_c["result"].to_numpy(float)).raw_coefs(MF)["d_chg_ol"]}
    up = x[~x.played & x.spread_line.notna() & (x.season == last)]
    mv = (up.proj_margin_c - up.proj_margin).abs()
    out["upcoming_games"] = {"n": int(len(up)), "board_margin_moves_mean": round(float(mv.mean()), 3) if len(up) else None,
                             "board_margin_moves_max": round(float(mv.max()), 3) if len(up) else None,
                             "raw_model_moves_max": round(float((up.model_margin_c - up.model_margin).abs().max()), 3) if len(up) else None}
    OUT.mkdir(exist_ok=True)
    (OUT / "ol_name_match_test.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
