"""Compare the board's projected starting quarterbacks with nfelo's, for the upcoming NFL games.

    python -m nflmodel.qbcheck

The board assumes last week's starter unless data/overrides.csv says otherwise; nfelo does the
same and its author corrects the list by hand when news breaks (github.com/greerreNFL/nfeloqb,
qb_elos.csv, updated Tuesday and Thursday mornings). A disagreement is worth a look before
betting: the board's starter pick is wrong about one game in fourteen.

Reads the board's upcoming games (docs/data.json slate: away_qb, home_qb) and nfelo's qb_elos.csv
rows not yet played (qb1 = home, qb2 = away). Writes data/qb_check/latest.json (this run's
comparison and whether the disagreements changed since the last run) and appends every comparison
to data/qb_check/log.csv. No Odds API credits.

    python -m nflmodel.qbcheck --apply

also takes nfelo's pick where the two disagree (Ryan, Nov 2, 2026): a `starter` row in
data/overrides.csv for that game, which the next board update uses. Two safeguards: a starter row
entered by hand for that game is never replaced (hand-entered news is newer than nfelo's
Tuesday/Thursday list), a quarterback with an `out` row for that game (benched, traded or
injured since nfelo's list) is never started, and a pick whose quarterback can't be matched to a
player id is only reported. Rows this check wrote earlier are replaced when nfelo changes its pick again. What was
applied or skipped is written to data/qb_check/applied.json.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import unicodedata
import urllib.request

import pandas as pd

from . import config as C

URL = "https://raw.githubusercontent.com/greerreNFL/nfeloqb/main/qb_elos.csv"
OUT = C.DATA / "qb_check"
ALIAS = {"OAK": "LV", "LAR": "LA", "SD": "LAC", "STL": "LA", "WSH": "WAS"}


def norm(name: str | None) -> str:
    """Compare names loosely: accents, punctuation and suffixes (Jr., II) don't count."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def nfelo_starters(today: dt.date) -> pd.DataFrame:
    req = urllib.request.Request(URL, headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, timeout=120) as r:
        q = pd.read_csv(io.BytesIO(r.read()), usecols=["date", "team1", "team2", "qb1", "qb2", "score1", "game_id"])
    q = q[(pd.to_datetime(q["date"]).dt.date >= today) & q["score1"].isna()]
    rows = []
    for x in q.itertuples():
        for team, qb in ((x.team1, x.qb1), (x.team2, x.qb2)):
            rows.append({"game_id": x.game_id, "date": x.date, "team": ALIAS.get(team, team), "nfelo_qb": qb})
    return pd.DataFrame(rows)


def board_starters(today: dt.date) -> pd.DataFrame:
    d = json.loads((C.DOCS / "data.json").read_text())
    rows = []
    for g in (d.get("slate") or {}).get("games", []):
        if pd.Timestamp(g["kickoff"]).date() < today:
            continue
        for side in ("home", "away"):
            rows.append({"game_id": g["id"], "team": g[side], "board_qb": (g.get(f"{side}_qb") or {}).get("name"),
                         "kickoff": g["kickoff"]})
    return pd.DataFrame(rows)


def compare(board: pd.DataFrame, nfelo: pd.DataFrame) -> pd.DataFrame:
    # each team's next game in nfelo's file (its game ids can spell the Raiders and Rams differently)
    nxt = nfelo.sort_values("date").drop_duplicates("team")[["team", "nfelo_qb"]]
    m = board.sort_values("kickoff").drop_duplicates("team").merge(nxt, on="team", how="left")
    m["agree"] = [norm(a) == norm(b) if isinstance(b, str) else None for a, b in zip(m.board_qb, m.nfelo_qb)]
    return m


def run(now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    today = now.astimezone(dt.timezone(dt.timedelta(hours=-5))).date()
    OUT.mkdir(parents=True, exist_ok=True)
    m = compare(board_starters(today), nfelo_starters(today))
    m.insert(0, "checked_utc", now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    log = OUT / "log.csv"
    m.to_csv(log, mode="a", header=not log.exists(), index=False)
    diff = m[m.agree == False]  # noqa: E712  (None = nfelo has no row for the game yet)
    key = sorted(f"{r.game_id}|{r.team}|{r.board_qb}|{r.nfelo_qb}" for r in diff.itertuples())
    last = OUT / "latest.json"
    prev = json.loads(last.read_text()).get("key", []) if last.exists() else []
    res = {"checked_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "games": int(m.game_id.nunique()),
           "compared": int(m.agree.notna().sum()), "not_in_nfelo": int(m.agree.isna().sum()),
           "disagree": [{"game_id": r.game_id, "team": r.team, "board": r.board_qb, "nfelo": r.nfelo_qb, "kickoff": r.kickoff}
                        for r in diff.itertuples()],
           "changed": key != prev, "key": key}
    last.write_text(json.dumps(res, indent=1))
    print(f"QB check: {res['compared']} starters compared over {res['games']} games, "
          f"{len(res['disagree'])} disagree{' (changed)' if res['changed'] else ''}, {res['not_in_nfelo']} not in nfelo's file yet")
    for x in res["disagree"]:
        print(f"  {x['game_id']} {x['team']}: board {x['board']}, nfelo {x['nfelo']}")
    return res


AUTO = "(auto, qbcheck)"


def qb_ids(season: int) -> pd.DataFrame:
    """Quarterbacks with a gsis id: name, the teams they have thrown for this season."""
    info = pd.read_csv(C.CACHE / f"player_info_{season}.csv", dtype=str)
    info = info[info["position"] == "QB"][["player_id", "name"]]
    try:
        pg = pd.read_csv(C.CACHE / f"player_games_{season}.csv", dtype={"player_id": str})
        teams = pg[pg["pass_att"].fillna(0) > 0].groupby("player_id")["posteam"].agg(lambda x: set(x))
    except Exception:
        teams = pd.Series(dtype=object)
    info["teams"] = info["player_id"].map(teams)
    info["key"] = info["name"].map(norm)
    return info


def pick_id(qbs: pd.DataFrame, name: str, team: str) -> tuple[str, str] | None:
    """(player_id, name) of nfelo's quarterback: a QB of that name who has thrown for the team this
    season, else the only QB of that name. None when it isn't clear."""
    m = qbs[qbs["key"] == norm(name)]
    on_team = m[m["teams"].map(lambda t: isinstance(t, set) and team in t)]
    hit = on_team if len(on_team) == 1 else (m if len(m) == 1 else None)
    return None if hit is None else (hit.iloc[0]["player_id"], hit.iloc[0]["name"])


def apply(res: dict, season: int, path=None) -> dict:
    """Write nfelo's pick as a starter override for each disagreement (see the module note)."""
    from . import overrides
    path = overrides.PATH if path is None else path
    ov = overrides.load(path)
    qbs = qb_ids(season)
    applied, skipped = [], []
    for x in res["disagree"]:
        mine = ov[(ov["game_id"] == x["game_id"]) & (ov["team"] == x["team"]) & (ov["kind"] == "starter")]
        if len(mine) and not mine["source"].str.contains(AUTO, regex=False).all():
            skipped.append({**x, "why": "starter entered by hand for this game; kept"})
            continue
        hit = pick_id(qbs, x["nfelo"], x["team"])
        if hit is None:
            skipped.append({**x, "why": "nfelo's quarterback could not be matched to a player id"})
            continue
        # nfelo's list lags hand-entered news (benchings, trades, injuries): never start a
        # quarterback an `out` row takes out of this game
        if len(ov[(ov["game_id"] == x["game_id"]) & (ov["kind"] == "out") & (ov["player_id"] == hit[0])]):
            skipped.append({**x, "why": "nfelo's quarterback has an out row for this game; kept"})
            continue
        ov = ov.drop(mine.index)
        row = {"game_id": x["game_id"], "team": x["team"], "kind": "starter", "player_id": hit[0], "name": hit[1], "status": "",
               "source": f"nfelo qb_elos.csv, checked {res['checked_utc'][:10]}: nfelo starts {x['nfelo']}, board had {x['board']} {AUTO}"}
        ov = pd.concat([ov, pd.DataFrame([row])], ignore_index=True)
        applied.append({**x, "player_id": hit[0]})
    if applied:
        ov[overrides.COLS].to_csv(path, index=False)
    out = {"checked_utc": res["checked_utc"], "applied": applied, "skipped": skipped}
    (OUT / "applied.json").write_text(json.dumps(out, indent=1))
    for a in applied:
        print(f"  applied: {a['game_id']} {a['team']} starter {a['board']} -> {a['nfelo']}")
    for k in skipped:
        print(f"  not applied: {k['game_id']} {k['team']} (board {k['board']}, nfelo {k['nfelo']}): {k['why']}")
    return out


def issue_body(res: dict) -> str:
    """Markdown for the GitHub issue the workflow opens or updates."""
    if not res["disagree"]:
        return f"Board and nfelo agree on every projected starter ({res['compared']} compared, checked {res['checked_utc']})."
    lines = [f"Checked {res['checked_utc']}: the board and nfelo disagree on {len(res['disagree'])} projected starter(s).", "",
             "| Game | Team | Board | nfelo |", "|---|---|---|---|"]
    lines += [f"| {x['game_id']} | {x['team']} | {x['board']} | {x['nfelo']} |" for x in res["disagree"]]
    lines += ["", "If the board is wrong, add a `starter` row to data/overrides.csv. nfelo updates its list Tuesday and Thursday mornings, "
              "so on other days its pick can be the stale one."]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    r = run()
    if "--apply" in sys.argv[1:]:
        from .data import current_season
        apply(r, current_season())
    if "--issue-body" in sys.argv[1:]:
        (OUT / "issue.md").write_text(issue_body(r))
