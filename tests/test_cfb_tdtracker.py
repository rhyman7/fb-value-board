import pandas as pd

from cfbmodel import tdtracker


def test_eligible_follows_college_report_rules():
    c = pd.DataFrame({"week": 5, "game": "g", "player_id": ["rb", "wr", "te", "qb1", "qb2", "rbx", "wr2", "idle"],
                      "name": ["a", "b", "c", "Starter", "Backup", "f", "g", "h"],
                      "pos": ["RB", "WR", "TE", "QB", "QB", "RB", "WR", "RB"], "team": "T", "opp": "O",
                      "status": [None, None, None, None, None, "Out", None, None], "chance": 0.4,
                      "starter": "Starter", "backfilled": False})
    rows = []
    for w in (1, 2, 3):
        rows += [dict(game_id=f"w{w}", week=w, team_id=1, player_id="rb", carries=8, receptions=1, rush_td=1, rec_td=0),   # 9 a game
                 dict(game_id=f"w{w}", week=w, team_id=1, player_id="wr", carries=0, receptions=3, rush_td=0, rec_td=1),   # 3: enough in college
                 dict(game_id=f"w{w}", week=w, team_id=1, player_id="te", carries=0, receptions=2, rush_td=0, rec_td=1),
                 dict(game_id=f"w{w}", week=w, team_id=1, player_id="qb1", carries=5, receptions=0, rush_td=1, rec_td=0),
                 dict(game_id=f"w{w}", week=w, team_id=1, player_id="qb2", carries=5, receptions=0, rush_td=1, rec_td=0),
                 dict(game_id=f"w{w}", week=w, team_id=1, player_id="rbx", carries=15, receptions=2, rush_td=2, rec_td=0),
                 dict(game_id=f"w{w}", week=w, team_id=1, player_id="wr2", carries=0, receptions=5, rush_td=0, rec_td=0)]   # no TD yet
    rows += [dict(game_id=f"w{w}", week=w, team_id=1, player_id="idle", carries=12, receptions=1, rush_td=1, rec_td=0) for w in (1, 2)]   # missed last game
    x = tdtracker._eligible(c, pd.DataFrame(rows), 5)
    assert sorted(x.player_id) == ["qb1", "rb", "te", "wr"]
