import pandas as pd

from nflmodel import tdtracker


def test_eligible_follows_report_rules():
    c = pd.DataFrame({"week": 5, "game_id": "g", "player_id": ["rb", "wr", "te", "qb1", "qb2", "rbx"],
                      "name": list("abcdef"), "pos": ["RB", "WR", "TE", "QB", "QB", "RB"], "team": "T", "opp": "O",
                      "status": [None, None, None, None, None, "Out"], "chance": 0.4, "pass_att_proj": [0, 0, 0, 30, 5, 0]})
    rows = []
    for w in (1, 2, 3):
        rows += [dict(game_id=f"w{w}", week=w, player_id="rb", carries=12, receptions=1, rush_td=1, rec_td=0),
                 dict(game_id=f"w{w}", week=w, player_id="wr", carries=0, receptions=3, rush_td=0, rec_td=1),   # 3 catches: under 3.5
                 dict(game_id=f"w{w}", week=w, player_id="te", carries=0, receptions=4, rush_td=0, rec_td=1),
                 dict(game_id=f"w{w}", week=w, player_id="qb1", carries=5, receptions=0, rush_td=1, rec_td=0),
                 dict(game_id=f"w{w}", week=w, player_id="qb2", carries=5, receptions=0, rush_td=1, rec_td=0),
                 dict(game_id=f"w{w}", week=w, player_id="rbx", carries=15, receptions=2, rush_td=2, rec_td=0)]
    x = tdtracker._eligible(c, pd.DataFrame(rows), 5)
    assert sorted(x.player_id) == ["qb1", "rb", "te"]
