import pandas as pd

from nflmodel import players as PL


def _team(rows):
    d = pd.DataFrame(rows).sort_values("att_share_raw", ascending=False)
    return PL._pick_starter(d)


def test_last_starter_active_keeps_job():
    rows = [{"player_id": "a", "prev_qb": "a", "att_share_raw": 0.7, "pass_att_sum": 200.0},
            {"player_id": "b", "prev_qb": "a", "att_share_raw": 0.1, "pass_att_sum": 10.0}]
    d = pd.DataFrame(rows, index=[10, 11])
    assert PL._pick_starter(d.sort_values("att_share_raw", ascending=False)) == 10


def test_backup_with_passes_beats_unknown_qb():
    # last starter (traded/out) not active; an unknown QB has the default share (~0.3),
    # the season-long backup has a small share but real recent passes: the backup starts
    rows = [{"player_id": "beck", "prev_qb": "brissett", "att_share_raw": 0.3, "pass_att_sum": 0.0},
            {"player_id": "minshew", "prev_qb": "brissett", "att_share_raw": 0.05, "pass_att_sum": 23.0}]
    d = pd.DataFrame(rows, index=[1, 2]).sort_values("att_share_raw", ascending=False)
    assert PL._pick_starter(d) == 2


def test_no_passes_anywhere_falls_back_to_share():
    rows = [{"player_id": "x", "prev_qb": "z", "att_share_raw": 0.3, "pass_att_sum": 0.0},
            {"player_id": "y", "prev_qb": "z", "att_share_raw": 0.2, "pass_att_sum": 0.0}]
    d = pd.DataFrame(rows, index=[5, 6]).sort_values("att_share_raw", ascending=False)
    assert PL._pick_starter(d) == 5
