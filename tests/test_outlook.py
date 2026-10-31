"""Player outlook for the Props section (nflmodel/propoutlook.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import propmarket as PM  # noqa: E402


def test_outlook_ranges_are_ordered_and_match_the_ladder():
    """docs/props/outlook.json: the chance of at least k never rises with k, lo <= mid <= hi,
    and the curve gives the same chance the prop pricing gives at a half-point line."""
    from nflmodel import config as C, propoutlook as OL, props as P
    model = P.PropModel.load()
    payload = json.loads((C.DOCS / "data.json").read_text())
    o = OL.build(payload, model)
    if not o["players"]:
        return
    for k, v in o["players"].items():
        s = np.array(v["s"])
        assert (np.diff(s) <= 1e-9).all() and (s <= 1).all(), k
        assert v["lo"] <= v["mid"] <= v["hi"], k
    pl = PM.board_players(payload)
    r = pl[(pl["group"] == "WR") & (pl["rec_yds_proj"] > 30)].head(1).reset_index(drop=True)
    if len(r):
        v = o["players"][f"{r.at[0, 'game_id']}|{r.at[0, 'player_id']}|rec_yds"]
        over, _, _ = model.probs(r, "rec_yds", "WR", np.array([29.5]))
        assert abs(v["s"][5] - over[0]) < 1e-3          # at least 30 = over 29.5
