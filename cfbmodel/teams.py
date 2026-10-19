"""FBS team names, abbreviations and colours for the board (ESPN's team file via cfbfastR)."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .data import _get


def load_teams(season: int, refresh: bool = False) -> pd.DataFrame:
    """school -> abbr, name, colours, conference, stadium location. Cached per season."""
    path = C.CACHE / f"teams_{season}.csv"
    if refresh or not path.exists():
        t = _get("teams", season)
        t = t[t["classification"] == "fbs"]
        out = pd.DataFrame({
            "school": t["school"], "team_id": t["team_id"].astype(int), "abbr": t["abbreviation"],
            "name": t["display_name"], "short": t["short_display_name"],
            "color": "#" + t["color"].fillna("7a857f").astype(str), "alt": "#" + t["alternate_color"].fillna("c0c9c3").astype(str),
            "conference": t["conference_short_name"].fillna(t["cfbd_conference"]),
            "lat": t["latitude"], "lon": t["longitude"], "dome": t["dome"].fillna(False).astype(bool),
            "venue": t["venue_name"], "timezone": t["timezone"],
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    t = pd.read_csv(path)
    # abbreviations must be unique: they are the codes on the board and in game ids
    dup = t["abbr"].duplicated(keep=False)
    t.loc[dup, "abbr"] = t.loc[dup, "abbr"] + t.loc[dup, "team_id"].astype(str).str[-2:]
    return t.set_index("school")
