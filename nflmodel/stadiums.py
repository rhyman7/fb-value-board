"""Which way each NFL field points, so a wind can be split into its along-field and
crosswind parts.

The model today sees only wind speed. A 15 mph wind blowing goal line to goal line
hurts passing and kicking far more than the same wind blowing across the field, so
the direction matters. Open-Meteo gives the direction the wind blows FROM (0 = from
the north, 90 = from the east); the tables below give the compass bearing of each
field's long axis, 0-179 degrees (0 = goal lines face north and south, 90 = east and
west). The two components:

    along = |wind x cos(dir - bearing)|      blowing down the field, one way or the other
    cross = |wind x sin(dir - bearing)|      blowing across it

The bearings were set from the public layout of each stadium (where the open end and
the home sideline face), not from a survey, and none was re-measured when this table
was written (Oct 2026). "measured" means the long axis is well established, north-south
within about 10 degrees; "approx" means it is from general knowledge of the layout and
may be off by 15 degrees or more. Domes and stadiums usually closed are tagged approx:
their wind is zero in the model anyway. Refine a bearing by reading it off a satellite
view (the angle of the goal lines from north), and keep the tag honest.
"""
from __future__ import annotations

import math

# Current home stadiums, by team abbreviation (config.TEAM_NAMES): (bearing, confidence).
FIELDS: dict[str, tuple[int, str]] = {
    "ARI": (0, "approx"),       # State Farm Stadium (retractable, usually closed)
    "ATL": (170, "approx"),     # Mercedes-Benz Stadium (retractable, usually closed)
    "BAL": (0, "measured"),     # M&T Bank Stadium, north-south
    "BUF": (20, "approx"),      # Highmark Stadium, roughly north-south, tilted a little east of north
    "CAR": (10, "approx"),      # Bank of America Stadium
    "CHI": (0, "measured"),     # Soldier Field, north-south along the lakefront
    "CIN": (10, "approx"),      # Paycor Stadium, roughly north-south
    "CLE": (75, "approx"),      # Huntington Bank Field, along the lakeshore; Dawg Pound at the east end
    "DAL": (100, "approx"),     # AT&T Stadium (retractable, usually closed)
    "DEN": (0, "measured"),     # Empower Field at Mile High, north-south
    "DET": (60, "approx"),      # Ford Field (dome)
    "GB": (0, "measured"),      # Lambeau Field, north-south
    "HOU": (170, "approx"),     # NRG Stadium (retractable, usually closed)
    "IND": (0, "approx"),       # Lucas Oil Stadium (retractable; window at the north end)
    "JAX": (10, "approx"),      # EverBank Stadium, roughly north-south
    "KC": (10, "approx"),       # GEHA Field at Arrowhead, roughly north-south
    "LA": (10, "approx"),       # SoFi Stadium (covered)
    "LAC": (10, "approx"),      # SoFi Stadium (covered)
    "LV": (90, "approx"),       # Allegiant Stadium (dome)
    "MIA": (0, "approx"),       # Hard Rock Stadium, home sideline on the west side
    "MIN": (90, "approx"),      # U.S. Bank Stadium (dome)
    "NE": (0, "approx"),        # Gillette Stadium, lighthouse at the north end
    "NO": (0, "approx"),        # Caesars Superdome (dome)
    "NYG": (20, "approx"),      # MetLife Stadium, roughly north-south
    "NYJ": (20, "approx"),      # MetLife Stadium
    "PHI": (0, "measured"),     # Lincoln Financial Field, north-south
    "PIT": (150, "approx"),     # Acrisure Stadium, open end toward the rivers and downtown (south-east)
    "SEA": (0, "measured"),     # Lumen Field, north-south, open to downtown at the north end
    "SF": (0, "approx"),        # Levi's Stadium, suite tower on the west sideline
    "TB": (0, "measured"),      # Raymond James Stadium, pirate ship at the north end
    "TEN": (10, "approx"),      # Nissan Stadium, roughly north-south along the river
    "WAS": (90, "approx"),      # Northwest Stadium (FedExField), roughly east-west
}

# Outdoor stadiums in the games file since 2014 that teams have since left, by the name the
# nflverse games file uses: (lat, lon, bearing, confidence). Looked up by stadium name first
# so the archive fetch asks for the weather where the game was actually played.
PAST: dict[str, tuple[float, float, int, str]] = {
    "Los Angeles Memorial Coliseum": (34.014, -118.288, 90, "measured"),   # Rams 2016-2019; peristyle at the east end
    "StubHub Center": (33.864, -118.261, 0, "approx"),                      # Chargers 2017-2019 (Dignity Health Sports Park)
    "Dignity Health Sports Park": (33.864, -118.261, 0, "approx"),
    "Oakland-Alameda County Coliseum": (37.752, -122.201, 0, "approx"),    # Raiders through 2019; Mount Davis on the east side
    "O.co Coliseum": (37.752, -122.201, 0, "approx"),
    "Oakland Coliseum": (37.752, -122.201, 0, "approx"),
    "Ring Central Coliseum": (37.752, -122.201, 0, "approx"),
    "Qualcomm Stadium": (32.783, -117.119, 0, "approx"),                   # Chargers through 2016
    "SDCCU Stadium": (32.783, -117.119, 0, "approx"),
    "TCF Bank Stadium": (44.976, -93.225, 90, "approx"),                   # Vikings 2014-2015; open end toward downtown (west)
}


def bearing(team: str, stadium: str | None = None) -> tuple[int, str] | None:
    """(bearing, confidence) for the field a team plays on, or None when unknown."""
    if stadium in PAST:
        _, _, b, conf = PAST[stadium]
        return b, conf
    return FIELDS.get(team)


def site(team: str, stadium: str | None = None) -> tuple[float, float, int, str] | None:
    """(lat, lon, bearing, confidence) for a home game: the old stadium when the games file
    names one in PAST, else the team's current stadium (features_ext.SITES for the location)."""
    if stadium in PAST:
        return PAST[stadium]
    from .features_ext import SITES
    if team in SITES and team in FIELDS:
        lat, lon, _ = SITES[team]
        b, conf = FIELDS[team]
        return lat, lon, b, conf
    return None


def components(wind_mph: float, dir_deg: float, field_bearing: float) -> tuple[float, float]:
    """Split a wind (speed, direction it blows from) into (along_mph, cross_mph) for a field
    whose long axis has the given bearing. Both are absolute values."""
    rad = math.radians(float(dir_deg) - float(field_bearing))
    along = abs(float(wind_mph) * math.cos(rad))
    cross = abs(float(wind_mph) * math.sin(rad))
    return along, cross


def crosswind(team: str, wind_mph: float, dir_deg: float, stadium: str | None = None) -> tuple[float, float]:
    """(along_mph, cross_mph) of a wind at a team's home field; (nan, nan) when the field's
    bearing, the speed or the direction is unknown."""
    b = bearing(team, stadium)
    if b is None or wind_mph is None or dir_deg is None:
        return math.nan, math.nan
    try:
        w, d = float(wind_mph), float(dir_deg)
    except (TypeError, ValueError):
        return math.nan, math.nan
    if math.isnan(w) or math.isnan(d):
        return math.nan, math.nan
    return components(w, d, b[0])
