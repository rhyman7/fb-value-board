"""Typical: the model's own 50/50 yardage for a player this week (Nov 18, 2026, Ryan's ask).

A projection is an average, and yardage has a long upside tail, so a correct projection is met
less than half the time. Typical is the line at which the board's prop distribution
(props.PropModel with the fitted widths in data/prop_params.json, or data/cfb/prop_params.json
for college) gives an even chance of over and under, at the player's projection. About half of
players should reach it; the Weekly Review's Players page scores that.

Shown next to the projection on both boards (Players tab, the player's detail, each game's
matchup tables), carried in the dashboards' feed (`t`), and printed in the matchup reports.
**Display only**: nothing reads it back; no projection, price or bet changes.

    attach(board, league)   sets p["typical"] = {"pass_yds": .., "rush_yds": .., "rec_yds": ..}
    medians(league, players, given_plays)   {(player id, game, stat): yards}
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

STATS = ("pass_yds", "rush_yds", "rec_yds")
MIN_PROJ = 1.0


def medians(league: str, players: list[dict], given_plays: bool = False) -> dict:
    """given_plays: college players shaded for a chance to sit are priced on the projection if
    they play (what a prop is priced on); the boards show the shaded number, so they pass False."""
    from . import parlays as PL, propmarket as PM, props as P
    players = copy.deepcopy(players)
    if league == "cfb":            # the college board carries reception share where the widths expect target share
        for p in players:
            why = p.get("why")
            if isinstance(why, dict):
                for a, b in (("tgt_share", "rec_share"), ("tgt_share_hist", "rec_share_hist")):
                    if why.get(a) is None and why.get(b) is not None:
                        why[a] = why[b]
            if given_plays and isinstance(p.get("proj_played"), dict):
                p["proj"] = dict(p["proj_played"])
    df = PM.board_players({"players": {"players": players}})
    model = P.PropModel.load(PL.prop_params_path(league))
    if model is None or df.empty:
        return {}
    out = {}
    for stat in STATS:
        spec = P.SPECS[stat]
        for grp in spec.groups:
            if (stat, grp) not in model.yards:
                continue
            d = df[df["group"] == grp].reset_index(drop=True)
            for c in spec.proj + P.SPECS[spec.vol].proj:
                if c not in d:
                    d[c] = 0.0
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
            d = d[d[spec.proj[0]] >= MIN_PROJ].reset_index(drop=True)
            if d.empty:
                continue
            mu = P.mean(d, stat)
            L = np.floor(mu[:, None] * np.linspace(0.05, 1.7, 67)[None, :]) + 0.5
            over, _, under = model.probs(d, stat, grp, L)
            dec = over / np.maximum(over + under, 1e-9)
            for i in range(len(d)):       # dec falls as the line rises: the line where it crosses a half
                out[(str(d.at[i, "player_id"]), d.at[i, "game_id"], stat)] = round(float(np.interp(0.5, dec[i][::-1], L[i][::-1])), 1)
    return out


def attach(board: dict, league: str) -> int:
    """Set p["typical"] on every board player with a yardage projection. Returns players touched."""
    pl = board.get("players") or []
    med = medians(league, pl)
    n = 0
    for p in pl:
        t = {s: med[(str(p["id"]), p["game"], s)] for s in STATS if (str(p["id"]), p["game"], s) in med}
        if t:
            p["typical"] = t
            n += 1
    return n
