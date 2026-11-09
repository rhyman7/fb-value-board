"""One-off (Nov 9, 2026): add the kneel-down columns to the committed player box-score caches.

    python scripts/add_kneels.py            # every season from players_data.FIRST

nflmodel/players_data.py writes `kneels` and `kneel_yds` from now on (the Update board Action
re-reduces the current season daily). This adds them to the seasons already cached without
re-reducing those files, so no other number in them can move: it downloads each season's
play-by-play, counts the kneel-downs per rusher per game, and merges the two columns onto the
existing rows by game and player. A player whose only plays were kneels has no row and gets none.
Safe to run again: the columns are replaced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import config as C, players_data as PD  # noqa: E402
from nflmodel.data import _download, current_season  # noqa: E402

KEY = ["game_id", "season", "week", "season_type", "posteam", "defteam"]


def main():
    seasons = [int(a) for a in sys.argv[1:]] or range(PD.FIRST, current_season() + 1)
    for season in seasons:
        path = C.CACHE / f"player_games_{season}.csv"
        if not path.exists():
            continue
        raw = C.RAW / f"pbp_kneel_{season}.parquet"
        _download(C.PBP_URL.format(season=season), raw)
        p = pd.read_parquet(raw, columns=["game_id", "season", "week", "season_type", "posteam", "defteam", "qb_kneel", "rush_attempt",
                                          "rusher_player_id", "rushing_yards", "two_point_attempt"])
        raw.unlink()
        p = p[(p["two_point_attempt"].fillna(0) == 0) & p["posteam"].notna()]
        k = PD.kneels(p, KEY)[["game_id", "player_id", "kneels", "kneel_yds"]]
        text = path.read_text()
        pg = pd.read_csv(path, dtype=str, keep_default_na=False)      # as text: nothing else is reformatted
        pg = pg.drop(columns=[c for c in ("kneels", "kneel_yds") if c in pg])
        out = pg.merge(k, on=["game_id", "player_id"], how="left")
        assert len(out) == len(pg)
        for c in ("kneels", "kneel_yds"):
            out[c] = out[c].fillna(0.0).round(2)
        out.to_csv(path, index=False)
        back = pd.read_csv(path, dtype=str, keep_default_na=False).drop(columns=["kneels", "kneel_yds"])
        old = pd.read_csv(pd.io.common.StringIO(text), dtype=str, keep_default_na=False).drop(columns=["kneels", "kneel_yds"], errors="ignore")
        assert back.equals(old), f"{season}: other columns changed"
        lost = len(k) - int((out["kneels"] > 0).sum())
        print(f"{season}: {int(out['kneels'].sum())} kneels on {int((out['kneels'] > 0).sum())} player-games, "
              f"{out['kneel_yds'].sum():.0f} yards; {lost} kneel-only player-games left without a row", flush=True)


if __name__ == "__main__":
    main()
