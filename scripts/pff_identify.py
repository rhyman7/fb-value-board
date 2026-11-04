"""Identify PFF Premium Stats per-week college exports (receiving / rushing tables, no season or
week columns) by matching their stat lines to the board's box scores, and file them as
hand/<season>/w<week>_<table>.csv in the private PFF data checkout (cfbmodel.pff_data.root(): the
rhyman7/pff-data clone). Never into this repo: it is public (CONVENTIONS.md).

    python scripts/pff_identify.py <dir with csvs> [--copy]

Prints one line per file: table, season, week, how many of the top lines matched; then the
season-weeks covered and missing. With --copy, writes the identified files there.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cfbmodel import config as C, pff_data  # noqa: E402

OUT = pff_data.root() / "hand" if pff_data.available() else None    # needed only for --copy


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def load_box(seasons, games, pinfo):
    out = {}
    for s in seasons:
        f = C.CACHE / f"player_games_{s}.csv"
        if not f.exists():
            continue
        pg = pd.read_csv(f).merge(games[["game_id", "week"]], on="game_id")
        pg["n"] = pg["player_id"].map(pinfo["name"]).map(norm)
        out[s] = pg
    return out


def identify(df: pd.DataFrame, box: dict) -> tuple[str, int | None, int | None, int, int]:
    cols = set(df.columns)
    if "yprr" in cols and "routes" in cols and "attempts" not in cols:
        table, stat, vol, pff_s, pff_v = "receiving", "rec_yds", "receptions", "yards", "receptions"
    elif "attempts" in cols and "ypa" in cols:
        table, stat, vol, pff_s, pff_v = "rushing", "rush_yds", "carries", "yards", "attempts"
    else:
        return "unknown", None, None, 0, 0
    top = df.sort_values(pff_s, ascending=False).head(40)
    hits = {}
    for s, pg in box.items():
        for r in top.itertuples():
            m = pg[(pg["n"] == norm(r.player)) & (pg[stat] == getattr(r, pff_s)) & (pg[vol] == getattr(r, pff_v))]
            for w in m["week"].unique():
                hits[(s, int(w))] = hits.get((s, int(w)), 0) + 1
    if not hits:
        return table, None, None, 0, len(top)
    (s, w), n = max(hits.items(), key=lambda kv: kv[1])
    return table, s, w, n, len(top)


def main():
    src = Path(sys.argv[1])
    copy = "--copy" in sys.argv
    games = pd.read_pickle("/tmp/cfb_games_all.pkl") if Path("/tmp/cfb_games_all.pkl").exists() else None
    pinfo = pd.read_pickle("/tmp/cfb_pinfo_all.pkl") if Path("/tmp/cfb_pinfo_all.pkl").exists() else None
    if games is None or pinfo is None:
        from cfbmodel import data, players_data
        games, _, _ = data.load_all(refresh=False, verbose=False)
        pinfo = players_data.load_player_games(verbose=False)[1]
    box = load_box(range(2017, 2027), games, pinfo)
    found = {}
    for f in sorted(src.glob("*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name}: unreadable ({exc})")
            continue
        table, s, w, n, k = identify(df, box)
        gc = sorted(df["player_game_count"].unique()) if "player_game_count" in df else []
        flag = "" if (n >= k * 0.5 and gc == [1]) else "  <-- check"
        print(f"{f.name:32s} {table:9s} {s} week {w}  ({n}/{k} lines matched; games per player {gc}){flag}")
        if s is not None and n >= k * 0.5 and gc == [1]:
            key = (table, s, w)
            if key in found:
                print(f"   duplicate of {found[key].name}")
                continue
            found[key] = f
            if copy:
                d = OUT / str(s)
                d.mkdir(parents=True, exist_ok=True)
                df.to_csv(d / f"w{w:02d}_{table}.csv", index=False)
    for s in sorted({k[1] for k in found}):
        weeks = sorted(games[(games.season == s) & games.played.astype(bool) & (games.postseason == 0)]["week"].unique())
        for table in ("receiving", "rushing"):
            have = sorted(k[2] for k in found if k[0] == table and k[1] == s)
            missing = [int(w) for w in weeks if int(w) not in have]
            print(f"{s} {table}: have weeks {have}; missing {missing}")


if __name__ == "__main__":
    main()
