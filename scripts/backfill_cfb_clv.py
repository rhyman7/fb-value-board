#!/usr/bin/env python3
"""One-time correction (Nov 11, 2026, Ryan's go-ahead): fill the closing line value the college
ledger and Best Bets results never got.

Until this date cfbmodel/pipeline.py handed ledger.grade the closing quotes under float ids
("401856733.0"), so no graded college row found its quote and close_line, close_prob and clv
were written empty. This fills those three columns, and only those, on rows where clv is empty
and a closing quote is on file (data/market/), exactly as ledger.grade would have: the sharp
book's last quote before kickoff, carried to the bet's line through the key-number curve fitted
on the seasons before this one. Results and units are not touched. Rows whose game has no
closing quote stay empty.

    python scripts/backfill_cfb_clv.py          # dry run: prints what would change
    python scripts/backfill_cfb_clv.py --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfbmodel import bets, config as C, data, teams as T          # noqa: E402
from cfbmodel.pipeline import close_by_id                          # noqa: E402
from nflmodel import ledger, market                                # noqa: E402

PAIRS = (("ledger.csv", "ledger_results.csv"), ("bestbets.csv", "bestbets_results.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    games, _, _ = data.load_all(refresh=False, verbose=False)
    season = int(games[games.played].season.max())
    names = T.load_teams(season, refresh=False)["name"].to_dict()
    done = games[(games.season == season) & games.played]
    close = close_by_id(market.closing_quotes(done, "cfb", {g.game_id: pd.Timestamp(g.kickoff) for g in done.itertuples()},
                                              names={v: k for k, v in names.items()}))
    dist = bets.fit_distribution(games[games.fbs_game == 1], season)
    curve = lambda df: bets.curves(df, dist, "line")               # noqa: E731
    for log, res in PAIRS:
        lp, rp = C.DATA / log, C.DATA / res
        if not lp.exists() or not rp.exists():
            continue
        led = pd.read_csv(lp, dtype={"row_id": str, "game_id": str, "data_id": str})
        out = pd.read_csv(rp, dtype={"row_id": str})
        if log.startswith("bestbets"):                 # several entries per bet: the graded one is the card at kickoff
            from nflmodel import bestbets
            rows = bestbets.final(lp, now=pd.Timestamp.now(tz="UTC").to_pydatetime()).set_index("row_id")
        else:
            rows = led.drop_duplicates("row_id", keep="last").set_index("row_id")
        n = 0
        for i, r in out.iterrows():
            if pd.notna(r.get("clv")) or r["row_id"] not in rows.index:
                continue
            b = rows.loc[r["row_id"]]
            cv = ledger._clv(b, close.get(str(b["data_id"])), curve)
            if cv["clv"] is None:
                continue
            out.loc[i, ["close_line", "close_prob", "clv"]] = [cv["close_line"], cv["close_prob"], cv["clv"]]
            n += 1
        print(f"{res}: {n} of {len(out)} rows filled; {int(out['clv'].isna().sum())} still empty (no closing quote)")
        if args.write and n:
            out.to_csv(rp, index=False)


if __name__ == "__main__":
    main()
