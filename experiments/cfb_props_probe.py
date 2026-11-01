"""How many college games have DraftKings or FanDuel player props, and which markets? A cheap check
before pricing college props live or backfilling past seasons.

    python experiments/cfb_props_probe.py [--days 7] [--max-credits 120]

1. Lists college events kicking off in the next --days (free).
2. Asks each one for a single market, passing yards (1 credit if posted, 0 if not).
3. For the first two games that have it, asks for every market the NFL props use (at most 13
   credits each), to see which college markets the books post.
Writes experiments/out/cfb_props_probe.json and prints the cost estimates from those counts.
Stops before spending more than --max-credits. Run it Thursday or Friday: books post most college
props a day or two before Saturday.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import oddsapi  # noqa: E402
from nflmodel.propmarket import DEFAULT_MARKETS  # noqa: E402

SPORT = "americanfootball_ncaaf"
OUT = Path(__file__).resolve().parent / "out" / "cfb_props_probe.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max-credits", type=int, default=120)
    a = ap.parse_args()
    c = oddsapi.Client()
    if not c.ready:
        sys.exit("ODDS_API_KEY is not set")
    now = datetime.now(timezone.utc)
    evs = [e for e in c.events(SPORT)
           if now < datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) < now + timedelta(days=a.days)]
    print(f"{len(evs)} college events in the next {a.days} days")
    spent, posted, full = 0, [], []
    for e in evs:
        if spent + 1 > a.max_credits:
            print("stopping: credit cap"); break
        before = c.remaining
        body = c.event_odds(e["id"], ["player_pass_yds"], sport=SPORT)
        spent += (before - c.remaining) if before is not None and c.remaining is not None else 1
        books = [b["key"] for b in body.get("bookmakers", []) if b.get("markets")]
        if books:
            posted.append({"game": f'{e["away_team"]} @ {e["home_team"]}', "kickoff": e["commence_time"], "books": books, "id": e["id"]})
    for g in posted[:2]:
        if spent + len(DEFAULT_MARKETS) > a.max_credits:
            break
        before = c.remaining
        body = c.event_odds(g["id"], DEFAULT_MARKETS, sport=SPORT)
        spent += (before - c.remaining) if before is not None and c.remaining is not None else len(DEFAULT_MARKETS)
        mk = sorted({m["key"] for b in body.get("bookmakers", []) for m in b.get("markets", [])})
        full.append({"game": g["game"], "markets": mk})
    n_m = round(sum(len(f["markets"]) for f in full) / len(full), 1) if full else None
    week = None if n_m is None else {"per_game_per_snapshot": n_m, "games": len(posted),
                                    "live_week_3_snapshots": round(len(posted) * n_m * 3)}
    res = {"at": now.isoformat(), "events": len(evs), "with_pass_yds_props": len(posted), "games": posted,
           "markets_seen": full, "credits_spent": spent, "estimate": week, "remaining": c.remaining}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "games"}, indent=1))


if __name__ == "__main__":
    main()
