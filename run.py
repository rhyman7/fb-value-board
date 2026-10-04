#!/usr/bin/env python3
"""Command line entry point.

    python run.py                 refresh data, re-rate, backtest, rebuild the dashboard
    python run.py --no-backtest   skip the backtest (reuses the last one); much faster
    python run.py --offline       use the data already on disk
    python run.py --page          just rebuild docs/index.html from docs/data.json
"""
import argparse

from nflmodel import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="NFL model: ratings, projections, EV against the market.")
    ap.add_argument("--no-backtest", action="store_true", help="skip the walk-forward backtest")
    ap.add_argument("--offline", action="store_true", help="do not download; use cached data")
    ap.add_argument("--page", action="store_true", help="only rebuild the dashboard page")
    args = ap.parse_args()
    if args.page:
        pipeline.build_page()
        return
    out = pipeline.run(refresh=not args.offline, with_backtest=not args.no_backtest)
    games = out["slate"]["games"]
    if not games:
        print("No upcoming games with lines.")
        return
    wk = out["slate"]["weeks"][0]
    print(f"\nWeek {wk}: top bets by market-adjusted EV")
    rows = sorted((b | {"game": f"{g['away']} @ {g['home']}"} for g in games if g["week"] == wk for b in g["bets"]),
                  key=lambda b: -b["ev"])[:10]
    print(f"{'Game':<12}{'Pick':<14}{'Odds':>6}{'Model':>8}{'Market':>8}{'EV':>8}{'Raw EV':>8}")
    for b in rows:
        print(f"{b['game']:<12}{b['pick']:<14}{b['odds']:>+6d}{b['prob']:>8.1%}{b['market_prob']:>8.1%}{b['ev']:>+8.1%}{b['raw_ev']:>+8.1%}")


if __name__ == "__main__":
    main()
