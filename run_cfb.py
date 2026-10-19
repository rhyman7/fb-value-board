"""Build the college value board.

    python run_cfb.py                  download data, rate teams, backtest, build docs/cfb/
    python run_cfb.py --no-backtest    daily refresh
    python run_cfb.py --offline        no ESPN lines or wind forecasts (for testing)
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from cfbmodel import pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--no-refresh", action="store_true", help="use cached data only")
    ap.add_argument("--offline", action="store_true", help="skip ESPN lines and wind forecasts")
    a = ap.parse_args()
    pipeline.run(refresh=not a.no_refresh, with_backtest=not a.no_backtest, live=not a.offline)
