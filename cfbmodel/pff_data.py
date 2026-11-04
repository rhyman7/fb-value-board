"""Reader for PFF's college data, which lives in the private repo rhyman7/pff-data, never here.

This repo is public and its root is the live site, so PFF rows must never be committed or written
to docs/. Where the data is found, in order:

  1. $PFF_DATA_DIR
  2. .pff-data/ in this repo (where the Actions check it out with the PFF_DATA_TOKEN secret;
     git-ignored)
  3. ../pff-data next to this repo (a local clone)

    from cfbmodel import pff_data
    rec = pff_data.load("receiving_summary", [2024, 2025])     # every week, all divisions

Tables are the files' names in pff-data (see its spec/operations.md): receiving_summary,
rushing_summary, offense_summary, report_receiving, ... One row per player per week, with
`season` and `week` (0-16 regular season, 17 conference championships, 18 bowls).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def root() -> Path:
    for cand in (os.environ.get("PFF_DATA_DIR"), ROOT / ".pff-data", ROOT.parent / "pff-data"):
        if cand and (Path(cand) / "ncaa").is_dir():
            return Path(cand)
    raise FileNotFoundError("PFF data not found: set PFF_DATA_DIR, or check out rhyman7/pff-data into .pff-data/ "
                            "(Actions: the PFF_DATA_TOKEN secret) or ../pff-data")


def available() -> bool:
    try:
        root()
        return True
    except FileNotFoundError:
        return False


def load(table: str, seasons, weeks=None) -> pd.DataFrame:
    base = root() / "ncaa"
    parts = []
    for s in seasons:
        for f in sorted((base / str(s)).glob(f"w*_{table}.csv.gz")):
            if f.name[4:-len(".csv.gz")] != table:        # w05_<table>.csv.gz, exact table name
                continue
            if weeks is not None and int(f.name[1:3]) not in weeks:
                continue
            parts.append(pd.read_csv(f, low_memory=False))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
