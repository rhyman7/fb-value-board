"""Fit the over-lean shade (nflmodel/propshade.py) and write it to data/prop_params.json.

    python experiments/props_shade_fit.py

Reads the committed output of experiments/props_over_lean_test.py (the "qb_rushing_out" version
at the close: every two-sided DraftKings/FanDuel main line of 2023-2025, quarterback rushing left
out) and takes, per shade group, the prop-weighted average of (over rate - books' fair over).
Re-run after that test is re-run with another season of lines (the January recheck); never from
a scheduled task. Only the "shade" key of the parameters file changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nflmodel import propshade  # noqa: E402

SRC = Path(__file__).resolve().parent / "out" / "props_over_lean_test.json"


def main():
    rows = json.loads(SRC.read_text())["qb_rushing_out"]["close"]["lean_by_stat"]
    acc = {}
    for r in rows:
        stat, pos = r["key"]
        g = propshade.group(stat, pos)
        if g is None:
            continue
        n, d = acc.get(g, (0, 0.0))
        acc[g] = (n + r["n"], d + r["n"] * (r["over_rate"] - r["fair_over"]))
    shade = {g: round(d / n, 4) for g, (n, d) in acc.items()}
    shade["fitted_on"] = "2023-2025 main lines at the close (props_over_lean_test.json)"
    shade["n"] = {g: n for g, (n, _) in acc.items()}
    params = json.loads(propshade.PARAMS_PATH.read_text())
    params["shade"] = shade
    propshade.PARAMS_PATH.write_text(json.dumps(params, indent=1))
    print(json.dumps(shade, indent=1))


if __name__ == "__main__":
    main()
