# PFF unit ratings vs production (descriptive check, Nov 11, 2026)

Does production fall when a unit's collective PFF rating falls, or when its regulars are missing?
Not a model test; nothing here feeds the boards. Results: FEATURES.md, "PFF unit ratings vs
production", and CFB_FEATURES.md.

Run from anywhere, with a scratch directory `D` that holds nflverse `games.csv` and
`pbp_<season>.parquet` (2015 on), and the private PFF checkout:

    python build.py ../pff-data D      # player-games and ratings (NFL)
    python build2.py D                 # unit lineup rating vs the usual lineup, regulars missing
    python analyze.py D                # production and market, 2015-2025 and 2026
    python split.py D                  # notable buckets by period
    python cfb.py ../pff-data D        # college (needs D/cfb_games.parquet from cfbmodel.data.reduce_games)
    python cfb_split.py D

No PFF rows are written to this repo; the scratch directory stays out of it.
