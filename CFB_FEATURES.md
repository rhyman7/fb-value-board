# College feature tests, October 2026

The college board uses the NFL board's method (see [FEATURES.md](FEATURES.md)) with
college data from cfbfastR. Each candidate was added to the model on its own and
scored walk-forward. A feature or setting earned a place only if it lowered the raw
model's average miss on the **tuning seasons (2017-2019, 2,206 FBS-vs-FBS games with
lines)**. The **holdout (2021 to week 7 of 2026, 4,143 games)** is the check, not the
selector. The shortened 2020 season is scored but is in neither set. 2015-2016 are
training only (the first season with lines is 2015, and the model needs two seasons
to fit on).

Reproduce with `python experiments/cfb_feature_test.py`, `python experiments/cfb_qb_test.py`
and `python experiments/cfb_backtest.py`.

## Rating settings

The NFL settings, then a grid on 2017-2019 (raw model margin miss, tuning / holdout):

| Setting | Tried | Chosen | Effect |
|---|---|---|---|
| Rating shrinkage (ridge) | NFL × 0.02 to 2 | **NFL × 0.05** | 12.61 at the NFL values → 12.45 (with the 8-week market half-life). College ratings need far less shrinkage: 135 teams with much wider gaps. |
| Market-implied ratings half-life | 2, 4, 8, 12 weeks | **8 weeks** (NFL 4) | −0.01 to −0.02 against 4 |
| Recency half-life | 5, 7, 9, 12 games | 9 (as NFL) | flat (within 0.01) |
| Prior-season carry | 0.4, 0.55, 0.7 | 0.55 (as NFL) | flat |
| Game-model ridge, margin / total | 2 to 5,000 | **150 / 600** (NFL 20) | −0.03 / −0.03 |

## Margin (spreads and moneylines)

Raw model with the NFL's feature set: 12.758 (tuning), 12.542 (holdout).
Current model: **12.374 / 12.354**. Closing line: 12.221 / 12.055.

Change in average miss; for kept features, the change from having them (negative is better):

| Candidate | What it measures | Tuning | Holdout | Kept |
|---|---|---|---|---|
| **Returning production** | share of last season's production back (difference) | **−0.148** | **−0.034** | **yes** |
| **Talent × early season** | 247 team talent gap, weighted to weeks 0-5 | **−0.071** | **−0.097** | **yes** |
| **QB gap × early season** | the quarterback gap, reweighted in weeks 0-5 | **−0.039** | **−0.010** | **yes** |
| **Talent** | 247 team talent gap, all season | **−0.022** | +0.015 | **yes** |
| Returning production × early | | +0.007 | +0.003 | no |
| Offensive / defensive returning production | split instead of overall | +0.019 / +0.032 | −0.000 / +0.001 | no |
| Returning starting QB | | −0.000 | −0.001 | no |
| Market-implied ratings × early | | +0.002 | −0.003 | no |
| Points, success rate × early | | +0.002 / +0.000 | −0.002 / −0.001 | no |
| Tempo (snaps per game) | | +0.003 | −0.001 | no |
| Travel distance | away team's trip | +0.011 | −0.011 | no |
| Elevation | home stadium | +0.011 | −0.003 | no |
| Conference game × home field | | −0.000 | −0.001 | no |

The college-only preseason numbers carry most of the improvement, as expected: early in
the season the play-by-play has seen two or three games and last season's ratings are
stale. Talent all season helps on the tuning seasons and hurts slightly since, but is
kept by the rule.

## Totals

Raw model with the NFL's feature set: 13.508 (tuning), 12.648 (holdout). Current model:
**13.45 / 12.60**. Closing total: 13.174 / 12.433.

| Candidate | Tuning | Holdout | Kept |
|---|---|---|---|
| **Returning defensive production (sum)** | **−0.044** | +0.002 | **yes** |
| **Tempo (snaps per game)** | **−0.011** | **−0.028** | **yes** |
| **Wind above 10 mph** | **−0.010** | **−0.022** | **yes** |
| Wind (all) | +0.001 | −0.014 | no |
| Returning offensive / overall production | −0.002 / −0.002 | −0.005 / −0.003 | no |
| Talent | +0.003 | +0.002 | no |
| Returning starting QB | +0.001 | −0.000 | no |
| Cold (below 45°F) | +0.005 | −0.002 | no |
| Elevation | +0.009 | +0.001 | no |

As with the NFL board, the totals weight in the market blend is capped at 5%.

## Quarterback changes

Games where either team started a different quarterback from its previous game (573 in
2017-2019, 987 since 2021). College backups are much worse than NFL backups, so the
NFL setting undervalued a switch.

| Setting | Tuning, QB-change games | Holdout, QB-change games | Tuning, all | Holdout, all |
|---|---|---|---|---|
| NFL-style (unknown QB −0.10, prior 200 dropbacks) | 12.330 | 12.815 | 12.420 | 12.395 |
| Unknown QB at −0.20 | 12.244 | 12.687 | 12.381 | 12.368 |
| **Current: unknown QB −0.30, prior 800** | **12.235** | **12.641** | **12.374** | **12.355** |
| Unknown QB at −0.35 | 12.233 | 12.629 | 12.372 | 12.351 |
| QB half-life 700 dropbacks | 12.233 | 12.579 | 12.382 | 12.332 |
| QB half-life 2,800 | 12.237 | 12.681 | 12.372 | 12.369 |
| QB prior 400 / 1,600 | 12.254 / 12.222 | 12.633 / 12.652 | 12.384 / 12.368 | 12.350 / 12.360 |
| No quarterback features | 12.459 | 13.064 | 12.429 | 12.441 |
| Closing line | 12.140 | 12.608 | 12.221 | 12.056 |

−0.35 and the 700 half-life were barely better or worse on tuning and were not taken
(differences under 0.005 are noise). On QB-change games the model is now within 0.03
points of the closing line since 2021.

## Data fix found during testing

The lines file and the schedule disagree on which team is home for a few neutral-site
games (Army-Navy among them). Their spreads were being applied to the wrong side; the
build now flips the lines file's numbers onto the schedule's sides.

## Missing players

There is no public college injury report and no game-day inactive list, so absences can
only be read from box scores. For each team and game, the regulars by usage over its
earlier games this season (20%+ of the carries, quarterbacks excluded; 12%+ of the
catches; two or more appearances) are checked against the team's previous game. A regular
with no stat there is presumed still out, which is all a bettor can know before kickoff.
Run `python experiments/cfb_missing_test.py`.

| Added to the current model | Tuning | Holdout | Kept |
|---|---|---|---|
| Running backs missing from the last game (difference) | +0.001 | +0.000 | no |
| Running backs, change from the three games before | −0.005 | +0.000 | no |
| Pass catchers missing from the last game | −0.000 | +0.004 | no |
| Pass catchers, change | +0.001 | +0.002 | no |
| Both groups | +0.000 | +0.003 | no |
| Totals: running backs / pass catchers missing (sum) | −0.003 / −0.001 | +0.005 / +0.002 | no |
| *Ceiling: regulars missing from this game's own box score* | +0.005 / −0.012 (totals) | −0.000 / +0.006 | *not usable* |

**Nothing helps**, not even the ceiling row, which uses who actually played (known only
after the game). Skill-position absences that show up in box scores are already in the
market-implied ratings and the quarterback adjustment. Offensive line and defensive
absences cannot be seen at all: box scores do not list linemen, and defensive stats are
only complete from 2023.

## Not tested yet

- The wind feature graded on forecasts instead of recorded wind (needs a season of the
  logged forecasts in `data/cfb/wind_log.csv`).
- Missing players again once a season or two of game rosters with listed starters exist
  (ESPN's college game rosters only mark starters from late 2025).
