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

## Where the edge lives

The walk-forward backtest split by segment, 2021 on (`python experiments/cfb_segments.py`,
full output with the 2017-2019 tuning seasons in `experiments/out/cfb_segments.md`). Miss is
the average absolute error of the raw model and of the closing line. The records bet the raw
model's side wherever it is 2+ or 3+ points off the spread or 3+ off the total, at −110 at
the line rounded to the half point; opener records use a run that assumes last week's
quarterbacks and a typical wind and leave out short-week games, as the board's opener report
does. P4 is the SEC, Big Ten, Big 12, ACC and Notre Dame, plus the Pac-12 through 2023; days
are Eastern.

**Spreads** (W-L, win rate, return per bet ± s.e., at -110)

| Segment | Games | Miss: model / close | vs close, 2+ | vs close, 3+ | vs opener, 2+ | vs opener, 3+ |
|---|---|---|---|---|---|---|
| Conference tier: P4 vs P4 | 1841 | 12.20 / 11.95 | 519-490 51.4% -1.8% ± 2.9% | 338-341 49.8% -4.9% ± 3.6% | 489-469 51.0% -2.5% ± 3.0% | 327-311 51.3% -2.1% ± 3.7% |
| Mixed (P4 vs G5) | 586 | 12.31 / 11.64 | 183-169 52.0% -0.7% ± 4.9% | 143-133 51.8% -1.1% ± 5.6% | 199-175 53.2% +1.6% ± 4.9% | 156-130 54.5% +4.1% ± 5.6% |
| G5 vs G5 | 1776 | 12.52 / 12.27 | 549-499 52.4% +0.0% ± 2.9% | 411-352 53.9% +2.8% ± 3.4% | 503-471 51.6% -1.4% ± 3.0% | 354-334 51.5% -1.7% ± 3.6% |
| Day of week: Saturday | 3586 | 12.49 / 12.17 | 1068-1001 51.6% -1.4% ± 2.1% | 761-720 51.4% -1.9% ± 2.4% | 1030-989 51.0% -2.6% ± 2.1% | 732-687 51.6% -1.5% ± 2.5% |
| Midweek (Tue-Fri) | 594 | 11.46 / 11.27 | 178-148 54.6% +4.1% ± 5.1% | 127-99 56.2% +7.0% ± 6.1% | 156-118 56.9% +8.5% ± 5.6% | 100-83 54.6% +4.2% ± 6.9% |
| Sunday/Monday | 23 | 13.56 / 12.52 | 5-9 35.7% -29.7% ± 23.7% | 4-7 36.4% -28.0% ± 26.6% | 5-8 38.5% -26.6% ± 26.8% | 5-5 50.0% -4.5% ± 31.8% |
| Week of season: Weeks 0-5 | 1568 | 12.58 / 11.88 | 476-494 49.1% -6.1% ± 3.0% | 355-380 48.3% -7.6% ± 3.4% | 470-498 48.6% -7.2% ± 3.0% | 368-360 50.5% -3.5% ± 3.5% |
| Weeks 6-10 | 1519 | 12.08 / 11.97 | 447-408 52.3% -0.2% ± 3.2% | 311-268 53.7% +2.5% ± 3.9% | 421-375 52.9% +1.0% ± 3.3% | 281-262 51.7% -1.2% ± 4.0% |
| Weeks 11+ | 1091 | 12.38 / 12.36 | 322-251 56.2% +7.2% ± 3.9% | 220-174 55.8% +6.5% ± 4.7% | 292-237 55.2% +5.2% ± 4.0% | 183-149 55.1% +5.1% ± 5.1% |
| Bowls | 25 | 12.87 / 12.90 | 6-5 54.5% +4.1% ± 30.1% | 6-4 60.0% +14.5% ± 31.2% | 8-5 61.5% +17.5% ± 26.8% | 5-4 55.6% +6.1% ± 33.5% |
| **All games** | **4203** | **12.35 / 12.04** | **1251-1158 51.9% -0.8% ± 1.9%** | **892-826 51.9% -0.9% ± 2.3%** | **1191-1115 51.6% -1.4% ± 2.0%** | **837-775 51.9% -0.9% ± 2.3%** |

**Totals** (W-L, win rate, return per bet ± s.e., at -110)

| Segment | Games | Miss: model / close | vs close, 3+ | vs opener, 3+ |
|---|---|---|---|---|
| Conference tier: P4 vs P4 | 1841 | 12.26 / 12.11 | 322-295 52.2% -0.4% ± 3.8% | 283-227 55.5% +5.9% ± 4.2% |
| Mixed (P4 vs G5) | 586 | 12.32 / 11.94 | 109-107 50.5% -3.6% ± 6.4% | 110-101 52.1% -0.5% ± 6.5% |
| G5 vs G5 | 1776 | 13.12 / 13.01 | 355-295 54.6% +4.2% ± 3.7% | 313-242 56.4% +7.6% ± 4.0% |
| Day of week: Saturday | 3586 | 12.63 / 12.49 | 681-592 53.5% +2.1% ± 2.6% | 620-488 56.0% +6.8% ± 2.8% |
| Midweek (Tue-Fri) | 594 | 12.69 / 12.42 | 99-99 50.0% -4.5% ± 6.7% | 82-77 51.6% -1.5% ± 7.6% |
| Sunday/Monday | 23 | 11.27 / 10.94 | 6-6 50.0% -4.5% ± 28.8% | 4-5 44.4% -15.2% ± 33.5% |
| Week of season: Weeks 0-5 | 1568 | 12.57 / 12.40 | 331-279 54.3% +3.5% ± 3.8% | 304-248 55.1% +5.1% ± 4.0% |
| Weeks 6-10 | 1519 | 12.88 / 12.70 | 251-253 49.8% -4.9% ± 4.2% | 214-194 52.5% +0.1% ± 4.7% |
| Weeks 11+ | 1091 | 12.37 / 12.25 | 195-160 54.9% +4.8% ± 5.0% | 181-123 59.5% +13.6% ± 5.4% |
| Bowls | 25 | 12.48 / 12.53 | 9-5 64.3% +22.7% ± 25.4% | 7-5 58.3% +11.4% ± 28.4% |
| **All games** | **4203** | **12.63 / 12.47** | **786-697 53.0% +1.2% ± 2.4%** | **706-570 55.3% +5.6% ± 2.6%** |

Reading: the only edge that holds up is the one the README reports, **totals against the
opener**, and it comes from the plain segments: Saturday games (56.0%, +6.8% ± 2.8%),
G5-vs-G5 (56.4%) and P4-vs-P4 (55.5%) games, and the start and end of the season (weeks
0-5 at 55.1%, weeks 11+ at 59.5%, +13.6% ± 5.4%); it is absent in midweek games (51.6% on 159
bets) and in P4-vs-G5 games, and weeks 6-10 are flat. Against the closing total the same
rule is +1.2% ± 2.4%, so most of that edge is the line moving the model's way, and it could
not be checked on 2017-2019, which have no openers (there the model lost against closing
totals, 50.1%). On **spreads** there is no edge overall at either line (51.9%, −0.9% ± 2.3%
at 3+); midweek games (54.6% to 56.9%, about 180 to 330 bets) and weeks 11+ (55% to 56%,
+5% to +7% ± 4% to 5%) are above water while weeks 0-5 lose (48% to 49%), but the 2017-2019
splits point the other way (weeks 0-5 won 57%, midweek 50%), so these look like noise rather
than structure. Every segment's standard error is 2.5 to 7 points of return, so no two
segments differ significantly, and the Sunday/Monday and bowl rows (12 to 25 games) say
nothing.

## Graded against Pinnacle (2025 totals, Oct 31, 2026)

The college record is graded against cfbfastR's consensus lines (and live, ESPN's one book).
`experiments/pinnacle_grade.py --league cfb --seasons 2025` regrades the one college edge that
has held, **totals 3+ off the opener**, against Pinnacle's real numbers: the backfilled
Pinnacle, DraftKings and FanDuel totals in `data/market/history/cfb/2025/` (totals only, to
fit the month's credits; spreads and 2024 are for a later month). The open is Monday 12:00 UTC;
a college "close" is 10 minutes to two hours before kickoff (one snapshot covers every kickoff
within two hours). 796 games matched; results in `experiments/out/pinnacle_grade_cfb.json`.

| Model total off the opener by | Bets at Pinnacle's Monday total and price | W-L-P | Return | Line moved toward the model | CLV (EV) vs Pinnacle's close | Same games, consensus opener at −110 |
|---|---|---|---|---|---|---|
| 2+ | 373 | 201-167-5 | +4.0% ± 4.9% | +0.42 pts | −2.7% ± 0.4% | +1.7% ± 5.1% (+0.63 pts) |
| 3+ | 216 | 118-95-3 | +5.6% ± 6.4% | +0.69 pts | −1.4% ± 0.6% | +10.7% ± 6.5% (+0.85 pts) |
| 5+ | 69 | 37-30-2 | +5.4% ± 11.4% | +1.09 pts | +1.0% ± 1.1% | +9.1% ± 12.0% (+1.66 pts) |

The direction holds in 2025 (the total moves toward the model, more the bigger the gap) and the
record is positive, but against Pinnacle's Monday number it is not significant and the move is
smaller than against the consensus opener: by Monday noon Pinnacle has already taken part of
it. The edge is in betting the first number, which is what the live market log and the opener
alerts are for; a Monday price is already late. The consensus close sits 0.43 points from
Pinnacle's close on totals.

**Price tier** (a DraftKings or FanDuel total beating Pinnacle's no-vig number by 1%): 25 at the
Monday open (13-12, CLV −2.6% ± 1.4%, 32% beat the close) and 17 at the close. Too few, and at
the open they did not hold their value: college retail totals that look better than Pinnacle's
early in the week mostly mean Pinnacle moves first.

## Not tested yet

- The wind feature graded on forecasts instead of recorded wind (needs a season of the
  logged forecasts in `data/cfb/wind_log.csv`).
- Missing players again once a season or two of game rosters with listed starters exist
  (ESPN's college game rosters only mark starters from late 2025).
- The derivative curves (team totals and first-half lines, `nflmodel/derivs.py`) are tested
  on both boards in FEATURES.md, "Derivative markets": the college half curve matches the
  key numbers at the half, and college overs on team totals and first-half totals hit 2-3
  points less often than the curve says at the high-chance lines. Grade them against
  Pinnacle's closing derivative lines once the market log has a season of them.
