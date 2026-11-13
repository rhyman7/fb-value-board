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

*Nov 4, 2026, under the keep rule (CONVENTIONS.md "Model tests"; FEATURES.md "Re-judging past
tests under the keep rule", "Applied"):* three more margin features and the unknown-quarterback
value −0.35 (below). `d_miss_defcov`, the share of the team's core coverage defenders (by box-score
production) who recorded no stat in the previous game; `d_dve`, drive value over expected, rated
like the other metrics; `d_mid`, a mid-season head-coaching change (`cfbmodel/extra.py`). Together,
on today's model: raw miss −0.022 ± 0.012 tuning / −0.017 ± 0.011 holdout, blended −0.009 ± 0.005 /
−0.007 ± 0.005; the holdout spread record (EV>0 sides, closing line) −0.6% → +0.1%.

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

*Nov 4, 2026, under the keep rule:* returning offensive production (`s_off_rtprod`; the offense-only
version beat overall head-to-head) and line yards per rush (`s_lyd`, rated) joined the total. Raw
total miss −0.003 ± 0.010 tuning / −0.005 ± 0.004 holdout; the blended total, 95% market, moved
+0.0001 ± 0.0007 / +0.0000 ± 0.0004 (noise). Ryan's call: applied on the raw measure.

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
(differences under 0.005 are noise). *Nov 4, 2026: −0.35 taken under the keep rule* (better on
QB-change and all games in both periods; on today's model raw −0.0017 / −0.0045, blended −0.0004 /
−0.0013). On QB-change games the model is now within 0.03
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

## Graded against Pinnacle: spreads and totals, 2024-2025 (Nov 1, 2026)

The 2024 and 2025 backfills now hold Pinnacle, DraftKings and FanDuel spreads and totals at a
Monday 12:00 UTC open and a close before each cluster of kickoffs. `python experiments/pinnacle_grade.py
--league cfb --seasons 2024 2025` (results in `experiments/out/pinnacle_grade_cfb.json`), 1,499
games with a Pinnacle Monday number; the model's side bet at Pinnacle's Monday line and price,
graded on the result and on closing line value against Pinnacle's close. Games after a short week
are left out (21).

| Model off Pinnacle's Monday number by | Bets | W-L | Return | CLV vs Pinnacle's close | Line moved toward the model |
|---|---|---|---|---|---|
| Spread, any | 1,499 | 735-732 | −3.5% ± 2.5% | −2.7% ± 0.2% | +0.21 pts |
| Spread 3+ | 537 | 270-255 | −0.9% ± 4.1% | −1.9% ± 0.4% | +0.36 pts |
| Spread 4+ | 368 | 191-170 | +1.8% ± 5.0% | −1.8% ± 0.4% | +0.38 pts |
| Total, any | 1,499 | 778-709 | −0.2% ± 2.5% | −3.5% ± 0.2% | +0.22 pts |
| Total 2+ | 718 | 400-309 | +7.5% ± 3.5% | −2.7% ± 0.3% | +0.40 pts |
| Total 3+ | 418 | 233-179 | +7.7% ± 4.6% | −1.7% ± 0.4% | +0.61 pts |
| Total 5+ | 125 | 64-57 | +1.1% ± 8.4% | −0.5% ± 0.8% | +0.82 pts |

**College spreads have no edge against Pinnacle.** The line drifts toward the model after
Monday, but by about a third of a point, short of Pinnacle's margin: closing line value is
negative at every gap and the record is flat. The model's college spread edge (CFB_FEATURES
above, the 0.52 haircut) was measured against consensus averages and one book's prices; against
the sharp number it is not there. **Totals are better but not proven**: 2+ and 3+ point gaps
won (+7.5% ± 3.5%, about two standard errors) and the line moves toward the model more the
bigger the gap, yet the price at Pinnacle's Monday number still did not beat its close. As with
the NFL, the edge, if any, is in the first number, which a Monday noon snapshot is already late for.

**Price tier** (a DraftKings or FanDuel price beating Pinnacle's no-vig number by 1%): 82 at the
open (44-38, CLV −1.8% ± 1.1%) and 62 at the close (30-32, CLV +2.5% ± 0.7%). Too few to say more.

The consensus close the backtests use sits 0.40 points from Pinnacle's close on spreads and 0.42
on totals, and differs on 33% of games.

## Graded against Pinnacle: moneylines, 2024-2025 (Nov 13, 2026)

Moneylines were added to the 2024 and 2025 backfills (h2h, 3,370 credits) and the grade re-run:
`python experiments/pinnacle_grade.py --league cfb --seasons 2024 2025`. On the college board a
moneyline is a bet only in the price tier (a DraftKings or FanDuel price beating Pinnacle's no-vig
number by 1%; never on the model's opinion, `sheet.annotate`), so that is what is graded.

| Taken at | Bets | W-L | Return | EV claimed | CLV vs Pinnacle's close |
|---|---|---|---|---|---|
| Monday open | 138 | 91-47 | −0.6% ± 6.5% | +1.8% | −0.3% ± 0.5% |
| Close | 70 | 38-32 | −12.9% ± 10.7% | +1.9% | (the close itself) |

**College moneyline prices that beat Pinnacle's don't hold.** At the open they beat Pinnacle's close
about half the time (51%) and lost the claimed edge by the close; at the close they lost. Nearly all are
favourites (131 of 138 at the open, median −193). By season at the open: 2024 −6.9% on 97 (closing line
value +0.1%), 2025 +14.4% on 41 (−1.1%). Every price-tier bet together (spreads, totals and moneylines)
at the open: 220 bets, +0.6% ± 5.9%, closing line value −0.9% ± 0.6%. Against the live record (the
price tier by book, `claude/near-misses-and-rechecks.md`) this is one more reason the January recheck
should look at college price bets by market.

The same run, on today's model, moved the opener-gap totals slightly: total 3+ 404 bets +11.5% ±
4.7%, line moved +0.66 pts toward the model, closing line value against Pinnacle still −1.4% ± 0.4%.
Spreads unchanged (no edge).

## Drive ratings, FEI / SP+ style (tested Nov 2, 2026: not kept; dve applied Nov 4 under the keep rule)

FEI rates college teams per drive (points against what the starting spot was worth) and SP+ has
a "finishing drives" factor; the board rates per play and per game. Two new rated metrics from
the same play-by-play, rated like the others (opponent-adjusted ridge, recency decay, last season
at 0.55), weighted by drives or trips: **dve**, points per drive minus the 2014-2016 average for
that starting spot, and **fin**, points per trip that reached the opponent's 40. Garbage-time
drives, drives starting with under a minute left in a half, and drives that ran out the half
without a score are dropped. About 10 drives and 5 trips per team-game.
`python experiments/cfb_drive_test.py` (results in `experiments/out/cfb_drive_test.json`).

Rule, set before running: ridge chosen on 2017-2019 from a three-value grid; kept only if the
average miss drops in tuning and holdout, the holdout by at least 2 paired standard errors.

| Added | Tuning (2017-19) | Holdout (2021-25) |
|---|---|---|
| Margin + drive value | −0.005 ± 0.009 | −0.004 ± 0.007 |
| Margin + finishing | +0.003 ± 0.005 | 0.000 ± 0.003 |
| Margin + both | −0.005 ± 0.009 | −0.005 ± 0.007 |
| Total + drive value | −0.006 ± 0.013 | +0.002 ± 0.010 |
| Total + finishing | +0.015 ± 0.011 | −0.001 ± 0.004 |
| Margin, points rating swapped for drive value | +0.075 ± 0.030 | +0.032 ± 0.024 |

**Nothing here.** Drive value moves the margin model by under a hundredth of a point, well inside
its noise (0.6 standard errors on the holdout); finishing drives does nothing. The per-play
ratings plus the points rating already carry what drives measure, and drive value is a worse
stand-in for the points rating, not a better one. Too small to count as a near-miss.

## Down splits, line yards and havoc (tested Nov 2, 2026: not kept; line yards in the total from Nov 4 under the keep rule)

CollegeFootballData's adjusted metrics and SP+ split success rate by standard and passing downs,
rate the line by line yards per rush, and track havoc (tackles for loss, sacks, interceptions,
forced fumbles, pass breakups per play). From cfbfastR's own columns, over the board's
non-garbage scrimmage plays, four new rated metrics: standard-down success, passing-down success,
line yards per rush (each run clipped to −6..7) and havoc allowed per play, each rated like the
others and weighted by its plays. `python experiments/cfb_downs_havoc_test.py` (results in
`experiments/out/cfb_downs_havoc_test.json`). Rule, set before running: ridge chosen on 2017-2019
from a three-value grid; kept only if the average miss drops in tuning and holdout, the holdout by
at least 2 paired standard errors.

| Added | Margin, tuning | Margin, holdout | Total, tuning | Total, holdout |
|---|---|---|---|---|
| Standard-down success | +0.009 ± 0.004 | −0.003 ± 0.002 | −0.002 ± 0.003 | +0.002 ± 0.002 |
| Passing-down success | +0.006 ± 0.005 | 0.000 ± 0.001 | 0.000 ± 0.002 | +0.002 ± 0.001 |
| Line yards per rush | +0.004 ± 0.014 | −0.018 ± 0.012 | −0.001 ± 0.001 | −0.001 ± 0.002 |
| Havoc allowed | +0.004 ± 0.005 | +0.006 ± 0.006 | −0.003 ± 0.006 | +0.003 ± 0.005 |
| All four | +0.017 ± 0.015 | −0.013 ± 0.012 | −0.006 ± 0.007 | +0.002 ± 0.005 |

**Nothing here.** Every one is worse on 2017-2019 for spreads, and nothing reaches 2 standard
errors on the holdout. Line yards was the closest (1.6 standard errors on the holdout, worse on
tuning), too weak for the near-miss list. The overall success rate and the pass and rush ratings
already carry what these splits measure.

## Returning production from PFF that counts incoming transfers (Nov 5, 2026): mixed, Ryan's call

`experiments/cfb_pff_returning_test.py` (rule committed before the first run). For each team-season,
the players on this season's CFBD roster are matched by name to last season's PFF college players on
any team (98,348 matches since 2018, 15,077 of them transfers in), and valued at last season's PFF
snaps × (snap-weighted grade − 60), clipped at 0. Offense and defense summed separately. Margin:
home minus away (`d_pffret`); total: both teams' sums, centred (`s_pffret_off`, `s_pffret_def`).
The first run had a numerical bug in the total features (a filled constant kept a rounding-size
spread that the ridge's standardisation blew up in 2018, +1.11 points); fixed and every version
re-run, disclosed in the script. Raw model, average miss (weeks 0-5 in brackets, reported only):

| Version | Tuning 2017-19 margin / total | Holdout 2021+ margin / total |
|---|---|---|
| Current model | 12.359 / 13.452 (12.420 / 13.169) | 12.334 / 12.619 (12.572 / 12.575) |
| **A: margin + d_pffret, total + s_pffret_off/def (chosen)** | **12.354 / 13.447** (12.394 / 13.168) | **12.313** / 12.630 (12.522 / 12.587) |
| B: margin + d_pffret × early only | 12.366 / 13.452 | 12.309 / 12.619 |
| C: A + d_pffret × early | 12.375 / 13.447 | 12.305 / 12.630 |
| D: A replacing CFBD returning production | 12.492 / 13.488 | 12.323 / 12.631 |

A against the current model (paired): tuning margin −0.005 (SE 0.010), total −0.005 (SE 0.012);
holdout margin **−0.022 (SE 0.009)**, total **+0.011 (SE 0.008)**. Margin better in 2019 and in four
of the five holdout seasons through 2025 (2026 to date −0.067); weeks 0-5 on the holdout −0.050. The
feature only exists from 2018, so on the tuning seasons only 2019 could move. Mixed (holdout total
worse), so not kept by the rule: it goes to Ryan. The margin and total models are fitted separately,
so A's margin half on its own is exactly A's margin numbers with the totals unchanged.

*Applied Nov 6, 2026 (Ryan's choice: the margin half only):* `cfbmodel/pff_returning.py` adds
`d_pffret` to the margin model; the totals are unchanged. The production code reproduces the test
(margin 12.354 / 12.313, totals 13.452 / 12.619). Update college board checks out PFF
`ncaa/**/*_defense_summary.csv.gz` too; without the PFF data `d_pffret` is 0 and the board runs as
before. Shown in the spread card under "Talent and returning production".

## Fixes from the Nov 11, 2026 review, tested (Nov 11, 2026): none kept

`experiments/review_fix_tests.py cfb` (rules written first; run with the PFF returning production).
College 2026 is what prompted both: the model misses by 12.38 against the line's 11.55 through week
10, where the usual gap is 0.2 to 0.3.

**T3. A first-FBS-season feature (`d_newfbs`): not kept.** Eleven team-seasons since 2015 (Charlotte
2015 to Sacramento State and North Dakota State 2026). Raw margin miss +0.048 (SE 0.022) on 2017-19
and +0.005 (SE 0.007) on 2021-25: worse on both. In the 84 earlier games with a first-year team the
feature made the miss 1.5 points worse (12.69 to 14.22): earlier newcomers played about as the model
already said. On 2026's 15 such games it helped by 0.37 (SE 0.48), which is one season and not in
the rule.

**T4. The blend weight sees this season's games: not kept.** Market-adjusted margin miss +0.023
(SE 0.015) on 2017-19 and 0.000 on 2021-25. In 2026 it would have walked the weight from 0.30 down to
0.24 by week 10 and lowered the miss by 0.014 (SE 0.007), again one season outside the rule.

## PFF unit ratings vs production, college (Nov 11, 2026): descriptive check

The college run of FEATURES.md "PFF unit ratings vs production" (`experiments/pff_unit_check/cfb.py`,
`cfb_split.py`; same definitions; points only, no play-by-play measures). Regular season 2017-2025,
8,892 FBS team-games with both teams' PFF units and three earlier games; 8,058 of 8,447 games matched
to PFF. Bowls did not match on week numbers and are not covered.

The continuous unit drop is unusable in college: its sign is backwards and large (own OL −2.15
points per grade point, t −9.1; against the spread −2.72, t −9.8), because blowouts put the backups
on the field, so the lineup follows the score. Only the count of regulars with zero snaps is read.

| Regulars missing | Games | Points vs expected | Points vs closing team total | Cover |
|---|---|---|---|---|
| Own OL, 2+ | 364 | +0.52 ± 0.66 | +0.42 ± 0.56 | 47.9% |
| Own WR/TE, 1 | 1,774 | −0.57 ± 0.31 | −0.11 ± 0.27 | 49.3% |
| Own WR/TE, 2+ | 225 | −0.84 ± 0.87 | −0.60 ± 0.76 | 48.9% |
| Opposing DL, 2+ | 86 | +1.41 ± 1.40 | +3.77 ± 1.26 | 68.6% |
| Opposing LB, 1 | 811 | −0.34 ± 0.46 | +0.59 ± 0.40 | 51.2% |
| Opposing DB, 1 | 1,648 | −0.04 ± 0.33 | +0.94 ± 0.28 | 54.5% |
| Opposing DB, 2+ | 184 | −0.25 ± 0.97 | +1.24 ± 0.84 | 51.6% |

(With no regular missing the baseline against the team total is about +0.3 to +0.6.) Nothing reaches
2 standard errors on points against the team's own expectation. One market lead: the team facing a
secondary that is missing a regular covered 989 of 1,823 (54.3%) in 2017-2025: 55.9% in 2017-21,
52.1% in 2022-25, and 66 of 109 in 2026. It does not come with more points than the offense's own
trend, so it may be marking teams that are coming apart (mid-season opt-outs and departures) more
than a coverage effect. It uses who actually played, which college books and bettors mostly could
not know before 2024's availability reports. A lead for a bet-time test on the official reports
(`docs/cfb/availability.json`, 2026 only so far), not a change.
