# Feature tests, October 2026

Which stats beyond the base model help find bets? Each candidate was added to
the model on its own and scored walk-forward. A feature earned a place only if
it lowered the raw model's average miss on the **tuning seasons (2011-2018,
out of sample)**. The **holdout (2019 to week 6 of 2026, 2,052 games)** is the
check, not the selector. Reproduce with `python experiments/feature_test.py`.

## Margin (spreads and moneylines)

Base model average miss: 10.251 points (tuning), 10.013 (holdout). Change when
each stat is added; negative is better.

| Candidate | What it measures | Tuning | Holdout | Kept |
|---|---|---|---|---|
| Sack rate | pass protection against pass rush | +0.005 | +0.002 | no |
| Explosive play rate | passes of 20+ yards, runs of 10+ | +0.002 | −0.001 | no |
| Early-down pass EPA | 1st and 2nd down dropbacks | +0.001 | −0.006 | no |
| Special teams EPA | kicks, punts, returns, net | +0.001 | +0.000 | no |
| Fumble-recovery luck | lost fumbles against half of all fumbles | +0.005 | +0.002 | no |
| QB CPOE gap | starter's completion % over expected vs the team's usual | +0.003 | +0.002 | no |
| Pace (seconds per snap) | neutral-situation tempo | −0.001 | +0.001 | no |
| Snaps per game | volume | +0.003 | +0.003 | no |
| Travel distance | away team's trip | +0.005 | +0.005 | no |
| Time zones crossed east | body clock | +0.010 | −0.000 | no |
| West Coast team, early kickoff | the classic 1 p.m. spot | +0.030 | +0.008 | no |
| Divisional game × home field | familiarity shrinking home edge | +0.000 | +0.003 | no |

**None helps.** Every change is a few hundredths of a point either way, which
is noise. The base model already carries what these stats say (EPA and success
rate absorb most of them; quarterback swaps and earlier lines carry the rest),
and the market prices the situational spots. The spread model is unchanged.

## Totals

Base model average miss: 10.629 (tuning), 10.522 (holdout).

| Candidate | Tuning | Holdout | Kept |
|---|---|---|---|
| **Wind (mph, outdoor games)** | **−0.023** | **−0.043** | **yes** |
| **Snaps per game (pace)** | **−0.011** | **−0.006** | **yes** |
| Wind above 10 mph only | −0.001 | −0.019 | no (wind covers it) |
| Seconds per snap | +0.004 | −0.005 | no |
| Explosive play rate | +0.015 | −0.006 | no |
| Sack rate | +0.002 | +0.002 | no |
| Early-down pass EPA | +0.001 | −0.003 | no |
| Fumble luck | +0.002 | +0.000 | no |
| Cold (below 45°F) | +0.003 | −0.001 | no |

With wind and snaps the totals model's holdout miss falls from 10.52 to 10.48
points. The closing total is still 10.39, so the model does not beat the market
on totals.

### Totals as bets

Totals the board flagged as positive EV lost money in every version tested,
before and after these features:

| Total features | EV-flagged totals, 2014-2026 | Return |
|---|---|---|
| Base | 61-76 | −12.4% |
| + snaps | 129-146 | −7.9% |
| + wind | 249-281 | −8.0% |
| + wind + snaps | 332-353 | −5.3% |

The new features made the projection better but also raised the weight the
board put on the model for totals (to 20% for 2026), which would have flagged
more of these losing bets. So the totals weight is now capped at 5%
(`TOTAL_WEIGHT_CAP`): totals stay close to the market.

### The wind under

Closing totals do not fully account for wind. Unders at the closing total in
outdoor games, by recorded wind:

| Wind | Unders | Win rate | Return | Close minus final |
|---|---|---|---|---|
| 0-9 mph | 884-945-21 | 48.3% | −5.9% | −0.9 |
| 10-14 mph | 358-276-5 | 56.5% | +9.8% | +0.9 |
| 15-19 mph | 140-103-4 | 57.6% | +11.8% | +2.2 |
| 20+ mph | 41-37-1 | 52.6% | +2.0% | +0.2 |

| 10-19 mph, by era | Unders | Win rate | Return (± 1 s.e.) |
|---|---|---|---|
| 2006-2013 | 96-92-3 | 51.1% | +0.1% ± 7.1% |
| 2014-2018 | 164-123-2 | 57.1% | +11.6% ± 5.7% |
| 2019 on | 238-164-4 | 59.2% | +14.2% ± 4.7% |

The effect is consistent since 2014 and about three standard errors above zero
since 2019. Two cautions:

1. The 10-19 mph band was chosen after looking at every season, so the record
   is optimistic. Track it going forward before staking much on it.
2. The backtest uses the wind recorded at the game. A live bet uses a forecast
   made earlier, which is less accurate, so real results will be weaker.

The board now shows a wind tag on games forecast at 10-19 mph, the forecast in
each game's header, and this table in the model record.

### Wind direction and gusts (set up Oct 29, 2026, untested)

The model sees only the wind's speed. A 15 mph wind blowing goal line to goal
line should hurt passing and kicking far more than the same wind across the
field, and a gusty 12 mph may play like 20. Nothing is tested yet; this is the
set-up:

- Both boards' forecast calls now also fetch the gusts and the direction the
  wind blows from (same Open-Meteo call) and log them next to the speed
  (`data/wind_log.csv`, `data/cfb/wind_log.csv`: `gust_mph`, `dir_deg`). The
  NFL log also has `along_mph` and `cross_mph`, the wind split along and across
  the field from `nflmodel/stadiums.py`, which holds each stadium's field
  bearing with a `measured` or `approx` tag (most are approx: refine them from
  a satellite view). The game header shows the gusts. The model is unchanged.
- `experiments/fetch_wind_history.py` (the **Wind history** Action, run by
  hand) fetches the archive wind at kickoff for every outdoor game since 2014
  into `data/cache/wind_history.csv`.
- `experiments/wind_direction_test.py` then (a) checks the games file's
  recorded wind against the archive, (b) adds gusts, the along-field part and
  the crosswind part to the total model one at a time, walk-forward, with the
  rule above (tuning 2014-2018, holdout 2019 on; kept only if better on tuning
  and not worse on the holdout), and (c) repeats the wind-under table by
  along-field and crosswind bands. If the along-field bands carry the effect
  and the crosswind bands don't, direction matters.
- `experiments/wind_forecast_grade.py` grades the wind under on the forecasts
  the board actually logged (last forecast before kickoff) against the recorded
  wind and the closing total. Eleven games so far; it prints what it has.

To run it: start **Wind history** in Actions (about 2,000 archive calls, a few
minutes, resumable), pull, then `python experiments/wind_direction_test.py`
(`--no-model` for parts a and c only, `--fixture` to check the script with
made-up numbers). Results go here when there are any.

## Quarterback changes

Prompted by Carolina at Philadelphia in week 6: with Jalen Hurts out, the raw
model still had Philadelphia by 0.7 while the market moved about 5 points
toward Carolina. The model valued the switch to Tanner McKee at only 2.3
points, because Hurts's rating leaned on five weak 2026 games and an unknown
backup was rated close to an average starter.

Games where either team started a different quarterback from its previous game
(388 in 2011-2018, 475 since 2019) were scored separately. Run
`python experiments/qb_test.py`.

| Setting | Tuning, QB-change games | Holdout, QB-change games | Holdout, all games |
|---|---|---|---|
| Before (20-week QB half-life, backup at −0.07) | 10.409 | 10.360 | 10.013 |
| QB half-life 80 weeks | 10.377 | 10.280 | 9.994 |
| + unknown backup at −0.13 | **10.362** | **10.251** | **9.987** |
| Closing line | 10.345 | 10.038 | 9.803 |

Both changes are kept. Also tested and rejected: removing the starter's share
of the market-implied rating when he is out (`d_qbm`; no gain on the tuning
seasons) and a heavier quarterback prior (400 dropbacks; slightly worse).

Carolina at Philadelphia under the new settings: raw model Carolina by 0.5
instead of Philadelphia by 0.7. Carolina won 25-14. Quarterback changes are
still where the model trails the market most (10.25 against 10.04): it cannot
know a backup's talent before he plays.

The game card's model column is now labelled "Raw model", so it no longer reads
as the board's pick when the market-adjusted win chance says the opposite.

## Injuries beyond the quarterback

Prompted by Philadelphia's offensive line in week 6. For each team and game,
`nflmodel/injuries.py` finds the core players in six position groups by snap
share in earlier games, then marks who is unavailable: Out or Doubtful on the
injury report, or not active on that week's roster (injured reserve, game-day
inactive, released). Players on injured reserve drop off the weekly injury
report, so the roster is what catches them; Lane Johnson (inactive weeks 4 to
6) and Cam Jurgens (off the roster after week 4) never appeared on a report.
Data: nflverse injury reports, snap counts and weekly rosters, 2013 on.

Two measures per group: the share of the core missing now, and the change from
what the team's last four games already reflect (its ratings have seen those).
Tuning seasons here are 2013-2018. Run `python experiments/injury_test.py`.

| Added to the margin model | Tuning change | Holdout change | Kept |
|---|---|---|---|
| **Offensive line, change** | **−0.015** | **−0.006** | **yes** |
| Offensive line, missing now | +0.003 | −0.015 | no |
| Pass catchers, missing now | +0.003 | −0.020 | no |
| Pass catchers, change | +0.005 | −0.017 | no |
| Running back | −0.007 / +0.003 | +0.014 / +0.007 | no |
| Defensive line | −0.002 / +0.000 | +0.016 / +0.007 | no |
| Linebackers | −0.003 / −0.007 | −0.003 / −0.004 | no (tiny) |
| Defensive backs | +0.015 / +0.018 | −0.009 / −0.009 | no |
| All six groups, missing now | +0.037 | −0.027 | no |
| All six groups, change | +0.020 | −0.011 | no |

Only the offensive line helps in both periods. The model values the whole line
turning over at about 2.5 points, roughly half a point per additional missing
starter. Several groups (pass catchers, defensive backs, all groups together)
help since 2019 but hurt before, so by the rule they are out. One possible
reason is data: rosters only mark game-day inactives from 2020 on. Re-test
them once a few more seasons are in.

The opening-line test runs without injury information, since most of it
arrives after openers are posted.

Carolina at Philadelphia with all of the above: raw model Carolina by 0.5,
market-adjusted Carolina by 2.6 (line Carolina −3, final Carolina by 11). The
line effect on this game was small (−0.1) because Carolina was also missing
linemen (16% of its core against Philadelphia's 19%).

### Pass rush, by production

The position-based defensive-line group missed many edge rushers, who are often
listed as linebackers (in 2025, half of the top ten defenders by QB hits were).
So pass rush was rebuilt from production: every defender weighted by QB hits
plus sacks from play-by-play, matched to rosters and injury reports by player
id, with half of last season's production counted for early weeks and new
signings. Run `python experiments/pass_rush_test.py`.

| Added to the current model | Tuning | Holdout | Kept |
|---|---|---|---|
| Pass rush missing now | +0.018 | −0.005 | no |
| Pass rush, change vs recent games | +0.013 | −0.009 | no |
| Pass rush in the total model | +0.007 / +0.021 | +0.011 / +0.006 | no |

The sign is right (losing a team's whole pass rush is worth about 4-5 points),
but the effect does not hold up: season by season it helped in 3 of 7 seasons
before 2020 and 3 of 7 since.

### Is it the data?

Rosters only mark game-day inactives from 2020 on, so earlier seasons see fewer
absences. Change in mean miss per season when each group is added:

| Group | 2013-2019 | 2020-2026 |
|---|---|---|
| Pass rush | +0.014 (better in 3 of 7) | −0.010 (3 of 7) |
| Pass catchers | +0.005 (2 of 7) | −0.013 (**7 of 7**) |
| Defensive backs | +0.017 (2 of 7) | −0.007 (5 of 7) |
| All groups | +0.039 (2 of 7) | −0.017 (6 of 7) |

Pass catchers have helped in every season since the better data starts. That
is a lead, not a decision: the split was chosen after seeing the results. The
honest test is to add pass catchers at the end of the 2026 season only if 2026
also improves, and to judge it on 2027.
(From October 29, 2026 it runs as a second shadow spread version; see "Edge
tests, late October 2026".)

## Defensive players, re-tested and rated (late October 2026)

Three questions: do the defensive features rejected above help now that the model also
has skill starters out, falling snap shares, wind and pace? Do they help if each missing
defender counts by how good he is instead of one each? And do new defender ratings (PFR
pressures, coverage and tackling, FTN blitz charting) help the spreads, totals or player
projections? Same rule as above: each candidate is added alone, walk-forward, and kept only
if it lowers the average miss on the tuning seasons and holds up on the holdout. Who is
missing uses only what is known by Friday (Out or Doubtful on the report, or off the
active roster the week before), as `nflmodel/game_roles.py`. Scripts:
`experiments/def_retest.py`, `def_ratings.py`, `def_test.py`, `def_se.py`,
`def_players_test.py`, `def_players_se.py`, `def_players_combined.py`,
`cfb_def_test.py`; every row, with each season, is in `experiments/out/def_*.csv`.

**Windows.** Head counts and play-by-play pass rush go back to 2013: tuning 2013-2018,
holdout 2019 on, as in October. PFR's defensive stats start in 2018. A rating needs
earlier games, and a walk-forward fit only learns a new input from the season after it
first appears, so 2020 is the first season the model can use them: **tuning 2020-2022
(838 games), holdout 2023 on (962)**. Three seasons is short, so read these with
caution. FTN charting starts in 2022, which leaves no tuning window at all; its rows are
reported for 2024 on and can't earn a place.

### The October rejects, re-tested

Change in the margin's (or total's) average miss when added; negative is better. "Seasons
better" counts seasons whose miss fell. The 2013-2019 against 2020-on split was chosen after
seeing the October results (rosters mark game-day inactives only from 2020), so it is
reported, not used to keep anything.

| Candidate | October tuning | October holdout | Now tuning | Now holdout | 2013-19 (seasons better) | 2020 on (seasons better) | Kept |
|---|---|---|---|---|---|---|---|
| Defensive line, missing now | −0.002 | +0.016 | +0.005 | +0.014 | +0.007 (3 of 7) | +0.014 (3 of 7) | no |
| Defensive line, change | +0.000 | +0.007 | +0.009 | +0.007 | +0.011 (2 of 7) | +0.005 (2 of 7) | no |
| Linebackers, missing now | −0.003 | −0.003 | +0.005 | −0.002 | +0.002 (4 of 7) | −0.001 (5 of 7) | no |
| Linebackers, change | −0.007 | −0.004 | +0.000 | −0.004 | −0.003 (3 of 7) | −0.002 (5 of 7) | no |
| Defensive backs, missing now | +0.015 | −0.009 | +0.017 | −0.008 | +0.014 (3 of 7) | −0.008 (5 of 7) | no |
| Defensive backs, change | +0.018 | −0.009 | +0.019 | −0.007 | +0.016 (3 of 7) | −0.008 (6 of 7) | no |
| Pass rush, missing now | +0.018 | −0.005 | +0.017 | −0.000 | +0.015 (1 of 7) | −0.000 (4 of 7) | no |
| Pass rush, change | +0.013 | −0.009 | +0.011 | −0.005 | +0.012 (4 of 7) | −0.008 (2 of 7) | no |
| Pass rush in the total, missing now | +0.007 | +0.011 | +0.007 | +0.011 | +0.006 (4 of 7) | +0.012 (4 of 7) | no |
| Pass rush in the total, change | +0.021 | +0.006 | +0.021 | +0.006 | +0.018 (2 of 7) | +0.007 (3 of 7) | no |
| All six groups, missing now | +0.037 | −0.027 | +0.089 | −0.000 | +0.080 (1 of 7) | −0.005 (4 of 7) | no |
| All six groups, change | +0.020 | −0.011 | +0.045 | +0.017 | +0.052 (2 of 7) | +0.005 (4 of 7) | no |

Same answer as October, a little worse: the new skill-player signals already carry part of
what the counts said. Against openers none helps on both periods.

### Weighted by rating

`experiments/def_ratings.py` rates every defender from earlier games only: decayed (half-life
10 games, last season x 0.6) and shrunk toward his position group (interior line, edge,
linebacker, corner, safety). Pass rush is QB hits plus sacks (play-by-play, 2012 on) or PFR
pressures (2018 on), per estimated pass-rush snap; coverage is yards allowed per estimated
coverage snap or per target; tackling is missed tackles per attempt. No free source counts
how many snaps a defender rushed or covered, so both use his defensive snaps times the share
of the opponent's plays that were dropbacks. A missing defender is worth his expected snap
share times (his rating minus his replacement's). Two replacements were tried: the group's
backup level, and the next available player at his depth-chart slot. The backup level did
better almost everywhere; its rows are below, the depth-chart rows are in the CSV.

| Candidate (vs backup level) | Window | Tuning | Holdout | Seasons better since 2020 | Passes the rule |
|---|---|---|---|---|---|
| Pass rush (play-by-play), all defenders, missing now | long | −0.001 | −0.004 | 5 of 7 | yes |
| Pass rush (play-by-play), all defenders, change | long | −0.001 | −0.004 | 5 of 7 | yes |
| Pass rush (play-by-play), defensive line, change | long | −0.000 | −0.001 | 4 of 7 | yes |
| Pass rush (play-by-play), linebackers, missing now | long | −0.000 | −0.003 | 4 of 7 | yes |
| Pass rush (play-by-play) in the total, missing now | long | −0.002 | −0.000 | 3 of 7 | yes |
| Coverage, defensive backs, missing now | PFR | −0.003 | −0.010 | 6 of 7 | yes |
| Coverage, defensive backs, change | PFR | −0.009 | +0.001 | 5 of 7 | no |
| Coverage, linebackers, missing now | PFR | +0.125 | −0.039 | 4 of 7 | no |
| Pressures (PFR), defensive line, missing now | PFR | +0.018 | −0.003 | 3 of 7 | no |
| All defenders, pressures + coverage, change | PFR | −0.012 | −0.016 | 6 of 7 | yes |
| All defenders, pressures + coverage, missing now | PFR | +0.019 | −0.034 | 3 of 7 | no |

### New ratings and team-level stats

| Candidate (vs backup level) | Model | Tuning | Holdout | Seasons better since 2020 | Passes the rule |
|---|---|---|---|---|---|
| Pressure rate lost, change | margin | −0.005 | −0.013 | 6 of 7 | yes |
| Pressure rate lost, missing now | margin | +0.007 | −0.018 | 5 of 7 | no |
| Coverage yards per snap lost, change | margin | −0.008 | −0.009 | 5 of 7 | yes |
| Coverage yards per snap lost, missing now | margin | +0.001 | −0.026 | 4 of 7 | no |
| Coverage yards per target lost, change | margin | +0.015 | −0.028 | 5 of 7 | no |
| Missed-tackle rate lost, change | margin | +0.021 | +0.000 | 4 of 7 | no |
| Missed-tackle rate lost, change (vs depth chart) | total | −0.013 | −0.004 | 4 of 7 | yes |
| Any of the four ratings, missing now or change | total | +0.004 to +0.11 | | | no |
| Team pressure rate | margin / total | +0.001 / +0.001 | −0.006 / +0.000 | | no |
| Team blitz rate | margin / total | +0.067 / −0.012 | −0.013 / +0.005 | | no |
| Team yards per target allowed | margin / total | +0.008 / +0.016 | −0.005 / −0.002 | | no |
| Team missed-tackle rate | margin / total | +0.010 / −0.043 | +0.002 / +0.002 | | no |
| FTN blitzers per dropback (2024 on only) | margin / total | | +0.007 / −0.028 | | can't qualify |
| FTN pass rushers per dropback (2024 on only) | margin / total | | −0.004 / +0.003 | | can't qualify |

**Is any of it real?** For each candidate that passed, `def_se.py` pairs the two models game
by game. Every change is within about one standard error: the largest, "pressures + coverage,
change", is −0.012 ± 0.013 on the tuning seasons and −0.016 ± 0.018 on the holdout; "pressure
rate lost, change" is −0.005 ± 0.010 and −0.013 ± 0.008. About 80 candidates were tried, so
a few were bound to pass by chance. The signs are right (losing a team's best pass rusher
and a starting corner is worth about a point at the 95th percentile), but none of these
beats noise, and none helps against openers on both periods. **The spread and total models
are unchanged.**

### Player projections

The projections already scale each player by what the opponent allows per target, carry and
attempt, and receiving uses the opponent's missing defensive backs and pass rush by head count.
Each candidate below adds one term to the last calibration step of the stats it should touch
(`def_players_test.py`; tuning 2020-2022, holdout 2023 on; players who played; standard errors
clustered by game, `def_players_se.py`).

| Candidate | Stat | Tuning | Holdout | Passes the rule |
|---|---|---|---|---|
| Opponent's team missed-tackle rate | receiving yards | −0.013 ± 0.003 | −0.022 ± 0.002 | yes |
| Opponent's team yards per target allowed | receiving yards | −0.013 ± 0.007 | −0.014 ± 0.005 | yes |
| Opponent's team pressure rate | interceptions | −0.003 ± 0.002 | −0.002 ± 0.001 | yes |
| Opponent's tackling lost to injury | receiving yards | −0.004 ± 0.001 | −0.002 ± 0.002 | yes, tiny |
| Opponent's pass rush lost (play-by-play) | passing, receiving | worse or ±0.002 | | no |
| Opponent's pressure lost (PFR) | passing yards | +0.20 | −0.02 | no |
| Opponent's coverage lost (PFR, DBs) | passing, receiving | ±0.01 | worse | no |
| Opponent's team pressure rate | passing yards | −0.09 | +0.21 | no |
| Opponent's team missed-tackle rate | rushing yards | +0.001 | −0.002 | no |

Together (`def_players_combined.py`: missed-tackle rate and yards per target in receiving
yards, pressure rate in interceptions), receiving yards improve by 0.025 ± 0.007 on the tuning
seasons and 0.025 ± 0.006 on the holdout, against an average miss of about 15 yards; that is
real but small. It would move this season's receiving projections by a median 0.6 yards
(5 at most) and interceptions by about 0.02. Not applied yet: these need PFR's weekly
defensive file in the data pull, and the board changes only after Ryan approves.

### College

College football has no snap counts, no coverage charting and no history of injury
designations. `experiments/cfb_def_test.py` weights each defender by his share of his team's
box-score production (sacks plus hurries, passes defended plus interceptions, tackles) and
marks him missing if he recorded no stat in the team's previous game, the only version that
can be tested. Tuning 2017-2019, holdout 2021 on:

| Candidate | Margin tuning / holdout | Total tuning / holdout |
|---|---|---|
| Pass rush, missed last game | +0.005 / −0.001 | −0.010 / +0.019 |
| Pass rush, change | −0.004 / +0.003 | −0.013 / +0.018 |
| Coverage, missed last game | −0.011 / −0.001 | +0.003 / −0.002 |
| Coverage, change | −0.002 / −0.001 | +0.018 / +0.002 |
| Tackling, missed last game | +0.001 / −0.001 | −0.007 / +0.013 |
| Tackling, change | −0.003 / +0.003 | −0.007 / +0.015 |
| Ceiling: coverage defenders absent from this game (known only after it) | −0.021 / −0.010 | +0.001 / +0.004 |

Coverage, missed last game, passes the rule on the margin, by about a hundredth of a point
against an average miss of 12.4. Even knowing exactly which defenders sat out (the ceiling
row, which can't be known in advance) is worth only 0.01 to 0.02. **The college board is
unchanged**: injured defenders stay listed with no change to the numbers.

### Shadow test (from October 29, 2026)

None of the defensive inputs beat noise on past seasons, so the most promising ones are run
alongside the board instead of on it (`nflmodel/shadow.py`). On every refresh, the Update board
Action refits the game model and two player projections with them and logs the results to
`data/shadow/`. A game's rows are frozen at kickoff, and nothing reaches the page, the bets or
the props. The inputs and the rule were set on October 29, 2026, before any of these games
were played:

| Shadow version | Inputs added | Past seasons, tuning / holdout (board's version of the inputs) |
|---|---|---|
| Spread | pressures and coverage yards lost to absences beyond the last four games (vs backup level) | −0.004 / −0.015 (2020-2022 / 2023 on) |
| Total | missed-tackle rate lost, beyond the last four games (vs depth-chart replacement) | −0.020 / +0.006 |
| Receiving yards | the opponent's missed-tackle rate and yards per target allowed | −0.054 ± 0.008 / −0.005 ± 0.006 |
| Interceptions | the opponent's pressure rate | −0.002 ± 0.002 / −0.002 ± 0.001 |

The past-season numbers differ a little from the tables above because the board's version
builds each season from it and the two before, as it will live. **Rule:** a shadow version
replaces the board's only if its average miss is lower on the rest of 2026 (week 9 on) and,
separately, on 2027. Score it with `python experiments/shadow_score.py`. Until then, don't
change the inputs.
A second spread version, pass catchers out, was added to the shadow on the same date and
under the same rule, logged in its own file (see "Edge tests, late October 2026", item 5).

## Player roles in the game model, October 2026

The player work added snap shares, depth charts and team play-calling
(`nflmodel/roles.py`). Tested in the margin and total models
(`experiments/game_roles_test.py`, 2014 on):

| Candidate | Model | Tuning | Holdout | Kept |
|---|---|---|---|---|
| Depth-chart starters (QB, RB, WR, TE) out, home minus away | margin | better | better | **yes, with the next** |
| Regular players' snap share falling, last two games vs average | margin | better | better | **yes** |
| Usage missing (target and carry shares of players out) | margin | no gain | | no |
| Neutral-situation pass rate | margin / total | no gain | | no |
| Any of these, summed | total | no gain | | no |

Together the two kept signals cut the margin's average miss by 0.013 points on the
tuning seasons and 0.019 since 2019. "Out" uses only what is known by Friday
(`nflmodel/game_roles.py`): last week's roster status plus players listed Out or
Doubtful, never the game-day inactive list. Against openers
(`experiments/game_roles_open_test.py`; side picked by the model against the opening
spread):

| | Average miss, tuning / since 2019 | Win rate at the opener, tuning / since 2019 |
|---|---|---|
| Before | 10.058 / 9.989 | 53.5% / 52.7% (1,204 / 1,825 games) |
| With both signals | 10.026 / 9.976 | 54.8% / 53.4% |

Better on both periods, but small: the gain since 2019 (+0.6 points) is inside
the noise for 1,825 games (about ±1.2). Kept because it helps on both and the
inputs are known in time. Totals unchanged.

## Edge tests, late October 2026

Six questions about where the board's edge is and how much of it to stake, each tested by
the rule above: a candidate is added alone, walk-forward, and kept only if it lowers the raw
model's average miss on the tuning seasons and holds on the holdout. Scripts:
`experiments/line_move_test.py`, `experiments/qb_draft_prior_test.py`,
`experiments/shadow_score.py`; the haircut and the cap are in `nflmodel/backtest.py`,
`nflmodel/bets.py` and `nflmodel/slate.py`. Base numbers here come from the production
engine with the injury groups (margin miss 10.237 tuning, 9.985 holdout, through week 8 of
2026); they differ a little from the October table, which used the wider test engine.

| Item | Result | Kept |
|---|---|---|
| 1. Line movement as a feature | tuning +0.012, holdout −0.014 | no |
| 1. Line movement as a filter on spread bets | no bucket holds on both periods | no |
| 2. Edge haircut (share of claimed EV that was real) | NFL 0.00 (slope −0.38 ± 0.28, 2018-2025); college 0.52 (± 0.39) | applied to stakes (`EDGE_HAIRCUT`) |
| 3. Per-game stake cap | 5% of bankroll per game, all sides together | yes (`MAX_GAME_STAKE`) |
| 4. Props book-disagreement tags | market-only chips, no model opinion | yes (tags only, flags stay off) |
| 5. Pass catchers in shadow | second shadow spread version, from Oct 29, 2026 | shadow only |
| 6. Backup quarterback prior by draft round | tuning −0.007, holdout +0.017 (QB-change games) | no |

### 1. Line movement

`move` is the closing spread less the nfelo opening spread (positive: the market moved toward
the home team), 0 for the 16% of games without an opener. Added to the margin model alone:

| Candidate | Tuning | Holdout | Seasons better since 2014 |
|---|---|---|---|
| move | +0.012 | −0.014 | 8 of 13 |
| move + its size | +0.008 | −0.008 | 7 of 13 |

Worse on the tuning seasons, so out by the rule. It would also break the board's rule that
this week's line is never a feature (README): the closing line is not known when the board
prices a game, and the opener is what the model is supposed to be judged against.

The same opener lets the backtest's market-adjusted spread bets (EV above 0, closing prices)
be split by whether the line moved toward the bet's side between the opener and the close:

| Line moved | All (1,871 bets) | Tuning 2014-2018 | Holdout 2019 on |
|---|---|---|---|
| toward the model's side | 279-267-8, 51.1%, −0.2% ± 4.1% | 142-141-5, 50.2%, −1.7% ± 5.7% | 137-126-3, 52.1%, +1.5% ± 6.0% |
| did not move | 219-167-8, 56.7%, +10.1% ± 4.8% | 90-60-4, 60.0%, +17.0% ± 7.7% | 129-107-4, 54.7%, +5.7% ± 6.2% |
| away from it | 475-422-26, 53.0%, +2.9% ± 3.2% | 264-200-14, 56.9%, +10.9% ± 4.4% | 211-222-12, 48.7%, −5.7% ± 4.5% |

The pattern flips between the periods ("away" was the best bucket on the tuning seasons and
the worst since), and every bucket is within about one standard error of the others on the
holdout, so there is no filter here. The unmoved bucket is the only one positive in both,
and it is the smallest.

### 2. Edge haircut

Claimed EV has run ahead of realized returns on every part of the board (props: +14.9%
claimed, +0.5% realized, PROPS.md). `backtest.edge_haircut` measures the share that was real:
on the graded market-adjusted bets with EV above 0 from earlier seasons, realized profit is
regressed on claimed EV through the origin; the slope is the share, clipped to [0, 1]. Over
the last eight seasons it is the factor the board stakes from.

| Board, 2018-2025 | Bets | Claimed EV | Realized | Slope ± 1 s.e. | Factor |
|---|---|---|---|---|---|
| NFL, all markets | 2,364 | +5.5% | −2.5% | −0.38 ± 0.28 | **0.00** |
| NFL, spreads only | 1,217 | +4.9% | −0.3% | +0.23 ± 0.41 | (0.23) |
| NFL, moneylines only | 1,145 | +6.1% | −4.8% | −0.71 ± 0.39 | (0.00) |
| College, spreads (the only market with prices) | 2,432 | +3.6% | +1.7% | +0.52 ± 0.39 | **0.52** |

NFL by season (slope): 2014 1.36, 2015 1.41, 2016 0.87, 2017 −0.41, 2018 1.38, 2019 −0.70,
2020 −0.88, 2021 0.48, 2022 −0.96, 2023 −2.66, 2024 −0.01, 2025 −2.87. The seasons the
settings were chosen on returned what they claimed and more; since 2019 the claimed edge has
not shown up at the close (the same thing the README's 51.9% says), and the moneylines,
which carry the biggest claimed edges, have been the worst. College by season: 2.27, −0.10,
0.75, −0.72, 0.86, 0.59, 0.71, 1.64 (2018-2025).

**What it does (`config.EDGE_HAIRCUT = True`, both boards):** the stake of every side, at the
anchor line and at each book, is the quarter-Kelly stake for the model probability shrunk
toward the market's no-vig number by the factor (`bets.stake_prob`): with factor h, the
stake is sized from market + h × (model − market). EV, probabilities, edge and fair price on
the page are still the model's; only the stake changes. The factor is in `meta.haircut`
(with the by-season and by-market tables) and `meta.haircut_factor`, and the pricing note
under the Every price table says what it is. **On the NFL board the factor is 0.0 as of this
test, so no NFL stake is sized until the claimed edge shows up in results**; the college board
stakes at about half. The switch is there to turn off (`EDGE_HAIRCUT = False`) if that is not
wanted; the number is honest either way. The Update board Actions refit it on their backtest
runs (Tuesday NFL, Monday college) and carry it from the previous `data.json` otherwise. The
factor's standard error is large (0.3-0.4 on 2,400 bets), so it will move season to season.

### 3. Per-game stake cap

`config.MAX_GAME_STAKE = 0.05` (both boards): when the stakes the page shows for one game,
all sides together (the best book's stake where its EV beats the anchor's, else the anchor's),
add up to more than 5% of bankroll, every stake in the game is scaled down by the same factor
(`slate.cap_game`; the college board calls the same function). This is what the props already
do per game (`propmarket.flag`, 3%). Each game carries `game_stake` (the sum before the cap)
and `stake_scale`. The per-bet cap (3%) still applies first. Tested in `tests/test_model.py`;
on the week 8-9 slates no game reached it (the largest, Baltimore at Buffalo, added up to
4.8% before the haircut), so it is a guard for the weeks the model likes several sides of one
game.

### 4. Props: book-disagreement tags

Flags stay off (`FLAGS_ON = False`). Two tags that need no model opinion, from
`propmarket.market_tags`, shown as small chips in the Props section and kept in the log:

- **book gap**: DraftKings and FanDuel post the same player's main line a full step apart
  (5 yards on yards props, half a point on counts). The better number at each book is the
  side to take: the over at the lower line, the under at the higher. When both of those are
  −110 or better it is a **middle** (both can win on a result between the lines). The log's
  `book_gap` names the two lines, `take` the side on that row, `middle` whether both prices
  qualify.
- **ladder gap**: an alternate-ladder rung whose no-vig chance at a book is more than 5
  points off what the same book's main line implies through the model's distribution (the
  main line's no-vig chance plus the model's difference between the two lines, the carry
  `proplog.grade` already uses for closing line value). `ladder_gap` is the rung's chance
  minus the implied one (negative: the over is cheaper than the main line says), `tag_book`
  the book, `take` the side.

The prop log is append-only and its header is fixed, so the tags live in a sibling file,
`data/props/prop_log_tags.csv`, keyed by `row_id` like `prop_log_new.csv`; only rows priced
from now on carry them. `proplog._append` now writes an existing file's own columns, so a
column added to `LOG_COLS` can never put new rows out of step with an old header. The Props
files (`docs/props/`) carry the tags (`bg`, `mid`, `lg`, `tb`, `take`) and each week's summary
counts them (`book_gaps`, `middles`, `ladder_gaps`). Whether the tagged sides win is a question
for the log: nothing is flagged or staked on them. Tested with a fixture in `tests/test_props.py`.

### 5. Pass catchers in shadow (from October 29, 2026)

"Pass catchers, missing now" (`d_miss_pc`, `nflmodel/injuries.py`: the snap-weighted share of a
team's core WR/TE unavailable) helped the margin in all 7 seasons since 2020 but hurt before,
and the rule says that split was chosen after the fact. So it joins the shadow test instead of
the board: `shadow.MARGIN_PC` is the board's margin features plus `d_miss_pc`, refitted on
every refresh and logged to `data/shadow/passcatchers_games.csv` (raw model margin, current
and shadow, per game) under the same freeze-at-kickoff rule as the defensive log. The first
shadow version's inputs (`shadow.MARGIN`, `TOTAL`, `PLAYER`) are unchanged. **Rule, set
October 29, 2026 before any of these games were played:** the pass-catchers version replaces
the board's spread model only if its average miss is lower on the rest of 2026 (week 9 on)
and, separately, on 2027. Score it with `python experiments/shadow_score.py`, which now
prints the pass-catchers rows next to the defensive ones.

### 6. Backup quarterback prior by draft capital

`config.QB_REPLACEMENT` gives every unknown quarterback the same prior, −0.13 EPA per dropback
against the league. nflverse's draft file lets the prior depend on where he was drafted:
round 1, rounds 2-3, later or undrafted. Among the 127 quarterbacks whose first 300 dropbacks
are in the data (careers from 2007), round 1 picks averaged −0.068 in those dropbacks,
rounds 2-3 −0.092, later or undrafted −0.091. Two versions, each scored as the quarterback
test above on games where either team changed starters (`experiments/qb_draft_prior_test.py`;
the rating engine takes a per-quarterback prior through `qb_priors`):

| Prior | Tuning, QB-change games | Holdout, QB-change games | Holdout, all games |
|---|---|---|---|
| Flat −0.13 (current) | 10.362 | 10.341 | 9.991 |
| Fitted by bucket, walk-forward (first-300 averages, shrunk) | 10.355 | 10.358 | 9.993 |
| Fixed: round 1 +0.04, late −0.04 | 10.370 | 10.353 | 9.987 |
| Fixed: round 1 +0.06, late −0.06 | 10.382 | 10.364 | 9.988 |
| Fixed: round 1 +0.08, late −0.04 | 10.387 | 10.371 | 9.990 |
| Fixed: round 1 +0.04, late −0.08 | 10.381 | 10.358 | 9.987 |
| Fixed: round 1 +0.10, late −0.10 | 10.417 | 10.390 | 9.993 |
| Closing line | 10.345 | 10.112 | 9.803 |

The fitted version helps the tuning seasons by 0.007 and hurts the holdout by 0.017; every
fixed offset is worse on the tuning seasons. Not kept. The buckets differ less than they
seem to (a first-round backup is about 0.02 EPA better than a late one over his first 300
dropbacks, worth under half a point), and the 200-dropback prior already gives way to a
backup's own games quickly. `QB_REPLACEMENT` stays at −0.13.

## What to try next

- **Opening lines for totals.** The model's sides do best against openers;
  nobody has tested totals that way yet, and wind forecasts are what a total
  opener cannot know. That needs a source of opening totals.
- **Grade the wind under on forecasts.** Every forecast the board fetches is
  saved to `data/wind_log.csv`; `experiments/wind_forecast_grade.py` grades the
  wind under on them. After a season of logs, take its table seriously.
- **Wind direction and gusts.** Set up above; fetch the archive and run the
  test.
- **Pass catchers.** In the shadow test from October 29, 2026 (above); score it with
  `experiments/shadow_score.py` after the 2026 season and again after 2027.
- **The edge haircut.** Re-read `meta.haircut` each season: the NFL factor is 0.0 now, and
  the first season whose bets return what they claim will move it.
- **Defensive ratings.** The shadow test decides (above). Score it with
  `experiments/shadow_score.py` after the 2026 season and again after 2027.
- **College football**, where the market is softer and rosters change more.
