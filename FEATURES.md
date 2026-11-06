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

### Wind direction and gusts (tested Oct 29, 2026: nothing kept)

The model sees only the wind's speed. A 15 mph wind blowing goal line to goal
line should hurt passing and kicking far more than the same wind across the
field, and a gusty 12 mph may play like 20. Tested on the first archive fetch
(1,115 of the 2,450 outdoor games since 2014; the archive refused the rest on
the first run, and a later run fills them in). **Nothing earns a place.**

| Added to the total model | Tuning | Holdout | Kept |
|---|---|---|---|
| Gusts (mph) | −0.000 | +0.003 | no |
| Along-field wind (mph) | −0.004 | +0.010 | no |
| Crosswind (mph) | +0.004 | +0.000 | no |
| Along-field and crosswind together | −0.001 | +0.009 | no |
| Archive speed in place of the recorded wind | +0.019 | +0.037 | no |

The wind under by along-field band (10-19 mph) went 51-42 and by crosswind band
47-31: both directions carry it, so direction adds nothing the speed does not
say, at least with bearings that are mostly approximate. Gusts are worse than
useless as a band (15-19 mph gusts: 45% unders).

The finding that matters is the sanity check. The games file's recorded wind
and the archive's hour-of-kickoff wind at the stadium agree only loosely
(correlation 0.72, average gap 2.5 mph), and the wind under is weaker on the
archive's number: 10-19 mph unders 122-107 (53.3%, +2.6% ± 6.5%) against
402-287 (58.3%) on the recorded wind, on the games that have both. The recorded
wind is what the stadium reported during the game; a bettor only has a
forecast. So part of the published wind-under edge may be hindsight, and the
honest grade is the one on the logged forecasts
(`experiments/wind_forecast_grade.py`), which needs a season of logs. Until
then treat the wind tag as a lead, and size it small.

The set-up:

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
| Opponent's team missed-tackle rate | receiving yards | −0.013 ± 0.003 | −0.022 ± 0.002 | yes (see below) |
| Opponent's team yards per target allowed | receiving yards | −0.013 ± 0.007 | −0.014 ± 0.005 | yes (see below) |
| Opponent's team pressure rate | interceptions | −0.003 ± 0.002 | −0.002 ± 0.001 | yes |
| Opponent's tackling lost to injury | receiving yards | −0.004 ± 0.001 | −0.002 ± 0.002 | yes, tiny |
| Opponent's pass rush lost (play-by-play) | passing, receiving | worse or ±0.002 | | no |
| Opponent's pressure lost (PFR) | passing yards | +0.20 | −0.02 | no |
| Opponent's coverage lost (PFR, DBs) | passing, receiving | ±0.01 | worse | no |
| Opponent's team pressure rate | passing yards | −0.09 | +0.21 | no |
| Opponent's team missed-tackle rate | rushing yards | +0.001 | −0.002 | no |

Together (`def_players_combined.py`: missed-tackle rate and yards per target in receiving
yards, pressure rate in interceptions), receiving yards improved by 0.025 ± 0.007 on the tuning
seasons and 0.025 ± 0.006 on the holdout. **That result does not hold up.** Each team's rate
was measured against an average over all seasons, 2026 included, so the reference used later
games. Measured against something a projection could know at the time, the receiving gain on
the holdout is mostly gone:

| Reference for "above or below average" | Receiving yards, tuning / holdout | Interceptions, tuning / holdout |
|---|---|---|
| Average of all seasons (uses later games; the first test) | −0.025 / −0.026 | −0.003 / −0.002 |
| This week's league average | −0.010 ± 0.007 / −0.007 ± 0.005 | −0.002 ± 0.002 / −0.003 ± 0.001 |
| League average of earlier seasons | −0.054 ± 0.008 / −0.005 ± 0.006 | −0.002 ± 0.002 / −0.002 ± 0.001 |

Every version passes the rule on both periods, but on the holdout the receiving change is
within about one standard error and the interceptions change about two, against average
misses of 15 yards and 0.7 interceptions. The receiving result also swings fivefold with the
choice of reference, which a real effect shouldn't. A board-ready version (PFR's weekly
defensive file, the earlier-seasons reference) was built on the branch
`feature/defense-in-player-projections` (commit 901b3c7, checked with
`experiments/def_players_production.py`); it moved this season's receiving projections by a
median 0.6 yards (5 at most). **Not applied.** The branch was removed in the Nov 3, 2026
branch cleanup; rebuild from this description if it's ever wanted.

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

## Derivative markets (October 30, 2026)

Team totals and first-half spreads, totals and moneylines, priced off the boards' full-game
curves (`nflmodel/derivs.py`; the market snapshots in `nflmodel/market.py`; test script
`experiments/derivs_test.py`). Books derive these from the main line with crude rules and
limits are low, so they are mispriced more often than full-game lines; the question was
whether the boards' curves describe them well enough to say when.

**How they are priced.** A team total comes from the margin and total curves combined: home
points are (total + margin) / 2, so the chance of every whole-number team score is the sum of
the joint probabilities of the (margin, total) pairs that give it, the two curves taken as
independent and pairs of different parity dropped. A first-half line uses a half-game curve
fitted exactly as `distributions.py` fits the full game, on half-time scores (nflverse
play-by-play for the NFL, ESPN's quarter linescores in the cfbfastR schedule file for
college; `data/cache/half_scores_<season>.csv` on both boards): the first-half margin is
centred on a fitted share of the full-game spread, the first-half total on a straight line in
the full-game total, with the halves' own key-number multipliers and widths. Where Pinnacle
quotes the derivative, the curve is anchored to its no-vig number at its line (as the
full-game curve is to the full-game line) and each book's price is read at that book's own
line, so "beats Pinnacle" means what it means on the main markets; where Pinnacle does not
quote it, the curve sits at the implied centre and nothing is flagged.

**Does the curve match history?** There are no closing derivative lines in the repo's history
(the market log began on Oct 29, 2026), so the test is of the curves: walk-forward, every
played game priced from its closing full-game line alone, scored against actual team points
and half-time scores at the implied derivative lines and at those lines moved 3.5 and 7
points each way. A curve matches history when its hit rates equal its chances:

| Curve, both leagues | Lines scored | Log loss | Brier | Predicted chance → hit rate by bucket |
|---|---|---|---|---|
| NFL team total, joint pmf | 34,030 | **0.6066** | **0.2088** | 19→21, 27→27, 36→33, 50→49, 63→62, 75→74, 81→79 |
| NFL team total, normal approximation | 34,030 | 0.6066 | 0.2089 | 19→21, 27→27, 36→34, 50→49, 63→62, 75→74, 81→80 |
| NFL first-half margin, half curve | 17,015 | **0.6239** | **0.2168** | 28→29, 37→38, 50→51, 61→63, 74→77 |
| NFL first-half margin, plain normal | 17,015 | 0.6242 | 0.2169 | 26→26, 37→36, 50→51, 63→63, 74→77 |
| NFL first-half total, half curve | 17,015 | **0.5982** | **0.2051** | 19→20, 27→26, 49→47, 62→61, 77→77, 81→79 |
| NFL first-half total, plain normal | 17,015 | 0.6002 | 0.2060 | 20→20, 26→24, 36→30, 50→47, 64→60, 74→72 |
| College team total, joint pmf | 69,144 | **0.6359** | 0.2223 | 28→28, 37→35, 50→48, 61→58, 74→71, 84→81 |
| College team total, normal approximation | 69,144 | 0.6361 | 0.2223 | 28→28, 37→35, 50→48, 61→58, 74→71, 84→81 |
| College first-half margin, half curve | 33,800 | **0.6380** | **0.2233** | 29→29, 38→38, 50→50, 60→60, 72→73 |
| College first-half margin, plain normal | 33,800 | 0.6391 | 0.2238 | 28→28, 39→37, 50→50, 61→60, 72→73 |
| College first-half total, half curve | 33,800 | **0.6312** | **0.2201** | 29→27, 37→34, 49→47, 59→58, 72→72 |
| College first-half total, plain normal | 33,800 | 0.6338 | 0.2213 | 27→25, 38→33, 50→47, 62→58, 73→72 |

NFL 2014-2026, college 2017-2026. The joint pmf and the normal approximation are the same
to three decimals on team totals (the pmfs' mean and variance carry nearly everything); the
joint pmf is used because it keeps the scoring structure for free. The half curve beats a
plain normal everywhere, mostly by getting the key numbers right:

| First-half margin | NFL observed | Half curve | Plain normal | College observed | Half curve | Plain normal |
|---|---|---|---|---|---|---|
| 0 (tied at the half) | 7.1% | 6.7% | 3.6% | 6.6% | 6.4% | 2.7% |
| 3 | 12.2% | 11.6% | 6.9% | 10.0% | 8.9% | 5.3% |
| 7 | 11.5% | 11.5% | 5.9% | 11.7% | 11.1% | 4.8% |
| 10 | 8.7% | 8.7% | 4.8% | 6.8% | 6.7% | 4.3% |

The fits (through 2025): NFL first-half margin 0.58 x the spread, first-half total
−0.2 + 0.51 x the total, widths 10.2 and 8.8 points; college 0.59 x the spread, 1.2 + 0.50 x
the total, widths 12.3 and 11.1. A first-half spread is more than half the full spread on
both boards: favourites build their lead early and second halves mean-revert. The limits:
NFL first-half favourites at the high-chance lines cover 2-3 points more often than either
curve says (74→77); college overs on team totals and first-half totals hit 2-3 points less
often than the curve says at the high-chance lines (74→71, 61→58: the loser of a college
blowout scores less than independence implies). Both are within what the anchoring to
Pinnacle's own quote absorbs, since the curve only carries its number between nearby lines.

**What it does on the boards.** Each game's page has a Derivatives table: every book's team
totals and first-half lines, the model's chance and the market's no-vig chance at that book's
line, EV, and a "beats Pinnacle" chip where the price beats Pinnacle's own derivative quote
by 1%. Those sides are **price-tier bets** on the Bet sheet and in the ledger
(`ledger.csv` market `Team total`, `1H Spread`, `1H Total`, `1H Moneyline`; first-half rows
graded on the half-time score, no closing line value yet) and count toward the game's 5% cap.
The curve's own opinion on a derivative is **not** a bet: the backtest above is of the
curve's calibration against results, not of an edge against closing derivative lines, and
there is none to test against until the market log has a season of them. Nothing here
changes the main-market bets.

**Credits.** The Odds API serves these markets one event at a time, (markets returned) x
(regions) a call, so `run_market.py derivatives` spends at most `market.DERIV_BUDGET` a run
(NFL 70: 16 games x 4 markets; college 160: 50 games x 3, the first-half moneyline dropped)
on the 14:05 UTC run on Thursday, Saturday and Sunday (NFL) and Friday (college): about 900
and 700 credits a month, the whole market workflow about 3,000 of the account's 20,000.

**Next.** After a season of derivative snapshots: grade the curve against Pinnacle's closing
derivative lines (the carry between lines is what the price bets rely on), test the model's
disagreement on first-half lines against the opener the way the full-game openers are
tested, and add first-quarter lines if the half curve holds.

## Graded against Pinnacle (NFL 2024-2025, Oct 31, 2026)

Every record above is graded against the nflverse consensus line, and closing line value
against the consensus close. `experiments/pinnacle_grade.py` regrades the board's edges against
Pinnacle's real numbers, from the backfilled lines in `data/market/history/nfl/` (README
"Market lines"): an "open" snapshot each Monday 12:00 UTC (Pinnacle opens Sunday evening, so
this is the line a half day in, not the first number) and a close ten minutes before each
cluster of kickoffs. Two seasons, 570 games. Results in `experiments/out/pinnacle_grade_nfl.json`.

**The consensus close the backtests use is close to Pinnacle's, not the same:** 0.27 points off
on spreads and 0.33 on totals on average, a half point or more on 41% of games.

**Price tier** (a DraftKings or FanDuel price beating Pinnacle's no-vig number by
`sheet.PRICE_MIN`, 1%, with no model; one bet per game and side):

| Taken at | Bets | W-L | Return | EV claimed | Closing line value | Beat the close |
|---|---|---|---|---|---|---|
| Monday open | 84 | 51-33 | +21.9% ± 12.9% | +1.9% | +1.3% ± 1.4% | 63% |
| Close | 69 | 35-34 | −3.1% ± 11.4% | +2.1% | (the close itself) | |

At the open, by market: spreads 22 bets, CLV +1.9% ± 2.8%; totals 28, +3.4% ± 2.3%;
moneylines 34, −0.7% ± 2.2% (their +43% return is luck: they did not beat the close). The
closing line value roughly matches the edge claimed, which is what a real price edge looks
like, but 84 bets in two seasons is far too few to prove it, and there are only about 40 such
prices a season at these two books at those two moments. It stays the Bet sheet's first tier;
the live ledger, which sees every snapshot, will say more than this backfill can.

**Opener gap** (the raw model's side, run without the week's starters or injuries, at
Pinnacle's Monday line and price, against Pinnacle's close; games after a Monday night left out):

| Model off Pinnacle's Monday line by | Bets | W-L-P | Return | Line moved toward the model | CLV (EV) | Same games, consensus opener at −110 |
|---|---|---|---|---|---|---|
| Spread 1+ | 284 | 151-126-7 | +5.7% ± 5.7% | +0.31 pts | −1.2% ± 0.5% | +4.0% ± 5.8% |
| Spread 2+ | 159 | 85-68-6 | +7.5% ± 7.5% | +0.39 pts | −0.4% ± 0.8% | +8.7% ± 7.5% |
| Spread 3+ | 85 | 43-40-2 | +0.9% ± 10.5% | +0.17 pts | −1.5% ± 1.2% | −4.4% ± 10.2% |
| Total 2+ | 144 | 77-66-1 | +4.0% ± 8.0% | +0.65 pts | +0.8% ± 0.7% | (no consensus opening totals) |
| Total 3+ | 85 | 45-39-1 | +3.2% ± 10.4% | +0.64 pts | +0.7% ± 0.9% | |

The direction holds (Pinnacle's line moves toward the model after Monday, more on totals than
on spreads), and the records are positive, but none of it clears two standard errors, and on
spreads the move is too small to beat Pinnacle's price: closing line value is about zero. The
2+ point spread rule's record against Pinnacle (+7.5%) matches its record against the
consensus on the same games (+8.7%), so the old grading did not flatter it; it is simply not
proven on two seasons. Pinnacle's first number (Sunday evening) would show more of the move
than a Monday noon snapshot; the live market log has it from Oct 30, 2026.

What this changes: nothing on the board. It confirms that the NFL market is close to efficient
against the model, that a retail price better than Pinnacle's is the cleanest edge, and that
the opener edge is real in direction but small. Derivatives are not in it: the bulk historical
endpoint has no team totals or first-half lines. College (2024-2025, about 3,400 credits with
spreads and totals) is waiting for the next month's credits.

## Where the edge lives (Oct 31, 2026)

The walk-forward backtest split by segment, the way the college board's was (CFB_FEATURES.md,
"Where the edge lives"): `python experiments/nfl_segments.py`, full output, with the 2014-2018
tuning seasons, the closing-spread columns, the model's miss per segment and the Pinnacle
regrade, in `experiments/out/nfl_segments.md`; the segment definitions and the rule are in
the script's docstring. It rebuilds the board's own two runs and first reproduces the
headline numbers: 3,403 games, model miss 10.02 against the line's 9.91, the model 2+ off the
opening spread 982 bets 54.5% +4.0% ± 3.0% (615 since 2019, +4.1%; the same sides at the close
51.8%), wind unders 498-379-9, and the model edges since 2019 1,070 bets 51.0% −1.3%.

Buckets are the ones the board bets: the raw model 2+ off the opener (graded at the opener's
price, games after a Monday night left out, as the opener report does), the raw model 3+ off
the closing total (there are no historical opening totals), the wind under in 10-19 mph at
the closing total, and the market-adjusted spread bets with EV above 0 at the close (the
model tier). Holdout seasons, 2019 on (W-L, return per bet ± one standard error):

| Segment | Opener 2+ | Totals 3+ at close | Wind unders | Model edges |
|---|---|---|---|---|
| Matchup: Division | 121-109 +0.4% ± 6.1% | 79-70 +2.1% ± 7.9% | 75-65 +3.5% ± 8.2% | 199-176 +2.5% ± 4.9% |
| Same conference, not division | 119-101 +3.3% ± 6.3% | 79-59 +9.7% ± 8.0% | 100-62 +19.0% ± 7.4% | 199-199 -2.8% ± 4.8% |
| Other conference | 87-62 +11.1% ± 7.4% | 48-51 -6.5% ± 9.6% | 63-37 +21.3% ± 9.1% | 136-139 -4.2% ± 5.8% |
| Slot: Prime time (TNF, SNF, MNF) | 61-59 -3.0% ± 8.7% | 41-28 +14.2% ± 11.4% | 43-24 +23.8% ± 11.3% | 101-96 -0.6% ± 6.9% |
| Sunday day | 231-192 +4.2% ± 4.5% | 152-139 +0.4% ± 5.6% | 175-124 +13.0% ± 5.5% | 389-381 -2.1% ± 3.4% |
| Saturday | 16-12 +8.9% ± 17.6% | 8-6 +9.8% ± 26.4% | 15-10 +14.8% ± 18.5% | 18-19 -5.8% ± 15.3% |
| Neutral site | 11-4 +37.8% ± 21.3% | 2-4 -35.2% ± 41.0% | 1-4 -61.3% ± 38.7% | 18-8 +34.3% ± 17.9% |
| Other (holiday day games, Fri, Tue, Wed) | 8-5 +18.0% ± 27.0% | 3-3 -3.1% ± 43.4% | 4-2 +28.8% ± 40.8% | 8-10 -11.3% ± 20.1% |
| Rest: Short week (either team 4-5 days) | 22-21 -1.9% ± 14.8% | 14-7 +27.4% ± 20.2% | 12-8 +15.7% ± 21.7% | 30-36 -11.7% ± 11.8% |
| Normal | 218-195 +0.8% ± 4.5% | 150-133 +1.8% ± 5.7% | 155-118 +9.7% ± 5.8% | 376-365 -1.7% ± 3.5% |
| Long (either team 10+ days) | 87-56 +15.9% ± 7.7% | 42-40 -0.9% ± 10.7% | 71-38 +25.2% ± 8.6% | 128-113 +2.9% ± 6.1% |
| Rest gap: Home 2+ days more rest | 42-30 +10.8% ± 10.8% | 25-17 +14.7% ± 14.8% | 35-19 +23.8% ± 12.0% | 60-67 -8.1% ± 8.4% |
| Within 1 day | 242-213 +1.6% ± 4.3% | 160-140 +2.4% ± 5.5% | 169-126 +10.6% ± 5.6% | 410-399 -1.8% ± 3.3% |
| Away 2+ days more rest | 43-29 +14.2% ± 11.1% | 21-23 -7.4% ± 14.8% | 34-19 +24.2% ± 12.9% | 64-48 +10.6% ± 9.1% |
| Roof: Dome or closed | 85-71 +4.0% ± 7.4% | 64-56 +2.4% ± 8.6% | - | 156-147 -0.8% ± 5.5% |
| Retractable, open | 9-3 +43.1% ± 24.9% | 4-2 +29.5% ± 41.0% | - | 12-6 +29.4% ± 22.2% |
| Outdoors | 233-198 +3.1% ± 4.5% | 138-122 +2.1% ± 5.9% | 238-164 +14.3% ± 4.7% | 366-361 -2.2% ± 3.5% |
| Wind: Indoors (incl. open roof) | 94-74 +6.7% ± 7.1% | 68-58 +3.7% ± 8.4% | - | 168-153 +0.9% ± 5.3% |
| Outdoors, under 10 mph | 143-117 +4.9% ± 5.7% | 75-68 +0.6% ± 8.0% | - | 199-205 -4.5% ± 4.7% |
| Outdoors, 10-19 mph | 61-57 -1.3% ± 8.7% | 53-35 +16.4% ± 10.2% | 238-164 +14.3% ± 4.7% | 119-104 +4.0% ± 6.4% |
| Outdoors, 20+ mph | 5-1 +50.6% ± 28.2% | 2-8 -61.8% ± 25.5% | - | 13-8 +19.6% ± 21.0% |
| Outdoors, no wind recorded | 24-23 -2.4% ± 13.8% | 8-11 -19.7% ± 22.2% | - | 35-44 -13.8% ± 10.8% |
| Season part: Weeks 1-4 | 89-75 +3.6% ± 7.2% | 52-56 -7.6% ± 9.2% | 46-30 +16.6% ± 10.9% | 114-120 -5.9% ± 6.2% |
| Weeks 5-9 | 78-71 -0.1% ± 7.6% | 48-44 +0.3% ± 10.0% | 59-45 +10.2% ± 9.4% | 161-130 +7.1% ± 5.6% |
| Weeks 10-14 | 71-57 +5.5% ± 8.1% | 57-41 +12.3% ± 9.6% | 73-37 +27.7% ± 8.6% | 126-132 -5.1% ± 5.9% |
| Weeks 15-18 | 74-58 +7.1% ± 8.1% | 43-35 +5.6% ± 10.9% | 45-44 -2.7% ± 10.3% | 114-114 -3.0% ± 6.2% |
| Playoffs | 15-11 +10.3% ± 18.9% | 6-4 +15.4% ± 31.4% | 15-8 +24.6% ± 18.8% | 19-18 -0.2% ± 16.2% |
| Line size: Pick'em to 3 | 110-98 +1.1% ± 6.5% | 68-77 -9.7% ± 7.9% | 92-52 +22.8% ± 7.7% | 187-188 -3.1% ± 4.9% |
| 3.5 to 7 | 130-102 +6.7% ± 6.0% | 91-65 +12.1% ± 7.6% | 94-68 +12.4% ± 7.5% | 218-197 +1.6% ± 4.7% |
| 7.5 to 10 | 50-43 +2.4% ± 9.8% | 28-27 -2.4% ± 13.1% | 30-27 +1.5% ± 12.4% | 73-72 -2.6% ± 8.0% |
| 10.5+ | 37-29 +6.8% ± 11.4% | 19-11 +21.5% ± 16.7% | 22-17 +9.2% ± 15.6% | 56-57 -3.8% ± 9.0% |
| Favourite: Home favourite | 191-167 +1.8% ± 4.9% | 120-108 +1.1% ± 6.3% | 144-98 +14.7% ± 6.0% | 325-317 -1.8% ± 3.7% |
| Road favourite | 136-105 +7.8% ± 6.1% | 86-72 +4.7% ± 7.6% | 94-66 +13.6% ± 7.5% | 209-197 -0.4% ± 4.7% |
| Quarterback: Quarterback change | 119-91 +7.9% ± 6.4% | 62-50 +6.5% ± 9.1% | 52-41 +7.8% ± 9.7% | 149-130 +2.9% ± 5.6% |
| No change | 208-181 +2.1% ± 4.7% | 144-130 +1.0% ± 5.8% | 186-123 +16.3% ± 5.4% | 385-384 -2.8% ± 3.4% |
| **All games** | **327-272 +4.1% ± 3.8%** | **206-180 +2.6% ± 4.9%** | **238-164 +14.3% ± 4.7%** | **534-514 -1.3% ± 2.9%** |

**The rule** (set before running): a segment is a lead only if its return is above zero in
both periods, the holdout return is at least two standard errors above zero, and it is still
above zero against Pinnacle's lines on 2024-2025. 167 segment-by-bucket cells were tested;
with no edge anywhere about one would pass by chance, and about four would clear the holdout
bar alone. Ten clear the holdout bar and **four pass**:

| Bucket | Segment | Holdout | 2014-2018 | Pinnacle 2024-25 | Holdout vs rest of bucket |
|---|---|---|---|---|---|
| Opener 2+ | Long rest (either team 10+ days) | 87-56 +15.9% ± 7.7% | 50-38 +8.7% ± 10.2% | 29-18 +19.1% ± 13.9% | +15.4% ± 8.8% |
| Wind unders | Weeks 10-14 | 73-37 +27.7% ± 8.6% | 53-26 +31.0% ± 10.4% | 17-11 +16.1% ± 18.0% | +18.5% ± 10.3% |
| Wind unders | Prime time | 43-24 +23.8% ± 11.3% | 18-16 +3.0% ± 16.9% | 11-10 +0.8% ± 21.5% | +11.5% ± 12.4% |
| Wind unders | No quarterback change | 186-123 +16.3% ± 5.4% | 119-98 +7.0% ± 6.6% | 39-33 +4.4% ± 11.3% | +8.5% ± 11.1% |

Reading: **none of the four is evidence that the edge lives somewhere in particular.**

- The three wind passes are the wind under itself. The bucket is +14.3% ± 4.7% on the holdout,
  so most of its large segments clear two standard errors without being different from the
  rest; none of the three beats the rest of the bucket by two standard errors. Weeks 10-14 is
  the most consistent (both periods and Pinnacle), and weeks 15-18 the weakest (−2.7% on the
  holdout, 8-20 at Pinnacle), which would fit a market that prices December weather better,
  but that is a story fitted to subsets, not a test.
- The long-rest opener bets (a team off a bye or ten days after a Thursday) are positive in 10
  of the 12 full seasons and whichever team the model backs (the rested one 57.7%, the other
  61.0%, holdout), so it is not a hidden rest feature. But it is 1.75 standard errors better
  than the rest of the opener bucket with 35 segments tried, which is what the best of 35
  looks like by chance, and 31-15 of its holdout record is 2024-2025.
- The Pinnacle check is not a fresh sample: 2024-2025 are the holdout's last two seasons,
  regraded at better lines. It confirms direction on the same games.
- New from the regrade: **the wind under itself went 52-52 at Pinnacle's close in 2024-2025**
  (−3.5% ± 9.3%; 53-52 at the consensus close), the only two-season stretch since 2014 below zero.
  Two seasons is within its noise, but it is worth reading with the forecast grade
  (`experiments/wind_forecast_grade.py`) after this season.
- Splits that look large and do not replicate: neutral-site and open-roof games (dozens of
  bets), opener bets in other-conference games (+11.1% holdout, +10.3% tuning, but 1.5
  standard errors), quarterback-change games (opener 2+ +7.9% holdout, −7.3% on 2014-2018).
  The model edges are flat to negative in almost every segment on the holdout.

What this changes: nothing on the board. No tier is restricted or extended. The two worth
watching, long-rest opener bets and wind unders in weeks 10-14, are named here so the live
ledger can be read for them after the 2026 and 2027 seasons without re-searching.

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
- **Derivatives against their own closing lines.** The market log has carried team totals
  and first-half lines since Oct 30, 2026; after a season, grade the curve's carry between
  lines and the model's first-half disagreement against Pinnacle's derivative close (above).

## Referee crews and Week 18 (Nov 1, 2026)

`python experiments/referee_week18_test.py` (results in `experiments/out/referee_week18.json`).

**Referee crews: nothing.** For each game, the referee's earlier games give his crew's average
total against the closing total and home margin against the closing spread, shrunk toward 0 by
30 games. Added to the board's models alone: totals tuning +0.0035 (worse), holdout −0.0005;
margin tuning −0.0008, holdout +0.024 (worse). Not kept. Against the close the crew's past
residual predicts nothing (correlation +0.013 ± 0.019 tuning, −0.038 ± 0.023 holdout), and
betting the over or under whenever a crew leaned a point or more went 734-804. The market prices
crews, or there is nothing to price.

**Week 18: the market knows something the model does not.** Regular-season Week 18 games
(2021 on, 80 games): the raw model missed the margin by 9.84 points, the closing line by 8.87, a
gap of about a point, against 0.11 in weeks 1-17 (10.15 against 10.04). On totals the model was
no worse (10.34 against 10.57). The blend weight that would have minimized the miss in Week 18
is 0 (the line alone), against 0.07 on weeks 1-17 of the same seasons. Starters rested and
teams with nothing to play for are news the play-by-play cannot see. There are no Week 18
games in the tuning seasons, so the usual rule cannot be applied; the gap is about 2.5 standard
errors. **Done (Nov 1, 2026, `config.FINAL_WEEK_MARKET`):** Week 18 spreads and moneylines are priced
from the market alone (blend weight 0), and the model's Week 18 sides stay off the Bet sheet's
opener and model tiers and the line-movement alerts; price-tier bets still count, and totals are
unchanged. The old final week (Week 17, 2011-2020, 160 games) showed nothing (model minus line
−0.19 ± 0.31), so it applies to Week 18 of the 17-game seasons only. Every other regular-season
week since 2011, for comparison: +0.13 ± 0.04. Eighty games found after looking: the live record should confirm it.

**Already tested earlier, not re-run:** cold weather for totals (FEATURES.md, Totals: no), a
team's pace and neutral-situation pass rate for player volume (PLAYERS.md: no gain), opening
totals against Pinnacle's Monday number (above, "Graded against Pinnacle": total 2+ +4.0% ±
8.0%, not significant). Rain and snow need a precipitation history the repo does not have yet.

## Moved lines: is the curve calibrated? (Nov 1, 2026)

The Parlays page prices every spread and total within 7 points of the market off the key-number
curve. `python experiments/alt_calibration.py` (results in `experiments/out/alt_calibration.json`)
scores it walk-forward: each season's curve fitted on earlier seasons, centred on the closing
line, every played game at the line moved 1 to 7 points each way in half points (NFL 2014 on,
194,351 lines; college 2017 on, 399,183), pushes left out.

**The shape is right.** In every band from 20% to 80% the share that covered is within about a
point and a half of what the curve said, on spreads and totals in both leagues, and the gap does
not grow as the line moves further from the market: a moved line costs what the curve says it does.

**There is a lean, and it does not last.** Home sides and overs won about a point less often than
a 50/50 split at the consensus close implies, on every band, which looked like a correction to
make. By period it is not stable: NFL home sides −1.0 points to 2018, −2.5 in 2019-2022, +0.4 since
2023; NFL overs −2.0, −1.8, +0.8; college overs −2.6, −1.4, +0.2. The consensus close used to lean
toward home teams and overs and no longer does. **Nothing changed**: the parlay legs keep the curve
as it is, centred on the sharp book's own no-vig price.

## Rain and snow (Nov 1, 2026)

`data/cache/weather_history.csv` (the **Weather history** Action, Open-Meteo's free archive)
holds the rain (mm), snow (cm) and temperature during 2,372 of the 2,397 outdoor home games
since 2014; `python experiments/precip_test.py` (results in `experiments/out/precip_test.json`).
Like the wind test, it uses the weather that fell, not a forecast, so it flatters both checks.
(First run on 2,187 games; rerun on the complete history the same day, numbers below. The
conclusions didn't change.)

| Added to the total model | Tuning (to 2018) | Holdout (2019 on) | Kept by the rule |
|---|---|---|---|
| Rain, mm during the game | −0.0079 | −0.0261 | passes |
| Snow, cm | +0.369 | +0.008 | no |
| Wet (2+ mm of rain or any snow) | +0.008 | −0.043 | no |

| Unders at the closing total | Tuning | Holdout | Points under the line |
|---|---|---|---|
| Wet (2+ mm or snow) | 51-33, +15.9% | 90-59, +15.3% | −2.7 / −2.4 |
| Rain 5+ mm | 9-6 | 33-17, +26.0% | −4.2 / −5.5 |
| Snow | 17-10 | 23-21, −0.2% | −1.7 / −0.5 |

Rain helps the total model a little on both periods, and the closing total does not fully price
heavy rain: wet-game unders went 141-92 (60.5%, about +15% ± 6%) over both periods. Snow is
already priced. **Not on the board yet**: the live board would need a precipitation forecast
(the Open-Meteo forecast it already calls for wind carries one), and as with the wind under, the
forecast version has to be logged and graded before it is trusted. Rain and wind overlap
(stormy games are both); the next step is a wet under tested alongside the wind under.

**Logging from Nov 1, 2026.** The board's Open-Meteo forecast call now also returns rain and snow,
summed over the kickoff hour and the three after it (`weather.GAME_HOURS`), logged with the wind
in `data/wind_log.csv` (`precip_mm`, `snow_cm`). Games forecast wet (2+ mm or any snow) carry a
"Rain N mm" or "Snow forecast" tag on the NFL board's game cards; nothing is staked on it.
`python experiments/precip_forecast_grade.py` grades the wet under on those forecasts. Rule, set
now: it joins the Bet sheet only if, after a full season of logged forecasts, the forecast version
is positive with closing line value above zero, the same bar the wind under has to clear.


## Staking rules from the audit (Nov 1, 2026)

Two rules, approved by Ryan, in `nflmodel/sheet.py` (both boards):

- **Wind unders at half stake** (`WIND_STAKE_SCALE = 0.5`). Their record was measured on wind
  recorded at the game; live they are bet on forecasts, and in 2024-2025 they went 52-52 at
  Pinnacle's close. Half stake until `experiments/wind_forecast_grade.py` has a full season.
- **A closing-line-value guard** (`clv_guard`, `CLV_GUARD_N = 50`, `CLV_GUARD_Z = 2`). Any kind of
  bet on the Bet sheet (price, opener, wind, model) whose live ledger shows 50 or more graded
  bets with closing line value, averaging below zero by more than two standard errors, carries
  no stake: the bets stay on the sheet, marked, and in the ledger, so the kind is still graded and
  comes back once it recovers. `meta.muted_tiers` names the kinds and the numbers; the Bet sheet
  says so. As of this change no kind has 50 graded bets yet.

## Closing line value dashboard (Nov 1, 2026)

The Model record page of both boards now opens with a closing-line-value section: how many graded bets
have a closing value, the average with its standard error, the share that beat the close, a table by kind
of bet and by market with each kind's stake-rule status (watching until 50 bets, staked, half stake for
wind unders, or no stake while the CLV guard has it muted), and a running-average chart per kind.

College bets had almost no closing value because Pinnacle rarely quotes college games. `closing_quotes`
now falls back to DraftKings' and then FanDuel's closing no-vig number for any game Pinnacle never quoted,
and records which book closed it (`close_book`). Those books' closes are softer than Pinnacle's, so college
CLV is a weaker signal than NFL CLV. Rows already graded stay blank; the ledger log is append-only.

## Feed for the matchup dashboards (Nov 1, 2026)

`nflmodel/dashfeed.py` writes `docs/feed/nfl.json` and `docs/feed/cfb.json` (about 50 KB and 220 KB)
for rhyman7/nfl and rhyman7/cfb, which read them from the same site:

- each game's projected line and total (the board's final projection) with the market's, keyed by
  the dashboards' game id (nflverse id for NFL, ESPN id for college), and the bets on its Bet sheet;
- each player's projection, keyed `TEAM|name` (lower case, no punctuation or Jr./III), with the
  books' main prop line for NFL players.

The dashboards show the line and bets on their game board and matchup header, and use the player
projections for the Weekly Edges prop tables in place of their old "opponent allowed ÷ average"
multiplier (raw position splits overstated matchups several times over; see the defense-vs-position
test above). Both pipelines write the feed after data.json, and the props run rewrites the NFL feed
with new lines. If the feed is missing or for another week, the dashboards hide those pieces.

## Capping long runs in yards per carry (Nov 1, 2026): no

`python experiments/capped_runs_test.py` (results in `experiments/out/capped_runs_test.json`). Every run
counted at no more than its season's 97.5th percentile (19-21 yards), scaled so the league average is
unchanged, as the input to each player's yards per carry and each defense's allowed. Keep rule set
first: holdout miss on rushing yards (RB and QB rows) better by 2 paired standard errors, tuning no
worse. Result: worse in both. Holdout 13.287 → 13.334 yards (+0.047 ± 0.008), tuning +0.012 ± 0.008;
RBs alone +0.037 ± 0.013 holdout. Not kept. The player model already shrinks yards per carry toward the
position average (60 carries of prior), which takes care of a few long runs; capping on top throws away
real information (who breaks long runs keeps doing it). The matchup sites still show capped yards on
the team cards as context, labeled as such.

## College prop history (Nov 1, 2026)

Probe first (`experiments/cfb_props_probe.py`, 431 credits): on Saturday Nov 1, 2025, 27 of the 43
college games in The Odds API's archive had DraftKings or FanDuel player props, eight markets each
(passing yards and TDs, rushing yards, receiving yards, anytime TD, and the three yardage ladders;
no receptions or attempts). `nflmodel/cfbpropbackfill.py` (workflow **College prop backfill**) saves
one snapshot per game ten minutes before kickoff under `data/props/history_cfb/<season>/`, newest
weeks first; about 2,200 credits a week. Ryan approved up to 32,000 credits for 2025. The grading
(board projections against these lines and the results) comes after.

## College props against real lines (Nov 1, 2026)

`python experiments/cfb_props_real_lines.py` (results in `experiments/out/cfb_props_real_lines.json`).
450 games of 2025 college props (DraftKings and FanDuel, ten minutes before kickoff), the college
board's walk-forward player projections, and the NFL prop distributions the Parlays page uses for
college legs. Rules were written into the script before it ran.

- **Main lines (4,328 two-sided props): the books win.** Brier: model alone 0.2551, books 0.2489
  (model worse by 0.0062 ± 0.0017). The NFL nudge (books moved 20% toward the model, at most 3
  points) came out 0.0004 ± 0.0003 better than the books alone: under the 2-standard-error bar.
  Verdict: college props are priced off the books alone; the model gets no nudge.
- **Ladders (67,061 over-only rungs, 403 games): the NFL reading of rung prices doesn't hold.**
  College rungs cleared 1-3 points less than the NFL reading says in 4 of 10 bands (outside 2
  standard errors), and raw implied odds overstate them by 3-8 points. A college reading fitted on
  these rungs (`cfb_reading_2025` in the results) is the starting point; it is checked on 2026's
  rungs before it's trusted. The model's own rung chances were worse than the NFL reading.
- **Anytime TD (7,979 props): a tie once the books are read properly.** With the NFL's assumed 10%
  margin the model looked 0.0058 better, but the books' college TD prices carry about 17.5%. Read
  at their best (in-sample, generous to the books), model minus books is -0.0004 ± 0.0008.
  Corrected the same way, the model ranks scorers slightly better (-0.0022 ± 0.0009), measured
  in-sample: a lead to recheck on 2026's live prices, not a finding.

So the "not market-tested" college legs on the Parlays page (the model alone) are the weakest
number available. College props should come from live book prices: two-sided lines at the books'
no-vig chance, ladders through a college reading, anytime TD at about a 17.5% margin.

## Live college prop prices (Nov 1, 2026)

`nflmodel/cfbprops.py` (workflow **College props**) pulls DraftKings and FanDuel college player props
for games in the next 40 hours every morning (12:13 UTC) and again before each Saturday kickoff
wave (16:13, 20:13 and 23:13 UTC, games in the next 5 hours), about three snapshots a game and
800-1,000 credits a week. One file per game under `data/props/history_cfb_live/<season>/`, rewritten
each snapshot, so the last one before kickoff is kept for grading 2026 (the ladder reading and the
anytime-TD ranking lead in "College props against real lines" get rechecked on it).
`parlays.cfb_book_legs` turns them into Parlays-page legs exactly as that test found: books' no-vig
chance on two-sided lines, `CFB_RUNG_CAL` on alternate rungs, `CFB_TD_MARGIN` (17.5%) on anytime TD,
no model nudge, only lines the books post. The model's own college legs remain only for players
and stats the books don't post, still tagged "not market-tested".

## SRS (Simple Rating System) (Nov 2, 2026): no, with one college lead

`python experiments/srs_test.py` (results in `experiments/out/srs_test.json`). Classic SRS (margin
adjusted for opponents, every game this season equal, no home field, no shrinkage) and OSRS/DSRS,
recomputed before each week from that season's earlier games, added to the walk-forward game models.
Rule, set first: lower the average miss in both periods, the holdout by 2+ standard errors.

| Change in average miss (points) | Tuning | Holdout |
|---|---|---|
| NFL margin, SRS added | +0.008 ± 0.013 | +0.008 ± 0.003 |
| NFL total, OSRS/DSRS added | +0.006 ± 0.005 | +0.002 ± 0.005 |
| NFL margin, points rating swapped for SRS | +0.035 ± 0.022 | +0.055 ± 0.020 |
| College margin, SRS added | +0.002 ± 0.006 | +0.003 ± 0.002 |
| College total, OSRS/DSRS added | +0.009 ± 0.014 | −0.017 ± 0.007 |
| College total, points rating swapped for OSRS/DSRS | +0.014 ± 0.021 | −0.046 ± 0.013 |

Not kept anywhere: the board's points rating (ratings.py: home field, recency, last season at lower
weight, shrinkage) already carries what SRS does, and swapping it out makes NFL margins worse. The
one lead: college totals did better since 2021 with this-season-only scoring ratings (OSRS/DSRS),
though not on 2017-2019. A guess worth testing on its own: the college points rating leans on last
season too much for totals in the transfer-portal years. That needs its own test with a fresh
split (choose on 2021-2023, check on 2024-2025), not a decision from this one.

### Follow-up: college totals and last season's weight (Nov 2, 2026): not kept

`python experiments/cfb_totals_carry_test.py` (results in `experiments/out/cfb_totals_carry_test.json`).
Only the points rating feeding the college total model was rebuilt, with last season carried at
0.55 (current), 0.40, 0.25, 0.10 or 0 and a 9- or 15-game half-life. Rule, set first, on a fresh
split: choose on 2021-2023, keep only if it beats the current setting on 2024-2025 by 2 standard
errors. Chosen: carry 0 (9-game half-life), 0.044 ± 0.016 better on 2021-2023; on 2024-2025 0.028 ±
0.018 better (1.5 SE): **not kept**. Every lower carry did a little better since 2021 and made no
difference on 2017-2019, so the direction is consistent, but the size is small (0.03 points on a
12.6-point average miss). Recheck after 2026 with 2024-2026 as the check.

## Head-coaching changes (Nov 2, 2026): no, with one college lead

ELWAY and SP+ adjust for coaching changes; the board does not. NFL coaches from nflverse's
home_coach / away_coach; college from Wikipedia's FBS coaching-change tables, 2014-2025
(`data/cfb/coach_changes.csv`, every firing, interim and hire with its date; in `experiments/data/` until Nov 4, 2026).
Flags per team-game: **new** (a different head coach from the start of last season; about 21% of
NFL and 16% of college team-games) and **mid** (the coach changed earlier this season; about 2%).
Features: the new-coach difference all season and weeks 1-4 only, new coach times last season's
average margin (negated: does a new coach's team drift to the middle more than the ratings allow),
and the mid-season difference; for totals, the sums. `python experiments/coach_change_test.py`
(results in `experiments/out/coach_change_test.json`). Rule, set before running: kept only if the
average miss drops in tuning and holdout, the holdout by at least 2 paired standard errors.

| Added | NFL tuning | NFL holdout (2019 on) | College tuning (2017-19) | College holdout (2021-25) |
|---|---|---|---|---|
| Margin + new coach | +0.003 ± 0.004 | +0.002 ± 0.002 | +0.013 ± 0.010 | +0.005 ± 0.006 |
| Margin + new coach, weeks 1-4 | +0.002 ± 0.003 | −0.002 ± 0.002 | +0.002 ± 0.006 | −0.001 ± 0.001 |
| Margin + new coach × last season's margin | +0.011 ± 0.008 | −0.005 ± 0.003 | +0.005 ± 0.006 | **−0.026 ± 0.009** |
| Margin + mid-season change | 0.000 ± 0.005 | +0.002 ± 0.002 | −0.001 ± 0.003 | −0.010 ± 0.009 |
| Total + new coach | 0.000 ± 0.017 | +0.009 ± 0.012 | +0.004 ± 0.004 | −0.001 ± 0.002 |
| Total + mid-season change | +0.003 ± 0.004 | +0.003 ± 0.002 | +0.003 ± 0.004 | +0.002 ± 0.001 |

**Nothing is kept.** A new coach by itself tells the model nothing the ratings and the market
pull do not already carry, in either sport, and mid-season firings add nothing. The one lead is
college: a new coach's team regressing toward the middle beat the holdout by 2.7 standard errors,
but it was slightly worse on 2017-2019, so it fails the rule. It may be real for the transfer-portal
era (rosters now turn over with the coach), or a holdout fluke; it is on the near-miss list for the
2026 recheck and the January bundle test, not on the board.

## Playing surface, grass vs turf (Nov 2, 2026): no

nfelo adds a surface adjustment to home field when the visitors play at home on the other kind of
surface. From nflverse's per-game `surface` (gaps filled from the same stadium's other games; 4
games left unknown), each game is grass or turf, and each team's home surface is that of most of
its home games that season. Visitors were off their home surface in 53% of non-neutral games, and
44% of games were on turf. `python experiments/surface_test.py` (results in
`experiments/out/surface_test.json`). Rule, set before running: kept only if the average miss
drops in tuning and holdout, the holdout by at least 2 paired standard errors.

| Added | Tuning | Holdout (2019 on) |
|---|---|---|
| Margin + visitors off their home surface | +0.005 ± 0.006 | +0.008 ± 0.007 |
| Margin + split: turf team on grass / grass team on turf | −0.009 ± 0.012 | +0.004 ± 0.012 |
| Total + game on turf | +0.025 ± 0.009 | +0.004 ± 0.003 |

**Nothing here, and not a near-miss.** Playing away from their home surface makes no difference to
visitors beyond ordinary home field, and turf adds nothing to totals beyond the indoor flag the
model already has.

## Turnover-free pass and rush ratings (Nov 2, 2026): no

ELWAY backs turnovers out of its efficiency ratings as mostly random. The board's pass and rush
ratings were rebuilt without turnover plays (NFL: interception or fumble lost; college: cfbfastR's
turnover_vec), same plays and ridge otherwise, then added next to the originals and swapped in for
them. `python experiments/turnover_free_test.py` (results in `experiments/out/turnover_free_test.json`;
the college plays come from `cfb_downs_havoc_test.py`'s cache). Rule as usual: lower average miss
in tuning and holdout, the holdout by 2 paired standard errors. Seasons through 2025.

| Change | NFL tuning | NFL holdout (2019-25) | College tuning | College holdout (2021-25) |
|---|---|---|---|---|
| Margin, added | −0.004 ± 0.006 | +0.001 ± 0.009 | +0.003 ± 0.006 | −0.005 ± 0.006 |
| Margin, swapped in | −0.010 ± 0.008 | −0.003 ± 0.005 | +0.003 ± 0.006 | −0.003 ± 0.004 |
| Total, added | +0.003 ± 0.004 | −0.003 ± 0.002 | +0.001 ± 0.006 | 0.000 ± 0.003 |
| Total, swapped in | +0.005 ± 0.005 | −0.002 ± 0.003 | +0.001 ± 0.010 | +0.003 ± 0.007 |

**Nothing here, and not a near-miss.** With turnovers in or out, the ratings predict the same, in
both sports. This agrees with the rejected fumble-luck test: once plays are clipped and the points
and market ratings are in the model, turnover luck is not something the board is missing.

*Nov 4, 2026: re-judged under the keep rule and applied.* The NFL margin swap is better on every
measure, however small, so it went on the board (see "Re-judging past tests under the keep rule",
"Applied"). The counts now live in `data/cache/team_extra_<season>.csv` (`pnt_n, pnt_epa, rnt_n,
rnt_epa`, written by `nflmodel/extra.py`); the script reads them from there.

## Totals at the opening total (Nov 2, 2026): a lead, not proven

Until now NFL totals were graded only at the close (no historical opening totals). nfelo's games
file, already downloaded for the opening spreads, has opening totals with prices from 2024 (513
games through early 2026 with both a result and a close; teams off a Monday night left out, as for
the opening spreads). `python experiments/open_totals_test.py` (results in
`experiments/out/open_totals_test.json`). Two runs from `nfl_segments.py`: "early" (quarterbacks
blind: what is known when the opener is posted) and the full run. "Moved" is how far the total
went from open to close toward the model's side, in points. Rule, set before running: a lead (for
the live log, not the board) only if, at 3+ points, the opener return is above zero and the line
moves toward the model by more than two standard errors.

| Model off the opening total by | Early run: record, return | Moved toward model | Full run: record, return | Moved |
|---|---|---|---|---|
| 1+ | 176-148-3, +3.5% ± 5.2% | +0.50 ± 0.07 | 188-161, +2.7% ± 5.1% | +0.47 ± 0.07 |
| 2+ | 103-78-1, +8.4% ± 7.0% | +0.64 ± 0.10 | 109-88, +5.3% ± 6.8% | +0.69 ± 0.10 |
| 3+ | 58-47-1, +5.0% ± 9.2% | +0.66 ± 0.13 | 62-44, +11.3% ± 9.2% | +0.82 ± 0.13 |
| 3+, games under 10 mph wind | 41-38-1, −1.4% ± 10.6% | +0.64 ± 0.14 | 51-34, +14.3% ± 10.2% | +0.72 ± 0.15 |
| 4+ | 32-21-1, +14.7% ± 12.7% | +0.79 ± 0.21 | 29-20, +12.6% ± 13.5% | +1.02 ± 0.22 |
| Same 3+ sides at the close | 52-53, −5.3% | | 56-50, +1.0% | |

Average miss: model 10.16 (early) / 10.12 (full), opening total 10.10, closing total 10.06.

**It passes the lead rule.** The total moves toward the model's side between open and close, by
about two-thirds of a point at a 3+ gap, five standard errors from zero, and in games without wind
too (so it is not the recorded wind leaking in). The same bets at the close are flat, which is the
pattern of an opener edge: the market comes to the model's number, then prices it. **Not proven
as profit**: 106 bets is far too few (the returns are 0.5 to 1.2 standard errors), and the full
run uses news from later in the week that the opener did not have. Like the opening spreads, this
belongs in the live log (the board already records each game's opening total) before anything
changes on the board.

## Blending with nfelo's model (Nov 2, 2026): passes on paper, not trusted

nfelo publishes, per game since 2009, its Elo difference at the opener (team Elo plus home field,
bye, divisional, surface, time-zone and quarterback adjustments, regressed part-way to the
market). `python experiments/nfelo_blend_test.py` (results in `experiments/out/nfelo_blend_test.json`)
blends it (nfelo_dif_open / 25, nfelo's points conversion) with the board's early run, weight
chosen on 2014-2018, graded as the opening spreads are (2,681 games, Monday games left out). Rule,
set before running: a lead if the blend lowers the holdout miss by 2+ paired standard errors and
its 2+ opener bets return more than ours.

| Holdout, 2019 on | Average miss | 2+ off the opener: record, return | Line moved toward it |
|---|---|---|---|
| Board's early run | 10.12 | 327-272-16, +4.1% ± 3.8% | +0.55 |
| nfelo | 9.93 | 110-78-3, +11.6% ± 6.8% | +1.53 |
| Blend (95% nfelo, chosen 2014-18) | 9.93 (−0.19 ± 0.05 vs ours) | 120-84-3, +12.2% ± 6.5% | +1.51 |
| Opening spread | 9.96 | | |

It passes the rule, and is **not trusted**, for three reasons found on checking:
1. **nfelo knows the starting quarterback.** Its QB adjustment uses who actually started, which at
   the opener is unknown when a team changes starters during the week. In those games nfelo
   misses by 10.27 against the opener's 10.42; in the 2,085 games with no change it misses 9.83
   against 9.82, no better than the opener. The 1.5-point line moves toward nfelo are mostly
   that news arriving.
2. **nfelo is regressed toward the opening line**, so beating our model's miss is largely the
   market beating it (our early run misses 9.93 in no-change games, the opener 9.82).
3. **In-sample:** nfelo's file is rebuilt with settings tuned on these same seasons.

What is left, games with no quarterback change: nfelo 2+ off the opener 65-47-2, +10.6% ± 8.8%
(1.2 standard errors), the line moving 1.1 points toward it. Worth a clean check, which only a
forward record can give: save nfelo's numbers each week before the games, as published.

### nfelo as published (Nov 2, 2026): no edge

nfelo also archives the spreads it published before each game, 2021 on (saved weekly to
`data/nfelo_log/` by `nflmodel/nfelolog.py`): its model line before the pull toward the market,
with the quarterbacks it assumed at the time. These are a fair forward record.
`python experiments/nfelo_published_test.py` (results in `experiments/out/nfelo_published_test.json`),
1,515 games. Rule, set before running: a lead only if nfelo's 2+ bets return more than zero by two
standard errors at the close, or at the opener with the line moving toward it.

| 2021 to now | Average miss | 2+ off the opener | 2+ off the close |
|---|---|---|---|
| nfelo, published (before its market pull) | 9.94 | 313-271-17, +2.3% ± 3.8% | 323-296-11, +0.1% ± 3.8% |
| Board's early run | 10.04 | 298-265-13, +1.1% ± 3.9% | 291-304-15, −5.9% ± 3.9% |
| Opening / closing spread | 9.85 / 9.73 | | |

**No lead.** As published, nfelo is a little more accurate than our model and less accurate than
the opener, and its bets are about break-even: strong in 2021-2022 (+19% and +16% at the opener),
negative in 2024-2025. The 2019-on result from its rebuilt file (+12%) was mostly knowing the
quarterback who started. The weekly save continues as a cheap record; the blend is not pursued.

## Pass rate over expected (Nov 2, 2026): no for totals and pass attempts; tiny gain on carries

PROE is how often a team passes compared with what down, distance, field position, score and
clock say an average team would (nflfastR's xpass); nfelo shows it, the board did not use it.
Per team-game in neutral situations (downs 1-3, win probability 0.20-0.80, more than two minutes
left in the half), 2012 on, rated as of each game like the volume ratings (recent games weighted,
last season at half, shrunk by 4 games): each offense's PROE and each defense's PROE allowed.
`python experiments/proe_test.py <pbp dir>` (results in `experiments/out/proe_test.json`). Rule,
set before running: lower miss on 2014-2018 and on 2019 on, the holdout by 2+ standard errors.

| Added | Tuning (2014-18) | Holdout (2019 on) | Kept |
|---|---|---|---|
| Team pass attempts (volume model; miss 6.18) | −0.027 ± 0.010 | −0.006 ± 0.008 | no |
| Team carries (miss 5.59) | −0.003 ± 0.007 | −0.015 ± 0.006 | passes |
| Game totals (s_proe; miss 10.47) | −0.003 ± 0.016 | +0.018 ± 0.008 worse | no |

**Little here.** A pass-happy offense does not score more than its efficiency ratings already say,
and the volume model's recent pass attempts already carry the tendency. The carries gain passes
the rule but is 0.015 carries on a 5.6-carry miss (0.3%), on the older chain projections; the
board shows the New projections for backs' carries, so it is not applied without a recheck there,
which at this size is not worth it.

*Nov 4, 2026: applied under the keep rule* in both volume regressions (see "Re-judging past tests
under the keep rule", "Applied"). PROE now lives in `data/cache/team_extra_<season>.csv` (`proe_n`,
`proe_oe`, `nflmodel/extra.py`), rated by `players.tendency`.

## nfelo's WEPA, weighted EPA (Nov 2, 2026): no

WEPA (github.com/greerreNFL/wepa) reweights each play's EPA by its kind (turnovers in part,
special teams at a learned weight, defensive plays boosted), refitted each season on earlier
seasons only; its point-in-time file gives every team's WEPA in every game since 2000 with no
look-ahead. Each team-game's offensive WEPA was added to the board's engine as one more rated
metric (opponent-adjusted, decayed, one observation per game), ridge chosen on the tuning
seasons. `python experiments/wepa_test.py` (results in `experiments/out/wepa_test.json`). Rule as
usual: lower miss on tuning and holdout, the holdout by 2+ paired standard errors.

| Change | Tuning (before 2019) | Holdout (2019 on) |
|---|---|---|
| Margin + WEPA rating | +0.002 ± 0.003 | +0.001 ± 0.004 |
| Margin, points rating swapped for WEPA | +0.036 ± 0.015 | +0.021 ± 0.016 |
| Total + WEPA rating | 0.000 ± 0.005 | −0.004 ± 0.006 |

**Nothing here, and not a near-miss.** WEPA adds nothing to the board's pass, rush, success-rate,
points and market ratings, and is a worse stand-in for the points rating, matching the earlier
turnover-free and special-teams results.

## Middles between DraftKings and FanDuel (Nov 2, 2026): never worth it

When the two books hang different numbers, betting the better side at each wins both if the result
lands between them and otherwise pays the vig. `python experiments/middles_test.py` (results in
`experiments/out/middles_test.json`, every opportunity in `middles_test.csv`), the saved 2024-2025
backfill (Monday-noon "open" and last-before-kickoff "close" snapshots), per $1 on each side.
Expected value uses Pinnacle's no-vig line at the same snapshot as the centre and how often past
games landed on each integer result around their line (NFL through 2023, college 2014-2023), so
key numbers count as often as they occur. Rule, set before running: worth flagging only if both
expected and realized are above zero, the realized by 2 standard errors.

| Gap between the books | NFL spreads (open / close) | NFL totals | College spreads | College totals |
|---|---|---|---|---|
| 0.5+ point: how often | 222 / 199 of ~570 games | 239 / 198 | 781 / 779 of ~1,540 | 692 / 660 |
| 0.5+: expected / realized per pair | −6.7% / −5.8% (open) | −6.8% / −9.3% | −7.0% / −6.8% | −6.7% / −8.0% |
| 1+ point: how often | 72 / 47 | 122 / 91 | 221 / 176 | 341 / 282 |
| 1+: expected / realized | −5.0% / −3.6% (open) | −6.1% / −8.9% | −5.9% / −7.3% | −6.1% / −8.3% |
| 1.5+ point: how often | 7 / 6 | 3 / 0 | 16 / 21 | 10 / 6 |

**No.** The two books disagree on about a third of NFL games and half of college games, but almost
always by half a point to a point, and a half-point middle lands about 1% of the time while each
pair pays two vigs. Expected value is negative at every gap, for every league and market, and the
realized returns agree. Gaps that straddle 3 or 7 essentially never happen (none in the NFL, one in
college). Middles need a wider book selection or a line that moves after you bet (the opener, then
the other side later), not two books at the same moment.

## College rushing projections: three tests and one kept (Nov 3, 2026)

Prompted by Messiah Burch at Miami (OH): chain 71.9, recent per-game average 103.6, projection
95.4, season average 83.7. Scripts in `experiments/`, results in `experiments/out/`.

**1. `cfb_rush_blend_test.py`: chain, recent average, their plain average or the projection,
closest to actual RB rushing yards?** 32,000 player-games 2017-2025 (chain or projection >= 20).
Mean absolute miss: chain 29.7 / 28.0 (tuning / holdout), recent 31.4 / 29.5, average 29.9 /
28.1, projection 30.3 / 28.6. The chain "wins" by 8 SE on the holdout while running 8.9 yards
low: rushing yards are right-skewed, absolute miss is minimised by the median, and a low
center is rewarded. Not a usable result on its own.

**2. `cfb_rush_center_test.py`: each number as the center of the prop distribution, Brier at the
2025 DraftKings/FanDuel rushing-yards main lines** (1,097 props, 386 games): chain 0.2853,
recent 0.2654, average 0.2650, projection 0.2608, the books 0.2500. The chain is the worst center
(5.4 SE worse than the projection, calls 68% of props unders when 46% of overs hit); the
projection is the best of the four and still 2.6 SE worse than the books, as on Nov 1. Nothing
changed. Burch's 95 stands, and his line is priced off the books.

**3. `cfb_rush_group_calib_test.py`: rushing calibration per position group. Kept.** The
calibration fitted one rushing-yards regression on everyone who appeared (mostly receivers and
quarterbacks with a few carries), settling on about 0.71 x chain + 0.43 x recent; running backs
alone want about 0.76 / 0.40, prop-relevant backs 0.82 / 0.30, stable every season 2019-2025.
Rule, written first: fit `rush_yds` and `carries` per group (RB; QB; WR+TE), keep only if RB
rushing-yards RMSE improves on tuning and holdout, the holdout by 2+ SE, and the Brier at the
2025 lines is no worse. Result: RMSE 40.154 -> 40.141 tuning (2.3 SE), 37.950 -> 37.937 holdout
(2.9 SE); bias +0.10 -> -0.01 on the holdout; Brier 0.2612 -> 0.2604 (1.5 SE better). Passes,
and the gain is small: about a yard on a lead back's projection. `cfbmodel.players.GROUP_STATS`,
`CALIB_GROUPS`.

**4. `cfb_rec_group_calib_test.py`: receiving calibration per group (WR; TE; RB). Kept, on a
corrected rule.** Receptions and receiving yards were fit on one pool too. First rule (both stats
must pass; TE and RB not worse; Brier no worse per group with 100+ props): failed on WR
receptions' holdout (1.2 SE) and on the RB receiving-yards Brier (0.4 SE worse on 129 props).
The rule was wrong about what the bet is: college books post no receptions lines, so
receptions are an input to the yards (the chain and the prop distribution's volume), not a
bet. Corrected rule, set before reading it against the same numbers: receiving-yards Brier at
the 2025 lines better by 2+ SE overall and no group with 300+ props worse. Result: overall
0.2580 -> 0.2561 (3.0 SE; 1,960 props), WR 0.2577 -> 0.2563 (2.6 SE), TE 0.2581 -> 0.2530
(3.5 SE); RB 129 props, below the floor. Backtest RMSE, holdout: WR receiving yards 34.17 ->
34.10 (4.7 SE), RB 22.95 -> 22.35 (6.8 SE) with the bias falling from +5.2 yards a game to
+1.5, TE 25.01 -> 24.97 (2.0 SE); WR receptions 1.2 SE, RB receptions 5.6 SE. Ryan's call,
Nov 3: accuracy decides. `CALIB_GROUPS` is now per stat; groups not listed get the pooled fit.

**NFL, same change, not kept** (`experiments/nfl_rush_group_calib_test.py`). On the NFL board
the linear calibration is live for quarterbacks' rushing; for running backs it is one input to
the boosted step. Rule: QB rushing RMSE better on tuning (2014-18) and holdout (2019+), the
holdout by 2+ SE; RB boosted RMSE and the Briers at the 2023-2025 closing lines no worse. QB
RMSE 16.116 -> 16.033 tuning (2.7 SE) and 17.829 -> 17.767 holdout (1.5 SE, short of 2); RB
boosted 32.290 -> 32.323 (0.6 SE, no change); Brier QB lines 0.2563 -> 0.2543 (1.5 SE), RB lines
0.2606 -> 0.2599 (0.8 SE). Every measure points the same way as college on a tenth of the
starting-QB sample; filed as a near-miss to re-run after the 2026 season.

**NFL receiving, same change, not kept: it is a small loss** (`experiments/nfl_rec_group_calib_test.py`).
Live for tight ends' receiving on the NFL board (WR and RB receiving come from the boosted step).
Rule: TE receiving-yards Brier at the 2023-2025 closing lines better by 2+ SE, or TE RMSE better
on both periods (holdout 2+ SE); nothing worse by 2 SE. TE receptions RMSE 2.215 -> 2.220 on
tuning (3.4 SE worse), receiving yards 28.82 -> 28.89 (2.2 SE worse); both flat on the holdout
(0.7 SE); Brier at the lines 0.2583 -> 0.2581 (0.7 SE); boosted WR and RB within 1.6 SE. The NFL
tight-end pool (about 2,500 tuning games against 14,000 college WR games) is too small to stand
alone and does better borrowing from receivers. Not a near-miss; the pooled fit stays.

## Niche props: probe and the tackle stat-crew check (Nov 2, 2026)

**What the books post** (probe, 320 credits: four 2025 week-1 games, the board's pricing snapshot
and the close, via the Prop price backfill Action):

| Market | DraftKings | FanDuel |
|---|---|---|
| Tackles + assists | ~8-9 players a game, both snapshots | none |
| Kicking points, field goals | 2 kickers a game, close only | none |
| Sacks | a few, close only | ~6 a game |
| Longest reception | ~10 a game | ~8 a game |
| Defensive interceptions | none | none |

**Tackles and the home stat crew.** Tackles are recorded by each home team's stat crew.
`python experiments/tackle_crew_test.py <stats dir> <pbp dir>` (nflverse weekly player stats and
play-by-play, 2016-2025): tackles per opponent play credited to *visiting* defenses in each stadium
(so the crew is what stays the same). The crew effect is real but modest: a stadium's factor in one
season correlates 0.42 with the next, and averaged over 2023-2025 runs from 4% below the league
rate (LAC) to 5% above it (NYG). It does **not** help project a player's tackles: scaling his
season average by the crew (prior two seasons, shrunk) made the average miss slightly worse at every
strength (+0.0003 ± 0.0006, +0.002 ± 0.001, +0.006 ± 0.002 on a 2.0-tackle miss, 2018-2025).
Rule, set before running: worth pricing tackle props on only if it helped by 2+ standard errors.
It does not, so no tackle-price backfill was bought.

## College props: where the gap lives, and college-fitted distributions (Nov 3, 2026): nothing kept

Two read-only steps from the research memo on getting the college player model past the books
(`claude/college-props-inputs-2026-11-03.md` in the Project). Same props, projections and chances
as "College props against real lines" above: the 2025 DraftKings/FanDuel main lines, the board's
walk-forward 2025 player table, the NFL prop distributions. Rushing yards 1,106 props, receiving
yards 1,960.

**1. `cfb_props_gap_diag.py`: which props carry the gap** (results in `experiments/out/cfb_props_gap_diag.json`).
Each prop's contribution is (model - hit)^2 - (books - hit)^2. At a main line the books' no-vig
chance is 50.0% on every prop, so "beating the books" here means only one thing: the model's lean
off the line has to be right more often than its size claims.

- **Over-confidence is the whole gap.** Reliability by band of the model's chance: rushing props the
  model put at 23% cleared 45%; at 74%, 44%. Receiving: 37% cleared 43%; 64% cleared 46%. The best
  single shrink of the model's chance toward 50% is 0.2 for both stats (in-sample), and even shrunk
  the model only ties the books (rushing -0.0008 ± 0.0008, receiving -0.0003 ± 0.0005). Where the
  model and books disagree by 5+ points, the model's side won 54.2% of 759 rushing props (2.3 SE
  above a coin) and 51.3% of 1,042 receiving props (nothing). The model's lean carries about a
  fifth of the information it claims.
- **Role disagreements carry most of it.** Props where the projection sits 30%+ from the line are
  16% of rushing props and 73% of the rushing gap (model 22% above the line on average; 40% of them
  cleared); 20% of receiving props and 65% of that gap (model 50% above the line; 49% cleared).
  Within 10% of the line the model is even with the books on rushing (-0.0031 ± 0.0038). When the
  model has a player for far more than the books do, the books are right.
- **Not the availability rule as such.** Props on players the model did not expect (post hoc: he
  appeared without playing the previous game) are 2-4% of props and carry none of the gap. Props
  where an expected regular teammate at the position sat out (post hoc) are about 20% of props and
  carry 34% of the rushing gap, 24% of receiving: more than their share, not most of it.
- **Rushing is an early-season problem** (weeks 3-8: 44% of props, 78% of the gap; weeks 9-16:
  +0.0040 ± 0.0049, even with the books). **Receiving is the reverse** (weeks 9-16: 76% of the gap).
  Quarterback and running-back rushing props carry the gap in proportion to their numbers.

**2. `cfb_propdist_test.py`: college-fitted prop distributions.** Rule, written first: refit the
PropModel widths per stat and position on the 2017-2024 college walk-forward table (mean held equal
to the projection) with the isotonic correction fitted on synthetic lines on the same seasons; keep
only if rushing and receiving yards each improve at the 2025 lines by 2+ SE over the NFL-fitted
chances and no stat is worse by 2 SE. Decision version: widths + correction.

| Stat | Books | Current (NFL-fitted) | College widths + correction (vs current) | College widths only (vs current, information) |
|---|---|---|---|---|
| Rushing yards | 0.2500 | 0.2604 | 0.2604 (0.0000 ± 0.0025) | 0.2582 (-0.0022 ± 0.0020, 1.1 SE) |
| Receiving yards | 0.2500 | 0.2561 | 0.2600 (+0.0038 ± 0.0022, worse) | 0.2545 (-0.0017 ± 0.0006, 3.0 SE) |
| Passing yards | 0.2500 | 0.2526 | 0.2512 (-0.0015 ± 0.0007, 2.3 SE) | 0.2506 (-0.0020 ± 0.0008, 2.5 SE) |
| Passing TDs | 0.2422 | 0.2439 | 0.2419 (-0.0019 ± 0.0012, 1.7 SE) | 0.2415 (-0.0024 ± 0.0014, 1.7 SE) |

**Fails the rule; nothing changes.** The isotonic correction, fitted on lines around the model's own
projection, makes the chances *more* extreme at the books' lines (receiving spread from 50%: 0.074
→ 0.112) and undoes the widths' gain. The widths alone point the right way on all four stats and
clear 2 SE on receiving and passing yards, not rushing (1.1 SE): a near-miss (#6 in the Project's
near-miss list), with a 2021-2024 fit giving the same picture. *Nov 4, 2026: the widths-only version
was applied under the keep rule (see "Re-judging past tests under the keep rule", "Applied", and PROPS.md
"College prop widths").* Even the best version is still
+0.0045 (receiving) and +0.0082 (rushing) behind the books: the distribution is not the main story.
The main story is the diagnostic's: the centre of the model's number is what has to improve, on
the props where it disagrees with the books by 30%+, which is a role and volume question
(candidates 1-5 in the memo), with early-season rushing roles (portal and recruiting priors,
candidate 6) moved up the list.

## College early-season role prior from CFBD background data (Nov 3, 2026): no

Candidate 1 of the college-props memo, aimed at the 78% of the rushing-yards prop gap that sits in
weeks 3-8. `python experiments/cfb_role_prior_test.py` (results in `experiments/out/cfb_role_prior_test.json`);
data from the new **CFBD player data** Action (`cfbmodel/cfbd_pull.py`, `data/cfb/cfbd/`: recruiting
ratings, transfer portal, team returning production, rosters, 2014-2026; CFBD roster ids are the
board's ESPN player ids, 98% matched).

A back's carry share starts each season from a flat prior (half the position average, worth one
game). The candidate replaced it with a ridge-fitted prior share for running backs: decayed history
entering the season and how much of it there is, 247 composite rating and stars, class year, portal
transfer, the team's returning rushing usage, rank among the team's backs; fitted walk-forward on
2015 to the season before. Weight w pseudo-games from {1, 2, 4, 8, 16}, chosen on tuning.

Rule, written first: RB rushing-yards and carries RMSE in weeks 1-8 better on the holdout (2021+)
by 2+ SE and not worse on tuning (2017-19); Brier at the 2025 rushing lines in weeks 3-8 better by
2+ SE.

- **The prior predicts the early-season share well on its own**: RMSE of a back's first-eight-weeks
  carry share 0.114 against 0.142 from his history alone and 0.140 from the flat prior (holdout).
  Rating carries most of the weight; stars are missing for 54% of back-seasons (walk-ons, older
  classes, JUCO), class year for 27%.
- **In the chain it does not help, at any weight.** w = 1 (the tuning pick): rushing RMSE 37.681 →
  37.670 (0.5 SE), carries 5.259 → 5.263 (worse, 1.3 SE). Every heavier weight is worse on both
  periods, by 2.4 SE (w = 2) to 9.6 SE (w = 16) on holdout rushing yards. Why: the shares are
  renormalised within the team, so a prior that lifts one back's share lowers his teammates', and
  the backs who already have history (most of the props) lose accuracy to those who don't; the
  calibration's recent-average term already does the early-season catching-up the prior was meant
  to do, and the chain's decayed history (last season at 60%) is a better prior than the fitted
  one for anyone who played last year.
- **Lines**: weeks 3-8 Brier 0.2677 → 0.2671 (-0.0006 ± 0.0009, 0.7 SE); all weeks 0.8 SE.
  Receiving, same features, as information: RMSE flat (1.2 SE worse on holdout), lines 0.7 SE worse.

Not kept, not a near-miss. The books' early-season edge on roles is not recruiting and portal data
applied this way. What the data may still be for: putting roster backs with no history into the
team's first games (today they get no row, and no projection, until they have played), which the
2025 lines cannot test since props start in week 3.

## College defense against each position (Nov 3, 2026): passes for five pairs, not past the books

Candidate 2 of the college-props memo. `python experiments/cfb_dvp_test.py` (results in
`experiments/out/cfb_dvp_test.json`): the NFL board's defense-against-position rating
(`nflmodel/dvp.py`: log of what a defense allowed to a position over what the projections expected,
6-game half-life, last season at half weight, shrunk by 4 games; projection x exp(b x rating), b by
least squares) on the college walk-forward projections, 2017 on.

**Stage 1, the NFL rating as is: worse on every pair but one, on the tuning seasons it was fitted
on too** (WR receiving yards 34.60 → 34.85 tuning, holdout 5.0 SE worse; RB carries 8.5 SE worse). The
NFL rating counts a player who was expected but did not appear as 0 yards allowed. With no college
injury report that is common, and the rating was measuring the opponents' absences, not the defense.

**Stage 2, rule written before it ran: the same rating from players who appeared.** Decision per
pair: RMSE better on the holdout (2021+) by 2+ SE and not worse on tuning (2017-19, where b is
fitted); then Brier for that group at the 2025 lines (b refitted on 2017-2024) no worse.

| Pair | b (2017-24) | Tuning RMSE | Holdout RMSE | Holdout z | 2025 lines, group z | Applied |
|---|---|---|---|---|---|---|
| WR receiving yards | 0.26 | 34.601 → 34.577 | 33.394 → 33.350 | -5.9 | -0.5 (1,442 props) | yes |
| WR receptions | 0.34 | 2.126 → 2.124 | 2.051 → 2.048 | -5.3 | no lines | yes |
| RB rushing yards | 0.27 | 40.192 → 40.128 | 38.022 → 37.975 | -3.0 | -0.5 (789) | yes |
| RB carries | 0.41 | 5.555 → 5.544 | 5.277 → 5.269 | -3.2 | no lines | yes |
| QB passing yards | 0.14 | 94.009 → 93.758 | 88.812 → 88.482 | -4.8 | -0.9 (628) | yes |
| TE receiving yards | 0.10 | 24.953 → 24.954 (worse) | 24.378 → 24.373 | -3.8 | -0.7 | no: tuning worse |
| RB receiving yards | 0.11 | 27.674 → 27.624 | 23.415 → 23.312 | -3.8 | **+2.0 worse** (126 props) | no: lines worse |
| TE receptions | 0.14 | | | -0.3 | | no |
| RB receptions | 0.37 | | | -1.5 | | no |

Five pairs pass; college passing yards passes where the NFL's did not. **The gains are small** (a few
hundredths of a yard on a 33-yard miss, as on the NFL board), and **against the books it is nothing**:
rushing yards -0.0004 ± 0.0007, receiving -0.0001 ± 0.0004, passing -0.0012 ± 0.0014. The books price
the matchup. It goes on the college board as a projection input and a card fact, on Ryan's go-ahead
(it changes the live projections slightly); it does not count toward closing the gap to the books.

## College targets from CFBD play stats (Nov 3, 2026): coverage check fails for 2021-2024; stop

Candidate 3 of the college-props memo. The college receiving chain runs on catch share because
cfbfastR's play-by-play names the receiver on too few incompletions from 2021 on. CFBD's own
per-play player stats (`/plays/stats`, pulled by the **CFBD player data** Action's `targets` mode
into `data/cfb/cfbd/targets_<year>.csv` and `types_<year>.csv`) record a "Target" row for the
intended receiver of an incompletion, separately from "Reception". Rule, written first: targets
over receptions must sit near the NFL's 1.5 in 2021-2025, or the test stops.

The endpoint returns about 2,000 rows a call, so a year-and-week call is a sample of 5-11 games a
week (about 135 games a season); the ratios are what matter:

| Season | Games sampled | Targets / receptions | Incompletions with a named receiver |
|---|---|---|---|
| 2021 | 129 | 1.32 | 56% |
| 2022 | 130 | 1.25 | 43% |
| 2023 | 134 | 1.21 | 36% |
| 2024 | 142 | 1.20 | 34% |
| 2025 | 138 | **1.50** | **88%** |

**Stopped.** 2021-2024 have the same hole as cfbfastR (both parse ESPN's play text): a third to a
half of incompletions carry a receiver, so a target share built on them would be a catch share
with noise. 2025 is the first season with real coverage. A walk-forward model needs fitting seasons,
so this waits: after the 2026 season there will be two seasons of real targets (2025 and 2026 live),
enough to test "target share where the season's coverage is 85%+, catch share otherwise" on the
2025 + 2026 lines. Filed on the near-miss list's rechecks, not as a near-miss. Pulling full seasons
(by game, about 800 calls a season) is not worth CFBD calls until then.

## College backs: usage by down type and player PPA from CFBD (Nov 3, 2026): one kept, one near-miss

Candidate 4 of the college-props memo. `python experiments/cfb_usage_ppa_test.py` (results in
`experiments/out/cfb_usage_ppa_test.json`); data from the **CFBD player data** Action's `usage` mode
(`data/cfb/cfbd/usage_<year>.csv`: a player's share of his team's plays on standard vs passing downs,
per season; `ppa_<year>.csv`: per player per game average PPA). Each candidate is one column multiplied
into the last calibration step (`players.CAL_EXTRA`: raw x column next to the chain and the recent
average, fitted walk-forward), rules written first.

**(a) Game script by role.** `usage_script` = (passing-downs usage - standard-downs usage, from the
back's previous season; 57% of RB rows have one) x the projected margin / 10, on carries and rushing
yards. Rule: RB carries RMSE better on tuning and holdout (2+ SE), rushing-yards Brier at the 2025
lines better by 2+ SE.

| | Tuning 2017-19 | Holdout 2021+ |
|---|---|---|
| RB carries RMSE | 5.555 → 5.546 (3.7 SE) | 5.277 → 5.271 (3.2 SE) |
| RB rushing yards RMSE | 40.192 → 40.101 (3.6 SE) | 38.021 → 37.958 (4.0 SE) |
| 2025 rushing lines, Brier (1,106 props) | | 0.2604 → 0.2592, -0.0012 ± 0.0007 (**1.9 SE**) |

Passes the projection bar on every count; the lines bar by a hair short (1.87 SE against the 2
required). **Not kept by the rule; near-miss #7.** It is the largest move toward the books of any
college candidate so far (a tenth of the rushing gap). Re-run on 2025 + 2026 lines (+ 2024 once bought).

**(b) PPA as the efficiency prior.** `ppa_rush` = the back's decayed average rushing PPA per play over
his earlier games (8-game half-life, last season at 60%, centred on the season's RB mean, shrunk by
two games; 77% of appearing RB rows have history), on rushing yards. Rule: RB rushing-yards RMSE
better on both periods, holdout 2+ SE; Brier at the 2025 lines no worse.

| | Tuning 2017-19 | Holdout 2021+ |
|---|---|---|
| RB rushing yards RMSE | 40.192 → 40.151 (2.4 SE) | 38.021 → 37.992 (3.2 SE) |
| 2025 rushing lines, Brier | | -0.0001 ± 0.0004 (no worse) |

**Passes; kept, pending the board change** (it needs the PPA file refreshed in season: the CFBD pull
gains a weekly step, and the column enters `players.build`). Size: 0.03 yards on a 38-yard miss,
nothing against the books: an accuracy input, like defense vs position.

## College PFF role inputs: routes per pass play and snap share (Nov 4, 2026): receiving kept, rushing not

Item 2 of the Nov 3 handoff. `PFF_DATA_DIR=../pff-data python experiments/cfb_pff_role_test.py`
(scores in `experiments/out/cfb_pff_role_test.json`). Data: PFF Premium Stats from the private repo
rhyman7/pff-data (`receiving_summary` for routes, `offense_summary` for snap counts), read through
`cfbmodel/pff_data.py`; no PFF rows are committed here. PFF covers every division; rows are kept only
for the board's FBS teams (the game-model frame, 136 teams in 2025). Franchises are matched to board
teams by shared players (all 1,181 FBS team-seasons 2017-2025 matched), players by name within a
team-season, weeks by team and week. Receiving yards agree exactly on 86% of matched rows (the rest are
small stat-crew corrections).

**Inputs**, as of each game from earlier games only (8-game half-life, last season at 60%):
`route_share` = routes / team pass plays; `snap_share` = offensive snaps / team snaps (team = the most
any player of the team logged that week). 84% of 2025 WR/TE/RB rows have a route share, 90% a snap
share. Each enters the last calibration step as a `CAL_EXTRA` column: receiving chain (receptions,
receiving yards) both; rushing chain (carries, rushing yards) snap share.

**Rule, written before running** (handoff memo; script docstring): fit on 2024 (one-season calibration
window, the same for the current chain), score on 2025. Each chain is kept only if all three improve by
2+ SE: (1) 2025 RMSE (receiving yards WR/TE/RB, rushing yards RB); (2) Brier at the 2025 DK/FD main
lines (receiving yards / rushing yards); (3) floor rungs: 2025 over-only alternates whose books'
reading (`CFB_RUNG_CAL`) is 65-95%, model Brier vs the current chain (betting strategy #1). Walk-forward
2017-2025 only after a pass.

| Receiving chain | Current | With PFF | Change |
|---|---|---|---|
| Receiving yards RMSE, 2025 (9,213 player-games) | 29.449 | 29.309 | **3.7 SE better** |
| Receptions RMSE (9,168) | 1.815 | 1.807 | 3.4 SE better |
| Receiving-yards lines, Brier (1,958 props; books 0.2499) | 0.2557 | 0.2552 | -0.0005 ± 0.0009 (0.6 SE) |
| Floor rungs 65-95%, Brier (6,717 rungs, 397 games; books 0.1668) | 0.1697 | 0.1690 | -0.0008 ± 0.0005 (1.6 SE) |

| Rushing chain | Current | With PFF | Change |
|---|---|---|---|
| RB rushing yards RMSE (3,647) | 37.658 | 37.653 | 1.2 SE better |
| RB carries RMSE (3,862) | 5.160 | 5.154 | 2.2 SE better |
| Rushing-yards lines, Brier (1,104 props; books 0.2500) | 0.2597 | 0.2592 | -0.0005 ± 0.0008 (0.7 SE) |
| Floor rungs 65-95%, Brier (4,118 rungs; books 0.1580) | 0.1662 | 0.1659 | -0.0003 ± 0.0002 (1.2 SE) |

**Verdict: not kept, either chain.** Route share makes the receiving projections clearly more accurate
(3.7 SE on yards, 3.4 on receptions), but that does not reach the prices: at the main lines the Brier
gain is 0.0005 (0.6 SE) and the model stays 0.0052 behind the books (2.1 SE); on the floor rungs it
moves 1.6 SE, short of 2. Snap share adds nothing to the rushing chain at the lines
or the rungs. No walk-forward (the rule runs it only after a pass). Nothing touches the board.

Worth a recheck rather than a near-miss slot: the receiving floor-rung score (1.6 SE) on 2025 + 2026
lines, and with the 2024 college prop backfill when Ryan buys it in December. The other 50 PFF tables
(depth, concept, scheme, blocking, pass rush, coverage; pulled Nov 4) are untested.

**Re-judged the same day under the new keep rule (CONVENTIONS.md, "Model tests"): anything that makes
the model more accurate is kept, however small.** The receiving chain was better on every 2025 measure.
The walk-forward (`--walk-forward`, default seasons and calibration window) decides:

| Walk-forward 2017-2025 | Tuning 2017-19 | Holdout 2021+ |
|---|---|---|
| Receiving yards RMSE, WR/TE/RB (route + snap share) | 31.435 → 31.418 (3.7 SE better) | 30.179 → 30.057 (**10.1 SE better**) |
| RB rushing yards RMSE (snap share) | 40.187 → 40.191 (1.8 SE worse) | 38.195 → 38.198 (1.2 SE worse) |

- **Receiving chain: kept.** Better on every measure: 2025 RMSE, the 2025 main lines, the floor rungs and
  both walk-forward periods. Applied: `cfbmodel/pff_roles.py`, `players.CAL_EXTRA` (receptions,
  receiving yards); the Update college board Action reads the private PFF data.
- **Rushing chain: not kept.** Slightly better on 2025 alone but worse on both walk-forward periods.
## Re-judging past tests under the keep rule (Nov 4, 2026)

Ryan's keep rule (CONVENTIONS.md, "Model tests", #193): anything that makes the model more accurate is
kept, however small: better on every measure, worse on none. Every past "not kept" result in FEATURES.md,
CFB_FEATURES.md, PLAYERS.md, CFB_PLAYERS.md and PROPS.md was inventoried (about 300 candidates). The
ones that would flip on their original numbers were **re-run on today's model**, because most were
measured before later changes were kept. Outputs and logs: `experiments/out/rejudge_2026-11-04/`.
NFL game-model candidates were re-run on the board's frame with today's features by
`experiments/rejudge_nfl_game.py`. The original feature test added total-model candidates to the
pre-October model, without wind or snaps.

**Better on every measure on today's model (kept under the rule; each needs a plan before the board):**

| Change | Measures (tuning / holdout, change in average miss; or as stated) |
|---|---|
| NFL margin + special-teams EPA (`d_st`) | -0.0005 / -0.0010 |
| NFL margin, turnover-free ratings swapped in | -0.0097 / -0.0031 |
| NFL team pass attempts + PROE | -0.0272 / -0.0045 |
| NFL team carries: three overlapping versions all better (PROE -0.0028 / -0.0151; play-calling pace + neutral pass rate 5.607→5.599 / 5.590→5.578; defense volume "vs usual" swapped in -0.0095 / -0.0068) | head-to-head needed to pick or combine |
| NFL prop nudge cap `PULL_MAX` 3 → 7.5 points | Brier vs the 3-point cap -0.00011 (2024) / -0.00008 (2025); return +8.4% vs +5.6%, -2.3% vs -4.2%; CLV better both |
| College margin + coverage defenders missed last game (or the change version, -0.0020 / -0.0013) | -0.0111 / -0.0005 |
| College unknown-QB value -0.30 → -0.35 | better on QB-change and all games, both periods |
| College margin + drive value (`d_dve`; with finishing -0.0041 / -0.0045) | -0.0043 / -0.0042 |
| College margin + mid-season coaching change | -0.0013 / -0.0095 |
| College total + returning offensive production (overall version -0.001 / -0.002) | -0.002 / -0.004 |
| College total + line yards per rush | -0.0010 / -0.0015 |
| College prop widths fitted on college misses, widths only (near-miss #6) | 2025 lines vs current: pass TD -0.0024, pass yards -0.0019, rec yards -0.0010, rush yards -0.0021 |
| NFL total + rain; wet-game unders (2+ mm or snow) | rain -0.0079 / -0.0259; unders +15.9% / +15.3%. On recorded weather; live needs forecasts (forecast log recheck) |
| NFL defense in player projections (shadow version) | flips on its Nov 2 numbers; in the shadow test (fixed until judged after 2027) |

**No longer better on today's model:**
- **College rushing PPA prior ("apply b"):** worse. RB rushing RMSE is 0.8 SE worse on 2017-19 and 2.6 SE worse on 2021+; it was kept on Nov 3 before defense vs position went on the board. Not to be applied.
- **College usage by downs (near-miss #7):** a trade-off. Rushing yards are better (2.3 / 2.0 SE) and better at the lines (1.8 SE), but carries are 0.6 SE worse on 2021+.
- **NFL total + wind above 10 mph:** worse now that wind is in the model.
- NFL linebacker injuries: the Oct re-test supersedes the first test (trade-off / zero).

Trade-offs (better somewhere, worse somewhere) and incomplete results (a measure missing) are listed
in the Project doc `claude/keep-rule-rejudge-2026-11-04.md`; trade-offs go to Ryan, incompletes need
a re-run with the missing measure before they can be judged.

### Applied (Nov 4, 2026, Ryan's go-ahead): the combined checks

Changes interact, so each set was re-checked together before it went on the board, against the model
`main` had, on the frame the Update board Action builds (injuries, wind, and the role signals of
`game_roles.py`, which the re-judge frame did not have). A set is kept only if it is still better on
every measure; a piece that makes the set worse is dropped and named.

**NFL game model** (`experiments/apply_check_nfl_game.py`, `experiments/out/apply_check_nfl_game.json`).
Measures: the raw model's and the blended projection's average margin miss, tuning (2014-18) and
holdout (2019 on), paired by game; the spread record at the closing line (market-adjusted sides with
EV above 0) alongside.

| Set | Raw tuning | Raw holdout | Blend tuning | Blend holdout | Spread EV>0, holdout return |
|---|---|---|---|---|---|
| main | 10.248 | 10.021 | | | −1.2% (n 1028) |
| + special teams `d_st` | +0.0002 ± 0.0043 | +0.0002 ± 0.0037 | +0.0008 ± 0.0022 | −0.0008 ± 0.0014 | −1.2% (n 1048) |
| turnover-free swap | −0.0057 ± 0.0057 | −0.0044 ± 0.0047 | −0.0025 ± 0.0030 | −0.0006 ± 0.0019 | −1.0% (n 1077) |
| both | −0.0021 ± 0.0065 | −0.0027 ± 0.0055 | −0.0006 ± 0.0033 | −0.0014 ± 0.0023 | −1.6% (n 1090) |

- **Kept: the turnover-free swap** (`d_pnt`, `d_rnt` for `d_pass`, `d_rush` in `model.MARGIN_FEATS`;
  `features_ext.PRODUCTION_METRICS` gained `pnt`, `rnt`; the counts are columns of
  `team_extra_<season>.csv`). Better on all four accuracy measures and on the holdout spread record.
- **Dropped: special teams.** On the board's frame it is a trade-off on its own (raw miss a hair worse
  in both periods) and it makes the swap's set worse on every measure. The re-judge's −0.0005 /
  −0.0010 came from a frame without the role signals. Not applied; it stays a trade-off for Ryan.

**NFL prop nudge cap** (`experiments/props_cap.py` re-run on the chain above, `experiments/out/props_cap.json`):
`parlays.PULL_MAX` 3 → 7.5 points. Against 3 points: Brier −0.00014 ± 0.00013 (2024) / −0.00008 ± 0.00010
(2025), CLV better in both folds, return better on 2025 and worse on 2024 (18 bets at 3 points; the
test reports bets, it does not choose on them). Details in PROPS.md "How far the model may move a prop".
On this week's Parlays page, legs priced at the same book number moved a median 0 and at most 7 points.

**College game model** (`experiments/apply_check_cfb_game.py`, `experiments/out/apply_check_cfb_game.json`; rule in
the docstring). Head-to-heads, each piece added alone (raw / blend, tuning / holdout): coverage defenders
*missed last game* −0.016 / −0.002, −0.009 / −0.002 beats the change version (−0.004 / −0.001, −0.003 /
−0.000); drive value with finishing added nothing to dve alone; returning *offensive* production (raw
−0.002 / −0.003) over overall (−0.001 / −0.002). The margin set (coverage missed + drive value + mid-season
coaching change + unknown QB −0.35) together: raw −0.022 ± 0.012 / −0.017 ± 0.011, blend −0.009 ± 0.005 /
−0.007 ± 0.005, better on all; nothing dropped (drive and the coaching change were each mixed on the blend
alone). Holdout spread record at the close, EV>0 sides: −0.6% (n 1792) → +0.1% (n 1874). The total set
(returning offensive production + line yards): raw −0.003 ± 0.010 / −0.005 ± 0.004, blend +0.0001 ± 0.0007
/ +0.0000 ± 0.0004: mixed by a ten-thousandth on a projection that is 95% market; **Ryan's call: applied.**
Plumbing: `cfbmodel/extra.py` (play splits and drives from the same play-by-play download as
`team_games`, defenders missing from the box scores, `data/cfb/coach_changes.csv`); the rating engine
solves `lyd` and `dve` like the base metrics. The college spread card gained rows "Coverage defenders
out" and "Mid-season coaching change"; "Returning defensive production" became "Returning production".
Week 10 board: raw spreads move 0.41 on average, max 2.9 (the coaching-change teams); projected lines at
most 0.84; totals at most 0.06; no bet added or removed.

**NFL defense in the player projections: checked, not applied** (`experiments/apply_check_nfl_defense_players.py`,
`experiments/out/apply_check_nfl_defense_players.json`; rule in the docstring). Ryan had said to apply the shadow
test's player version now, but on today's chain (after the volume changes) it is mixed, so it stays in the shadow
test under its pre-registered rule (Ryan, Nov 4: "just move on"). RMSE change, tuning 2020-22 / holdout 2023 on:

| | Tuning | Holdout |
|---|---|---|
| WR rec_yds | -0.0138 ± 0.0182 | +0.0031 ± 0.0138 |
| TE rec_yds | -0.0102 ± 0.0172 | -0.0183 ± 0.0128 |
| RB rec_yds | -0.0031 ± 0.0088 | +0.0011 ± 0.0067 |
| QB ints | -0.0007 ± 0.0017 | -0.0008 ± 0.0012 |

Receiving-yards Brier at the real lines: 2023 pricing 0.2563 → 0.2563; 2023 close 0.2587 → 0.2592; 2024 pricing 0.2580 → 0.2585; 2024 close 0.2620 → 0.2622; 2025 pricing 0.2569 → 0.2564; 2025 close 0.2576 → 0.2571. The re-judge's −0.054 / −0.005 was measured on the
October chain. It would have moved this week's receiving projections by a median 0.35 yards (max 3.9).

**College prop widths** (near-miss #6; `experiments/cfb_propdist_test.py --save`, `experiments/out/cfb_propdist_test.json`):
the widths-only version, refitted on the chain above, is better than the NFL-fitted chances on every stat at the
2025 lines (passing TDs −0.0025, passing yards −0.0019, receiving yards −0.0010, rushing yards −0.0013) and
is applied through `data/cfb/prop_params.json` (PROPS.md "College prop widths").

**NFL team volume** (`experiments/apply_check_nfl_volume.py`, `experiments/out/apply_check_nfl_volume.json`
and `_pieces.json`; rule in the script's docstring), on the turnover-free spread model. The test's
measures: the team regressions' average miss on team-games, tuning (2014-18) / holdout (2019 on).

Pass attempts + PROE (offense's, and the opponent's allowed): **−0.0145 ± 0.0092 / −0.0035 ± 0.0079**, kept.

Carries, head-to-head of the three re-judged versions and their combinations (miss, change vs the
current 5.6107 / 5.5906):

| Carries version | Tuning | Holdout |
|---|---|---|
| PROE | 5.601 (+0.0094 ± 0.0088) | 5.572 (−0.0188 ± 0.0064) |
| play-calling (pace, neutral pass rate; team and opponent) | 5.553 (−0.0392 ± 0.0112) | 5.575 (−0.0157 ± 0.0074) |
| vs usual, in place of the raw count | 5.571 (−0.0206 ± 0.0065) | 5.581 (−0.0095 ± 0.0045) |
| PROE + play-calling | 5.563 (−0.0295 ± 0.0123) | 5.570 (−0.0211 ± 0.0085) |
| **PROE + vs usual** | 5.577 (−0.0151 ± 0.0101) | **5.564 (−0.0263 ± 0.0081)** |
| play-calling + vs usual | 5.561 (−0.0314 ± 0.0126) | 5.571 (−0.0198 ± 0.0084) |
| all three | 5.569 (−0.0229 ± 0.0134) | 5.565 (−0.0253 ± 0.0095) |

PROE alone is now a trade-off (it was better on both in the re-judge, before the spread model
changed). Five versions are unbeaten on both periods; the pre-set tie-break (lowest holdout miss)
picks **PROE + vs usual**, which went on the board (`players.CAR_FEATS`). Play-calling was not
plumbed in; it stays available in `roles.py` for a later head-to-head.

Through the player chain (projection RMSE, tuning / holdout; `_pieces.json` has each piece alone):
the carries piece improves RB carries −0.0031 ± 0.0015 / −0.0034 ± 0.0014 and rushing yards −0.004 /
−0.010 and touches nothing else; the attempts piece improves QB pass attempts −0.012 / −0.002,
completions −0.005 / −0.003, passing yards −0.022 / −0.046 and TE receiving, and makes WR targets
+0.0002 / +0.0001 and WR receiving yards +0.010 ± 0.004 / +0.002 ± 0.004 worse. At the real 2023-25
lines (Brier vs the books, both pieces): passing yards better in five of six snapshots (2024 close
+0.0005 worse), rushing yards better in two and worse in three by 0.0001-0.0003, carries (2025)
better at pricing and worse at the close, receiving yards flat; "all" −0.0002 to 0.0000. **Ryan's
call (Nov 4): apply both**; the team-level measures the test reports pass and the downstream
differences are noise-sized. The WR receiving-yards and rushing-yards line numbers are recorded here
as the mixed part of the result.

## Margin curve by total and spread size (Nov 5, 2026): no, both boards

Both boards price every spread side, alternate spread, teaser leg and derivative off one margin curve
per season (one width and one set of key-number multipliers for every game). Teaser and alternate-line
bettors usually condition on the total, so `python experiments/total_curve_test.py` (and `--cfb`) refits
the curve walk-forward exactly as production does and tries: **A** width scaling with the closing total,
**B** A plus key-number multipliers by total band, **C** A plus width growing with the spread's size.
Measures (log loss, paired by game): results against the closing spread moved ±0.5 to ±6.5 points, both
6-point teaser legs, and Wong legs only. Rule, set before running: kept only if better on all three in
both the tuning and holdout seasons (NFL 2014-18 / 2019-25; college 2017-19 / 2020-25).

| Candidate | NFL tuning (alts / teaser / Wong) | NFL holdout | College tuning | College holdout |
|---|---|---|---|---|
| A | -0.00001 / -0.00001 / +0.00015 | -0.00005 / +0.00012 / -0.00001 | +0.00001 / +0.00021 / +0.00056 | +0.00003 / +0.00029 / +0.00039 |
| B | -0.00003 / -0.00004 / -0.00085 | -0.00011 / +0.00010 / +0.00004 | +0.00005 / +0.00032 / +0.00007 | +0.00004 / +0.00033 / +0.00052 |
| C | -0.00001 / -0.00001 / +0.00015 | +0.00008 / +0.00048 / +0.00055 | +0.00001 / +0.00021 / +0.00056 | +0.00003 / +0.00029 / +0.00039 |

**None passes; nothing changes.** The fitted total exponent was 0 in most NFL seasons and 0.2-0.6 in
college, the spread-size term 0 in both, and every difference is a fraction of a standard error
(`experiments/out/total_curve_test_<league>.json`). The single curve, anchored to the market's line and
price, is already as well calibrated on alternate lines and teaser legs as these refinements.

Side finding, not a rule (Nov 5, 2026 consultant review): the curve puts both kinds of Wong leg at about
75%. Since 2019, closing lines, pushes set aside, underdog legs (+1.5 to +2.5 teased to +7.5/+8.5) cleared
77.6% (330) and favourite legs (-7.5 to -8.5 teased to -1.5/-2.5) 71.1% (159), below the 73.9% a two-team
teaser at -120 needs; on 2014-18 both cleared about 76%. The gap is 1.5 standard errors, so the board's
pricing is unchanged; it is a candidate for the live teaser record (`claude/betting-strategies.md`).

## Quarterback value from PFF grades (Nov 5, 2026): kept, on the board

`experiments/qb_pff_grade_test.py` (rule written and committed before the first run). A
quarterback's value is his decayed EPA per dropback shrunk toward a prior; today every prior is
−0.13. Here each PFF NFL game (passing_summary, 2014 on; 8,212 quarterback games, 100% of
dropbacks matched to nflverse ids) is turned into an EPA-scale score by a walk-forward linear
map fitted on earlier seasons' games (game EPA per dropback on PFF's pass grade, alone or with
big-time and turnover-worthy throw rates; about +0.014 EPA per dropback per grade point), and a
quarterback's PFF value is his decayed, dropback-weighted score shrunk toward −0.13 with Pg
dropbacks. Versions: the EPA rating shrinks toward the PFF value instead of −0.13 ("prior"),
or the PFF value is blended into the rating ("blend"). Chosen on the tuning seasons 2015-2018
(2014 is PFF's burn-in), judged on the holdout 2019 on. Raw model, average miss:

| Version | Tuning margin, all / QB-change | Tuning total | Holdout margin, all / QB-change | Holdout total |
|---|---|---|---|---|
| Current model | 9.887 / 10.278 | 10.695 | 9.994 / 10.367 | 10.467 |
| Prior, grade, Pg 100 | 9.882 / 10.282 | 10.692 | 9.992 / 10.368 | 10.465 |
| Prior, grade, Pg 200 | 9.882 / 10.271 | 10.692 | 9.991 / 10.365 | 10.465 |
| **Prior, grade, Pg 400 (chosen)** | **9.884 / 10.265** | **10.693** | **9.991 / 10.363** | **10.466** |
| Prior, grade+BTT/TWP, Pg 100 / 200 / 400 | 9.885-9.886 / 10.270-10.293 | 10.688-10.691 | 9.992-9.993 / 10.364-10.368 | 10.465-10.466 |
| Blend, a 0.25 / 0.5 | 9.896 / 9.933, 10.231 / 10.244 | 10.688 / 10.697 | 9.998 / 10.009, 10.378 / 10.413 | 10.465 / 10.467 |
| Prior + blend 0.25 | 9.898 / 10.246 | 10.686 | 9.997 / 10.381 | 10.464 |

Chosen version against the current model (paired): tuning margin −0.003 (SE 0.007), QB-change
games −0.013 (SE 0.023), total −0.003 (SE 0.004); holdout margin −0.003 (SE 0.004), QB-change
games −0.004 (SE 0.012), total −0.001 (SE 0.003). Better on all six measures, so it passes the
keep rule; margin better in 9 of 12 seasons, total in 8 of 12. Small: the model margin moves
0.11 points on average (at most 1.35). Blending the grade into a starter's rating helps
QB-change games on the tuning seasons but hurts everything on the holdout: the grade earns its
place as the starting point for a quarterback with few dropbacks, not as a second opinion on an
established starter. Big-time and turnover-worthy rates add nothing over the grade. Props: not
affected (the player projections don't read the game model's quarterback rating).

Applied Nov 5, 2026 (Ryan's go-ahead): `nflmodel/pff_qb.py`, passed to the engine as
`qb_prior_model` by the pipeline. The production code reproduces the test's six numbers exactly.
Without the PFF data (no `.pff-data` checkout) the prior is −0.13, the old model.

## Whole-roster availability weighted by PFF grades (Nov 5, 2026): no

`experiments/pff_availability_test.py` (rule committed before the first run). Every non-quarterback
player who is known to be out before kickoff (Out or Doubtful on the injury report, or on a reserve
list: RES, PUP, NWT, SUS, RSN, RSR, EXE; game-day inactives not counted) is weighted by his usual
snap share (team's last 8 PFF games, decayed) times his PFF overall grade above a replacement level
(decayed over his earlier games, shrunk with 300 snaps). Summed by unit (OL, skill, front,
secondary) into home-minus-away margin features and both-teams total features, on top of the
production model (the QB prior, d_chg_ol, d_depth1_out and d_snap_drop included). Data: PFF
offense_summary and defense_summary 2014 on (247,514 player-games, linked by nflverse's players file;
the weekly rosters carry no pff_id for offensive linemen), nflverse injury reports and rosters.
Raw model, average miss:

| Version | Tuning 2015-18 margin / total | Holdout 2019+ margin / total |
|---|---|---|
| Current model | 9.843 / 10.693 | 9.958 / 10.466 |
| A offense/defense, replacement 60 | 9.920 / 10.701 | 9.956 / 10.475 |
| B four units, 60 | 10.044 / 10.701 | 9.957 / 10.475 |
| C offense/defense, 50 | 9.927 / 10.695 | 9.953 / 10.457 |
| D offense/defense, 70 | 9.864 / 10.702 | 9.966 / 10.472 |
| E as A, replacing d_chg_ol and d_depth1_out | 9.923 / 10.701 | 9.955 / 10.475 |

No version beats the current model on the tuning seasons, so nothing is carried to the holdout and
nothing is kept. Margin better in 6 of 12 PFF-era seasons, total in 7 (version D against the current
model). The closest to a trade-off is C (better on both holdout measures by 0.005 and 0.009, worse on
the tuning margin by 0.084): not close enough to argue for. Week 18 rest games carry the largest
values and swing the fits. The opener record at 2+ points of disagreement looks better with D
(holdout 393-287, +10.1%, against 382-289, +8.6%), but availability here is Friday's knowledge, which
the opener doesn't have, so that record is flattered by construction and isn't evidence.
Props: not applicable (the player projections don't read the game model's features).

## PFF prop inputs for the floors strategy, NFL (Nov 6, 2026): both chains mixed, Ryan's call

`experiments/nfl_pff_prop_inputs_test.py` (rule committed before the first run; two plumbing fixes in
the sanity check before any measure ran: PFF's own `team` field, and linking PFF team names to board
teams through offensive players). Added to the New projections (route share already in), as of each
game from earlier games: receiving — aDOT, slot share, contested-target rate, YAC per reception, the
opponent's coverage grade; rushing — yards before contact per carry, breakaway rate, zone share, the
opponent's run-defense grade. Same walk-forward as the board; scored on RMSE (tuning 2015-18,
holdout 2019-25), Brier at the 2023-25 DK/FD main lines and Brier on the floor rungs (alternate
rungs whose books' reading is 65-95%), pricing snapshot.

| Receiving chain | Current | With PFF inputs | Change |
|---|---|---|---|
| WR receptions RMSE, tuning / holdout | 1.8257 / 1.8054 | 1.8317 / 1.8054 | +0.0060 ± 0.0030 / +0.0000 ± 0.0015 |
| RB receptions RMSE | 1.4880 / 1.3954 | 1.4856 / 1.3963 | −0.0024 ± 0.0031 / +0.0010 ± 0.0017 |
| WR receiving yards RMSE | 28.261 / 27.555 | 28.311 / 27.566 | +0.050 ± 0.049 / +0.011 ± 0.026 |
| RB receiving yards RMSE | 15.180 / 13.796 | 15.192 / 13.804 | +0.012 ± 0.031 / +0.009 ± 0.017 |
| Main lines, Brier (10,356; books 0.24898) | 0.25388 | 0.25354 | −0.00034 ± 0.00047 |
| Floor rungs 65-95%, Brier (5,492; books 0.15952; 78.8% cleared) | 0.16610 | 0.16549 | −0.00061 ± 0.00048 |

| Rushing chain | Current | With PFF inputs | Change |
|---|---|---|---|
| RB carries RMSE, tuning / holdout | 4.4527 / 4.1566 | 4.4800 / 4.1482 | +0.0273 ± 0.0119 / −0.0084 ± 0.0059 |
| RB rushing yards RMSE | 24.976 / 24.716 | 25.017 / 24.702 | +0.041 ± 0.070 / −0.013 ± 0.039 |
| Main lines, Brier (4,311; books 0.24979) | 0.25360 | 0.25345 | −0.00016 ± 0.00095 |
| Floor rungs 65-95%, Brier (3,312; books 0.16294; 78.2% cleared) | 0.17266 | 0.17191 | −0.00075 ± 0.00084 |

Receiving: better on 3 of 10 (the projections get slightly worse; the small Brier gains are within
noise). Rushing: better on 4 of 6 (worse on the tuning seasons, carries by 2.3 SE; better on the
holdout and both Brier measures, all within noise). Neither is kept by the rule; both are Ryan's call.
The larger finding: on the floor rungs the books read the outcome far better than the model, with or
without these inputs (receiving 0.1595 vs 0.1661, rushing 0.1629 vs 0.1727, gaps of 4-6 SE). The
rungs that clear 78% of the time are already priced well by the books.

## Predicting the spread's move from the opener (Nov 6, 2026): mixed, leave as is

`experiments/line_move_predict_test.py` (rule committed before the first run; one crash fixed
before any measure). Target: closing minus opening spread. Opener-time model (last week's
starters, no late news, post-Monday games out), nfelo's openers, 3,316 games 2009 on, refitted
each season on earlier seasons. Gap-only (move ~ model minus opener, what the board implies) against
the full model (+ nfelo's rating gap without its QB adjustment, last game's starter under 75% of
dropbacks for each team, the opening spread).

| | Tuning 2014-18 | Holdout 2019+ |
|---|---|---|
| Average miss of the predicted move, gap-only | 1.2340 | 1.1743 |
| Full model | 1.2399 (+0.006 ± 0.005) | 1.1760 (+0.002 ± 0.003) |
| Bets at the opener, today's rule (model off the opener by 2+) | 203-157, +7.6% ± 4.9%, CLV +0.65 | 345-282, +4.9% ± 3.7%, CLV +0.54 |
| Full rule (predicted move 0.5+ points; threshold set on tuning) | 146-104, +11.4% ± 5.9%, CLV +0.85 | 264-216, +5.0% ± 4.3%, CLV +0.64 |

Better on 2 of 4 (both returns), worse on both move measures: mixed. The extra inputs don't predict
the move better than the gap alone; the full rule's better tuning return comes from betting fewer,
bigger disagreements (about 0.14 points of move per point of gap, so 0.5 points of predicted move
is a gap of about 3.5), and on the holdout it matches today's rule (+5.0% vs +4.9%). The one input
with a sizeable fitted effect: when the home team's starter threw under 75% of its dropbacks last
game, the line moves about 0.6 points toward the away team (2.7% of games): QB news the market
prices during the week. A lead for the backup-QB work, not a rule.

## Trenches matchup from PFF: pressure allowed against pass-rush win rate (Nov 6, 2026): no

`experiments/trenches_test.py` (rule committed before the first run). Per team, decayed over earlier
games: offense pressure allowed (pressured dropbacks / dropbacks), defense pass-rush win rate, and
the offense's pressure-to-sack rate (PFF passing_pressure and defense_pass_rush, 2014 on). Expected
pressure on each offense = its pressure allowed + the opposing pass-rush win rate (centred);
expected sacks = that × its pressure-to-sack rate relative to the league. Margin: away minus home;
total: the sum. Added to the production model. Raw model, average miss:

| Version | Tuning 2015-18 margin / total | Holdout 2019+ margin / total |
|---|---|---|
| Current model | 9.843 / 10.693 | 9.958 / 10.466 |
| A pressure | 9.822 / 10.702 | 9.955 / 10.461 |
| B sacks | 9.826 / 10.706 | 9.950 / 10.461 |
| C both | 9.820 / 10.709 | 9.954 / 10.463 |

Every version is worse on the tuning total, so none is chosen and none is kept; the QB passing-yards
props check (run only for a chosen version) did not run. Version A against the current model:
tuning margin −0.021 (SE 0.018), total +0.009 (SE 0.015); holdout margin −0.004 (SE 0.013), total
−0.005 (SE 0.007): all within noise, and the totals, where the trenches should matter most, go the
wrong way on the tuning seasons.

## Backup-QB starter pick in the player projections (Nov 6, 2026): kept, on the board

`experiments/backup_qb_pick_test.py` (rule committed before the first run; an earlier draft of the
rule, with different candidates, was replaced before anything ran). When last game's starter is not
among a team's pregame-active QBs (468 team-games since 2013), `players._pick_starter` gave the passes
to the active QB with the largest share, where a QB with no games starts at half the position
average and a season-long backup with a few passes sits near 0 (Week 9, 2026: Carson Beck over
Gardner Minshew). Candidates for that fallback only: most decayed pass attempts (volume), most
decayed starts (starts), unknown QBs ranked last (known).

| Fallback games | Tuning 2013-18 (139) pick right / starter pass-yds RMSE | Holdout 2019+ (286) |
|---|---|---|
| Current (largest share) | 74.1% / 124.5 | 55.9% / 147.8 |
| **Volume (chosen)** | **79.9% / 115.5** | **73.4% / 123.3** |
| Starts | 77.7% / 119.0 | 72.4% / 125.8 |
| Known | 78.4% / 120.4 | 61.9% / 140.3 |

Volume is better on all four measures, so it passes the keep rule. 116 picks change: 29 were right
before, 85 after. Over all team-games the pick is right 94.9% / 93.6% (tuning / holdout) against
94.6% / 92.4%. Props: fallback-game QB passing-yards main lines 2023-25 (74 lines, 35 games), Brier
0.2450 against 0.2453 (books 0.2497). Applied Nov 6, 2026 (Ryan's go-ahead): `players._pick_starter`.
