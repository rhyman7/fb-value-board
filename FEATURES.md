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

## What to try next

- **Opening lines for totals.** The model's sides do best against openers;
  nobody has tested totals that way yet, and wind forecasts are what a total
  opener cannot know. That needs a source of opening totals.
- **Grade the wind under on forecasts.** Every forecast the board fetches is
  now saved to `data/wind_log.csv`. After a season of logs, re-run the wind
  table on the logged forecast instead of the recorded wind.
- **Pass catchers.** Re-test after the 2026 season (see above).
- **College football**, where the market is softer and rosters change more.
