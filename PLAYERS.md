# Player projections

The Players tab projects passing, rushing and receiving for every quarterback,
running back, receiver and tight end with a role in the coming games. Code:
`nflmodel/players.py` (model) and `nflmodel/players_data.py` (box scores).

## How a projection is built

Opportunity times efficiency, using only what was known before each game.

1. **Team volume.** Pass attempts and carries per game, from the team's recent
   volume, what the opponent has allowed, the game model's projected margin and
   total (teams that trail throw more), wind and home field. A regression fitted
   on the previous eight seasons.
2. **Player share.** The player's share of team targets, carries and pass
   attempts in games he was active, recent games weighted more (6-game
   half-life, last season at 60%). Active status comes from weekly rosters and
   injury reports, so a game missed injured is not a zero. When teammates are
   out, the shares of those available are scaled up to fill the gap.
3. **Starting quarterback.** Last game's starter if he is active, unless another
   active quarterback is clearly the regular (80% or more of the team's recent
   passes), as when a starter returns from injury. With neither, the active
   quarterback with the largest share. He takes all the passes; the others get
   none. Right on 92.7% of team-games since 2019.
4. **Efficiency.** Catch rate, yards per target and per carry, completion rate,
   yards per attempt, touchdown and interception rates, each shrunk toward the
   position average. Rushing and passing are multiplied by what this defense
   has allowed compared with the league.
5. **Touchdowns.** Team touchdowns from the game model's implied points, split
   into passing and rushing by the team's tendency, then shared out by red-zone
   targets and carries. Anytime-TD chance assumes a Poisson count.
6. **Calibration.** A last regression per stat, fitted on earlier seasons, that
   blends the chain above with the player's recent per-game average (per start,
   for quarterbacks) and takes in wind and, for receivers, the defense's missing
   backs and pass rushers. Passing is fitted only on games the projected starter
   actually started (official starter from the schedule data), so a
   quarterback's line is what he throws **if he starts**. A passing prop is void
   if he does not play.

## Backtest

Walk-forward over every player-game since 2013, each season fitted only on
earlier ones. Average miss per player-game since 2019 (lower is better).
Quarterbacks are scored as the projected starter in games he started, with an
earlier start so the per-start baselines exist.

| Stat | Model | Season average | Recent average | Bias |
|---|---|---|---|---|
| Passing yards | 59.2 | 64.9 | 61.8 | +1.4 |
| Pass attempts | 6.74 | 7.33 | 6.98 | +0.2 |
| Completions | 4.77 | 5.21 | 4.97 | +0.2 |
| Passing TDs | 0.90 | 0.99 | 0.95 | −0.03 |
| Rushing yards | 15.9 | 16.5 | 16.1 | −0.1 |
| Carries | 2.72 | 2.84 | 2.78 | −0.1 |
| Receptions | 1.28 | 1.35 | 1.30 | −0.1 |
| Receiving yards | 16.4 | 17.3 | 16.7 | −0.3 |
| Anytime TD, receiving / rushing (Brier) | 0.194 / 0.208 | 0.223 / 0.237 | 0.207 / 0.221 | |

The model beats both baselines on every stat in both 2013-2018 and 2019 on.
Rerun with `python experiments/players_test.py`.

### The passing fix, October 2026

Passing yards had run about 7 yards low. The yardage math was fine; the cause
was the starter. About one projected starter in ten did not start (benched, or
a backup who had kept the job when the regular came back) and threw next to
nothing, and the calibration learned from those games, shrinking every
starter's projection. Starters who did start came out about 13 yards low.
Two changes, both better on 2013-2018 and on 2019 on:

| Since 2019 | Before | After |
|---|---|---|
| Passing yards bias, projected starter who started | −12.9 | +1.4 |
| Passing yards miss, every actual starter | 69.6 | 68.2 |
| Starter picked right, team-games since 2019 | 91.1% | 92.7% |

An earlier version of the test script lined up its season-average baseline
with the wrong rows after the first stat, which made that baseline look worse
than it is (for example 18.4 receiving yards instead of 17.3). Fixed; the
board's own table was not affected.

## What earns its place

Each piece removed in turn (`python experiments/players_ablation.py`):

- **Teammates out, usage passed on**: the biggest gain, for receiving,
  rushing and passing.
- **Game script** (projected margin and total): clear gain for passing yards.
- **Wind**: gain for passing yards, mostly since 2019.
- **What the defense allows**: helps passing and rushing; no help for
  receivers, so receivers use their own rates.
- **Defensive injuries**: small; kept only for receiving, where they helped on
  the tuning seasons.

The defense matters less than usage and game situation, the same pattern as
the game model.

### More inputs tested, October 2026

`python experiments/players_more_test.py` (average miss; tuning 2014-2018, holdout 2019 on):

| Change | Tuning | Holdout | Kept? |
|---|---|---|---|
| Team pass attempts with pace and neutral pass rate | 6.445 → 6.431 | 6.176 → 6.179 | No: no gain |
| Team carries, same | 5.607 → 5.599 | 5.588 → 5.577 | No: too small to matter |
| Boosted last step for quarterbacks (yards, attempts, completions, TDs) | worse on all four | worse on all four | No |
| Questionable multiplier (below), rushing yards, RBs and others | 15.07 → 14.05 | 12.61 → 11.88 | Yes, New projections |
| Same, receiving yards | 20.37 → 19.20 | 18.09 → 17.28 | Yes |
| Same, carries / targets / receptions | 4.58 → 4.32 / 2.40 → 2.27 / 1.77 → 1.68 | 3.12 → 3.03 / 1.92 → 1.87 / 1.51 → 1.47 | Yes |

Players listed Questionable who play get less than their usual usage, more so the less
they practiced: about 0.9 of the projection after a full practice, 0.8 after limited, 0.7
after a week of no practice (receiving yards, all seasons before 2026). The new
projections, which already see practice status, still ran high on these players. So the
New projections are scaled by a multiplier per position and practice status, fitted each
season on earlier seasons only and shrunk toward 1 by n / (n + 200)
(`players_ml.questionable`). Shrinkage of 50 scored better on the tuning seasons, 200
since 2019; 200 was taken because 50 overcorrected in recent seasons (projections then ran
low by about 2.5 yards). Scored on Questionable players who played, which is what a prop
settles on; a player who sits voids the bet.

### Route participation and snap share (Oct 31, 2026)

`python experiments/routes_test.py` asks whether route participation (how often a player is on
the field when his team drops back) would improve the target projections: the role measure PFF
sells weekly. nflverse publishes who was on the field for every play (NGS 2016-2022, FTN 2023
on) but only after the season (2026 is not out), so the board cannot use it live. Walk-forward
by season (fitted on the seasons before, tested 2019-2025), on wide receivers, tight ends and
backs who played, the error (mean squared) of the target share:

| Added to the model's projected target share | WR | TE | RB | All | Every season 2019-2025 |
|---|---|---|---|---|---|
| Snap share, as of the game and last game (free, weekly) | −2.2% | −2.6% | −1.9% | −2.25% | −1.8% to −2.6% |
| Snap share + route share + targets per route (PFF, weekly) | −3.0% | −3.8% | −2.6% | −3.1% | −2.9% to −3.4% |
| What routes add over snaps | −0.8% | −1.2% | −0.7% | **−0.85%** | −0.6% to −1.2% |

Targets per game: average miss 1.965 → 1.931 with both. **Three quarters of the gain is free**:
snap share, which nflverse publishes every week (from Pro Football Reference) and which the
board's main projections did not use then (only the New projections, `players_ml.py`, did; on the
2025 real lines they cut receiving yards' Brier gap to the books from 0.0070 to 0.0051 in the final run below,
`experiments/out/props_real_lines_ml_2025.json`). Routes would add under a point more. So the
next step is the free one: bring snap share into the main projections (or adopt the New
projections) and test on the 2023-2025 real lines; paid route data is worth considering only
after that.

### The New projections become the projections (Oct 31, 2026)

The New projections (`players_ml.py`, snap share and the other role inputs) now replace the
current ones for every stat and position they cover (`players_ml.ADOPT`: RB carries and rushing
yards; WR and RB receptions and receiving yards), on every page. The Players page's record grades
them: on backtest runs the boosted step runs walk-forward over the holdout seasons too.
Passing stays on the current projections: a boosted last step for quarterbacks was worse on all
four passing stats on both the tuning and holdout seasons (table above), and tight ends were not
better either, so they keep the current model.

Tested first on the real DraftKings and FanDuel lines, 2023-2025
(`python experiments/props_real_lines_ml.py --season <year>`; 2023 and 2024 have the yardage
markets only). Our Brier score minus the books' (lower is better; ± one standard error,
clustered by game), at the pricing snapshot and at the close:

| Stat (positions) | Season | Pricing: current → New | Close: current → New |
|---|---|---|---|
| Rushing yards (RB) | 2023 | +0.0105 → +0.0014 | +0.0117 → +0.0070 |
| | 2024 | +0.0048 → +0.0101 (worse) | +0.0098 → +0.0134 (worse) |
| | 2025 | +0.0008 → −0.0008 | +0.0091 → +0.0071 |
| | **All three** | **+0.0048 → +0.0033** | **+0.0102 → +0.0092** |
| Receiving yards (WR, RB) | 2023 | +0.0052 → +0.0051 | +0.0094 → +0.0077 |
| | 2024 | +0.0089 → +0.0092 | +0.0124 → +0.0098 |
| | 2025 | +0.0070 → +0.0051 | +0.0079 → +0.0067 |
| | **All three** | **+0.0071 → +0.0065** | **+0.0099 → +0.0081** |
| Carries (RB) | 2025 | +0.0063 → +0.0018 | +0.0187 → +0.0056 |
| Receptions (WR, RB) | 2025 | +0.0102 → +0.0043 | +0.0106 → +0.0089 |

New is better on every stat over the three seasons and in 2023 and 2025 on all of them. 2024
went the other way for RB rushing yards (about one standard error), so the gain there is
smaller and less certain than for receiving. The books are still the better forecast on every
line (the gap is above zero almost everywhere), so this narrows the gap; it does not close it,
and no prop bets are flagged.

## Limits

- No cornerback-by-receiver matchups: that needs coverage charting (PFF or Next
  Gen Stats), which is not public. The defense enters by position group and
  through its injured backs.
- Game-day inactives are named about 90 minutes before kickoff. Earlier in the
  week only injured reserve and Out or Doubtful count.
- A quarterback's line assumes he starts. The starter pick is wrong about one
  game in fourteen, mostly benchings announced during the week; check the news.
- Prop probabilities and the comparison with sportsbook lines: [PROPS.md](PROPS.md).

### Defense against each position (Nov 1, 2026)

The matchup dashboards split pass defense by position (yards allowed to wide receivers, tight
ends and backs). `experiments/def_vs_position.py` built that split as a rating per defense,
position and stat: the log of what it allowed over what the projections expected those players
to get (so it is already adjusted for the offenses it faced), averaged over its earlier games
(6-game half-life, last season at half weight) and shrunk toward 0 by 4 games. The projection
is multiplied by exp(b x rating), b fitted on earlier seasons (`nflmodel/dvp.py`).

| Position, stat | Chain projections, holdout 2019 on (b fitted 2013-2018) | New projections, 2022 on (b fitted 2019-2021) |
|---|---|---|
| WR receiving yards | 19.91 → −0.09 ± 0.01 | 18.69 → −0.02 ± 0.004 |
| TE receiving yards | 12.44 → −0.05 ± 0.004 | (chain shown) |
| RB receiving yards | 8.97 → −0.12 ± 0.006 | 8.55 → −0.13 ± 0.009 |
| WR / TE / RB receptions | −0.0023 / −0.0027 / −0.0072 | −0.0010 / (chain) / −0.0030 |
| RB rushing yards | 16.41 → −0.11 ± 0.01 | 15.96 → −0.07 ± 0.01 |
| RB carries | −0.0061 ± 0.0011 | −0.0084 ± 0.0021 |
| QB passing yards | 68.93 → −0.21 ± 0.14 (not kept) | |

Every pair but passing yards clears two standard errors on both sets of projections
(`experiments/dvp_ml_test.py`), so it is applied to the projections the board shows: the chain
for tight ends, the New projections for the stats in `players_ml.ADOPT`. Gains are small, as
the defense's were before: usage and game script still do most of the work.

**It is not an edge against the books.** On 2025 DraftKings and FanDuel main lines the rating
tied the books' own chance (pooled Brier −0.00003 ± 0.00018), and the props it moved 2+ points
off the books went 155-162, −7.4% ± 4.8%, closing line value −4.7% ± 0.4%. The books price
the matchup. Receiving yards alone scored better than the books on 2025 (−0.00061 ± 0.00021),
one of several stats tried: a lead for the live prop log, not a result.
