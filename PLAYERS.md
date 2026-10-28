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

## Limits

- No cornerback-by-receiver matchups: that needs coverage charting (PFF or Next
  Gen Stats), which is not public. The defense enters by position group and
  through its injured backs.
- Game-day inactives are named about 90 minutes before kickoff. Earlier in the
  week only injured reserve and Out or Doubtful count.
- A quarterback's line assumes he starts. The starter pick is wrong about one
  game in fourteen, mostly benchings announced during the week; check the news.
- Prop probabilities and the comparison with sportsbook lines: [PROPS.md](PROPS.md).
