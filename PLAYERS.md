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
3. **Starting quarterback.** The active quarterback with the largest share of
   the team's recent passes takes the snaps; the others get none.
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
   backs and pass rushers.

## Backtest

Walk-forward over every player-game since 2013, each season fitted only on
earlier ones. Average miss per player-game since 2019 (lower is better):

| Stat | Model | Season average | Recent average |
|---|---|---|---|
| Passing yards (starters) | 65.4 | 78.4 | 68.3 |
| Completions | 5.36 | 6.54 | 5.63 |
| Rushing yards | 15.9 | 18.0 | 16.2 |
| Carries | 2.73 | 3.19 | 2.79 |
| Receptions | 1.29 | 1.46 | 1.31 |
| Receiving yards | 16.6 | 18.4 | 16.8 |
| Anytime TD (Brier score) | 0.199-0.212 | 0.234-0.254 | 0.210-0.223 |

The model beats both baselines on every stat in both 2013-2018 and 2019 on.
Rerun with `python experiments/players_test.py`.

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
- Quarterback passing projections run a few yards low on average (about 7 yards
  since 2019); worth fixing before comparing with prop lines.
- Not yet compared with sportsbook prop lines.
