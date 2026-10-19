# College player projections

The college Players tab projects passing, rushing and receiving for every quarterback,
running back, receiver and tight end with a role in the coming FBS games. Code:
`cfbmodel/players.py` (model) and `cfbmodel/players_data.py` (box scores). It follows the
NFL board's method ([PLAYERS.md](PLAYERS.md)) with what public college data allows.

## How a projection is built

Opportunity times efficiency, using only what was known before each game.

1. **Team volume.** Pass attempts and carries per game, from the team's recent volume,
   what the opponent has allowed, the game model's projected margin and total, home
   field and wind. A regression fitted on the previous eight seasons.
2. **Player share.** The player's share of team carries, pass attempts and catches in
   games he played, recent games weighted more (5-game half-life, last season at 60%).
   Targets are not used: the college play-by-play is missing the receiver on many
   incompletions in 2021-2024 (targets ran barely above catches), so catch share is the
   consistent measure.
3. **Who plays.** There is no public injury report or game-day inactive list. A player
   is expected to play if he recorded a stat in his team's last game (in its first game,
   if he played for it last season). A regular who did not is treated as out and his
   share goes to the players expected to play.
4. **Starting quarterback.** Last game's starter, unless another expected quarterback is
   clearly the regular (80% or more of recent passes). Right on 93.4% of team-games
   since 2021.
5. **Efficiency.** Completion rate, yards per attempt, touchdown and interception rates,
   yards per carry and per catch, shrunk toward the position average. Passing and
   rushing are multiplied by what this defense has allowed compared with the league.
6. **Touchdowns.** Team touchdowns from the game model's implied points, split into
   passing and rushing by the team's tendency, shared out by each player's usage and
   touchdown rate. Anytime-TD chance assumes a Poisson count.
7. **Calibration.** A last regression per stat on earlier seasons that blends the chain
   with the player's recent per-game average (per start, for quarterbacks). Passing is
   fitted on projected starters who started, so a quarterback's line is what he throws
   if he starts.

Stats are official college box scores, which count sacks as quarterback runs (that is
how college rushing props settle).

## Backtest

Walk-forward over every FBS-vs-FBS player-game since 2017, each season fitted only on
earlier ones, scored on players who played. Average miss per player-game (lower is
better).

| Stat | Model | Season average | Recent average | Model, 2017-2019 | Season average, 2017-2019 |
|---|---|---|---|---|---|
| | **since 2021** | | | | |
| Passing yards | 64.2 | 68.4 | 66.4 | 68.5 | 73.4 |
| Pass attempts | 6.75 | 7.21 | 6.99 | 7.05 | 7.55 |
| Completions | 4.64 | 4.93 | 4.80 | 4.76 | 5.11 |
| Passing TDs | 0.95 | 1.07 | 1.03 | 0.98 | 1.11 |
| Rushing yards | 24.0 | 25.7 | 24.9 | 25.3 | 27.4 |
| Carries | 3.28 | 3.40 | 3.33 | 3.41 | 3.55 |
| Receptions | 1.23 | 1.30 | 1.26 | 1.23 | 1.30 |
| Receiving yards | 18.5 | 19.6 | 18.8 | 18.9 | 20.2 |
| Anytime TD (Brier) | 0.239 | 0.295 | 0.275 | 0.241 | 0.296 |

The model beats both baselines on every stat, in the tuning seasons and since.
Rushing yards run about 1.3 yards low (bias), the largest bias in the table.

## What earns its place

Each piece removed in turn, on 2017-2019:

- **Game script** (projected margin and total): the biggest gain for touchdowns (Brier
  0.241 against 0.257 without it) and passing touchdowns (0.98 against 1.05).
- **What the defense allows**: helps passing yards (68.5 against 68.9) and rushing yards
  (25.3 against 25.6).
- **Who plays**: built into which players are projected, so it cannot be removed on its
  own; as a game-model feature the same box-score signal added nothing (CFB_FEATURES.md).

## Limits

- No injury report: a player who misses a game without missing the one before is
  projected anyway, and a returning starter is not projected until he plays.
- Transfers and freshmen have no line until their first game for the team.
- Positions come from ESPN game rosters; players with no listing are placed by usage.
- Not compared with sportsbook prop lines: there is no free source of college props.
