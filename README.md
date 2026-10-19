# NFL Value Board

A model that projects every NFL game, compares its probabilities with the
betting market's, and prices each spread, total and moneyline by expected value.
The output is a static site in `docs/` that GitHub Pages can serve as is.

## What the site shows

- **Games.** A card for every game with the market and model spread, total and
  win probability, and the best price if one clears break-even.
- **A page for each game**, opened from its card:
  - game summary: first tracked line, current line, model, market-adjusted
    projection, projected score and win probability
  - how the spread is built: team ratings, home field, rest and each
    quarterback, in points
  - quarterbacks: rating, effect on the spread and an eight-game EPA log
  - how they match up: each offense against the defense it faces across 15
    stats, with league ranks
  - recent form: last ten games straight up, against the spread and over-under
  - every price with EV, fair price and stake
- **Players**: projected passing, rushing and receiving for every player with a
  role, with how each number is built (share, team volume, efficiency, the
  defense, teammates out). See [PLAYERS.md](PLAYERS.md).
- **Team ratings** and the **model record** (backtest, key numbers, method).

## What it found

Backtested walk-forward on 3,387 games, 2014 through week 6 of 2026, graded at
closing prices:

| | Model | Closing line |
|---|---|---|
| Average miss on the margin | 10.04 pts | 9.91 pts |
| Average miss on the total | 10.56 pts | 10.46 pts |
| Picks the winner | 65.5% | 66.0% |

| Raw model, one bet on every game | Record | Win rate | Return |
|---|---|---|---|
| Spreads | 1,690-1,610-87 | 51.2% | -0.6% |
| Totals | 1,684-1,672-31 | 50.2% | -2.8% |
| Moneylines | 1,701-1,673-12 | 50.4% | -2.2% |

**The model is close to the closing line and does not beat it.** A spread bettor
needs 52.4% at -110. Since 2019, the seasons the settings were not tuned on, the
spread picks are 50.5%.

So the dashboard prices bets from a market-adjusted projection: the market line,
moved toward the model by only as much as the model has earned. That weight is
currently 17% on sides and 5% on totals (totals are capped there; see
FEATURES.md). Market-adjusted spread bets with EV above 2% went 603-513-25
(54.0%, +5.4% return, standard error 2.9%), but only 51.9% since 2019, so the
improvement over earlier versions comes mostly from seasons the settings were
chosen on and is not yet proof.

### Against opening lines

The picture is better when the bet is placed at the opening spread instead of
the closing one. On 2,667 games since 2014:

| Model disagrees with the opener by | Record | Win rate | Return |
|---|---|---|---|
| any amount | 1,348-1,260-59 | 51.7% | -1.3% |
| 1+ points | 895-805-38 | 52.6% | +0.5% |
| 2+ points | 521-433-23 | 54.6% | +4.2% |
| 3+ points | 265-228-9 | 53.8% | +2.5% |

On the 2+ point bets the line later moved toward the model 51% of the time
and away from it 28%, for an average of +0.6 points of closing line value.
The same sides bet at the close won 52.0%.

This test is deliberately strict: the model is not told about quarterback
changes announced during the week, and 345 games where a team had played the
previous Monday night are left out. It is still not proof. The 2+ point return
is 1.4 standard errors above zero, and opening numbers move fast and take
small bets, so real results would land between the opening and closing records.

Games four or more days away are marked on the board as early lines and priced
with the weight the model has earned against opening spreads (17%, against 17%
for closing lines). The difference is small because that is what the data
supports: replaying that pricing on past openers flagged 825 bets that went
430-376-19 (53.4%) and returned +1.9%, about half the +3.4% the pricing claimed.

**So the model's value is early in the week.** By kickoff the market has
absorbed what it knows. Use it to find spreads that look off when they open, and
as a fair price to shop against.

### Wind and totals

Closing totals do not fully account for wind. Unders at the closing total in
outdoor games with 10 to 19 mph of recorded wind went 498-379-9 (56.8%, +10.4%
return), and 238-164-4 (59.2%, +14.2% ± 4.7%) since 2019. The effect shows up
from 2014 on and not before. The band was chosen after looking at every season
and the backtest uses wind measured at the game rather than a forecast, so it is
a lead to keep testing. The board tags games forecast in that band.

### Other stats tested

Sack rate, explosive plays, early-down passing, special teams, fumble luck,
quarterback CPOE, pace, travel, time zones, the West Coast early kickoff and
divisional games were each tested as additions to the model. None improved the
spread model. Wind and snaps per game improved the total model a little. Full
results are in [FEATURES.md](FEATURES.md); rerun them with
`python experiments/feature_test.py`.

## Quick start

```bash
pip install -r requirements.txt
python run.py                  # download data, rate teams, backtest, build docs/
python run.py --no-backtest    # weekly refresh, about 20 seconds
python -m pytest -q            # tests
```

Open `docs/index.html` in a browser. The data is inlined, so it works from disk.
Each game has its own address, for example `docs/index.html#2026_04_IND_WAS`.

To price bets at your own book's numbers, copy `lines.example.csv` to
`lines.csv`, edit the prices and re-run. Projections stay anchored to the
consensus line; only the prices being graded change.

## Publishing on GitHub Pages

1. Push the repo.
2. Settings, Pages: deploy from the `main` branch, `/docs` folder. (If Pages is
   set to the repository root instead, `index.html` there forwards to the board.)
3. Settings, Actions, General: allow workflows read and write permission.

`.github/workflows/update.yml` then refreshes the board daily during the season
and re-runs the full backtest on Tuesdays. The workflow has not been run yet;
check its first run.

## How the model works

1. **Team ratings** (`nflmodel/ratings.py`). Pass and rush EPA per play, success
   rate and points, each solved as a ridge regression over all teams at once so
   ratings are opponent-adjusted. Recent games count more (9-week half-life),
   prior seasons count 55% as much, garbage time is dropped and single-play EPA
   is capped.
2. **Quarterback adjustment.** Each quarterback has a decayed, shrunk EPA per
   dropback, judged over about 80 team-weeks so a starter's value does not
   collapse after a few bad games; an unknown backup starts at 0.13 EPA per
   dropback below average. The feature is the gap between this week's starter and the
   quarterbacks who produced the team's pass rating.
3. **Market-implied ratings.** Earlier weeks' closing spreads and totals are
   solved into a rating per team. This carries injury and roster information the
   play-by-play cannot see. This week's line is never used as a feature.
4. **Game model** (`nflmodel/model.py`). Ridge regression for margin and total
   on the ratings, quarterback gap, a trailing home-field estimate, rest,
   missing offensive-line starters (`nflmodel/injuries.py`) and roof. The total model also uses wind and a snaps-per-game rating (pace).
   Refitted each season on earlier seasons only. The margin model has no
   intercept, so a neutral-site game gives neither team a home edge.
5. **Market blend.** `projection = line + w * (model - line)`, with `w` measured
   on the previous eight seasons. The totals weight is capped at 5%.
6. **Wind** (`nflmodel/weather.py`). Upcoming outdoor games within a week get
   the Open-Meteo forecast for the kickoff hour; played games use the recorded
   wind. Every forecast is saved to `data/wind_log.csv`.
7. **Probabilities** (`nflmodel/distributions.py`). A discretised curve with
   key-number multipliers (a spread of 3 pushes 9.7% of the time; a bell curve
   says 3.5%; this says 8.6%). The curve is anchored so that with no
   disagreement each side equals the market's no-vig price.
8. **Pricing** (`nflmodel/bets.py`, `nflmodel/odds.py`). EV per dollar, fair
   price, and a quarter-Kelly stake capped at 3% of bankroll.

## Honest limits

- Rating settings were tuned on 2014 to 2018. Some structural choices (adding
  market-implied ratings, the eight-season blend window) were made after seeing
  all seasons, so even the post-2019 numbers are slightly optimistic.
- Starting quarterbacks in the backtest are the actual starters from the games
  file. Those are normally known before kickoff, but not always.
- Injuries other than quarterback changes and the offensive line are only seen
  through earlier lines and the current line. For upcoming games, game-day
  inactives are not known, so only injured reserve and Out/Doubtful count.
- Weather is wind only. The backtest uses wind recorded at the game, so the
  wind feature and the wind under will look better in testing than they will
  live, where a forecast is used.
- Opening lines cover spreads only, for about 88% of games, and their exact
  posting time is not known. `data/line_history.csv` also records each game's
  line the first time the board sees it and whenever it moves.
- Matchup ranks are season to date, so they are noisy in the first few weeks.
- Moneyline no-vig prices use the proportional method, which slightly
  overstates underdogs.

## Layout

```
run.py                    command line entry point
nflmodel/
  config.py               paths and settings
  data.py                 download nflverse data, cache per-season aggregates
  extra.py                extra per-game stats (sacks, pace, special teams, CPOE ...)
  features_ext.py         candidate features and the extended rating engine
  weather.py              wind forecasts for upcoming games, and the forecast log
  injuries.py             missing starters by position group (reports, rosters, snaps)
  players_data.py         player box scores from play-by-play
  players.py              player projections, backtest summary and board output
  ratings.py              team, quarterback and market-implied ratings
  model.py                margin and total models, market blend
  distributions.py        key-number curve, anchoring, width fit
  odds.py                 implied probability, vig removal, EV, Kelly
  bets.py                 price every bet on a set of games
  backtest.py             walk-forward grading and summaries
  slate.py                upcoming games, line tracking and the ratings table
  matchup.py              per-game pages: breakdown, quarterbacks, ranks, trends
  pipeline.py             runs everything, writes docs/
site/template.html        site template (no build step, no dependencies)
docs/                     generated site: index.html and data.json
index.html                forwards to docs/ if Pages serves the repository root
data/cache/               small per-season aggregates (committed)
data/line_history.csv     lines as first seen and as they move
data/wind_log.csv         wind forecasts as fetched
experiments/feature_test.py  test candidate features walk-forward
experiments/qb_test.py       quarterback-change settings
experiments/injury_test.py   missing starters by position group
experiments/players_test.py      player projections against baselines
experiments/players_ablation.py  which pieces of the player model help
PLAYERS.md                how player projections are built, and how they have done
FEATURES.md               results of the October 2026 feature tests
tests/test_model.py
```

## Data

Play-by-play, schedules, results and closing lines come from the
[nflverse](https://github.com/nflverse) project. Opening lines come from the
[nfelo](https://github.com/greerreNFL/nfelo) project's public game file. Wind
forecasts come from [Open-Meteo](https://open-meteo.com). Check their terms
before redistributing the data.

For research and entertainment. Nothing here is betting advice.
