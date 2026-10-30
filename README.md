# NFL Value Board

A model that projects every NFL game, compares its probabilities with the
betting market's, and prices each spread, total and moneyline by expected value.
The output is a static site in `docs/` that GitHub Pages can serve as is.

## What the site shows

- **This Week's Matchups** (`#matchups`): this week's games laid out like the
  matchup dashboards' home page (one row per game: spread, total, implied
  score from the line, stadium), in their graphite look. Each game (`#matchups/<game id>`)
  opens the two teams' cards with this game's projections instead of season
  stats: projected points, win chance, line and rating; projected pass and
  rush yards for and against (dials ranked among this week's teams); pass
  attempts and carries; season efficiency ranks from the team ratings; and
  every player's projected passing, rushing and receiving line. The Week select
  opens a past week from the archive (`#matchups/<season>_w<week>`): the same
  board and cards with the predictions as frozen at kickoff, and under each
  one, in blue, what happened: the final score, the result against the line,
  team yards, attempts and carries, and every projected player's actual line.
- **Games.** A card for every game with the market and model spread, total and
  win probability, and the best price if one clears break-even.
- **A page for each game**, opened from its card:
  - game summary: first tracked line, current line, model, market-adjusted
    projection, projected score and win probability
  - how the spread is built: every input of the game model in points (team
    ratings, home field, rest, each quarterback, offensive line injuries, skill
    starters out, falling snap shares), adding up exactly to the raw model spread;
    the same for the total
  - quarterbacks: rating, effect on the spread and an eight-game EPA log
  - how they match up: each offense against the defense it faces across 15
    stats, with league ranks
  - recent form: last ten games straight up, against the spread and over-under
  - every price with EV, fair price and stake
- **Players**: projected passing, rushing and receiving for every player with a
  role, with how each number is built (share, team volume, efficiency, the
  defense, teammates out). See [PLAYERS.md](PLAYERS.md).
- **Past weeks**: every game's prediction and player projections as the board
  showed them at the last update before kickoff, graded against the result:
  the board's side of each spread and total, flagged bets, and each player's
  projection next to his box score. The player table has a Proj and an
  Actual column for each stat and sorts by any of them (click a header; again
  to flip it). Stored in `docs/archive/` by
  `nflmodel/archive.py` (frozen at kickoff, never rewritten after).
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

## College football board

A college version of the same board lives in `docs/cfb/` (on GitHub Pages:
`.../docs/cfb/`), built by `python run_cfb.py` from free cfbfastR data and refreshed by
`.github/workflows/update-cfb.yml` (daily August to January, plus Saturday midday and
afternoon runs; full backtest on Mondays). Code is in `cfbmodel/`, which reuses the NFL
rating engine, odds math, key-number curve and page design. The page is made from the
NFL page by `python site/make_cfb_template.py`, which stops if an NFL page change breaks
one of its edits.

Walk-forward on FBS-against-FBS games, settings chosen on 2017-2019, graded at −110 at
the consensus closing line:

| Since 2021 (4,143 games) | Model | Closing line |
|---|---|---|
| Average miss on the margin | 12.35 pts | 12.05 pts |
| Average miss on the total | 12.60 pts | 12.43 pts |
| Picks the winner | 73.4% | 73.5% |

It trails the line by about a point in weeks 0 to 5 and is level from week 6 on. Against
opening totals (2021 on), betting the model's side when it is 3+ points off went
694-562-12 (55.3%, +5.4% ± 2.7%). See [CFB_FEATURES.md](CFB_FEATURES.md) for every test
(including "Where the edge lives", the record split by conference tier, day of week and
week of season) and [CFB_PLAYERS.md](CFB_PLAYERS.md) for the player projections.

College limits: no official injury report (players listed Out or Doubtful on Covers.com,
read daily into `docs/cfb/designations.json`, get no projection; see
`cfbmodel/designations.py`), historical lines are consensus averages with no prices, and current
prices come from ESPN's scoreboard (one book). When the `CFBD_API_KEY` secret is set, games
ESPN has not priced take their line from CollegeFootballData.com instead (`cfbmodel/cfbd.py`:
the consensus line, else DraftKings, else the first book listed; the game's line source says
which); without the key nothing changes. For research and entertainment.

## Quick start

```bash
pip install -r requirements.txt
python run.py                  # download data, rate teams, backtest, build docs/
python run.py --no-backtest    # weekly refresh, about 20 seconds
python -m pytest -q            # tests
python run_props.py all        # player props: grade, price, log (needs ODDS_API_KEY; PROPS.md)
```

Open `docs/index.html` in a browser. The data is inlined, so it works from disk.
Each game has its own address, for example `docs/index.html#2026_04_IND_WAS`.

To price bets at your own book's numbers, copy `lines.example.csv` to
`lines.csv`, edit the prices and re-run. Projections stay anchored to the
consensus line; only the prices being graded change.

## Publishing on GitHub Pages

1. Push the repo.
2. Settings, Pages: deploy from the `main` branch, `/docs` folder. (The live site is served from the
   repository root, where `index.html` is the Weekly Value Boards landing page
   linking to both boards; a URL with a hash such as `#games` forwards to the
   NFL board.)
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
   price, and a quarter-Kelly stake capped at 3% of bankroll and 5% a game, all
   sides together. The stake is sized from the model probability shrunk toward the
   market's by the share of claimed edge that has shown up in the last eight
   seasons' graded bets (`backtest.edge_haircut`; 0 on the NFL board as of
   October 2026, so no NFL stake is sized until that changes); the EV and
   probabilities shown are the model's. See FEATURES.md, "Edge tests, late
   October 2026".

## Market lines

`data/market/` holds snapshots of every upcoming game's spread, total and moneyline at
Pinnacle, DraftKings, FanDuel, BetMGM and Kalshi, from The Odds API (`nflmodel/market.py`,
`run_market.py`, the **Market lines** Action: three a day in season and hourly from Sunday
night to Tuesday, when the week's lines open). Only rows that changed are stored, so a
month of both leagues is a few thousand rows. Pinnacle is the sharp reference: the boards
anchor their projections to its no-vig price where it quotes the game, and score every other
book's price at that book's own line. Kalshi contract prices are charged the exchange's fee
(`odds.kalshi_effective_odds`: 7% x p x (1-p) per contract as a taker, 1.75% as a maker)
before EV. `run_market.py status` says what is stored; `run_market.py alerts` lists games where
the published raw model is 2+ points (NFL) or 3+ (college) off Pinnacle's current line.

On the boards, a game Pinnacle quotes is anchored to Pinnacle's line and prices rather than
the consensus (its card and page say so), each side's table shows the best price among
DraftKings, FanDuel, BetMGM and Kalshi at that book's own line with a "beats Pinnacle" chip
when the price is better than the sharp no-vig number on its own, and Past weeks grade every
flagged bet's closing line value against Pinnacle's close. The Market lines Action posts
opener alerts (raw model well off Pinnacle early in the week) to a GitHub issue labelled
`opener-alerts`.

## Honest limits

- Rating settings were tuned on 2014 to 2018. Some structural choices (adding
  market-implied ratings, the eight-season blend window) were made after seeing
  all seasons, so even the post-2019 numbers are slightly optimistic.
- Starting quarterbacks in the backtest are the actual starters from the games
  file. Those are normally known before kickoff, but not always.
- Injuries other than quarterback changes and the offensive line are only seen
  through earlier lines and the current line. For upcoming games, game-day
  inactives are not known, so only injured reserve and Out/Doubtful count.
  News the official sources have not caught up with yet (a starting quarterback
  named midweek, a player ruled out before Friday's report) can be entered by hand
  in `data/overrides.csv`, one row per player per game (see `nflmodel/overrides.py`).
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
  overrides.py            hand-entered starters and outs from data/overrides.csv
  market.py               game lines from several books (Pinnacle, DK, FD, BetMGM, Kalshi), data/market/
  def_ratings.py          defender ratings and missing-defender features (shadow test only)
  defense.py              team defensive rates from PFR (shadow test only)
  shadow.py               the shadow test of the defensive inputs, logged to data/shadow/
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
  propdist.py             player-prop distributions (counts, yards, early exit, lines)
  props.py                prop probabilities from the player projections
  propsim.py              same-game combinations (correlations by kind of pair)
  oddsapi.py              The Odds API client, with a credit log
  propmarket.py           vig removal, edge, EV, flags and stakes for props
  proplog.py              the append-only prop log, grading, closing line value
  propbackfill.py         past prop prices from the historical endpoint
site/template.html        site template (no build step, no dependencies)
docs/                     generated site: index.html and data.json
index.html                Weekly Value Boards landing page (links to both boards)
img/                      league logos for the landing page
data/cache/               small per-season aggregates (committed)
data/line_history.csv     lines as first seen and as they move
data/wind_log.csv         wind forecasts as fetched
experiments/feature_test.py  test candidate features walk-forward
experiments/qb_test.py       quarterback-change settings
experiments/qb_draft_prior_test.py  a backup's prior by draft round (not kept)
experiments/line_move_test.py       line movement as a feature and as a bet filter (not kept)
experiments/injury_test.py   missing starters by position group
experiments/players_test.py      player projections against baselines
experiments/players_ablation.py  which pieces of the player model help
PLAYERS.md                how player projections are built, and how they have done
run_props.py              player props: price, log, grade, backfill
run_market.py             game-line snapshots, status and opener alerts
experiments/props_calibration.py  prop distributions, calibration report, fitted parameters
experiments/props_real_lines.py   the prop model against past DraftKings and FanDuel lines
PROPS.md                  player props: method, calibration, market, log
data/prop_params.json     fitted prop distributions
data/props/               prop log, results, snapshots, credit usage, past prices
FEATURES.md               results of the October 2026 feature tests and the edge tests
tests/test_model.py
```

## Data

Play-by-play, schedules, results and closing lines come from the
[nflverse](https://github.com/nflverse) project. Opening lines come from the
[nfelo](https://github.com/greerreNFL/nfelo) project's public game file. Player-prop
prices come from [The Odds API](https://the-odds-api.com) (DraftKings and FanDuel). Wind
forecasts come from [Open-Meteo](https://open-meteo.com). Check their terms
before redistributing the data.

For research and entertainment. Nothing here is betting advice.
