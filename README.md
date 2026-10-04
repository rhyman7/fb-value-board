# NFL Value Board

A model that projects every NFL game, compares its probabilities with the
betting market's, and prices each spread, total and moneyline by expected value.
The output is a static dashboard in `docs/` that GitHub Pages can serve as is.

## What it found

Backtested walk-forward on 3,344 games, 2014 through week 3 of 2026, graded at
closing prices:

| | Model | Closing line |
|---|---|---|
| Average miss on the margin | 10.10 pts | 9.93 pts |
| Average miss on the total | 10.62 pts | 10.47 pts |
| Picks the winner | 65.4% | 66.1% |

| Raw model, one bet on every game | Record | Win rate | Return |
|---|---|---|---|
| Spreads | 1,668-1,592-84 | 51.2% | -0.7% |
| Totals | 1,634-1,679-31 | 49.3% | -4.4% |
| Moneylines | 1,667-1,665-11 | 50.0% | -2.5% |

**The model is close to the closing line and does not beat it.** A spread bettor
needs 52.4% at -110. Since 2019, the seasons the settings were not tuned on, the
spread picks are 50.0%.

So the dashboard prices bets from a market-adjusted projection: the market line,
moved toward the model by only as much as the model has earned. That weight is
currently 15% on sides and 2% on totals. Market-adjusted spread bets with EV
above 2% went 490-432-21 (53.1%, +3.6% return, standard error 3.1%), which is
not distinguishable from zero.

Use it as a pricing tool. It gives a fair price for every bet, so a book
offering better than that price is the value, and it shows where the model and
market disagree most.

## Quick start

```bash
pip install -r requirements.txt
python run.py                  # download data, rate teams, backtest, build docs/
python run.py --no-backtest    # weekly refresh, about 20 seconds
python -m pytest -q            # tests
```

Open `docs/index.html` in a browser. The data is inlined, so it works from disk.

To price bets at your own book's numbers, copy `lines.example.csv` to
`lines.csv`, edit the prices and re-run. Projections stay anchored to the
consensus line; only the prices being graded change.

## Publishing on GitHub Pages

1. Push the repo.
2. Settings, Pages: deploy from the `main` branch, `/docs` folder.
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
   dropback. The feature is the gap between this week's starter and the
   quarterbacks who produced the team's pass rating.
3. **Market-implied ratings.** Earlier weeks' closing spreads and totals are
   solved into a rating per team. This carries injury and roster information the
   play-by-play cannot see. This week's line is never used as a feature.
4. **Game model** (`nflmodel/model.py`). Ridge regression for margin and total
   on the ratings, quarterback gap, a trailing home-field estimate, rest and
   roof. Refitted each season on earlier seasons only.
5. **Market blend.** `projection = line + w * (model - line)`, with `w` measured
   on the previous eight seasons.
6. **Probabilities** (`nflmodel/distributions.py`). A discretised curve with
   key-number multipliers (a spread of 3 pushes 9.7% of the time; a bell curve
   says 3.5%; this says 8.6%). The curve is anchored so that with no
   disagreement each side equals the market's no-vig price.
7. **Pricing** (`nflmodel/bets.py`, `nflmodel/odds.py`). EV per dollar, fair
   price, and a quarter-Kelly stake capped at 3% of bankroll.

## Honest limits

- Rating settings were tuned on 2014 to 2018. Some structural choices (adding
  market-implied ratings, the eight-season blend window) were made after seeing
  all seasons, so even the post-2019 numbers are slightly optimistic.
- Starting quarterbacks in the backtest are the actual starters from the games
  file. Those are normally known before kickoff, but not always.
- Injuries other than quarterback changes are only seen through earlier lines.
- Weather is not modeled beyond indoors or outdoors.
- Only closing lines are available, so closing line value cannot be measured.
- Moneyline no-vig prices use the proportional method, which slightly
  overstates underdogs.

## Layout

```
run.py                    command line entry point
nflmodel/
  config.py               paths and settings
  data.py                 download nflverse data, cache per-season aggregates
  ratings.py              team, quarterback and market-implied ratings
  model.py                margin and total models, market blend
  distributions.py        key-number curve, anchoring, width fit
  odds.py                 implied probability, vig removal, EV, Kelly
  bets.py                 price every bet on a set of games
  backtest.py             walk-forward grading and summaries
  slate.py                upcoming games and the ratings table
  pipeline.py             runs everything, writes docs/
site/template.html        dashboard template (no build step, no dependencies)
docs/                     generated dashboard: index.html and data.json
data/cache/               small per-season aggregates (committed)
tests/test_model.py
```

## Data

Play-by-play, schedules, results and betting lines come from the
[nflverse](https://github.com/nflverse) project. Check their terms before
redistributing the data.

For research and entertainment. Nothing here is betting advice.
