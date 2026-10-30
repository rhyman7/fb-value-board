# Conventions: Weekly Value Boards

Read this before changing anything in this repo, and update it in the same pull request
whenever a convention changes. README.md describes the model and every section of the
boards in detail; this file is the rules.

- Live site: GitHub Pages, branch `main`, **repository root**
  - Landing page: https://rhyman7.github.io/fb-value-board/ (`index.html`)
  - NFL board: https://rhyman7.github.io/fb-value-board/docs/ (`docs/index.html`)
  - College board: https://rhyman7.github.io/fb-value-board/docs/cfb/ (`docs/cfb/index.html`)
- The matchup dashboards (rhyman7/nfl, rhyman7/cfb) and the Weekly Gridiron Board
  (rhyman7.github.io) are separate sites with their own CONVENTIONS.md.

## How changes get made

- **Pull requests, merged after checks.** Don't push hand edits straight to `main`: `main` is
  the live site. Open a pull request (one topic per PR), run the checks in "Checks before
  merging", and merge it yourself once they pass (squash, delete the branch). Then tell Ryan
  in a line or two what went live, with the PR link. If a check fails, don't merge: fix it, or
  leave the PR open and tell Ryan why.
- **After merging**, confirm the Pages deploy for the merge commit succeeded and load the live
  board(s) you changed. If something broke, revert right away (a revert PR, merged the same
  way) and tell Ryan.
- **Both boards.** Design and feature changes go to both the NFL and the college board
  unless Ryan says otherwise. Don't call a change finished until both are done.
- **Ask first** if anything is unclear or a change could break the live site.
- **Kept separate for now, with one exception.** Don't link the value boards from the matchup
  dashboards or rhyman7.github.io, and don't link those from here, until Ryan says otherwise.
  The exception (Ryan's call, Oct 2026): the dashboards' **game packets** include this board's
  matchup page. Their Build PDFs Actions check out this repo, open the Matchups view
  (`docs/index.html#matchups`, `docs/cfb/index.html#matchups`), press Print all and take one
  landscape page per game, matched to the dashboard by game id (the college board's
  `game_id` is the ESPN id the CFB dashboard uses). If you change the Matchups view's print
  layout, its row links (`#matchups/<id>`) or the game ids, check the dashboards' packets
  still map every game; their Action logs print the counts.
- **What commits straight to `main`** (automated, not hand edits): the **Update board** and
  **Update college board** Actions (`.github/workflows/update.yml`, `update-cfb.yml`), which
  rebuild `docs/` and the committed files under `data/` (for the NFL board including the
  shadow test log, `data/shadow/`). They share the `board-commits`
  concurrency group so they never push at once. Everything else goes through a PR.
- **Shadow test (NFL): `data/shadow/`.** The **Update board** Action also writes
  `defense_games.csv` and `defense_players.csv` (`nflmodel/shadow.py`): the game model and two
  player projections refitted with the defensive-player inputs, logged and never shown on the
  board. A game's rows are rewritten until kickoff and never after; don't edit or delete rows
  by hand. The inputs and the rule are fixed until the test is judged (after the 2027 season,
  FEATURES.md "Shadow test"): don't change `shadow.MARGIN`, `TOTAL` or `PLAYER`, and score it
  with `experiments/shadow_score.py`. A shadow failure never stops the board build. A second
  spread version, pass catchers out (`shadow.MARGIN_PC`, from Oct 29, 2026), is logged the same
  way to `passcatchers_games.csv` under the same rule: don't change `MARGIN_PC` either.
- The **Player props** and **Prop price backfill** Actions (`props.yml`, `props-backfill.yml`)
  also commit straight to `main`, only under `data/props/` (the prop log, results, snapshots,
  credit usage and past prices) and, for Player props, `docs/props/` (the board's Props files). They share the `board-commits` group too. The backfill runs
  only by hand.
- The **Wind history** Action (`wind-history.yml`, by hand only) runs
  `experiments/fetch_wind_history.py` and commits only `data/cache/wind_history.csv`: the
  archive wind at kickoff (speed, gusts, direction, along-field and crosswind parts) for
  every played outdoor NFL home game since 2014, one Open-Meteo archive call per stadium per
  game day, resumable. It shares the `board-commits` group. The test that reads it is
  `experiments/wind_direction_test.py` (FEATURES.md "Wind direction and gusts").
- **Wind logs (both boards): `data/wind_log.csv`, `data/cfb/wind_log.csv`.** Every forecast
  the board builds fetch, one row per game per fetch: `game_id, fetched` (UTC), `wind_mph,
  gust_mph, dir_deg` (direction the wind blows from), and on the NFL log `along_mph,
  cross_mph`. Append-only; the Actions write them. Rows logged before Oct 29, 2026 have
  only the speed (blank in the newer columns); read them with `weather.read_log`. The
  model uses only `wind_mph`; the gusts show in the game header.
- **Stadium orientation (NFL): `nflmodel/stadiums.py`.** `FIELDS` gives each current home
  field's long-axis bearing (0-179, 0 = north-south) with a `measured` or `approx` tag, and
  `PAST` the location and bearing of outdoor stadiums since 2014 that teams have left, by the
  games file's stadium name. `crosswind(team, mph, dir)` splits a wind into its along-field
  and crosswind parts. Keep the tag honest: `measured` only for a bearing known within about
  10 degrees (checked against a satellite view); `approx` otherwise.
- If a change touches what those Actions build or where it lands, check their steps and
  wording still match the site structure.
- **Player props (NFL) and The Odds API** ([PROPS.md](PROPS.md)). The key is the
  `ODDS_API_KEY` repository secret; never print it or write it to a file. The account has
  20,000 credits a month (Oct 2026). Live logging (DraftKings and FanDuel, ten markets plus
  the three yardage ladders, open, pricing and closing snapshots) runs about 2,500 a month; every call's cost is in
  `data/props/api_usage.csv`. **Ask Ryan before** anything that would use most of the
  month's credits (a backfill), changes what the board shows, or can't be undone.
  `data/props/prop_log.csv`, `prop_log_new.csv`, `prop_log_tags.csv` (the market-only tags,
  by `row_id`) and `prop_results.csv` are append-only: never edit or delete a row. The log's
  header is fixed: `proplog._append` writes an existing file's own columns, so a new column
  needs a sibling file keyed by `row_id`, as the tags and the new projections have. `data/prop_params.json` is written only by `experiments/props_calibration.py`; refit
  it there (methods chosen on 2013-2018, holdout 2019 on, run once) rather than by hand.
  Bet flags are off (`propmarket.FLAGS_ON = False`, Oct 2026): against real 2025 lines the
  model did worse than the books. Turning them back on needs an out-of-sample win over the
  books' prices (PROPS.md) and Ryan's go-ahead.
- **Market lines (both boards): `data/market/`.** The **Market lines** Action
  (`.github/workflows/market.yml`) snapshots every upcoming NFL and college game's spread,
  total and moneyline at Pinnacle, DraftKings, FanDuel, BetMGM and Kalshi through The Odds
  API (`nflmodel/market.py`, `run_market.py snapshot`; 3 credits per league per snapshot, three
  a day plus hourly Sunday night to Tuesday, about 1,400 credits a month) and commits only the
  rows that changed to `data/market/<league>_<YYYY-MM>.csv`. Append-only: never edit or delete
  a row. It shares the `board-commits` group. The board builds read these files and fall back
  to their usual lines when a game has no quotes; a failed snapshot never stops anything.
  Kalshi prices are the contract price; `odds.kalshi_effective_odds` adds the exchange fee
  (taker 7% x p x (1-p) per contract, maker 1.75%) before any EV is computed.
  **Derivatives (both boards, Oct 30, 2026):** the same Action fetches the derivative markets
  (`team_totals`, `spreads_h1`, `totals_h1`, `h2h_h1`; `nflmodel/derivs.py`) into the same
  files on its 14:05 UTC run, one event at a time as the API serves them, under
  `market.DERIV_BUDGET` credits a run (NFL 70 on Thursday, Saturday and Sunday; college 160 on
  Friday, three markets): about 1,500 credits a month, so the whole market run stays under
  about 3,000. `run_market.py derivatives` runs it by hand (the workflow's `derivatives`
  input does both leagues). Don't raise the budgets or add derivative days without checking
  `data/props/api_usage.csv` against the month's 20,000.
  **On the boards:** where Pinnacle quotes a game, its line and two-way prices are the anchor
  the projection is blended from (`slate.with_market`, both boards; the game's `anchor` says
  which) and the consensus line is only the fallback; a game Pinnacle quotes before the
  consensus has a line is on the board already. Every side carries each retail book's price at
  that book's own line (`bets.shop`: `books`, `best*`, and `beats_sharp` when a price beats the
  anchor's no-vig number with no model at all), the page shows the best book, and the archive
  grades each flagged bet's closing line value against Pinnacle's last quote before kickoff
  (`archive.closing_value`; `clv` and `beat_close` in each week's record). The Action also runs
  `run_market.py alerts`: raw model 2+ points off Pinnacle's spread (NFL; 3+ college) or 3+ off
  its total goes to `data/market/alerts.csv` and is posted to the open `opener-alerts` issue.
- **CFBD lines (college board): the `CFBD_API_KEY` repository secret.** Optional. When it is
  set, the **Update college board** Action passes it to the build and `cfbmodel/cfbd.py`
  fetches CollegeFootballData.com's lines for the upcoming weeks; a game ESPN's scoreboard has
  not priced takes CFBD's consensus line, else DraftKings, else the first book listed, and its
  `line_source` says so ("CFBD DraftKings"). ESPN's price always comes first, the cfbfastR
  consensus stays the fallback, and without the key or on any failure nothing changes. Never
  print the key or write it to a file.
- **Hand overrides (NFL board): `data/overrides.csv`.** One row per player per game:
  `starter` sets that game's starting quarterback for the game model and makes him the
  quarterback who takes the team's passes in the player projections (whoever started the
  team's last game, and whatever the injury report says), `out` takes a player out of the
  player projections. Use it only for confirmed news the nflverse schedule or the
  official injury report hasn't picked up yet, and put the source and date in `source`. Rows
  only touch games not yet played, so they lapse by themselves; delete stale rows when you
  see them. Edits go through a PR like any other change (the Actions never write this
  file); after merging, run **Update board** so the board picks them up.
- **Injury designations (college board): `docs/cfb/designations.json`.** The daily college
  check task writes it from Covers.com's NCAAF injury page
  (https://www.covers.com/sport/football/ncaaf/injuries) and merges it through a PR like any
  other change, then starts **Update college board** so the board uses it; the Actions never
  write it. Shape: `{source, url, checked: "YYYY-MM-DD", players: [{team, name, pos, d,
  injury, updated, season}]}`: `team` is the board abbreviation, `name` the board's name for
  quarterbacks, backs and receivers (Covers' own, e.g. "J. Smith", for linemen and defense),
  `d` one of O, D, Q, P, `season` true when he is out for the season. What it does
  (`cfbmodel/designations.py`, upcoming games only): Out and Doubtful players get no
  projection and their usage goes to teammates, a starting quarterback out is replaced by
  the next one in the game model, and all of them are listed under "Out" on the team card;
  Questionable and Probable are only shown as a superscript after the name; linemen and
  defenders change no numbers. Each designation covers only the team's next game (every
  game if `season`), and the pipeline writes it into that game's player status, so the
  superscript shows on that week's row only. A designation the player has played through (a game for his team after its `updated` date) is ignored, since Covers keeps old entries; a newer one, posted after that game, stands. The page and the pipeline ignore the file once `checked` is
  more than 3 days old, and `tests/test_cfb.py` checks it. Ryan approved this one exception
  to "model inputs belong to the Model project" (Oct 21, 2026). The NFL board shows the same
  superscript from its own official-report status.
- **Starting quarterbacks (college board): the CFB dashboard's weekly list.** Each build
  downloads `overrides/qb_starters.json` from rhyman7/cfb (the CFB dashboard's daily task
  checks every FBS team's expected starter on every run) and keeps a copy in
  `docs/cfb/starters.json`; if the download fails, the saved copy is used. What it does
  (`cfbmodel/starters.py`, upcoming games only): when a team's next game is in the list's
  week, the listed quarterback starts that game and the team's later games this season in
  the game model, takes the team's passes in the player projections, and the team's other
  quarterbacks get no designed runs. A listed quarterback with no pass this season counts as
  replacement level only when the list marks the change confirmed or reported. Injury
  designations are applied after it, so a listed starter who is Out or Doubtful is still
  replaced. A list for another season or week changes nothing; `tests/test_cfb.py` checks
  the rules and the saved copy. Ryan approved this second exception to "model inputs belong
  to the Model project" (Oct 22, 2026). Don't edit the copy by hand: fix the list in
  rhyman7/cfb and re-run **Update college board**. The daily value-board check may make that
  fix itself (Ryan's call, Oct 24, 2026): when news confirms that a team's listed starter is out
  or a different starter is named for a game in the next 3 days, it changes only that team's
  entry in rhyman7/cfb's `overrides/qb_starters.json` through a PR there, then re-runs
  **Update college board**. Its task prompt has the rules.
- **Stakes (both boards).** Quarter Kelly, capped at 3% a bet and 5% a game, all sides
  together (`config.MAX_GAME_STAKE`, `slate.cap_game`: every stake in the game is scaled
  down by one factor). With `config.EDGE_HAIRCUT` on, the stake is sized from the model
  probability shrunk toward the market's by the share of claimed edge that has shown up in
  the last eight seasons' graded bets (`backtest.edge_haircut`, `meta.haircut`); the EV and
  probabilities shown are never shrunk. On the NFL board that share is 0 as of Oct 29, 2026,
  so no NFL stake is sized until it moves (FEATURES.md, "Edge tests, late October 2026");
  turning the switch off is Ryan's call. **The haircut fails closed** (`pipeline.haircut_factor`):
  it is only measured on a full-backtest run, and a quick refresh with no measured factor on
  hand (the first refresh after a change, or a lost `meta.haircut`) sizes no model stake at
  all rather than every stake at the full claimed edge. After merging anything that touches
  the graded bets, run **Update board** and **Update college board** with the backtest
  input ticked so the factor is measured again. A price bet (below) is sized against the
  sharp line, so it carries a stake whatever the factor; opener and wind bets the model
  sizes nothing for get a `bucket_stake` from their bucket's own record
  (`sheet.bucket_stakes`: quarter Kelly on the bucket's win rate shrunk halfway to
  break-even, at the book's net price).
- **The bet sheet (both boards): `nflmodel/sheet.py`.** Every bet on a game entry carries
  `tier`, `gap`, `opener_gap`, `opener_src`, `move` and `why` (the docstring has the rules).
  The tiers are what the page's Bet sheet groups by: `price` (a retail price beats
  Pinnacle's no-vig number by `PRICE_MIN`, 1%, with no model), `opener` (the model 2+ points
  off the opener, 3+ in college and on totals), `wind` (NFL only: an under in 10-19 mph), and
  `model` (anything else +EV on a spread or total). **Moneylines are never a bet outside the
  price tier**: on the graded bets since the holdout seasons they have lost at every claimed
  edge. The opener is Pinnacle's first stored quote when the market log caught the open (its
  first snapshot no later than the day the board first saw the game), else the board's first
  line, else the data's opener. Each game also carries `kickoff` (UTC), `sharp_path`
  (Pinnacle's spread and total by snapshot, `market.path`), and `alts`/`teasers`
  (`nflmodel/alts.py`: P(cover) at every half point within 7 of the line and with 6 or 7
  points added, on the market curve and the projection) and `derivs` (`nflmodel/derivs.py`:
  every book's team totals and first-half lines priced off the curves, bet-shaped records
  with `deriv: true`; a retail price that beats Pinnacle's own derivative quote by
  `PRICE_MIN` is a `price` tier bet, nothing else in them is a bet, and the page lists them
  in the game's Derivatives table); `meta.inputs` says when the board,
  the book prices, the injury report (NFL) and the designations (college) were last read.
  `meta.half` is the first-half curve's fit (`derivs.fit_half`, on the half-time scores in
  `data/cache/half_scores_<season>.csv` and `data/cfb/cache/half_scores_<season>.csv`, which
  the data loaders write and the Actions commit with the rest of the cache).
  The page's timing verdicts (bet now / can wait / wait or size down) are in `timing()` in
  `site/template.html`; keep them consistent with the sheet's wording if either changes.
- **The ledger (both boards): `data/ledger.csv`, `data/ledger_results.csv`, and the same
  under `data/cfb/`** (`nflmodel/ledger.py`). Every bet the sheet flags (a stake on the shown
  basis, or a price, opener or wind tier) is written down the first time it is flagged, at
  the retail book, line and net price it would be taken at, and never changed; results are
  appended once the game is played (W/L/P, units at the price taken, and closing line value
  against Pinnacle's last quote before kickoff through the key-number curve). Derivative rows
  (market `Team total`, `1H Spread`, `1H Total`, `1H Moneyline`) are graded on the team's points
  or the half-time score and carry no closing line value yet. Append-only:
  the Update Actions write them, never edit or delete a row by hand, and a local build run
  must not commit its rows (they would be sandbox prices). `data.json` carries the summary as
  `ledger` (records overall, by tier, by market and by week, plus every row) and the Model
  record page leads with it as **Live record**; `tests/test_sheet.py` checks the files.
- **The spread and totals card shows every model input.** A game page's "How the spread is
  built" rows add up to the raw model spread and total (`matchup.breakdown()`,
  `cfbmodel/board.py`). A feature added to `MARGIN_FEATS` or `TOTAL_FEATS` needs a row
  (`MARGIN_ROWS`/`TOTAL_ROWS` in `nflmodel/matchup.py`, `uncovered()` on the college
  board) or the tests fail; the tests also check that Adjusted lies between market and model.
- **Out means no projection, on both boards.** A player Out, Doubtful or on injured reserve
  for a game never has a projection for that game: the pipelines leave him out (his usage
  goes to teammates) and list him under "Out" on the team card, the page drops any player row
  whose status says otherwise, and `tests/test_cfb.py` checks both boards' `data.json`. The
  superscript comes from the board data, not from fetching `designations.json` (that file
  only adds the injury and date to the tooltip), so it always matches the projections.

## Site structure

- `docs/index.html` and `docs/cfb/index.html` are **generated**. Don't edit them by hand;
  edit the template and rebuild:
  - NFL: `site/template.html`, built by `python run.py` (`--no-backtest` for a quick refresh)
  - College: `site/cfb-template.html`, which is itself generated from the NFL template by
    `python site/make_cfb_template.py`. Change `site/template.html`, then re-run that
    script; if it stops, fix the college substitution it names rather than editing
    `cfb-template.html` directly.
- The pipelines inject the week's data into the template in place of `/*__DATA__*/null`;
  keep that placeholder.
- `docs/archive/` holds each week's predictions frozen at kickoff. Never rewrite past weeks.
  Grading adds the results: the final score, each bet's result, and under `actuals` the whole
  box score (every player with a touch; one not projected before the game carries his name,
  pos and team), with `full` listing the games whose actuals are complete. The Update Actions
  fill `full` in for any graded game still missing it. `index.json` carries each week's record
  and, as `pmiss`, the player projections' average miss and bias by position and stat
  (`nflmodel/archive.player_misses`), which the Past weeks page's **Season to date** sections
  read, so the page never has to load every week's file for the season view.
- Both boards share the same sections and anchors, in this order:
  This Week's Matchups `#matchups` | Games `#games` | Players `#players` | Past weeks `#history` |
  Team ratings `#ratings` | Model record `#model`. Each board's nav links to the other board.
- **Best bets (both boards): `#bestbets` and `best-bets.pdf`** (Oct 30, 2026). The week's card
  on one document, built from the whole board by `renderBestBets()` in `site/template.html`:
  the Bet sheet's bets in tier order with book, price, EV, stake, why and when; the 6-point
  teaser legs that clear −120 and the best pairs; every game's market, model and adjusted
  numbers with the board's lean; the week's prop flags (NFL; book disagreements when none
  are flagged); the bucket records and the disclaimer. The Bet sheet's head links to it
  (**Download the best bets (PDF)** and **Print view**), the landing page's cards link the
  PDFs, and the view's own buttons are Download PDF, Print and Back. Printing uses the
  portrait `#bb-page` style, switched on by `route()` only for `#bestbets`, and the
  `html.bb-print` rules (black on white, nothing but the card). The PDFs
  (`docs/best-bets.pdf`, `docs/cfb/best-bets.pdf`) are rendered by `scripts/build_pdf.py`
  (Playwright Chromium, from the generated page on disk) in the Update Actions after each
  build and committed with `docs/`; a render failure never stops the build. They show the
  board's first listed week at a $1,000 bankroll; the Print button prints whichever week and
  bankroll the reader has selected. `#bestbets` is not a nav section (the nav order above
  stands); don't add it to the landing page's section lists beyond the PDF link.
- **The Games view's order** (Oct 30, 2026): filters, **Bet sheet**, the boost calculator,
  **Teasers and alternate lines** (legs from `teasers`, the two-leg price entered by the
  reader; the per-game ladders are on each game's page under "Alternate lines and teasers"),
  then Model against the market, the game cards and Every price. The Model record view leads
  with **Live record** (the ledger), then the backtest with the holdout columns first. The
  Props view hides lines on players the board has ruled out and passing lines with no
  projection, and says how many.
- **Props `#props` is NFL only** (Ryan's call, Oct 2026), between Players and Past weeks. Its
  markup, styles and script in `site/template.html` sit between `<!--NFL-ONLY-->` and
  `<!--/NFL-ONLY-->` (HTML) or `/*NFL-ONLY*/` and `/*/NFL-ONLY*/` (CSS and script), which
  `site/make_cfb_template.py` cuts from the college page; it checks the number of blocks, so
  update its count when adding one. The section reads `docs/props/` (one file per week,
  `index.json` for the list), written by `nflmodel/propboard.py` on every props run; it never
  edits past weeks by hand, since the files are rebuilt from the append-only log.
- **This Week's Matchups** copies the matchup dashboards' look on purpose (graphite colors,
  condensed Archivo, team-color blocks lettered in the secondary color, never black): the
  board of this week's games, and `#matchups/<game id>` with each team's card showing this
  game's projections, not season stats. While it is open, `data-mx` on `<html>` switches the
  page tokens to graphite; every other section keeps the board's own look. Its styles are
  all under `.mx`. If the dashboards' week board or team card changes look, bring these along.
- **Past weeks on the Matchups board.** The board's Week select lists the archived weeks
  before this one; `#matchups/<season>_w<week>` (e.g. `#matchups/2026_w07`) is that week's
  board and `#matchups/<season>_w<week>/<game id>` one of its games, read from
  `docs/archive/<season>_w<week>.json` (the Past weeks table's game names link there too).
  Nothing is rebuilt for it: the page shows the predictions as frozen at kickoff, and once the
  game is graded the final score and each projected player's actual line under the projection,
  in blue (`--act`). Season ratings and efficiency ranks are left off past weeks (they are
  today's, not that week's); the result badge and the margin/total comparison take their place.
  This week's board and `#matchups/<game id>` are unchanged, which the dashboards' packets
  rely on. Players in the box score who were not projected are listed too, tagged "Not
  projected", and count in the team's actual yards.
- Each matchup (`#matchups/<game id>`) has a **Print matchup** button, like the dashboards'.
  It prints both team cards side by side on one landscape letter page in black on white, with
  a printed heading (week, teams, kickoff, stadium, line) and the disclaimer. The print rules
  are the `@media print` block after the `.mx` styles and only apply while a matchup is open
  (the landscape `@page` lives in the `#mx-page` style, which `route()` turns on only for a
  matchup). Don't use a named `@page` for this: Chrome prints a blank sheet before and after
  a named page. Other sections print as before.
- The matchups board (`#matchups`) has a **Print all** button that prints every game on the
  board, one landscape page per game, in the same layout as Print matchup. It follows the
  board's team/conference filter (the label becomes "Print N shown"). The games are built
  into `#mx-print-all` only for the print (`html.mx-printing` shows them in place of the
  board) and cleared on `afterprint`. Both prints share `gameBlock()` and the `.mx-game`
  print rules, so a change to one game's printed layout applies to both. Ctrl+P on the board
  without the button still prints the board list.
- **Favicon:** the RH monogram, as `favicon.ico`, `favicon-192.png` and `apple-touch-icon.png` at the repo root (the same three files in rhyman7/nfl, rhyman7/cfb, rhyman7.github.io and rhyman7/fb-value-board). Every page links all three in `<head>`, right after the viewport meta; a new page gets the same three lines. The boards are one and two folders below the root, so the template links `../favicon.ico` and so on, and `site/make_cfb_template.py` changes those to `../../` for the college board.

## Checks before merging

- `python -m pytest -q` passes (required for anything touching `nflmodel/`, `cfbmodel/` or
  the pipelines). `pytest.ini` limits collection to `tests/`: the `experiments/*_test.py`
  scripts are run by hand, and a `test_*` function in one of them would otherwise break the
  Update board Actions' test step.
- If a template changed: `python site/make_cfb_template.py`, then rebuild both boards
  (`python run.py --no-backtest`, `python run_cfb.py --no-backtest`) and commit the generated
  `docs/` files with the template.
- Serve the repo root (`python -m http.server`) and load the landing page and both boards'
  sections you touched at desktop and phone widths with no page errors.

## Landing page (`index.html`)

- Titled **Weekly Value Boards**. Same layout, tokens, Archivo type and NCAA/NFL logos as
  the Weekly Gridiron Board (rhyman7.github.io); keep it styled consistently with it.
- College card on the left, NFL on the right. Each card links to its board and to the six
  sections above (the NFL card also to Props); if a section is added, renamed or reordered,
  update both cards. Each card's first link is its board's **Best Bets (PDF)**
  (`docs/best-bets.pdf`, `docs/cfb/best-bets.pdf`), which the Update Actions write.
- Logos live in `img/` (copies of rhyman7.github.io's `img/ncaa.png` and `img/nfl.png`).
- All links are relative (`docs/`, `docs/cfb/`, `img/`) so the page works at
  `/fb-value-board/`.
- A URL with a hash (e.g. `/fb-value-board/#games`) forwards to the NFL board, so links
  from when the root redirected there keep working. Keep that script.
- The Actions don't touch this file.

## Out of scope here

- This repo is the betting model as well as its site. Model, prediction and betting-logic
  work belongs to the CFB/NFL Model project; page, layout and landing-page work belongs to
  the Dashboards project.
- Exception (Ryan's call, Oct 29, 2026): the defensive-player study (FEATURES.md, "Defensive
  players, re-tested and rated", scripts `experiments/def_*.py` and `cfb_def_test.py`) is
  done here, and anything from it that earns a place goes on both boards' games, cards and
  player projections through its own PR. Ryan sees the results and the plan before anything
  that changes the board is merged.
- Every page keeps the disclaimer: for research and entertainment, past results do not
  predict future results, and nothing here is betting advice.
