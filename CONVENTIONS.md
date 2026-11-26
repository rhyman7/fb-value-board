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
- **What the dashboards read from here** (`docs/feed/`, written by every board build): the
  matchup feed `<league>.json` (`nflmodel/dashfeed.py`; it carries only the week the board is on,
  `meta.week`, even when the board also lists next week's players, Nov 25, 2026; when the week changes, the last feed of the
  week just left is kept as `<league>_prev.json`, which the dashboards' reports read for that week's
  late games, Nov 14, 2026; each player's `R` holds his yardage ladders from the live prop logs,
  per book the main line and the next three alternate overs below it, which the reports show and
  track for Best Overall Matchups) and the Likeliest TD Scorers tracker
  `td_tracker_<league>.json` (`nflmodel/tdtracker.py` for the NFL, also run by the props run;
  `cfbmodel/tdtracker.py` for college, from Nov 4, 2026). The two trackers write the same shape,
  which each report's `build_report.py` turns into its Touchdown tracker page: keep them in step,
  and keep each one's eligibility rules the same as its report's `td_rows`. The college tracker
  reads the board's archived weeks (chances frozen at kickoff, box scores in `actuals`; weeks the
  walk-forward backfill wrote are marked "rebuilt") and the live college props for prices.

## How changes get made

- **Pull requests, merged after checks.** Don't push hand edits straight to `main`: `main` is
  the live site. Open a pull request (one topic per PR), run the checks in "Checks before
  merging", and merge it yourself once they pass (squash, delete the branch). Then tell Ryan
  in a line or two what went live, with the PR link. If a check fails, don't merge: fix it, or
  leave the PR open and tell Ryan why.
- **Branches.** The repo deletes a pull request's branch when it merges (Settings, Nov 3,
  2026). For leftovers, run the **Delete merged branches** Action (`delete-merged-branches.yml`)
  by hand: dry run first (the default), then untick it. It only deletes branches with nothing
  that isn't on `main`, never one with an open pull request.
- **After merging**, confirm the Pages deploy for the merge commit succeeded and load the live
  board(s) you changed. If something broke, revert right away (a revert PR, merged the same
  way) and tell Ryan.
- **Both boards.** Design and feature changes go to both the NFL and the college board
  unless Ryan says otherwise. Don't call a change finished until both are done.
- **Ask first** if anything is unclear or a change could break the live site.
- **One site with the dashboards (Ryan's call, Nov 20, 2026).** The value boards, Parlays and the
  Weekly Review are sections of the Weekly Gridiron Board (rhyman7.github.io), which lists them on
  its hub page. This replaced the earlier "kept separate" rule.
- **The manual loads the site bar too (Nov 26, 2026)**, as `<script src="/site-bar.js" data-shell="2"></script>`,
  so it has the Back arrow inside the home-screen app. Its text sits in `<div class="page">` (the
  width and padding that used to be on `<body>`) so the bar spans the full screen; print is unchanged.
- **Site bar (Nov 20, 2026).** Every page here (the landing page, both boards, Parlays, the Weekly Review, Recommended) loads the Weekly Gridiron Board's site bar as the first
  thing in `<body>`: `<script src="/site-bar.js"></script>`. The script lives in
  rhyman7/rhyman7.github.io and writes the strip that links the hub, both matchup dashboards,
  both value boards, Parlays and the Weekly Review (Basketball as a side link). Change the bar
  there, not here; a new page gets the same line. It is hidden in print, and a page opened from
  disk or a local server simply has no bar.
  The generated boards get the line from `site/template.html`.
- **Game packets.** The dashboards' **game packets** include this board's
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
  **Exception: the College availability Action** (`cfb-availability.yml`) has its own group,
  `cfb-availability` (Nov 11, 2026). In `board-commits` a newer board run waiting its turn
  replaced it, so a Game Day report could miss kickoff. It commits only its own files and
  rebases and retries its push. Besides its GitHub schedule, the **Watchdog** Action (below, under
  the Market lines Action) starts it in the Game Day report hours (Tue-Fri 3:20-8 pm ET, Saturday
  9:20 am-10 pm ET) unless it ran in the last 40 minutes, because GitHub dropped most scheduled
  runs on Nov 10-11. Until Nov 25, 2026 two Claude scheduled tasks did this.
- **Push retries (Nov 24, 2026).** Every Action that commits to `main` pushes in a loop: pull
  `--rebase` and push, up to 4 tries 15 s apart, then fail the run (a QB starter check push
  failed on a passing GitHub "Internal Server Error"). College availability keeps its own
  5-try loop. A new Action that pushes copies the same loop.
- **Play-by-play values hold switch (`config.HOLD_PBP_VALUES`, normally False).** While True,
  `data.load_pbp_aggregates` and `extra.load_extra` do not re-download the current season, so its
  `team_games`, `qb_games`, `stat_games`, `team_extra` and `qb_extra` tables stay on their cached
  values while everything else refreshes; games finished in the meantime are missing from those
  tables. Used Nov 23-24, 2026, when nflverse republished its play-by-play with different EPA and
  success values for plays already on file (same play counts; pass EPA down, rush EPA up). Ryan
  lifted it after the old-vs-revised comparison (Project doc
  `claude/nflverse-epa-revision-check-2026-11-23.md`): game projections moved 0.15 points at most.
  The cached earlier seasons were not rebuilt and still differ from nflverse's current files;
  rebuilding them is a model test under the usual rule, not a refresh.
- **The "Browser for the best-bets PDF" step never fails a board build** (`continue-on-error`,
  and it falls back to installing Chromium without system packages). On Nov 11, 2026 Ubuntu's
  mirrors were unreachable from the runners and every Update board / Update college board run
  failed there for hours. `scripts/build_pdf.py` skips the PDF without a browser.
- **Shadow test (NFL): `data/shadow/`.** The **Update board** Action also writes
  `defense_games.csv` and `defense_players.csv` (`nflmodel/shadow.py`): the game model and two
  player projections refitted with the defensive-player inputs, logged and never shown on the
  board. A game's rows are rewritten until kickoff and never after; don't edit or delete rows
  by hand. The inputs and the rule are fixed until the test is judged (after the 2027 season,
  FEATURES.md "Shadow test"): don't change `shadow.MARGIN`, `TOTAL` or `PLAYER`, and score it
  with `experiments/shadow_score.py`. A shadow failure never stops the board build. A second
  spread version, pass catchers out (`shadow.MARGIN_PC`, from Oct 29, 2026), is logged the same
  way to `passcatchers_games.csv` under the same rule: don't change `MARGIN_PC` either.
- **The Recommended list (both leagues): `data/recommended/`, `docs/recommended/`** (`nflmodel/recommended.py`,
  Nov 20, 2026, Ryan's call; the rules as approved are in the Project doc
  `claude/plan-recommended-list-2026-11-20.md`). The public record of picks the site stands behind:
  six *tracks*, each one kind of pick under one rule fixed for the season (college spreads at the
  opener from week 5, college totals at the opener, NFL spreads at the opener, NFL price with the
  model agreeing, Wong underdog teasers, NFL prop unders the model agrees with). **Every track
  starts on trial.** One was stopped on its first day: NFL prop unders (`STATUS` "held", Nov 20, 2026).
  A held track publishes no new picks; the picks it had already published stay in the files, are
  graded and count in the totals, and it shows on the page as "Stopped". It starts again only on
  Ryan's word. `recommended.STATUS` is changed by hand at a season review only (100+ graded
  picks across a full season; return above zero by 1 standard error and closing line value not
  below zero for Recommended, 2 standard errors for Proven), never mid-season and never by code.
  **Don't change a rule constant, a track or `STATUS` without Ryan**; a rule change waits for the
  next season and starts a new track key so the old record stays.
  Every Update board and Update college board run (`recommended.update_games`) and every Player
  props run (`recommended.update_props`) publishes each pick the first time its rule is met before
  kickoff, at the best of DraftKings, FanDuel and Kalshi (`ledger.MY_BOOKS`; Kalshi from Nov 26, 2026, at its price after the fee, a dated change to the pre-set rule made at Ryan's request; teasers and props stay DraftKings and FanDuel, `ledger.SPORTSBOOKS`), to
  `data/recommended/<league>_picks.csv`, grades finished games into `<league>_results.csv`
  (`ledger.grade_frame` for game lines; teasers at -120; props from the Props section's results)
  and rewrites `docs/recommended/<league>.js`. Both CSVs are **append-only, written by the Actions
  only**: never edit or delete a row, never backfill, and never commit rows from a local build
  (they would be sandbox prices). A failure never stops a build. No extra Odds API calls: it reads
  the slate and the Props files the build already made.
  **Rolling total:** $10 (`recommended.UNIT`) on every pick; a bet that lands on two tracks counts
  once in the totals and in each track's own record.
  **Page:** `docs/recommended/index.html`, hand-made (not generated): the record page, and each
  week's report per league at `#<league>/<season>/<week>` (picks added as they are published,
  results after the games). It loads the two `.js` files as scripts so it also works from disk.
  `scripts/build_pdf.py` (`render_recommended`, run by both Update Actions) writes each report as
  `docs/recommended/<league>_<season>_w<WW>.pdf`: the newest two weeks every run, older weeks only
  when missing. The Weekly Gridiron Board's site bar and hub link to the page. What the product may
  say about the record (live figures with counts and ranges only; no "accurate" or "winning" for a
  track that is not Proven) is in the Project doc.
- **Wong teaser record (NFL): `data/teasers/`** (`nflmodel/teaserlog.py`, from Nov 6, 2026; the
  Project doc `claude/betting-strategies.md`, strategy 2). Every **Update board** run logs each side
  whose DraftKings, FanDuel or anchor line is in a Wong range (underdog +1.5 to +2.5, favourite -7.5 to
  -8.5) to `nfl_log.csv` (a row whenever its lines or total change, until kickoff; append-only),
  grades it after the game into `nfl_results.csv` (append-only) and writes the summary
  `nfl_record.json`. It reads the slate and writes only `data/teasers/`; the board shows none of it,
  and a failure never stops the build. The strategy's rule (underdog legs, total 49.5 or less, two-team
  teasers at -120 or better) is fixed in the module's constants; change it only with Ryan.
- **Offensive-line watch (NFL): `data/ol_watch/`** (`nflmodel/olwatch.py`, from Nov 13, 2026; Project
  doc `claude/plan-unit-tracker-this-week-2026-11-13.md`). A paper record of the one lead from the
  Nov 11 unit check: teams missing two or more of their usual five linemen. Every **Update board** run
  logs each team with two or more of its `data/cache/ol_core_<season>.csv` linemen out (report,
  reserve list, or a confirmed-news row in `data/overrides.csv`) to `nfl_log.csv` (a row whenever the
  linemen out or the team's spread or total change, until kickoff; append-only), grades it after the
  game into `nfl_results.csv` (append-only: points against the implied team total and cover, at the
  first flag and at the last line before kickoff) and writes the summary `nfl_record.json`. It writes
  only `data/ol_watch/`; the board shows none of it, no bet is flagged from it, and a failure never
  stops the build. The reading rule is in the module's docstring (read after the 2027 season; a
  model test only if the shortfall against the flag-time team total reaches 2 standard errors);
  change it only with Ryan. The "NFL late availability" tasks mention these teams in their report
  ("Offensive-line watch") but write nothing here.
- **Forecast log (both boards): `data/forecast_log/<league>_<season>.csv`** (`nflmodel/forecastlog.py`,
  from Nov 11, 2026; Project doc `claude/plan-uncertain-qb-test-2026-11-11.md`). Every **Update board**
  and **Update college board** run appends a row for each game that has not kicked off when its line,
  either quarterback or his status changed, or its forecast or weight moved past a small tolerance
  (`forecastlog.TOL`: the model is refitted every build and wobbles by a hundredth of a point): the
  raw model, the line it was blended with, the blend shown, the weight, both quarterbacks and, on the
  NFL board, each quarterback's level on the week's injury report and Friday practice. Append-only:
  never edit or delete rows. The board shows none of it and a failure never stops a build. Why it
  exists: `docs/archive/` keeps only the last refresh before kickoff, so the opener and midweek
  forecasts lived only in git history; a test of the blend weight at the times the board is used
  reads this log. Rows marked `backfill` were rebuilt from that history on Nov 11, 2026
  (`scripts/forecast_log_backfill.py`, one-off; NFL from Oct 7, college from Oct 19) and carry no
  quarterback status. `qb_flag` is deliberately not in the archive's game keys: a past week's page
  would start showing the tag.
  **Inputs (from Nov 19, 2026):** with each row written, the raw game-model inputs the build priced that
  game from (every margin and total feature plus each side's quarterback delta) go to
  `data/forecast_log/<league>_<season>_inputs.csv` in long form (`at, game_id, input, value`; joined to the
  forecast row on `at` and `game_id`), so a feature-list change never breaks the file. Same rules:
  append-only, never shown, a failure never stops a build. It is the record of what the board knew at
  bet time, for fitting and scoring the game model on bet-time information (Project doc
  `claude/plan-bettime-information-tests-2026-11-19.md`); the backtest's own inputs know the actual
  starters and game-day inactives.
- The **Player props** and **Prop price backfill** Actions (`props.yml`, `props-backfill.yml`)
  also commit straight to `main`, only under `data/props/` (the prop log, results, snapshots,
  credit usage and past prices) and, for Player props, `docs/props/` (the board's Props files). They share the `board-commits` group too. The backfill runs
  only by hand; to resume one, start a new run with the same inputs, never **Re-run** (it
  checks out the old commit and fetches the season again, ~16,000 credits).
- The **Market line backfill** Action (`market-backfill.yml`, by hand only;
  `nflmodel/marketbackfill.py`) commits only `data/market/history/` and
  `data/props/api_usage.csv`, in the `board-commits` group: past Pinnacle, DraftKings and
  FanDuel game lines for one league-season per run, an "open" snapshot each Monday 12:00 UTC
  and a "close" before each cluster of kickoffs. Its `plan` input (on by default) prints the
  snapshots and their cost and spends nothing; **ask Ryan before a run that spends credits**.
  One file per snapshot, never edited; a header-only file is a snapshot the API had nothing
  for. Resume a stopped run with a new run and the same inputs, never **Re-run**. The boards
  never read these files; `experiments/pinnacle_grade.py` does.
- **Frozen test fits: `experiments/frozen/`.** A pre-set test that is scored on weeks still to
  be played freezes its fit here before those weeks are scored (first: the NFL floor-rung
  re-reading, `rung_recalibration_map.json`, Nov 18, 2026, fitted on data through 2026 week 10
  and scored on weeks 11-13; Project doc `claude/plan-rung-recalibration-test-2026-11-18.md`).
  Second: the main-line curve for floor rungs, `market_ladder_shape.json` (Nov 18, 2026, fitted on
  2025 only, scored on 2026 weeks 8-13; `claude/plan-market-ladder-test-2026-11-18.md`).
  Never edit or refit a frozen file; the script refuses to overwrite one. Only coefficients go
  here, never PFF rows. The boards do not read these files.
- The **Outside ratings** Action (`outside-ratings.yml`, by hand only) runs
  `experiments/fetch_outside_ratings.py` and commits only `data/cfb/outside/`: The Prediction
  Tracker's college season files (`ncaa<season>.csv`, every tracked system's predicted margin per
  game, recorded before kickoff), this week's file under `live/<date>.csv` (kept so the archive can
  be checked for after-the-fact revisions) and `manifest.json` (source, time, sha256, columns per
  file). The files are saved byte for byte as served; never edit them. They exist for the pre-set
  test in the Project doc `claude/plan-cfb-outside-ratings-test-2026-11-20.md`; the boards and the
  model do not read them. The source says its predictions are for research and informational use,
  so nothing built on these files goes into a product that is sold until the data-rights check.
- The **Wind history** Action (`wind-history.yml`, by hand only) runs
  `experiments/fetch_wind_history.py` and commits only `data/cache/wind_history.csv`: the
  archive wind at kickoff (speed, gusts, direction, along-field and crosswind parts) for
  every played outdoor NFL home game since 2014, one Open-Meteo archive call per stadium per
  game day, resumable. It shares the `board-commits` group. The test that reads it is
  `experiments/wind_direction_test.py` (FEATURES.md "Wind direction and gusts").
- The **Forecast history** Action (`forecast-history.yml`, by hand only) runs
  `experiments/fetch_forecast_history.py` and commits only `data/cache/forecast_history.csv`: the
  rain and snow *forecast* for every played outdoor NFL home game since 2022 from Open-Meteo's
  forecast archives (game-morning forecasts from 2022; 1, 2 and 3 days ahead from 2024, the
  earliest those exist), summed over the board's game window, up to two calls per stadium per
  season, resumable. It shares the `board-commits` group. Only
  `experiments/rain_forecast_backtest.py` reads it (FEATURES.md "Rain and snow"); the boards and
  the model do not.
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
  20,000 credits a month (Oct 2026; 100,000 for one month from Oct 30, 2026 for the backfills). Live logging (DraftKings and FanDuel, ten markets plus
  the three yardage ladders, open, pricing and closing snapshots) runs about 2,500 a month; every call's cost is in
  `data/props/api_usage.csv`. **Ask Ryan before** anything that would use most of the
  month's credits (a backfill), changes what the board shows, or can't be undone.
  `data/props/prop_log.csv`, `prop_log_new.csv`, `prop_log_tags.csv` (the market-only tags,
  by `row_id`) and `prop_results.csv` are append-only: never edit or delete a row. (One sanctioned correction: Nov 9, 2026, `scripts/regrade_kneels.py` moved 99 quarterback rushing rows to the official line; PROPS.md "Kneel-downs and rushing props".) Rushing yards and rush attempts settle with kneel-downs counted (`players_data.SETTLE`): `carries` and `rush_yds` in the box scores stay without them for the projections, and anything that settles a rushing prop adds `kneels` / `kneel_yds`. A quarterback's rush-attempts prop is also priced with his expected kneel-downs (`nflmodel/kneels.py`, `props.OFFICIAL_KEY`; PROPS.md): only that prop reads them. Don't add them to `carries_proj` or show them as his carries, and don't move rushing yards the same way (tested Nov 9, 2026: worse on every measure). A starting quarterback's carries and rushing yards are projected from his starts (`players.QB_RUSH`, PLAYERS.md "Quarterback rushing per start"); his passing already was. The log's
  header is fixed: `proplog._append` writes an existing file's own columns, so a new column
  needs a sibling file keyed by `row_id`, as the tags and the new projections have. `data/prop_params.json` is written only by `experiments/props_calibration.py` (and its `shade` key by `experiments/props_shade_fit.py`, the over-lean shade on the books' number, PROPS.md; refit at the January recheck only); refit
  it there (methods chosen on 2013-2018, holdout 2019 on, run once) rather than by hand. The
  production widths are fitted on seasons from `FIT_FROM` (2019, Oct 31, 2026: backs' outcomes
  changed then; PROPS.md "The low end, refitted on recent seasons"); `--production` refits only
  them and keeps the correction and correlations.
  Bet flags are off (`propmarket.FLAGS_ON = False`, Oct 2026): against real 2025 lines the
  model did worse than the books, and a market-anchored blend tested out of sample on 2025
  (Oct 30, 2026, `experiments/props_blend.py`) did not earn them back. Turning them back on
  needs an out-of-sample win over the books' prices with closing line value to match
  (PROPS.md) and Ryan's go-ahead.
- **Market lines (both boards): `data/market/`.** The **Market lines** Action
  (`.github/workflows/market.yml`) snapshots every upcoming NFL and college game's spread,
  total and moneyline at Pinnacle, DraftKings, FanDuel, BetMGM and Kalshi through The Odds
  API (`nflmodel/market.py`, `run_market.py snapshot`; 3 credits per league per snapshot, three
  a day plus hourly Sunday night to Tuesday, about 1,400 credits a month; plus, from Oct 31,
  2026, hourly game-day snapshots of the league playing: college Saturdays 13-23 UTC, NFL
  Sundays 13-20 UTC and Thursday nights 21-22 UTC, about 360 credits a month, each followed
  by a quick rebuild of that board dispatched from the Action, `actions: write`; from Nov 3,
  2026 any snapshot that sends a line-movement alert also rebuilds that league's board; from Nov 3,
  2026 the **Kickoff refresh** Action, `kickoff.yml` + `nflmodel/kickoff.py`, dispatches a
  `league`-scoped snapshot and that board's rebuild 35 and 20 minutes before each kickoff window,
  about 150 credits a week in a full week, retrying a failed run until 10 minutes past its time;
  after each window's 20-minute refresh it sends that window's Best bets card from
  `bestbets.csv` to ntfy and the week's `best-bets` issue (one per Tuesday-to-Monday week, the old
  one closed by the new week's first alert), nothing when the card is empty, at
  DraftKings, FanDuel and Kalshi only (`kickoff.MY_BOOKS`, which is `ledger.MY_BOOKS`; Kalshi added Nov 26, 2026 and shown at its price after the fee; a bet none of them qualifies is listed as NOT A BET); it only
  dispatches existing workflows, reads and sends, and never changes the model) and commits only the
  rows that changed to `data/market/<league>_<YYYY-MM>.csv`. Append-only: never edit or delete
  a row. It shares the `board-commits` group. The board builds read these files and fall back
  to their usual lines when a game has no quotes; a failed snapshot never stops anything.
  **Backup for skipped runs (Nov 11, 2026):** GitHub often drops the three daily snapshots (02:05,
  14:05, 20:05 UTC; on Nov 11 the board was built on prices 8 hours old), so the **Watchdog**
  Action (`watchdog.yml`, `scripts/watchdog.py`, Nov 25, 2026; before that a Claude scheduled task,
  "Market lines backup") checks from 40 minutes after each one and starts one plain snapshot (no
  derivatives, no alternates, about 6 credits) unless a Market lines run created since the slot
  succeeded or one is queued or running. A failed run is retried, at most twice per slot. It
  starts nothing but this and the College availability check, and edits nothing. **The Watchdog
  keeps itself going:** each run ends by waiting until the next check is due and starting the
  next run, since a schedule can't fix a schedule that doesn't fire; a two-hourly cron only
  revives the chain. To stop it, set the repository variable `WATCHDOG_CHAIN` to `off`. If the
  three daily times or the report hours change, change `MARKET_SLOTS` / `availability_window`
  in the script (and `tests/test_watchdog.py`) with them.
  Kalshi prices are the contract price; `odds.kalshi_effective_odds` adds the exchange fee
  (taker 7% x p x (1-p) per contract, maker 1.75%; the schedule and its date are cited in
  `odds.py`, check it each season) before any EV is computed. Every Kalshi quote on a side
  (main markets and derivatives, `bets.with_post_at`) carries `post_at` and `post_match`,
  the maker prices in cents (`odds.kalshi_post_at`), computed against the chance the stakes
  are sized from (`p_post` on each quote, `bets.stake_prob`: the model shrunk toward the sharp
  no-vig number by the haircut, so Pinnacle's own number on the NFL board while its haircut
  is 0); derivative quotes, which carry no `p_post`, use the model's chance. The page shows them as "Post on Kalshi
  at Xc" with a tooltip saying a resting order may not fill. The ledger still records a
  Kalshi bet at the taker price it was flagged at.
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
  `run_market.py alerts` (Oct 31, 2026): every side the model is off the opener by the
  `sheet.MOVE_GAP` threshold, measured at Pinnacle's newest stored line (`sheet.line_moves`),
  is written to `docs/moves.json` and `docs/cfb/moves.json` (the boards' **Line movement**
  section reads them when their `board` stamp matches the published board, else the
  `slate.moves` the build wrote). Sides with a backtested edge (`sheet.bucket_record`: the
  opener report's highest threshold the opener gap clears, when its holdout ROI is positive)
  that are new or change status (Live, Partly gone, Missed) are alerts: appended to
  `data/market/gaps.csv`, the last status per side kept in `data/market/gaps_state.json`
  (both written only by the Action), pushed to the phone through ntfy when the `NTFY_TOPIC`
  secret is set, and posted to the open `opener-alerts` issue with an @-mention of the owner.
  `data/market/alerts.csv` is the old raw-model log, no longer written. Step guards in the
  workflow test `/tmp` files with the shell, never `hashFiles()` (which only sees the
  workspace).
- **CFBD lines (college board): the `CFBD_API_KEY` repository secret.** Optional. When it is
  set, the **Update college board** Action passes it to the build and `cfbmodel/cfbd.py`
  fetches CollegeFootballData.com's lines for the upcoming weeks; a game ESPN's scoreboard has
  not priced takes CFBD's consensus line, else DraftKings, else the first book listed, and its
  `line_source` says so ("CFBD DraftKings"). ESPN's price always comes first, the cfbfastR
  consensus stays the fallback, and without the key or on any failure nothing changes. Never
  print the key or write it to a file.
- **College player projections, defense against each position.** `cfbmodel.players.walk_forward`
  ends with `nflmodel.dvp` applied to `players.DVP_PAIRS`, fitted on the seasons it projected
  before each one (it projects `DVP_FIT_SEASONS` seasons before `first_test` for that and drops
  them). The rating uses players who appeared (`active := appeared`), not the NFL's expected-and-
  active rule; keep that if the two modules are ever merged. The factor is `dvpf_<stat>` on the
  projection rows and `why.dvp[<stat>]` in the board JSON; the matchup tooltip row that shows
  it lives in `site/template.html` (shared with the NFL page, which has no `why.dvp` yet, so the
  row only renders on the college board).
- **PFF data (college and NFL) lives in the private repo rhyman7/pff-data, never here.** This repo is public
  and its root is the live site; PFF's data is licensed to Ryan's PFF Pro account. Don't commit PFF
  rows (or a file that reproduces them row by row) anywhere in this repo, and don't put them in
  `docs/`, an issue, a PDF or an alert. Test scores, fitted weights and projections computed from
  them are fine. `tests/test_pff_private.py` fails if PFF files are committed. Moved out on Nov 4,
  2026; the files remain in this repo's history from #190 (left as is).
  - NFL tables (Nov 5, 2026): pff-data's `nfl/` (`pff_data.load(table, seasons, league="nfl")`).
    The man/zone tables are read only by `experiments/coverage_scheme_test.py`; the board does not
    use them (tested, not kept: PLAYERS.md "Man and zone coverage").
  - **Used on the NFL board** (Nov 5, 2026): `nflmodel/pff_routes.py` adds route share (decayed,
    last game, trend) and targets per route to the New projections (`players_ml.ROUTE_F`, through
    `players_ml.add_features`; PLAYERS.md "PFF route share in the New projections"). The **Update
    board** Action checks out only `nfl/**/*_receiving_summary.csv.gz` into `.pff-data/`; a new
    workflow that builds the NFL board must add the same step. pff-data pulls that table daily at
    10:35 UTC, before Update board. Without the data `players_ml.feats_for` drops the route
    columns and the board is what it was before; the build log says "PFF route inputs: on/off".
    The route columns never go into the board JSON or any file under `docs/` or `data/`.
  - **Used in the college margin model** (Nov 6, 2026): `cfbmodel/pff_returning.py` matches this
    season's CFBD roster to last season's PFF college players (offense and defense summaries) and
    adds `d_pffret`, transfer-aware returning production (CFB_FEATURES.md). Update college board
    checks out `ncaa/**/*_defense_summary.csv.gz` for it; without the data `d_pffret` is 0. Only
    the team-level difference reaches the board, never a PFF grade.
  - **Used in the NFL game model** (Nov 5, 2026): `nflmodel/pff_qb.py` gives each quarterback a
    prior from his PFF pass grades (`passing_summary`, 2014 on), which the rating engine's
    EPA-per-dropback rating shrinks toward in place of the flat `QB_REPLACEMENT` (engine argument
    `qb_prior_model`; FEATURES.md "Quarterback value from PFF grades"). Update board checks out
    `nfl/**/*_passing_summary.csv.gz` too, and pff-data's daily NFL pull includes that table.
    Without the data the prior is −0.13 for everyone, exactly the old model; the build log says
    "quarterback prior: PFF grades" or "flat -0.13". Only the derived quarterback values (EPA
    scale) reach the board, never a PFF grade.
  - **NFL passing-yards projection** (Nov 13, 2026, provisional; January recheck): `nflmodel/pff_passing.py`
    adds the quarterback's PFF passing grade and the opponent's PFF coverage grade (`team_overview`) to
    the player table (`players.build`), used in `players.CAL_EXTRA["pass_yds"]` (FEATURES.md "The college
    PFF pairs on the NFL projections"). Update board checks out `nfl/**/*_team_overview.csv.gz` too.
    team_overview is **not** on pff-data's daily NFL pull unless it is added there: if it falls more than a
    week behind, the opponent coverage column is 0 for every row (the calibration refits without it) and
    the build log says so; if the passing table is stale, both columns are 0. Build log: "PFF passing
    inputs: ...". Never written to docs/ or data/.
  - Reading it: `cfbmodel/pff_data.py` (`pff_data.load(table, seasons)`) looks in `$PFF_DATA_DIR`,
    then `.pff-data/` (git-ignored), then `../pff-data`. An Action that needs it checks it out with
    the **PFF_DATA_TOKEN** secret (a fine-grained token: read-only Contents on rhyman7/pff-data
    only): `actions/checkout@v5` with `repository: rhyman7/pff-data`, `token:
    ${{ secrets.PFF_DATA_TOKEN }}`, `path: .pff-data`. Never `git add` that folder; the Update
    Actions add only named paths, keep it that way.
  - **Used on the college board** (Nov 4, 2026): `cfbmodel/pff_roles.py` adds route share and snap
    share to the player table (`players.build`) for the receiving calibration (`players.CAL_EXTRA`).
    From Nov 13, 2026 `cfbmodel/pff_skill.py` adds three more (yards per route run, the quarterback's
    passing grade, the opponent's coverage grade) to the passing and receiving calibration
    (provisional; January recheck). The **Update college board** Action checks out only the tables
    it needs (`receiving_summary`, `offense_summary`, `defense_summary`, `passing_summary`,
    `team_overview`) into `.pff-data/`: a new PFF input needs its table added there; every college rebuild (kickoff,
    line alerts) runs through that Action, so a new workflow that builds the college board must
    add the same step. With no data, or PFF more than a week behind the board's latest games,
    every PFF column is 0 and the board is what it was before. No PFF column goes into the board
    JSON, the card or any file under `docs/` or `data/`.
  - The pull (`pff_pull.py`, the **PFF pull** Action, the `PFF_API_KEY` secret) and its rules are
    in pff-data's CONVENTIONS.md. No PFF key is stored in this repo.
  - `scripts/pff_identify.py` (stamps season and week on hand exports from the website by matching
    box scores) writes into the pff-data checkout (`hand/`), never here.
- **Official availability reports (college board): `docs/cfb/availability.json`**, written only
  by the **College availability** Action (`cfb-availability.yml`, `cfbmodel/availability_log.py
  --conferences`, standard library only, no pip step; every two hours Wednesday to Friday,
  hourly on Saturdays). The four power conferences publish NFL-style availability reports for
  conference games (SEC, ACC, Big 12, Big Ten: Wednesday to Friday evenings and a Game Day report
  90 minutes to two hours before kickoff), the MAC for every game and the American for conference
  games (a Game Day report three hours out); players not listed are available. All six come
  through one vendor app (app.hdintelligence.com, public JSON; the MAC's "Miami" is renamed to the
  board's "Miami (OH)"). The Mountain West publishes its own on themw.com and isn't read; the
  Pac-12, Sun Belt and CUSA publish none (Nov 2026). The file holds each team's newest report
  while its game is still ahead: `{source, fetched, today, teams: {school: {conf, abbr, report
  (Initial | Update | Game Day), published, posted, game_date, game_time, opponent, players:
  [{name, pos, number, status, season_out}]}}}`, statuses in the conferences' own words,
  "Exempt" rows (deep-roster players exempt from reporting) left out. **The board build uses it
  over Covers** (`cfbmodel/designations.load`, from Nov 9, 2026): for every team in it, the
  report's designations replace the team's Covers rows (a Covers row that says out for the season
  stays unless the report lists him; a live report listing nobody means everyone is available),
  mapped Out O, Doubtful D, Questionable Q, Probable P, Game Time Decision Q, Out (1st Half) P,
  with `src` the conference; the Teammates-out box names the source. **Questionable and Game Time
  Decision on an official report** (not Covers') are shaded by the chance to play
  (`designations.PLAY_CHANCE`: Questionable 0.66, Game Time Decision 0.64; Nov 9, 2026 test, kept,
  Project doc `claude/questionable-gtd-test-results-2026-11-09.md`): `players.shade_for_availability`
  keeps that share of his receptions, receiving yards, carries and rushing yards and gives the rest
  to undesignated teammates in proportion; passing and touchdowns stay as projected given he plays.
  His board entry carries `play_chance` and `proj_played` (full projection given he plays), a
  teammate's `avail_plus` (what he gained). **College props price from `proj_played`**
  (`cfbpropboard.with_target_share`; a prop is void if he sits). Refit the chances at the January
  recheck, not mid-season. The file is ignored once
  `fetched` is more than 5 days old. When the reports in it change (not just its stamps), the
  Action starts a quick **Update college board** run so Game Day reports reach the projections
  before kickoff (Ryan's call, Nov 9, 2026); it skips that on Saturdays 13:00-23:59 UTC, when the
  Market lines Action already rebuilds the college board hourly, and when an Update college board
  run is already queued or running. Don't edit it by hand. The CFB dashboard (rhyman7/cfb) reads
  the same vendor app into its own `data/availability.json` with the same rules.
- **College availability history: `data/cfb/availability/<season>/`**, written by the same
  Action: `conf_log.csv` and `conf_archive.json` (every version of every conference report,
  season to date), `covers_log.csv` (every change in the Covers.com designations the board used)
  and ESPN's college injury feed (`latest.json`, `log.csv`; empty for college on the Nov 3, 2026
  dry runs, kept in case it fills). History for the recheck of kickoff-time availability against
  the 2026 live prop log; `experiments/availability_grade.py` grades the archived designations
  against who played (PFF snaps where the private data is checked out, else box-score stats); the
  manual **Availability grade** Action (`availability-grade.yml`) runs it with PFF and prints
  aggregate tables only (no artifact, no commit). Its `--out` row file is never committed when PFF
  was used. Don't edit these files by hand. The box-score absence features stay in
  `cfbmodel/availability.py`; the two modules are unrelated.
- **CFBD player data: `data/cfb/cfbd/`** (`recruits_`, `portal_`, `returning_`, `roster_<season>.csv`),
  written by the manual **CFBD player data** Action (`cfbmodel/cfbd_pull.py`, `CFBD_API_KEY`).
  Inputs to experiments only (role prior, Nov 3, 2026: not kept); the board does not read them.
- **Hand overrides (NFL board): `data/overrides.csv`.** One row per player per game:
  `starter` sets that game's starting quarterback for the game model and makes him the
  quarterback who takes the team's passes in the player projections (whoever started the
  team's last game, and whatever the injury report says), `out` takes a player out of the
  player projections. **An `out` row for a core offensive lineman also reaches the game model**
  (Nov 11, 2026; `injuries.override_keys`, `load_injuries(overrides=)`): he counts as unavailable
  in `d_chg_ol` for that game, as a game-day inactive does in the seasons the model is fitted on.
  Until then the live board only knew Out / Doubtful and the reserve lists, so news after Friday's
  report (a Questionable lineman ruled out, the inactives 90 minutes before kickoff) never reached
  it. Offensive line only: the other position groups feed the shadow test, whose inputs are fixed.
  The fit is unchanged (played games are never touched), so this is a data input, not a model
  change. Each build writes `data/cache/ol_core_<season>.csv`: the five linemen the model counts
  for each team's next game (`name` as the snap counts spell it, `player_id`, `share`, `out`,
  `reason`). A lineman row must use that file's `name`, since the match is by name.
  **Offensive-line names are matched loosely across the source files** (Nov 11, 2026, Ryan's call
  after a flat pre-set test; `injuries.alias_map`, FEATURES.md "Offensive-line input: linemen the
  name match loses"): a lineman the snap counts, roster and report spell differently (a dropped
  "Jr.", "DJ" for "Delmar") used to count as missing every week. Provisional like any applied
  change: re-run `experiments/ol_name_match_test.py` at the January recheck. The other position
  groups still use the exact match (`ALIAS_GROUPS`); widening it changes the player projections'
  and the shadow test's inputs and needs its own test with a props check.
  **Late news rows are entered by four Claude scheduled tasks** ("NFL late availability": Saturday
  6:40 pm, Sunday 9:40 and 11:40 am, Sunday 3:05 and 7:05 pm, Monday and Thursday 7:05 pm ET), which
  read the official inactives, team announcements and reports by Adam Schefter, Ian Rapoport or Tom
  Pelissero, add `out` / `starter` rows marked "(auto, news)" through a pull request they merge,
  and start Update board. Their wording is `scripts/tasks/nfl_late_availability.md`; they touch
  nothing but this file. Kickoffs those times miss get a one-time run of the same wording about 75
  minutes before kickoff (Nov 12, 2026: ten for the rest of the 2026 regular season: Munich,
  Thanksgiving eve and day, the Friday after, Saturday Dec 19, Christmas; the list is in the Project
  doc `claude/late-availability-task-prompt-2026-11-11.md`). Week 18 and the playoffs need theirs
  once the league sets the times. Use `out`
  only for confirmed news the nflverse schedule or the
  official injury report hasn't picked up yet, and put the source and date in `source`. Rows
  only touch games not yet played, so they lapse by themselves; delete stale rows when you
  see them. Hand edits go through a PR like any other change; after merging, run **Update
  board** so the board picks them up. One Action writes this file: the **QB starter check**
  (`qb-check.yml`, `nflmodel/qbcheck.py`; Mon-Sat, once a day from noon ET, since GitHub often
  starts scheduled runs hours late) commits nfelo's pick as a `starter` row marked
  "(auto, qbcheck)" where nfelo and the board disagree, then starts Update board. It never
  replaces a hand-entered starter row and never starts a quarterback with an `out` row for
  that game, so when nfelo's list is behind the news, a hand `starter` row (or `out` row) wins.
  It commits `data/qb_check/` and `data/overrides.csv` straight to `main`.
- **Covers.com designations (college board): `docs/cfb/designations.json`.** The source for
  teams without a live official report (above). The daily college
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
- **NFL Questionable players: `proj` is what he does if he plays**, and everything that prices
  (props, parlays, best bets, alternates, outlook) reads `proj`. Since Nov 9, 2026 the board also
  gives each RB/WR/TE on this week's report as Questionable a display-only `proj_ev` and
  `play_chance` (`nflmodel/qshade.py`, constants in `data/questionable_shade.json`, written only
  by `experiments/nfl_questionable_test.py --shade --write-constants` at the January recheck):
  receptions, receiving yards, carries and rushing yards x (1 − λ(1 − chance)). The page swaps it
  in for display (`adoptNew` in site/template.html keeps `proj_played`); teammates don't change.
  Never write the shaded numbers into `proj`, and key the shade to this week's report
  (`report_lvl`), not the roster listing's game status, which carries last week's until the new
  report. The board's Questionable / Probable superscript is kept only when this week's report
  says so (`players.this_week`, Nov 9, 2026); Out and Doubtful stay as listed, since the
  projections treat those players as out until the new report.
  PLAYERS.md "Questionable players: a shade on top of the clean multiplier".
- **Questionable starting quarterback flag (NFL, Nov 9, 2026; `qshade.qb_flags`).** When the game
  model's quarterback (`away_qb` / `home_qb`) is on this week's report as Questionable after a
  limited or no Friday practice, the game gets `qb_flag`: a "QB questionable" tag on its tile and
  a warning in the game page's Quarterbacks section. It changes no number and is not a bet rule;
  the daily QB check switching the starter clears it. Rates (2016-2026, the starter from the
  team's last game listed Questionable): started 91% after a full practice (no flag), 54% limited,
  43% none.
  The Questionable multiplier (`players_ml.questionable`) is fitted only on earlier Questionable
  players who got on the field: the pipeline passes `snaps=R["snaps"]`. Don't fit it, or anything
  else about who played, on roster status "ACT" alone: before 2019 the weekly rosters don't mark
  game-day inactives, so 19-24% of "active" skill-player rows then had no snap. Any NFL player test
  that asks about players who might sit must project without the inactive list (the usual backtest
  knows it from 2019; `experiments/nfl_questionable_test.py` shows how).
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
- **Stakes (both boards).** Quarter Kelly, capped at 3% a bet, 5% a game, all sides
  together (`config.MAX_GAME_STAKE`, `slate.cap_game`: every stake in the game is scaled
  down by one factor), and 10% a week, everything the sheet flags that week together
  (`config.MAX_WEEK_STAKE`, `sheet.cap_week` after the bucket stakes and before the ledger;
  `meta.week_cap` per week, `week_scale` per game; the sheet note says when it bit). With `config.EDGE_HAIRCUT` on, the stake is sized from the model
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
  (`sheet.bucket_stakes`: quarter Kelly on Pinnacle's chance for the side at the book's own
  line and price, from `ev_line`, plus the bucket's edge over a fair line, its win rate
  shrunk halfway to break-even less 0.5; the book is the one with the best `ev_line`, written
  as `bucket_book`, and the ledger records the bet there; once the line has moved toward an
  opener bet the rate is the bucket's `close_win_pct`).
- **The bet sheet (both boards): `nflmodel/sheet.py`.** Every bet on a game entry carries
  `tier`, `gap`, `opener_gap`, `opener_src`, `move` and `why` (the docstring has the rules).
  The tiers are what the page's Bet sheet groups by: `price` (a retail price beats
  Pinnacle's no-vig number by `PRICE_MIN`, 1%, with no model), `opener` (the model 2+ points
  off the opener, 3+ in college and on totals), `wind` (NFL only: an under in 10-19 mph), and
  `model` (anything else +EV on a spread or total). **Moneylines are never a bet outside the
  price tier**: on the graded bets since the holdout seasons they have lost at every claimed
  edge. **College spreads are never the opener tier** (they lost against the opener on the
  holdout seasons below 5 points; college totals are). **The opener tier is judged at the
  current line** (Oct 31, 2026): a side stays in it while the line has moved no more than
  `sheet.MOVED` (half a point) toward it, or the gap at the current line still clears the
  threshold; once the market has taken it, it drops to the model tier with `missed: True`
  and the page says "Missed". A wind under drops the same way once the total has fallen
  `sheet.WIND_MOVED` (1.5) since the open, and wind unders say "Bet now". The opener is the
  data's opener first (nfelo, cfbfastR: the number the bucket was backtested on), else
  Pinnacle's first stored quote when the market log caught the open (its first snapshot no
  later than the day the board first saw the game), else the board's first line. Each game also carries `kickoff` (UTC), `sharp_path`
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
- **College game-model inputs (`cfbmodel/extra.py`, from Nov 4, 2026).** `data/cfb/cache/`
  holds, per season, `play_extra_<s>.csv` and `drive_raw_<s>.csv` (written by `data.load_season`
  from the same play-by-play download as `team_games`) and `def_missing_<s>.csv` (from the
  cfbfastR box scores); the Update college board Action refreshes the current season's with the
  rest of the cache. `data/cfb/coach_changes.csv` is **hand-kept**: every FBS head-coaching
  change with its date (team as the games file names it, `COACH_ALIAS` in `extra.py` for the
  few that differ). When a head coach is fired, resigns or an interim is named during the
  season, add the row in a PR the same week; the mid-season-change feature reads it, and no
  scheduled task writes it. Offseason hires can wait for the season's first rows.
- **The ledger (both boards): `data/ledger.csv`, `data/ledger_results.csv`, and the same
  under `data/cfb/`** (`nflmodel/ledger.py`). Every bet the sheet flags (a stake on the shown
  basis, or a price, opener or wind tier) is written down the first time it is flagged, at
  the retail book, line and net price it would be taken at, and never changed; results are
  appended once the game is played (W/L/P, units at the price taken, and closing line value
  against Pinnacle's last quote before kickoff through the key-number curve). Derivative rows
  (market `Team total`, `1H Spread`, `1H Total`, `1H Moneyline`) are graded on the team's points
  or the half-time score and carry no closing line value yet. Append-only:
  the Update Actions write them, never edit or delete a row by hand, and a local build run
  must not commit its rows (they would be sandbox prices). From Oct 31, 2026,
  `ledger_status.csv` beside each ledger (append-only, written by `ledger.track` inside
  `record`) logs each time a row leaves the sheet or comes back before kickoff, so the record
  reads both ways: `kept` (still on the sheet at kickoff) and `withdrawn`; each summary row
  has `kept`. The record is also split by where the bet would be placed (`by_books`, `ledger.by_books`, Nov 11, 2026): `mine` is DraftKings, FanDuel and, since Nov 26, 2026, Kalshi, where Ryan bets (applied to every row, so the Kalshi rows flagged earlier moved from `other`; `ledger.MY_BOOKS`; `ledger.SPORTSBOOKS` is the two sportsbooks, for teasers and props, the same pair as `kickoff.MY_BOOKS`), `other` is BetMGM, Kalshi and consensus rows; both records' tables on the Model record page show the two rows, and the Best Bets record carries the same split. **College closing line value** (Nov 11, 2026): `cfbmodel/pipeline.close_by_id` keys the closing quotes by the whole-number ESPN id before grading; until then they were keyed "401856733.0" and no college row had a `clv`. One sanctioned correction: `scripts/backfill_cfb_clv.py --write` filled `close_line`, `close_prob` and `clv` (those three columns only) on the 151 college ledger results and 86 Best Bets results graded before the fix; `tests/test_cfb.py` fails if most graded college rows lack one. `data.json` carries the summary as
  `ledger` (records overall, kept, withdrawn, by tier, by market and by week, plus every row) and the Model
  record page leads with it as **Live record**; `tests/test_sheet.py` checks the files.
- **The Best Bets record (both boards): `data/bestbets.csv`, `data/bestbets_results.csv`, and
  the same under `data/cfb/`** (`nflmodel/bestbets.py`, Oct 31, 2026). Only the bets on the
  Best bets card (`bestbets.on_card`, the page's `ON_CARD`: every tiered bet but the NFL
  model tier), as the card stood at each game's kickoff: each build appends an entry when a
  card bet is new, changes book, line, price or stake, or leaves the card (`on_card` 0), for
  games not yet kicked off; the record reads the last entry before kickoff. Graded by
  `ledger.grade_frame` like the ledger. Append-only, written by the Update Actions only.
  `data.json` carries it as `best_record` (all, by week, by tier, by market, rows, `pending`);
  the Model record page shows it as **Best Bets record** under the Live record, and the Best
  bets card's header carries the record so far.
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

## Scheduled tasks' instructions (`scripts/tasks/`)

- **A Claude scheduled task's wording lives in this repo, not in the task.** The task's own prompt
  is a few lines: get the repo, read the file, follow it. So changing what a task does is an
  ordinary pull request on the file, reviewed like code, and nobody retypes a long prompt.
  - `scripts/tasks/daily_check.md`: **Value Boards daily check** (moved here Nov 25, 2026).
  - `scripts/tasks/nfl_late_availability.md`: the **NFL late availability** tasks.
  - `scripts/tasks/weekly_review.md`: the two **Weekly Review** tasks.
  - `scripts/tasks/weekly_digest.md`: the **Weekly digest** task (Tuesday 2:40 pm ET, Nov 25, 2026):
    one message a week on what was recommended, how it did, what changed in the model and what
    needs a decision. It reads this repo and Ryan's private My Bets page, saves the digest to
    that page and writes nothing to any repo. His own bets never go in a repo or on the site.
- **Web addresses are the one exception.** In scheduled runs WebFetch only opens an address written
  out in the task's prompt or returned by a search; an address read from a file is not trusted. The
  Value Boards daily check fetches Covers team pages, so its prompt ends with the full Covers list
  (the same 138 addresses as the end of rhyman7/cfb's WEEKLY_PROMPT.md, which is the copy to edit).
  If a Covers address changes, change it there and in the prompts of the Value Boards daily check
  and the CFB dashboard daily update.
- A task's file never lets it change the model or its betting logic (the keep rule, below).

## Model tests: the keep rule

- **Rule from Nov 11, 2026 (Ryan; replaces the Nov 4, 2026 "any improvement, however small" rule,
  which itself replaced "holdout better by 2+ standard errors").** A model change is kept only if:
  1. the rule and its measures are written into the test script (and a Project plan doc) before the run;
  2. it is better on **every** accuracy measure the test reports, on both the tuning seasons and the
     holdout, and worse on none: projection RMSE (or the game model's average miss), and Brier at the
     real DraftKings/FanDuel lines where lines exist, props included;
  3. the holdout improvement is at least **1 standard error** (paired by game or by player-game).
- **Anything kept is provisional.** It is rechecked on the next full season it has not seen and
  removed if it is worse there. Why: the 2019-on holdout has been used by well over a hundred tests,
  so a fresh season is the only real out-of-sample check left, and gains smaller than their standard
  error are as likely noise as signal (the Nov 4-6, 2026 college additions were 0.011 to 0.016 points
  better on the tested seasons and 0.015 worse on 2026). The January recheck judges everything kept
  since Nov 4, 2026 on the 2026 season alone.
- A mixed result (better on one measure, worse on another), or better everywhere but under 1 standard
  error on the holdout, is not kept. It goes to Ryan as a trade-off for him to decide.
- Every model test includes a props check where relevant.
- **Player-projection tests are scored at bet time too (Nov 11, 2026).** Any change whose input is
  who is expected to play (shares, team sums, usage handed to teammates, availability) is scored on
  bet-time projections (`players.build(pregame=True)`, the pricing-snapshot lines) as well as on the
  game-day table, and must pass on both. The game-day table knows the inactives (from 2019); the
  board, mid-week, does not and counts about one more receiver per team. Scaling receivers to fit
  the quarterback was better on every measure on played games and worse on every measure at bet
  time (PLAYERS.md, `experiments/team_totals_test.py --bettime`).
- Still run the 2017-2025 walk-forward (or the NFL equivalent) before anything is kept, report results
  plainly including when nothing is kept, and still give Ryan a short plan and wait for his go-ahead
  before a kept change goes on the live boards.
- Changes kept under the earlier rules stay until the January recheck; they are not re-judged now.

## What's live (`docs/whats-live/`, Nov 25, 2026)

- **One list of every provisional change:** `data/whats_live.json`. Each row has what the change
  does, where its setting lives (a constant in a `.py` file or a key in a data file), how to turn it
  off, the stretch it is judged on and the rule that decides it. `python scripts/whats_live.py`
  reads each setting's current value from the code and writes `docs/whats-live/index.html`
  (`/fb-value-board/docs/whats-live/`, not linked from the boards, `noindex`). The page is
  generated: never edit it by hand.
- **Applying a provisional change means adding its row in the same pull request**, with the rule
  as written before the test, then running the script and committing the page. When a recheck
  decides a change, remove its row (or move a setting to "waiting") in the pull request that acts
  on the decision. Changing a listed setting's value needs the page regenerated too.
- **`python scripts/whats_live.py --check`** fails if a listed setting is no longer in the code, if
  a file under `nflmodel/` or `cfbmodel/` says "provisional" and no row names it, or if the page is
  stale. The Smoke test Action runs it on every pull request. It is not a pytest test on purpose:
  the board builds run pytest, and a stale list must never stop a board from building.
- The list holds changes that are **live**. Tests not kept, leads and forward-only records stay in
  the January recheck documents. The changes kept Nov 4 to 10, 2026 under the earlier rule are one
  group row, not itemised.

## One site: the shared bar and the dark look (Nov 26, 2026)

- The Weekly Gridiron Board is one site with one bar and one look (plan: Project doc
  `claude/plan-one-site-redesign-2026-11-25.md`). Pages load the bar as
  `<script src="/site-bar.js" data-shell="2"></script>`; its sections and pages are `SECTIONS` in
  rhyman7.github.io's `site-bar.js`. A page does not carry its own links to other sections.
- **Dark, with the dashboards' graphite and Archivo:** page `#131416`, panels `#1b1d21`, raised
  `#24272c`, rules `#353a42`, text `#f2f1ee`, secondary text `#9da2ab`, good `#4ade80`, bad
  `#f87171`, caution `#fbbf24`. Headings are Archivo, condensed (`font-stretch`), not Big Shoulders
  Display. Moved so far: Recommended, What's live (through `scripts/whats_live.py`), Parlays
  (`site/parlays.html` and its copy in `docs/`), the Weekly Review (already dark) and the manual
  (dark on screen since Nov 26; its print rules and PDF stay black on white. Its screenshots were
  re-shot dark, with the shared bar, for the sixth edition the same day).
  The two value boards moved the same day (below).
- **The value boards (Nov 26, 2026).** `site/template.html` has one colour set, the graphite above,
  in `:root`; it no longer follows the device's light or dark setting. The field keeps its turf and
  chalk. `--display` is Archivo, written `extra-condensed` inside each `font:` shorthand (the
  shorthand takes the keyword, not a percentage). A general `@media print` block right after `:root`
  puts the light colours back, so every print view and the best-bets and leans PDFs stay black on
  white.
- **The boards' own bar is the fallback only.** With the shared bar on (`html.wgb-shell-on`) `.bar`
  is hidden and the shared bar's second row lists the sections; the board's bar shows only when
  `/site-bar.js` does not load (a page opened from disk). A new board section is added to both the
  template and `SECTIONS` in rhyman7.github.io's `site-bar.js`.
- **"This Week's Matchups" is off the menu but its addresses still work** (`#matchups`,
  `#matchups/<id>`, `#matchups/<season>_w<week>...`): the dashboards' game packets print from them
  and Past weeks links each game and week to them. Do not remove the view without moving both.
- **Print views and documents stay black on white:** the Recommended weekly report
  (`body.report`), anything under `@media print`, and the PDFs. A page that turns dark resets its
  colour tokens to light inside those.
- Parlays' own green header with links to the boards is gone: the shared bar does that job.

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
  Manual `#manual` | This Week's Matchups `#matchups` | Games `#games` | Players `#players` |
  Props `#props` | Past weeks `#history` | Team ratings `#ratings` | Model record `#model`.
  The site bar above the nav links to the other board (the nav's own link came out Nov 20, 2026).
- **The manual (both boards): `#manual` and `docs/manual/`** (Oct 31, 2026). The Value Boards
  manual lives once, in `docs/manual/` (`index.html`, its screenshots in `img/`, and
  `Value_Boards_Manual.pdf`), hand-made, not generated by a build: the Update Actions never
  touch it. Each board's Manual view loads it in a frame sized to its content on first open
  (`openManual()`; the college page points one folder up, `../manual/`), with buttons to
  download the PDF and open it on its own page. A new edition replaces those files through a PR. To make one (fifth edition, Nov 11, 2026): serve the repo root, run `python scripts/manual_shots.py` (it re-shoots every screenshot in `docs/manual/img/` from the built pages; one entry per image in `SHOTS`), rewrite the captions, the two "How I read" boxes and the dated sentences in `index.html` from what the new screenshots show, then render the PDF from the page with Chromium (`page.pdf(prefer_css_page_size=True, print_background=True)` after setting every image to load eagerly). Say on the cover which day the screenshots are from. **Sixth edition (Nov 26, 2026):** adds chapters 20 to 22 (the Recommended list, the Weekly Review, and the site, the menu and the phone app), corrects the Kalshi and college Props passages, and re-shoots all 28 screenshots from the dark boards with the shared bar in them (serve the hub's `site-bar.js` at the root and this repo under `/fb-value-board/`, then `--base http://localhost:<port>/fb-value-board`; the pictures were taken from pages built locally from `site/template.html` and the day's `data.json`, with Archivo installed so the font matches the live site). Captions, the worked examples and every dated figure were rewritten from that day's pages. Backtest figures in the running text that no picture shows were spot-checked, not all re-derived.
- **Best bets (both boards): `#bestbets` and `best-bets.pdf`** (Oct 30, 2026). The week's card
  on one document, built from the whole board by `renderBestBets()` in `site/template.html`:
  the Bet sheet's bets in tier order with book, price, EV, stake, why and when; the 6-point
  teaser legs that clear −120 and the best pairs; the week's prop flags (NFL; book disagreements when none
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
- **Model leans (both boards): `#leans` and `model-leans.pdf`** (Oct 31, 2026). The model's
  opinions that are not bets, on their own document so the card holds only bets
  (`renderLeans()`): on the NFL board the model-tier sides (no stake while the haircut is 0;
  they are off the Bet sheet and the Best bets card, `ON_CARD` in the template), and on both
  boards every game's market, model and adjusted numbers with the board's lean. The Bet
  sheet's head has **Model leans, not bets (PDF)** and **Print the leans**; the landing page
  links both PDFs; `scripts/build_pdf.py` renders it beside `best-bets.pdf`. Printing uses
  the same `#bb-page` / `bb-print` rules. Not a nav section.
- **Record-adjusted EV (both boards, Nov 11, 2026; Ryan approved the mockup).** Every bet carries `ev_rec` and every book quote `ev_rec`: the EV at the chance the stake is sized from (`bets.stake_prob`, the model shrunk toward the anchor's no-vig number by `meta.haircut_factor`). Wherever the page shows an EV that comes from the model (`evPair` / `evRecOf` in `site/template.html`: model-tier rows on the Bet sheet and the Best bets card, Model leans, a game's price table, its headline sentence, the Games lede, the chart hover) the record-adjusted figure leads and the model's own claim sits under it as "model +x%". Opener and wind rows show both in small type, because their Record column is what leads. Price-beats-Pinnacle figures need no model and are unchanged, as are stakes, sorting, tiers, the ledger and `ev` itself. With the haircut off, on the raw-model basis, or on data without the field, the page shows the model's EV as before. Keep the wording honest: the note says the share is "not yet distinguishable from zero" while the factor is under two standard errors.
- **Stale prices.** A price-tier row is greyed, and its verdict reads "Check the price",
  when the board's book prices (`meta.inputs.market`) are more than 3 hours old (`STALE_H`).
- **Same page pieces on both boards** (Oct 31, 2026): the Model record's "By part of the season" table (`backtest.by_week`, holdout seasons; NFL weeks 1-4, 5-9, 10-14, 15-18 and playoffs, measured on full-backtest runs) and the player card's "From chain to projection" box (`why.chain`, `why.recent`, and on the NFL `why.ml` for the stats the boosted step replaced) are in `site/template.html` for both boards, no longer college-only substitutions.
- **The Games view's order** (Oct 31, 2026): filters, **Bet sheet**, **Line movement**
  (`sheet.line_moves`: the opener-gap sides with opener, now, the share of the gap closed,
  status and the side's 6-point teaser leg; sides with no backtested edge folded under a
  details line), the boost calculator,
  **Teasers and alternate lines** (legs from `teasers`, the two-leg price entered by the
  reader; the per-game ladders are on each game's page under "Alternate lines and teasers"),
  then Model against the market, the game cards and Every price. The Model record view leads
  with **Live record** (the ledger), then the backtest with the holdout columns first. The
  Props view hides lines on players the board has ruled out and passing lines with no
  projection, and says how many. Every Props table column sorts (Nov 7, 2026), on what it shows: Game by
  kickoff, Stat in the dropdown's order, Range by its low end, DK/FD by the over price, Close by
  the move; a blank always sorts last.
- **Player outlook (Props section): `docs/props/outlook.json`** (`nflmodel/propoutlook.py`),
  written by **Update board** (from the data.json it just wrote) and by every props run, never
  by hand: each board player's median, 80% range and chance of reaching at least every number.
  The page's Range column and the outlook panel at the top of an open row read it; the panel
  says these are the model's chances, not bets, and gives the fair price. Keep that wording if
  the panel changes: the books have been the better forecast on props (PROPS.md).
- **Every-game floors (both boards): `players[].floor`** (`nflmodel/floors.py`, Nov 5, 2026),
  written by **Update board** and **Update college board** after the player projections, never by
  hand, for players already on the board only (no player is added for it). The most yards he
  reached in every game he played this season: current team only, a QB's starts only, early exits
  left out (NFL: snap share under half his median; college: touches under 40% of his median).
  Stats by position: QB passing and rushing; RB rushing, receiving and rush + receiving; WR and TE
  receiving; 3+ games. Flags (`floors.SHOW_FLAGS`, set by `experiments/floor_tag_test.py`): ⚠ tough
  defense, bigger underdog, new QB, wind; ▲ teammate out; blowout risk is off. Shown under the
  projections on Players, the Matchups team cards (this week only), the player card's "Every-game
  floor" box and the Props table's Floor column and ladder panel. **Display only**: it never feeds
  the model, the bets, the Bet sheet or the Parlays page, and the Props section never marks rungs
  under it (graded on 2025 rungs they lost about 9% per bet, PROPS.md "Every-game floors"). Keep the
  "information, not a bet" wording (`FL_NOTE` in the template) if the display changes. A floors
  failure never stops a board build.
- **The dashboards' Highest Floors pages** (Nov 7, 2026): rhyman7/nfl and rhyman7/cfb's Matchup Report
  (`scripts/build_report.py`) reads the floors from `docs/data.json` / `docs/cfb/data.json`: `players[]`
  `name`, `pos`, `team`, `game`, `status`, `proj.pass_att` and `floor.{pass,rush,rec,rr}` (`floor`, `g`,
  `avg`), plus the teams' `out` lists. Keep those fields or update the report in both dashboards.
- **Props `#props` is on both boards** (NFL from Oct 2026; college from Nov 7, 2026, Ryan's call),
  between Players and Past weeks. One section in `site/template.html`; `site/make_cfb_template.py`
  swaps in the college wording (not market-tested banner, the college pull times, the over-only
  rung note, "Off" for flagged bets). Only the Best bets card's props summary is still marked
  `/*NFL-ONLY*/` and cut from the college page (the script checks the number of blocks: update
  its count when adding one). Each page reads `props/` next to it, one file per week plus
  `index.json` and `outlook.json`:
  - NFL: `docs/props/`, written by `nflmodel/propboard.py` on every props run from the
    append-only prop log.
  - College: `docs/cfb/props/`, written by `nflmodel/cfbpropboard.py` in **Update college board**
    (after the build; a failure never stops it). It spends no Odds API credits: it reads the
    files the **College props** Action pulls (`data/props/history_cfb_live/`). The first time a
    line appears it appends one row to `data/props/cfb_prop_log.csv` with the books' prices, the
    books' vig-free chance (the Parlays page's college legs start from it; on two-sided main lines
    the page then adds the provisional model nudge, below) and the model's chance then
    (college projections on `data/cfb/prop_params.json`, reception share standing in for target
    share, `cfbpropboard.with_target_share`). The row is never rewritten. Close comes from the
    game's file after kickoff; results from the archive's `actuals`. Not market-tested, so
    `flags_on` is false and nothing is flagged. Display only: it never feeds the model, the bets,
    the Bet sheet or the Parlays page.
  Never edit past weeks by hand; the files are rebuilt from the logs.
- **NFL Friday weight for current-line games** (Nov 19, 2026, provisional, Ryan's call on a trade-off;
  FEATURES.md "Bet-time information"; Project docs `claude/bettime-information-results-2026-11-19.md`,
  `claude/plan-nfl-friday-weight-2026-11-19.md`, `claude/nfl-friday-weight-test-results-2026-11-19.md`):
  `nflmodel/bettime.py`. With `config.FRIDAY_WEIGHT = True`, NFL games priced at the current line (not early
  lines, not the final week) use `w_margin` = the weight the board's model earns on a Friday reconstruction
  of the previous 8 seasons (last week's quarterback unless the Friday report or a returning starter says
  otherwise; `injuries._season(bettime=True)`, no game-day inactives). The model, its fit, early lines,
  totals and the college board are unchanged; the player projections are built before this step and keep
  the game-day weight, as with early lines. Measured on Update board runs with the backtest, and whenever
  `data.json` has none for the season; carried in `meta.blend.friday` otherwise. The reconstruction is
  cached per past season in `data/cache/bettime_qb_<season>.csv` and `injury_groups_bt_<season>.csv` (the
  first build of a new season downloads that season's injury reports and rosters once, about a minute).
  A failure keeps the game-day weight. Rechecked in January on the forecast log's logged inputs (Project doc
  `claude/january-recheck-additions.md`); `FRIDAY_WEIGHT = False` puts the board back as it was.
- **College market-residual margin** (Nov 19, 2026, provisional, Ryan's call on an 11-of-12 trade-off;
  CFB_FEATURES.md "Market-residual margin"; Project docs `claude/plan-fresh-eyes-structure-tests-2026-11-19.md`,
  `claude/fresh-eyes-structure-test-results-2026-11-19.md`): `cfbmodel/model.market_residual`. With
  `cfbmodel/config.STACK_MARGIN = True` the college adjusted spread (`proj_margin`) for games priced at the
  current line is the line plus a ridge fit of (result - line) on `MARGIN_FEATS` and the line, no intercept,
  each training season weighted 0.5 ** (age / `STACK_HALF_LIFE` = 4), penalty `STACK_ALPHA` = 500. The raw
  model (`model_margin`), its breakdown, totals and early lines (the opener weight, `pipeline.py`) are
  unchanged; `w_margin` is still measured and shown for early lines and as the fallback. The page wording
  follows `meta.blend.stacked` (`site/make_cfb_template.py`). Don't change the penalty or half-life without a
  new pre-set test. Rechecked in January on 2026 (Project doc `claude/january-recheck-additions.md`);
  `STACK_MARGIN = False` puts the board back as it was. NFL not changed (nothing passed there).
- **College QB rushing chance shift** (Nov 13, 2026, provisional; FEATURES.md "Correcting the college QB
  rushing chance", `experiments/cfb_qb_rush_calibration_test.py`): `data/cfb/prop_params.json` carries
  `"chance_shift": {"rush_yds|QB": 0.30}`, a logit shift on the chance of the over that
  `props.PropModel.probs` applies after the correction, for that stat and group only. Every college prop
  chance goes through `probs`, so the Props page, the Parlays legs and nudge, the outlook ranges and the
  floors read it alike. The NFL parameters (`data/prop_params.json`) have no shift. A refit of the college
  file (`experiments/cfb_propdist_test.py --save`) carries it over; don't change the value without a new
  pre-set test. Rechecked in January on the full 2026 season. The Props page's logged rows keep the
  chance they were first priced at; lines first seen after the change carry the shift.
- **Role change in the prop spread** (Nov 17, 2026, provisional; FEATURES.md "Prop spread that widens
  after a role change", `experiments/role_change_spread_test.py`, `nflmodel/rolechange.py`): each player's
  role change u (his share in his last appearance against his decayed share; carries, and targets in the
  NFL / catches in college) widens or narrows the distribution through one term, `CountFit.d` (counts) and
  `YardsFit.e` (yards). It is on only where it passed: **NFL WR/TE/RB receptions and receiving yards**
  (`data/prop_params.json`) and **college RB carries and rushing yards** (`data/cfb/prop_params.json`),
  listed in `rolechange.APPLIED`; every other fit has the term at 0 and prices as before. The boards write
  u to `why.role_chg_car` / `why.role_chg_rec` (`players.board(..., table=ptable)`), `propmarket.board_players`
  reads it and `props.role` hands it to the fits, so the Props pages, Parlays legs, outlook and floors all
  see it. A refit keeps it: `experiments/props_calibration.py --production` and `--role-change`, and
  `experiments/cfb_propdist_test.py --save` and `--role-change`, refit those fits with the term (each
  started from its fit without it). Don't switch it on for another stat without a new pre-set test. The
  pieces not applied (college receiving, NFL rushing) and the applied ones are rechecked in January on
  2026 alone (Project doc `claude/january-recheck-additions.md`). `PropModel.save` keeps top-level keys it
  does not own (e.g. `shade`) since Nov 17, 2026; before that a refit through `save` dropped them.
- **WOPR standing in WR/TE receiving yards** (Nov 18, 2026, provisional, Ryan's call on a 14-of-15
  trade-off; FEATURES.md "NFL: receivers whose WOPR is high for their projection",
  `experiments/wopr_standing_test.py`, `nflmodel/woprstanding.py`): after the boosted and Questionable
  steps and before defense-vs-position, a wide receiver's or tight end's receiving-yards projection
  (10+ yards) is multiplied by `1 + b x z`, z being how far his WOPR sits from the usual for his
  projection. The line, its spread and b are constants in `data/wopr_standing.json`, written only by
  `experiments/wopr_standing_test.py --write-constants` (fitted on game-day projections before 2026).
  Don't change the constants, the 10-yard cut-off or the clip without a new pre-set test; refit only
  at the January recheck. Receptions and running backs are untouched. `woprstanding.APPLIED = False`
  (or removing the file) puts the board back as it was.
- **College quarterbacks on big underdogs** (Nov 19, 2026, provisional, Ryan's call on a 7-of-8 trade-off;
  CFB_PLAYERS.md "What earns its place", `experiments/chatgpt_player_changes_test.py`): `cfbmodel/players.underdog_qb`
  takes `yards_per_point` x (projected losing margin beyond `margin`) off a quarterback's passing yards, from
  `from_season` on. The three constants live in `data/cfb/qb_underdog.json` (2.871, 7, 2026: the 2017-2025 fit
  the test scored 2026 with). It runs last in `players.walk_forward`, so no fit and no earlier season sees it;
  `pass_yds_pre_dog` and `dog_adj` keep the projection before it and the yards taken off. Don't change the
  constants or extend it to other stats or to favourites without a new pre-set test; refit only at the January
  recheck. Removing the file puts the board back as it was.
- **College prop legs on the Parlays page** (`parlays.cfb_book_legs`): priced off DraftKings and
  FanDuel. From Nov 13, 2026 (Ryan's call, provisional) two-sided main lines in passing yards,
  passing TDs, rushing yards and receiving yards carry the model nudge: the books' no-vig chance
  moved `CFB_W_MAIN` (0.2) toward the model, at most `CFB_PULL_MAX` (3 points), legs more than
  `CONFLICT` apart left at the books'. That is exactly the version tested
  (`experiments/cfb_props_nudge_test.py`, FEATURES.md "The model nudge on college main lines": 19 of
  20 measures); don't change the weight or cap without a new pre-set test. Value is scored on the
  books' number (`wv`), never the nudged one. Ladders and anytime TD are not nudged. It is rechecked
  in the January 2027 pass on the full 2026 season's live lines (Project doc
  `claude/january-recheck-additions.md`) and removed if it is worse than the books there.
- **One set of player projections** (Ryan's call, Oct 31, 2026). The New projections
  (`nflmodel/players_ml.py`) replace the current ones wherever `players_ml.ADOPT` covers the stat
  and position (RB carries and rushing yards; WR and RB receptions and receiving yards); every
  other stat keeps the current projection. `players.board` writes them into `proj`; there is no
  `proj_new` and no "New" column on Players, Past weeks, Matchups or Props. Older files that still
  carry both (archived weeks' `proj_new`, props weeks with `pno`/`pnu`/`mun`) are shown through the
  page's normalisers (`adoptNew`, `adoptNewProp`), which put the New values in place of the old ones;
  the logs themselves are never rewritten. Don't bring back a side-by-side column: test a new
  projection in `experiments/` and replace, as here, when it wins.
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

## Typical (Nov 18, 2026)

- **Typical** is the model's own 50/50 yardage for a player this week: the line at which the
  prop distribution gives an even chance of over and under at his projection
  (`nflmodel/typical.py`; both pipelines attach `p["typical"]` right after the floors). It is
  labelled "Typical" everywhere (Ryan's choice) and shown above the floor: the Players tab, the
  player's detail card and each game's matchup tables on both boards. The dashboards' feed carries
  it as `t`, and the matchup pages and report PDFs in rhyman7/nfl and rhyman7/cfb print it beside
  the board's projection. **Display only**: nothing reads it back.
- The Weekly Review's Players page scores it (about half of players should reach it).
- **Glossary hints (Nov 25, 2026).** In the Players note, the words "typical" and "floor" are
  `<span data-glossary="…">`, so the site's glossary hints (rhyman7.github.io `glossary.js`) underline
  them and show the Start here definition. The player card's "Typical" and "Every-game floor"
  headings are marked by the same script. The per-row "typical 233" / "floor 96" keep their own
  hover text and are not marked, so the table isn't underlined on every row.

## Weekly Review (`review/`, Nov 17, 2026)

Every finished game looked back on: what went right, what went wrong, what was chance, and what
can be improved. Live at https://rhyman7.github.io/fb-value-board/review/. **Display only**: it
reads the archive and play-by-play and never feeds the model, the bets or the boards.

- **Page:** `review/index.html`, hand-made (not generated by a build): landing page styled like
  this repo's landing page and the Weekly Gridiron Board (CFB card left, NFL right, the same
  logos from `../img/`), then per league the dashboards' top bar and week strip, the week's
  game list, one page per game and a Season patterns page. Routes are plain hashes:
  `#nfl`, `#nfl-10`, `#nfl-10-ne-det` (away-home, lower case, URI-encoded), `#nfl-patterns`.
  Team blocks use `meta.colors` from the board's data.json, lettered in the secondary color,
  never black (the dashboards' `tcLetter`).
- **Facts:** `python -m nflmodel.review <nfl|cfb> <season> <weeks...> [--pbp FILE]` writes
  `review/data/<league>/<season>_w<week>.json` from the archived week (`docs/archive`,
  `docs/cfb/archive`; graded games only) and the season's play-by-play (nflverse; cfbfastR,
  matched by the ESPN `game_id`), and rewrites `review/data/<league>/index.json`. A lean is the
  side of the market line the model's own number (`model`, not the blended `proj`) sat on.
- **Context** (`nflmodel/reviewctx.py`, Nov 18, 2026, Ryan's ask: think outside the box): each
  game also gets `flow` (quarters, drive outcomes, when the total went over, garbage-time points,
  the score that settled the cover), `context` (previous game and its cover / model miss, record
  against the spread, rest, bye, division or conference game, rematch, weather, slot, referee,
  coaches), `short` (players who came up short, with usage, share, snaps, injury report, script,
  defenders, previous game, season average, former team), `beat` (one line: Ryan cares less about
  these), `dnp`, `personnel` (defensive playmakers, players used far more than usual, regulars who
  didn't play) and `former_team`. NFL sources: nflverse schedule, snap counts, injuries, weekly
  stats and rosters; college: `data/cfb/cache` (games, player_games, player_info). Facts files are
  written compact (no indentation) to keep them small.
- **Players page** (`nflmodel/reviewplayers.py`, Nov 18, 2026, Ryan's ask): `#nfl-players`,
  `#nfl-players-10`, linked right of Season patterns. `review/data/<league>/<season>_w<week>.players.json`
  (written by the same `nflmodel.review` run) has one row per player and stat with the frozen
  projection, the actual and a status. Ryan's minimums (Nov 18, 2026, `reviewplayers.on_page`): a WR projected 20 yards or
  fewer, a TE 15 or fewer, a back's rushing 15 or fewer and a back's receiving under 10 are left
  off the page and not counted; a quarterback's projection under 10 yards is listed but not
  scored. The scorecard counts the rest for players who played their usual game: met = actual
  at or above the projection. Each row also carries **Typical** (`md`, `reviewplayers.typical`):
  the model's own 50/50 yardage for the player from the prop distribution (`props.PropModel`,
  today's fitted widths) at the frozen projection; each scorecard tile shows the share who
  reached it, where about 50% means the spread is right. The Met / Short tag stays on the
  projection. **Left early is never
  counted against the model** and has its own labelled scorecard: a quarterback who threw under
  60% of his team's passes; NFL, a regular (25%+ of snaps) under half his usual snap share (the
  floors' rule); college, touches under 40% of his usual with none after halftime (an estimate,
  labelled). NFL players with no offensive snaps are "Didn't play"; on the field with no stats
  is a real zero. College players with no box-score line are "No stats recorded" (no public
  snap counts, so unknown) and not counted in the main scorecard.
  **Shut-outs counted (college, Nov 18, 2026, Ryan's ask).** When the private PFF data is there
  (`PFF_DATA_DIR`, `.pff-data/` or `../pff-data`), `reviewplayers.pff_snaps` reads that week's
  offensive snap counts and each no-stat player is one of: **shut out** (played, at least half
  his usual snaps: a real zero), **limited** (under half the median of his earlier weeks, two
  weeks minimum: not counted, like left early), **sat** (no snaps), or **unknown** (a name two
  players share that week, or a team whose no-stat rows are 40%+ of its scored rows, which is a
  box-score gap; most of week 1). The page shows a second scorecard with the shut-outs added as
  zeros (`card.shutout`) and the four counts. **Only totals are written**: PFF's rows are
  licensed and this repo is public, so no player's row says which he was
  (`reviewplayers.PUBLISH_PLAYER = False`; changing it is Ryan's call). Without the PFF data the
  page is as before, with no second scorecard, so a review built without it loses that block.
- **Best Matchups page** (Nov 18, 2026, Ryan's ask): `#nfl-best`, `#nfl-best-10`, linked right of
  Players. The Players page filtered to the players on that week's Best Overall Matchups list in
  the dashboards' record (`data/best_overall.json` in rhyman7/nfl and rhyman7/cfb, read from
  raw.githubusercontent.com by `reviewctx.Sources`, or `--local best=FILE`): each team's pick as
  the list stood at its kickoff, on his list stat (QB passing; a back's rushing and receiving;
  WR/TE receiving). `reviewplayers.best_matchups` marks the rows (`bm`, his list number) and writes
  `best` (the picks and their scorecard, plus the record's own "beat his average" grade) into the
  week's `.players.json`; the index carries each week's `best` scorecard. Weeks without a record
  (before NFL week 7, college week 8) say so.
  **On the ladder** (Nov 18, 2026, Ryan's ask): did each pick clear the books' O/U, or one, two
  or three rungs below it? `reviewplayers.ladder` reads the record's `lad` (per book the main line
  and the next three alternate overs, at his kickoff) and `ly` (his yards on that stat; backs on
  the rushing ladder) into each pick's `lad`: the book shown (DraftKings where posted, else
  FanDuel), every line with over/under, and `deep` (first line cleared: 0 the O/U, 1-3 the rungs).
  `card.rungs` (`rung_card`): `ou` (every pick with a ladder against the O/U) and `n`/`hit` (picks
  with the full four lines: how many had cleared by the O/U, -1, -2, -3), at the book shown;
  left-early picks aren't counted. The page shows four tiles with the season so far, an "On the
  ladder" table (FanDuel's line under DraftKings' where different, and FanDuel's result when it
  differs) and two week-by-week columns. Ladders exist from NFL week 8 and college week 10 (when
  alternate lines were first logged); earlier weeks say so.
- **Chance ledger values** are fitted on 2023-25 by `experiments/review_chance_fit.py` into
  `data/review_chance.json` (points each item moved the margin past the closing line: fumble
  bounces, non-offensive TDs, missed field goals from 45 yards or closer). Refit with the script,
  don't hand-edit. A lean's verdict is "chance" when removing those items (or a starting QB
  leaving early) flips it.
- **Reads:** `review/data/<league>/<season>_w<week>.reads.json`. `python -m nflmodel.reviewdraft
  <league> <season> <weeks...>` writes a first draft from the numbers (`"by": "draft"`, shown as
  "First draft from the numbers"); the weekly review task rewrites every game in its own words
  and marks it `"by": "claude"` ("Written by Claude"), and the week's notes with `"week_by":
  "claude"`. The draft script never overwrites a game marked "claude" (unless `--force`). Keep
  each read to what the facts file and the archive show.
- **Who writes it:** the **Weekly Review** scheduled tasks (NFL Tuesday after the morning grade;
  college Monday), through a PR merged after the checks below. Each run reviews the newest
  graded week, then upgrades the oldest week still on drafts. "What can be improved" only names
  candidate tests; nothing changes the model without the keep rule.
- Linked from this repo's landing page (each card and a cross-link) and from the Weekly
  Gridiron Board.

## Checks before merging

- **Smoke test (added Nov 25, 2026).** `python scripts/smoke.py` loads the pages listed in
  `smoke.json` in headless Chromium, from this checkout, and fails on a script error, a missing
  script or stylesheet, a table with too few rows, or "undefined"/"NaN" showing on a page. The
  **Smoke test** Action (`smoke.yml`) runs it on every pull request and on `main` when a merge
  changes page code; data-only commits don't start it. A red run on a pull request means don't
  merge. If a page is added, renamed or removed, or an element it checks is renamed, update
  `smoke.json` in the same change. The counts are set for the season; out of season, lower any
  that an empty schedule would trip. `scripts/smoke.py` is the same file in rhyman7/nfl,
  rhyman7/cfb, rhyman7/fb-value-board and rhyman7.github.io: change it in all four.
  The two boards are tested as `site/template.html` and `site/cfb-template.html` with the
  current `docs/data.json` inlined, the same way the build does it, so a template change is
  checked before the board is rebuilt.
- `python -m pytest -q` passes (required for anything touching `nflmodel/`, `cfbmodel/` or
  the pipelines). `pytest.ini` limits collection to `tests/`: the `experiments/*_test.py`
  scripts are run by hand, and a `test_*` function in one of them would otherwise break the
  Update board Actions' test step.
- If a template changed: `python site/make_cfb_template.py`, then rebuild both boards
  (`python run.py --no-backtest`, `python run_cfb.py --no-backtest`) and commit the generated
  `docs/` files with the template.
- Serve the repo root (`python -m http.server`) and load the landing page and both boards'
  sections you touched at desktop and phone widths with no page errors.

## The boards on a phone (Nov 25, 2026; 760px and under, screen only)

Ryan approved the mockup (Design canvas "Value Board phone mockup") and chose the short summary
and the long columns behind a tap. Display only: nothing here changes a number, a pick or what
is listed. Wider screens and print must stay pixel-identical; compare old against new there.

- **Summary.** `#lede-stats` (three numbers: bets on the sheet, bets clearing break-even, the
  best EV) and `#lede-short` (one line) show in place of the `#lede` paragraph, which opens from
  "Read the full summary". `renderHero()` fills all three from the same values.
- **Jump strip.** `#jump`, the first child of `#view-games`, holds one button per section of
  the Games page, built from each section's `<h2>`; a new section gets a button automatically.
  They are buttons that scroll, not links: the page's hash is its router. It sticks under the
  top bar (`--bar-h`, measured in script).
- **Tables as cards.** `table()` puts each column's name on its cells (`data-label`); the phone
  rules lay out `table.sheet` (bet sheet, line movement) and `#bets` (best prices) as one card
  per row. The card rules place cells by position (`td:nth-child(n)`), so **adding, removing
  or reordering a column in one of those three tables means updating its card rules** in the
  phone block of `site/template.html`.
- **Bet sheet detail.** Each row ends with a cell that has no column (`td.more-cell`, hidden
  except on phones) holding the "Why, timing and record" button; it opens the three long
  columns (`tr.open`). The Stake cell carries `class="end"` so it stays the last column on
  wider screens.
- **Long lists.** `foldLong()` shows the first 5 bets of a tier, the first 5 line-movement
  sides and the first 16 game cards, then a "Show all N" button. Long section notes show three
  lines and "Read more" (`.note.clamp`). All of these buttons are hidden on wider screens.
- Colors come from the page's own tokens (`--surface`, `--rule`, `--ink`), so the cards follow
  the light and dark themes and any later restyle.
- **The other sections (Nov 26, 2026; Ryan approved the mockup, Design canvas "Value Board phone
  mockup: the other sections").** Every table in Players, Props, Past weeks, Team ratings and
  Model record shows as one card per row: `table()` adds the class `m-cards` to a table inside
  one of those views (`M_VIEWS`), and the three hand-built tables (`#players-table`,
  `#props-table`, `#hist-players`) add it themselves and put `data-label` on their cells. A new
  table built with `table()` in those views needs nothing extra. Rules to keep:
  - A blank number ("–") gets the class `nil` and is left out of a player's card.
  - Players and Props keep their headings as a strip of sort buttons, and a row still opens on
    a tap. A prop's Range, Floor, Close and Result show only while its row is open; the card
    rules place the props columns by position (`td:nth-child(n)`), so **a column added, removed
    or reordered in `#props-table` or `#ratings-table` means updating its card rules**.
  - Past weeks' players show "projection → actual" from `data-act` on the projection cell.
  - Team ratings: rank, team and rating; a tap opens offense, defense, market and quarterback
    (`tr.open`). The `.m-hint` line above it shows on phones only.
  - Long lists fold with `foldTable()` (it wraps `foldLong()` and keeps a table open through its
    redraws once "Show all" is tapped): 20 players, props and bets, 6 past games, 40 teams, and
    the first 6 rows of any other card table longer than 8.
  - Model record's headline numbers (`.tiles`) sit two to a row.
  - Every section's intro note folds to three lines; "Read more" shows only where there is more.
  - The phone block sits above the page's base rules, so a phone rule needs a more specific
    selector than the base rule it replaces (that is why some carry `main` or `.m-cards`).
- **A game's own page, the teaser table and a prop's ladder (Nov 26, 2026; Ryan approved the
  before-and-after preview and chose "best book, then Show every book").** Every price is one
  card per pick (`table.g-prices`). Every book is a small card per book under its pick
  (`table.g-books`): a phone shows each pick's first row (`tr.group`, the best price) and the
  "Show every book" button (`everyBook()`) adds the class `all` for the rest; Derivatives carries
  `all` from the start because its first row per pick is the sharp book. A prop's ladder is one
  card per rung. The page's other tables (game summary, matchup ranks, recent form, alternate
  lines) and `#tease-legs` stay tables, with tighter padding and wrapping headings so they fit.
- **Recommended, the Weekly Review and What's live on a phone (Nov 26, 2026; Ryan approved the
  before-and-after preview).** Tables that ran off the side show one card per row at 760px and
  under; wider screens and print are unchanged.
  - `docs/recommended/index.html`: its four tables carry `m-cards` plus `m-open`, `m-tracks`,
    `m-weeks` or `m-graded`. The labels are written in the page's style block by column
    position, so **a column added, removed or reordered there means updating that block**. Open
    picks shows the first 10 and "Every pick ever published" the first 15, each with a "Show
    all" button (`.m-more`).
  - `review/index.html`: `phoneCards()` runs after every redraw and labels each too-wide table
    in a `.scroll` box from its own heading row, so a new column or table needs nothing. More
    than 20 rows fold behind "Show all N", which stays open through a re-sort or filter.
    Column sorting is desktop-only.
  - `scripts/whats_live.py`: the waiting-changes table is `table.m-cards`, labels by column
    position in the script's style block.
- Not reworked for phones yet: the Best bets and Model leans print views.

## Landing page (`index.html`)

- Titled **Weekly Value Boards**. Same layout, tokens, Archivo type and NCAA/NFL logos as
  the Weekly Gridiron Board (rhyman7.github.io); keep it styled consistently with it.
- A second cross-link goes to the Recommended list (`docs/recommended/`, Nov 20, 2026).
- A cross-link under the cards goes to the Weekly Review (`review/`), and each card lists
  **Weekly Review** after Past Weeks.
- College card on the left, NFL on the right. Each card links to its board and to the seven
  sections above (Props included on both cards since Nov 7, 2026); if a section is added, renamed or reordered,
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

## Cross-links between pages (Nov 26, 2026)

Small links that carry a game or a player from one page to its counterpart. None changes a number.

- **Board game page -> matchup dashboard:** "Full matchup on the dashboard" beside "All games", for
  the board's first listed week only (`DASH_DIR` in `site/template.html`; the college template gets
  `/cfb` and uses the game's ESPN `game_id`). The dashboards' "Open on the value board" link comes
  back to the same game page (`#<game id>`), no longer to the unlisted Matchups view.
- **Past weeks -> Weekly Review:** each graded game gets a "Review" link when
  `review/data/<league>/index.json` lists that week (`REVIEW_DIR`; the address is
  `review/#<league>-<week>-<away>-<home>` in lower case, the review's own slug).
- **Weekly Review game -> the board's past matchup** (`#matchups/<season>_w<week>/<game id>`):
  "Projected against actual on the value board" under the game's header.
- **Player card -> Prop Ladder:** `/prop-ladder/#sport=NFL&player=<name>` (CFB on the college
  board). Prop Ladder finds the player by name and says so when no props are posted.
- **Recommended -> board game:** an open pick's game links to `../#<game id>` (the id is inside the
  pick's id). A game not yet on the board opens the board's Games page. Graded picks stay plain
  text, and in the weekly report the links print as plain text.
