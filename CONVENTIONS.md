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
  matchup feed `<league>.json` (`nflmodel/dashfeed.py`) and the Likeliest TD Scorers tracker
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
- **Wong teaser record (NFL): `data/teasers/`** (`nflmodel/teaserlog.py`, from Nov 6, 2026; the
  Project doc `claude/betting-strategies.md`, strategy 2). Every **Update board** run logs each side
  whose DraftKings, FanDuel or anchor line is in a Wong range (underdog +1.5 to +2.5, favourite -7.5 to
  -8.5) to `nfl_log.csv` (a row whenever its lines or total change, until kickoff; append-only),
  grades it after the game into `nfl_results.csv` (append-only) and writes the summary
  `nfl_record.json`. It reads the slate and writes only `data/teasers/`; the board shows none of it,
  and a failure never stops the build. The strategy's rule (underdog legs, total 49.5 or less, two-team
  teasers at -120 or better) is fixed in the module's constants; change it only with Ryan.
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
  `bestbets.csv` to ntfy and the open `best-bets` issue, nothing when the card is empty, at
  DraftKings and FanDuel only (`kickoff.MY_BOOKS`; a bet neither qualifies is listed as NOT A BET); it only
  dispatches existing workflows, reads and sends, and never changes the model) and commits only the
  rows that changed to `data/market/<league>_<YYYY-MM>.csv`. Append-only: never edit or delete
  a row. It shares the `board-commits` group. The board builds read these files and fall back
  to their usual lines when a game has no quotes; a failed snapshot never stops anything.
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
  - Reading it: `cfbmodel/pff_data.py` (`pff_data.load(table, seasons)`) looks in `$PFF_DATA_DIR`,
    then `.pff-data/` (git-ignored), then `../pff-data`. An Action that needs it checks it out with
    the **PFF_DATA_TOKEN** secret (a fine-grained token: read-only Contents on rhyman7/pff-data
    only): `actions/checkout@v5` with `repository: rhyman7/pff-data`, `token:
    ${{ secrets.PFF_DATA_TOKEN }}`, `path: .pff-data`. Never `git add` that folder; the Update
    Actions add only named paths, keep it that way.
  - **Used on the college board** (Nov 4, 2026): `cfbmodel/pff_roles.py` adds route share and snap
    share to the player table (`players.build`) for the receiving calibration (`players.CAL_EXTRA`).
    The **Update college board** Action checks out only the two tables it needs
    (`receiving_summary`, `offense_summary`) into `.pff-data/`; every college rebuild (kickoff,
    line alerts) runs through that Action, so a new workflow that builds the college board must
    add the same step. With no data, or PFF more than a week behind the board's latest games,
    both columns are 0 and the board is what it was before. Neither column goes into the board
    JSON, the card or any file under `docs/` or `data/`.
  - The pull (`pff_pull.py`, the **PFF pull** Action, the `PFF_API_KEY` secret) and its rules are
    in pff-data's CONVENTIONS.md. No PFF key is stored in this repo.
  - `scripts/pff_identify.py` (stamps season and week on hand exports from the website by matching
    box scores) writes into the pff-data checkout (`hand/`), never here.
- **College availability log: `data/cfb/availability/<season>/`** (`latest.json`, `log.csv`), written
  only by the **College availability** Action (`cfbmodel/availability_log.py`, standard library
  only, no pip step): ESPN's per-team college injury designations, which carry the power
  conferences' availability reports, with one `log.csv` row per change (new, status, details,
  cleared) and the time it was first seen. History for the 2026 recheck of kickoff-time
  availability; the board's Covers.com designations (`docs/cfb/designations.json`) are not
  read from it. Don't edit these files by hand. The box-score absence features stay in
  `cfbmodel/availability.py`; the two modules are unrelated.
- **CFBD player data: `data/cfb/cfbd/`** (`recruits_`, `portal_`, `returning_`, `roster_<season>.csv`),
  written by the manual **CFBD player data** Action (`cfbmodel/cfbd_pull.py`, `CFBD_API_KEY`).
  Inputs to experiments only (role prior, Nov 3, 2026: not kept); the board does not read them.
- **Hand overrides (NFL board): `data/overrides.csv`.** One row per player per game:
  `starter` sets that game's starting quarterback for the game model and makes him the
  quarterback who takes the team's passes in the player projections (whoever started the
  team's last game, and whatever the injury report says), `out` takes a player out of the
  player projections. Use it only for confirmed news the nflverse schedule or the
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
  has `kept`. `data.json` carries the summary as
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

## Model tests: the keep rule

- **Kept if it makes the model more accurate, however small (Ryan, Nov 4, 2026; replaces the
  old "holdout better by 2+ standard errors" bar).** Small gains add up.
- "More accurate" means the change's number is better on **every** accuracy measure the test
  reports, and worse on none: projection RMSE (or the game model's average miss) on the tuning
  seasons and on the holdout, and Brier at the real DraftKings/FanDuel lines where lines exist,
  props included. No standard-error threshold; standard errors are still reported.
- A mixed result (better on one measure, worse on another) is not kept. It goes to Ryan as a
  trade-off for him to decide.
- Still write the measures into the test script before running, still run the 2017-2025
  walk-forward (or the NFL equivalent) before anything is kept, and still give Ryan a short plan
  and wait for his go-ahead before a kept change goes on the live boards.

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
  Each board's nav links to the other board.
- **The manual (both boards): `#manual` and `docs/manual/`** (Oct 31, 2026). The Value Boards
  manual lives once, in `docs/manual/` (`index.html`, its screenshots in `img/`, and
  `Value_Boards_Manual.pdf`), hand-made, not generated by a build: the Update Actions never
  touch it. Each board's Manual view loads it in a frame sized to its content on first open
  (`openManual()`; the college page points one folder up, `../manual/`), with buttons to
  download the PDF and open it on its own page. A new edition replaces those files through a PR.
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
    vig-free chance (priced as the Parlays page prices college legs) and the model's chance then
    (college projections on `data/cfb/prop_params.json`, reception share standing in for target
    share, `cfbpropboard.with_target_share`). The row is never rewritten. Close comes from the
    game's file after kickoff; results from the archive's `actuals`. Not market-tested, so
    `flags_on` is false and nothing is flagged. Display only: it never feeds the model, the bets,
    the Bet sheet or the Parlays page.
  Never edit past weeks by hand; the files are rebuilt from the logs.
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
