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
- **Kept separate for now.** Don't link the value boards from the matchup dashboards or
  rhyman7.github.io, and don't link those from here, until Ryan says otherwise.
- **What commits straight to `main`** (automated, not hand edits): the **Update board** and
  **Update college board** Actions (`.github/workflows/update.yml`, `update-cfb.yml`), which
  rebuild `docs/` and the committed files under `data/`. They share the `board-commits`
  concurrency group so they never push at once. Everything else goes through a PR.
- If a change touches what those Actions build or where it lands, check their steps and
  wording still match the site structure.

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
- Both boards share the same sections and anchors, in this order:
  This Week's Matchups `#matchups` | Games `#games` | Players `#players` | Past weeks `#history` |
  Team ratings `#ratings` | Model record `#model`. Each board's nav links to the other board.
- **This Week's Matchups** copies the matchup dashboards' look on purpose (graphite colors,
  condensed Archivo, team-color blocks lettered in the secondary color, never black): the
  board of this week's games, and `#matchups/<game id>` with each team's card showing this
  game's projections, not season stats. While it is open, `data-mx` on `<html>` switches the
  page tokens to graphite; every other section keeps the board's own look. Its styles are
  all under `.mx`. If the dashboards' week board or team card changes look, bring these along.

## Checks before merging

- `python -m pytest -q` passes (required for anything touching `nflmodel/`, `cfbmodel/` or
  the pipelines).
- If a template changed: `python site/make_cfb_template.py`, then rebuild both boards
  (`python run.py --no-backtest`, `python run_cfb.py --no-backtest`) and commit the generated
  `docs/` files with the template.
- Serve the repo root (`python -m http.server`) and load the landing page and both boards'
  sections you touched at desktop and phone widths with no page errors.

## Landing page (`index.html`)

- Titled **Weekly Value Boards**. Same layout, tokens, Archivo type and NCAA/NFL logos as
  the Weekly Gridiron Board (rhyman7.github.io); keep it styled consistently with it.
- College card on the left, NFL on the right. Each card links to its board and to the six
  sections above; if a section is added, renamed or reordered, update both cards.
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
- Every page keeps the disclaimer: for research and entertainment, past results do not
  predict future results, and nothing here is betting advice.
