# Weekly Review task (set up Nov 17, 2026 in the CFB/NFL Model project)

Instructions for the two **Weekly Review** scheduled tasks: NFL on Tuesday afternoon, college on
Monday afternoon. The task's prompt says which league (`nfl` or `cfb`). Written by Claude at
Ryan's request; Ryan wants, for every finished game: what went right, what went wrong, what
was chance, and what can be improved, so he and Claude can see where the misses are and where
the predictions were close. The page is https://rhyman7.github.io/fb-value-board/review/.

**Scope.** This task only writes files under `review/data/`. It never changes the model, the
betting logic, the boards, the archive, the ledger, `data/review_chance.json` or any code. If a
fix to `nflmodel/review.py` or `nflmodel/reviewdraft.py` seems needed, stop and tell Ryan.
"What can be improved" only names candidate tests: nothing changes the model without Ryan's
pre-set keep rule.

GitHub here: GraphQL is blocked, so use `gh api` (REST) only; `gh pr create` fails.

## Steps

1. **Repo.** cd /home/claude/fb-value-board (if missing: `git clone https://github.com/rhyman7/fb-value-board.git /home/claude/fb-value-board`, allow 10 minutes). `git fetch origin main && git checkout main && git merge --ff-only origin/main`. Read `CONVENTIONS.md`, section "Weekly Review". `gh api repos/rhyman7/fb-value-board --jq .permissions.push` must print true; if not, do the work anyway, don't push, and make the first line of your report "Push access: NONE".
   `pip install --break-system-packages -q -r requirements.txt pyarrow`.

2. **Which week.** In `docs/archive/` (NFL) or `docs/cfb/archive/` (college), `index.json` lists each week's `games` and `graded`. The target is the newest week whose games are all graded and that has no `review/data/<league>/<season>_wNN.json` yet, or whose file has fewer games than the archive now has graded. If the newest finished week isn't fully graded yet (NFL: Monday night's game; college: late Saturday games), wait 2 hours and look once more; if it is still short, review the graded games you have and say which are missing (the next run fills them in). If there is nothing new, skip to step 6.

3. **Facts and drafts.**
   `python -m nflmodel.review <league> <season> <week>` (downloads that season's play-by-play: nflverse for the NFL, cfbfastR for college; it prints how many games it wrote and how many had no play-by-play yet. If play-by-play is missing for some games, the source hasn't caught up: say so; the next run picks them up).
   Then `python -m nflmodel.reviewdraft <league> <season> <week>`: first-draft reads from the numbers, marked `"by": "draft"`.

4. **Write the reads.** Open `review/data/<league>/<season>_wNN.json` (facts) and `..._wNN.reads.json` (drafts). For **every game**, rewrite the read in your own words and set `"by": "claude"`. Keep the shape: `headline` (one or two sentences: the story of the game against the model's read), `verdict` (leave the draft's values unless the facts plainly show otherwise), and four lists, each 1-4 short bullets:
   - **right**: what the model got right: a lean that won on the merits (give the line, the model's number and the result), being closer than the market, player projections that held.
   - **wrong**: leans that lost on the merits, the market being closer, and the biggest real player misses with a reason the facts support (game script, a teammate out, a role change).
   - **chance**: only what the facts file shows: a starting QB leaving early (quarter and time), fumble bounces, defensive/return TDs, short missed field goals, the margin with even luck, the points defense and returns put on the total, a team that out-played its opponent (success rate, EPA per play) and still lost. Say when chance flipped a lean and when it didn't.
   - **improve**: concrete candidate tests or data fixes (a role the board missed, a projected player who didn't play, a pattern worth a pre-set test), or say plainly that one game is no evidence. Never recommend changing the model directly.
   Rules: every number must come from the facts file or the archive. No outside news, no guesses about injuries the facts don't show. Plain words, short sentences. Use the facts file's team abbreviations. For college (60+ games) keep each list to 1-3 bullets.
   Then rewrite `week_notes` (3-6 bullets: the week's records, model vs market on the average miss, the biggest real misses and what they had in common, how much chance decided, anything that repeats from earlier weeks; read `review/data/<league>/index.json` for the season so far) and set `"week_by": "claude"` and the top-level `"by": "claude"`.
   Validate: `python -c "import json,sys; json.load(open(sys.argv[1]))" <file>`, then `python -c "import nflmodel.review as R; R.write_index('<league>')"`.

5. **Upgrade one older week.** If time allows, take the oldest week in `review/data/<league>/` whose reads file is not `"by": "claude"` and do step 4 for it too (its facts are already written; don't rerun step 3 for it).

6. **Check and publish.** `python -m pytest -q` must pass. Serve the repo root (`python -m http.server 8765 &`) and load `http://localhost:8765/review/#<league>-<week>` and two of its game pages headlessly with Playwright (Chromium at /opt/pw-browsers if available), at 1280 and 390 wide: no page errors, the tiles filled in, the reads showing "Written by Claude". Commit only `review/data/` on a new branch `claude/review-<league>-<season>-w<week>`, push, open a pull request (`gh api repos/rhyman7/fb-value-board/pulls -f title="Weekly Review: <LEAGUE> week <week>" -f head=<branch> -f base=main -F body=@<file>`), then merge it (`gh api -X PUT repos/rhyman7/fb-value-board/pulls/<n>/merge -f merge_method=squash`). If the merge is refused, leave the pull request open and tell Ryan to merge it.

7. **Report** to Ryan with SendUserMessage (load it with ToolSearch if hidden), a few lines: the week reviewed and the link (https://rhyman7.github.io/fb-value-board/review/#<league>-<week>); side and total leans, split into on the merits and decided by chance; model vs market average miss; the two or three biggest real misses in one line each; any candidate test that came up in more than one game; any older week upgraded; games still missing; the pull request link and whether it merged. If the season is over (NFL past mid-February, college past mid-January), say so, so he can turn the task off.

If anything is unclear, stop and tell Ryan instead of guessing.
