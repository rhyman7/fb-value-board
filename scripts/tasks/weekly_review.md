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

2. **Which week.** In `docs/archive/` (NFL) or `docs/cfb/archive/` (college), `index.json` lists each week's `games` and `graded`. The targets are every week whose games are all graded and that has no `review/data/<league>/<season>_wNN.json` yet, or whose file has fewer games than the archive now has graded; do them oldest first (steps 3-4 for each), so no week is skipped. If the newest finished week isn't fully graded yet (NFL: Monday night's game; college: late Saturday games), wait 2 hours and look once more; if it is still short, review the graded games you have and say which are missing (the next run fills them in). If there is nothing new, skip to step 6.

3. **Facts and drafts.**
   `python -m nflmodel.review <league> <season> <week>` (downloads that season's play-by-play: nflverse for the NFL, cfbfastR for college; for the NFL also the schedule, snap counts, injury reports, weekly player stats and four seasons of weekly rosters; college context comes from data/cfb/cache, which the Update board Action keeps current; it prints how many games it wrote and how many had no play-by-play yet. If play-by-play is missing for some games, the source hasn't caught up: say so; the next run picks them up).
   Then `python -m nflmodel.reviewdraft <league> <season> <week>`: first-draft reads from the numbers, marked `"by": "draft"`.

4. **Write the reads.** Open `review/data/<league>/<season>_wNN.json` (facts) and `..._wNN.reads.json` (drafts). For **every game**, rewrite the read in your own words and set `"by": "claude"`. Keep the shape: `headline` (one or two sentences: the story of the game against the model's read), `verdict` (leave the draft's values unless the facts plainly show otherwise, e.g. a total decided by garbage-time points can be `"chance"`), and four lists of short bullets.

   **What Ryan wants (Nov 18, 2026).** The point of the review is to think outside the box: examine what happened in the game and raise possibilities, even though one game is not evidence. Three things matter most:
   - **Players who came up short** (`short` in the facts). Ryan cares much less about players who beat their projection (`beat`: one line at most). For each player who fell short, work out why from the facts: usage (targets, carries, attempts against the projection; target share against his usual), efficiency (yards per touch), snaps against his usual (NFL), the injury report and practice status (NFL), game script (`script`: share of his team's plays trailing or leading by 9+), the defenders who broke up or picked off his targets, his previous game and season average (was the projection chasing one big game?), a quarterback's sacks, interceptions and completion % over expected, and whether he faced a former team.
   - **Why the total and the cover landed where they did** (`flow`): each team's drives (TDs, field goals, punts, turnovers, turnovers on downs, missed field goals, points per drive), scoring by quarter, when the total went over, garbage-time points (scored with a team up 17+ in the fourth quarter), and the score that settled the cover (a late swing or settled early). Say plainly when garbage time or a late backdoor score decided a total or a cover.
   - **Possibilities** (`context`, `personnel`, `former_team`): previous game for each team (a good or bad week by the cover, the model's miss. Careful with "its rating is behind": the Nov 18, 2026 check found a run of misses does not carry into a team's next NFL game, and carries only about a tenth in college, with the line already holding it. Say so when you raise it, and look for what changed instead: a quarterback, injuries, a coach), record against the spread, rest, a bye, a short week, a division or conference game, a rematch, time zones, weather, kickoff slot; regulars who didn't play and players who played far more than usual (a new quarterback, a backup lineman, a defender back from injury), the defensive playmakers; players facing a former team (each side may know the other's habits; a coach facing a former team counts too). Get creative, but label every one as a possibility, not a conclusion.

   You may also look up what happened during the game (WebSearch, mode "standard"; a recap from ESPN, the AP, the team's site or a beat writer): an in-game injury, an ejection, weather, a benching, a coaching decision. Name the source in the bullet. Do this for every NFL game; for college, for games where a lean lost, the total missed by 14+, or a player fell 50%+ short. Never use a source you can't name, and never present news as a fact the facts file contradicts.

   The lists: **right** (leans that won on the merits with the line, the model's number and the result; the model closer than the market; projections that held), **wrong** (leans that lost on the merits and why, from `flow`; the players who came up short and why), **chance** (what the model couldn't see: an early exit, fumble bounces, defensive/return TDs, short missed field goals, garbage time, a late backdoor score; whether chance flipped a lean), **improve** (possibilities and candidate pre-set tests or data fixes; never a direct change to the model). 2-5 bullets each for the NFL; 1-4 for college.
   Every number must come from the facts file, the archive or a named source. Plain words, short sentences. Use the facts file's team abbreviations.
   Then rewrite `week_notes` (3-6 bullets: the week's records, model vs market on the average miss, the biggest real misses and what they had in common, how much chance and garbage time decided, players or teams that keep coming up short, anything that repeats from earlier weeks; read `review/data/<league>/index.json` and the previous week's reads for the season so far) and set `"week_by": "claude"` and the top-level `"by": "claude"`.
   Validate: `python -c "import json,sys; json.load(open(sys.argv[1]))" <file>`, then `python -c "import nflmodel.review as R; R.write_index('<league>')"`.

5. **Upgrade one older week.** If time allows, take the oldest week in `review/data/<league>/` whose reads file is not `"by": "claude"` and do step 4 for it too (its facts are already written; don't rerun step 3 for it).

6. **Check and publish.** `python -m pytest -q` must pass. Serve the repo root (`python -m http.server 8765 &`) and load `http://localhost:8765/review/#<league>-<week>` and two of its game pages headlessly with Playwright (Chromium at /opt/pw-browsers if available), at 1280 and 390 wide: no page errors, the tiles filled in, the reads showing "Written by Claude". Commit only `review/data/` on a new branch `claude/review-<league>-<season>-w<week>`, push, open a pull request (`gh api repos/rhyman7/fb-value-board/pulls -f title="Weekly Review: <LEAGUE> week <week>" -f head=<branch> -f base=main -F body=@<file>`), then merge it (`gh api -X PUT repos/rhyman7/fb-value-board/pulls/<n>/merge -f merge_method=squash`). If the merge is refused, leave the pull request open and tell Ryan to merge it.

7. **Report** to Ryan with SendUserMessage (load it with ToolSearch if hidden), a few lines: the week reviewed and the link (https://rhyman7.github.io/fb-value-board/review/#<league>-<week>); side and total leans, split into on the merits and decided by chance; model vs market average miss; the two or three biggest real misses in one line each; any candidate test that came up in more than one game; any older week upgraded; games still missing; the pull request link and whether it merged. If the season is over (NFL past mid-February, college past mid-January), say so, so he can turn the task off.

If anything is unclear, stop and tell Ryan instead of guessing.
