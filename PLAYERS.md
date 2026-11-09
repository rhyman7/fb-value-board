# Player projections

The Players tab projects passing, rushing and receiving for every quarterback,
running back, receiver and tight end with a role in the coming games. Code:
`nflmodel/players.py` (model) and `nflmodel/players_data.py` (box scores).

## How a projection is built

Opportunity times efficiency, using only what was known before each game.

1. **Team volume.** Pass attempts and carries per game, from the team's recent
   volume, what the opponent has allowed, the game model's projected margin and
   total (teams that trail throw more), wind and home field. A regression fitted
   on the previous eight seasons. Since Nov 4, 2026 (the keep rule) both regressions
   also take the offense's pass rate over expected and the PROE its opponent has
   allowed, and the carries regression reads the opponent's carries allowed against
   what its opponents usually run ("vs usual") in place of the raw count
   (`players.ATT_FEATS`, `CAR_FEATS`, `tendency`; FEATURES.md "Re-judging past tests
   under the keep rule").
2. **Player share.** The player's share of team targets, carries and pass
   attempts in games he was active, recent games weighted more (6-game
   half-life, last season at 60%). Active status comes from weekly rosters and
   injury reports, so a game missed injured is not a zero. When teammates are
   out, the shares of those available are scaled up to fill the gap.
3. **Starting quarterback.** Last game's starter if he is active, unless another
   active quarterback is clearly the regular (80% or more of the team's recent
   passes), as when a starter returns from injury. With neither, the active
   quarterback with the largest share. He takes all the passes; the others get
   none. Right on 92.7% of team-games since 2019.
4. **Efficiency.** Catch rate, yards per target and per carry, completion rate,
   yards per attempt, touchdown and interception rates, each shrunk toward the
   position average. Rushing and passing are multiplied by what this defense
   has allowed compared with the league.
5. **Touchdowns.** Team touchdowns from the game model's implied points, split
   into passing and rushing by the team's tendency, then shared out by red-zone
   targets and carries. Anytime-TD chance assumes a Poisson count.
6. **Calibration.** A last regression per stat, fitted on earlier seasons, that
   blends the chain above with the player's recent per-game average (per start,
   for quarterbacks) and takes in wind and, for receivers, the defense's missing
   backs and pass rushers. Passing is fitted only on games the projected starter
   actually started (official starter from the schedule data), so a
   quarterback's line is what he throws **if he starts**. A passing prop is void
   if he does not play.

## Backtest

Walk-forward over every player-game since 2013, each season fitted only on
earlier ones. Average miss per player-game since 2019 (lower is better).
Quarterbacks are scored as the projected starter in games he started, with an
earlier start so the per-start baselines exist.

| Stat | Model | Season average | Recent average | Bias |
|---|---|---|---|---|
| Passing yards | 59.2 | 64.9 | 61.8 | +1.4 |
| Pass attempts | 6.74 | 7.33 | 6.98 | +0.2 |
| Completions | 4.77 | 5.21 | 4.97 | +0.2 |
| Passing TDs | 0.90 | 0.99 | 0.95 | −0.03 |
| Rushing yards | 15.9 | 16.5 | 16.1 | −0.1 |
| Carries | 2.72 | 2.84 | 2.78 | −0.1 |
| Receptions | 1.28 | 1.35 | 1.30 | −0.1 |
| Receiving yards | 16.4 | 17.3 | 16.7 | −0.3 |
| Anytime TD, receiving / rushing (Brier) | 0.194 / 0.208 | 0.223 / 0.237 | 0.207 / 0.221 | |

The model beats both baselines on every stat in both 2013-2018 and 2019 on.
Rerun with `python experiments/players_test.py`.

### The passing fix, October 2026

Passing yards had run about 7 yards low. The yardage math was fine; the cause
was the starter. About one projected starter in ten did not start (benched, or
a backup who had kept the job when the regular came back) and threw next to
nothing, and the calibration learned from those games, shrinking every
starter's projection. Starters who did start came out about 13 yards low.
Two changes, both better on 2013-2018 and on 2019 on:

| Since 2019 | Before | After |
|---|---|---|
| Passing yards bias, projected starter who started | −12.9 | +1.4 |
| Passing yards miss, every actual starter | 69.6 | 68.2 |
| Starter picked right, team-games since 2019 | 91.1% | 92.7% |

An earlier version of the test script lined up its season-average baseline
with the wrong rows after the first stat, which made that baseline look worse
than it is (for example 18.4 receiving yards instead of 17.3). Fixed; the
board's own table was not affected.

## What earns its place

Each piece removed in turn (`python experiments/players_ablation.py`):

- **Teammates out, usage passed on**: the biggest gain, for receiving,
  rushing and passing.
- **Game script** (projected margin and total): clear gain for passing yards.
- **Wind**: gain for passing yards, mostly since 2019.
- **What the defense allows**: helps passing and rushing; no help for
  receivers, so receivers use their own rates.
- **Defensive injuries**: small; kept only for receiving, where they helped on
  the tuning seasons.

The defense matters less than usage and game situation, the same pattern as
the game model.

### More inputs tested, October 2026

`python experiments/players_more_test.py` (average miss; tuning 2014-2018, holdout 2019 on):

| Change | Tuning | Holdout | Kept? |
|---|---|---|---|
| Team pass attempts with pace and neutral pass rate | 6.445 → 6.431 | 6.176 → 6.179 | No: no gain |
| Team carries, same | 5.607 → 5.599 | 5.588 → 5.577 | No: too small to matter |
| Boosted last step for quarterbacks (yards, attempts, completions, TDs) | worse on all four | worse on all four | No |
| Questionable multiplier (below), rushing yards, RBs and others | 15.07 → 14.05 | 12.61 → 11.88 | Yes, New projections |
| Same, receiving yards | 20.37 → 19.20 | 18.09 → 17.28 | Yes |
| Same, carries / targets / receptions | 4.58 → 4.32 / 2.40 → 2.27 / 1.77 → 1.68 | 3.12 → 3.03 / 1.92 → 1.87 / 1.51 → 1.47 | Yes |

Players listed Questionable who play get less than their usual usage, more so the less
they practiced: about 0.9 of the projection after a full practice, 0.8 after limited, 0.7
after a week of no practice (receiving yards, all seasons before 2026). The new
projections, which already see practice status, still ran high on these players. So the
New projections are scaled by a multiplier per position and practice status, fitted each
season on earlier seasons only and shrunk toward 1 by n / (n + 200)
(`players_ml.questionable`). Shrinkage of 50 scored better on the tuning seasons, 200
since 2019; 200 was taken because 50 overcorrected in recent seasons (projections then ran
low by about 2.5 yards). Scored on Questionable players who played, which is what a prop
settles on; a player who sits voids the bet.

### Route participation and snap share (Oct 31, 2026)

`python experiments/routes_test.py` asks whether route participation (how often a player is on
the field when his team drops back) would improve the target projections: the role measure PFF
sells weekly. nflverse publishes who was on the field for every play (NGS 2016-2022, FTN 2023
on) but only after the season (2026 is not out), so the board cannot use it live. Walk-forward
by season (fitted on the seasons before, tested 2019-2025), on wide receivers, tight ends and
backs who played, the error (mean squared) of the target share:

| Added to the model's projected target share | WR | TE | RB | All | Every season 2019-2025 |
|---|---|---|---|---|---|
| Snap share, as of the game and last game (free, weekly) | −2.2% | −2.6% | −1.9% | −2.25% | −1.8% to −2.6% |
| Snap share + route share + targets per route (PFF, weekly) | −3.0% | −3.8% | −2.6% | −3.1% | −2.9% to −3.4% |
| What routes add over snaps | −0.8% | −1.2% | −0.7% | **−0.85%** | −0.6% to −1.2% |

Targets per game: average miss 1.965 → 1.931 with both. **Three quarters of the gain is free**:
snap share, which nflverse publishes every week (from Pro Football Reference) and which the
board's main projections did not use then (only the New projections, `players_ml.py`, did; on the
2025 real lines they cut receiving yards' Brier gap to the books from 0.0070 to 0.0051 in the final run below,
`experiments/out/props_real_lines_ml_2025.json`). Routes would add under a point more. So the
next step is the free one: bring snap share into the main projections (or adopt the New
projections) and test on the 2023-2025 real lines; paid route data is worth considering only
after that.

### The New projections become the projections (Oct 31, 2026)

The New projections (`players_ml.py`, snap share and the other role inputs) now replace the
current ones for every stat and position they cover (`players_ml.ADOPT`: RB carries and rushing
yards; WR and RB receptions and receiving yards), on every page. The Players page's record grades
them: on backtest runs the boosted step runs walk-forward over the holdout seasons too.
Passing stays on the current projections: a boosted last step for quarterbacks was worse on all
four passing stats on both the tuning and holdout seasons (table above), and tight ends were not
better either, so they keep the current model.

Tested first on the real DraftKings and FanDuel lines, 2023-2025
(`python experiments/props_real_lines_ml.py --season <year>`; 2023 and 2024 have the yardage
markets only). Our Brier score minus the books' (lower is better; ± one standard error,
clustered by game), at the pricing snapshot and at the close:

| Stat (positions) | Season | Pricing: current → New | Close: current → New |
|---|---|---|---|
| Rushing yards (RB) | 2023 | +0.0105 → +0.0014 | +0.0117 → +0.0070 |
| | 2024 | +0.0048 → +0.0101 (worse) | +0.0098 → +0.0134 (worse) |
| | 2025 | +0.0008 → −0.0008 | +0.0091 → +0.0071 |
| | **All three** | **+0.0048 → +0.0033** | **+0.0102 → +0.0092** |
| Receiving yards (WR, RB) | 2023 | +0.0052 → +0.0051 | +0.0094 → +0.0077 |
| | 2024 | +0.0089 → +0.0092 | +0.0124 → +0.0098 |
| | 2025 | +0.0070 → +0.0051 | +0.0079 → +0.0067 |
| | **All three** | **+0.0071 → +0.0065** | **+0.0099 → +0.0081** |
| Carries (RB) | 2025 | +0.0063 → +0.0018 | +0.0187 → +0.0056 |
| Receptions (WR, RB) | 2025 | +0.0102 → +0.0043 | +0.0106 → +0.0089 |

New is better on every stat over the three seasons and in 2023 and 2025 on all of them. 2024
went the other way for RB rushing yards (about one standard error), so the gain there is
smaller and less certain than for receiving. The books are still the better forecast on every
line (the gap is above zero almost everywhere), so this narrows the gap; it does not close it,
and no prop bets are flagged.

## Long plays (tested October 2026, not adopted)

Ryan's question: would projections be more accurate if a player's efficiency ignored his
unusually long plays? Tested by capping each run and catch at 15, 20 or 30 yards when
measuring yards per carry and per target (yards beyond the cap came back at the position's
average rate), with and without capping the recent average the calibration uses. The
outcomes were never capped. `python experiments/players_capped_test.py`.

No variant helped. Average miss per player-game, 2019 on (same player-games for each):

| | No cap | Cap 15 | Cap 20 | Cap 30 |
|---|---|---|---|---|
| Rushing yards, RB | 21.99 | 22.06 | 22.04 | 22.02 |
| Rushing yards, QB | 14.79 | 14.78 | 14.79 | 14.79 |
| Receiving yards, WR | 23.23 | 23.22 | 23.22 | 23.23 |
| Receiving yards, TE | 17.89 | 17.93 | 17.92 | 17.90 |
| Receiving yards, RB | 14.16 | 14.19 | 14.18 | 14.17 |

The tuning seasons (2013-2018) showed the same: differences in the second decimal, mixed in
direction, and capping the recent average was worse everywhere. The model already damps one
long play: yards per carry is shrunk toward the position average by 60 carries and yards per
target by 40 targets, so a single 70-yard run moves a back's rate little. The switch is
`players.EFF_CAP` (off).

## Limits

- No cornerback-by-receiver matchups: that needs coverage charting (PFF or Next
  Gen Stats), which is not public. The defense enters by position group and
  through its injured backs. PFF's man/zone split (by defense and by receiver) was
  tested on Nov 5, 2026 and not kept (below, "Man and zone coverage").
- Game-day inactives are named about 90 minutes before kickoff. Earlier in the
  week only injured reserve and Out or Doubtful count.
- A quarterback's line assumes he starts. The starter pick is wrong about one
  game in fourteen, mostly benchings announced during the week; check the news.
- Prop probabilities and the comparison with sportsbook lines: [PROPS.md](PROPS.md).

### Defense against each position (Nov 1, 2026)

The matchup dashboards split pass defense by position (yards allowed to wide receivers, tight
ends and backs). `experiments/def_vs_position.py` built that split as a rating per defense,
position and stat: the log of what it allowed over what the projections expected those players
to get (so it is already adjusted for the offenses it faced), averaged over its earlier games
(6-game half-life, last season at half weight) and shrunk toward 0 by 4 games. The projection
is multiplied by exp(b x rating), b fitted on earlier seasons (`nflmodel/dvp.py`).

| Position, stat | Chain projections, holdout 2019 on (b fitted 2013-2018) | New projections, 2022 on (b fitted 2019-2021) |
|---|---|---|
| WR receiving yards | 19.91 → −0.09 ± 0.01 | 18.69 → −0.02 ± 0.004 |
| TE receiving yards | 12.44 → −0.05 ± 0.004 | (chain shown) |
| RB receiving yards | 8.97 → −0.12 ± 0.006 | 8.55 → −0.13 ± 0.009 |
| WR / TE / RB receptions | −0.0023 / −0.0027 / −0.0072 | −0.0010 / (chain) / −0.0030 |
| RB rushing yards | 16.41 → −0.11 ± 0.01 | 15.96 → −0.07 ± 0.01 |
| RB carries | −0.0061 ± 0.0011 | −0.0084 ± 0.0021 |
| QB passing yards | 68.93 → −0.21 ± 0.14 (not kept) | |

Every pair but passing yards clears two standard errors on both sets of projections
(`experiments/dvp_ml_test.py`), so it is applied to the projections the board shows: the chain
for tight ends, the New projections for the stats in `players_ml.ADOPT`. Gains are small, as
the defense's were before: usage and game script still do most of the work.

**It is not an edge against the books.** On 2025 DraftKings and FanDuel main lines the rating
tied the books' own chance (pooled Brier −0.00003 ± 0.00018), and the props it moved 2+ points
off the books went 155-162, −7.4% ± 4.8%, closing line value −4.7% ± 0.4%. The books price
the matchup. Receiving yards alone scored better than the books on 2025 (−0.00061 ± 0.00021),
one of several stats tried: a lead for the live prop log, not a result.

### What a defense allows in volume, against what its opponents usually do (Nov 1, 2026)

The team-volume regression uses the raw pass attempts and carries a defense's opponents have had
against it. That count also carries the schedule (the offenses it happened to face) and those
games' script. `python experiments/volume_allowed_test.py` (results in
`experiments/out/volume_allowed_test.json`) tried two cleaner versions, from the defense's earlier
games only: opponents' attempts against it minus those offenses' own recent averages ("vs
usual"), and minus what the volume model predicted for those games, which also knows their
projected margin and total ("vs expected"). Each replaced or joined the raw count. Average miss
on team-games (tuning 2014-2018 / holdout 2019 on; current 6.445 / 6.176 pass attempts, 5.607 /
5.586 carries):

| Variant | Pass attempts, tuning | holdout | Carries, tuning | holdout |
|---|---|---|---|---|
| vs usual, replacing the raw count | −0.020 ± 0.012 | +0.001 ± 0.005 | −0.010 ± 0.005 | −0.007 ± 0.004 |
| vs usual, added | −0.037 ± 0.018 | +0.009 ± 0.005 | −0.008 ± 0.005 | +0.001 ± 0.002 |
| vs expected, replacing | −0.006 ± 0.008 | +0.005 ± 0.005 | +0.019 ± 0.005 | +0.002 ± 0.005 |
| vs expected, added | −0.022 ± 0.007 | +0.012 ± 0.006 | +0.006 ± 0.004 | +0.005 ± 0.002 |

**Not kept.** For pass attempts every version helped the tuning seasons and none held on the
holdout. For carries "vs usual" in place of the raw count passes the rule by a hair, but the gain is
under a hundredth of a carry per team-game (5.586 → 5.580), too small to change any projection.
Defenses' tendencies to face more or fewer passes are mostly the offenses they meet and the score:
the raw count, the team's own averages and the projected margin already carry what there is.

*Nov 4, 2026: applied for carries under the keep rule*, together with PROE, after the head-to-head in
FEATURES.md "Re-judging past tests under the keep rule", "Applied".

### Coaching changes and playing surface on the player projections (Nov 2, 2026): not applied

Both were rejected for the game models (FEATURES.md); this checks the props side.
`python experiments/props_context_test.py` then `python experiments/props_context_books.py`
(results in `experiments/out/props_context_test.json`, `props_context_books.json`). Flags per
player-game: new head coach since last season (all season, and weeks 1-4), mid-season coaching
change, game on turf, on turf outdoors. Each acts on a projection as a shift (x exp(b x flag)) or a
pull toward the group average, fitted per stat and position on earlier seasons; scored on the New
projections for the stats the board shows them for (2022 on), the chain projections otherwise
(2019 on). Rule, set before running: 2+ standard errors better on the scored seasons, no worse on
the fitted ones; then checked against the books' 2025 lines.

**Projections:** 24 of 100 stat-by-flag versions pass, all shrinking projections a little (players
on new-coach teams and in turf games came in slightly under their projections). All are tiny:
the largest are backs' receiving yards −0.03 yards on an 8.6-yard miss (new coach; turf), tight
ends' receiving yards −0.025 on 12.4 (turf outdoors), backs' rushing yards −0.012 on 16.0 (new
coach), quarterbacks' passing yards −0.18 ± 0.08 on 68.9 (turf). With 100 versions tried, a few
of the borderline ones are chance.

**Against the books (2025 main lines):** none helps. Brier against the book, for the ones with
lines to fit on: QB passing yards / turf −0.0001 ± 0.0014; RB receiving / new coach −0.0007 ±
0.0009; RB rushing / new coach 0.0000 ± 0.0008; RB receiving / turf +0.0008 ± 0.0005 worse; TE and
WR versions +0.0008 to +0.0013 worse. The books already price coaching changes and surface.

Not applied: gains of a few hundredths of a yard aren't worth the extra inputs, and they don't
move any prop against the market.

### Top receiver against the rest: a shadow-corner proxy (Nov 2, 2026)

No coverage charting is public, so cornerback-by-receiver matchups can't be modelled directly.
The proxy: label each team's wide receiver with the highest projected receiving yards its WR1 and
the rest WR2+, and build the defense-against-position rating (nflmodel/dvp.py) separately for
each. An elite corner who follows the WR1 should make a defense's WR1 rating lower than its WR2+
rating, consistently. `python experiments/wr1_matchup_test.py` (results in
`experiments/out/wr1_matchup_test.json`), chain projections, tuning 2013-2018, holdout 2019 on.

**Is it a stable trait?** Across 417 defense-seasons with 8+ games, odd weeks against even weeks:
how much a defense held WR1s below projection correlates −0.02 between the halves (none), the WR2+
version 0.20, and the WR1-minus-WR2+ gap 0.08. A defense's effect on the other team's top
receiver is not repeatable; its effect on the secondary receivers is. Denver: WR1s at 99% of
projection in 2023, 90% in 2024, 47% in 2025, 65% in 2026 so far (WR2+ 68%): a strong pass
defense overall, not a steady WR1-specific effect.

**Does it improve projections?** Receiving yards, average miss 19.82 on the holdout:

| Change | Tuning | Holdout |
|---|---|---|
| The board's single WR rating, applied at its own strength for WR1 and for WR2+ (b 0.10 / 0.48 instead of 0.27) plus each role's own rating | −0.054 ± 0.006 | −0.059 ± 0.005 |
| of which the role's own rating (over the split strength alone) | −0.015 ± 0.002 | −0.020 ± 0.002 |
| ... for WR1s | +0.002 | +0.001 |

It passes the projection rule (receptions too: −0.0013 ± 0.0003), but the gain is in the WR2+
rows: matchups move secondary receivers about four to five times as much as WR1s, whose
projection barely depends on the defense. The WR1 rows get slightly worse (+0.05 yards), so
nothing here supports cutting a top receiver for a shadow corner. Size: 0.06 yards on a 19.8-yard
miss (0.3%). Not applied: the board shows the New projections for WR receiving (players_ml.ADOPT),
so it would need re-checking on those and against the books' 2025 lines first, where the
position rating itself tied the books.

**Recheck where it would be used (Nov 2, 2026): not applied.** `python experiments/wr1_split_recheck.py`
(results in `experiments/out/wr1_split_recheck.json`). Rule, set before running: on the board
only if it lowers the miss on the New projections (the ones shown for wide receivers; fitted
2019-2021, scored 2022 on) by more than 2 standard errors, and is no worse than the current WR
rating against the books' 2025 lines.

| Check | Receiving yards | Receptions |
|---|---|---|
| New projections, 2022 on (miss 18.67 yards / 1.263) | **+0.069 ± 0.009 worse** (WR1 +0.068, WR2+ +0.069) | **+0.0037 ± 0.0007 worse** |
| Against the books, 2025 Brier, split minus current WR rating | −0.0004 ± 0.0004 (no difference) | not scored (no 2023-24 lines to fit on) |

The split fails the first check. On the New projections it makes both WR1s and WR2+ worse: those
projections already carry each receiver's role and usage (target and route share, roles.features),
so the role split repeats what they know and the extra coefficients fit noise (the role ratings
come out negative). The gain was real only on the older chain projections, which lack those role
inputs. The board keeps its single WR rating.

### Man and zone coverage (PFF), Nov 5, 2026: not kept

Ryan asked whether the model can tell which cornerback a receiver will face. PFF doesn't sell
receiver-by-corner charting, but its Receiving vs Scheme and Defense Coverage Scheme tables split
every receiver's routes and every defender's coverage snaps into man and zone, by game week, 2014 on
(rhyman7/pff-data, `nfl/`, private). `python experiments/coverage_scheme_test.py` (results in
`experiments/out/coverage_scheme_test.json`; rule in its docstring, committed before the run).

**Which half is a trait?** A defense's man rate is steady: odd weeks against even weeks 0.82 (416
team-seasons), one season to the next 0.56. A receiver's man-minus-zone yards per route is not:
odd against even weeks 0.03-0.05, one season to the next 0.19 for wide receivers (691) and 0.11 for
tight ends, against about 0.5 for his overall yards per route. College is the same or weaker
(defense 0.80; receivers 0.04 and 0.07).

**Inputs, known before the game** (PFF weeks through the week before): D, the defense's man rate this
season plus 750 snaps of last season's, centred on the week's average; G, the receiver's log ratio of
man to zone rate per route (yards or receptions), each shrunk 150 routes toward his overall rate,
earlier seasons at 0.6 a season, capped at ±0.5. Candidates on top of the board's defense-vs-position
rating: **A** adds a × D, **B** adds a × D + c × D × G, coefficients by position group (WR, TE, RB).

| Measure | Receiving yards: base → A / B | Receptions: base → A / B |
|---|---|---|
| Chain RMSE, 2014-2018 (fitted here) | 23.020 → 23.018 / 23.014 better | 1.6674 → 1.6670 / 1.6669 better |
| Chain RMSE, 2019 on | 22.050 → 22.052 / 22.051 **worse** | 1.6225 → 1.6223 / 1.6223 better |
| Board's projections (New for WR/RB, chain TE), fit 2019-21, RMSE 2022 on | 21.407 → 21.408 / 21.408 **worse** | 1.5755 → 1.5756 / 1.5758 **worse** |
| 2025 DK/FD main lines, Brier (book 0.25020) | 0.24945 → 0.24962 / 0.24964 **worse** | not scored (lines for 2025 only, nothing to fit on) |

**Verdict: not kept, neither candidate.** Each is better on 3 of the 7 numbers and worse on 4, and the
four are the ones that matter: the holdout for yards, the projections the board actually shows, and the
books. No difference reaches 1.5 standard errors (the largest are the fitted seasons: A on 2014-2018
receptions −0.0012 ± 0.0008, B on 2014-2018 yards −0.28 ± 0.20, in squared error). It's the in-sample fit that looks good; out of sample it adds noise. Waiting two weeks for
PFF's data (lag 2) gives the same picture. One thing to watch, not a result: on the board's projections
the receiver edge (B) helps wide receivers alone (26.918 → 26.916) and hurts tight ends and backs, the
same direction as its season-to-season 0.19; one subgroup of three, chosen after the fact.

Why it adds nothing: the New projections already carry each receiver's role and his
defense-vs-position matchup, and a defense's coverage mix is part of what its WR/TE/RB rating already
measures. The receiver's own man/zone split needs a season or more of routes before it means anything.
Recheck only if PFF's receiver splits ever get a two-season sample in the board's own projections; the
pull is kept current by hand (pff-data, `--league nfl`).

### PFF route share in the New projections (Nov 5, 2026): one measure flat, a trade-off for Ryan

`PFF_DATA_DIR=../pff-data python experiments/nfl_pff_routes_test.py` (results in
`experiments/out/nfl_pff_routes_test.json`; PFF rows stay in the private rhyman7/pff-data). The real
weekly PFF numbers (NFL `receiving_summary`, 2014-2026, pulled Nov 5) in the boosted New projections
for WR and RB receptions and receiving yards: route share (decayed and last game), its trend, and
targets per route run, from earlier games only. 99.9% of PFF rows match the board's players by
nflverse's `pff_id`. Rule, set before running: kept only if better on all 12 measures below.

| Projection RMSE (walk-forward) | Tuning 2015-2018 | Holdout 2019-2025 |
|---|---|---|
| WR receptions | −0.0065 ± 0.0027 | −0.0020 ± 0.0016 |
| WR receiving yards | −0.053 ± 0.049 | −0.057 ± 0.027 |
| RB receptions | −0.0020 ± 0.0030 | −0.0001 ± 0.0019 |
| RB receiving yards | −0.072 ± 0.034 | **+0.0007 ± 0.019** |

| Brier at the real DK/FD lines (WR + RB), pooled | Pricing (pregame) | Close |
|---|---|---|
| Receiving yards, 2023-2025 | −0.00052 ± 0.00058 | −0.00064 ± 0.00046 |
| Receptions, 2025 | −0.00160 ± 0.00092 | −0.00143 ± 0.00090 |

Better on 11 of 12, and in every season at both snapshots against the books (the gap to the books
narrows by about a tenth). The one miss is RB receiving yards on the holdout, worse by 0.0007 yards
(0.04 standard errors, effectively flat). Under the keep rule that is a trade-off, so it is not
applied; Ryan decides. If applied, the NFL `receiving_summary` needs adding to the PFF pull's
schedule, and PFF posts NFL weeks a few days late (week 9 was still empty on Thursday, Nov 5), so
the latest game's routes may be missing at Thursday-night and early pricing.

**Ryan (Nov 5, 3:43 PM): apply as-is. Applied Nov 5, 2026** (`nflmodel/pff_routes.py`, added to
`players_ml.add_features`; `players_ml.ROUTE_F`). The board's code reproduces the test's walk-forward
numbers exactly (`experiments/apply_check_nfl_pff_routes.py --rmse`, results in
`experiments/out/apply_check_nfl_pff_routes.json`). One addition: rookies whose nflverse record has
no PFF id yet are matched by name (four 2026 rookies were otherwise counted at 0 routes every game,
up to 17 yards low). Two guards for games PFF hasn't charted yet were tried and dropped, each
making the backs' holdout worse (the board keeps the test's rule: a player who played with no
PFF row that week ran 0 routes). On the first upcoming week the change moves WR receiving yards by
a median 1.5 yards (95th percentile 4.6, largest 7.6), RB receiving yards 0.9 (largest 5.1),
receptions about 0.1. The data comes from pff-data's daily NFL pull (10:35 UTC, before Update
board); without it the board falls back to the projections from before.

### Quarterback rushing per start (Nov 9, 2026): six of eight, a trade-off for Ryan

`python experiments/qb_rush_per_start_test.py` (results in `experiments/out/qb_rush_per_start_test.json`;
rule committed before the run). Found from one row: Carson Wentz, the projected starter in week 10, at
0.6 carries against a book line of 2.5.

**The fault.** A player's share of the carries is averaged over the games he was active for. A backup
quarterback is active every week without playing, so his share is mostly zeros. When he takes over,
the starter rule gives him every pass and his passing form is already his per-start average, but his
rushing still uses the diluted share and the diluted per-game average. Projected starters in games
they started, 2014 on, by how much of their recent active time were starts:

| Recent active games that were starts | Starts | Carries: projected / actual | Rushing yards: projected / actual | Rushing-yards RMSE |
|---|---|---|---|---|
| Under 30% | 320 | 0.69 / 3.04 | 3.9 / 15.8 | 20.5 |
| 30-60% | 459 | 1.36 / 2.45 | 7.6 / 13.2 | 15.6 |
| 60-85% | 625 | 2.28 / 2.95 | 13.1 / 16.5 | 16.2 |
| 85% and up (established) | 5,046 | 2.79 / 2.81 | 16.3 / 16.5 | 16.0 |

**The change tested ("per start"),** for the projected starter's carries and rushing yards only: his
share of the carries in the games he started (prior: the average starter's share in earlier seasons),
his per-start average as his form, and the same small last-step regression fitted on projected
starters who started. It mirrors what passing already does. No other row or stat is refitted.

| Measure (per start against current) | Rush attempts | Rushing yards |
|---|---|---|
| RMSE, tuning 2014-2018 (2,530 starts) | 1.848 → 1.784 (**−0.064**) | 14.638 → 14.450 (**−0.188**) |
| RMSE, holdout 2019 on (3,920) | 2.264 → 2.116 (**−0.148**) | 17.220 → 16.708 (**−0.512**) |
| Brier at the lines, pricing snapshot | +0.0070 ± 0.0051 (223, 2025) | **−0.0006 ± 0.0024** (1,276, 2023-25) |
| Brier at the lines, close | +0.0067 ± 0.0067 (273, 2025) | **−0.0048 ± 0.0026** (1,907) |

**Six of eight: not kept by the rule.** The two misses are the 2025 rush-attempt lines, the only
season with them. With the change, under-30% starters project for 2.83 carries and 15.6 yards (actual
3.04 and 15.8) and their rushing-yards RMSE falls from 20.5 to 16.1. The cost is on established
starters, who move from 2.79 to 2.95 carries (actual 2.81) and 16.3 to 17.5 yards (actual 16.5): a
slight over-projection that pushes the average over chance at the 2025 attempt lines to 46.7% when
44.7% went over (current 44.1%). 2026 alone (255 starts): carries 2.048 → 1.945, yards 15.941 →
15.592.

Week 10 under the change: Wentz 0.6 → 2.5 carries and 2.8 → 9.8 yards; Drew Lock 0.4 → 1.7 and 3.0 →
12.9; Gardner Minshew 0.3 → 1.4 and 1.5 → 7.4; established starters within about half a carry
(the largest, Jayden Daniels 5.5 → 4.6).

Seen on the way, not changed: passing yards run high for new starters (under 30%: projected 221.0,
actual 204.3; 30-60%: 226.0 against 208.4; established 243.4 against 245.3).

**Nothing changes on the board.** For Ryan: apply as tested, or test a narrower version first (the
per-start numbers only for quarterbacks without an established run of starts).
