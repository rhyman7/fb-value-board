"""Make site/cfb-template.html from the NFL page (site/template.html).

The college board uses the NFL board's design. Rather than keep two copies that drift
apart, this script applies the college changes to the NFL page: names and wording, rank
bands sized to the number of FBS teams, receptions instead of targets, the opening-total
table, the projection's own grade in past weeks, and the college method and limits.
Every change must find its text exactly once (or the stated number of times); if the NFL
page has changed underneath one, the script stops and says which.

    python site/make_cfb_template.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "template.html", HERE / "cfb-template.html"

P = []   # (old, new, expected count or None for "at least one")


def sub(old, new, n=1):
    P.append((old, new, n))


# ---- names and storage ------------------------------------------------------------
sub("<title>NFL Value Board</title>", "<title>College Football Value Board</title>")
sub('href="../favicon', 'href="../../favicon', 2)   # icons: docs/cfb/ is one folder deeper than docs/
sub('href="../apple-touch-icon', 'href="../../apple-touch-icon')
sub('<i aria-hidden="true"></i>NFL Value Board</a>', '<i aria-hidden="true"></i>College Value Board</a>')
sub('        <li><a href="cfb/" data-view="cfb">College board</a></li>\n', '        <li><a href="../" data-view="nfl">NFL board</a></li>\n')
sub(' | NFL Value Board"; }', ' | College Value Board"; }', 2)   # game page, matchups board
sub(' | NFL Value Board`;', ' | College Value Board`;')              # one matchup (this week or a past week)
sub('document.title = "NFL Value Board";', 'document.title = "College Value Board";')
sub('"NFL Value Board: best bets, Week "', '"College Value Board: best bets, Week "')
sub('"nflvb.bankroll"', '"cfbvb.bankroll"', None)
sub("No data yet. Run python run.py to build the board.", "No data yet. Run python run_cfb.py to build the board.")

# ---- this week's matchups: college names, conferences and no targets ---------------------------
sub('  const LEAGUE = "nfl";', '  const LEAGUE = "cfb";')
sub('const LG = "nfl";', 'const LG = "cfb";')   # the bet sheet: wind unders are an NFL bucket only

# ---- games ---------------------------------------------------------------------------
sub('$("#stamp").textContent = M.season + " season. Results through " + M.games_through + ". Updated " + M.generated + ".";',
    '$("#stamp").textContent = M.season + " season, FBS games against FBS opponents. Results through " + M.games_through + ". Lines: " + M.lines_from + ". Updated " + M.generated + ".";')
sub("ticks = [-21, -17, -14, -10, -7, -3, 0, 3, 7, 10, 14, 17, 21].filter(t => t > lo && t < hi);",
    "ticks = [-42, -35, -28, -21, -14, -10, -7, -3, 0, 3, 7, 10, 14, 21, 28, 35, 42].filter(t => t > lo && t < hi);")
sub('mark.append(S("text", { x: 4, y: cy + 6.5, class: "disp t-ink", "font-size": 20, "font-weight": 800 }, g.away + " @ " + g.home));',
    'mark.append(S("text", { x: 4, y: cy + 5.5, class: "disp t-ink", "font-size": 16, "font-weight": 800 }, g.away + " @ " + g.home));')
sub('const left = 96, right = wide ? 138 : 6', 'const left = 112, right = wide ? 150 : 6')
# the opening total, when the data has one
sub('["Total", g.opening_spread != null ? "" : (o ? o.total.toFixed(1) : "n/a"), g.market.total.toFixed(1)',
    '["Total", g.opening_total != null ? g.opening_total.toFixed(1) : (o ? o.total.toFixed(1) : "n/a"), g.market.total.toFixed(1)')

# ---- matchup ranks: bands are fifths of however many FBS teams there are ------------------
sub('const rankClass = r => r == null ? "q5" : r <= 6 ? "q1" : r <= 13 ? "q2" : r <= 19 ? "q3" : r <= 26 ? "q4" : "q5";',
    'let RANK_N = 136;\nconst rankClass = r => r == null ? "q5" : r <= RANK_N * 0.2 ? "q1" : r <= RANK_N * 0.4 ? "q2" : r <= RANK_N * 0.6 ? "q3" : r <= RANK_N * 0.8 ? "q4" : "q5";')
sub('''    const key = H("div", { class: "rank-key" }, document.createTextNode("League rank:"),
      ...[["q1", "1 to 6"], ["q2", "7 to 13"], ["q3", "14 to 19"], ["q4", "20 to 26"], ["q5", "27 to 32"]].map(([c, t]) => H("span", { class: "rk " + c, text: t })));''',
    '''    RANK_N = d.ranks.teams || 136;
    const cut = k => Math.floor(RANK_N * k);
    const key = H("div", { class: "rank-key" }, document.createTextNode("FBS rank:"),
      ...[["q1", "1 to " + cut(0.2)], ["q2", (cut(0.2) + 1) + " to " + cut(0.4)], ["q3", (cut(0.4) + 1) + " to " + cut(0.6)], ["q4", (cut(0.6) + 1) + " to " + cut(0.8)], ["q5", (cut(0.8) + 1) + " to " + RANK_N]].map(([c, t]) => H("span", { class: "rk " + c, text: t })));''')
sub("season to date. Each offense is set against the defense it faces. Rank 1 is best for that unit, so a defense ranked 1 in points allows the fewest.`",
    "season to date, ranked among FBS teams (games against FCS teams count). Each offense is set against the defense it faces. Rank 1 is best for that unit, so a defense ranked 1 in points allows the fewest; plays per game ranks the fastest first.`")
sub("`Each piece is worth this many points toward one team. Team ratings are points against an average team before the quarterback: ${g.home} ${snum(rt.home_rating)}, ${g.away} ${snum(rt.away_rating)}.`",
    "`Each piece is worth this many points toward one team. Team ratings are points against an average FBS team before the quarterback and preseason numbers: ${g.home} ${snum(rt.home_rating)}, ${g.away} ${snum(rt.away_rating)}. Talent is the 247Sports team talent composite; returning production is the share of last season's production back this year. Both matter most in the first weeks.`")

# ---- ratings ---------------------------------------------------------------------------
sub('"Rating is points per game better or worse than an average team on a neutral field, with this week\'s quarterback. It combines play-by-play efficiency, scoring and what earlier betting lines implied about each team."',
    '"Rating is points per game better or worse than an average FBS team on a neutral field, with this week\'s quarterback. It combines play-by-play efficiency, scoring and what earlier betting lines implied about each team. Hover a team on the chart, or a row in the table, to find it."')
sub('"aria-label": "Offensive and defensive efficiency for all 32 teams"', '"aria-label": "Offensive and defensive efficiency for every FBS team"')
sub('"font-size": on ? 24 : 17, "font-weight": 800,', '"font-size": on ? 22 : 11, "font-weight": on ? 800 : 700,')
sub("if (Math.abs(b.px - a.px) < 30 && Math.abs(dy) < 15) { const push = (15 - Math.abs(dy)) / 2 + 0.1",
    "if (Math.abs(b.px - a.px) < 24 && Math.abs(dy) < 10) { const push = (10 - Math.abs(dy)) / 2 + 0.1")
sub("const step = 0.05;", "const step = 0.1;")

# ---- model record ------------------------------------------------------------------------
sub("each projected using only earlier seasons and graded at the closing price. The model's settings were chosen on ${BT.first_season} to ${BT.holdout_from - 1}, so results since ${BT.holdout_from} are the cleaner test.`",
    "each projected using only earlier games and graded at −110 at the consensus closing line, rounded to the half point (the data has no historical prices). The model's settings and features were chosen on ${BT.tuning[0]} to ${BT.tuning[1]}; the shortened 2020 season is in neither set, so results since ${BT.holdout_from} are the cleaner test.`")
sub('tile("Average miss on the margin", A.margin_mae.model.toFixed(2) + " pts", "The closing spread missed by " + A.margin_mae.market.toFixed(2)),',
    'tile("Average miss on the margin", A.margin_mae.model.toFixed(2) + " pts", "The closing spread missed by " + A.margin_mae.market.toFixed(2) + ". From week 6 on the two are about even"),')
sub('''  table($("#records"), [{ t: "Market", l: 1 }''', '''  if (BT.by_week) {
    const host = $("#records").closest("section"), t = H("table");
    table(t, [{ t: "Weeks", l: 1 }, { t: "Games" }, { t: "Model miss" }, { t: "Adjusted miss" }, { t: "Line miss" }, { t: "Model picks winner" }, { t: "Line picks winner" }],
      BT.by_week.filter(w => w.n).map(w => [w.weeks, comma(w.n), w.margin_mae.model.toFixed(2), w.margin_mae.blend.toFixed(2), w.margin_mae.market.toFixed(2), pct(w.winner_pct.model), pct(w.winner_pct.market)]));
    host.append(H("div", { class: "section-head" }, H("h3", { text: "By part of the season, since " + BT.holdout_from }),
      H("p", { class: "note", text: "Early in the season the model leans on last year and on talent, and trails the line by about a point. From week 6 on it is level with the closing line on the margin and picks winners slightly better." })),
      H("div", { class: "scroll" }, t));
  }
  table($("#records"), [{ t: "Market", l: 1 }''')

# ---- wind ---------------------------------------------------------------------------------
sub("$(\"#wind-note\").textContent = `Unders at the closing total in outdoor games with ${W.band[0]} to ${W.band[1] - 1} mph of recorded wind went ${a.w}-${a.l}-${a.p} (${pct(a.win_pct)}), a return of ${spct(a.roi)} plus or minus ${pct(a.roi_se)}. The effect shows up from 2014 on and not before. The band was chosen after looking at every season, and live bets use a forecast rather than the wind measured at the game, so treat it as a lead to keep testing. Games flagged on the board carry a wind tag; wind is also a feature in the total model.`;",
    "$(\"#wind-note\").textContent = `Unders at the closing total in outdoor games with ${W.band[0]} to ${W.band[1] - 1} mph of recorded wind went ${a.w}-${a.l}-${a.p} (${pct(a.win_pct)}), a return of ${spct(a.roi)} plus or minus ${pct(a.roi_se)}. That is much weaker than in the NFL and within noise, so the college board does not tag wind games as a bet. Wind above 10 mph is a feature in the total model (it helped on the tuning seasons), and each outdoor game's forecast shows on its page.`;")
sub("[...W.eras.map(r => row({ ...r, label: `${W.band[0]}–${W.band[1] - 1} mph, ${r.label}` })), ...W.tiers.map(row)]",
    "[...W.eras.map(r => row({ ...r, label: `${W.band[0]}–${W.band[1] - 1} mph, ${r.label}` })), ...W.tiers.map(r => row({ ...r, label: r.label + \", all seasons\" }))]")

# ---- opening lines: spreads as on the NFL page, plus totals ------------------------------------
sub("`The same model, but with the bet placed at the opening spread instead of the closing one: ${comma(OP.games)} games from ${OP.first_season} to ${OP.last_season}. To keep the test fair the model is not told about quarterback changes announced during the week, and ${OP.left_out_monday} games where a team had played the previous Monday night are left out, because those openers were posted before that game.`",
    "`The same model, but with the bet placed at the opening line instead of the closing one: ${comma(OP.games)} games from ${OP.first_season} to ${OP.last_season} (opening lines start in 2021, so every one of these seasons is out of sample). To keep the test fair the model is not told about quarterback changes announced during the week or the wind, and ${OP.left_out_monday} games where a team had played in the previous five days are left out, because those openers were posted before that game.`")
sub("tile(\"Return on those bets\", spct(r2.roi), `Plus or minus ${pct(r2.roi_se)}. Since ${BT ? BT.holdout_from : 2019}: ${spct(r2.holdout.roi)} on ${r2.holdout.n} bets`),",
    "tile(\"Return on those bets\", spct(r2.roi), `Plus or minus ${pct(r2.roi_se)}, on ${comma(r2.n)} bets`),")
sub('''          <figure>
            <figcaption><h3 id="open-t-h">Betting the model's side at the opener</h3><p class="note" id="open-table-note"></p></figcaption>
            <div class="scroll"><table id="open-table" aria-labelledby="open-t-h"></table></div>
          </figure>
        </div>''', '''          <figure>
            <figcaption><h3 id="open-t-h">Betting the model's side at the opening spread</h3><p class="note" id="open-table-note"></p></figcaption>
            <div class="scroll"><table id="open-table" aria-labelledby="open-t-h"></table></div>
          </figure>
        </div>
        <figure>
          <figcaption><h3 id="open-tot-h">Betting the model's side at the opening total</h3><p class="note" id="open-tot-note"></p></figcaption>
          <div class="scroll"><table id="open-tot-table" aria-labelledby="open-tot-h"></table></div>
        </figure>''')
sub('''  const z = r2.roi / r2.roi_se;
  const fl = OP.flagged;''', '''  const z = r2.roi / r2.roi_se;
  const fl = OP.flagged;
  if (OP.total) {
    const T = OP.total, t2 = T.rows.find(r => r.threshold === 2), t3 = T.rows.find(r => r.threshold === 3);
    table($("#open-tot-table"), [{ t: "Bets placed", l: 1 }, { t: "Bets" }, { t: "Record" }, { t: "Win rate" }, { t: "Return" }, { t: "Toward / away" }, { t: "CLV" }, { t: "Same side at the close" }],
      T.rows.map(r => [r.rule, comma(r.n), `${r.w}-${r.l}-${r.p}`, { v: pct(r.win_pct), cls: "strong" }, spct(r.roi) + " \\u00b1" + pct(r.roi_se), pct(r.toward, 0) + " / " + pct(r.away, 0), snum(r.clv, 2), pct(r.close_win_pct)]));
    $("#open-tot-note").textContent = `The strongest result in the college backtest. When the model's total is 3 or more points off the opener, betting its side went ${t3.w}-${t3.l}-${t3.p} (${pct(t3.win_pct)}), a return of ${spct(t3.roi)} plus or minus ${pct(t3.roi_se)}, and the total later moved toward the model ${pct(t3.toward, 0)} of the time. That is about ${(t3.roi / t3.roi_se).toFixed(1)} standard errors above zero on seasons the model was not tuned on, so it is a lead worth following, not proof. The same sides at the close won ${pct(t3.close_win_pct)}. Graded at −110.`;
  }''')
sub("Graded at the opening price. The 2+ point row is ${z.toFixed(1)} standard errors above zero, so it is encouraging and not yet proof.",
    "Graded at −110 at the opening spread. The 2+ point row is ${z.toFixed(1)} standard errors from zero: for college sides the opener is not where the model does well, unlike the NFL.")

# ---- method and limits ----------------------------------------------------------------------
sub('''    ["Rate every team each week.", "Pass and rush efficiency (expected points added per play), success rate and scoring, solved for all teams at once so each number is adjusted for opponents. Recent games count more, last season counts less, and garbage time is dropped."],''',
    '''    ["Rate every team each week.", "Pass and rush efficiency (expected points added per play), success rate, scoring and tempo, solved for every FBS team at once (FCS opponents pooled as one team) so each number is adjusted for opponents. Recent games count more, last season counts less, and garbage time (a lead of 43, 37, 27 or 21 points by quarter) is dropped."],
    ["Add what is known before the season.", "Team talent (the 247Sports composite) and returning production, which matter most in the first weeks when the play-by-play has seen two or three games."],''')
sub('"Each quarterback carries his own rating. When this week\'s starter is not the one who produced the team\'s numbers, the projection moves."',
    '"Each quarterback carries his own rating. When this week\'s starter is not the one who produced the team\'s numbers, the projection moves. An unknown college backup is rated well below an NFL one."')
sub('"A regression fitted on earlier seasons combines the ratings with home field, rest and roof."',
    '"A regression fitted on earlier seasons combines the ratings with home field, rest, roof and, for totals, wind."')
sub('''    "It has not beaten closing lines. Against opening lines it has done better, so its value is early in the week, before the market catches up.",
    "Injuries other than a quarterback change are only seen through earlier lines.",
    "Weather is not modeled beyond whether the game is indoors.",
    "Lines come from one consensus source. Your own book's prices go in lines.csv and are priced against the same projections.",
    "Opening totals and moneylines are not available, so only spreads are tested against the opener.",''',
    '''    "It has not beaten closing lines. It trails them by about a point in the first five weeks and is level after that. Against opening totals it has done better; against opening spreads it has not.",
    "There is no official college injury report. Players listed Out or Doubtful on Covers.com get no projection (a starting quarterback is replaced by the next one in the game model); everyone else who has played lately is projected to play. Injured linemen and defenders are listed but change no numbers, and none of this can be tested on past seasons.",
    "Only games between two FBS teams are projected; games against FCS teams have no lines.",
    "Historical lines are consensus averages with no prices, so the record is graded at −110 and moneylines are not graded.",
    "Current prices come from ESPN's scoreboard, one book. Check your own book before betting.",''')

# ---- players ------------------------------------------------------------------------------
sub('''    All: [["pass_yds", "Pass yds", f0], ["rush_yds", "Rush yds", f0], ["receptions", "Rec", f1], ["rec_yds", "Rec yds", f0], ["td", "Anytime TD", null]],''',
    '''    All: [["pass_yds", "Pass yds", f0], ["rush_yds", "Rush yds", f0], ["receptions", "Rec", f1], ["rec_yds", "Rec yds", f0], ["td", "Anytime TD", null]],''')
sub('''    RB: [["carries", "Carries", f1], ["rush_yds", "Rush yds", f0], ["targets", "Targets", f1], ["receptions", "Rec", f1], ["rec_yds", "Rec yds", f0], ["td", "Anytime TD", null]],
    WR: [["targets", "Targets", f1], ["receptions", "Rec", f1], ["rec_yds", "Rec yds", f0], ["td", "Anytime TD", null]],''',
    '''    RB: [["carries", "Carries", f1], ["rush_yds", "Rush yds", f0], ["rush_td", "Rush TD", f1], ["receptions", "Rec", f1], ["rec_yds", "Rec yds", f0], ["td", "Anytime TD", null]],
    WR: [["receptions", "Rec", f1], ["rec_yds", "Rec yds", f0], ["rec_td", "Rec TD", f1], ["td", "Anytime TD", null]],''')
sub('''$("#players-note").textContent = "Projected box scores for every quarterback, running back, receiver and tight end with a role this week. Each projection is volume times efficiency: the team's expected passes and runs (from its pace, the opponent, the projected score and wind), the player's share of them (recent games weighted more, with injured teammates' usage passed on to those available), and his own efficiency, adjusted for what this defense allows. Select a player to see how his numbers are built.";''',
    '''$("#players-note").textContent = "Projected box scores for every quarterback, running back, receiver and tight end with a role this week, in official college stats (sacks count as quarterback runs). Each projection is volume times efficiency: the team's expected passes and runs (from its pace, the opponent, the projected score and wind), the player's share of them (recent games weighted more; players listed Out or Doubtful on Covers.com get no projection and their usage is passed on), and his own efficiency, adjusted for what this defense allows. Select a player to see how his numbers are built.";''')
sub('''    else grid.append(box("Share", tx("Projected "), b(pctx(w.tgt_share) + " of targets"), tx(" and "), b(pctx(w.car_share) + " of carries"),
      tx(`; ${pctx(w.tgt_share_hist)} and ${pctx(w.car_share_hist)} in his recent games` + (Math.abs((w.tgt_share || 0) - (w.tgt_share_hist || 0)) > 0.01 || Math.abs((w.car_share || 0) - (w.car_share_hist || 0)) > 0.02 ? ", adjusted for who is out." : "."))));
    if (p.pos !== "QB") grid.append(box("Efficiency", tx(`${f1(w.ypt)} yards per target and a ${pctx(w.catch)} catch rate` + ((w.car_share || 0) > 0.05 ? `; ${f1(w.ypc)} yards per carry` : "") + `, from ${f1(w.games)} weighted games, shrunk toward his position's average.`)));''',
    '''    else grid.append(box("Share", tx("Projected "), b(pctx(w.rec_share) + " of catches"), tx(" and "), b(pctx(w.car_share) + " of carries"),
      tx(`; ${pctx(w.rec_share_hist)} and ${pctx(w.car_share_hist)} in his recent games` + (Math.abs((w.rec_share || 0) - (w.rec_share_hist || 0)) > 0.01 || Math.abs((w.car_share || 0) - (w.car_share_hist || 0)) > 0.02 ? ", adjusted for who is out." : "."))));
    if (p.pos !== "QB") grid.append(box("Efficiency", tx(`${f1(w.ypr)} yards per catch` + ((w.car_share || 0) > 0.05 ? `; ${f1(w.ypc)} yards per carry` : "") + `, from ${f1(w.games)} weighted games, shrunk toward his position's average. Targets are not used: the college play-by-play is missing receivers on many incompletions.`)));''')
sub('''    if (p.pos === "QB" || (w.tgt_share || 0) > 0.05) d.push''', '''    if (p.pos === "QB" || (w.rec_share || 0) > 0.05) d.push''')
sub('''    if (t.opp_miss_db > 0.05 || t.opp_miss_pr > 0.05) dtext += ` Out for ${p.opp}: ${pctx(t.opp_miss_db)} of its usual defensive backs and ${pctx(t.opp_miss_pr)} of its pass rush.`;
''', "")
sub('''    if (t.out && t.out.length) grid.append(box("Teammates out", tx(t.out.map(o => `${o.name} (${o.pos}, ${o.status}${o.tgt_share >= 0.05 ? ", " + pctx(o.tgt_share) + " of targets" : ""}${o.car_share >= 0.1 ? ", " + pctx(o.car_share) + " of carries" : ""})`).join("; ") + ".")));''',
    '''    if (t.out && t.out.length) grid.append(box("Teammates out", tx(t.out.map(o => `${o.name} (${[o.pos, o.status, o.att_share >= 0.5 ? pctx(o.att_share) + " of passes" : "", o.rec_share >= 0.05 ? pctx(o.rec_share) + " of catches" : "", o.car_share >= 0.1 ? pctx(o.car_share) + " of carries" : ""].filter(Boolean).join(", ")})`).join("; ") + ". From Covers.com.")));''')
sub('''$("#players-foot").textContent = rows.length ? `${rows.length} players. Status comes from the weekly roster and injury report; game-day inactives are named about 90 minutes before kickoff, so check late news.` + (HAS_NEW ? NEW_NOTE : "") + dzFoot() : "No players match.";''',
    '''$("#players-foot").textContent = rows.length ? `${rows.length} players. College football has no official injury report: players listed Out or Doubtful on Covers.com are left out, everyone else who has played lately is projected, so check late news.` + (HAS_NEW ? NEW_NOTE : "") + dzFoot() : "No players match.";''')
sub("`Average miss per player-game since ${BTp.from}, scored only with what was known before each game.",
    "`Average miss per player-game since ${BTp.from} (the model also beat both averages on every stat in 2017 to 2019, the seasons it was set up on), scored only with what was known before each game, on players who played.")

# ---- past weeks -------------------------------------------------------------------------------
sub('  const LEAGUE_REC = "targets";', '  const LEAGUE_REC = "receptions";')
sub('''      seg($("#seg-hweek"), index.map(w => ({ v: w.file, t: "Week " + w.week })), () => cur, v => { cur = v; show(); });''',
    '''      seg($("#seg-hweek"), index.map(w => ({ v: w.file, t: "Week " + w.week + (w.backfilled ? "*" : "") })), () => cur, v => { cur = v; show(); });''')
sub('''      box(String(w.graded) + " of " + w.games, pend ? "games graded (" + pend + " to play)" : "games graded"));''',
    '''      box(String(w.graded) + " of " + w.games, pend ? "games graded (" + pend + " to play)" : "games graded"),
      ...(w.graded && w.winners ? [box(pr(w.winners), `Winners picked (the line ${pr(w.line_winners)})`),
        box(w.miss.margin.toFixed(1), `Average miss on the margin (line ${w.miss.line_margin.toFixed(1)})`),
        box(w.miss.total.toFixed(1), `Average miss on the total (line ${w.miss.line_total.toFixed(1)})`)] : []));
    const done = index.filter(x => x.season === w.season && x.graded);
    if (done.length > 1) {
      const sum = k => done.reduce((a, x) => a + (x[k] || 0), 0), n = sum("graded");
      const add = k => [0, 1, 2].map(i => done.reduce((a, x) => a + ((x[k] || [])[i] || 0), 0));
      const avg = k => done.reduce((a, x) => a + (x.miss ? x.miss[k] * x.graded : 0), 0) / n;
      $("#hist-rec").append(H("p", { class: "note", style: "flex-basis:100%;margin:4px 0 0", text: `Season so far (${n} games): board's side ${pr(add("spread"))} on spreads and ${pr(add("total"))} on totals; winners ${pr(add("winners"))} against the line's ${pr(add("line_winners"))}; average miss ${avg("margin").toFixed(2)} on the margin (line ${avg("line_margin").toFixed(2)}) and ${avg("total").toFixed(2)} on the total (line ${avg("line_total").toFixed(2)}).` }));
    }''')
sub('''    $("#hist-foot").textContent = `"Side" is whichever side the board's market-adjusted chance favours, with that chance; most are close to 50%, so a week's record is mostly noise. Flagged bets are the ones the board staked. Predictions frozen ${fz.length > 1 ? "between " + fz[0] + " and " + fz[fz.length - 1] : "at " + (fz[0] || "kickoff")}.`;''',
    '''    const replay = rows.some(g => g.backfilled);
    $("#hist-foot").textContent = `"Side" is whichever side the board's market-adjusted chance favours, with that chance; most are close to 50%, so a week's record is mostly noise. Flagged bets are the ones the board staked. ` + (replay
      ? "* Weeks marked with a star are a replay: the board was not live yet, so these are the same model's walk-forward predictions, made only from earlier games and priced at the closing consensus line."
      : `Predictions frozen ${fz.length > 1 ? "between " + fz[0] + " and " + fz[fz.length - 1] : "at " + (fz[0] || "kickoff")}.`);''')
sub('''    WR: [["targets", "Targets"], ["receptions", "Rec"], ["rec_yds", "Rec yds"]],''', '''    WR: [["receptions", "Rec"], ["rec_yds", "Rec yds"], ["rec_td", "Rec TD"]],''')
sub('"No player projections were archived for this week (they start with week 7)."', '"No player projections were archived for this week."')

# ---- player card: the last calibration step ----------------------------------------
# The chain (volume x share x efficiency x defense) is not the projection: a regression fitted
# on earlier seasons blends it with the player's recent average. Show both inputs and the gap.
CAL_BOX = r'''
    if (w.chain && w.recent) {
      const keys = (p.pos === "QB" ? [["pass_yds", "Passing yards"], ["rush_yds", "Rushing yards"]]
        : [["rush_yds", "Rushing yards"], ["rec_yds", "Receiving yards"]])
        .filter(([k]) => w.chain[k] != null && ((p.proj[k] || 0) >= 10 || (w.chain[k] || 0) >= 10));
      if (keys.length) {
        const kids = [tx("The chain is team volume × share × efficiency × defense. The projection comes from a regression on earlier seasons that blends the chain with his recent average.")];
        keys.forEach(([k, label]) => {
          const lift = (p.proj[k] || 0) - (w.chain[k] || 0);
          const rec = p.pos === "QB" && k === "pass_yds" ? "per start" : "per game";
          kids.push(H("br"), b(label + ": "), tx(`chain ${f1(w.chain[k])}, recent ${rec} ${f1(w.recent[k])}, projection ${f1(p.proj[k])} (${lift >= 0 ? "+" : "−"}${f1(Math.abs(lift))} from calibration).`));
        });
        grid.append(box("From chain to projection", ...kids));
      }
    }'''
sub('    grid.append(box("The defense", tx(dtext)));', '    grid.append(box("The defense", tx(dtext)));' + CAL_BOX)


# NFL-only parts of the page (the Props section, Oct 2026: props are NFL only, Ryan's call),
# marked in template.html and cut from the college page: (pattern, expected count).
NFL_ONLY = [(r"<!--NFL-ONLY-->.*?<!--/NFL-ONLY-->\n?", 2), (r"/\*NFL-ONLY\*/.*?/\*/NFL-ONLY\*/\n?", 5)]


def main():
    html = SRC.read_text()
    for pat, n in NFL_ONLY:
        html, c = re.subn(pat, "", html, flags=re.S)
        if c != n:
            raise SystemExit(f"make_cfb_template: expected {n} NFL-only blocks matching {pat}, found {c}")
    for old, new, n in P:
        c = html.count(old)
        if (n is None and c == 0) or (n is not None and c != n):
            raise SystemExit(f"make_cfb_template: expected {n or 'at least 1'} of this in template.html, found {c}:\n{old[:200]}")
        html = html.replace(old, new)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(P)} changes)")


if __name__ == "__main__":
    main()
