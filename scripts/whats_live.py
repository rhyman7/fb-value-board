"""What's live: one page of every model change that is live but provisional.

    python scripts/whats_live.py          # rewrite docs/whats-live/index.html
    python scripts/whats_live.py --check  # exit 1 if the list and the code disagree

data/whats_live.json is the list: each provisional change, where its setting lives, and the rule
that decides it. This script reads every setting's CURRENT value from the code or data file (so
the page can't say 0.2 when the code says 0.3) and writes the page.

--check fails when:
  - a setting in the list can't be found in the code any more (renamed, removed, moved);
  - a file under nflmodel/ or cfbmodel/ says "provisional" and no row in the list names it
    (a provisional change was applied without adding it here);
  - docs/whats-live/index.html is not what this script would write now (a setting changed and
    the page was not regenerated).
The Smoke test Action runs --check on every pull request. It is deliberately NOT a pytest test:
the board builds run pytest, and a stale list must never stop a board from building.

Needs only the standard library: settings are read as text, nothing is imported.
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "data" / "whats_live.json"
PAGE = ROOT / "docs" / "whats-live" / "index.html"
REPO = "https://github.com/rhyman7/fb-value-board"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def value(s: dict):
    """A setting's current value, read from its file. None if it is not there."""
    f = ROOT / s["file"]
    if not f.is_file():
        return None
    text = f.read_text()
    if "json" in s:
        v = json.loads(text)
        for k in s["json"]:
            if not isinstance(v, dict) or k not in v:
                return None
            v = v[k]
        return v
    m = re.search(s["regex"], text, re.M | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def shown(s: dict, v) -> str:
    if v is None:
        return "NOT FOUND"
    if s.get("show") == "points":                    # a long table of pairs: say how many, not all of them
        return str(str(v).count("(")) + " points, from 0 to 1"
    return json.dumps(v) if not isinstance(v, str) else v


def day(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def problems(data: dict) -> list[str]:
    out = []
    named = set()
    for sec in ("applied", "waiting"):
        for e in data[sec]:
            named.update(e.get("code", []))
            for s in e["settings"]:
                named.add(s["file"])
                if value(s) is None:
                    out.append(f"{e['id']}: setting {s['label']} was not found in {s['file']}")
    for pkg in ("nflmodel", "cfbmodel"):
        for f in sorted((ROOT / pkg).glob("*.py")):
            rel = f.relative_to(ROOT).as_posix()
            if re.search(r"provisional", f.read_text(), re.I) and rel not in named:
                out.append(f'{rel} says "provisional" but no row in data/whats_live.json names it')
    return out


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def doc_link(d: str) -> str:
    """A repo document as a link; a Project doc (claude/...) as plain text, since it is private."""
    if d.startswith("claude/"):
        return f'<span class="doc">{esc(d)}</span>'
    name, _, section = d.partition(": ")
    label = f"{name}, “{section}”" if section else name
    return f'<a href="{REPO}/blob/main/{esc(name)}">{esc(label)}</a>'


def settings_html(e: dict) -> str:
    rows = "".join(f'<div><dt>{esc(s["label"])}</dt><dd>{esc(shown(s, value(s)))}</dd><dd class="where">{esc(s["file"])}</dd></div>'
                   for s in e["settings"])
    return f'<dl class="set">{rows}</dl>'


def render(data: dict) -> str:
    applied, waiting, groups = data["applied"], data["waiting"], data["groups"]
    rc = data["recheck"]
    order = {"NFL": 0, "College": 1, "Both": 2}
    cards = []
    for e in sorted(applied, key=lambda e: (order[e["league"]], e["since"])):
        prs = " ".join(f'<a href="{REPO}/pull/{n}">#{n}</a>' for n in e.get("prs", []))
        docs = "".join(f"<li>{doc_link(d)}</li>" for d in e.get("docs", []))
        cards.append(f"""
<article class="item" id="{esc(e['id'])}">
  <header><span class="chip {esc(e['league'].lower())}">{esc(e['league'])}</span><span class="area">{esc(e['area'])}</span>
    <h3>{esc(e['title'])}</h3><p class="since">Live since {day(e['since'])} · {esc(e['basis'])}</p></header>
  <p>{esc(e['what'])}</p>
  <div class="cols">
    <div><h4>Setting now</h4>{settings_html(e)}<p class="off"><strong>To turn it off:</strong> {esc(e['off'])}</p></div>
    <div><h4>What decides it</h4><p class="rule">{esc(e['rule'])}</p><p class="unseen"><strong>Judged on:</strong> {esc(e['unseen'])}</p></div>
  </div>
  <footer><ul>{docs}</ul>{f'<span class="prs">Pull requests {prs}</span>' if prs else ''}</footer>
</article>""")
    wait_rows = "".join(f"""<tr id="{esc(e['id'])}"><td class="l"><span class="chip {esc(e['league'].lower())}">{esc(e['league'])}</span></td>
<td class="l t"><strong>{esc(e['title'])}</strong></td><td class="l">{settings_html(e)}</td>
<td class="w">{esc(e['why'])}</td><td class="w">{esc(e['rule'])}</td></tr>""" for e in waiting)
    group_html = "".join(f"""<article class="item group" id="{esc(g['id'])}"><header><span class="chip both">{esc(g['league'])}</span>
<h3>{esc(g['title'])}</h3></header><p>{esc(g['what'])}</p><p class="rule"><strong>What decides them:</strong> {esc(g['rule'])}</p>
<p class="note">{esc(g['note'])}</p><footer><ul>{''.join(f'<li>{doc_link(d)}</li>' for d in g['docs'])}</ul></footer></article>""" for g in groups)
    by = {k: sum(1 for e in applied if e["league"] == k) for k in order}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex">
<link rel="icon" href="../../favicon.ico" sizes="32x32">
<title>What's live · Weekly Gridiron Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap">
<style>
/* GENERATED by scripts/whats_live.py from data/whats_live.json. Don't edit this file: edit the
   list or the script and run it. Same dark look as the rest of the site. */
:root {{
  --page: #131416; --surface: #1b1d21; --raised: #24272c; --ink: #f2f1ee; --ink-2: #9da2ab;
  --grid: #2a2e35; --axis: #4a505a; --rule: #353a42; --turf: #1b5e3a; --chalk: #f6f8f3;
  --flag: #fbbf24; --flag-ink: #241d00; --link: #f2f1ee;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --sans: "Archivo", "Helvetica Neue", Arial, system-ui, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --gutter: clamp(16px, 4vw, 40px);
  color-scheme: dark;
}}
h1, h2, .tile .value {{ font-stretch: 75%; }}
h1 {{ font-size: clamp(36px, 8vw, 64px); line-height: 0.95; }}
*, *::before, *::after {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink); font: 400 15px/1.5 var(--sans); font-variant-numeric: tabular-nums; -webkit-font-smoothing: antialiased; }}
h1, h2, h3, h4, p, ul, dl, dd {{ margin: 0; }}
a {{ color: var(--link); }}
.wrap {{ max-width: 1180px; margin-inline: auto; padding: 0 var(--gutter) 72px; display: grid; gap: 48px; }}
header.top {{ display: grid; gap: 14px; padding-top: 40px; }}
h1 {{ font: 800 clamp(36px, 10.5vw, 112px)/0.86 var(--display); letter-spacing: 0.005em; }}
h2 {{ font: 800 34px/1 var(--display); letter-spacing: 0.01em; }}
h3 {{ font-size: 19px; line-height: 1.25; font-weight: 750; }}
h4 {{ font-size: 12.5px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-2); margin-bottom: 8px; }}
.lede {{ font-size: 18px; line-height: 1.45; max-width: 66ch; color: var(--ink-2); }}
.note {{ font-size: 13.5px; line-height: 1.5; color: var(--ink-2); max-width: 80ch; }}
section {{ display: grid; gap: 18px; min-width: 0; }}
.tiles {{ display: flex; flex-wrap: wrap; border-top: 2px solid var(--ink); border-bottom: 1px solid var(--rule); }}
.tile {{ flex: 1 1 200px; padding: 14px 20px 16px 0; display: grid; gap: 2px; align-content: start; }}
.tile + .tile {{ padding-left: 20px; border-left: 1px solid var(--rule); }}
.tile .label {{ font-size: 13px; color: var(--ink-2); font-weight: 600; }}
.tile .value {{ font: 800 44px/1 var(--display); white-space: nowrap; }}
.tile .sub {{ font-size: 13px; color: var(--ink-2); }}
@media (max-width: 520px) {{ .tile + .tile {{ padding-left: 0; border-left: 0; border-top: 1px solid var(--rule); }} }}
.items {{ display: grid; gap: 14px; }}
.item {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 20px; display: grid; gap: 12px; }}
.item header {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; align-items: center; }}
.item header h3, .item header .since {{ grid-column: 1 / -1; }}
.area {{ font-size: 12.5px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-2); }}
.since {{ font-size: 13px; color: var(--ink-2); }}
.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 16px 32px; border-top: 1px solid var(--grid); padding-top: 12px; }}
.set {{ display: grid; gap: 6px; }}
.set div {{ display: grid; grid-template-columns: minmax(0, auto) 1fr; gap: 0 12px; align-items: baseline; }}
.set dt {{ font: 600 13px/1.4 var(--mono); overflow-wrap: anywhere; }}
.set dd {{ font: 700 13.5px/1.4 var(--mono); overflow-wrap: anywhere; }}
.set dd.where {{ grid-column: 1 / -1; font: 400 12px/1.4 var(--mono); color: var(--ink-2); }}
.off, .unseen {{ font-size: 13.5px; color: var(--ink-2); margin-top: 8px; }}
.rule {{ font-size: 14.5px; }}
.item footer {{ display: flex; flex-wrap: wrap; gap: 6px 24px; justify-content: space-between; border-top: 1px solid var(--grid); padding-top: 10px; font-size: 12.5px; color: var(--ink-2); }}
.item footer ul {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 4px 18px; }}
.doc {{ font-family: var(--mono); font-size: 12px; }}
.chip {{ display: inline-block; padding: 2px 9px; border-radius: 3px; font-weight: 800; font-size: 12px; letter-spacing: 0.02em; border: 1.5px solid var(--ink); justify-self: start; }}
.chip.nfl {{ background: var(--ink); color: var(--page); }}
.chip.college {{ background: var(--turf); border-color: var(--turf); color: var(--chalk); }}
.chip.both {{ background: var(--raised); }}
.item.group {{ border-style: dashed; }}
.scroll {{ overflow-x: auto; }}
/* Phones: each waiting change is a card. Labels are set here by column position, so a column added,
   removed or reordered in the table means updating them. */
@media screen and (max-width: 760px) {{
  table.m-cards, table.m-cards > tbody {{ display: block; }}
  table.m-cards > thead {{ display: none; }}
  table.m-cards > tbody > tr {{ display: grid; gap: 8px; margin-top: 10px; padding: 12px 14px; background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; }}
  table.m-cards > tbody > tr > td {{ display: block; min-width: 0; max-width: none; width: auto; padding: 0; border: 0; text-align: left; white-space: normal; }}
  table.m-cards > tbody > tr > td::before {{ display: block; font-size: 12px; line-height: 1.3; color: var(--ink-2); }}
  table.m-cards > tbody > tr > td:nth-child(2) {{ font-size: 16px; }}
  table.m-cards > tbody > tr > td:nth-child(3)::before {{ content: "Value now"; }}
  table.m-cards > tbody > tr > td:nth-child(4)::before {{ content: "Why it waits"; }}
  table.m-cards > tbody > tr > td:nth-child(5)::before {{ content: "What decides it"; }}
}}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ padding: 10px 12px 10px 0; text-align: left; vertical-align: top; border-bottom: 1px solid var(--grid); }}
th {{ font-weight: 600; color: var(--ink-2); font-size: 12.5px; border-bottom: 2px solid var(--ink); }}
td.t {{ min-width: 200px; }} td .set {{ min-width: 250px; }} td.w {{ min-width: 230px; color: var(--ink-2); font-size: 13.5px; }}
footer.page {{ font-size: 13px; color: var(--ink-2); display: grid; gap: 6px; }}
</style>
</head>
<body>
<script src="/site-bar.js" data-shell="2"></script>
<div class="wrap">
<header class="top">
  <h1>WHAT'S LIVE</h1>
  <p class="lede">Every change to the model that is live on the boards but still provisional: what it does, the setting it runs on today, and the rule that decides whether it stays.</p>
  <p class="note">The settings are read from the code each time this page is built, so they are what the boards are running. The rules were written before each test and are judged in the <strong>{esc(rc['task'])}</strong> run on {day(rc['date'])}.</p>
</header>
<section aria-label="Summary">
  <div class="tiles">
    <div class="tile"><span class="label">Provisional changes live</span><span class="value">{len(applied)}</span><span class="sub">{by['NFL']} NFL · {by['College']} college · {by['Both']} both boards</span></div>
    <div class="tile"><span class="label">Settings waiting on 2026</span><span class="value">{len(waiting)}</span><span class="sub">not changes: fitted values due a check</span></div>
    <div class="tile"><span class="label">Rechecked</span><span class="value">{day(rc['date']).split(',')[0]}</span><span class="sub">{day(rc['date']).split(', ')[1]}, once the 2026 season is in</span></div>
  </div>
</section>
<section id="applied">
  <h2>PROVISIONAL CHANGES</h2>
  <div class="items">{''.join(cards)}</div>
</section>
<section id="earlier">
  <h2>KEPT UNDER THE EARLIER RULE</h2>
  <div class="items">{group_html}</div>
</section>
<section id="waiting">
  <h2>SETTINGS WAITING ON A CHECK</h2>
  <p class="note">Fitted values the boards use that are due a check on 2026 data. Nothing here was a model change under the keep rule.</p>
  <div class="scroll"><table class="m-cards"><thead><tr><th></th><th>Setting</th><th>Value now</th><th>Why it waits</th><th>What decides it</th></tr></thead><tbody>{wait_rows}</tbody></table></div>
</section>
<footer class="page">
  <p>Built by <code>scripts/whats_live.py</code> from <code>data/whats_live.json</code>. Tests that were not kept, leads and forward-only records are not here: they are in the January recheck documents ({', '.join(esc(r) for r in rc['reads'])}).</p>
  <p>For information only. Nothing on this page is a bet.</p>
</footer>
</div>
</body>
</html>
"""


def main() -> int:
    data = json.loads(LIST.read_text())
    bad = problems(data)
    page = render(data)
    if "--check" in sys.argv[1:]:
        if not PAGE.is_file() or PAGE.read_text() != page:
            bad.append("docs/whats-live/index.html is out of date: run python scripts/whats_live.py and commit it")
        for b in bad:
            print("PROBLEM  " + b)
        print(f"{len(bad)} problem(s)" if bad else f"What's live: {len(data['applied'])} provisional changes, every setting found, page up to date.")
        return 1 if bad else 0
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page)
    print(f"Wrote {PAGE.relative_to(ROOT)}: {len(data['applied'])} provisional changes, {len(data['waiting'])} settings waiting.")
    for b in bad:
        print("PROBLEM  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
