#!/usr/bin/env python3
"""Re-shoot the Value Boards manual's screenshots (docs/manual/img/) from the built pages.

    python -m http.server 8770 &          # from the repo root
    python scripts/manual_shots.py [--base http://localhost:8770] [--only nfl_sheet_0,cfb_sheet_0]

One entry per image in SHOTS: the page, the view (hash), what to scroll to, and the crop. "view" is
the browser window (1280 wide) with the anchor at the top; "sect" is a 1116-wide crop that starts at
the anchor, as the fifth edition's were taken. Needs Playwright's Chromium. The manual's text and
captions are written by hand from what these show (docs/manual/index.html); after re-shooting,
update the captions and worked examples, then re-render the PDF (see CONVENTIONS.md, "The manual").
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "manual" / "img"
NFL, CFB, PAR = "/docs/index.html", "/docs/cfb/index.html", "/docs/parlays.html"
GAME = "__GAME__"          # replaced by the first game with an opener-tier bet (else the first game)
# name: (page, hash, anchor css or None for the top, mode, height, js to run first or None)
SHOTS = {
    "nfl_top": (NFL, "#games", None, "view", 900, None),
    "nfl_sheet_0": (NFL, "#games", "#sheet-h", "sect", 1500, None),
    "nfl_moves": (NFL, "#games", "#moves-h", "sect", 640, None),
    "nfl_boost": (NFL, "#games", "#boost-h", "sect", 265, None),
    "nfl_tease_0": (NFL, "#games", "#tease-h", "sect", 1370, None),
    "nfl_slate": (NFL, "#games", "#slate-h", "view", 819, None),
    "nfl_cards_0": (NFL, "#games", "#games-h", "sect", 620, None),
    "nfl_bets": (NFL, "#games", "#bets-h", "view", 818, None),
    "nfl_bestbets": (NFL, "#bestbets", None, "view", 1036, None),
    "nfl_leans": (NFL, "#leans", None, "view", 825, None),
    "nfl_game_0": (NFL, "#" + GAME, None, "view", 1147, None),
    "nfl_game_4": (NFL, "#" + GAME, "text=Every book", "view", 1147, None),
    "nfl_matchup": (NFL, "#matchups/" + GAME, None, "view", 900, None),
    "nfl_players_0": (NFL, "#players", "#players-h", "view", 1147, None),
    "nfl_player_card": (NFL, "#players", "#players-table tr.prow[aria-expanded=true]", "sect", 600,
                        "(() => { const rows = [...document.querySelectorAll('#players-table tr.prow')]; (rows.find(r => /RB/.test(r.innerText.split('\\n')[0])) || rows[0]).click(); })()"),
    "nfl_props_0": (NFL, "#props", "#props-h", "view", 1147, None),
    "nfl_history_0": (NFL, "#history", "#hist-h", "view", 1147, None),
    "nfl_ratings_0": (NFL, "#ratings", "#ratings-h", "view", 1147, None),
    "nfl_model_live": (NFL, "#model", "#live-h", "sect", 800, None),
    "nfl_model_bbrec": (NFL, "#model", "#bbrec-h", "sect", 660, None),
    "nfl_model_clv": (NFL, "#model", "#clvdash-h", "sect", 1100, None),
    "nfl_model_record_0": (NFL, "#model", "#record-h", "view", 1147, None),
    "nfl_model_openers": (NFL, "#model", "#open-h", "view", 830, None),
    "nfl_model_opentotals": (NFL, "#model", "text=Totals at the opener", "sect", 302, None),
    "cfb_sheet_0": (CFB, "#games", "#sheet-h", "sect", 1400, None),
    "cfb_moves_0": (CFB, "#games", "#moves-h", "sect", 1486, None),
    "parlays_boost": (PAR, "", None, "view", 1150,
                      "document.querySelector('[data-tab=boost]').click(); document.querySelector('#f-boost').value='30'; document.querySelector('#f-boost').dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('#go').click()"),
    "parlays_record": (PAR, "", "#record", "sect", 1077, "document.querySelector('[data-tab=record]').click()"),
}


async def main(base: str, only: set | None) -> None:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1280, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:160]))
        await pg.goto(base + NFL + "#games")
        await pg.wait_for_timeout(1200)
        game = await pg.evaluate("(() => { const gs = DATA.slate.games.filter(g => g.week === DATA.slate.weeks[0]); const o = gs.find(g => g.bets.some(b => b.tier === 'opener')); return (o || gs[0]).id; })()")
        for name, (page, hsh, anchor, mode, height, js) in SHOTS.items():
            if only and name not in only:
                continue
            await pg.set_viewport_size({"width": 1280, "height": height if mode == "view" else 1000})
            await pg.goto("about:blank")
            await pg.goto(base + page + hsh.replace(GAME, game))
            await pg.wait_for_timeout(1300)
            if js:
                await pg.evaluate(js)
                await pg.wait_for_timeout(900)
            y, x = 0.0, 0.0
            if anchor:
                el = pg.locator(anchor).first
                try:
                    await el.scroll_into_view_if_needed(timeout=4000)
                    box = await el.evaluate("e => { const r = e.getBoundingClientRect(); return {x: r.left + window.scrollX, y: r.top + window.scrollY}; }")
                    x, y = box["x"], max(box["y"] - 8, 0)
                except Exception as exc:
                    print(f"  {name}: anchor {anchor!r} not found ({str(exc)[:60]}); shot skipped")
                    continue
            total = await pg.evaluate("document.documentElement.scrollHeight")
            if mode == "view":
                clip = {"x": 0, "y": y, "width": 1280, "height": min(height, max(total - y, 200))}
            else:
                clip = {"x": max(x - 4, 0), "y": y, "width": 1116, "height": min(height, max(total - y, 120))}
            await pg.screenshot(path=str(OUT / f"{name}.jpg"), type="jpeg", quality=80, full_page=True, clip=clip)
            print(f"  {name}: {int(clip['width'])}x{int(clip['height'])}")
        await b.close()
        print("game used:", game, "| page errors:", errs or "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8770")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    asyncio.run(main(a.base.rstrip("/"), set(filter(None, a.only.split(","))) or None))
