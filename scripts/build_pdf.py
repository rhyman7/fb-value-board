"""Render each board's best-bets card (#bestbets) to a PDF next to the board.

    python scripts/build_pdf.py            both boards
    python scripts/build_pdf.py nfl        one board (nfl or cfb)

Writes docs/best-bets.pdf and docs/cfb/best-bets.pdf from the generated pages, using
Playwright's Chromium (pip install playwright && playwright install chromium). The page
is opened from disk, so it needs no server; the props file is fetched relative to the page
and is skipped when the browser cannot read it from disk (the card says so). The Update
Actions run this after each build and commit the PDFs with docs/. A failure here never
stops a build: the script exits 0 and prints why.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARDS = {"nfl": ROOT / "docs", "cfb": ROOT / "docs" / "cfb"}
MARGIN = "0.45in"


def render(board: str, verbose: bool = True) -> Path | None:
    folder = BOARDS[board]
    page_file, out = folder / "index.html", folder / "best-bets.pdf"
    if not page_file.exists():
        print(f"  best-bets pdf {board}: no {page_file.relative_to(ROOT)}; skipped")
        return None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"  best-bets pdf {board}: playwright is not installed; skipped")
        return None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(page_file.resolve().as_uri() + "#bestbets", wait_until="load")
            page.wait_for_selector("#bb-doc h1", timeout=15000)
            page.wait_for_timeout(1500)                       # the props file, when the browser can read it
            page.emulate_media(media="print")
            page.pdf(path=str(out), format="Letter", print_background=True,
                     margin={"top": MARGIN, "bottom": MARGIN, "left": MARGIN, "right": MARGIN})
            browser.close()
    except Exception as exc:                                   # never fail the build over the PDF
        print(f"  best-bets pdf {board}: skipped ({type(exc).__name__}: {exc})")
        return None
    if verbose:
        print(f"  best-bets pdf {board}: wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    return out


if __name__ == "__main__":
    for b in (sys.argv[1:] or list(BOARDS)):
        render(b)
