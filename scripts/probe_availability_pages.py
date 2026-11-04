"""Render the four power conferences' football availability-report pages headlessly and save what
is there (text, links, images, iframes, PDFs, a screenshot), so the parsers for the College
availability Action can be written against the real pages. One-off: the **Probe availability
pages** Action uploads the output folder as a run artifact. Writes nothing to the repo.

    python scripts/probe_availability_pages.py out/

Depth 1: every link on a landing page whose text or address mentions report/availability is
rendered too (at most LIMIT per site).
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

PAGES = {
    "sec": "https://www.secsports.com/fbreports",
    "bigten": "https://bigten.org/FBReports",
    "acc": "https://theacc.co/fbreporting",
    "acc_policy": "https://theacc.com/sports/2025/8/28/availability-reporting.aspx",
    "big12": "https://big12sports.com/sports/2025/8/21/availability-reporting.aspx",
}
LIMIT = 4
KEY = re.compile(r"report|availab", re.I)


def dump(page, url, out: Path, name: str) -> list[str]:
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
    except Exception as exc:  # noqa: BLE001
        (out / f"{name}.error.txt").write_text(f"{url}\n{exc}")
        return []
    page.wait_for_timeout(2500)
    info = page.evaluate("""() => ({
        url: location.href, title: document.title,
        text: document.body ? document.body.innerText : "",
        links: [...document.querySelectorAll('a[href]')].map(a => ({text: a.innerText.trim().slice(0,120), href: a.href})),
        images: [...document.querySelectorAll('img')].map(i => ({alt: i.alt, src: i.currentSrc || i.src, w: i.naturalWidth, h: i.naturalHeight})),
        iframes: [...document.querySelectorAll('iframe')].map(f => f.src),
        tables: document.querySelectorAll('table').length,
    })""")
    (out / f"{name}.json").write_text(json.dumps(info, indent=1))
    (out / f"{name}.txt").write_text(info["text"])
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    print(f"{name}: {len(info['text'])} chars, {len(info['links'])} links, {len(info['images'])} images, {len(info['iframes'])} iframes, {info['tables']} tables")
    # the artifact may be unreachable from the workspace: say what matters in the log too
    print(f"  ::url {info['url']} | {info['title']}")
    for f in info["iframes"][:8]:
        print(f"  ::iframe {f}")
    for ln in info["links"]:
        if KEY.search(ln["text"]) or KEY.search(ln["href"]) or ln["href"].lower().endswith((".pdf", ".jpg", ".png")):
            print(f"  ::link {ln['text'][:60]!r} {ln['href']}")
    for im in info["images"]:
        if (im.get("w") or 0) >= 400:
            print(f"  ::image {im['alt'][:60]!r} {im['src'][:160]} {im['w']}x{im['h']}")
    t = re.sub(r"\s+", " ", info["text"])
    i = t.lower().find("availab")
    print("  ::text " + (t[max(0, i - 300): i + 1500] if i >= 0 else t[:1200]))
    picks = []
    for ln in info["links"]:
        href = ln["href"]
        if KEY.search(ln["text"]) or KEY.search(href) or href.lower().endswith(".pdf"):
            if href not in picks and href.rstrip("/") != info["url"].rstrip("/"):
                picks.append(href)
    return picks[:LIMIT]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "probe_out")
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 1000}, user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36")
        for name, url in PAGES.items():
            for i, sub in enumerate(dump(page, url, out, name)):
                dump(page, urljoin(url, sub), out, f"{name}_sub{i+1}")
        b.close()


if __name__ == "__main__":
    main()
