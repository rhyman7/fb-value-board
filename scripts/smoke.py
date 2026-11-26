"""Browser smoke test: load this repo's main pages in headless Chromium and fail on a page
error, a missing script or an empty table.

    pip install playwright && python -m playwright install chromium
    python scripts/smoke.py            # reads smoke.json at the repo root
    python scripts/smoke.py -v         # also print what each page was checked for

How it works. Every page is opened at its live address (https://rhyman7.github.io/<base>/...),
and each request under this repo's base is answered from the files on disk, so the pages under
test are the ones in this checkout. Requests for the other repos' files on rhyman7.github.io
(the site bar, the value boards' feed) go to the live site. Every other host (ESPN, Open-Meteo,
Google Fonts) is refused, so a run never depends on a third party, and the pages must cope
without them as they are meant to.

A page fails when:
  - it throws an uncaught script error;
  - a script or stylesheet under this repo's base is missing;
  - a selector in its "need" list finds fewer elements than asked within the wait;
  - its visible text contains "undefined" or "NaN" (turn off per page with "allow_text": true).

smoke.json: {"base": "/nfl/", "pages": [{"path": "epa.html", "need": {"#epaLeague tbody tr": 30}}]}
  path           relative to the base; may carry ?query or #hash
  need           CSS selector -> fewest elements that must be there
  follow         {"link": selector, "need": {...}}: open the first matching link and check that page
  from_template  {"template": "site/template.html", "data": "docs/data.json"}: serve this page as
                 the template with the data inlined (the same replace the board build does), so a
                 template change is tested before the board is rebuilt
  optional       skip the page if its file is not in the checkout
  wait           seconds to wait for "need" (default 30)

This file is the same in rhyman7/nfl, rhyman7/cfb, rhyman7/fb-value-board and rhyman7.github.io;
change it in all four. Each repo's smoke.json is its own.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
SITE = "rhyman7.github.io"
BAD_TEXT = re.compile(r"\bundefined\b|\bNaN\b")


def main() -> int:
    from playwright.sync_api import sync_playwright

    verbose = "-v" in sys.argv[1:]
    cfg = json.loads((ROOT / "smoke.json").read_text())
    base = cfg["base"]
    templates = {p["path"].split("?")[0].split("#")[0]: p["from_template"] for p in cfg["pages"] if p.get("from_template")}

    def local(url_path: str) -> Path | None:
        """The file on disk for a URL path under this repo's base, else None."""
        if not url_path.startswith(base):
            return None
        rel = url_path[len(base):]
        if base == "/" and re.match(r"(nfl|cfb|nba|cbb|fb-value-board|prop-ladder)/", rel):
            return None                                  # other repos' pages: the live site
        f = ROOT / rel
        if rel == "" or rel.endswith("/") or f.is_dir():
            f = f / "index.html"
        return f

    def rel_of(f: Path) -> str:
        return f.relative_to(ROOT).as_posix()

    missing: list[str] = []

    def handle(route):
        u = urlparse(route.request.url)
        if u.netloc != SITE:
            return route.abort()
        f = local(u.path)
        if f is None:
            return route.continue_()
        t = templates.get(rel_of(f)) or templates.get(rel_of(f).removesuffix("index.html"))
        if t:
            data = json.loads((ROOT / t["data"]).read_text())
            blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
            html = (ROOT / t["template"]).read_text().replace("/*__DATA__*/null", blob)
            return route.fulfill(body=html, content_type="text/html; charset=utf-8")
        if f.is_file():
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith(("json", "javascript")):
                ctype += "; charset=utf-8"
            return route.fulfill(path=str(f), content_type=ctype)
        if f.suffix in (".js", ".css", ".html"):
            missing.append(u.path)
        return route.fulfill(status=404, body="not found")

    failures: list[str] = []
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(proxy={"server": proxy} if proxy else None)
        ctx = browser.new_context(ignore_https_errors=bool(proxy), viewport={"width": 1280, "height": 900},
                                  timezone_id="America/New_York", locale="en-US")
        ctx.route("**/*", handle)

        def check(page, label: str, spec: dict, errors: list[str]) -> None:
            wait = int(spec.get("wait", 30)) * 1000
            problems = []
            for sel, least in (spec.get("need") or {}).items():
                try:
                    page.wait_for_function("([s, n]) => document.querySelectorAll(s).length >= n", arg=[sel, least], timeout=wait)
                except Exception:
                    problems.append(f"{sel}: found {page.locator(sel).count()}, need {least}")
            page.wait_for_timeout(500)                    # let late script errors surface
            if not spec.get("allow_text"):
                m = BAD_TEXT.search(page.inner_text("body"))
                if m:
                    text = page.inner_text("body")
                    problems.append(f'the page shows "{m.group(0)}": …{text[max(0, m.start() - 60):m.end() + 40]}…'.replace("\n", " "))
            problems += [f"script error: {e}" for e in errors]
            problems += [f"missing file: {m}" for m in missing]
            errors.clear()
            missing.clear()
            if problems:
                failures.extend(f"{label}: {p}" for p in problems)
            need = ", ".join(f"{s} ≥ {n}" for s, n in (spec.get("need") or {}).items())
            print(f"{'FAIL' if problems else 'ok  '}  {label}" + (f"   [{need}]" if verbose else ""))
            for p in problems:
                print(f"        {p}")

        for spec in cfg["pages"]:
            path = spec["path"]
            f = local(base + path.split("?")[0].split("#")[0])
            if spec.get("optional") and not (f and f.is_file()):
                print(f"skip  {path} (not in this checkout)")
                continue
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e).splitlines()[0][:300]))
            try:
                page.goto(f"https://{SITE}{base}{path}", wait_until="load", timeout=60000)
                check(page, path or "(home)", spec, errors)
                fol = spec.get("follow")
                if fol:
                    href = page.locator(fol["link"]).first.get_attribute("href") if page.locator(fol["link"]).count() else None
                    if not href:
                        failures.append(f"{path}: no link matches {fol['link']}")
                        print(f"FAIL  {path}: no link matches {fol['link']}")
                    else:
                        page.goto(page.evaluate("h => new URL(h, location.href).href", href), wait_until="load", timeout=60000)
                        check(page, f"{path} → {href}", fol, errors)
            except Exception as e:                        # a page that never loads is a failure too
                msg = f"{path}: could not be loaded: {str(e).splitlines()[0][:200]}"
                failures.append(msg)
                print("FAIL  " + msg)
            page.close()
        browser.close()

    print(f"\n{len(failures)} problem(s)" if failures else "\nAll pages loaded.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
