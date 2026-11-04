"""PFF data is licensed to Ryan's account and lives in the private repo rhyman7/pff-data. This repo
is public (its root is the live site), so no PFF rows may be committed here (CONVENTIONS.md)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_pff_data_committed():
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        return                                       # not a git checkout: nothing to check
    bad = [f for f in files if f.startswith((".pff-data/", "data/cfb/pff/", "docs/")) and "pff" in f.lower()
           and not f.endswith((".py", ".md"))]
    bad += [f for f in files if f.startswith(".pff-data/")]
    assert not bad, f"PFF data must not be committed to this public repo: {bad[:5]}"


def test_pff_checkout_ignored():
    assert ".pff-data/" in (ROOT / ".gitignore").read_text().split("\n")
