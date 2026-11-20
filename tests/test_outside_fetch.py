"""experiments/fetch_outside_ratings.py: the parsing and the guard against saving an error page.
No network: the fetch itself runs only in the Outside ratings Action."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fetch_outside_ratings", Path(__file__).resolve().parents[1] / "experiments" / "fetch_outside_ratings.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)


def test_archive_links_by_season():
    html = ('<h2>NCAA Prediction Archive</h2><b>2026 Season:</b> <a href="ncaa2026.csv">comma seperated file</a><br>'
            '<b>2025 Season:</b> <a href="/archive/ncaa2025.csv">comma seperated file</a> '
            '<b>2024 Season:</b> <a href="ncaa2024.html">page</a>')
    links = F.parse_archive(html)
    assert links == {2026: "https://www.thepredictiontracker.com/ncaa2026.csv",
                     2025: "https://www.thepredictiontracker.com/archive/ncaa2025.csv"}


def test_archive_falls_back_to_year_in_file_name():
    html = '<a href="ncaa2019.csv">2019</a> <a href=other.csv>x</a>'
    assert F.parse_archive(html) == {2019: "https://www.thepredictiontracker.com/ncaa2019.csv"}


def test_live_link():
    page = "https://www.thepredictiontracker.com/predncaa.html"
    assert F.parse_live('<a href="ncaapredictions.csv">Comma Separated File</a>', page) == \
        "https://www.thepredictiontracker.com/ncaapredictions.csv"
    assert F.parse_live("<p>no file</p>", page) is None


def test_only_a_real_predictions_file_is_saved():
    good = "home,road,lineopen,line,lineavg,linesag,linemassey\n" + "A,B,1,2,3,4,5\n" * 6
    assert F.looks_like_csv(good)
    assert not F.looks_like_csv("<html><body>Not found</body></html>")
    assert not F.looks_like_csv("home,road\nA,B\n")
    assert F.describe(good) == {"rows": 6, "columns": ["home", "road", "lineopen", "line", "lineavg", "linesag", "linemassey"]}


def test_standard_library_only():
    """The Action installs requirements.txt and nothing else (a missing package failed the first run)."""
    import ast
    import sys
    src = (Path(__file__).resolve().parents[1] / "experiments" / "fetch_outside_ratings.py").read_text()
    tops = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            tops |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            tops.add(node.module.split(".")[0])
    assert tops - {"__future__"} <= set(sys.stdlib_module_names)
