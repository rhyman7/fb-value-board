"""The bet sheet (nflmodel/sheet.py), alternate lines and teasers (alts.py), the ledger
(ledger.py) and the haircut fail-safe (pipeline.haircut_factor)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflmodel import alts as ALT, bets as B, ledger as LG, sheet as SH
from nflmodel import config as C
from nflmodel.pipeline import haircut_factor

ROOT = Path(__file__).resolve().parent.parent


def _game(**over):
    g = {"id": "2026_08_BAL_BUF", "week": 8, "away": "BAL", "home": "BUF", "anchor": "pinnacle", "roof": "outdoors",
         "market": {"spread": 3.5, "total": 54.0}, "model": {"margin": 9.6, "total": 49.1},
         "open": {"seen": "2026-10-20", "spread": 2.5, "total": 53.5, "moves": 3}, "opening_spread": 3.5,
         "sharp_open": {"seen": "2026-10-30 02:41", "spread": 3.5, "total": 54.0},
         "wind": {"mph": 14.0, "under_angle": True},
         "detail": {"breakdown": {"margin": [{"key": "ratings", "label": "Team ratings", "value": 3.8}, {"key": "home_qb", "label": "BUF quarterback", "value": 0.1},
                                             {"key": "away_qb", "label": "BAL quarterback", "value": 2.2}, {"key": "hfa", "label": "Home field", "value": 2.1}],
                                  "total": [{"key": "base", "label": "League baseline", "value": 47.4}, {"key": "wind", "label": "Wind", "value": -2.2}]}},
         "bets": [{"market": "Spread", "side": "home", "pick": "BUF -3.5", "odds": 108, "ev": 0.095, "best": "fanduel", "best_ev": 0.08, "beats_sharp": None, "stake": 0.0,
                   "books": [{"book": "fanduel", "line": -3.0, "odds": -112, "odds_net": -112, "ev": 0.08, "stake": 0.0}]},
                  {"market": "Spread", "side": "away", "pick": "BAL +3.5", "odds": -122, "ev": -0.14, "stake": 0.0},
                  {"market": "Moneyline", "side": "home", "pick": "BUF ML", "odds": -175, "ev": 0.065, "stake": 0.01, "best": "betmgm", "best_ev": 0.07, "beats_sharp": None,
                   "books": [{"book": "betmgm", "line": None, "odds": -165, "odds_net": -165, "ev": 0.07, "stake": 0.012}]},
                  {"market": "Moneyline", "side": "away", "pick": "BAL ML", "odds": 153, "ev": -0.18, "stake": 0.0, "beats_sharp": "kalshi", "sharp_ev": 0.02,
                   "best": "kalshi", "best_ev": -0.1, "books": [{"book": "kalshi", "line": None, "odds": 156, "odds_net": 150, "ev": -0.1, "ev_line": 0.02, "stake": 0.0}]},
                  {"market": "Total", "side": "over", "pick": "Over 54", "odds": -112, "ev": -0.04, "stake": 0.0},
                  {"market": "Total", "side": "under", "pick": "Under 54", "odds": -104, "ev": -0.02, "stake": 0.0}]}
    g.update(over)
    return g


def test_sheet_tiers_and_opener_source():
    g = _game()
    SH.annotate(g, "nfl")
    by = {b["pick"]: b for b in g["bets"]}
    # the data's opener (nfelo) is the one the bucket was backtested on, so it comes first
    assert by["BUF -3.5"]["opener_src"] == "data" and by["BUF -3.5"]["opener_gap"] == pytest.approx(6.1)
    assert by["BUF -3.5"]["gap"] == pytest.approx(6.1) and by["BUF -3.5"]["move"] == pytest.approx(0.0)
    assert by["BUF -3.5"]["tier"] == "opener" and by["BAL +3.5"]["tier"] is None
    assert by["BUF ML"]["tier"] is None                       # a +EV moneyline is not a bet unless the price beats the sharp line
    assert by["BAL ML"]["tier"] == "price"
    g5 = _game(); g5["bets"][3]["sharp_ev"] = 0.004; SH.annotate(g5, "nfl")
    assert {b["pick"]: b for b in g5["bets"]}["BAL ML"]["tier"] is None    # under 1% over the sharp line is not a bet
    assert by["Under 54"]["tier"] == "wind" and by["Over 54"]["tier"] is None
    assert [w["label"] for w in by["BUF -3.5"]["why"][:2]] == ["Team ratings", "BAL quarterback"] and by["BUF -3.5"]["why"][1]["news"]
    assert all(w["label"] != "League baseline" for w in by["Under 54"]["why"])
    # college: no wind bucket, and spreads are never the opener bucket (they lost on the holdout seasons)
    g2 = _game(); SH.annotate(g2, "cfb")
    by2 = {b["pick"]: b for b in g2["bets"]}
    assert by2["Under 54"]["tier"] is None and by2["BUF -3.5"]["tier"] == "model"
    # without the data's opener: the sharp opener when the log caught the open, else the board's first line
    g4 = _game(opening_spread=None, sharp_open={"seen": "2026-10-19 23:00", "spread": 2.0, "total": 53.0}); SH.annotate(g4, "nfl")
    assert {b["pick"]: b for b in g4["bets"]}["BUF -3.5"]["opener_src"] == "sharp"
    g6 = _game(opening_spread=None); SH.annotate(g6, "nfl")
    assert {b["pick"]: b for b in g6["bets"]}["BUF -3.5"]["opener_src"] == "board"


def test_opener_tier_judged_at_the_current_line():
    # opened 3.5, model 6.0: 2.5 off. At 4.5 (a point toward, 1.5 left) the market has taken it: model tier, missed
    g = _game(model={"margin": 6.0, "total": 49.1}, market={"spread": 4.5, "total": 54.0}); SH.annotate(g, "nfl")
    b = {x["pick"]: x for x in g["bets"]}["BUF -3.5"]
    assert b["move"] == pytest.approx(1.0) and b["tier"] == "model" and b["missed"]
    # half a point toward still counts as the opener
    g = _game(model={"margin": 6.0, "total": 49.1}, market={"spread": 4.0, "total": 54.0}); SH.annotate(g, "nfl")
    b = {x["pick"]: x for x in g["bets"]}["BUF -3.5"]
    assert b["tier"] == "opener" and not b["missed"]
    # a big gap that is still 2+ at the current line stays in the bucket after a move
    g = _game(model={"margin": 9.0, "total": 49.1}, market={"spread": 5.0, "total": 54.0}); SH.annotate(g, "nfl")
    assert {x["pick"]: x for x in g["bets"]}["BUF -3.5"]["tier"] == "opener"


def test_wind_under_drops_once_the_total_has_moved():
    g = _game(market={"spread": 3.5, "total": 52.0}, sharp_open={"seen": "2026-10-19 23:00", "spread": 3.5, "total": 54.0}); SH.annotate(g, "nfl")
    b = {x["pick"]: x for x in g["bets"]}["Under 54"]
    assert b["move"] == pytest.approx(2.0) and b["tier"] != "wind" and b["missed"]


def test_bucket_stakes_size_opener_and_wind_only():
    g = _game(); SH.annotate(g, "nfl")
    openers = {"rows": [{"threshold": 2, "win_pct": 0.545, "roi": 0.04, "holdout": {"n": 615, "win_pct": 0.546, "roi": 0.041}}]}
    wind = {"eras": [{"label": "2014-2018", "win_pct": 0.571}, {"label": "2019-", "win_pct": 0.592}]}
    n = SH.bucket_stakes([g], openers, wind, 0.25, 0.03)
    by = {b["pick"]: b for b in g["bets"]}
    assert by["BUF -3.5"]["bucket_book"] == "fanduel"
    assert n == 2 and by["BUF -3.5"]["bucket_stake"] > 0 and by["Under 54"]["bucket_stake"] > 0
    assert by["BUF ML"]["bucket_stake"] == 0 and by["BAL ML"]["bucket_stake"] == 0
    assert by["Under 54"]["bucket_stake"] > by["BUF -3.5"]["bucket_stake"]      # the stronger bucket, similar prices
    assert by["BUF -3.5"]["bucket_stake"] <= 0.03
    # a bucket that lost on the holdout seasons sizes nothing
    assert SH.bucket_stakes([_game_annotated()], {"rows": [{"threshold": 2, "win_pct": 0.55, "roi": 0.03, "holdout": {"n": 50, "win_pct": 0.5, "roi": -0.02}}]}, None, 0.25, 0.03) == 0


def _game_annotated():
    g = _game(); SH.annotate(g, "nfl"); return g


def test_haircut_fails_closed(monkeypatch):
    monkeypatch.setattr(C, "EDGE_HAIRCUT", True)
    assert haircut_factor(None, verbose=False) == 0.0
    assert haircut_factor({"factor": None}, verbose=False) == 0.0
    assert haircut_factor({"factor": 0.52}, verbose=False) == 0.52
    monkeypatch.setattr(C, "EDGE_HAIRCUT", False)
    assert haircut_factor(None, verbose=False) == 1.0


def _dist():
    return {"margin": np.ones(len(B.D.MARGINS)), "total": np.ones(len(B.D.TOTALS)), "sd_margin": 13.0, "sd_total": 13.5, "sd_win": 12.0, "n": 1}


def test_alts_and_teasers():
    df = pd.DataFrame({"game_id": ["g"], "season": [2026], "week": [8], "home_team": ["BUF"], "away_team": ["BAL"],
                       "spread_line": [3.5], "total_line": [54.0], "home_spread_odds": [-110.0], "away_spread_odds": [-110.0],
                       "over_odds": [-110.0], "under_odds": [-110.0], "home_moneyline": [-175.0], "away_moneyline": [150.0],
                       "proj_margin": [5.2], "proj_total": [53.0], "model_margin": [9.6], "model_total": [49.0]})
    out = ALT.price(df, _dist())["g"]
    sp = out["alts"]["spread"]
    assert sp[0]["line"] == -3.5 and sp[-1]["line"] == 10.5 and len(sp) == 29
    homes = [x["home"] for x in sp]
    assert all(a >= b - 1e-9 for a, b in zip(homes, homes[1:]))        # home covers less often as it lays more
    at = next(x for x in sp if x["line"] == 3.5)
    assert abs(at["home"] - 0.5) < 0.01 and at["home_proj"] > at["home"]  # anchored at the market; the projection likes the home side
    t6 = out["teasers"]["6"]
    assert t6["home"] > at["home"] and t6["away"] > at["away"] and out["teasers"]["7"]["home"] > t6["home"]
    assert t6["over"] > 0.5 and t6["under"] > 0.5
    assert ALT.wong(8.0, "home") and ALT.wong(2.0, "away") and not ALT.wong(3.5, "home") and not ALT.wong(8.0, "away")


def test_ledger_records_once_and_grades(tmp_path):
    g = _game_annotated()
    kick = {"2026_08_BAL_BUF": pd.Timestamp("2026-11-01T18:00Z")}
    path, res = tmp_path / "ledger.csv", tmp_path / "ledger_results.csv"
    now = pd.Timestamp("2026-10-30T09:00Z").to_pydatetime()
    assert LG.record([g], 2026, kick, path, now=now) == 3                  # the opener bet, the price bet and the wind under
    assert LG.record([g], 2026, kick, path, now=now) == 0                  # never twice
    led = pd.read_csv(path)
    row = led[led.pick == "BUF -3.5"].iloc[0]
    assert row.book == "fanduel" and row.line == -3.0 and row.odds == -112 and row.tier == "opener"
    assert led[led.pick == "BAL ML"].iloc[0].book == "kalshi"
    assert LG.record([g], 2026, kick, path, now=pd.Timestamp("2026-11-01T19:00Z").to_pydatetime()) == 0   # kicked off
    # grade: BUF wins 27-20 (covers -3 at FanDuel); BAL ML loses
    scores = {"2026_08_BAL_BUF": (27.0, 20.0)}
    close = {"2026_08_BAL_BUF": {"spread_line": 4.0, "home_spread_odds": -110.0, "away_spread_odds": -110.0, "total_line": 54.0,
                                 "over_odds": -110.0, "under_odds": -110.0, "home_moneyline": -190.0, "away_moneyline": 160.0}}
    n = LG.grade(path, res, scores, close, lambda df: B.curves(df, _dist(), "line"))
    assert n == 3 and LG.grade(path, res, scores, close, None) == 0
    r = pd.read_csv(res).set_index("row_id")
    assert r.loc["2026_08_BAL_BUF|Spread|home", "result"] == "W" and r.loc["2026_08_BAL_BUF|Spread|home", "units"] == pytest.approx(100 / 112, abs=1e-3)
    assert r.loc["2026_08_BAL_BUF|Spread|home", "clv"] > 0                 # -3 taken, closed at 4: beat the close
    assert r.loc["2026_08_BAL_BUF|Moneyline|away", "result"] == "L"
    s = LG.summary(path, res)
    assert r.loc["2026_08_BAL_BUF|Total|under", "result"] == "W"           # 47 points, under 54
    assert s["all"]["graded"] == 3 and s["all"]["w"] == 2 and s["by_tier"]["opener"]["w"] == 1 and s["by_tier"]["price"]["l"] == 1
    assert s["rows"][0]["pick"] in ("BUF -3.5", "BAL ML", "Under 54") and s["since"] == "2026-10-30"
    assert LG.settle("Spread", "away", 3.0, 3.0, 50.0) == ("P", 0.0)
    assert LG.settle("Total", "under", 44.5, 3.0, 44.0) == ("W", 1.0)


def test_ledger_files_are_consistent():
    """The committed ledgers (both boards): one row per bet, results only for ledger rows."""
    for led, res in ((C.DATA / "ledger.csv", C.DATA / "ledger_results.csv"), (ROOT / "data" / "cfb" / "ledger.csv", ROOT / "data" / "cfb" / "ledger_results.csv")):
        if not led.exists():
            continue
        x = LG._read(led, LG.COLS)
        assert x["row_id"].is_unique and set(x["tier"].dropna()) <= set(SH.TIERS) | {""}
        if res.exists():
            r = LG._read(res, LG.RES_COLS)
            assert r["row_id"].is_unique and set(r["row_id"]) <= set(x["row_id"])


def test_board_data_carries_the_sheet():
    """Every bet in both boards' data.json has a tier field, and the games carry alts and teasers."""
    for f in (C.DOCS / "data.json", ROOT / "docs" / "cfb" / "data.json"):
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        games = d["slate"]["games"]
        if not games or "ledger" not in d:          # a build from before the sheet
            continue
        assert all("tier" in b and "why" in b for g in games for b in g["bets"])
        assert all(b["tier"] in (None, "price") for g in games for b in g["bets"] if b["market"] == "Moneyline")
        assert all("kickoff" in g and ("alts" in g) == ("teasers" in g) for g in games)
        assert "inputs" in d["meta"] and d["meta"]["inputs"].get("generated")
        if d["meta"].get("haircut_on") and not (d["meta"].get("haircut") or {}).get("factor"):
            assert d["meta"]["haircut_factor"] == 0.0                     # fails closed


def test_best_bets_card_is_wired():
    """Both templates carry the best-bets view, its print page style and the buttons; the PDF
    script parses and names both boards."""
    import ast
    for t in (ROOT / "site" / "template.html", ROOT / "site" / "cfb-template.html"):
        html = t.read_text()
        assert 'id="view-bestbets"' in html and 'id="bb-page"' in html and 'id="bb-doc"' in html
        assert 'href="best-bets.pdf"' in html and 'href="#bestbets"' in html
        assert 'h === "bestbets"' in html and "renderBestBets()" in html
        # the model's opinions that are not bets: their own document, button and PDF
        assert 'id="view-leans"' in html and 'href="model-leans.pdf"' in html and 'href="#leans"' in html and "renderLeans()" in html
    assert 'id="view-bestbets"' in (ROOT / "site" / "cfb-template.html").read_text()
    src = (ROOT / "scripts" / "build_pdf.py").read_text()
    ast.parse(src)
    assert '"nfl"' in src and '"cfb"' in src and "best-bets.pdf" in src and "model-leans.pdf" in src
    landing = (ROOT / "index.html").read_text()
    assert 'href="docs/best-bets.pdf"' in landing and 'href="docs/cfb/best-bets.pdf"' in landing
    assert 'href="docs/model-leans.pdf"' in landing and 'href="docs/cfb/model-leans.pdf"' in landing


def test_bucket_stake_picks_the_book_on_price_and_uses_the_close_rate_after_a_move():
    openers = {"rows": [{"threshold": 2, "win_pct": 0.545, "roi": 0.04, "holdout": {"n": 615, "win_pct": 0.546, "roi": 0.041, "close_win_pct": 0.518}}]}
    g = _game(); SH.annotate(g, "nfl")
    b = g["bets"][0]
    # Kalshi at plus money on a worse number (2.7% under Pinnacle) vs FanDuel at -112 (2% under): FanDuel, on price
    b["books"] = [{"book": "kalshi", "line": -3.5, "odds": 113, "odds_net": 105.4, "ev": 0.02, "ev_line": -0.027, "stake": 0.0},
                  {"book": "fanduel", "line": -3.0, "odds": -112, "odds_net": -112, "ev": 0.08, "ev_line": -0.02, "stake": 0.0}]
    SH.bucket_stakes([g], openers, None, 0.25, 0.03)
    assert b["bucket_book"] == "fanduel" and 0 < b["bucket_stake"] < 0.02
    # once the line has moved toward the side (within the half point that keeps it in the bucket), the close rate sizes nothing
    g = _game(market={"spread": 4.0, "total": 54.0}); SH.annotate(g, "nfl")
    SH.bucket_stakes([g], openers, None, 0.25, 0.03)
    assert g["bets"][0]["tier"] == "opener" and g["bets"][0]["bucket_stake"] == 0


def test_line_moves_status_teaser_and_bucket():
    openers = {"rows": [{"threshold": 2, "win_pct": 0.545, "roi": 0.04, "holdout": {"n": 615, "win_pct": 0.546, "roi": 0.041}}]}
    alts = {"spread": [{"line": float(x) / 2, "home": 0.5 + x / 100, "away": 0.5 - x / 100} for x in range(-30, 31)], "total": []}
    g = _game(alts=alts)
    rows = SH.line_moves([g], "nfl", openers)
    r = next(x for x in rows if x["market"] == "Spread")
    assert r["side"] == "home" and r["pick"] == "BUF -3.5" and r["status"] == "Live" and r["bucket"]["threshold"] == 2
    assert r["tease"]["line"] == pytest.approx(2.5) and not r["tease"]["wong"]
    # a fresher sharp line overrides the board's: BUF -9 has reached the model (9.6 vs 9.0 leaves 0.6, a 5.5 move)
    r = next(x for x in SH.line_moves([g], "nfl", openers, {g["id"]: {"spread": 9.0}}) if x["market"] == "Spread")
    assert r["fresh"] and r["status"] == "Partly gone" and r["closed"] == pytest.approx(0.9, abs=0.01)
    r = next(x for x in SH.line_moves([g], "nfl", openers, {g["id"]: {"spread": 10.0}}) if x["market"] == "Spread")
    assert r["status"] == "Missed"
    assert SH.move_status(3.0, 1.0, 2.0) == "Live" and SH.move_status(1.5, 0.5, 2.0) == "Live" and SH.move_status(1.5, 1.0, 2.0) == "Partly gone"
    # college spreads under 5 off the opener have no backtested edge
    cfb_op = {"rows": [{"threshold": 3, "win_pct": 0.519, "roi": -0.009, "holdout": {"n": 1639, "win_pct": 0.519, "roi": -0.009}},
                       {"threshold": 5, "win_pct": 0.543, "roi": 0.036, "holdout": {"n": 735, "win_pct": 0.543, "roi": 0.036}}]}
    assert SH.bucket_record(cfb_op, "Spread", 4.0) is None and SH.bucket_record(cfb_op, "Spread", 6.1)["threshold"] == 5


def test_cap_week_scales_a_week_together():
    g1, g2 = _game(), _game(id="2026_08_X_Y")
    for g in (g1, g2):
        SH.annotate(g, "nfl")
    openers = {"rows": [{"threshold": 2, "win_pct": 0.545, "roi": 0.04, "holdout": {"n": 615, "win_pct": 0.546, "roi": 0.041}}]}
    SH.bucket_stakes([g1, g2], openers, {"eras": [{"win_pct": 0.592}]}, 0.25, 0.03)
    before = sum(SH.taken_stake(b) for g in (g1, g2) for b in g["bets"])
    out = SH.cap_week([g1, g2], 0.03)
    after = sum(SH.taken_stake(b) for g in (g1, g2) for b in g["bets"])
    assert out["8"]["stake"] == pytest.approx(before, abs=1e-3) and out["8"]["scale"] < 1
    assert after == pytest.approx(0.03, abs=2e-3) and g1["week_scale"] == out["8"]["scale"]
    assert SH.cap_week([_game_annotated()], 1.0)["8"]["scale"] == 1.0


def test_ledger_marks_bets_withdrawn_before_kickoff(tmp_path):
    import datetime as dt
    path = tmp_path / "ledger.csv"
    g = _game(); SH.annotate(g, "nfl")
    ko = {g["id"]: pd.Timestamp("2026-11-01T18:00Z")}
    t0 = dt.datetime(2026, 10, 30, 12, tzinfo=dt.timezone.utc)
    assert LG.record([g], 2026, ko, path, now=t0) > 0
    # the board changes its mind: BUF -3.5 is no longer flagged
    g2 = _game(); SH.annotate(g2, "nfl"); g2["bets"][0]["tier"] = None
    LG.record([g2], 2026, ko, path, now=t0 + dt.timedelta(hours=20))
    st = pd.read_csv(LG.status_path(path), dtype={"row_id": str})
    assert list(st["on_sheet"]) == [0] and st["row_id"].iloc[0].endswith("|Spread|home")
    LG.record([g2], 2026, ko, path, now=t0 + dt.timedelta(hours=21))       # no change, no new row
    assert len(pd.read_csv(LG.status_path(path))) == 1
    s = LG.summary(path, tmp_path / "ledger_results.csv")
    assert s["withdrawn"]["n"] == 1 and s["kept"]["n"] == s["all"]["n"] - 1
    assert next(r for r in s["rows"] if r["row_id"].endswith("|Spread|home"))["kept"] is False
    # after kickoff nothing is tracked
    LG.record([g], 2026, ko, path, now=dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc))
    assert len(pd.read_csv(LG.status_path(path))) == 1


def test_kalshi_post_at_uses_the_haircut_chance():
    out = {("g", "Spread", "home"): [
        {"book": "kalshi", "odds": 100.0, "odds_net": 96.0, "prob": 0.60, "p_push": 0.0, "ev": 0.1, "p_post": 0.50},
        {"book": "draftkings", "odds": -110.0, "odds_net": -110.0, "prob": 0.60, "p_push": 0.0, "ev": 0.14, "p_post": 0.50}]}
    k = B.with_post_at(out)[("g", "Spread", "home")]
    k = next(q for q in k if q["book"] == "kalshi")
    assert k["post_at"] is not None and k["post_at"] <= 50          # against the 50% the stakes are sized from, not the model's 60%
