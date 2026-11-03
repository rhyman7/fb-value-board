import datetime as dt

from nflmodel import kickoff as K

U = dt.timezone.utc


def t(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=U)


def test_windows_group_games_within_half_an_hour():
    ko = [t("2026-11-08 18:00"), t("2026-11-08 18:00"), t("2026-11-08 21:05"), t("2026-11-08 21:25"), t("2026-11-09 01:20")]
    assert K.windows(ko) == [t("2026-11-08 18:00"), t("2026-11-08 21:05"), t("2026-11-09 01:20")]


def test_plan_35_and_20_minutes_before_each_window_only_ahead_and_in_horizon():
    ko = {"nfl": [t("2026-11-08 18:00"), t("2026-11-08 21:05"), t("2026-11-08 21:25"), t("2026-11-09 01:20")]}
    p = K.plan(t("2026-11-08 17:30"), horizon_min=240, ko=ko)       # 17:25 already passed; 01:20 window beyond the horizon
    assert [(a, lg) for a, lg, *_ in p] == [(t("2026-11-08 17:40"), "nfl"), (t("2026-11-08 20:30"), "nfl"), (t("2026-11-08 20:45"), "nfl")]


def test_plan_merges_leagues_refreshing_together():
    ko = {"nfl": [t("2026-11-08 18:00")], "cfb": [t("2026-11-08 18:02"), t("2026-11-08 20:30")]}
    p = K.plan(t("2026-11-08 12:00"), horizon_min=600, ko=ko)
    assert [(a, lg) for a, lg, *_ in p] == [(t("2026-11-08 17:25"), "both"), (t("2026-11-08 17:40"), "both"),
                                           (t("2026-11-08 19:55"), "cfb"), (t("2026-11-08 20:10"), "cfb")]


def test_kickoffs_read_the_boards(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"slate": {"games": [{"kickoff": "2026-11-08T18:00Z"}, {"kickoff": "2026-11-08T18:00Z"}, {}]}}')
    assert K.kickoffs({"nfl": p, "cfb": tmp_path / "missing.json"}) == {"nfl": [t("2026-11-08 18:00")]}


def test_plan_marks_the_last_refresh_of_each_window_for_the_alert():
    ko = {"nfl": [t("2026-11-08 18:00")], "cfb": [t("2026-11-08 18:02")]}
    p = K.plan(t("2026-11-08 12:00"), horizon_min=600, ko=ko)
    assert [x[3] for x in p] == ["-", f"nfl@{int(t('2026-11-08 18:00').timestamp())},cfb@{int(t('2026-11-08 18:02').timestamp())}"]


CSV = """row_id,at,season,week,game_id,data_id,kickoff,market,side,pick,tier,book,line,odds,odds_net,ev,stake,anchor_spread,anchor_total,on_card
g1|Spread|away,2026-11-08 12:00,2026,10,2026_10_SEA_LV,g1,2026-11-08T18:00Z,Spread,away,SEA -6,price,fanduel,-5.5,-105,-105,0.0361,0.0051,-6,44.5,1
g1|Total|under,2026-11-08 12:00,2026,10,g1,g1,2026-11-08T18:00Z,Total,under,Under 44.5,price,draftkings,45.5,-110,-110,0.02,0.004,-6,44.5,1
g1|Total|under,2026-11-08 17:40,2026,10,g1,g1,2026-11-08T18:00Z,Total,under,Under 44.5,price,draftkings,44.5,-110,-110,0.0,0.0,-6,44.5,0
g2|Moneyline|home,2026-11-08 17:40,2026,10,g2,g2,2026-11-08T18:25Z,Moneyline,home,DAL ML,opener,kalshi,,108,101,0.03,0.006,-3,50,1
g3|Spread|home,2026-11-08 12:00,2026,10,g3,g3,2026-11-08T21:05Z,Spread,home,LV +6,price,fanduel,6.5,-110,-110,0.03,0.005,-6,44.5,1
"""


BOARD = {"slate": {"games": [
    {"id": "2026_10_SEA_LV", "bets": [{"market": "Spread", "side": "away", "tier": "price", "books": [
        {"book": "fanduel", "line": -5.5, "odds": -105, "ev": 0.0361, "ev_line": 0.0181, "stake": 0.0051},
        {"book": "draftkings", "line": -6, "odds": -110, "ev": -0.0259, "ev_line": -0.0435, "stake": 0}]}]},
    {"id": "g2", "bets": [{"market": "Moneyline", "side": "home", "tier": "opener", "books": [
        {"book": "kalshi", "line": None, "odds": 108, "ev": 0.03, "stake": 0.006},
        {"book": "draftkings", "line": None, "odds": -105, "ev": -0.01, "stake": 0}]}]},
]}}


def test_alert_shows_only_draftkings_and_fanduel_and_says_when_neither_qualifies(tmp_path):
    import json
    p, b = tmp_path / "bestbets.csv", tmp_path / "data.json"
    p.write_text(CSV)
    b.write_text(json.dumps(BOARD))
    title, body = K.message([("nfl", t("2026-11-08 18:00"))], {"nfl": p}, {"nfl": b})
    assert title == "Best bets: NFL Sun 1:00 PM ET (1 bet, 1 not at DK/FD)"   # the under left the card; g3 is a later window
    assert body.splitlines()[1:] == [
        "SEA@LV: SEA -5.5 -105 · FanDuel · stake 0.5% · EV +3.6%",
        "DAL ML · NOT A BET at DraftKings or FanDuel (not +EV; best there: DraftKings DAL ML -105, EV -1.0%; card has it at Kalshi)"]


def test_a_price_bet_needs_the_book_to_beat_the_sharp_number():
    b = {"tier": "price"}
    assert K.qualifies(b, {"ev": 0.03, "ev_line": 0.012}) and not K.qualifies(b, {"ev": 0.03, "ev_line": 0.005})
    assert K.qualifies({"tier": "opener"}, {"ev": 0.001}) and not K.qualifies({"tier": "opener"}, {"ev": -0.001})


def test_alert_sends_nothing_for_an_empty_card(tmp_path):
    p = tmp_path / "bestbets.csv"
    p.write_text(CSV)
    assert K.message([("nfl", t("2026-11-08 23:00"))], {"nfl": p}) is None
