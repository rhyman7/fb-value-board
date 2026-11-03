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
    assert [(a, lg) for a, lg, _ in p] == [(t("2026-11-08 17:40"), "nfl"), (t("2026-11-08 20:30"), "nfl"), (t("2026-11-08 20:45"), "nfl")]


def test_plan_merges_leagues_refreshing_together():
    ko = {"nfl": [t("2026-11-08 18:00")], "cfb": [t("2026-11-08 18:02"), t("2026-11-08 20:30")]}
    p = K.plan(t("2026-11-08 12:00"), horizon_min=600, ko=ko)
    assert [(a, lg) for a, lg, _ in p] == [(t("2026-11-08 17:25"), "both"), (t("2026-11-08 17:40"), "both"),
                                           (t("2026-11-08 19:55"), "cfb"), (t("2026-11-08 20:10"), "cfb")]


def test_kickoffs_read_the_boards(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"slate": {"games": [{"kickoff": "2026-11-08T18:00Z"}, {"kickoff": "2026-11-08T18:00Z"}, {}]}}')
    assert K.kickoffs({"nfl": p, "cfb": tmp_path / "missing.json"}) == {"nfl": [t("2026-11-08 18:00")]}
