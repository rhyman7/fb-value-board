"""The Watchdog's rules (scripts/watchdog.py): when a skipped Action is started, and when it isn't."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location("watchdog", Path(__file__).resolve().parent.parent / "scripts" / "watchdog.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
UTC = timezone.utc


def run(i, created, status="completed", conclusion="success"):
    return {"id": i, "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": status, "conclusion": conclusion}


def test_market_slot_is_checked_from_40_minutes_to_3_hours_after():
    d = lambda h, m: datetime(2026, 11, 25, h, m, tzinfo=UTC)
    assert W.market_slot(d(14, 30)) is None                      # too soon: GitHub may still start it
    assert W.market_slot(d(14, 45)) == d(14, 5)
    assert W.market_slot(d(17, 0)) == d(14, 5)
    assert W.market_slot(d(17, 10)) is None
    assert W.market_slot(datetime(2026, 11, 26, 0, 30, tzinfo=UTC)) is None
    assert W.market_slot(datetime(2026, 11, 26, 2, 50, tzinfo=UTC)) == datetime(2026, 11, 26, 2, 5, tzinfo=UTC)
    assert W.market_slot(datetime(2026, 5, 10, 14, 50, tzinfo=UTC)) is None      # out of season


def test_availability_hours_match_the_old_tasks():
    et = lambda y, mo, d, h, m: datetime(y, mo, d, h, m, tzinfo=W.ET).astimezone(UTC)
    assert W.availability_window(et(2026, 11, 28, 9, 22))        # Saturday
    assert W.availability_window(et(2026, 11, 28, 21, 22))
    assert not W.availability_window(et(2026, 11, 28, 8, 0))
    assert not W.availability_window(et(2026, 11, 28, 22, 30))
    assert W.availability_window(et(2026, 11, 25, 15, 22))       # Wednesday
    assert W.availability_window(et(2026, 11, 24, 19, 22))       # Tuesday
    assert not W.availability_window(et(2026, 11, 25, 12, 0))
    assert not W.availability_window(et(2026, 11, 23, 16, 0))    # Monday
    assert not W.availability_window(et(2026, 11, 29, 16, 0))    # Sunday
    assert not W.availability_window(et(2026, 4, 4, 12, 0))      # out of season


def test_a_run_github_started_means_nothing_to_do():
    slot = datetime(2026, 11, 25, 14, 5, tzinfo=UTC)
    since = slot - timedelta(minutes=20)
    assert W.decide([run(1, slot + timedelta(minutes=9))], since)[0] is False
    assert W.decide([run(1, slot + timedelta(minutes=30), "in_progress", None)], since)[0] is False
    assert W.decide([run(1, slot - timedelta(hours=6), "queued", None)], since)[0] is False   # anything active


def test_a_skipped_slot_is_started_and_old_runs_do_not_count():
    slot = datetime(2026, 11, 25, 14, 5, tzinfo=UTC)
    start, why = W.decide([run(1, slot - timedelta(hours=12))], slot - timedelta(minutes=20))
    assert start and "did not start" in why


def test_a_failed_run_is_retried_once_then_left_alone():
    slot = datetime(2026, 11, 25, 14, 5, tzinfo=UTC)
    since = slot - timedelta(minutes=20)
    one = [run(2, slot + timedelta(minutes=5), conclusion="failure")]
    start, why = W.decide(one, since)
    assert start and "failure" in why
    two = [run(3, slot + timedelta(minutes=45), conclusion="startup_failure")] + one
    start, why = W.decide(two, since)
    assert not start and why.startswith("gave up")               # never a third: credits are not burned


def test_next_check_is_soon_in_report_hours_and_never_past_five_hours():
    sat = datetime(2026, 11, 28, 12, 0, tzinfo=W.ET).astimezone(UTC)
    assert W.next_sleep(sat, False) <= 21 * 60 + 60
    quiet = datetime(2026, 11, 30, 3, 0, tzinfo=UTC)             # Sunday night ET, after the 02:05 check
    assert W.MIN_SLEEP <= W.next_sleep(quiet, False) <= W.MAX_SLEEP
    just_before = datetime(2026, 11, 30, 14, 40, tzinfo=UTC)     # 5 minutes before the 14:45 check
    assert W.next_sleep(just_before, False) <= 6 * 60 + 60
    assert W.next_sleep(quiet, True) <= 21 * 60 + 60             # after starting a run, look again soon
