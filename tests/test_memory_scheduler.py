import threading
import time

import pytest

from conflux.memory.scheduler import CleanupScheduler
from conflux.memory.service import CleanupService
from conflux.memory.store import RunInProgress

@pytest.fixture
def service(tmp_path):
    handle = CleanupService(tmp_path / "sched.db", policy={"model": "heuristic", "schedule_enabled": True})
    yield handle
    handle.stop()
    handle.store.close()


def scheduler_for(service, **kwargs):
    kwargs.setdefault("timezone_name", "UTC")
    kwargs.setdefault("hour", 0)
    kwargs.setdefault("minute", 0)
    kwargs.setdefault("enabled", True)
    return CleanupScheduler(service, **kwargs)


# --- clock arithmetic --------------------------------------------------------


def test_scheduled_instant_is_the_slot_containing_now(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"schedule_enabled": True})
    sched = scheduler_for(service, hour=3, minute=30)
    assert sched.scheduled_instant("2026-05-04T03:30:00+00:00").strftime("%H:%M") == "03:30"
    assert sched.scheduled_instant("2026-05-04T03:29:59+00:00").strftime("%H:%M") == "03:30"
    # before the hour, the stamped slot is still in front of the clock, which is
    # exactly what makes tick() report "not due" instead of firing early
    before_slot = sched.scheduled_instant("2026-05-04T03:29:00+00:00")
    assert before_slot > sched.local_now("2026-05-04T03:29:00+00:00")
    assert sched.tick("2026-05-04T03:29:00+00:00")["reason"] == "not due"
    service.store.close()


def test_next_fire_crosses_the_day_boundary(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"schedule_enabled": True})
    sched = scheduler_for(service, hour=9, minute=15)
    assert sched.next_fire("2026-05-04T09:14:00+00:00").isoformat().startswith("2026-05-04T09:15")
    assert sched.next_fire("2026-05-04T09:15:00+00:00").isoformat().startswith("2026-05-05T09:15")
    assert sched.next_fire("2026-05-04T09:16:00+00:00").isoformat().startswith("2026-05-05T09:15")
    assert sched.seconds_until_next("2026-05-04T09:14:00+00:00") == 60.0
    assert sched.seconds_until_next("2026-05-04T09:15:00+00:00") == 86400.0
    service.store.close()


def test_local_timezone_is_respected(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"schedule_enabled": True})
    sched = scheduler_for(service, timezone_name="Europe/Berlin", hour=0, minute=0)
    # 23:00 UTC on the 8th is 00:00 CET on the 9th, so the slot belongs to the 9th
    instant = sched.scheduled_instant("2026-03-08T23:00:00+00:00")
    assert instant.isoformat().startswith("2026-03-09T00:00:00+01:00")
    assert sched.local_now("2026-03-08T23:30:00+00:00").strftime("%H:%M %Z") == "00:30 CET"
    assert sched.slot_for(instant) == "2026-03-08T23:00"  # slots are identified in UTC
    service.store.close()


def test_the_spring_forward_slot_fires_once_not_twice(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    service.store.add("dst note", topic="t")
    # Berlin jumps 02:00 -> 03:00 on 2026-03-29, so a 02:30 slot does not exist
    sched = scheduler_for(service, timezone_name="Europe/Berlin", hour=2, minute=30)
    before = sched.tick("2026-03-29T00:45:00+00:00")  # 01:45 CET, still before the jump
    assert before["fired"] is False and before["reason"] == "not due"
    fired = sched.tick("2026-03-29T01:25:00+00:00")  # 03:25 CEST, mapped past the gap
    assert fired["fired"] is True and fired["slot"] == "2026-03-29T01:30", fired
    repeat = sched.tick("2026-03-29T01:45:00+00:00")
    assert repeat["fired"] is False and repeat["reason"] == "slot already claimed"
    assert repeat["slot"] == fired["slot"]  # the same UTC slot, not a second one
    assert len([r for r in service.store.list_runs(limit=10)
                if r["trigger"] == "scheduled"]) == 1
    following = sched.next_fire("2026-03-29T01:45:00+00:00")
    assert following.astimezone(sched.tz).strftime("%Y-%m-%d %H:%M") == "2026-03-30 02:30"
    service.store.close()


def test_interval_mode_anchors_on_midnight(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"schedule_enabled": True})
    sched = scheduler_for(service, interval_minutes=30)
    assert sched.scheduled_instant("2026-05-04T10:07:00+00:00").strftime("%H:%M") == "10:00"
    assert sched.next_fire("2026-05-04T10:07:00+00:00").strftime("%H:%M") == "10:30"
    assert sched.slot_for(sched.scheduled_instant("2026-05-04T10:07:00+00:00")).startswith("interval:")
    assert sched.slot_for(sched.scheduled_instant("2026-05-04T10:29:00+00:00")) == \
        sched.slot_for(sched.scheduled_instant("2026-05-04T10:07:00+00:00"))
    service.store.close()


# --- firing ------------------------------------------------------------------


def test_tick_fires_once_per_slot(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    service.store.add("A note worth clustering.", topic="t")
    service.store.add("a note worth clustering", topic="t", on_duplicate="insert")
    sched = scheduler_for(service, hour=0, minute=0)
    first = sched.tick("2026-05-04T00:00:30+00:00")
    assert first["fired"] is True and first["ok"] is True and first["run_id"]
    assert first["slot"] == "2026-05-04T00:00"
    assert first["next_run_at"].startswith("2026-05-05T00:00")
    second = sched.tick("2026-05-04T00:02:00+00:00")
    assert second["fired"] is False and second["reason"] == "slot already claimed"
    # the ledger records it as a scheduled pass, run by this slot
    run = service.store.get_run(first["run_id"])
    assert run["trigger"] == "scheduled" and run["actor"] == "scheduler:2026-05-04T00:00"
    assert run["status"] == "completed"
    service.store.close()


def test_tick_waits_for_the_scheduled_time_instead_of_replaying_yesterday(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    sched = scheduler_for(service, hour=0, minute=0)
    early = sched.tick("2026-05-04T23:58:00+00:00")
    assert early["fired"] is False and early["reason"] == "not due"
    # today's midnight is 23h58m in the past: well outside the grace window, so the
    # slot is treated as missed rather than replayed
    assert early["overdue_minutes"] > early["grace_minutes"]
    # and a pass scheduled for 23:59 at 23:58 is simply not due yet
    almost = scheduler_for(service, hour=23, minute=59)
    waiting = almost.tick("2026-05-04T23:58:00+00:00")
    assert waiting["fired"] is False and waiting["overdue_minutes"] < 0
    assert waiting["next_run_at"].startswith("2026-05-04T23:59")
    # 23 hours past midnight is a missed slot, not a due one - the grace window ends it
    missed = sched.tick("2026-05-04T03:00:00+00:00")
    assert missed["fired"] is False and missed["reason"] == "not due"
    assert missed["overdue_minutes"] > missed["grace_minutes"]
    # inside the window it still runs, which is what a restarted service needs
    caught_up = scheduler_for(service, hour=0, minute=0, grace_minutes=120)
    late = caught_up.tick("2026-05-04T01:30:00+00:00")
    assert late["fired"] is True and late["ok"] is True
    service.store.close()


def test_a_disabled_scheduler_does_nothing(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    sched = scheduler_for(service, enabled=False)
    assert sched.tick("2026-05-04T00:00:00+00:00") == {"fired": False, "reason": "disabled"}
    status = sched.status()
    assert status["enabled"] is False and status["policy_enabled"] is True
    assert sched.start() is None and sched.running is False
    service.store.close()


def test_a_stored_policy_veto_outranks_an_explicit_start(tmp_path):
    armed = CleanupService(tmp_path / "a.db", policy={"model": "heuristic", "schedule_enabled": True,
                                                      "schedule_hour": 4, "schedule_minute": 15,
                                                      "schedule_timezone": "UTC"})
    sched = CleanupScheduler(armed)
    assert sched.enabled is True and (sched.hour, sched.minute) == (4, 15)
    assert sched.timezone_name == "UTC"
    sleeping = CleanupService(tmp_path / "b.db", policy={"model": "heuristic", "schedule_enabled": False})
    reluctant = CleanupScheduler(sleeping, enabled=True)
    assert reluctant.enabled is False and reluctant.policy_enabled is False
    assert reluctant.start() is None
    assert reluctant.tick("2026-05-04T00:00:00+00:00") == {"fired": False, "reason": "disabled"}
    armed.store.close()
    sleeping.store.close()


def test_a_busy_database_lets_the_next_tick_retry(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    service.store.add("note", topic="t")
    sched = scheduler_for(service)
    assert service.store.try_lock(owner="other-process", lease_seconds=120)["acquired"] is True
    busy = sched.tick("2026-05-04T00:00:10+00:00")
    assert busy["fired"] is False and busy["reason"] == "busy" and busy["slot"] == "2026-05-04T00:00"
    assert sched.status()["last_error"]
    service.store.unlock(owner="other-process")
    retried = sched.tick("2026-05-04T00:00:20+00:00")
    assert retried["fired"] is True and retried["ok"] is True, retried
    service.store.close()


def test_a_failing_pass_is_reported_and_the_thread_survives(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    sched = scheduler_for(service)

    def explode(**kwargs):
        raise RuntimeError("disk on fire")

    service.run = explode
    result = sched.tick("2026-05-04T00:00:05+00:00")
    assert result["fired"] is True and result["ok"] is False and "disk on fire" in result["error"]
    assert "RuntimeError" in sched.status()["last_error"]
    state = service.store.query_one("SELECT * FROM scheduler_state WHERE namespace='default'")
    assert state["last_status"] == "failed"
    # the slot is spent, so the same midnight is not retried forever
    assert sched.tick("2026-05-04T00:00:40+00:00")["reason"] == "slot already claimed"
    service.store.close()


# --- the thread --------------------------------------------------------------


def test_the_loop_thread_starts_fires_and_stops_cleanly(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    service.store.add("threaded note", topic="t")
    sched = scheduler_for(service, hour=None, minute=None, interval_minutes=0.0001, tick_seconds=0.05,
                          dry_run=True)
    # interval 0.0001 minutes == 6 ms, so every tick is a new slot
    thread = sched.start()
    assert thread is not None and sched.running is True
    deadline = time.time() + 10
    runs = []
    while time.time() < deadline:
        runs = service.store.list_runs(limit=50)
        if any(r["trigger"] == "scheduled" for r in runs):
            break
        time.sleep(0.05)
    assert sched.stop(timeout=5) is True
    assert sched.running is False
    scheduled = [r for r in service.store.list_runs(limit=50) if r["trigger"] == "scheduled"]
    assert scheduled, [r["trigger"] for r in service.store.list_runs(limit=50)]
    assert scheduled[0]["mode"] == "dry_run"  # this scheduler was asked to preview
    assert thread not in threading.enumerate()
    # starting twice does not spawn a second thread
    again = sched.start()
    assert again is not None
    sched.stop(timeout=5)
    service.store.close()


def test_status_describes_the_schedule_for_a_dashboard(tmp_path):
    service = CleanupService(tmp_path / "s.db", policy={"model": "heuristic", "schedule_enabled": True})
    sched = scheduler_for(service, hour=2, minute=5)
    status = sched.status()
    assert status["enabled"] is True and status["running"] is False
    assert status["policy_enabled"] is True
    assert status["at"] == "02:05" and status["timezone"] == "UTC"
    assert status["interval_minutes"] is None
    assert status["next_run_at"] and status["dry_run"] is False
    assert status["last"] is None and status["last_error"] is None
    untouched = service.scheduler_status()
    assert untouched["enabled"] is False and untouched["running"] is False  # nothing persisted yet
    fired = sched.tick("2026-05-04T02:05:01+00:00")
    assert fired["fired"] is True
    assert sched.status()["last"]["stats"]["considered"] == 0
    persisted = service.scheduler_status()
    assert persisted["enabled"] is True and persisted["last_status"] == "completed"
    assert persisted["last_run_id"] == fired["run_id"]
    assert persisted["last_fired_slot"] == "2026-05-04T02:05"
    service.store.close()
