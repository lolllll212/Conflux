"""Nightly scheduler: fire a cleanup pass at midnight in a chosen timezone.

The loop is deliberately stupid — sleep, call :meth:`CleanupScheduler.tick`,
repeat — so all the interesting logic is reachable in tests without waiting or
sleeping: :meth:`next_fire` is pure arithmetic and :meth:`tick` takes a ``now``.

Two persistence details make it safe to run unattended:

* the slot it consumes (``scheduler_state.last_fired_slot``) is claimed in the
   database, so a restart inside the same minute cannot fire twice;
* a busy store is *not* a fired run: if :class:`RunInProgress` comes back, the
   slot is released and retried on the next tick, so a manual run at 00:00
   cannot silently eat the nightly pass.
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .store import RunInProgress, iso, parse_ts, utcnow

log = logging.getLogger("conflux.memory.scheduler")


class CleanupScheduler:
    """Threaded "every night at HH:MM in ``timezone_name``" trigger.

    ``enabled=False`` (or ``interval_minutes`` instead of hour/minute) is all it
    takes to switch it off for tests and development; nothing else in the
    package starts a thread.
    """

    def __init__(self, service, *, timezone_name=None, hour=None, minute=None,
                 interval_minutes=None, enabled=True, tick_seconds=5.0, dry_run=False,
                 namespace=None, logger=None, trigger="scheduled", grace_minutes=None):
        self.service = service
        policy = service.policy()
        self.namespace = namespace or service.namespace
        self.timezone_name = timezone_name or policy.schedule_timezone or "UTC"
        self.tz = ZoneInfo(self.timezone_name)
        self.hour = int(policy.schedule_hour if hour is None else hour)
        self.minute = int(policy.schedule_minute if minute is None else minute)
        self.interval_minutes = float(interval_minutes if interval_minutes is not None
                                       else policy.schedule_interval_minutes or 0.0)
        self.grace_minutes = float(grace_minutes if grace_minutes is not None
                                   else policy.schedule_grace_minutes)
        self.policy_enabled = bool(policy.schedule_enabled)
        self.enabled = bool(enabled and self.policy_enabled)
        self.tick_seconds = max(0.05, float(tick_seconds))
        self.dry_run = bool(dry_run)
        self.trigger = trigger
        self.log = logger or log
        self._stop = threading.Event()
        self._thread = None
        self._last = None
        self._last_error = None
        if policy.schedule_timezone and timezone_name is None:
            self.timezone_name = policy.schedule_timezone

    # -- clock arithmetic ----------------------------------------------------

    def local_now(self, now=None):
        reference = parse_ts(now) if now is not None else utcnow()
        return reference.astimezone(self.tz)

    def scheduled_instant(self, now=None):
        """Start of the slot that contains ``now`` (may lie in the future)."""
        local = self.local_now(now)
        if self.interval_minutes > 0:
            seconds = float(self.interval_minutes) * 60.0
            anchor = local.replace(hour=0, minute=0, second=0, microsecond=0)
            steps = int((local - anchor).total_seconds() // seconds)
            return anchor + timedelta(seconds=seconds * steps)
        return local.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)

    def next_fire(self, after=None):
        """Next fire instant (UTC) strictly after ``after``, for reporting.

        ``scheduled_instant`` returns the slot stamped on the same day, which is
        still ahead of the clock before that hour - so today's slot has to be
        offered before it is stepped forward, or a service asking at 09:14 is
        told the 09:15 pass is tomorrow.
        """
        local = self.local_now(after)
        current = self.scheduled_instant(local)
        step = timedelta(minutes=float(self.interval_minutes)) if self.interval_minutes > 0 \
            else timedelta(days=1)
        following = current if current > local else current + step
        while following <= local:
            following = following + step
        return following.astimezone(timezone.utc)

    def slot_for(self, local_instant):
        """Stable identity for one slot — what makes firing exactly-once work."""
        as_utc = local_instant.astimezone(timezone.utc)
        if self.interval_minutes > 0:
            return f"interval:{int(as_utc.timestamp() // (float(self.interval_minutes) * 60))}"
        return iso(as_utc)[:16]

    def seconds_until_next(self, now=None):
        reference = parse_ts(now) if now is not None else utcnow()
        return max(0.0, (self.next_fire(reference) - reference).total_seconds())

    # -- the loop ------------------------------------------------------------

    def tick(self, now=None):
        """Fire if the current slot is due. Returns a small report.

        Safe to call straight from a test: it never sleeps and starts no
        thread, and ``now`` lets a test stand in for midnight.
        """
        reference = parse_ts(now) if now is not None else utcnow()
        if not self.enabled:
            return {"fired": False, "reason": "disabled"}
        current = self.scheduled_instant(reference)
        overdue = (self.local_now(reference) - current).total_seconds() / 60.0
        # Grace window: a service started at 23:00 waits for midnight instead of
        # immediately replaying the slot it missed 23 hours ago.
        grace = float(self.interval_minutes) if self.interval_minutes > 0 else self.grace_minutes
        if overdue < 0 or overdue > grace:
            return {"fired": False, "reason": "not due", "overdue_minutes": round(overdue, 2),
                    "grace_minutes": grace, "next_run_at": iso(self.next_fire(reference))}
        slot = self.slot_for(current)
        next_at = iso(self.next_fire(reference))
        if not self.service.store.claim_slot(slot, namespace=self.namespace):
            return {"fired": False, "reason": "slot already claimed", "slot": slot,
                    "next_run_at": next_at}
        try:
            report = self.service.run(dry_run=self.dry_run, trigger=self.trigger,
                                      actor=f"scheduler:{slot}")
        except RunInProgress as exc:
            # Busy is not "done": give the slot back so the next tick retries.
            self.service.store.release_slot(slot, namespace=self.namespace)
            self._last_error = str(exc)
            self.log.info("scheduled pass skipped, another cleanup is running (%s)", exc)
            return {"fired": False, "reason": "busy", "slot": slot, "next_run_at": next_at}
        except Exception as exc:
            self._last_error = repr(exc)
            self.service.store.save_scheduler_state(namespace=self.namespace, last_run_at=iso(utcnow()),
                                                    last_status="failed", next_run_at=next_at)
            self.log.exception("scheduled cleanup failed")
            return {"fired": True, "ok": False, "error": repr(exc), "slot": slot}
        self._last = {"slot": slot, "run_id": report.run_id, "at": iso(utcnow()),
                      "stats": report.stats}
        self._last_error = None
        self.service.store.save_scheduler_state(
            namespace=self.namespace, enabled=True, timezone_name=self.timezone_name, hour=self.hour,
            minute=self.minute, last_run_id=report.run_id, last_run_at=iso(utcnow()),
            last_status="completed", next_run_at=next_at)
        self.log.info("scheduled cleanup %s: %s", report.run_id,
                      {k: report.stats.get(k) for k in
                       ("considered", "merges_proposed", "merges_auto", "prunes_proposed")})
        return {"fired": True, "ok": True, "run_id": report.run_id, "slot": slot,
                "next_run_at": next_at}

    def start(self):
        if not self.enabled:
            self.log.info("scheduler disabled%s; no background thread started",
                          " (policy schedule_enabled=False)" if not self.policy_enabled else "")
            return None
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="conflux-memory-scheduler",
                                        daemon=True)
        self._thread.start()
        self.service.store.save_scheduler_state(namespace=self.namespace, enabled=True,
                                                timezone_name=self.timezone_name, hour=self.hour,
                                                minute=self.minute, next_run_at=iso(self.next_fire()))
        self.log.info("scheduler armed for %s at %02d:%02d (tick %.1fs)", self.timezone_name,
                      self.hour, self.minute, self.tick_seconds)
        return self._thread

    def stop(self, timeout=5.0):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        return True

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # a scheduler thread must never die on a bad pass
                self.log.exception("scheduler tick failed")
            pause = min(self.tick_seconds, max(0.05, self.seconds_until_next()))
            if self._stop.wait(pause):
                return

    # -- reporting -----------------------------------------------------------

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        return {"enabled": self.enabled, "running": self.running, "timezone": self.timezone_name,
                "policy_enabled": self.policy_enabled,
                "at": f"{self.hour:02d}:{self.minute:02d}" if self.interval_minutes <= 0 else None,
                "interval_minutes": self.interval_minutes or None,
                "next_run_at": iso(self.next_fire()) if self.enabled else None,
                "tick_seconds": self.tick_seconds, "dry_run": self.dry_run,
                "last": self._last, "last_error": self._last_error}
