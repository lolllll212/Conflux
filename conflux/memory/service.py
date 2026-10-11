"""Service object that owns the store, the engine, the run lease and the schedule.

One instance per process, shared by the CLI, the FastAPI app and the background
scheduler. It exists so the three entry points cannot disagree about how a run
is started:

* an in-process lock keeps two threads of this process out of each other's
  transaction;
* the ``run_lock`` table keeps *other* processes out, with a lease, so a killed
  run cannot wedge the database forever;
* every run — scheduled or manual — goes through :meth:`run`, so the same
  guards, dry-run semantics and audit rows apply to both.
"""

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager

from .cleanup import CleanupEngine
from .model import build_model
from .policy import CleanupPolicy
from .store import MemoryStore, RunInProgress, StoreError, iso, utcnow

log = logging.getLogger("conflux.memory.service")

DEFAULT_LEASE_SECONDS = 1800.0


class CleanupService:
    """Facade over ``MemoryStore`` + ``CleanupEngine`` + schedule."""

    def __init__(self, db_path=None, *, store=None, namespace="default", policy=None, model=None,
                 lease_seconds=DEFAULT_LEASE_SECONDS, scheduler=None, logger=None,
                 backup_dir=None, durability="durable", read_only="auto"):
        if store is None and not db_path:
            raise ValueError("CleanupService needs a db_path or a store")
        # durability and the read-only fallback describe the *volume*, not the
        # cleanup rules, so they are constructor arguments and never policy keys
        self.store = store or MemoryStore(db_path, namespace=namespace, durability=durability,
                                          read_only=read_only)
        self.namespace = self.store.namespace
        self.backup_dir = backup_dir
        self.lease_seconds = float(lease_seconds)
        self.log = logger or log
        self._local = threading.RLock()
        self._active = None
        self._scheduler = scheduler
        self._policy_override = policy
        self._model_override = model
        if policy is not None:
            self._policy = policy if isinstance(policy, CleanupPolicy) else \
                CleanupPolicy.from_dict(policy, known_only=True)
        else:
            self._policy = self.store.get_policy(namespace=self.namespace)
        self._model = model

    # -- configuration -------------------------------------------------------

    def policy(self):
        """The live policy: stored per-namespace unless the caller pinned one."""
        if self._policy_override is not None:
            return self._policy
        return self.store.get_policy(namespace=self.namespace)

    def engine(self):
        policy = self.policy()
        model = self._model if self._model is not None else build_model(
            policy.model, dict(policy.model_options))
        return CleanupEngine(self.store, policy=policy, model=model, logger=self.log)

    def settings(self):
        policy = self.policy()
        return {"namespace": self.namespace, "policy": policy.to_dict(), "summary": policy.summary(),
                "stored": self.store.query_one("SELECT updated_at, updated_by FROM settings WHERE "
                                               "namespace=?", (self.namespace,)) or None,
                "model": self.describe_model()}

    def update_settings(self, updates, *, actor=None, replace_all=False):
        """Validate, persist and activate a policy change. Bad input raises."""
        base = self.policy()
        candidate = CleanupPolicy.from_dict(updates) if replace_all else base.merged(updates or {})
        candidate.validate()
        self.store.set_policy(candidate, namespace=self.namespace, updated_by=actor or "api")
        with self._local:
            self._policy = candidate
            self._policy_override = candidate
        self.log.info("policy updated by %s: %s", actor or "api", candidate.summary())
        return candidate

    def reset_settings(self, *, actor=None):
        default = CleanupPolicy()
        self.store.set_policy(default, namespace=self.namespace, updated_by=actor or "api")
        with self._local:
            self._policy = default
            self._policy_override = default
        return default

    def describe_model(self):
        policy = self.policy()
        try:
            model = self._model if self._model is not None else build_model(
                policy.model, dict(policy.model_options))
            return {"spec": policy.model, "name": model.name, "available": bool(model.available()),
                    "fallback": "heuristic merge (no model needed)"}
        except Exception as exc:
            return {"spec": policy.model, "name": "unavailable", "available": False,
                    "error": repr(exc), "fallback": "heuristic merge (no model needed)"}

    # -- runs ----------------------------------------------------------------

    @contextmanager
    def _exclusive(self, *, trigger, actor=None):
        """One run at a time, in-process and cross-process. Never queues.

        The in-process lock is taken non-blocking, so an overlapping trigger
        fails immediately instead of waiting to double-apply, and the database
        lease covers other processes (and heals if a holder is killed).
        """
        owner = uuid.uuid4().hex[:8]
        if not self._local.acquire(blocking=False):
            raise RunInProgress(f"run {self._active} is already in progress in this process")
        try:
            lease = self.store.try_lock(owner=owner, run_id=None, lease_seconds=self.lease_seconds,
                                        namespace=self.namespace)
            if not lease.get("acquired"):
                raise RunInProgress(f"cleanup already running: holder={lease.get('owner')}, "
                                    f"run={lease.get('run_id')}, lease until "
                                    f"{lease.get('lease_expires_at')}")
            self._active = f"{trigger}:{owner}"
            try:
                yield owner
            finally:
                self._active = None
                self.store.unlock(owner=owner, namespace=self.namespace)
        finally:
            self._local.release()

    def run(self, *, dry_run=True, trigger="manual", actor=None, force=False, limit=None):
        """One cleanup pass under an exclusive lease. Raises ``RunInProgress``."""
        with self._exclusive(trigger=trigger, actor=actor) as owner:
            report = self.engine().run(dry_run=dry_run, trigger=trigger, namespace=self.namespace,
                                       actor=actor or f"{trigger}:{owner}", force=force, limit=limit)
            if not dry_run:
                self.checkpoint()
            return report

    def checkpoint(self, *, truncate=True):
        """Fold the write-ahead log back into the main database file.

        WAL only grows while nothing checkpoints it, so a long-lived service that
        never checkpoints ends up with a -wal file larger than the database and a
        slow open after a crash. Best-effort by design: a busy log is not a reason
        to fail the pass that just succeeded.
        """
        try:
            busy, log_size, framed = self.store.checkpoint(truncate=truncate)
        except (StoreError, sqlite3.Error) as exc:
            self.log.warning("wal checkpoint skipped: %s", exc)
            return {"skipped": True, "reason": str(exc)}
        return {"skipped": False, "busy": bool(busy), "wal_pages": log_size, "framed": framed,
                "truncate": truncate}

    def start_run_async(self, *, dry_run=True, trigger="manual", actor=None, force=False):
        """Kick a pass off in a worker thread; the caller polls ``/runs``.

        A rejected trigger (busy) is reported through ``/status`` and the run
        ledger, not back to this caller — the response has already gone out.
        """
        state = {"accepted": True, "error": None, "done": threading.Event()}

        def worker():
            try:
                state["run_id"] = self.run(dry_run=dry_run, trigger=trigger, actor=actor,
                                           force=force).run_id
            except RunInProgress as exc:
                state["error"] = str(exc)
            except Exception as exc:
                self.log.exception("background cleanup failed")
                state["error"] = repr(exc)
            finally:
                state["done"].set()

        thread = threading.Thread(target=worker, name=f"conflux-memory-run-{uuid.uuid4().hex[:6]}",
                                  daemon=True)
        thread.start()
        state["thread"] = thread
        return state

    def apply_run(self, run_id, *, actor=None, force=False):
        """Execute the plan recorded by an earlier run (revalidated row by row)."""
        with self._exclusive(trigger="apply", actor=actor) as owner:
            return self.engine().apply_run(run_id, namespace=self.namespace,
                                          actor=actor or f"apply:{owner}", force=force)

    def review(self, review_id, decision, *, actor=None, note=None):
        """Approve (and execute) or reject one queued item."""
        if decision != "approved":
            result = self.store.resolve_review(review_id, "rejected", note=note, actor=actor or "api",
                                               namespace=self.namespace)
            if result.get("changed") and result.get("merge_id"):
                self.store.set_merge_state(result["merge_id"], "rejected",
                                          note=note or "rejected in review", namespace=self.namespace)
            return result
        with self._exclusive(trigger="review", actor=actor):
            return self.engine().approve_review(review_id, actor=actor or "api", note=note,
                                                namespace=self.namespace)

    # -- read-side convenience ----------------------------------------------

    def status(self):
        counts = self.store.counts(namespace=self.namespace)
        runs = self.store.list_runs(namespace=self.namespace, limit=5)
        lock = self.store.lock_info(self.namespace)
        return {"healthy": self.store.integrity_check() == ["ok"],
                "namespace": self.namespace, "database": self.store.info(),
                "counts": counts, "last_runs": runs,
                "running": bool(lock.get("held")), "lock": lock,
                "open_reviews": self.store.reviews(state="open", namespace=self.namespace, limit=20),
                "proposed_merges": self.store.merges(state=None, namespace=self.namespace, limit=20),
                "model": self.describe_model(), "policy": self.policy().summary(),
                "scheduler": self.scheduler_status(), "server_time": iso(utcnow())}

    def health(self):
        integrity = self.store.integrity_check()
        writable = True
        try:
            with self.store.transaction():
                self.store._execute("SELECT 1").fetchone()
        except Exception:
            writable = False
        scheduler = self.scheduler_status()
        info = self.store.info()
        counts = self.store.counts(namespace=self.namespace)
        wal = 0
        try:
            import os
            wal = os.path.getsize(self.store.path + "-wal") if os.path.exists(
                self.store.path + "-wal") else 0
        except OSError:
            pass
        degraded_reason = None
        if not writable:
            degraded_reason = (self.store.degraded or "the database rejected a test write")
        elif integrity != ["ok"]:
            degraded_reason = "integrity_check reported: " + "; ".join(integrity[:3])
        return {"status": "ok" if integrity == ["ok"] and writable else "degraded",
                "degraded_reason": degraded_reason,
                "read_only": bool(info.get("read_only")),
                "durability": info.get("durability"),
                "interrupted_runs": len(getattr(self.store, "interrupted_runs", ()) or ()),
                "bytes": info.get("bytes"), "wal_bytes": wal,
                "rows": counts.get("total"), "reviews_open": counts.get("open_reviews"),
                "merges_proposed": counts.get("proposed_merges"),
                "integrity": integrity, "writable": writable,
                "schema_version": self.store.schema_version,
                "latest_schema_version": info["latest_schema_version"],
                "schema_version_stale": info["schema_version"] != info["latest_schema_version"],
                "namespace": self.namespace, "server_time": iso(utcnow()),
                "running": bool(self.store.lock_info(self.namespace).get("held")),
                "scheduler": {k: scheduler.get(k) for k in
                             ("enabled", "timezone", "at", "next_run_at", "last_run_at", "last_status")}}

    def scheduler_status(self):
        stored = self.store.scheduler_state(self.namespace) or {}
        live = self._scheduler.status() if self._scheduler is not None else {}
        out = dict(stored)
        out.update(live)
        if not out:
            out = {"enabled": False, "running": False}
        out.setdefault("enabled", False)
        out.setdefault("running", self._scheduler is not None and self._scheduler.running)
        return out

    def backup(self, destination=None):
        import os
        from pathlib import Path
        stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
        if destination is None:
            root = Path(self.backup_dir or os.path.dirname(os.path.abspath(self.store.path)))
            root.mkdir(parents=True, exist_ok=True)
            destination = root / f"{Path(self.store.path).stem}-{stamp}.sqlite"
        self.store.vacuum_into(destination)
        return str(destination)

    # -- lifecycle -----------------------------------------------------------

    def start_scheduler(self, **kwargs):
        if self._scheduler is None:
            from .scheduler import CleanupScheduler
            self._scheduler = CleanupScheduler(self, **kwargs)
        if self._scheduler.enabled:
            self._scheduler.start()
        return self._scheduler

    def stop(self, *, timeout=10.0):
        if self._scheduler is not None:
            self._scheduler.stop(timeout=timeout)
        self.store.close()

    def close(self):
        # fold the WAL on the way out, so the next start reads one file instead of
        # replaying a log: the difference between a fast restart and a surprising one
        try:
            if not self.store.read_only:
                self.store.checkpoint(truncate=True)
        except (StoreError, sqlite3.Error) as exc:
            self.log.info("shutdown checkpoint skipped: %s", exc)
        self.stop()
