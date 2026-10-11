"""FastAPI surface for the memory layer: inspect, trigger, review, configure.

Only the endpoints a local operator actually needs. Two invariants:

* the API never bypasses the service — every mutation goes through
  :meth:`CleanupService.run` / :meth:`apply_run` / :meth:`review`, so a manual
  trigger and the nightly scheduler are held to the same lease and the same
  safety floors;
* the *default* for a trigger is a dry run, and the default for a setting is the
  stored policy. Nothing here can delete anything: the destructive primitives
  (state transitions, hard delete) stay behind policy flags and review.

The module is imported lazily by the CLI, so ``import conflux.memory`` keeps
working with no FastAPI installed. Bind to loopback by default and, if you must
expose it further, set a bearer token with ``CONFLUX_MEMORY_TOKEN``.
"""

import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from .policy import CleanupPolicy
from .scheduler import CleanupScheduler
from .service import CleanupService
from .store import (STATES, MemoryNotFound, PolicyViolation, RunInProgress, StoreBusy,
                    StoreCorrupt, StoreError, StoreUnavailable, utcnow)

MODEL_DOC = ("model consolidation is optional: 'null' and 'heuristic' need no model at all, "
             "'ollama'/'local' talk to a loopback server, and a non-local endpoint is refused "
             "unless allow_remote is set - see docs/MEMORY.md")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")


class MemoryCreate(BaseModel):
    body: str
    kind: str = "note"
    topic: Optional[str] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}
    source_agent: Optional[str] = None
    uri: Optional[str] = None
    confidence: Optional[float] = None
    importance: float = 1.0
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    # A relative expiry, so a caller can say "valid for 30 days" without
    # computing a date. Mutually exclusive with expires_at.
    ttl_days: Optional[float] = None
    idempotency_key: Optional[str] = None
    on_duplicate: Literal["touch", "insert", "error"] = "touch"

    model_config = ConfigDict(extra="forbid")


class RunRequest(BaseModel):
    dry_run: bool = True
    force: bool = False
    wait: bool = True
    trigger: str = "manual"
    limit: Optional[int] = None
    namespace: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class ReviewDecision(BaseModel):
    decision: str
    note: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class SettingsUpdate(BaseModel):
    policy: Dict[str, Any] = {}
    replace: bool = False
    dry_run: bool = False

    model_config = ConfigDict(extra="forbid")


class MemoryBatch(BaseModel):
    """Up to 500 memories at once: one transaction, per-row receipts."""
    items: List[MemoryCreate] = Field(default_factory=list, max_length=500)
    namespace: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class EventCompaction(BaseModel):
    older_than_days: float = Field(default=90.0, gt=0)
    dry_run: bool = True
    model_config = ConfigDict(extra="forbid")


class SeenRequest(BaseModel):
    count: int = 1

    model_config = ConfigDict(extra="forbid")


MAX_BODY_BYTES = 1 << 20  # one MiB: a memory is a sentence, not a document


def create_app(service=None, *, db_path=None, namespace="default", policy=None, scheduler=None,
               start_scheduler=None, token=None, db_read_only_hint=False,
               max_body_bytes=MAX_BODY_BYTES, durability="durable"):
    """Build the ASGI app. ``service`` may be pre-built (tests embed one)."""
    owns_service = service is None
    if service is None:
        if not db_path:
            raise ValueError("create_app needs a service or a db_path")
        service = CleanupService(db_path, namespace=namespace, policy=policy,
                                 durability=durability)
    token = token if token is not None else os.environ.get("CONFLUX_MEMORY_TOKEN") or None

    if scheduler is None:
        live_policy = service.policy()
        scheduler = CleanupScheduler(
            service, enabled=start_scheduler if start_scheduler is not None
            else live_policy.schedule_enabled,
            timezone_name=live_policy.schedule_timezone, hour=live_policy.schedule_hour,
            minute=live_policy.schedule_minute,
            interval_minutes=live_policy.schedule_interval_minutes or None,
            dry_run=False)
    service._scheduler = scheduler

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if scheduler is not None and scheduler.enabled:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()
            if app.state.owns_service:
                service.store.close()

    app = FastAPI(
        title="Conflux self-pruning memory",
        version="0.1.0",
        description=("Local-first memory hygiene for AI agents: duplicate folding, semantic "
                     "grouping, staleness review. Nothing leaves this machine. " + MODEL_DOC),
        lifespan=lifespan,
    )
    # tunables that are about the service, not the store: what a request may carry,
    # and how many problems this process has admitted to
    app.state.max_body_bytes = int(max_body_bytes or 0)
    app.state.problem_count = 0
    app.state.service = service
    app.state.owns_service = owns_service
    app.state.scheduler = scheduler

    def guard(authorization: Optional[str] = Header(default=None)):
        if not token:
            return
        given = (authorization or "").strip()
        expected = f"Bearer {token}"
        # bytes, because a header is attacker-controlled text and compare_digest
        # raises TypeError on non-ASCII *strings* - which turns a bad token into a 500
        if not secrets.compare_digest(given.encode("utf-8", "replace"),
                                       expected.encode("utf-8")):
            # 401, not 403: the caller may retry with a real credential
            raise HTTPException(status_code=401, detail="missing or invalid bearer token",
                                headers={"WWW-Authenticate": "Bearer"})

    def problem(status_code, exc):
        # every 4xx/5xx this app produces, counted: "is anyone hitting the error
        # paths?" is the first question an operator asks of a service like this
        app.state.problem_count += 1
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.exception_handler(MemoryNotFound)
    async def _not_found(request, exc):
        return problem(404, exc)

    @app.exception_handler(RunInProgress)
    async def _busy(request, exc):
        return problem(409, exc)

    @app.exception_handler(PolicyViolation)
    async def _blocked(request, exc):
        return problem(409, exc)

    @app.exception_handler(ValueError)
    async def _bad_value(request, exc):
        return problem(400, exc)

    @app.exception_handler(StoreError)
    async def _bad_store(request, exc):
        return problem(400, exc)

    # A full disk, a read-only remount and a bad WHERE clause all arrive as
    # sqlite errors; telling them apart is what lets a client retry one and page a
    # human for the other. Starlette picks the most specific registered handler,
    # so these win over the generic StoreError case above.
    @app.exception_handler(StoreUnavailable)
    async def _store_unavailable(request, exc):
        response = problem(503, exc)
        response.headers["Retry-After"] = "5"
        response.headers["X-Conflux-Store"] = "unavailable"
        return response

    @app.exception_handler(StoreBusy)
    async def _store_busy(request, exc):
        response = problem(503, exc)
        response.headers["Retry-After"] = "1"
        response.headers["X-Conflux-Store"] = "busy"
        return response

    @app.exception_handler(StoreCorrupt)
    async def _store_corrupt(request, exc):
        response = problem(503, exc)
        response.headers["X-Conflux-Store"] = "corrupt"
        response.headers["Retry-After"] = "0"
        return response

    # -- liveness ------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    def health():
        return service.health()

    @app.middleware("http")
    async def _bound_request_size(request, call_next):
        """Refuse an oversized request before it is parsed, not after.

        The store already caps a body at 100k characters, but that check runs on the
        parsed value: a client (or a bug in one) that posts a 200 MB JSON document
        would otherwise be buffered and decoded first. 413 with a number in the
        message, because the fix is always "your payload is too big".
        """
        limit = app.state.max_body_bytes
        header = request.headers.get("content-length")
        if limit and header:
            try:
                declared = int(header)
            except ValueError:
                declared = 0
            if declared > limit:
                app.state.problem_count += 1
                return JSONResponse(status_code=413,
                                    content={"detail": f"request body is {declared} bytes; this "
                                                       f"service accepts at most {limit}",
                                              "max_body_bytes": limit})
        return await call_next(request)

    @app.get("/metrics", tags=["ops"])
    def metrics():
        """Prometheus text exposition of the handful of series worth alerting on.

        Deliberately unauthenticated, like /health: a scraper should not need the
        write token. The interesting signals are "can it be written", "is the audit
        trail growing without bound" and "when did a pass last finish".
        """
        import time as _time

        health = service.health()
        lines = []

        def add(name, value, kind="gauge", help_text=None, labels=None):
            if value is None:
                return
            # HELP/TYPE must name the metric exactly as it is exposed, prefix and
            # all, or a Prometheus parser drops the comment and the series type
            full = f"conflux_memory_{name}"
            if help_text:
                lines.append(f"# HELP {full} {help_text}")
                lines.append(f"# TYPE {full} {kind}")
            suffix = ("{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}"
                      if labels else "")
            lines.append(f"{full}{suffix} {value}")

        add("up", 1, help_text="Always 1 when the service answers.")
        add("healthy", 1 if health["status"] == "ok" else 0,
            help_text="0 means degraded: read the /health payload.")
        add("writable", 1 if health["writable"] else 0,
            help_text="0 means the volume rejected a test write (full or read-only).")
        add("read_only", 1 if health["read_only"] else 0)
        add("degraded", 1 if health["status"] != "ok" else 0)
        add("interrupted_runs", health["interrupted_runs"],
            help_text="Runs left 'running' by a process that died, reaped at open.")
        add("schema_version", health["schema_version"])
        add("rows", health["rows"])
        add("reviews_open", health["reviews_open"])
        add("merges_proposed", health["merges_proposed"])
        add("db_bytes", health["bytes"])
        add("wal_bytes", health["wal_bytes"], help_text="Write-ahead log size; checkpoint trims it.")
        events = service.store.query_one("SELECT COUNT(*) AS n FROM memory_events WHERE namespace=?",
                                         (service.namespace,))
        add("events", events["n"] if events else 0, help_text="Append-only audit rows.")
        add("run_lock_held", 1 if health["running"] else 0)
        add("http_problems_total", app.state.problem_count, kind="counter",
            help_text="4xx/5xx responses this process has produced.")
        scheduler = health["scheduler"] or {}
        add("scheduler_enabled", 1 if scheduler.get("enabled") else 0)
        last = scheduler.get("last_run_at")
        if last:
            from .store import parse_ts
            add("last_run_age_seconds", round(_time.time() - parse_ts(last).timestamp()))
        add("last_run_ok", 1 if scheduler.get("last_status") in ("completed", "ok") else 0)
        for state, value in service.store.counts(namespace=service.namespace).get(
                "by_state", {}).items():
            add("memories", value, labels={"state": state})
        return PlainTextResponse("\n".join(lines) + "\n",
                                media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/status", tags=["ops"])
    def status():
        return service.status()

    @app.get("/settings", tags=["config"], response_model=None)
    def get_settings():
        return service.settings()

    @app.get("/settings/schema", tags=["config"])
    def settings_schema():
        """Everything the policy understands: type, default, group and help."""
        defaults = CleanupPolicy()
        fields = defaults.schema()
        return {"fields": fields, "defaults": defaults.to_dict(), "summary": defaults.summary(),
                "model_options_hint": MODEL_DOC,
                "groups": sorted({v["group"] for v in fields.values()})}

    @app.put("/settings", tags=["config"], dependencies=[Depends(guard)])
    def put_settings(payload: SettingsUpdate):
        if payload.dry_run:
            # a preview must report the problem rather than raise it: that is the
            # whole point of asking "would this be accepted?"
            try:
                candidate = (CleanupPolicy.from_dict(payload.policy) if payload.replace
                             else service.policy().merged(payload.policy))
                candidate.validate()
            except ValueError as exc:
                return {"dry_run": True, "valid": False, "errors": [str(exc)],
                        "submitted": payload.policy}
            current = service.policy().to_dict()
            return {"dry_run": True, "valid": True,
                    "changes": {k: v for k, v in candidate.to_dict().items() if current.get(k) != v}}
        policy = service.update_settings(payload.policy, actor="api", replace_all=payload.replace)
        return {"updated": True, "policy": policy.to_dict(), "summary": policy.summary()}

    @app.post("/settings/reset", tags=["config"], dependencies=[Depends(guard)])
    def reset_settings():
        policy = service.reset_settings(actor="api")
        return {"reset": True, "policy": policy.to_dict()}

    # -- memories ------------------------------------------------------------

    @app.post("/memories", tags=["memories"], dependencies=[Depends(guard)])
    def add_memory(payload: MemoryCreate):
        # ttl_days / expires_at conflicts are the store's rule, so the CLI and
        # this route cannot disagree about what an expiry means.
        receipt = service.store.add(
            payload.body, kind=payload.kind, topic=payload.topic, namespace=payload.namespace,
            metadata=payload.metadata, provenance=payload.provenance,
            source_agent=payload.source_agent, uri=payload.uri, confidence=payload.confidence,
            importance=payload.importance, embedding=payload.embedding,
            embedding_model=payload.embedding_model, created_at=payload.created_at,
            expires_at=payload.expires_at, ttl_days=payload.ttl_days,
            idempotency_key=payload.idempotency_key,
            on_duplicate=payload.on_duplicate, actor="api")
        # 201 for a new row, 200 when the write folded into an existing one. The
        # receipt is spread at the top level so `id` and `created` are one hop away,
        # and the stored record rides along so a client never needs a second request.
        return JSONResponse(status_code=200 if not receipt["created"] else 201,
                            content={**receipt,
                                     "memory": service.store.get(receipt["id"],
                                                                 namespace=receipt["namespace"]).to_dict()})

    @app.post("/memories/batch", tags=["memories"], dependencies=[Depends(guard)])
    def add_batch(payload: MemoryBatch):
        """Many memories in one transaction.

        An agent that finished a session with 200 facts to record should not pay a
        durability round-trip per fact, and should not have to choose between "all or
        nothing" and "whatever the loop wrote before it crashed".
        """
        rows = [{"body": item.body,
                 **{k: v for k, v in item.model_dump().items()
                    if k not in ("body", "namespace") and v not in (None, {}, [])}}
                for item in payload.items]
        receipts = service.store.add_many(rows, namespace=payload.namespace, actor="api")
        return {"written": sum(1 for r in receipts if r["created"]),
                "folded": sum(1 for r in receipts if not r["created"]),
                "receipts": receipts}

    @app.get("/memories", tags=["memories"])
    def list_memories(state: str = "active", kind: Optional[str] = None, topic: Optional[str] = None,
                      limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0),
                      q: Optional[str] = None):
        """One page of memories, plus how big the whole filter is.

        Two traps this closes: an unknown ``state`` used to answer "no memories"
        (a typo indistinguishable from an empty store), and ``q`` used to be applied
        in Python *after* the limit, so ``?q=x&limit=5`` could report zero matches
        while the store held hundreds.
        """
        if state not in ("", "all") and state not in STATES:
            raise ValueError(f"unknown state {state!r}; use 'all' or one of "
                             f"{', '.join(sorted(STATES))}")
        wanted = {"state": None if state in ("", "all") else state, "kind": kind, "topic": topic,
                  "q": q or None, "namespace": service.namespace}
        rows = service.store.list(limit=limit, offset=offset, **wanted)
        total = service.store.count(**wanted)
        return {"count": len(rows), "total": total, "state": state, "offset": offset,
                "has_more": offset + len(rows) < total, "items": [m.to_dict() for m in rows]}

    @app.get("/memories/{memory_id}", tags=["memories"])
    def get_memory(memory_id: str):
        if not ID_RE.match(memory_id or ""):
            raise HTTPException(status_code=400, detail="invalid memory id")
        return service.store.get(memory_id).to_dict()

    @app.get("/memories/{memory_id}/events", tags=["memories"])
    def memory_events(memory_id: str, limit: int = Query(100, ge=1, le=1000)):
        return {"memory_id": memory_id, "events": service.store.timeline(memory_id, limit=limit)}

    @app.post("/memories/{memory_id}/seen", tags=["memories"], dependencies=[Depends(guard)])
    def mark_seen(memory_id: str, payload: SeenRequest):
        if service.store.get(memory_id) is None:  # raises MemoryNotFound -> 404
            raise MemoryNotFound(f"no memory {memory_id!r}")
        updated = service.store.mark_seen(memory_id, count=payload.count, actor="api")
        record = service.store.get(memory_id)
        return {"updated": updated, "access_count": record.access_count,
                "last_seen_at": record.last_seen_at}

    @app.post("/memories/{memory_id}/restore", tags=["memories"], dependencies=[Depends(guard)])
    def restore_memory(memory_id: str):
        record = service.store.get(memory_id)
        if record.state == "active":
            return {"restored": False, "reason": "already active"}
        ok = service.store.set_state(memory_id, "active", actor="api", action="restore",
                                      detail={"from": record.state})
        return {"restored": bool(ok), "state": "active" if ok else record.state}

    # -- runs ----------------------------------------------------------------

    @app.post("/runs", tags=["cleanup"], dependencies=[Depends(guard)])
    def trigger_run(payload: RunRequest):
        if not payload.wait:
            state = service.start_run_async(dry_run=payload.dry_run, trigger=payload.trigger,
                                            force=payload.force, actor="api")
            return {"accepted": True, "finished": state["done"].is_set(),
                    "error": state["error"], "hint": "poll /runs or /status"}
        report = service.run(dry_run=payload.dry_run, trigger=payload.trigger, force=payload.force,
                             limit=payload.limit, actor="api")
        return report.to_dict()

    @app.get("/runs", tags=["cleanup"])
    def list_runs(limit: int = Query(20, ge=1, le=200), status: Optional[str] = None):
        return {"items": service.store.list_runs(limit=limit, status=status)}

    @app.get("/runs/{run_id}", tags=["cleanup"])
    def get_run(run_id: str, include_plan: bool = True):
        run = service.store.get_run(run_id)
        if not include_plan:
            run.pop("plan", None)
        return run

    @app.get("/runs/{run_id}/items", tags=["cleanup"])
    def run_items(run_id: str):
        run = service.store.get_run(run_id)
        plan = run.get("plan") or {}
        return {"run_id": run_id, "status": run["status"], "mode": run["mode"],
                "merges": plan.get("merges", []), "prunes": plan.get("prunes", []),
                "reviews": plan.get("reviews", []), "stats": run.get("stats", {})}

    @app.post("/runs/{run_id}/apply", tags=["cleanup"], dependencies=[Depends(guard)])
    def apply_run(run_id: str):
        new_run, outcome = service.apply_run(run_id, actor="api")
        return {"run_id": new_run, "applied": outcome.to_dict()}

    # -- proposals and review ------------------------------------------------

    @app.get("/merges", tags=["review"])
    def list_merges(state: Optional[str] = "proposed", memory_id: Optional[str] = None,
                    limit: int = Query(50, ge=1, le=500)):
        wanted = None if state in ("", "all") else state
        return {"items": service.store.merges(state=wanted, memory_id=memory_id, limit=limit)}

    @app.get("/merges/{merge_id}", tags=["review"])
    def get_merge(merge_id: int):
        return service.store.get_merge(merge_id)

    @app.post("/merges/{merge_id}/revert", tags=["review"], dependencies=[Depends(guard)])
    def revert_merge(merge_id: int, note: Optional[str] = None):
        return service.store.revert_merge(merge_id, actor="api", note=note)

    @app.get("/reviews", tags=["review"])
    def list_reviews(state: str = "open", limit: int = Query(50, ge=1, le=500)):
        return {"items": service.store.reviews(state=state or None, limit=limit)}

    @app.post("/reviews/{review_id}", tags=["review"], dependencies=[Depends(guard)])
    def decide_review(review_id: int, payload: ReviewDecision):
        if payload.decision not in ("approved", "rejected"):
            raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")
        return service.review(review_id, payload.decision, actor="api", note=payload.note)

    # -- maintenance ---------------------------------------------------------

    @app.post("/backups", tags=["ops"], dependencies=[Depends(guard)])
    def create_backup():
        return {"path": service.backup()}

    @app.post("/maintenance/events", tags=["ops"], dependencies=[Depends(guard)])
    def compact_events(payload: EventCompaction):
        """Trim observation-only audit rows, dry run unless told otherwise.

        Everything that changed a memory's state stays forever; this clears only rows
        that repeat what the surviving record already says - and the trim itself is
        written to the trail it is trimming.
        """
        cutoff = utcnow() - timedelta(days=float(payload.older_than_days))
        result = service.store.compact_events(older_than=cutoff, dry_run=payload.dry_run,
                                             namespace=service.namespace)
        if not payload.dry_run and result["deleted"]:
            service.store.event("events_compacted", namespace=service.namespace, actor="api",
                                detail={"deleted": result["deleted"],
                                        "older_than": result["older_than"],
                                        "droppable": result["droppable"]})
        return result

    @app.post("/maintenance/checkpoint", tags=["ops"])
    def checkpoint(truncate: bool = True):
        """Fold the WAL into the main file; the response says whether it was busy."""
        return service.checkpoint(truncate=truncate)

    @app.post("/maintenance/purge", tags=["ops"], dependencies=[Depends(guard)])
    def purge(retain_days: float = Query(90.0, ge=0.0), limit: int = Query(1000, ge=1, le=10000)):
        """Hard delete long-retired rows. Refused unless the policy opts in."""
        policy = service.policy()
        if not policy.allow_hard_delete:
            raise HTTPException(status_code=409,
                                detail="allow_hard_delete is False; retired records are kept on disk")
        from datetime import timedelta

        from .store import utcnow
        cutoff = utcnow() - timedelta(days=float(retain_days))
        deleted = service.store.delete_rows(state="pruned", older_than=cutoff, limit=limit,
                                           actor="api", run_id=None)
        return {"deleted": deleted, "cutoff": cutoff.isoformat()}

    @app.get("/safety", tags=["ops"])
    def safety():
        """The knobs that keep a pass from doing damage, as currently in force."""
        policy = service.policy()
        counts = service.store.counts()
        return {"policy": policy.summary(), "active": counts.get("active"),
                "min_active_memories": policy.min_active_memories,
                "would_block_total_removal": (counts.get("active", 0) - policy.min_active_memories) <= 0,
                "auto_apply_merges": policy.auto_apply_merges,
                "auto_merge_exact": policy.auto_merge_exact,
                "allow_auto_prune": policy.allow_auto_prune,
                "allow_hard_delete": policy.allow_hard_delete,
                "open_reviews": counts.get("open_reviews"),
                "dry_run_by_default": True}

    return app
