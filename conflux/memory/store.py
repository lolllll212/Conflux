"""SQLite-backed memory store: the durable half of the memory layer.

Conventions borrowed from the rest of Conflux: writes are recorded rather than
applied in place (``memory_events`` is append-only, like the action journal),
ids are opaque strings, and everything public is safe to call from the
scheduler thread and a request thread at the same time.

Safety properties the rest of the package relies on:

* one connection, guarded by an :class:`threading.RLock`, with explicit
  ``BEGIN IMMEDIATE`` transactions and nested ``SAVEPOINT`` support, so a
  multi-statement transition is all-or-nothing;
* ``memories`` rows are never deleted by a cleanup pass — only their ``state``
  changes — and every transition carries the full pre-image in the audit trail;
* a single-row ``run_lock`` with a lease makes runs mutually exclusive across
  processes as well as threads, and self-heals if a holder dies.
"""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .policy import CleanupPolicy
from .schema import PRAGMAS, apply_migrations, SCHEMA_VERSION, user_version
from .similar import (content_fingerprint, normalize_text, pack_vector, signature, unpack_vector)

ACTIVE = "active"
STATES = (ACTIVE, "merged", "pruned", "superseded", "archived")


# --- time -------------------------------------------------------------------


def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def parse_ts(value):
    """Accept an ISO string, an epoch number or a datetime; return aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z") or text.endswith("z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid timestamp {value!r} (expected ISO-8601)") from exc
    else:
        raise ValueError(f"invalid timestamp {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def age_days(value, now=None):
    """Days elapsed since ``value`` (negative for future timestamps)."""
    moment = parse_ts(value)
    if moment is None:
        return None
    reference = parse_ts(now) if now is not None else utcnow()
    return (reference - moment).total_seconds() / 86400.0


def new_id(prefix="mem"):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _dumps(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _loads(text, default=None):
    if not text:
        return {} if default is None else default
    try:
        return json.loads(text)
    except ValueError:
        return {} if default is None else default


# --- records ----------------------------------------------------------------


@dataclass(frozen=True)
class Memory:
    """One memory row. ``vector`` is decoded on demand and never compared."""

    id: str
    namespace: str = "default"
    kind: str = "note"
    topic: str = None
    body: str = ""
    body_norm: str = ""
    fingerprint: str = ""
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    source_agent: str = None
    uri: str = None
    confidence: float = None
    importance: float = 1.0
    state: str = ACTIVE
    superseded_by: str = None
    merge_id: int = None
    embedding_model: str = None
    embedding_dim: int = None
    created_at: str = None
    updated_at: str = None
    last_seen_at: str = None
    expires_at: str = None
    access_count: int = 1
    revision: int = 1
    idempotency_key: str = None
    vector: tuple = field(default=None, repr=False, compare=False)

    @classmethod
    def from_row(cls, row, with_vector=True):
        blob = row["embedding"] if "embedding" in row.keys() else None
        vector = None
        if with_vector and blob:
            vector = tuple(unpack_vector(blob, row["embedding_dim"]) or ())
        return cls(
            id=row["id"], namespace=row["namespace"], kind=row["kind"], topic=row["topic"],
            body=row["body"], body_norm=row["body_norm"], fingerprint=row["fingerprint"],
            metadata=_loads(row["metadata"]), provenance=_loads(row["provenance"]),
            source_agent=row["source_agent"], uri=row["uri"], confidence=row["confidence"],
            importance=row["importance"], state=row["state"], superseded_by=row["superseded_by"],
            merge_id=row["merge_id"], embedding_model=row["embedding_model"],
            embedding_dim=row["embedding_dim"], created_at=row["created_at"],
            updated_at=row["updated_at"], last_seen_at=row["last_seen_at"],
            expires_at=row["expires_at"], access_count=row["access_count"],
            revision=row["revision"], idempotency_key=row["idempotency_key"],
            vector=vector,
        )

    @property
    def origin_ids(self):
        """Every upstream memory id recorded for this record (provenance)."""
        return list(self.provenance.get("origin_ids") or [])

    def signature(self):
        return signature(self.body, self.vector or None)

    def to_dict(self, include_body=True):
        out = {
            "id": self.id, "namespace": self.namespace, "kind": self.kind, "topic": self.topic,
            "metadata": self.metadata, "provenance": self.provenance,
            "source_agent": self.source_agent, "uri": self.uri, "confidence": self.confidence,
            "importance": self.importance, "state": self.state,
            "superseded_by": self.superseded_by, "merge_id": self.merge_id,
            "embedding_model": self.embedding_model, "embedding_dim": self.embedding_dim,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at, "expires_at": self.expires_at,
            "access_count": self.access_count, "revision": self.revision,
            "fingerprint": self.fingerprint,
        }
        if include_body:
            out["body"] = self.body
        return out


# --- errors -----------------------------------------------------------------


class StoreError(RuntimeError):
    pass


class MemoryNotFound(StoreError, KeyError):
    pass


class DuplicateMemory(StoreError):
    pass


class RunInProgress(StoreError):
    """Raised when a cleanup run cannot take the exclusive lease."""


class PolicyViolation(StoreError):
    """A mutation was refused by a safety guard."""


class PlanStale(StoreError):
    """A plan's target row changed after planning; the step is skipped, not guessed."""


# --- store ------------------------------------------------------------------


class MemoryStore:
    """Thread-safe SQLite handle for one memory database.

    Opens (creating and migrating as needed) the database at ``path``, then
    exposes record, merge, review, run-ledger and lock operations. Callers that
    need atomic multi-row writes go through :meth:`transaction`; every helper
    here already does.
    """

    def __init__(self, path, namespace="default", *, timeout=30.0, create=True):
        self.path = str(path)
        self.namespace = namespace or "default"
        self._lock = threading.RLock()
        self._depth = 0
        if not create and self.path != ":memory:" and not self.path.startswith("file:"):
            import os
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"no memory database at {self.path}")
        self._conn = sqlite3.connect(self.path, timeout=timeout, isolation_level=None,
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            for pragma in PRAGMAS:
                self._conn.execute(pragma)
            apply_migrations(self._conn)
        except Exception:
            self._conn.close()
            raise

    # -- lifecycle ------------------------------------------------------------

    @classmethod
    def open(cls, path, namespace="default", **kwargs):
        return cls(path, namespace=namespace, **kwargs)

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def schema_version(self):
        return user_version(self._conn)

    def integrity_check(self):
        with self._lock:
            rows = self._conn.execute("PRAGMA integrity_check").fetchall()
        return [r[0] for r in rows]

    # -- transactions ---------------------------------------------------------

    @contextmanager
    def transaction(self, immediate=True):
        """One atomic unit of work. Nested uses savepoints, not new BEGINs."""
        with self._lock:
            if self._depth:
                name = f"sp_{self._depth}"
                self._conn.execute(f"SAVEPOINT {name}")
                self._depth += 1
                try:
                    yield self._conn
                except Exception:
                    self._conn.execute(f"ROLLBACK TO {name}")
                    self._conn.execute(f"RELEASE {name}")
                    raise
                else:
                    self._conn.execute(f"RELEASE {name}")
                    self._depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN DEFERRED")
            self._depth += 1
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                self._depth -= 1
                raise
            self._conn.execute("COMMIT")
            self._depth -= 1

    def _execute(self, sql, params=()):
        return self._conn.execute(sql, params)

    def query(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def query_one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- memories -------------------------------------------------------------

    def add(self, body, *, kind="note", topic=None, namespace=None, metadata=None, provenance=None,
             source_agent=None, uri=None, confidence=None, importance=1.0, embedding=None,
             access_count=1, embedding_model=None, idempotency_key=None, on_duplicate="touch",
             memory_id=None,
             created_at=None, updated_at=None, last_seen_at=None, expires_at=None, run_id=None,
             ttl_days=None, actor=None, source="agent", origin_ids=None, state=ACTIVE):
        """Insert (or idempotently refresh) one memory, returning a receipt.

        ``ttl_days`` is the relative form of ``expires_at`` ("valid for 30
        days"), resolved here so every entry point means the same thing by an
        expiry; setting both is refused rather than silently preferred.

        ``on_duplicate="touch"`` is the default so an agent that re-states a
        fact it already wrote does not grow the table: the existing row gets a
        provenance union and an access bump, and the receipt reports
        ``created=False``. ``"insert"`` keeps both rows for the cleanup pass to
        reason about; ``"error"`` raises :class:`DuplicateMemory`.
        """
        clean = str(body or "").strip()
        if not clean:
            raise ValueError("memory body must not be empty")
        if len(clean) > 100_000:
            raise ValueError("memory body too long (max 100000 chars)")
        if on_duplicate not in ("touch", "insert", "error"):
            # refuse loudly instead of silently behaving like "insert"
            raise ValueError("on_duplicate must be one of 'touch', 'insert', 'error', "
                             f"got {on_duplicate!r}")
        ns = namespace or self.namespace
        norm = normalize_text(clean)
        fp = content_fingerprint(clean, kind, topic)
        prov = dict(provenance or {})
        prov.setdefault("source", source)
        if source_agent:
            prov.setdefault("agent", source_agent)
        if uri:
            prov.setdefault("uri", uri)
        origins = list(dict.fromkeys([*(prov.get("origin_ids") or []), *(origin_ids or [])]))
        if origins:
            prov["origin_ids"] = origins
        if ttl_days is not None:
            try:
                days = float(ttl_days)
            except (TypeError, ValueError):
                raise ValueError(f"ttl_days must be a number of days, got {ttl_days!r}")
            if not days > 0:
                # a zero or negative TTL would retire the row as it is written
                raise ValueError(f"ttl_days must be positive, got {ttl_days!r}")
            if expires_at:
                raise ValueError("set either ttl_days or expires_at, not both")
            expires_at = utcnow() + timedelta(days=days)
        stamp = iso(created_at) or iso(utcnow())
        update_stamp = iso(updated_at) or stamp
        seen_stamp = iso(last_seen_at) or update_stamp
        vector_blob = None
        dim = None
        if embedding is not None:
            vector_blob = pack_vector(embedding)
            dim = len(embedding)
        with self.transaction():
            if idempotency_key:
                hit = self._execute(
                    "SELECT id, state FROM memories WHERE namespace=? AND idempotency_key=?",
                    (ns, idempotency_key)).fetchone()
                if hit is not None:
                    self._event("replay", memory_id=hit["id"], namespace=ns, run_id=run_id,
                                actor=actor, detail={"idempotency_key": idempotency_key,
                                                     "state": hit["state"]})
                    return {"id": hit["id"], "created": False, "reason": "idempotency_key",
                            "namespace": ns, "revision": None}
            exact = self._execute(
                "SELECT id, revision, provenance FROM memories "
                "WHERE namespace=? AND fingerprint=? AND state=? ORDER BY updated_at DESC",
                (ns, fp, ACTIVE)).fetchall()
            if exact and on_duplicate in ("touch", "error"):
                row = exact[-1]
                if on_duplicate == "error":
                    raise DuplicateMemory(f"memory {row['id']} already holds this body")
                merged_prov = _union_provenance(_loads(row["provenance"]), prov)
                other = prov.get("agent")
                if other:
                    told = list(dict.fromkeys([*(merged_prov.get("also_reported_by") or []), other]))
                    merged_prov["also_reported_by"] = told[-20:]
                merged_prov["seen_at"] = list(dict.fromkeys([*(merged_prov.get("seen_at") or []),
                                                              stamp]))[-50:]
                self._execute(
                    "UPDATE memories SET access_count=access_count+1, last_seen_at=?, provenance=?, "
                    "updated_at=MAX(updated_at, ?) WHERE namespace=? AND id=?",
                    (stamp, _dumps(merged_prov), stamp, ns, row["id"]))
                seq = self._event("touch", memory_id=row["id"], namespace=ns, run_id=run_id,
                                  actor=actor, detail={"duplicate_of": row["id"], "body": clean})
                return {"id": row["id"], "created": False, "reason": "exact_duplicate",
                        "namespace": ns, "event_seq": seq, "revision": row["revision"]}
            memory_id = memory_id or new_id()
            columns, placeholders, values = [], [], []
            for name, value in (
                ("id", memory_id), ("namespace", ns), ("kind", kind or "note"), ("topic", topic),
                ("body", clean), ("body_norm", norm), ("fingerprint", fp),
                ("metadata", _dumps(dict(metadata or {}))), ("provenance", _dumps(prov)),
                ("source_agent", source_agent), ("uri", uri), ("confidence", confidence),
                ("importance", float(importance)), ("embedding", vector_blob),
                ("embedding_model", embedding_model if embedding is not None else None),
                ("embedding_dim", dim), ("state", state), ("created_at", stamp),
                ("updated_at", update_stamp),
                ("last_seen_at", seen_stamp), ("expires_at", iso(expires_at)),
                ("access_count", max(1, int(access_count))),
                ("revision", 1), ("idempotency_key", idempotency_key),
            ):
                columns.append(name)
                if name == "revision":
                    placeholders.append("1")  # revision is always 1 for a fresh row
                else:
                    placeholders.append("?")
                    values.append(value)
            self._execute(
                f"INSERT INTO memories ({', '.join(columns)}) VALUES ({', '.join(placeholders)})",
                values)
            detail = {"body": clean, "kind": kind, "topic": topic}
            if exact:
                detail["duplicate_of"] = [r["id"] for r in exact]
            seq = self._event("add", memory_id=memory_id, namespace=ns, run_id=run_id,
                              actor=actor, detail=detail, after=self._snapshot(memory_id))
            return {"id": memory_id, "created": True, "namespace": ns, "event_seq": seq,
                    "duplicate_of": [r["id"] for r in exact] if exact else [],
                    "fingerprint": fp, "revision": 1}

    def get(self, memory_id, *, namespace=None, with_vector=True):
        ns = namespace or self.namespace
        row = self._execute("SELECT * FROM memories WHERE namespace=? AND id=?", (ns, memory_id)).fetchone()
        if row is None:
            raise MemoryNotFound(f"no memory {memory_id!r} in namespace {ns!r}")
        return Memory.from_row(row, with_vector=with_vector)

    def find(self, ids, *, namespace=None, with_vector=True):
        ids = [i for i in (ids or []) if i]
        if not ids:
            return []
        ns = namespace or self.namespace
        marks = ",".join("?" * len(ids))
        rows = self._execute(
            f"SELECT * FROM memories WHERE namespace=? AND id IN ({marks}) ORDER BY updated_at DESC",
            (ns, *ids)).fetchall()
        return [Memory.from_row(r, with_vector=with_vector) for r in rows]

    def list(self, *, state=ACTIVE, kind=None, topic=None, namespace=None, limit=100, offset=0,
             updated_since=None, with_vector=False, order="updated_at"):
        """Page over memories; ``state=None`` returns every state."""
        ns = namespace or self.namespace
        sql = ["SELECT * FROM memories WHERE namespace=?", ]
        params = [ns]
        if state is not None:
            sql.append("AND state=?")
            params.append(state)
        if kind is not None:
            sql.append("AND kind=?")
            params.append(kind)
        if topic is not None:
            sql.append("AND topic=?")
            params.append(topic)
        if updated_since is not None:
            sql.append("AND updated_at>?")
            params.append(iso(updated_since))
        order = order if order in ("updated_at", "created_at", "id") else "updated_at"
        sql.append(f"ORDER BY {order} DESC, id ASC LIMIT ? OFFSET ?")
        params.extend([int(limit), int(offset)])
        rows = self._execute(" ".join(sql), params).fetchall()
        return [Memory.from_row(r, with_vector=with_vector) for r in rows]

    def active(self, *, namespace=None, limit=10000, with_vector=False):
        return self.list(state=ACTIVE, namespace=namespace, limit=limit, with_vector=with_vector)

    def counts(self, *, namespace=None):
        ns = namespace or self.namespace
        rows = self._execute("SELECT state, COUNT(*) AS n FROM memories WHERE namespace=? "
                             "GROUP BY state", (ns,)).fetchall()
        by_state = {r["state"]: r["n"] for r in rows}
        total = sum(by_state.values())
        kinds = self._execute("SELECT kind, COUNT(*) AS n FROM memories WHERE namespace=? AND state=? "
                              "GROUP BY kind ORDER BY n DESC", (ns, ACTIVE)).fetchall()
        return {"total": total, "by_state": by_state, ACTIVE: by_state.get(ACTIVE, 0),
                "kinds": {r["kind"]: r["n"] for r in kinds},
                "open_reviews": self._execute("SELECT COUNT(*) AS n FROM reviews WHERE namespace=? "
                                              "AND state='open'", (ns,)).fetchone()["n"],
                "proposed_merges": self._execute("SELECT COUNT(*) AS n FROM merges WHERE namespace=? "
                                                  "AND state='proposed'", (ns,)).fetchone()["n"]}

    def set_embedding(self, memory_id, vector, *, model="local-hash", namespace=None, run_id=None):
        ns = namespace or self.namespace
        blob = pack_vector(vector)
        with self.transaction():
            cur = self._execute("UPDATE memories SET embedding=?, embedding_model=?, embedding_dim=? "
                                "WHERE namespace=? AND id=?",
                                (blob, model, len(vector), ns, memory_id))
            if cur.rowcount:
                self._event("embedding_set", memory_id=memory_id, namespace=ns, run_id=run_id,
                            detail={"model": model, "dim": len(vector)})
            return cur.rowcount > 0

    def mark_seen(self, memory_id, *, namespace=None, run_id=None, count=1, at=None, actor=None):
        ns = namespace or self.namespace
        with self.transaction():
            cur = self._execute(
                "UPDATE memories SET access_count=access_count+?, last_seen_at=? "
                "WHERE namespace=? AND id=?", (int(count), iso(at or utcnow()), ns, memory_id))
            if cur.rowcount:
                self._event("seen", memory_id=memory_id, namespace=ns, run_id=run_id, actor=actor,
                            detail={"count": int(count)})
            return cur.rowcount > 0

    def touch_body(self, memory_id, *, body=None, metadata=None, topic=None, kind=None,
                   confidence=None, expires_at=None, namespace=None, run_id=None, actor=None,
                   provenance=None):
        """Content-level update; bumps ``revision`` so open plans go stale."""
        ns = namespace or self.namespace
        sets, params = ["updated_at=?"], [iso(utcnow())]
        snapshot = {}
        if body is not None:
            clean = str(body).strip()
            if not clean:
                raise ValueError("memory body must not be empty")
            row = self.get(memory_id, namespace=ns)
            sets += ["body=?", "body_norm=?", "fingerprint=?", "revision=revision+1"]
            params += [clean, normalize_text(clean), content_fingerprint(clean, kind or row.kind,
                                                                         topic or row.topic)]
            snapshot["body"] = clean
        if metadata is not None:
            sets += ["metadata=?", "revision=revision+1"]
            params.append(_dumps(dict(metadata)))
            snapshot["metadata"] = dict(metadata)
        if provenance is not None:
            sets += ["provenance=?", "revision=revision+1"]
            params.append(_dumps(dict(provenance)))
            snapshot["provenance"] = dict(provenance)
        if topic is not None:
            sets += ["topic=?", "revision=revision+1"]
            params.append(topic)
            snapshot["topic"] = topic
        if kind is not None:
            sets += ["kind=?", "revision=revision+1"]
            params.append(kind)
            snapshot["kind"] = kind
        if confidence is not None:
            sets.append("confidence=?")
            params.append(float(confidence))
        if expires_at is not None:
            sets.append("expires_at=?")
            params.append(iso(expires_at))
        with self.transaction():
            before = self._snapshot(memory_id, namespace=ns)
            cur = self._execute(f"UPDATE memories SET {', '.join(sets)} WHERE namespace=? AND id=?",
                                (*params, ns, memory_id))
            if not cur.rowcount:
                raise MemoryNotFound(f"no memory {memory_id!r} in namespace {ns!r}")
            self._event("update", memory_id=memory_id, namespace=ns, run_id=run_id, actor=actor,
                        detail=snapshot, before=before, after=self._snapshot(memory_id, namespace=ns))
            return self.get(memory_id, namespace=ns)

    def set_state(self, memory_id, state, *, superseded_by=None, merge_id=None, namespace=None,
                  run_id=None, actor=None, action=None, detail=None, expect_revision=None):
        """Move a memory to another state (the only mutation a cleanup performs).

        ``expect_revision`` makes the write a compare-and-set: if the row moved
        since the plan was built, this returns ``False`` and the caller skips
        the step instead of clobbering newer information.
        """
        if state not in STATES:
            raise ValueError(f"unknown memory state {state!r}")
        ns = namespace or self.namespace
        sql = ("UPDATE memories SET state=?, superseded_by=?, merge_id=?, revision=revision+1, "
               "updated_at=? WHERE namespace=? AND id=?")
        params = [state, superseded_by, merge_id, iso(utcnow()), ns, memory_id]
        if expect_revision is not None:
            sql += " AND revision=?"
            params.append(int(expect_revision))
        with self.transaction():
            before = self._snapshot(memory_id, namespace=ns)
            cur = self._execute(sql, params)
            if not cur.rowcount:
                return False
            self._event(action or f"state:{state}", memory_id=memory_id, namespace=ns, run_id=run_id,
                        actor=actor, detail=dict(detail or {}, to=state, superseded_by=superseded_by,
                                                 merge_id=merge_id),
                        before=before, after=self._snapshot(memory_id, namespace=ns))
            return True

    def delete_rows(self, *, state="pruned", older_than=None, namespace=None, limit=1000,
                    run_id=None, actor=None):
        """Hard delete — the one destructive primitive, and it keeps the trail.

        Only rows already in ``state`` and older than ``older_than`` are
        removed, the audit ``before`` snapshot of every removed row is written
        first, and the rows themselves stay recoverable from that trail.
        """
        ns = namespace or self.namespace
        if older_than is None:
            raise ValueError("delete_rows requires an explicit older_than cutoff")
        with self.transaction():
            rows = self._execute(
                "SELECT * FROM memories WHERE namespace=? AND state=? AND updated_at<? "
                "ORDER BY updated_at ASC LIMIT ?", (ns, state, iso(older_than), int(limit))).fetchall()
            for row in rows:
                snapshot = dict(row)
                snapshot.pop("embedding", None)
                self._event("hard_delete", memory_id=row["id"], namespace=ns, run_id=run_id,
                            actor=actor, detail={"state": state, "cutoff": iso(older_than)},
                            before=snapshot)
            marks = [r["id"] for r in rows]
            if marks:
                self._execute(
                    f"DELETE FROM memories WHERE namespace=? AND state=? AND id IN "
                    f"({','.join('?' * len(marks))})", (ns, state, *marks))
            return len(marks)

    def _snapshot(self, memory_id, namespace=None):
        ns = namespace or self.namespace
        row = self._execute("SELECT * FROM memories WHERE namespace=? AND id=?", (ns, memory_id)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data.pop("embedding", None)
        return data

    # -- audit trail ----------------------------------------------------------

    def _event(self, action, *, memory_id=None, namespace=None, target=None, detail=None,
               run_id=None, actor=None, before=None, after=None, ts=None):
        ns = namespace or self.namespace
        cur = self._execute(
            "INSERT INTO memory_events (ts, namespace, action, memory_id, target, run_id, actor, "
            "detail, before, after) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (iso(ts or utcnow()), ns, action, memory_id, target, run_id, actor,
             _dumps(detail or {}), _dumps(before) if before is not None else None,
             _dumps(after) if after is not None else None))
        return cur.lastrowid

    def event(self, action, **kwargs):
        """Public audit hook (usable outside a transaction for notes)."""
        with self.transaction():
            return self._event(action, **kwargs)

    def events(self, *, memory_id=None, run_id=None, namespace=None, action=None, limit=200,
               after_seq=0):
        ns = namespace or self.namespace
        sql = ["SELECT * FROM memory_events WHERE namespace=? AND seq>?"]
        params = [ns, int(after_seq)]
        if memory_id:
            sql.append("AND memory_id=?")
            params.append(memory_id)
        if run_id:
            sql.append("AND run_id=?")
            params.append(run_id)
        if action:
            sql.append("AND action LIKE ?")
            params.append(action)
        sql.append("ORDER BY seq DESC LIMIT ?")
        params.append(int(limit))
        rows = self._execute(" ".join(sql), params).fetchall()
        out = []
        for row in reversed(rows):
            item = dict(row)
            item["detail"] = _loads(item.get("detail"))
            item["before"] = _loads(item.get("before"), None) if item.get("before") else None
            item["after"] = _loads(item.get("after"), None) if item.get("after") else None
            out.append(item)
        return out

    def timeline(self, memory_id, *, namespace=None, limit=100):
        return self.events(memory_id=memory_id, namespace=namespace, limit=limit)

    # -- merges ---------------------------------------------------------------

    def find_open_merge(self, cluster_key, *, namespace=None):
        """The still-open proposal for this cluster, if a previous pass made one."""
        if not cluster_key:
            return None
        ns = namespace or self.namespace
        row = self._execute("SELECT * FROM merges WHERE namespace=? AND cluster_key=? AND "
                            "state IN ('proposed', 'needs_review') ORDER BY id DESC LIMIT 1",
                            (ns, cluster_key)).fetchone()
        return dict(row) if row else None

    def propose_merge(self, *, merged_body, source_ids, run_id=None, topic=None, confidence=None,
                      similarity=None, reason=None, model=None, namespace=None, state="proposed",
                      merged_memory_id=None, sources=None, note=None, cluster_key=None):
        """Persist a merge proposal with its provenance, atomically.

        ``sources`` is the durable provenance document (one entry per input
        memory: id, body, kind, timestamps, agent); ``merge_links`` is its
        queryable projection so a memory's merge history can be walked later.
        """
        ns = namespace or self.namespace
        merged_norm = normalize_text(merged_body)
        with self.transaction():
            existing = self.find_open_merge(cluster_key, namespace=ns)
            if existing is not None and existing["merged_norm"] == merged_norm:
                self._execute("UPDATE merges SET run_id=?, confidence=?, similarity=?, reason=?, "
                              "model=?, note=COALESCE(?, note), sources=? WHERE id=?",
                              (run_id, confidence, similarity, reason, model, note,
                               _dumps(list(sources or [])), existing["id"]))
                for memory_id in dict.fromkeys(source_ids):
                    self._execute("INSERT OR IGNORE INTO merge_links (merge_id, namespace, memory_id, "
                                  "role) VALUES (?,?,?,'source')", (existing["id"], ns, memory_id))
                self._event("merge_refreshed", namespace=ns, run_id=run_id,
                            detail={"merge_id": existing["id"], "source_ids": list(source_ids),
                                    "confidence": confidence})
                return existing["id"]
            cur = self._execute(
                "INSERT INTO merges (namespace, run_id, created_at, state, topic, merged_body, "
                "merged_norm, fingerprint, merged_memory_id, confidence, similarity, reason, model, "
                "note, sources, cluster_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ns, run_id, iso(utcnow()), state, topic, str(merged_body).strip(), merged_norm,
                 content_fingerprint(merged_body), merged_memory_id, confidence, similarity, reason,
                 model, note, _dumps(list(sources or [])), cluster_key))
            merge_id = cur.lastrowid
            links = []
            for memory_id in dict.fromkeys(source_ids):
                links.append((merge_id, ns, memory_id, "source"))
            for entry in (sources or []):
                if isinstance(entry, dict) and entry.get("id"):
                    pair = (merge_id, ns, entry["id"], entry.get("role", "source"))
                    if pair not in links:
                        links.append(pair)
            self._conn.executemany("INSERT OR IGNORE INTO merge_links (merge_id, namespace, "
                                   "memory_id, role) VALUES (?,?,?,?)", links)
            self._event("merge_proposed", namespace=ns, target=merged_norm[:80], run_id=run_id,
                        detail={"merge_id": merge_id, "source_ids": list(source_ids),
                                "confidence": confidence, "reason": reason, "model": model},
                        after={"merged_body": merged_body})
            return merge_id

    def get_merge(self, merge_id, *, namespace=None):
        ns = namespace or self.namespace
        row = self._execute("SELECT * FROM merges WHERE id=? AND (namespace=? OR namespace='*')",
                            (int(merge_id), ns)).fetchone()
        if row is None:
            raise MemoryNotFound(f"no merge {merge_id!r}")
        merge = dict(row)
        merge["sources"] = _loads(merge.get("sources") or "[]")
        merge["links"] = [dict(r) for r in self._execute(
            "SELECT memory_id, role FROM merge_links WHERE merge_id=? ORDER BY memory_id",
            (int(merge_id),)).fetchall()]
        return merge

    def merges(self, *, state=None, run_id=None, namespace=None, memory_id=None, limit=100):
        ns = namespace or self.namespace
        sql = ["SELECT * FROM merges WHERE namespace=?"]
        params = [ns]
        if state == "open":  # convenience: anything not yet decided
            sql.append("AND state IN ('proposed', 'needs_review')")
        elif state:
            sql.append("AND state=?")
            params.append(state)
        if run_id:
            sql.append("AND run_id=?")
            params.append(run_id)
        if memory_id:
            sql.append("AND id IN (SELECT merge_id FROM merge_links WHERE namespace=? AND memory_id=?)")
            params.extend([ns, memory_id])
        sql.append("ORDER BY id DESC LIMIT ?")
        params.append(int(limit))
        out = []
        for row in self._execute(" ".join(sql), params).fetchall():
            item = dict(row)
            item["sources"] = _loads(item.get("sources") or "[]")
            out.append(item)
        return out

    def set_merge_state(self, merge_id, state, *, run_id=None, note=None, merged_memory_id=None,
                        applied_at=None, reverted_at=None, namespace=None):
        ns = namespace or self.namespace
        sets, params = ["state=?"], [state]
        if state == "applied":
            sets.append("applied_at=?")
            params.append(iso(applied_at or utcnow()))
        if state == "reverted":
            sets.append("reverted_at=?")
            params.append(iso(reverted_at or utcnow()))
        if note is not None:
            sets.append("note=?")
            params.append(note)
        if merged_memory_id is not None:
            sets.append("merged_memory_id=?")
            params.append(merged_memory_id)
        with self.transaction():
            self._execute(f"UPDATE merges SET {', '.join(sets)} WHERE id=? AND namespace=?",
                          (*params, int(merge_id), ns))
            self._event(f"merge_{state}", namespace=ns, run_id=run_id,
                        detail={"merge_id": int(merge_id), "note": note})
            return True

    def apply_merge(self, merge_id, *, source_ids, merged_body, kind="merged", topic=None,
                    confidence=None, importance=1.0, metadata=None, reason=None, model=None,
                    expected=None, run_id=None, actor=None, namespace=None):
        """Fold a cluster into one new memory, atomically.

        Verifies every source is still ``active`` at the revision captured when
        the plan was built, inserts the merged record, transitions the sources
        to ``merged`` pointing at it, and marks the merge applied. Any mismatch
        raises :class:`PlanStale` and the whole step rolls back — a cluster is
        never half-merged.
        """
        ns = namespace or self.namespace
        expected = {str(k): int(v) for k, v in dict(expected or {}).items()}
        with self.transaction():
            marks = ",".join("?" * len(source_ids))
            rows = self._execute(
                f"SELECT * FROM memories WHERE namespace=? AND id IN ({marks})",
                (ns, *source_ids)).fetchall()
            by_id = {r["id"]: r for r in rows}
            for memory_id in source_ids:
                row = by_id.get(memory_id)
                if row is None:
                    raise PlanStale(f"source {memory_id} disappeared")
                if row["state"] != ACTIVE:
                    raise PlanStale(f"source {memory_id} is {row['state']}, not active")
                want = expected.get(memory_id)
                if want is not None and int(row["revision"]) != want:
                    raise PlanStale(f"source {memory_id} revision {row['revision']} != planned {want}")
            origins = []
            for row in rows:
                for origin in (_loads(row["provenance"]).get("origin_ids") or []):
                    if origin not in origins and origin not in source_ids:
                        origins.append(origin)
            record_meta = dict(metadata or {})
            record_meta.setdefault("merged_from", list(source_ids))
            record_meta.setdefault("merge_id", int(merge_id))
            receipt = self.add(
                merged_body, kind=kind, topic=topic, namespace=ns, metadata=record_meta,
                provenance={"source": "cleanup", "agent": actor or "cleanup", "detail": reason or "",
                            "origin_ids": list(source_ids) + origins},
                confidence=None if confidence is None else round(float(confidence), 4),
                importance=float(importance), on_duplicate="insert", run_id=run_id,
                actor=actor, state=ACTIVE)
            merged_id = receipt["id"]
            for memory_id in source_ids:
                cur = self._execute(
                    "UPDATE memories SET state='merged', superseded_by=?, merge_id=?, "
                    "revision=revision+1, updated_at=? WHERE namespace=? AND id=? AND state=?",
                    (merged_id, int(merge_id), iso(utcnow()), ns, memory_id, ACTIVE))
                if not cur.rowcount:  # unreachable while the transaction holds the write lock
                    raise PlanStale(f"source {memory_id} changed mid-transaction")
                self._event("state:merged", memory_id=memory_id, namespace=ns, run_id=run_id,
                            actor=actor,
                            detail={"merge_id": int(merge_id), "superseded_by": merged_id,
                                    "confidence": confidence, "reason": reason, "model": model},
                            before=dict(by_id[memory_id]), after=self._snapshot(merged_id, namespace=ns))
            self._execute("UPDATE merges SET state='applied', applied_at=?, merged_memory_id=? "
                          "WHERE id=? AND namespace=?", (iso(utcnow()), merged_id, int(merge_id), ns))
            self._event("merge_applied", namespace=ns, run_id=run_id, actor=actor,
                        target=merged_id,
                        detail={"merge_id": int(merge_id), "sources": list(source_ids),
                                "merged_memory_id": merged_id, "model": model})
            return merged_id

    def revert_merge(self, merge_id, *, namespace=None, run_id=None, actor=None, note=None):
        """Undo a merge: sources go back to active, the merged record is retired."""
        ns = namespace or self.namespace
        with self.transaction():
            row = self._execute("SELECT * FROM merges WHERE id=? AND namespace=?",
                                (int(merge_id), ns)).fetchone()
            if row is None:
                raise MemoryNotFound(f"no merge {merge_id!r}")
            if row["state"] == "reverted":
                return {"merge_id": int(merge_id), "reverted": False, "reason": "already reverted"}
            source_ids = [r["memory_id"] for r in self._execute(
                "SELECT memory_id FROM merge_links WHERE merge_id=? ORDER BY memory_id",
                (int(merge_id),)).fetchall()]
            merged_id = row["merged_memory_id"]
            restored = []
            for memory_id in source_ids:
                cur = self._execute(
                    "UPDATE memories SET state=?, superseded_by=NULL, merge_id=NULL, "
                    "revision=revision+1, updated_at=? WHERE namespace=? AND id=? AND state='merged'",
                    (ACTIVE, iso(utcnow()), ns, memory_id))
                if cur.rowcount:
                    restored.append(memory_id)
                    self._event("merge_reverted", memory_id=memory_id, namespace=ns, run_id=run_id,
                                actor=actor, detail={"merge_id": int(merge_id), "note": note})
            if merged_id:
                self._execute("UPDATE memories SET state='superseded', revision=revision+1, "
                              "updated_at=? WHERE namespace=? AND id=?", (iso(utcnow()), ns, merged_id))
                self._event("merge_reverted", memory_id=merged_id, namespace=ns, run_id=run_id,
                            actor=actor, detail={"merge_id": int(merge_id), "retired": True})
            self._execute("UPDATE merges SET state='reverted', reverted_at=?, note=? WHERE id=?",
                          (iso(utcnow()), note, int(merge_id)))
            self._event("merge_reverted", namespace=ns, run_id=run_id, actor=actor,
                        detail={"merge_id": int(merge_id), "restored": restored})
            return {"merge_id": int(merge_id), "reverted": True, "restored": restored,
                    "retired_merged": merged_id}

    # -- review queue ---------------------------------------------------------

    def open_review(self, *, kind, memory_ids, reason, run_id=None, confidence=None, merge_id=None,
                    detail=None, namespace=None):
        ns = namespace or self.namespace
        subject = _dumps(sorted(memory_ids))
        with self.transaction():
            cur = self._execute(
                "INSERT OR IGNORE INTO reviews (namespace, run_id, created_at, kind, state, "
                "confidence, reason, memory_ids, merge_id, detail) VALUES (?,?,?,?,'open',?,?,?,?,?)",
                (ns, run_id, iso(utcnow()), kind, confidence, reason, subject, merge_id,
                 _dumps(detail or {})))
            review_id = cur.lastrowid
            if review_id is None:
                row = self._execute("SELECT id FROM reviews WHERE namespace=? AND run_id IS ? AND "
                                    "kind=? AND memory_ids=?", (ns, run_id, kind, subject)).fetchone()
                return row["id"] if row else None
            for memory_id in memory_ids:
                self._event("review_opened", memory_id=memory_id, namespace=ns, run_id=run_id,
                            detail={"review_id": review_id, "kind": kind, "reason": reason,
                                    "confidence": confidence})
            return review_id

    def find_open_review(self, kind, memory_ids, *, namespace=None):
        """An equivalent review is opened once, never re-created per run."""
        ns = namespace or self.namespace
        subject = _dumps(sorted(memory_ids))
        row = self._execute("SELECT * FROM reviews WHERE namespace=? AND state='open' AND kind=? "
                            "AND memory_ids=? ORDER BY id DESC LIMIT 1", (ns, kind, subject)).fetchone()
        return dict(row) if row else None

    def reviews(self, *, state="open", namespace=None, memory_id=None, run_id=None, limit=100):
        """List the queue. ``state`` is 'open', 'resolved' (anything decided), an
        exact state, or None for everything."""
        ns = namespace or self.namespace
        sql = ["SELECT * FROM reviews WHERE namespace=?"]
        params = [ns]
        if state == "resolved":
            sql.append("AND state!='open'")
        elif state:
            sql.append("AND state=?")
            params.append(state)
        if run_id:
            sql.append("AND run_id=?")
            params.append(run_id)
        if memory_id:
            sql.append("AND memory_ids LIKE ?")
            params.append(f'%"{memory_id}"%')
        sql.append("ORDER BY id DESC LIMIT ?")
        params.append(int(limit))
        out = []
        for row in self._execute(" ".join(sql), params).fetchall():
            item = dict(row)
            item["memory_ids"] = _loads(item.get("memory_ids") or "[]")
            item["detail"] = _loads(item.get("detail") or "{}")
            out.append(item)
        return out

    def resolve_review(self, review_id, decision, *, note=None, actor=None, namespace=None, run_id=None):
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        ns = namespace or self.namespace
        with self.transaction():
            row = self._execute("SELECT * FROM reviews WHERE id=? AND namespace=?",
                                (int(review_id), ns)).fetchone()
            if row is None:
                raise MemoryNotFound(f"no review {review_id!r}")
            if row["state"] != "open":
                return {"id": int(review_id), "state": row["state"], "changed": False}
            self._execute("UPDATE reviews SET state=?, resolved_at=?, resolved_by=?, resolution_note=? "
                          "WHERE id=? AND state='open'",
                          (decision, iso(utcnow()), actor, note, int(review_id)))
            for memory_id in _loads(row["memory_ids"] or "[]"):
                self._event("review_resolved", memory_id=memory_id, namespace=ns, run_id=run_id,
                            actor=actor, detail={"review_id": int(review_id), "kind": row["kind"],
                                                 "decision": decision, "note": note})
            return {"id": int(review_id), "state": decision, "changed": True,
                    "kind": row["kind"], "memory_ids": _loads(row["memory_ids"] or "[]"),
                    "merge_id": row["merge_id"]}

    # -- run ledger and locks -------------------------------------------------

    def create_run(self, *, namespace=None, trigger="manual", mode="dry_run", actor=None,
                   policy=None, model=None, run_id=None):
        ns = namespace or self.namespace
        run_id = run_id or f"run-{utcnow().strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:6]}"
        with self.transaction():
            self._execute(
                "INSERT INTO runs (id, namespace, trigger, mode, status, started_at, actor, model, "
                "policy) VALUES (?,?,?,?,'running',?,?,?,?)",
                (run_id, ns, trigger, mode, iso(utcnow()), actor, model, _dumps(policy or {})))
            self._event("run_started", namespace=ns, target=run_id, run_id=run_id, actor=actor,
                        detail={"trigger": trigger, "mode": mode, "model": model})
        return run_id

    def update_run(self, run_id, *, status=None, stats=None, plan_hash=None, plan_json=None,
                   error=None, namespace=None, finished=False, actor=None):
        sets, params = [], []
        if status:
            sets.append("status=?")
            params.append(status)
        if stats is not None:
            sets.append("stats=?")
            params.append(_dumps(stats))
        if plan_hash is not None:
            sets.append("plan_hash=?")
            params.append(plan_hash)
        if plan_json is not None:
            sets.append("plan_json=?")
            params.append(plan_json if isinstance(plan_json, str) else _dumps(plan_json))
        if error is not None:
            sets.append("error=?")
            params.append(str(error))
        if finished:
            sets.append("finished_at=?")
            params.append(iso(utcnow()))
        if not sets:
            return False
        ns = namespace or self.namespace
        with self.transaction():
            self._execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=? AND namespace=?",
                          (*params, run_id, ns))
        return True

    def get_run(self, run_id, *, namespace=None):
        ns = namespace or self.namespace
        row = self._execute("SELECT * FROM runs WHERE id=? AND namespace=?", (run_id, ns)).fetchone()
        if row is None:
            raise MemoryNotFound(f"no run {run_id!r}")
        run = dict(row)
        run["policy"] = _loads(run.get("policy") or "{}")
        run["stats"] = _loads(run.get("stats") or "{}")
        run["plan"] = _loads(run["plan_json"], None) if run.get("plan_json") else None
        return run

    def list_runs(self, *, namespace=None, limit=20, status=None):
        ns = namespace or self.namespace
        sql = ["SELECT id, namespace, trigger, mode, status, started_at, finished_at, model, "
               "plan_hash, error, stats FROM runs WHERE namespace=?"]
        params = [ns]
        if status:
            sql.append("AND status=?")
            params.append(status)
        sql.append("ORDER BY started_at DESC LIMIT ?")
        params.append(int(limit))
        out = []
        for row in self._execute(" ".join(sql), params).fetchall():
            item = dict(row)
            item["stats"] = _loads(item.get("stats") or "{}")
            out.append(item)
        return out

    def find_completed_plan(self, plan_hash, *, namespace=None):
        """Runs are idempotent: an identical fingerprint already applied wins."""
        ns = namespace or self.namespace
        row = self._execute("SELECT id FROM runs WHERE namespace=? AND plan_hash=? AND "
                            "mode='apply' AND status='completed' ORDER BY finished_at DESC LIMIT 1",
                            (ns, plan_hash)).fetchone()
        return row["id"] if row else None

    def try_lock(self, *, owner, run_id=None, lease_seconds=900.0, namespace=None):
        """Take the run lease, or return the current holder. ``True`` = acquired."""
        ns = namespace or self.namespace
        now = utcnow()
        with self.transaction():
            row = self._execute("SELECT * FROM run_lock WHERE namespace=?", (ns,)).fetchone()
            if row is not None:
                expires = parse_ts(row["lease_expires_at"])
                if expires > now:
                    return {"acquired": False, "owner": row["owner"], "run_id": row["run_id"],
                            "acquired_at": row["acquired_at"], "lease_expires_at": row["lease_expires_at"]}
                self._event("run_lock_expired", namespace=ns, target=row["run_id"], run_id=row["run_id"],
                            detail={"stale_owner": row["owner"], "lease_expires_at": row["lease_expires_at"]})
            self._execute(
                "INSERT INTO run_lock (namespace, owner, run_id, acquired_at, lease_expires_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(namespace) DO UPDATE SET owner=excluded.owner, "
                "run_id=excluded.run_id, acquired_at=excluded.acquired_at, "
                "lease_expires_at=excluded.lease_expires_at",
                (ns, owner, run_id, iso(now), iso(now + timedelta(seconds=float(lease_seconds)))))
            return {"acquired": True}

    def renew_lock(self, *, owner, lease_seconds=900.0, namespace=None):
        ns = namespace or self.namespace
        now = utcnow()
        with self.transaction():
            cur = self._execute("UPDATE run_lock SET lease_expires_at=? WHERE namespace=? AND owner=?",
                                (iso(now + timedelta(seconds=float(lease_seconds))), ns, owner))
            return cur.rowcount > 0

    def unlock(self, *, owner, namespace=None):
        ns = namespace or self.namespace
        with self.transaction():
            cur = self._execute("DELETE FROM run_lock WHERE namespace=? AND owner=?", (ns, owner))
            return cur.rowcount > 0

    def lock_info(self, namespace=None):
        ns = namespace or self.namespace
        row = self._execute("SELECT * FROM run_lock WHERE namespace=?", (ns,)).fetchone()
        if row is None:
            return {"held": False}
        held = parse_ts(row["lease_expires_at"]) > utcnow()
        return {"held": held, "owner": row["owner"], "run_id": row["run_id"],
                "acquired_at": row["acquired_at"], "lease_expires_at": row["lease_expires_at"]}

    # -- review cursor --------------------------------------------------------

    def reviewed_fingerprints(self, ids, *, namespace=None):
        """``{memory_id: fingerprint}`` for ids already reviewed at that content."""
        ids = [i for i in ids if i]
        if not ids:
            return {}
        ns = namespace or self.namespace
        marks = ",".join("?" * len(ids))
        rows = self._execute(
            f"SELECT memory_id, fingerprint FROM review_cursor WHERE namespace=? AND memory_id IN ({marks})",
            (ns, *ids)).fetchall()
        return {r["memory_id"]: r["fingerprint"] for r in rows}

    def record_review(self, memory_id, fingerprint, *, namespace=None, run_id=None, verdict=None):
        ns = namespace or self.namespace
        self._execute(
            "INSERT INTO review_cursor (namespace, memory_id, fingerprint, reviewed_at, run_id, verdict) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(namespace, memory_id) DO UPDATE SET fingerprint=excluded."
            "fingerprint, reviewed_at=excluded.reviewed_at, run_id=excluded.run_id, verdict=excluded.verdict",
            (ns, memory_id, fingerprint, iso(utcnow()), run_id, verdict))

    def clear_review_cursor(self, *, namespace=None, memory_id=None):
        ns = namespace or self.namespace
        with self.transaction():
            if memory_id:
                cur = self._execute("DELETE FROM review_cursor WHERE namespace=? AND memory_id=?",
                                    (ns, memory_id))
            else:
                cur = self._execute("DELETE FROM review_cursor WHERE namespace=?", (ns,))
            return cur.rowcount

    # -- policy ---------------------------------------------------------------

    def get_policy(self, namespace=None, default=None):
        ns = namespace or self.namespace
        row = self._execute("SELECT policy FROM settings WHERE namespace=?", (ns,)).fetchone()
        if row is None:
            return default or CleanupPolicy()
        return CleanupPolicy.from_json(row["policy"], known_only=True)

    def set_policy(self, policy, *, namespace=None, updated_by=None):
        ns = namespace or self.namespace
        if isinstance(policy, dict):
            policy = CleanupPolicy.from_dict(policy)
        policy.validate()
        with self.transaction():
            self._execute(
                "INSERT INTO settings (namespace, policy, updated_at, updated_by) VALUES (?,?,?,?) "
                "ON CONFLICT(namespace) DO UPDATE SET policy=excluded.policy, "
                "updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                (ns, policy.to_json(), iso(utcnow()), updated_by))
            self._event("policy_updated", namespace=ns, actor=updated_by, target=ns,
                        detail={"policy": policy.summary()})
        return policy

    # -- scheduler state ------------------------------------------------------

    def scheduler_state(self, namespace=None):
        ns = namespace or self.namespace
        row = self._execute("SELECT * FROM scheduler_state WHERE namespace=?", (ns,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["enabled"] = bool(item["enabled"])
        return item

    def save_scheduler_state(self, *, namespace=None, enabled=None, timezone_name=None, hour=None,
                             minute=None, last_run_id=None, last_run_at=None, last_status=None,
                             next_run_at=None):
        ns = namespace or self.namespace
        fields = {"enabled": None if enabled is None else int(bool(enabled)),
                  "timezone": timezone_name, "hour": hour, "minute": minute,
                  "last_run_id": last_run_id, "last_run_at": last_run_at, "last_status": last_status,
                  "next_run_at": next_run_at}
        present = {k: v for k, v in fields.items() if v is not None}
        with self.transaction():
            columns = ["namespace", "updated_at"] + list(present)
            marks = ",".join("?" * len(columns))
            updates = ", ".join(f"{c}=excluded.{c}" for c in columns if c != "namespace")
            self._execute(
                f"INSERT INTO scheduler_state ({', '.join(columns)}) VALUES ({marks}) "
                f"ON CONFLICT(namespace) DO UPDATE SET {updates}",
                (ns, iso(utcnow()), *present.values()))
        return self.scheduler_state(ns)

    def claim_slot(self, slot, *, namespace=None):
        """Exactly-once claim for a scheduled fire time, persisted across restarts."""
        ns = namespace or self.namespace
        with self.transaction():
            row = self._execute("SELECT last_fired_slot FROM scheduler_state WHERE namespace=?",
                                (ns,)).fetchone()
            if row is not None and row["last_fired_slot"] == slot:
                return False
            if row is None:
                self._execute("INSERT INTO scheduler_state (namespace, updated_at, last_fired_slot) "
                              "VALUES (?,?,?)", (ns, iso(utcnow()), slot))
            else:
                self._execute("UPDATE scheduler_state SET last_fired_slot=?, updated_at=? "
                              "WHERE namespace=?", (slot, iso(utcnow()), ns))
            return True

    def release_slot(self, slot, *, namespace=None):
        ns = namespace or self.namespace
        with self.transaction():
            self._execute("UPDATE scheduler_state SET last_fired_slot=NULL WHERE namespace=? AND "
                          "last_fired_slot=?", (ns, slot))

    # -- maintenance ----------------------------------------------------------

    def vacuum_into(self, destination):
        """Online backup: SQLite's own consistent snapshot to a new file."""
        with self._lock:
            self._conn.execute("VACUUM INTO ?", (str(destination),))
        return str(destination)

    def checkpoint(self):
        with self._lock:
            return self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()

    def info(self):
        row = self._execute("PRAGMA page_count").fetchone()
        free = self._execute("PRAGMA freelist_count").fetchone()
        return {"path": self.path, "namespace": self.namespace, "schema_version": self.schema_version,
                "latest_schema_version": SCHEMA_VERSION, "pages": row[0] if row else None,
                "free_pages": free[0] if free else None, "states": STATES}


def _union_provenance(existing, incoming):
    """Merge provenance documents additively — never drop a source reference."""
    out = dict(existing or {})
    for key, value in (incoming or {}).items():
        if key == "origin_ids":
            out[key] = list(dict.fromkeys([*(out.get("origin_ids") or []), *(value or [])]))
        elif key == "seen_at":
            seen = list(out.get("seen_at") or [])
            if value and value not in seen:
                seen.append(value)
            out["seen_at"] = seen[-50:]
        elif key not in out or out[key] in (None, "", {}, []):
            out[key] = value
    return out
