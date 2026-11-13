"""CrewAI drop-in adapters backed by Conflux's CRDT state fabric.

Provides three integrations for CrewAI multi-agent crews and Flows:

1. :class:`ConfluxCrewStorage` — implements CrewAI's ``StorageBackend`` protocol
   (``save``, ``search``, ``delete``, ``update``, ``get_record``, ``list_records``,
   ``get_scope_info``, ``list_scopes``, ``list_categories``, ``count``, ``reset``,
   plus async counterparts) so ``Crew(memory=Memory(storage=ConfluxCrewStorage(...)))``
   replicates agent memories deterministically across a Conflux cluster.
2. :class:`ConfluxFlowPersistence` — implements CrewAI's ``FlowPersistence``
   interface (``init_db``, ``save_state``, ``load_state``) so ``@persist(persistence=...)``
   CrewAI Flows checkpoint their structured/dict state into Conflux.
3. :func:`create_crewai_tools` — returns drop-in tools (native CrewAI ``BaseTool``
   subclasses when ``crewai`` is installed, or callable tool descriptors otherwise)
   giving CrewAI agents lock-free access to shared CRDT registers, counters, and sets.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .base import BackendAdapter, coerce_backend

try:  # Optional CrewAI runtime types
    from crewai.flow.persistence.base import FlowPersistence as _CrewFlowPersistence
    from crewai.memory.types import MemoryRecord as _CrewMemoryRecord
    from crewai.memory.types import ScopeInfo as _CrewScopeInfo
    from crewai.tools import BaseTool as _CrewBaseTool
    from pydantic import PrivateAttr

    _HAS_CREWAI = True
except ImportError:  # pragma: no cover - standard when crewai is not installed
    _HAS_CREWAI = False
    _CrewFlowPersistence = None
    _CrewMemoryRecord = None
    _CrewScopeInfo = None
    _CrewBaseTool = None
    PrivateAttr = None


@dataclass
class FallbackMemoryRecord:
    """Lightweight stand-in for ``crewai.memory.types.MemoryRecord`` when CrewAI is not installed."""

    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    scope: str = "/"
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: list[float] | None = None
    source: str | None = None
    private: bool = False


@dataclass
class FallbackScopeInfo:
    """Lightweight stand-in for ``crewai.memory.types.ScopeInfo`` when CrewAI is not installed."""

    path: str
    record_count: int = 0
    categories: list[str] = field(default_factory=list)
    oldest_record: datetime | None = None
    newest_record: datetime | None = None
    child_scopes: list[str] = field(default_factory=list)


def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _normalize_scope(scope: str | None) -> str:
    if not scope or scope == "/":
        return "/"
    s = "/" + scope.strip("/")
    return s


def _scope_matches(record_scope: str, scope_prefix: str | None) -> bool:
    if not scope_prefix or scope_prefix == "/":
        return True
    norm_prefix = _normalize_scope(scope_prefix)
    norm_rec = _normalize_scope(record_scope)
    return norm_rec == norm_prefix or norm_rec.startswith(norm_prefix + "/")


def _to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class ConfluxCrewStorage:
    """CrewAI ``StorageBackend`` implementation backed by Conflux CRDTs.

    Key layout:
    - ``{prefix}ids`` -> ``ORSet`` of active ``record_id`` strings
    - ``{prefix}rec/{record_id}`` -> ``LWWRegister`` holding the serialized record dict
    """

    def __init__(self, backend: Any, *, prefix: str = "crew.mem/"):
        self.backend: BackendAdapter = coerce_backend(backend)
        self.prefix = prefix if prefix.endswith("/") else f"{prefix}/"

    def _ids_key(self) -> str:
        return f"{self.prefix}ids"

    def _rec_key(self, record_id: str) -> str:
        return f"{self.prefix}rec/{record_id}"

    def _serialize_record(self, rec: Any) -> dict[str, Any]:
        if isinstance(rec, dict):
            record_id = str(rec.get("id") or uuid4())
            created = rec.get("created_at")
            accessed = rec.get("last_accessed")
            return {
                "id": record_id,
                "content": str(rec["content"]),
                "scope": _normalize_scope(rec.get("scope", "/")),
                "categories": list(rec.get("categories") or []),
                "metadata": dict(rec.get("metadata") or {}),
                "importance": float(rec.get("importance", 0.5)),
                "created_at": created.isoformat() if isinstance(created, datetime) else (created or datetime.utcnow().isoformat()),
                "last_accessed": accessed.isoformat() if isinstance(accessed, datetime) else (accessed or datetime.utcnow().isoformat()),
                "embedding": list(rec["embedding"]) if rec.get("embedding") is not None else None,
                "source": rec.get("source"),
                "private": bool(rec.get("private", False)),
                "deleted": False,
            }
        created = getattr(rec, "created_at", None) or datetime.utcnow()
        accessed = getattr(rec, "last_accessed", None) or datetime.utcnow()
        emb = getattr(rec, "embedding", None)
        return {
            "id": str(getattr(rec, "id", None) or uuid4()),
            "content": str(rec.content),
            "scope": _normalize_scope(getattr(rec, "scope", "/")),
            "categories": list(getattr(rec, "categories", []) or []),
            "metadata": dict(getattr(rec, "metadata", {}) or {}),
            "importance": float(getattr(rec, "importance", 0.5)),
            "created_at": created.isoformat() if isinstance(created, datetime) else str(created),
            "last_accessed": accessed.isoformat() if isinstance(accessed, datetime) else str(accessed),
            "embedding": [float(x) for x in emb] if emb is not None else None,
            "source": getattr(rec, "source", None),
            "private": bool(getattr(rec, "private", False)),
            "deleted": False,
        }

    def _deserialize_record(self, data: dict[str, Any]) -> Any:
        cls = _CrewMemoryRecord if _HAS_CREWAI and _CrewMemoryRecord is not None else FallbackMemoryRecord
        return cls(
            id=data["id"],
            content=data["content"],
            scope=data.get("scope", "/"),
            categories=list(data.get("categories", [])),
            metadata=dict(data.get("metadata", {})),
            importance=float(data.get("importance", 0.5)),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            embedding=data.get("embedding"),
            source=data.get("source"),
            private=bool(data.get("private", False)),
        )

    def _iter_active_dicts(self) -> list[dict[str, Any]]:
        out = []
        for rec_id in sorted(self.backend.read_set(self._ids_key())):
            raw = self.backend.read(self._rec_key(str(rec_id)))
            if isinstance(raw, dict) and not raw.get("deleted"):
                out.append(raw)
        return out

    # ------------------------------------------------------------------
    # CrewAI StorageBackend Protocol Methods
    # ------------------------------------------------------------------
    def save(self, records: list[Any]) -> None:
        for rec in records:
            payload = self._serialize_record(rec)
            rec_id = payload["id"]
            self.backend.register_set(self._rec_key(rec_id), payload)
            self.backend.set_add(self._ids_key(), rec_id, tag=f"cmem:{rec_id}")

    def update(self, record: Any) -> None:
        self.save([record])

    def get_record(self, record_id: str) -> Any | None:
        raw = self.backend.read(self._rec_key(str(record_id)))
        if not isinstance(raw, dict) or raw.get("deleted"):
            return None
        return self._deserialize_record(raw)

    def search(
        self,
        query_embedding: list[float] | str,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        scored: list[tuple[Any, float]] = []
        is_text_query = isinstance(query_embedding, str)
        q_tokens = [t for t in query_embedding.lower().split() if t] if is_text_query else []

        for raw in self._iter_active_dicts():
            if not _scope_matches(raw.get("scope", "/"), scope_prefix):
                continue
            if categories:
                rec_cats = set(raw.get("categories") or [])
                if not rec_cats.intersection(categories):
                    continue
            if metadata_filter:
                meta = raw.get("metadata") or {}
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue

            if is_text_query:
                if not q_tokens:
                    score = 1.0
                else:
                    haystack = (raw.get("content", "") + " " + json.dumps(raw.get("metadata", {}))).lower()
                    hits = sum(1 for tok in q_tokens if tok in haystack)
                    score = hits / len(q_tokens)
            elif query_embedding and raw.get("embedding"):
                score = _cosine_similarity(list(query_embedding), raw["embedding"])
            else:
                score = float(raw.get("importance", 0.5))

            if score >= min_score and (not is_text_query or not q_tokens or score > 0):
                scored.append((self._deserialize_record(raw), round(score, 6)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        deleted_count = 0
        id_filter = set(record_ids) if record_ids is not None else None
        cutoff = _to_utc_naive(older_than) if older_than is not None else None

        for raw in self._iter_active_dicts():
            rec_id = raw["id"]
            if id_filter is not None and rec_id not in id_filter:
                continue
            if not _scope_matches(raw.get("scope", "/"), scope_prefix):
                continue
            if categories and not set(raw.get("categories") or []).intersection(categories):
                continue
            if metadata_filter:
                meta = raw.get("metadata") or {}
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            if cutoff is not None:
                created = _to_utc_naive(datetime.fromisoformat(raw["created_at"]))
                if created >= cutoff:
                    continue

            tombstone = dict(raw, deleted=True)
            self.backend.register_set(self._rec_key(rec_id), tombstone)
            self.backend.set_remove(self._ids_key(), f"cmem:{rec_id}")
            deleted_count += 1
        return deleted_count

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Any]:
        matching = [
            self._deserialize_record(raw)
            for raw in self._iter_active_dicts()
            if _scope_matches(raw.get("scope", "/"), scope_prefix)
        ]
        matching.sort(key=lambda r: _to_utc_naive(r.created_at), reverse=True)
        return matching[offset : offset + limit]

    def get_scope_info(self, scope: str) -> Any:
        norm = _normalize_scope(scope)
        records = [
            self._deserialize_record(raw)
            for raw in self._iter_active_dicts()
            if _scope_matches(raw.get("scope", "/"), norm)
        ]
        cats = sorted({c for r in records for c in (r.categories or [])})
        dates = [r.created_at for r in records] if records else []
        children = self.list_scopes(parent=norm)
        cls = _CrewScopeInfo if _HAS_CREWAI and _CrewScopeInfo is not None else FallbackScopeInfo
        return cls(
            path=norm,
            record_count=len(records),
            categories=cats,
            oldest_record=min(dates) if dates else None,
            newest_record=max(dates) if dates else None,
            child_scopes=children,
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        norm_parent = _normalize_scope(parent)
        children: set[str] = set()
        prefix = "/" if norm_parent == "/" else norm_parent + "/"
        for raw in self._iter_active_dicts():
            sc = _normalize_scope(raw.get("scope", "/"))
            if sc == norm_parent or not sc.startswith(prefix):
                continue
            remainder = sc[len(prefix) :]
            first_seg = remainder.split("/", 1)[0]
            if first_seg:
                child = ("/" + first_seg) if norm_parent == "/" else f"{norm_parent}/{first_seg}"
                children.add(child)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for raw in self._iter_active_dicts():
            if not _scope_matches(raw.get("scope", "/"), scope_prefix):
                continue
            for cat in raw.get("categories") or []:
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        return sum(
            1
            for raw in self._iter_active_dicts()
            if _scope_matches(raw.get("scope", "/"), scope_prefix)
        )

    def reset(self, scope_prefix: str | None = None) -> None:
        self.delete(scope_prefix=scope_prefix)

    async def asave(self, records: list[Any]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float] | str,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.delete,
            scope_prefix,
            categories,
            record_ids,
            older_than,
            metadata_filter,
        )


# ----------------------------------------------------------------------
# CrewAI FlowPersistence adapter
# ----------------------------------------------------------------------
if _HAS_CREWAI and _CrewFlowPersistence is not None:

    class ConfluxFlowPersistence(_CrewFlowPersistence):
        """CrewAI ``FlowPersistence`` adapter backed by Conflux CRDT state."""

        persistence_type: str = "conflux"
        prefix: str = "crew.flow/"
        _backend: BackendAdapter = PrivateAttr()

        def __init__(self, backend: Any, *, prefix: str = "crew.flow/", **data: Any):
            super().__init__(prefix=prefix if prefix.endswith("/") else f"{prefix}/", **data)
            self._backend = coerce_backend(backend)
            self.init_db()

        def init_db(self) -> None:
            return None

        def save_state(self, flow_uuid: str, method_name: str, state_data: Any) -> None:
            if hasattr(state_data, "model_dump"):
                state_dict = state_data.model_dump(mode="json")
            elif isinstance(state_data, dict):
                state_dict = dict(state_data)
            else:
                state_dict = dict(vars(state_data))
            envelope = {
                "flow_uuid": flow_uuid,
                "method_name": method_name,
                "state": state_dict,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._backend.register_set(f"{self.prefix}{flow_uuid}", envelope)

        def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
            envelope = self._backend.read(f"{self.prefix}{flow_uuid}")
            if not isinstance(envelope, dict):
                return None
            return dict(envelope.get("state") or {})

else:

    class ConfluxFlowPersistence:  # type: ignore[no-redef]
        """CrewAI ``FlowPersistence``-compatible adapter backed by Conflux CRDT state."""

        persistence_type = "conflux"

        def __init__(self, backend: Any, *, prefix: str = "crew.flow/"):
            self._backend: BackendAdapter = coerce_backend(backend)
            self.prefix = prefix if prefix.endswith("/") else f"{prefix}/"
            self.init_db()

        def init_db(self) -> None:
            return None

        def save_state(self, flow_uuid: str, method_name: str, state_data: Any) -> None:
            if hasattr(state_data, "model_dump"):
                state_dict = state_data.model_dump(mode="json")
            elif isinstance(state_data, dict):
                state_dict = dict(state_data)
            else:
                state_dict = dict(vars(state_data))
            envelope = {
                "flow_uuid": flow_uuid,
                "method_name": method_name,
                "state": state_dict,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._backend.register_set(f"{self.prefix}{flow_uuid}", envelope)

        def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
            envelope = self._backend.read(f"{self.prefix}{flow_uuid}")
            if not isinstance(envelope, dict):
                return None
            return dict(envelope.get("state") or {})


# ----------------------------------------------------------------------
# CrewAI Agent Tools for shared CRDT state
# ----------------------------------------------------------------------
class ConfluxCallableTool:
    """Zero-dependency callable tool wrapper exposing ``name``, ``description``, and ``run(...)``."""

    def __init__(self, name: str, description: str, fn: Any):
        self.name = name
        self.description = description
        self._fn = fn

    def run(self, *args: Any, **kwargs: Any) -> str:
        return self._fn(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self._fn(*args, **kwargs)


def create_crewai_tools(backend: Any, *, namespace: str = "crew.") -> list[Any]:
    """Build CrewAI tools for reading and mutating shared Conflux CRDT state.

    When ``crewai`` is installed, returns native ``crewai.tools.BaseTool`` instances
    that plug directly into ``Agent(tools=create_crewai_tools(client))``. When
    ``crewai`` is not installed, returns :class:`ConfluxCallableTool` objects with
    the identical ``.name``, ``.description``, and ``.run(...)`` interface.
    """
    adapter = coerce_backend(backend)
    ns = namespace

    def _qualify(key: str) -> str:
        if not ns or key.startswith(ns):
            return key
        return f"{ns}{key}"

    def read_state(key: str) -> str:
        """Read a shared key (register, counter, or set) from Conflux."""
        val = adapter.read(_qualify(key))
        if isinstance(val, (set, frozenset)):
            val = sorted(val)
        return json.dumps({"key": _qualify(key), "value": val})

    def write_register(key: str, value: Any) -> str:
        """Set a Last-Writer-Wins (LWW) register in shared Conflux state."""
        qkey = _qualify(key)
        adapter.register_set(qkey, value)
        return json.dumps({"status": "ok", "key": qkey, "value": adapter.read(qkey)})

    def increment_counter(key: str, by: int = 1) -> str:
        """Increment or decrement (if negative) a lock-free shared PNCounter in Conflux."""
        qkey = _qualify(key)
        amount = int(by)
        if amount >= 0:
            adapter.counter_inc(qkey, amount)
        else:
            adapter.counter_dec(qkey, abs(amount))
        return json.dumps({"status": "ok", "key": qkey, "total": adapter.read(qkey)})

    def add_to_set(key: str, item: Any) -> str:
        """Add an element to a shared Observed-Remove Set (ORSet) in Conflux."""
        qkey = _qualify(key)
        adapter.set_add(qkey, item)
        members = sorted(str(x) for x in adapter.read_set(qkey))
        return json.dumps({"status": "ok", "key": qkey, "members": members})

    specs = [
        (
            "conflux_read_state",
            "Read a shared key (counter, LWW register, or ORSet) from the Conflux CRDT mesh.",
            read_state,
        ),
        (
            "conflux_write_register",
            "Write a JSON-serializable value to a shared LWW register in Conflux.",
            write_register,
        ),
        (
            "conflux_increment_counter",
            "Atomically increment (or decrement when by < 0) a lock-free shared counter in Conflux.",
            increment_counter,
        ),
        (
            "conflux_add_to_set",
            "Add a member to a conflict-free shared ORSet in Conflux.",
            add_to_set,
        ),
    ]

    if _HAS_CREWAI and _CrewBaseTool is not None:
        from crewai.tools import tool as crew_tool_decorator

        tools = []
        for name, desc, fn in specs:
            fn.__doc__ = desc
            wrapped = crew_tool_decorator(name)(fn)
            tools.append(wrapped)
        return tools

    return [ConfluxCallableTool(name, desc, fn) for name, desc, fn in specs]
