"""AutoGen (v0.4+ / v0.7+ AgentChat & Core) drop-in adapters backed by Conflux.

Provides two integrations for AutoGen multi-agent teams:

1. :class:`ConfluxAutoGenMemory` — implements AutoGen's ``autogen_core.memory.Memory``
   interface (``add``, ``query``, ``update_context``, ``clear``, ``close``) over
   Conflux's replicated CRDT state so ``AssistantAgent(..., memory=[ConfluxAutoGenMemory(node)])``
   automatically injects converged multi-agent facts into every agent's LLM context.
2. :func:`create_autogen_tools` — produces ``FunctionTool`` instances (or plain
   typed functions when ``autogen-core`` is not installed) for ``AssistantAgent(tools=...)``
   so AutoGen agents can concurrently mutate shared CRDT counters, LWW registers,
   and Observed-Remove Sets without locks or central coordinators.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .base import BackendAdapter, coerce_backend

try:  # Optional AutoGen v0.4+/v0.7+ runtime classes
    from autogen_core.memory import (
        Memory as _AutoGenMemoryBase,
        MemoryContent,
        MemoryMimeType,
        MemoryQueryResult,
        UpdateContextResult,
    )
    from autogen_core.models import SystemMessage
    from autogen_core.tools import FunctionTool

    _HAS_AUTOGEN = True
except ImportError:  # pragma: no cover - standard when autogen-core is not installed
    _HAS_AUTOGEN = False
    _AutoGenMemoryBase = object  # type: ignore[assignment,misc]
    FunctionTool = None  # type: ignore[assignment,misc]

    @dataclass
    class MemoryContent:  # type: ignore[no-redef]
        content: Any
        mime_type: str = "text/plain"
        metadata: dict[str, Any] | None = field(default_factory=dict)

    class MemoryMimeType:  # type: ignore[no-redef]
        TEXT = "text/plain"
        JSON = "application/json"
        MARKDOWN = "text/markdown"

    @dataclass
    class MemoryQueryResult:  # type: ignore[no-redef]
        results: list[MemoryContent]

    @dataclass
    class UpdateContextResult:  # type: ignore[no-redef]
        memories: MemoryQueryResult

    @dataclass
    class SystemMessage:  # type: ignore[no-redef]
        content: str


class ConfluxAutoGenMemory(_AutoGenMemoryBase):  # type: ignore[misc]
    """Drop-in ``autogen_core.memory.Memory`` backed by Conflux's CRDT fabric.

    Memories added on any Conflux node are stored as deterministic CRDT entries:
    - ``{prefix}{name}/ids`` -> ``ORSet`` of active entry IDs
    - ``{prefix}{name}/item/{entry_id}`` -> ``LWWRegister`` with content, mime_type,
      metadata, and timestamp.

    When ``update_context(model_context)`` runs before an AutoGen turn, all
    converged memories (plus optional live shared-state snapshot keys) are
    appended to the agent's ``ChatCompletionContext`` as a ``SystemMessage``.
    """

    component_type = "memory"

    def __init__(
        self,
        backend: Any,
        *,
        name: str = "shared_team_memory",
        prefix: str = "ag.mem/",
        state_watch_prefix: str | None = None,
    ) -> None:
        self.backend: BackendAdapter = coerce_backend(backend)
        self._name = name
        base = prefix if prefix.endswith("/") else f"{prefix}/"
        self.namespace_prefix = f"{base}{name}/"
        self.state_watch_prefix = state_watch_prefix

    @property
    def name(self) -> str:
        return self._name

    def _ids_key(self) -> str:
        return f"{self.namespace_prefix}ids"

    def _item_key(self, entry_id: str) -> str:
        return f"{self.namespace_prefix}item/{entry_id}"

    def _load_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for entry_id in sorted(self.backend.read_set(self._ids_key())):
            raw = self.backend.read(self._item_key(str(entry_id)))
            if isinstance(raw, dict) and not raw.get("deleted"):
                entries.append(raw)
        entries.sort(key=lambda e: (e.get("created_at", ""), e.get("id", "")))
        return entries

    def _to_memory_content(self, raw: dict[str, Any]) -> MemoryContent:
        mime = raw.get("mime_type", "text/plain")
        return MemoryContent(
            content=raw["content"],
            mime_type=mime,
            metadata=dict(raw.get("metadata") or {}),
        )

    @property
    def content(self) -> list[MemoryContent]:
        return [self._to_memory_content(e) for e in self._load_entries()]

    def add_sync(self, content: MemoryContent | str, metadata: dict[str, Any] | None = None) -> str:
        if isinstance(content, str):
            body: Any = content
            mime_str = "text/plain"
            meta = dict(metadata or {})
        else:
            body = content.content
            mime = getattr(content, "mime_type", "text/plain")
            mime_str = mime.value if hasattr(mime, "value") else str(mime)
            meta = dict(getattr(content, "metadata", None) or {})
            if metadata:
                meta.update(metadata)

        entry_id = str(meta.get("id") or uuid4())
        record = {
            "id": entry_id,
            "content": body,
            "mime_type": mime_str,
            "metadata": meta,
            "agent": self.backend.agent_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deleted": False,
        }
        self.backend.register_set(self._item_key(entry_id), record)
        self.backend.set_add(self._ids_key(), entry_id, tag=f"agm:{entry_id}")
        return entry_id

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Any = None,
    ) -> None:
        _ = cancellation_token
        await asyncio.to_thread(self.add_sync, content)

    def query_sync(self, query: str | MemoryContent = "") -> list[MemoryContent]:
        q_str = query if isinstance(query, str) else str(query.content)
        tokens = [t for t in q_str.lower().split() if t]
        entries = self._load_entries()
        if not tokens:
            return [self._to_memory_content(e) for e in entries]

        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in entries:
            haystack = (
                str(entry.get("content", "")) + " " + json.dumps(entry.get("metadata") or {})
            ).lower()
            hits = sum(1 for tok in tokens if tok in haystack)
            if hits > 0:
                scored.append((hits / len(tokens), entry))
        scored.sort(key=lambda pair: (pair[0], pair[1].get("created_at", "")), reverse=True)
        return [self._to_memory_content(e) for _, e in scored]

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: Any = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        _ = cancellation_token, kwargs
        results = await asyncio.to_thread(self.query_sync, query)
        return MemoryQueryResult(results=results)

    async def update_context(
        self,
        model_context: Any,
    ) -> UpdateContextResult:
        memories = await asyncio.to_thread(lambda: self.content)
        state_snapshot = (
            await asyncio.to_thread(self.backend.view, self.state_watch_prefix)
            if self.state_watch_prefix
            else {}
        )

        lines: list[str] = []
        if memories:
            lines.append("Shared Conflux memory (converged across agents):")
            for idx, mem in enumerate(memories, 1):
                lines.append(f"{idx}. {mem.content}")

        if state_snapshot:
            lines.append("Shared Conflux CRDT state:")
            for k in sorted(state_snapshot):
                val = state_snapshot[k]
                if isinstance(val, (set, frozenset)):
                    val = sorted(val)
                lines.append(f"- {k}: {json.dumps(val)}")

        if lines:
            text = "\n" + "\n".join(lines) + "\n"
            await model_context.add_message(SystemMessage(content=text))

        return UpdateContextResult(memories=MemoryQueryResult(results=memories))

    async def clear(self) -> None:
        def _clear():
            for entry in self._load_entries():
                eid = entry["id"]
                self.backend.register_set(self._item_key(eid), dict(entry, deleted=True))
                self.backend.set_remove(self._ids_key(), f"agm:{eid}")

        await asyncio.to_thread(_clear)

    async def close(self) -> None:
        return None


def create_autogen_tools(backend: Any, *, namespace: str = "team.") -> list[Any]:
    """Create AutoGen ``FunctionTool`` objects for shared Conflux CRDT mutations.

    When ``autogen-core`` is installed, returns ``list[FunctionTool]`` ready to pass
    into ``AssistantAgent(..., tools=create_autogen_tools(node))``. When ``autogen-core``
    is not installed, returns the underlying typed Python callables directly.
    """
    adapter = coerce_backend(backend)
    ns = namespace

    def _qualify(key: str) -> str:
        if not ns or key.startswith(ns):
            return key
        return f"{ns}{key}"

    def conflux_get_state(key: str) -> str:
        """Read a shared key (counter, LWW register, or ORSet) from Conflux."""
        qkey = _qualify(key)
        val = adapter.read(qkey)
        if isinstance(val, (set, frozenset)):
            val = sorted(val)
        return json.dumps({"key": qkey, "value": val})

    def conflux_set_register(key: str, value: str) -> str:
        """Write a string or JSON value to a deterministic LWW register in Conflux."""
        qkey = _qualify(key)
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = value
        adapter.register_set(qkey, parsed)
        return json.dumps({"status": "ok", "key": qkey, "value": adapter.read(qkey)})

    def conflux_increment_counter(key: str, by: int = 1) -> str:
        """Increment or decrement (when by < 0) a shared lock-free counter in Conflux."""
        qkey = _qualify(key)
        if by >= 0:
            adapter.counter_inc(qkey, by)
        else:
            adapter.counter_dec(qkey, abs(by))
        return json.dumps({"status": "ok", "key": qkey, "total": adapter.read(qkey)})

    def conflux_add_set_member(key: str, member: str) -> str:
        """Add an element to a conflict-free Observed-Remove Set (ORSet) in Conflux."""
        qkey = _qualify(key)
        adapter.set_add(qkey, member)
        members = sorted(str(x) for x in adapter.read_set(qkey))
        return json.dumps({"status": "ok", "key": qkey, "members": members})

    def conflux_snapshot(prefix: str = "") -> str:
        """Return a JSON snapshot of all shared Conflux state under the namespace."""
        search_prefix = _qualify(prefix) if prefix else ns
        view = adapter.view(prefix=search_prefix or None)
        serializable = {
            k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
            for k, v in sorted(view.items())
        }
        return json.dumps({"state_hash": adapter.hash(), "state": serializable})

    raw_fns = [
        (
            conflux_get_state,
            "Read a shared key (counter, register, or set) from the Conflux CRDT state backend.",
        ),
        (
            conflux_set_register,
            "Write a JSON or string value into a deterministic Last-Writer-Wins register in Conflux.",
        ),
        (
            conflux_increment_counter,
            "Atomically increment or decrement a lock-free shared counter in Conflux.",
        ),
        (
            conflux_add_set_member,
            "Add a member to a conflict-free Observed-Remove Set in Conflux.",
        ),
        (
            conflux_snapshot,
            "Inspect all shared Conflux state keys and the canonical SHA-256 state hash.",
        ),
    ]

    if _HAS_AUTOGEN and FunctionTool is not None:
        return [FunctionTool(fn, description=desc, name=fn.__name__) for fn, desc in raw_fns]

    return [fn for fn, _ in raw_fns]
