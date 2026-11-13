"""LangGraph drop-in adapters backed by Conflux's CRDT state fabric.

Provides two integrations:

1. :class:`ConfluxCheckpointSaver` — implements LangGraph's
   ``BaseCheckpointSaver`` so compiled ``StateGraph`` workflows persist, fork,
   resume, and replicate checkpoints across a Conflux node or multi-node mesh.
2. :class:`ConfluxStore` — implements LangGraph's cross-thread ``BaseStore``
   (``put``, ``get``, ``delete``, ``search``, ``list_namespaces``, plus CRDT
   counter/set helpers) so independent LangGraph agents share lock-free state.

Both classes subclass LangGraph's native base classes automatically when
``langgraph`` is installed, and remain usable standalone with zero third-party
dependencies when ``langgraph`` is absent.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from .base import BackendAdapter, coerce_backend

try:  # Optional LangGraph dependency
    from langgraph.checkpoint.base import (
        WRITES_IDX_MAP,
        BaseCheckpointSaver,
        CheckpointTuple,
        get_checkpoint_id,
        get_checkpoint_metadata,
    )
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        Item,
        ListNamespacesOp,
        PutOp,
        SearchItem,
        SearchOp,
    )

    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover - exercised in zero-dep environments
    _HAS_LANGGRAPH = False
    WRITES_IDX_MAP = {}

    class CheckpointTuple:  # type: ignore[no-redef]
        def __init__(self, config, checkpoint, metadata, parent_config=None, pending_writes=None):
            self.config = config
            self.checkpoint = checkpoint
            self.metadata = metadata
            self.parent_config = parent_config
            self.pending_writes = pending_writes

    class Item:  # type: ignore[no-redef]
        def __init__(self, *, value, key, namespace, created_at, updated_at):
            self.value = value
            self.key = key
            self.namespace = namespace
            self.created_at = created_at
            self.updated_at = updated_at

    class SearchItem(Item):  # type: ignore[no-redef]
        def __init__(self, namespace, key, value, created_at, updated_at, score=None):
            super().__init__(
                value=value,
                key=key,
                namespace=namespace,
                created_at=created_at,
                updated_at=updated_at,
            )
            self.score = score

    def get_checkpoint_id(config):  # type: ignore[misc]
        return (config or {}).get("configurable", {}).get("checkpoint_id")

    def get_checkpoint_metadata(config, metadata):  # type: ignore[misc]
        return dict(metadata or {})

    class BaseCheckpointSaver:  # type: ignore[no-redef]
        serde = None

        def __init__(self, serde=None):
            self.serde = serde

        def get_next_version(self, current, channel=None):
            if current is None:
                return 1
            if isinstance(current, int):
                return current + 1
            if isinstance(current, str):
                prefix = current.split(".")[0]
                if prefix.isdigit():
                    return f"{int(prefix) + 1:032d}.0"
            return str(current) + ".1"

    class BaseStore:  # type: ignore[no-redef]
        pass


def _encode_typed(serde: Any, obj: Any) -> dict[str, str]:
    """Encode an arbitrary value via LangGraph's serde (if present) or JSON into a wire-safe dict."""
    if serde is not None:
        type_str, raw_bytes = serde.dumps_typed(obj)
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")
        return {"t": type_str, "b": base64.b64encode(raw_bytes).decode("ascii")}
    return {"t": "json", "b": json.dumps(obj, sort_keys=True)}


def _decode_typed(serde: Any, payload: dict[str, str]) -> Any:
    if serde is not None and payload.get("t") != "json":
        raw = base64.b64decode(payload["b"].encode("ascii"))
        return serde.loads_typed((payload["t"], raw))
    if payload.get("t") == "json":
        return json.loads(payload["b"])
    raw = base64.b64decode(payload["b"].encode("ascii"))
    return json.loads(raw.decode("utf-8"))


def _safe_segment(text: str) -> str:
    """Encode arbitrary string segments safely for Conflux keys without collisions."""
    if not text:
        return "~empty~"
    return (
        text.replace("%", "%25")
        .replace("/", "%2F")
        .replace(":", "%3A")
    )


def _unsafe_segment(text: str) -> str:
    if text == "~empty~":
        return ""
    return (
        text.replace("%3A", ":")
        .replace("%2F", "/")
        .replace("%25", "%")
    )


class ConfluxCheckpointSaver(BaseCheckpointSaver):
    """LangGraph ``BaseCheckpointSaver`` backed by Conflux's deterministic CRDT fabric.

    Key layout (all within ``MAX_KEY_LENGTH`` and Conflux's schema limits):
    - ``{prefix}threads`` -> ``ORSet`` of ``"{thread_id}/{checkpoint_ns}"``
    - ``{prefix}idx/{thread}/{ns}`` -> ``ORSet`` of ``checkpoint_id``
    - ``{prefix}ckpt/{thread}/{ns}/{ckpt_id}`` -> ``LWWRegister`` holding
      checkpoint envelope + encoded channel values + parent id
    - ``{prefix}writes/{thread}/{ns}/{ckpt_id}`` -> ``ORSet`` of JSON-encoded write entries
    """

    def __init__(self, backend: Any, *, prefix: str = "lg.ckpt/", serde: Any = None):
        super().__init__(serde=serde)
        self.backend: BackendAdapter = coerce_backend(backend)
        self.prefix = prefix if prefix.endswith("/") else f"{prefix}/"

    def _threads_key(self) -> str:
        return f"{self.prefix}threads"

    def _idx_key(self, thread_id: str, ns: str) -> str:
        return f"{self.prefix}idx/{_safe_segment(thread_id)}/{_safe_segment(ns)}"

    def _ckpt_key(self, thread_id: str, ns: str, ckpt_id: str) -> str:
        return (
            f"{self.prefix}ckpt/{_safe_segment(thread_id)}/"
            f"{_safe_segment(ns)}/{_safe_segment(ckpt_id)}"
        )

    def _writes_key(self, thread_id: str, ns: str, ckpt_id: str) -> str:
        return (
            f"{self.prefix}writes/{_safe_segment(thread_id)}/"
            f"{_safe_segment(ns)}/{_safe_segment(ckpt_id)}"
        )

    # ------------------------------------------------------------------
    # Sync CheckpointSaver interface
    # ------------------------------------------------------------------
    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(checkpoint["id"])
        parent_checkpoint_id = configurable.get("checkpoint_id")

        ckpt_copy = dict(checkpoint)
        channel_values: dict[str, Any] = dict(ckpt_copy.pop("channel_values", {}))

        # Preserve channel values across versions: carry forward prior blobs if unchanged
        encoded_channels: dict[str, dict[str, str]] = {}
        if parent_checkpoint_id:
            parent_raw = self.backend.read(self._ckpt_key(thread_id, checkpoint_ns, str(parent_checkpoint_id)))
            if isinstance(parent_raw, dict):
                encoded_channels.update(parent_raw.get("channels", {}))

        for k in ckpt_copy.get("channel_versions", {}).keys():
            if k in channel_values:
                encoded_channels[k] = _encode_typed(self.serde, channel_values[k])
            elif k in new_versions:
                encoded_channels.pop(k, None)

        resolved_meta = get_checkpoint_metadata(config, metadata)
        record = {
            "checkpoint": _encode_typed(self.serde, ckpt_copy),
            "channels": encoded_channels,
            "metadata": _encode_typed(self.serde, resolved_meta),
            "parent_checkpoint_id": parent_checkpoint_id,
        }

        self.backend.register_set(self._ckpt_key(thread_id, checkpoint_ns, checkpoint_id), record)
        thread_pair = f"{_safe_segment(thread_id)}/{_safe_segment(checkpoint_ns)}"
        self.backend.set_add(self._threads_key(), thread_pair, tag=f"th:{thread_pair}")
        self.backend.set_add(
            self._idx_key(thread_id, checkpoint_ns),
            checkpoint_id,
            tag=f"ck:{_safe_segment(thread_id)}:{_safe_segment(checkpoint_ns)}:{_safe_segment(checkpoint_id)}",
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable["checkpoint_id"])
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)

        existing_entries = self._read_writes_raw(thread_id, checkpoint_ns, checkpoint_id)
        existing_keys = {(e["task_id"], e["idx"]) for e in existing_entries}

        for idx, (channel, val) in enumerate(writes):
            mapped_idx = WRITES_IDX_MAP.get(channel, idx)
            if mapped_idx >= 0 and (task_id, mapped_idx) in existing_keys:
                continue
            entry = {
                "task_id": task_id,
                "idx": mapped_idx,
                "channel": channel,
                "value": _encode_typed(self.serde, val),
                "task_path": task_path,
            }
            serialized = json.dumps(entry, sort_keys=True)
            tag = f"wr:{_safe_segment(checkpoint_id)}:{_safe_segment(task_id)}:{mapped_idx}"
            self.backend.set_add(writes_key, serialized, tag=tag)

    def _read_writes_raw(self, thread_id: str, ns: str, ckpt_id: str) -> list[dict[str, Any]]:
        raw_set = self.backend.read_set(self._writes_key(thread_id, ns, ckpt_id))
        parsed = []
        for item in raw_set:
            if isinstance(item, str):
                parsed.append(json.loads(item))
        parsed.sort(key=lambda e: (e["task_path"], e["task_id"], e["idx"]))
        return parsed

    def _hydrate_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        raw_record: dict[str, Any],
        requested_config: dict[str, Any] | None = None,
    ) -> CheckpointTuple:
        ckpt_base = _decode_typed(self.serde, raw_record["checkpoint"])
        channels_encoded = raw_record.get("channels", {})
        channel_values = {}
        for ch_name in ckpt_base.get("channel_versions", {}).keys():
            if ch_name in channels_encoded:
                channel_values[ch_name] = _decode_typed(self.serde, channels_encoded[ch_name])

        full_checkpoint = {**ckpt_base, "channel_values": channel_values}
        metadata = _decode_typed(self.serde, raw_record["metadata"])
        parent_id = raw_record.get("parent_checkpoint_id")

        writes_raw = self._read_writes_raw(thread_id, checkpoint_ns, checkpoint_id)
        pending_writes = [
            (w["task_id"], w["channel"], _decode_typed(self.serde, w["value"]))
            for w in writes_raw
        ]

        out_config = requested_config if (requested_config and get_checkpoint_id(requested_config)) else {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )
        return CheckpointTuple(
            config=out_config,
            checkpoint=full_checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id:
            raw = self.backend.read(self._ckpt_key(thread_id, checkpoint_ns, str(checkpoint_id)))
            if not isinstance(raw, dict):
                return None
            return self._hydrate_tuple(thread_id, checkpoint_ns, str(checkpoint_id), raw, config)

        ckpt_ids = self.backend.read_set(self._idx_key(thread_id, checkpoint_ns))
        if not ckpt_ids:
            return None
        latest_id = max(str(cid) for cid in ckpt_ids)
        raw = self.backend.read(self._ckpt_key(thread_id, checkpoint_ns, latest_id))
        if not isinstance(raw, dict):
            return None
        return self._hydrate_tuple(thread_id, checkpoint_ns, latest_id, raw)

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        before_id = get_checkpoint_id(before) if before else None

        if config is not None:
            configurable = config.get("configurable", {})
            target_thread = str(configurable["thread_id"])
            target_ns = configurable.get("checkpoint_ns")
            target_ckpt_id = get_checkpoint_id(config)
            pairs = []
            for raw_pair in sorted(self.backend.read_set(self._threads_key())):
                parts = str(raw_pair).split("/", 1)
                if len(parts) != 2:
                    continue
                t_id, ns = _unsafe_segment(parts[0]), _unsafe_segment(parts[1])
                if t_id != target_thread:
                    continue
                if target_ns is not None and ns != target_ns:
                    continue
                pairs.append((t_id, ns))
            if not pairs and target_ns is not None:
                pairs = [(target_thread, str(target_ns))]
            elif not pairs:
                pairs = [(target_thread, "")]
        else:
            target_ckpt_id = None
            pairs = []
            for raw_pair in sorted(self.backend.read_set(self._threads_key())):
                parts = str(raw_pair).split("/", 1)
                if len(parts) == 2:
                    pairs.append((_unsafe_segment(parts[0]), _unsafe_segment(parts[1])))

        remaining = limit
        for thread_id, checkpoint_ns in pairs:
            ckpt_ids = sorted(
                (str(c) for c in self.backend.read_set(self._idx_key(thread_id, checkpoint_ns))),
                reverse=True,
            )
            for ckpt_id in ckpt_ids:
                if target_ckpt_id and ckpt_id != target_ckpt_id:
                    continue
                if before_id and ckpt_id >= str(before_id):
                    continue
                raw = self.backend.read(self._ckpt_key(thread_id, checkpoint_ns, ckpt_id))
                if not isinstance(raw, dict):
                    continue
                tup = self._hydrate_tuple(thread_id, checkpoint_ns, ckpt_id, raw)
                if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                    continue
                if remaining is not None:
                    if remaining <= 0:
                        return
                    remaining -= 1
                yield tup

    def delete_thread(self, thread_id: str) -> None:
        thread_id = str(thread_id)
        for raw_pair in list(self.backend.read_set(self._threads_key())):
            parts = str(raw_pair).split("/", 1)
            if len(parts) != 2:
                continue
            t_id, ns = _unsafe_segment(parts[0]), _unsafe_segment(parts[1])
            if t_id != thread_id:
                continue
            for ckpt_id in list(self.backend.read_set(self._idx_key(t_id, ns))):
                self.backend.register_set(self._ckpt_key(t_id, ns, str(ckpt_id)), None)
                tag = f"ck:{_safe_segment(t_id)}:{_safe_segment(ns)}:{_safe_segment(str(ckpt_id))}"
                self.backend.set_remove(self._idx_key(t_id, ns), tag)
            self.backend.set_remove(self._threads_key(), f"th:{raw_pair}")

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------
    async def aget_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def alist(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item


class ConfluxStore(BaseStore):
    """LangGraph cross-thread ``BaseStore`` backed by Conflux's CRDT state fabric.

    Supports standard LangGraph ``store.put(namespace, key, value)``,
    ``store.get(namespace, key)``, ``store.search(namespace_prefix, query=..., filter=...)``,
    ``store.delete(namespace, key)``, and ``store.list_namespaces(...)``, PLUS
    direct CRDT primitives (:meth:`counter_inc`, :meth:`counter_read`, :meth:`set_add`,
    :meth:`set_members`) for lock-free concurrent aggregation across parallel LangGraph workers.
    """

    def __init__(self, backend: Any, *, prefix: str = "lg.store/"):
        self.backend: BackendAdapter = coerce_backend(backend)
        self.prefix = prefix if prefix.endswith("/") else f"{prefix}/"

    def _ns_str(self, namespace: tuple[str, ...] | list[str]) -> str:
        return "/".join(_safe_segment(str(p)) for p in namespace)

    def _parse_ns(self, encoded: str) -> tuple[str, ...]:
        if not encoded:
            return ()
        return tuple(_unsafe_segment(p) for p in encoded.split("/"))

    def _item_key(self, namespace: tuple[str, ...], key: str) -> str:
        ns_part = self._ns_str(namespace)
        return f"{self.prefix}item/{ns_part}:{_safe_segment(key)}"

    def _index_key(self) -> str:
        return f"{self.prefix}index"

    # ------------------------------------------------------------------
    # Direct CRDT helpers for LangGraph nodes
    # ------------------------------------------------------------------
    def counter_inc(self, namespace: tuple[str, ...], key: str, by: int = 1) -> int:
        full_key = f"{self.prefix}cnt/{self._ns_str(namespace)}:{_safe_segment(key)}"
        self.backend.counter_inc(full_key, by=by)
        return int(self.backend.read(full_key) or 0)

    def counter_dec(self, namespace: tuple[str, ...], key: str, by: int = 1) -> int:
        full_key = f"{self.prefix}cnt/{self._ns_str(namespace)}:{_safe_segment(key)}"
        self.backend.counter_dec(full_key, by=by)
        return int(self.backend.read(full_key) or 0)

    def counter_read(self, namespace: tuple[str, ...], key: str) -> int:
        full_key = f"{self.prefix}cnt/{self._ns_str(namespace)}:{_safe_segment(key)}"
        return int(self.backend.read(full_key) or 0)

    def set_add(self, namespace: tuple[str, ...], key: str, value: Any, tag: str | None = None) -> frozenset:
        full_key = f"{self.prefix}set/{self._ns_str(namespace)}:{_safe_segment(key)}"
        self.backend.set_add(full_key, value, tag=tag)
        return self.backend.read_set(full_key)

    def set_members(self, namespace: tuple[str, ...], key: str) -> frozenset:
        full_key = f"{self.prefix}set/{self._ns_str(namespace)}:{_safe_segment(key)}"
        return self.backend.read_set(full_key)

    # ------------------------------------------------------------------
    # Internal single-op implementations
    # ------------------------------------------------------------------
    def _do_put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any] | None) -> None:
        namespace = tuple(namespace)
        item_key = self._item_key(namespace, key)
        idx_entry = f"{self._ns_str(namespace)}:{_safe_segment(key)}"
        tag = f"st:{idx_entry}"
        now_iso = datetime.now(timezone.utc).isoformat()

        if value is None:
            self.backend.register_set(item_key, {"deleted": True, "updated_at": now_iso})
            self.backend.set_remove(self._index_key(), tag)
            return

        existing = self.backend.read(item_key)
        created_at = (
            existing.get("created_at", now_iso)
            if isinstance(existing, dict) and not existing.get("deleted")
            else now_iso
        )
        record = {
            "namespace": list(namespace),
            "key": key,
            "value": dict(value),
            "created_at": created_at,
            "updated_at": now_iso,
            "deleted": False,
        }
        self.backend.register_set(item_key, record)
        self.backend.set_add(self._index_key(), idx_entry, tag=tag)

    def _do_get(self, namespace: tuple[str, ...], key: str) -> Item | None:
        namespace = tuple(namespace)
        raw = self.backend.read(self._item_key(namespace, key))
        if not isinstance(raw, dict) or raw.get("deleted"):
            return None
        return Item(
            value=dict(raw["value"]),
            key=raw["key"],
            namespace=tuple(raw["namespace"]),
            created_at=datetime.fromisoformat(raw["created_at"]),
            updated_at=datetime.fromisoformat(raw["updated_at"]),
        )

    def _do_search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        prefix = tuple(namespace_prefix)
        matches: list[SearchItem] = []
        q_tokens = [t for t in (query or "").lower().split() if t]

        for entry in sorted(self.backend.read_set(self._index_key())):
            ns_enc, _, key_enc = str(entry).rpartition(":")
            ns = self._parse_ns(ns_enc)
            if len(ns) < len(prefix) or ns[: len(prefix)] != prefix:
                continue
            key = _unsafe_segment(key_enc)
            item = self._do_get(ns, key)
            if item is None:
                continue
            if filter and not all(item.value.get(fk) == fv for fk, fv in filter.items()):
                continue
            score: float | None = None
            if q_tokens:
                haystack = json.dumps(item.value, sort_keys=True).lower()
                hits = sum(1 for tok in q_tokens if tok in haystack)
                if hits == 0:
                    continue
                score = round(hits / len(q_tokens), 4)
            matches.append(
                SearchItem(
                    namespace=item.namespace,
                    key=item.key,
                    value=item.value,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    score=score,
                )
            )

        matches.sort(
            key=lambda s: (
                s.score if s.score is not None else 1.0,
                s.updated_at.timestamp(),
                s.key,
            ),
            reverse=True,
        )
        return matches[offset : offset + limit]

    def _do_list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        seen: set[tuple[str, ...]] = set()
        for entry in self.backend.read_set(self._index_key()):
            ns_enc, _, key_enc = str(entry).rpartition(":")
            ns = self._parse_ns(ns_enc)
            key = _unsafe_segment(key_enc)
            if self._do_get(ns, key) is None:
                continue
            if prefix and (len(ns) < len(prefix) or ns[: len(prefix)] != prefix):
                continue
            if suffix and (len(ns) < len(suffix) or ns[-len(suffix) :] != suffix):
                continue
            if max_depth is not None:
                ns = ns[:max_depth]
            seen.add(ns)
        ordered = sorted(seen)
        return ordered[offset : offset + limit]

    # ------------------------------------------------------------------
    # LangGraph BaseStore batch + convenience methods
    # ------------------------------------------------------------------
    def batch(self, ops: Sequence[Any]) -> list[Any]:
        results: list[Any] = []
        for op in ops:
            if _HAS_LANGGRAPH and isinstance(op, GetOp):
                results.append(self._do_get(tuple(op.namespace), op.key))
            elif _HAS_LANGGRAPH and isinstance(op, PutOp):
                self._do_put(
                    tuple(op.namespace),
                    op.key,
                    dict(op.value) if op.value is not None else None,
                )
                results.append(None)
            elif _HAS_LANGGRAPH and isinstance(op, SearchOp):
                results.append(
                    self._do_search(
                        tuple(op.namespace_prefix),
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            elif _HAS_LANGGRAPH and isinstance(op, ListNamespacesOp):
                prefix = None
                suffix = None
                for cond in op.match_conditions or ():
                    if getattr(cond, "match_type", None) == "prefix":
                        prefix = tuple(cond.path)
                    elif getattr(cond, "match_type", None) == "suffix":
                        suffix = tuple(cond.path)
                results.append(
                    self._do_list_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            else:
                raise TypeError(f"unsupported store operation {type(op)!r}")
        return results

    async def abatch(self, ops: Sequence[Any]) -> list[Any]:
        return await asyncio.to_thread(self.batch, list(ops))

    # Convenience methods (work identically whether or not langgraph is installed)
    if not _HAS_LANGGRAPH:  # pragma: no cover

        def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
            self._do_put(tuple(namespace), key, value)

        def get(self, namespace: tuple[str, ...], key: str) -> Item | None:
            return self._do_get(tuple(namespace), key)

        def delete(self, namespace: tuple[str, ...], key: str) -> None:
            self._do_put(tuple(namespace), key, None)

        def search(
            self,
            namespace_prefix: tuple[str, ...],
            *,
            query: str | None = None,
            filter: dict[str, Any] | None = None,
            limit: int = 10,
            offset: int = 0,
        ) -> list[SearchItem]:
            return self._do_search(
                tuple(namespace_prefix), query=query, filter=filter, limit=limit, offset=offset
            )

        def list_namespaces(
            self,
            *,
            prefix: tuple[str, ...] | None = None,
            suffix: tuple[str, ...] | None = None,
            max_depth: int | None = None,
            limit: int = 100,
            offset: int = 0,
        ) -> list[tuple[str, ...]]:
            return self._do_list_namespaces(
                prefix=prefix, suffix=suffix, max_depth=max_depth, limit=limit, offset=offset
            )
