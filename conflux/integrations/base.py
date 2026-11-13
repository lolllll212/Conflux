"""Backend abstraction unifying Conflux targets (Replica, Server, DistributedApp, Client).

Every framework integration (LangGraph, CrewAI, AutoGen) talks to Conflux
through :class:`BackendAdapter` so developers can swap between an in-process
:class:`~conflux.Replica`, a :class:`~conflux.Cluster` node, a local
:class:`~conflux.Server`, or a networked :class:`~conflux.Client` (TCP/WS +
HMAC/Ed25519 signing) without changing a single line of orchestration code.
"""

from __future__ import annotations

import threading
from typing import Any

from ..client import Client
from ..framework import DistributedApp
from ..protocol import state_hash
from ..server import Server
from ..state import Replica


class BackendAdapter:
    """Uniform read/write facade over any Conflux runtime object."""

    def __init__(self, target: Replica | Server | DistributedApp | Client):
        self.target = target
        self._lock = threading.RLock()

        if isinstance(target, Client):
            self._kind = "client"
            self.agent_id = target.agent_id
        elif isinstance(target, Server):
            self._kind = "server"
            self.agent_id = target.node_id
        elif isinstance(target, DistributedApp):
            self._kind = "app"
            self.agent_id = target.node_id
        elif isinstance(target, Replica):
            self._kind = "replica"
            self.agent_id = target.agent_id
        else:
            raise TypeError(
                f"expected Replica, Server, DistributedApp, or Client; got {type(target).__name__}"
            )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def counter_inc(self, key: str, by: int = 1):
        with self._lock:
            if self._kind == "server":
                return self._server_emit("counter_inc", key, {"by": int(by)})
            return self.target.counter_inc(key, int(by))

    def counter_dec(self, key: str, by: int = 1):
        with self._lock:
            if self._kind == "server":
                return self._server_emit("counter_dec", key, {"by": int(by)})
            return self.target.counter_dec(key, int(by))

    def register_set(self, key: str, value: Any):
        with self._lock:
            if self._kind == "server":
                return self._server_emit("lww_set", key, {"value": value})
            return self.target.register_set(key, value)

    def register_set_weighted(self, key: str, value: Any, weight: float):
        with self._lock:
            if self._kind == "server":
                return self._server_emit("lww_set", key, {"value": value, "weight": weight})
            return self.target.register_set_weighted(key, value, weight)

    def mv_set(self, key: str, value: Any, supersedes: list[str] | tuple[str, ...] | None = None):
        with self._lock:
            if self._kind == "server":
                with self.target.lock:
                    if supersedes is None:
                        cands = self.target.replica.candidates(key)
                        actual_sup = [c["dot"] for c in cands]
                    else:
                        actual_sup = [str(s) for s in supersedes]
                return self._server_emit("mv_set", key, {"value": value, "supersedes": actual_sup})
            return self.target.mv_set(key, value, supersedes=supersedes)

    def mv_resolve(self, key: str, resolved_value: Any):
        with self._lock:
            if self._kind == "server":
                with self.target.lock:
                    dots = [c["dot"] for c in self.target.replica.candidates(key)]
                return self._server_emit("mv_set", key, {"value": resolved_value, "supersedes": dots})
            return self.target.mv_resolve(key, resolved_value)

    def set_add(self, key: str, value: Any, tag: str | None = None):
        with self._lock:
            if self._kind == "server":
                with self.target.lock:
                    actual_tag = tag or f"{self.target.node_id}:{self.target.replica.timestamp() + 1}"
                return self._server_emit("orset_add", key, {"value": value, "tag": actual_tag})
            return self.target.set_add(key, value, tag=tag)

    def set_remove(self, key: str, tag: str):
        with self._lock:
            if self._kind == "server":
                return self._server_emit("orset_remove", key, {"tag": tag})
            return self.target.set_remove(key, tag)

    def _server_emit(self, op: str, key: str, params: dict[str, Any]):
        server: Server = self.target  # type: ignore[assignment]
        with server.lock:
            action = server.replica._emit(op, key, params)
            if server.store is not None:
                rotate = server.store.append(action)
                if rotate:
                    server.store.rotate(server.replica.state._map)
        return action

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def read(self, key: str) -> Any:
        with self._lock:
            val = self.target.read(key)
            if isinstance(val, list):
                # Wire read from Client converts frozenset -> sorted list; normalize when needed by caller
                return val
            return val

    def read_set(self, key: str) -> frozenset:
        val = self.read(key)
        if val is None:
            return frozenset()
        if isinstance(val, (set, frozenset, list, tuple)):
            # Elements stored as JSON primitives/tuples
            return frozenset(val)
        return frozenset({val})

    def view(self, prefix: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self._kind == "client":
                all_data = self.target.read_all() or {}
            elif self._kind == "server":
                all_data = self.target.view()
            elif self._kind == "app":
                all_data = self.target.replica.state.view()
            else:
                all_data = self.target.state.view()
        if prefix is None:
            return dict(all_data)
        return {k: v for k, v in all_data.items() if k.startswith(prefix)}

    def keys(self, prefix: str | None = None) -> list[str]:
        return sorted(self.view(prefix=prefix).keys())

    def conflicts(self, prefix: str | None = None) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return self.target.conflicts(prefix=prefix)

    def candidates(self, key: str) -> list[dict[str, Any]]:
        with self._lock:
            return self.target.candidates(key)

    def hash(self) -> str:
        with self._lock:
            if self._kind == "client":
                return self.target.hash()
            if self._kind == "server":
                return self.target.hash()
            if self._kind == "app":
                return state_hash(self.target.replica.state)
            return state_hash(self.target.state)


def coerce_backend(target: BackendAdapter | Replica | Server | DistributedApp | Client) -> BackendAdapter:
    if isinstance(target, BackendAdapter):
        return target
    return BackendAdapter(target)
