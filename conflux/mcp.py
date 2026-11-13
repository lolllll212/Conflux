"""Zero-dependency Model Context Protocol (MCP) server for Conflux.

Exposes Conflux's lock-free CRDT state fabric — including **conflict-preserving
Multi-Value Registers (`MVRegister`)**, **conflict detection & resolution**,
**lease-based file/resource claims**, **reactive key watching**, and **MCP
Prompts + Resources + Resource Templates** — over standard MCP JSON-RPC 2.0
(stdio transport, supporting both newline-delimited JSON and ``Content-Length``
framing).

Works seamlessly with Claude Code, Claude Desktop, Cursor, Windsurf, Cline,
and any MCP-compliant agent orchestrator.

Two deployment modes via ``conflux-mcp`` CLI:

1. **Networked mesh mode (``--connect host:port``)**:
   Connects as a signed or unsigned ``Client`` to a live ``conflux-server``
   mesh so multiple MCP agent processes converge in real time.

2. **Standalone durable mode (``--data ./dir`` or ``--listen host:port``)**:
   Starts an embedded ``Server`` (optional TCP/WS listener + peer list + disk
   journal) so a local multi-agent team coordinates with zero setup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, BinaryIO

from . import __version__
from .client import Client
from .integrations.base import BackendAdapter, coerce_backend
from .server import Server
from .state import Replica

LATEST_PROTOCOL_VERSION = "2025-03-26"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "conflux_read",
        "description": (
            "Read the current converged value of a shared CRDT key "
            "(PNCounter int, LWWRegister JSON value, MVRegister winning value + active conflicts, or ORSet list)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Shared state key (e.g. 'fleet.jobs_done' or 'arch.decision').",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "conflux_snapshot",
        "description": (
            "Return a full JSON snapshot of all shared CRDT keys (optionally filtered "
            "by prefix), active MVRegister conflict count, and the canonical 64-hex SHA-256 state_hash."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Optional key prefix filter (e.g. 'crew.' or 'tasks/').",
                }
            },
        },
    },
    {
        "name": "conflux_register_set",
        "description": (
            "Write any JSON value to a Last-Writer-Wins (LWW) register in shared Conflux "
            "state. Concurrent writes resolve deterministically with zero locks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Register key to set."},
                "value": {
                    "description": "Any JSON-serializable value (string, number, boolean, object, array)."
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "conflux_mv_write",
        "description": (
            "Write a JSON value to a conflict-preserving Multi-Value Register (MVRegister). "
            "Sequential writes cleanly replace prior values, while un-synced concurrent writes "
            "from parallel agents are BOTH preserved in conflux_conflicts until resolved!"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "MVRegister key (e.g. 'spec.api_schema')."},
                "value": {"description": "JSON-serializable value proposed by this agent."},
                "supersedes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of 'tick:agent' dots to supersede (defaults to all currently observed dots on this replica).",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "conflux_conflicts",
        "description": (
            "List all unresolved concurrent write conflicts across MVRegister keys, showing "
            "every competing agent's candidate value, causal dot ('tick:agent'), and current canonical winner."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Optional key prefix to filter conflicts.",
                }
            },
        },
    },
    {
        "name": "conflux_resolve_conflict",
        "description": (
            "Causally resolve all currently observed concurrent MVRegister candidates on 'key' "
            "into a single merged/agreed JSON value, clearing the conflict across the mesh."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Conflicted MVRegister key to resolve."},
                "resolved_value": {"description": "Merged/agreed JSON value superseding all prior candidates."},
            },
            "required": ["key", "resolved_value"],
        },
    },
    {
        "name": "conflux_counter_add",
        "description": (
            "Atomically increment (delta > 0) or decrement (delta < 0) a lock-free "
            "shared PNCounter across concurrent agents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Counter key."},
                "delta": {
                    "type": "integer",
                    "description": "Non-zero integer to add (positive increments, negative decrements).",
                    "default": 1,
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "conflux_set_add",
        "description": "Add an item to a conflict-free Observed-Remove Set (ORSet) in shared state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Set key (e.g. 'swarm.findings')."},
                "value": {"description": "JSON-serializable element to add."},
                "tag": {
                    "type": "string",
                    "description": "Optional unique tag (save this tag if you plan to remove the item later).",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "conflux_set_remove",
        "description": "Tombstone-remove an element tag from a shared Observed-Remove Set (ORSet).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Set key."},
                "tag": {"type": "string", "description": "Tag previously used in conflux_set_add."},
            },
            "required": ["key", "tag"],
        },
    },
    {
        "name": "conflux_batch",
        "description": (
            "Execute multiple CRDT mutations sequentially in one tool call "
            "(ops: register_set, mv_write, counter_add, set_add, set_remove)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "description": "List of mutation specs: {op, key, value/delta/tag}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": [
                                    "register_set",
                                    "mv_write",
                                    "counter_add",
                                    "set_add",
                                    "set_remove",
                                ],
                            },
                            "key": {"type": "string"},
                            "value": {},
                            "delta": {"type": "integer"},
                            "tag": {"type": "string"},
                        },
                        "required": ["op", "key"],
                    },
                }
            },
            "required": ["operations"],
        },
    },
    {
        "name": "conflux_claim_resource",
        "description": (
            "Advisory file/task claim with TTL lease so parallel coding agents (Claude Code, Cursor) "
            "avoid editing the same file simultaneously. Reports 'conflict' if another agent holds an active lease."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "description": "File path or resource identifier (e.g. 'src/auth.py').",
                },
                "ttl_seconds": {
                    "type": "number",
                    "description": "Lease duration in seconds before automatic expiry (default 120).",
                    "default": 120,
                },
                "intent": {
                    "type": "string",
                    "description": "Short note describing what this agent is doing with the resource.",
                },
                "force": {
                    "type": "boolean",
                    "description": "Steal the claim even if another agent's lease has not expired yet.",
                    "default": False,
                },
            },
            "required": ["resource"],
        },
    },
    {
        "name": "conflux_release_resource",
        "description": "Release an advisory resource/file lease previously claimed by this agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource": {"type": "string", "description": "File path or resource identifier."}
            },
            "required": ["resource"],
        },
    },
    {
        "name": "conflux_watch_wait",
        "description": (
            "Wait up to timeout_seconds until 'key' changes from 'since_value' (or becomes non-null), "
            "enabling event-driven handoffs between parallel MCP agents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to watch."},
                "since_value": {
                    "description": "Return immediately once current value differs from since_value."
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max seconds to wait (capped at 30s).",
                    "default": 2.0,
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "conflux_task_update",
        "description": (
            "Create or update a replicated multi-agent task record under 'tasks/<task_id>' "
            "(status: created | running | done | failed) so parallel agents coordinate work queues."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Unique task identifier."},
                "status": {
                    "type": "string",
                    "enum": ["created", "running", "done", "failed"],
                    "description": "New lifecycle status for the task.",
                },
                "kind": {
                    "type": "string",
                    "description": "Task category/type (used when creating a task).",
                    "default": "agent_task",
                },
                "step": {"type": "string", "description": "Optional current step label."},
                "result": {"description": "Optional JSON result payload when status='done'."},
                "error": {"type": "string", "description": "Optional error message when status='failed'."},
            },
            "required": ["task_id", "status"],
        },
    },
]


MCP_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "resolve_state_conflicts",
        "description": (
            "Inspect all active Conflux MVRegister conflicts and guide the LLM agent "
            "to synthesize and commit a conflict-free merged value."
        ),
        "arguments": [
            {
                "name": "prefix",
                "description": "Optional key prefix to scope conflict inspection.",
                "required": False,
            }
        ],
    },
    {
        "name": "coordinate_parallel_swarm",
        "description": (
            "Bootstrap a parallel multi-agent coding/research session with Conflux "
            "file-claim leases, shared counters, and MVRegister conflict hygiene."
        ),
        "arguments": [
            {
                "name": "goal",
                "description": "High-level team objective for the agent crew.",
                "required": False,
            }
        ],
    },
]


class ConfluxMCPServer:
    """Full-featured MCP JSON-RPC 2.0 server backed by any Conflux runtime object."""

    def __init__(self, backend: Any, *, namespace: str = "", server_name: str = "conflux-state"):
        self.backend: BackendAdapter = coerce_backend(backend)
        self.namespace = namespace
        self.server_name = server_name

    def _qualify(self, key: str) -> str:
        if not self.namespace or key.startswith(self.namespace):
            return key
        return f"{self.namespace}{key}"

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})

        if name == "conflux_read":
            qkey = self._qualify(str(args["key"]))
            val = _json_safe(self.backend.read(qkey))
            cands = self.backend.candidates(qkey)
            resp: dict[str, Any] = {
                "key": qkey,
                "value": val,
                "state_hash": self.backend.hash(),
            }
            if cands:
                distinct = {json.dumps(_json_safe(c["value"]), sort_keys=True) for c in cands}
                resp["has_conflict"] = len(distinct) > 1
                resp["candidates"] = _json_safe(cands)
            return resp

        if name == "conflux_snapshot":
            raw_prefix = args.get("prefix")
            prefix = self._qualify(str(raw_prefix)) if raw_prefix else (self.namespace or None)
            view = _json_safe(self.backend.view(prefix=prefix))
            conflicts = _json_safe(self.backend.conflicts(prefix=prefix))
            return {
                "agent_id": self.backend.agent_id,
                "state_hash": self.backend.hash(),
                "keys": len(view),
                "conflict_count": len(conflicts),
                "conflicts": conflicts,
                "state": view,
            }

        if name == "conflux_register_set":
            qkey = self._qualify(str(args["key"]))
            value = args["value"]
            action = self.backend.register_set(qkey, value)
            return {
                "status": "ok",
                "action_id": action.action_id,
                "key": qkey,
                "value": _json_safe(self.backend.read(qkey)),
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_mv_write":
            qkey = self._qualify(str(args["key"]))
            supersedes = args.get("supersedes")
            action = self.backend.mv_set(qkey, args["value"], supersedes=supersedes)
            cands = self.backend.candidates(qkey)
            distinct = {json.dumps(_json_safe(c["value"]), sort_keys=True) for c in cands}
            return {
                "status": "ok",
                "action_id": action.action_id,
                "dot": f"{action.tick}:{action.agent}",
                "key": qkey,
                "winner": _json_safe(self.backend.read(qkey)),
                "has_conflict": len(distinct) > 1,
                "candidates": _json_safe(cands),
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_conflicts":
            raw_prefix = args.get("prefix")
            prefix = self._qualify(str(raw_prefix)) if raw_prefix else (self.namespace or None)
            conflicts = self.backend.conflicts(prefix=prefix)
            formatted = {
                k: {
                    "winner": _json_safe(self.backend.read(k)),
                    "candidate_count": len(cands),
                    "candidates": _json_safe(cands),
                }
                for k, cands in conflicts.items()
            }
            return {
                "conflict_count": len(formatted),
                "conflicts": formatted,
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_resolve_conflict":
            qkey = self._qualify(str(args["key"]))
            before_cands = self.backend.candidates(qkey)
            action = self.backend.mv_resolve(qkey, args["resolved_value"])
            return {
                "status": "resolved",
                "action_id": action.action_id,
                "key": qkey,
                "superseded_dots": [c["dot"] for c in before_cands],
                "value": _json_safe(self.backend.read(qkey)),
                "has_conflict": len(self.backend.conflicts(prefix=qkey)) > 0,
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_counter_add":
            qkey = self._qualify(str(args["key"]))
            delta = int(args.get("delta", 1))
            if delta == 0:
                raise ValueError("delta must be a non-zero integer")
            if delta > 0:
                action = self.backend.counter_inc(qkey, delta)
            else:
                action = self.backend.counter_dec(qkey, abs(delta))
            return {
                "status": "ok",
                "action_id": action.action_id,
                "key": qkey,
                "total": self.backend.read(qkey),
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_set_add":
            qkey = self._qualify(str(args["key"]))
            tag = args.get("tag")
            action = self.backend.set_add(qkey, args["value"], tag=tag)
            return {
                "status": "ok",
                "action_id": action.action_id,
                "tag": action.params.get("tag"),
                "key": qkey,
                "members": _json_safe(self.backend.read_set(qkey)),
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_set_remove":
            qkey = self._qualify(str(args["key"]))
            tag = str(args["tag"])
            action = self.backend.set_remove(qkey, tag)
            return {
                "status": "ok",
                "action_id": action.action_id,
                "key": qkey,
                "members": _json_safe(self.backend.read_set(qkey)),
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_batch":
            operations = list(args.get("operations") or [])
            applied = []
            for spec in operations:
                op = spec["op"]
                key = spec["key"]
                if op == "register_set":
                    res = self.call_tool("conflux_register_set", {"key": key, "value": spec["value"]})
                elif op == "mv_write":
                    res = self.call_tool("conflux_mv_write", {"key": key, "value": spec["value"]})
                elif op == "counter_add":
                    res = self.call_tool(
                        "conflux_counter_add", {"key": key, "delta": spec.get("delta", 1)}
                    )
                elif op == "set_add":
                    res = self.call_tool(
                        "conflux_set_add",
                        {"key": key, "value": spec["value"], "tag": spec.get("tag")},
                    )
                elif op == "set_remove":
                    res = self.call_tool("conflux_set_remove", {"key": key, "tag": spec["tag"]})
                else:
                    raise ValueError(f"unsupported batch op {op!r}")
                applied.append({"op": op, "key": res["key"], "action_id": res["action_id"]})
            return {
                "status": "ok",
                "applied_count": len(applied),
                "applied": applied,
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_claim_resource":
            resource = str(args["resource"])
            ttl = float(args.get("ttl_seconds", 120.0))
            force = bool(args.get("force", False))
            intent = str(args.get("intent") or "")
            claim_key = self._qualify(f"claims/{resource}")
            now = time.time()

            current = self.backend.read(claim_key)
            if (
                isinstance(current, dict)
                and current.get("status") == "claimed"
                and current.get("owner") != self.backend.agent_id
                and float(current.get("expires_at", 0.0)) > now
                and not force
            ):
                return {
                    "status": "conflict",
                    "acquired": False,
                    "resource": resource,
                    "held_by": current.get("owner"),
                    "holder_intent": current.get("intent", ""),
                    "seconds_remaining": round(float(current["expires_at"]) - now, 2),
                    "state_hash": self.backend.hash(),
                }

            lease = {
                "resource": resource,
                "status": "claimed",
                "owner": self.backend.agent_id,
                "intent": intent,
                "claimed_at": round(now, 3),
                "expires_at": round(now + ttl, 3),
            }
            self.backend.register_set(claim_key, lease)
            return {
                "status": "claimed",
                "acquired": True,
                "lease": lease,
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_release_resource":
            resource = str(args["resource"])
            claim_key = self._qualify(f"claims/{resource}")
            current = self.backend.read(claim_key)
            if (
                isinstance(current, dict)
                and current.get("status") == "claimed"
                and current.get("owner") not in (None, self.backend.agent_id)
            ):
                return {
                    "status": "conflict",
                    "released": False,
                    "reason": f"resource {resource!r} is owned by {current.get('owner')!r}",
                    "state_hash": self.backend.hash(),
                }
            released = {
                "resource": resource,
                "status": "released",
                "owner": None,
                "released_by": self.backend.agent_id,
                "released_at": round(time.time(), 3),
            }
            self.backend.register_set(claim_key, released)
            return {
                "status": "released",
                "released": True,
                "resource": resource,
                "state_hash": self.backend.hash(),
            }

        if name == "conflux_watch_wait":
            qkey = self._qualify(str(args["key"]))
            has_since = "since_value" in args
            since_val = _json_safe(args.get("since_value"))
            timeout = max(0.0, min(30.0, float(args.get("timeout_seconds", 2.0))))
            deadline = time.time() + timeout
            while True:
                cur = _json_safe(self.backend.read(qkey))
                changed = (cur != since_val) if has_since else (cur is not None)
                if changed or time.time() >= deadline:
                    return {
                        "key": qkey,
                        "changed": changed,
                        "value": cur,
                        "state_hash": self.backend.hash(),
                    }
                time.sleep(0.04)

        if name == "conflux_task_update":
            task_id = str(args["task_id"])
            status = str(args["status"])
            task_key = self._qualify(f"tasks/{task_id}")
            existing = dict(self.backend.read(task_key) or {})
            record = {
                "id": task_id,
                "kind": str(args.get("kind") or existing.get("kind") or "agent_task"),
                "agent": self.backend.agent_id,
                "status": status,
                "step": args.get("step"),
                "result": args.get("result", existing.get("result")),
                "error": args.get("error"),
            }
            self.backend.register_set(task_key, record)
            return {
                "status": "ok",
                "task": record,
                "state_hash": self.backend.hash(),
            }

        raise KeyError(f"unknown tool: {name}")

    # ------------------------------------------------------------------
    # Prompts builder
    # ------------------------------------------------------------------
    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        if name == "resolve_state_conflicts":
            conflicts_payload = self.call_tool("conflux_conflicts", {"prefix": args.get("prefix")})
            text = (
                "Inspect the active Conflux Multi-Value Register conflicts below. "
                "For each conflicted key, compare every agent's candidate value and causal dot, "
                "synthesize the correct merged value, and call `conflux_resolve_conflict` "
                "so all replicas converge cleanly:\n\n"
                f"{json.dumps(conflicts_payload, indent=2, sort_keys=True)}"
            )
            return {
                "description": "Resolve concurrent MVRegister conflicts across agents.",
                "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
            }

        if name == "coordinate_parallel_swarm":
            goal = args.get("goal") or "Collaborate safely without overwriting peer agent work."
            snap = self.call_tool("conflux_snapshot", {})
            text = (
                f"You are agent `{self.backend.agent_id}` connected to a shared Conflux CRDT state mesh.\n"
                f"Team objective: {goal}\n\n"
                "Coordination protocol:\n"
                "1. Before editing a shared file/module, call `conflux_claim_resource` to acquire a TTL lease.\n"
                "2. Use `conflux_mv_write` for shared specs/decisions where concurrent edits must be preserved rather than overwritten.\n"
                "3. Periodically check `conflux_conflicts` and resolve divergences with `conflux_resolve_conflict`.\n"
                "4. Track progress with `conflux_counter_add` and `conflux_task_update`, and release leases via `conflux_release_resource`.\n\n"
                f"Current mesh snapshot:\n{json.dumps(snap, indent=2, sort_keys=True)}"
            )
            return {
                "description": "Swarm coordination protocol over Conflux MCP.",
                "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
            }

        raise KeyError(f"unknown prompt: {name}")

    # ------------------------------------------------------------------
    # JSON-RPC 2.0 message handler
    # ------------------------------------------------------------------
    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # Notifications (no id) never produce a response in JSON-RPC 2.0
        if msg_id is None:
            return None

        try:
            if method == "initialize":
                requested_version = params.get("protocolVersion") or LATEST_PROTOCOL_VERSION
                result: dict[str, Any] = {
                    "protocolVersion": requested_version,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": self.server_name,
                        "version": __version__,
                    },
                }
                return {"jsonrpc": "2.0", "id": msg_id, "result": result}

            if method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": MCP_TOOLS}}

            if method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments") or {}
                try:
                    payload = self.call_tool(tool_name, tool_args)
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}
                            ],
                            "structuredContent": payload,
                            "isError": False,
                        },
                    }
                except Exception as exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": f"Error: {exc}"}],
                            "isError": True,
                        },
                    }

            if method == "prompts/list":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": MCP_PROMPTS}}

            if method == "prompts/get":
                prompt_name = params.get("name", "")
                prompt_args = params.get("arguments") or {}
                try:
                    prompt_res = self.get_prompt(prompt_name, prompt_args)
                    return {"jsonrpc": "2.0", "id": msg_id, "result": prompt_res}
                except KeyError as exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32602, "message": str(exc)},
                    }

            if method == "resources/list":
                resources = [
                    {
                        "uri": "conflux://state/snapshot",
                        "name": "Conflux Shared State Snapshot",
                        "description": "Live JSON snapshot and SHA-256 state_hash of all converged CRDT keys.",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "conflux://state/conflicts",
                        "name": "Conflux Active MVRegister Conflicts",
                        "description": "Live feed of concurrent multi-agent write conflicts awaiting causal resolution.",
                        "mimeType": "application/json",
                    },
                ]
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": resources}}

            if method == "resources/templates/list":
                templates = [
                    {
                        "uriTemplate": "conflux://key/{key}",
                        "name": "Read Individual Conflux Key",
                        "description": "Inspect the current CRDT value and candidate list of any key.",
                        "mimeType": "application/json",
                    }
                ]
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"resourceTemplates": templates}}

            if method == "resources/read":
                uri = params.get("uri", "")
                if uri == "conflux://state/snapshot":
                    data = self.call_tool("conflux_snapshot", {})
                elif uri == "conflux://state/conflicts":
                    data = self.call_tool("conflux_conflicts", {})
                elif uri.startswith("conflux://key/"):
                    target_key = uri[len("conflux://key/") :]
                    data = self.call_tool("conflux_read", {"key": target_key})
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32602, "message": f"Unknown resource URI: {uri}"},
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(data, indent=2, sort_keys=True),
                            }
                        ]
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:  # pragma: no cover - defensive guard
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(exc)},
            }

    # ------------------------------------------------------------------
    # Stdio loop (supports both MCP newline-delimited JSON and Content-Length framing)
    # ------------------------------------------------------------------
    def serve_stdio(
        self,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
    ) -> None:
        in_stream = stdin if stdin is not None else sys.stdin.buffer
        out_stream = stdout if stdout is not None else sys.stdout.buffer

        while True:
            line = in_stream.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            use_content_length = False
            if stripped.lower().startswith(b"content-length:"):
                use_content_length = True
                length = int(stripped.split(b":", 1)[1].strip())
                while True:
                    hdr = in_stream.readline()
                    if not hdr or hdr.strip() == b"":
                        break
                raw_body = in_stream.read(length)
                if not raw_body:
                    break
            else:
                raw_body = stripped

            try:
                msg = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                self._write_response(out_stream, err_resp, use_content_length)
                continue

            resp = self.handle_message(msg)
            if resp is not None:
                self._write_response(out_stream, resp, use_content_length)

    @staticmethod
    def _write_response(out_stream: BinaryIO, payload: dict[str, Any], framed: bool) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if framed:
            header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
            out_stream.write(header + encoded)
        else:
            out_stream.write(encoded + b"\n")
        out_stream.flush()


def _split_addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port:
        raise SystemExit(f"invalid address {text!r}, expected host:port")
    return host, int(port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="conflux-mcp",
        description="Conflux MCP Server — conflict-aware CRDT shared state for Claude, Cursor, and MCP agents.",
    )
    parser.add_argument("--agent-id", default="mcp-agent", help="Agent identity for signed/CRDT actions")
    parser.add_argument(
        "--connect",
        default=None,
        help="Connect as a Client to an existing Conflux Server (e.g. 127.0.0.1:7001)",
    )
    parser.add_argument(
        "--transport",
        choices=("tcp", "ws"),
        default="tcp",
        help="Network transport when --connect or --listen is used",
    )
    parser.add_argument("--secret", default=None, help="Optional HMAC secret for signed agent actions")
    parser.add_argument(
        "--listen",
        default=None,
        help="Start an embedded Conflux Server bound to host:port (e.g. 127.0.0.1:7001)",
    )
    parser.add_argument(
        "--peers",
        default="",
        help="Comma-separated peer host:port addresses when running an embedded Server",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Durable journal + snapshot directory for embedded Server mode",
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="Optional key prefix automatically prepended to MCP tool keys",
    )
    args = parser.parse_args(argv)

    cleanup = lambda: None
    if args.connect:
        host, port = _split_addr(args.connect)
        client = Client(
            host,
            port,
            agent_id=args.agent_id,
            secret=args.secret,
            transport=args.transport,
        )
        backend: Any = client
        cleanup = client.close
    elif args.listen or args.data or args.peers:
        bind_addr = _split_addr(args.listen) if args.listen else ("127.0.0.1", 0)
        peers = [_split_addr(p) for p in args.peers.split(",") if p.strip()]
        srv = Server(
            args.agent_id,
            bind_addr,
            peers=peers,
            data_dir=args.data,
            transport=args.transport,
        ).start()
        backend = srv
        cleanup = srv.stop
    else:
        backend = Replica(args.agent_id)

    try:
        mcp_server = ConfluxMCPServer(backend, namespace=args.namespace)
        mcp_server.serve_stdio()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
