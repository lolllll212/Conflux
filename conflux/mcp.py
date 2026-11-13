"""Zero-dependency Model Context Protocol (MCP) server for Conflux.

Exposes Conflux's lock-free CRDT state fabric over standard MCP JSON-RPC 2.0
(stdio transport, newline-delimited + Content-Length framing support) so
Claude Code, Claude Desktop, Cursor, Windsurf, Cline, and any MCP client can
share convergent state across parallel coding/research agents.

Two deployment modes via ``conflux-mcp`` CLI:

1. **Networked mesh mode (`--connect host:port`)**:
   Connects as a signed or unsigned ``Client`` to a live ``conflux-server``
   mesh so multiple MCP agent processes converge in real time.

2. **Standalone durable mode (`--data ./dir` or in-memory)**:
   Starts an embedded ``Server`` (optional TCP listener + peer list + disk
   journal) so a local multi-agent team can coordinate with zero setup.

Example Claude Desktop / Cursor MCP config (``mcp.json``):

.. code-block:: json

    {
      "mcpServers": {
        "conflux": {
          "command": "conflux-mcp",
          "args": ["--agent-id", "claude-1", "--data", "./.conflux-mcp"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, BinaryIO, TextIO

from . import __version__
from .app import TaskStore
from .client import Client
from .integrations.base import BackendAdapter, coerce_backend
from .server import Server
from .state import Replica

LATEST_PROTOCOL_VERSION = "2024-11-05"


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
            "(PNCounter integer, LWWRegister JSON value, or ORSet list)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Shared state key (e.g. 'fleet.jobs_done' or 'plan.leader').",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "conflux_snapshot",
        "description": (
            "Return a full JSON snapshot of all shared CRDT keys (optionally filtered "
            "by prefix) together with the canonical 64-hex SHA-256 state_hash."
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
        "description": (
            "Add an item to a conflict-free Observed-Remove Set (ORSet) in shared state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Set key (e.g. 'swarm.findings')."},
                "value": {"description": "JSON-serializable element to add."},
                "tag": {
                    "type": "string",
                    "description": "Optional unique tag (defaults to auto-generated monotonic agent tag); save this tag if you plan to remove the item later.",
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
        "name": "conflux_task_update",
        "description": (
            "Create or update a replicated multi-agent task record under 'tasks/<task_id>' "
            "(status: created | running | done | failed) so parallel agents can coordinate work queues."
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


class ConfluxMCPServer:
    """MCP JSON-RPC 2.0 dispatcher backed by any Conflux runtime object."""

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
            return {"key": qkey, "value": val, "state_hash": self.backend.hash()}

        if name == "conflux_snapshot":
            raw_prefix = args.get("prefix")
            prefix = self._qualify(str(raw_prefix)) if raw_prefix else (self.namespace or None)
            view = _json_safe(self.backend.view(prefix=prefix))
            return {
                "agent_id": self.backend.agent_id,
                "state_hash": self.backend.hash(),
                "keys": len(view),
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

            if method == "resources/list":
                resources = [
                    {
                        "uri": "conflux://state/snapshot",
                        "name": "Conflux Shared State Snapshot",
                        "description": "Live JSON snapshot and SHA-256 state_hash of all converged CRDT keys.",
                        "mimeType": "application/json",
                    }
                ]
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": resources}}

            if method == "resources/read":
                uri = params.get("uri", "")
                if uri == "conflux://state/snapshot":
                    snap = self.call_tool("conflux_snapshot", {})
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "contents": [
                                {
                                    "uri": uri,
                                    "mimeType": "application/json",
                                    "text": json.dumps(snap, indent=2, sort_keys=True),
                                }
                            ]
                        },
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": f"Unknown resource URI: {uri}"},
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
                # Consume remaining headers up to blank line
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
        description="Conflux MCP Server — shared lock-free CRDT state for Claude, Cursor, and MCP agents.",
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
