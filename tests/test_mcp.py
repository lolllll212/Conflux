"""Tests for the zero-dependency Conflux MCP (Model Context Protocol) server."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time

from conflux import Client, Cluster, InMemoryTransport, Server, state_hash
from conflux.mcp import ConfluxMCPServer, MCP_TOOLS


def test_mcp_initialize_tools_and_resources_over_cluster():
    cluster = Cluster({"claude-a", "cursor-b"}, transport=InMemoryTransport())
    mcp_a = ConfluxMCPServer(cluster.node("claude-a"), namespace="swarm.")
    mcp_b = ConfluxMCPServer(cluster.node("cursor-b"), namespace="swarm.")

    # 1. Initialize handshake
    init_resp = mcp_a.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
    )
    assert init_resp["result"]["protocolVersion"] == "2024-11-05"
    assert init_resp["result"]["serverInfo"]["name"] == "conflux-state"

    # Notifications produce no response
    assert mcp_a.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    # 2. List tools
    tools_resp = mcp_a.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert names == {t["name"] for t in MCP_TOOLS}

    # 3. Concurrent CRDT tool calls across two MCP servers
    mcp_a.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "conflux_counter_add", "arguments": {"key": "bugs_fixed", "delta": 2}},
        }
    )
    add_res = mcp_a.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "conflux_set_add",
                "arguments": {"key": "files_checked", "value": "auth.py", "tag": "t-auth"},
            },
        }
    )
    assert add_res["result"]["isError"] is False

    mcp_b.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "conflux_counter_add", "arguments": {"key": "bugs_fixed", "delta": 3}},
        }
    )
    mcp_b.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "conflux_task_update",
                "arguments": {
                    "task_id": "audit-1",
                    "status": "done",
                    "result": {"passed": True},
                },
            },
        }
    )

    cluster.gossip_round()

    # 4. Verify convergence via resources/read and conflux_read
    read_resp = mcp_a.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "conflux_read", "arguments": {"key": "bugs_fixed"}},
        }
    )
    assert read_resp["result"]["structuredContent"]["value"] == 5

    res_list = mcp_b.handle_message({"jsonrpc": "2.0", "id": 8, "method": "resources/list"})
    assert res_list["result"]["resources"][0]["uri"] == "conflux://state/snapshot"

    res_read = mcp_b.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "resources/read",
            "params": {"uri": "conflux://state/snapshot"},
        }
    )
    snap = json.loads(res_read["result"]["contents"][0]["text"])
    assert snap["state"]["swarm.bugs_fixed"] == 5
    assert snap["state"]["swarm.files_checked"] == ["auth.py"]
    assert snap["state"]["swarm.tasks/audit-1"]["status"] == "done"

    # Remove tag from ORSet
    mcp_b.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "conflux_set_remove", "arguments": {"key": "files_checked", "tag": "t-auth"}},
        }
    )
    cluster.gossip_round()
    after_rm = mcp_a.call_tool("conflux_read", {"key": "files_checked"})
    assert after_rm["value"] == []

    assert state_hash(cluster.node("claude-a").replica.state) == state_hash(
        cluster.node("cursor-b").replica.state
    )


def test_mcp_stdio_newline_and_content_length_framing():
    server = Server("mcp-hub", ("127.0.0.1", 0)).start()
    try:
        # Test Content-Length framing in-memory
        req_obj = {"jsonrpc": "2.0", "id": 42, "method": "ping"}
        req_bytes = json.dumps(req_obj).encode("utf-8")
        framed_in = io.BytesIO(f"Content-Length: {len(req_bytes)}\r\n\r\n".encode("ascii") + req_bytes)
        framed_out = io.BytesIO()
        ConfluxMCPServer(server).serve_stdio(stdin=framed_in, stdout=framed_out)
        raw_reply = framed_out.getvalue()
        assert raw_reply.startswith(b"Content-Length:")
        body = json.loads(raw_reply.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
        assert body == {"jsonrpc": "2.0", "id": 42, "result": {}}

        # Test live subprocess CLI (`python -m conflux.mcp --connect host:port`) with newline JSON-RPC
        host, port = server.node_address
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "conflux_register_set",
                        "arguments": {"key": "mcp.status", "value": "connected"},
                    },
                }
            ),
        ]
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "conflux.mcp",
                "--agent-id",
                "claude-cli",
                "--connect",
                f"{host}:{port}",
            ],
            input=("\n".join(lines) + "\n").encode("utf-8"),
            capture_output=True,
            timeout=15,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8")
        out_lines = [json.loads(ln) for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]
        assert len(out_lines) == 2
        assert out_lines[1]["result"]["structuredContent"]["value"] == "connected"
        assert server.read("mcp.status") == "connected"
    finally:
        server.stop()
