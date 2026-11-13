"""Comprehensive tests for MVRegister conflict preservation/resolution and the upgraded Conflux MCP server."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time

from conflux import Client, Cluster, InMemoryTransport, MVRegister, Replica, Server, Stamp, from_dict, state_hash
from conflux.mcp import ConfluxMCPServer, MCP_PROMPTS, MCP_TOOLS


# ----------------------------------------------------------------------
# MVRegister mathematical join laws + causal conflict lifecycle
# ----------------------------------------------------------------------
def test_mvregister_join_laws_and_causal_conflict_resolution():
    a = MVRegister().write("postgres", Stamp(1, "architect-a"))
    b = MVRegister().write("sqlite", Stamp(1, "architect-b"))
    c = MVRegister().write("cockroachdb", Stamp(2, "architect-c"))

    # Semilattice laws: commutative, idempotent, associative
    assert a.merge(b) == b.merge(a)
    assert a.merge(a) == a
    assert a.merge(b).merge(c) == a.merge(b.merge(c))

    # Concurrent un-synced writes preserve BOTH candidates!
    merged = a.merge(b)
    assert merged.has_conflict() is True
    assert {cand["value"] for cand in merged.candidates()} == {"postgres", "sqlite"}

    # Serialization round-trip preserves conflicts & superseded dots
    reloaded = from_dict(merged.to_dict())
    assert reloaded == merged
    assert reloaded.has_conflict() is True

    # Causal resolution supersedes prior dots cleanly
    resolved = merged.write("postgres+read-replica", Stamp(3, "lead"), supersedes=merged.active_dots())
    assert resolved.has_conflict() is False
    assert resolved.read() == "postgres+read-replica"
    assert len(resolved.candidates()) == 1

    # Re-merging stale pre-resolution replicas is idempotent (superseded dots stay pruned!)
    assert resolved.merge(a).merge(b) == resolved


def test_mvregister_networked_client_server_conflicts_and_resolve():
    s1 = Server("n1", ("127.0.0.1", 0), gossip_interval=0.08).start()
    s2 = Server("n2", ("127.0.0.1", 0), gossip_interval=0.08).start()
    try:
        # Before peering (partitioned!), two agents write conflicting specs on s1 and s2
        with Client("127.0.0.1", s1.node_address[1], agent_id="agent-alpha") as c1:
            c1.mv_set("arch.auth_mode", {"scheme": "jwt", "ttl": 900})
        with Client("127.0.0.1", s2.node_address[1], agent_id="agent-beta") as c2:
            c2.mv_set("arch.auth_mode", {"scheme": "mtls", "ttl": 3600})

        # Heal partition -> gossip merges both concurrent writes WITHOUT losing either!
        s1.add_peer(s2.node_address)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if s1.hash() == s2.hash() and "arch.auth_mode" in s1.conflicts():
                break
            time.sleep(0.05)

        assert s1.hash() == s2.hash()
        conflicts = s1.conflicts()
        assert "arch.auth_mode" in conflicts
        assert len(conflicts["arch.auth_mode"]) == 2

        # Agent resolves conflict via Client -> both servers clear conflict & converge
        with Client("127.0.0.1", s1.node_address[1], agent_id="lead-reviewer") as resolver:
            seen_conflicts = resolver.conflicts()
            assert "arch.auth_mode" in seen_conflicts
            resolver.mv_resolve("arch.auth_mode", {"scheme": "mtls+jwt", "ttl": 1800})

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if s1.hash() == s2.hash() and not s2.conflicts():
                break
            time.sleep(0.05)

        assert s1.conflicts() == {}
        assert s2.conflicts() == {}
        assert s2.read("arch.auth_mode") == {"scheme": "mtls+jwt", "ttl": 1800}
    finally:
        s1.stop()
        s2.stop()


# ----------------------------------------------------------------------
# Upgraded MCP Server: conflicts, leases, batch, watch, prompts, templates
# ----------------------------------------------------------------------
def test_mcp_conflicts_leases_prompts_and_resources_over_cluster():
    cluster = Cluster({"claude-a", "cursor-b"}, transport=InMemoryTransport())
    mcp_a = ConfluxMCPServer(cluster.node("claude-a"), namespace="swarm.")
    mcp_b = ConfluxMCPServer(cluster.node("cursor-b"), namespace="swarm.")

    # 1. Initialize handshake advertises tools, resources, AND prompts
    init_resp = mcp_a.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )
    caps = init_resp["result"]["capabilities"]
    assert "tools" in caps and "resources" in caps and "prompts" in caps
    assert mcp_a.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    # 2. List tools & prompts
    tools_resp = mcp_a.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert {t["name"] for t in tools_resp["result"]["tools"]} == {t["name"] for t in MCP_TOOLS}

    prompts_resp = mcp_a.handle_message({"jsonrpc": "2.0", "id": 3, "method": "prompts/list"})
    assert {p["name"] for p in prompts_resp["result"]["prompts"]} == {p["name"] for p in MCP_PROMPTS}

    # 3. Concurrent MVRegister writes on claude-a and cursor-b BEFORE gossip
    mcp_a.call_tool("conflux_mv_write", {"key": "api.version", "value": "v2-graphql"})
    mcp_b.call_tool("conflux_mv_write", {"key": "api.version", "value": "v2-rest"})

    # Batch operations on claude-a
    batch_res = mcp_a.call_tool(
        "conflux_batch",
        {
            "operations": [
                {"op": "counter_add", "key": "bugs_fixed", "delta": 2},
                {"op": "set_add", "key": "files_checked", "value": "auth.py", "tag": "t-auth"},
                {"op": "register_set", "key": "phase", "value": "implementation"},
            ]
        },
    )
    assert batch_res["applied_count"] == 3

    mcp_b.call_tool("conflux_counter_add", {"key": "bugs_fixed", "delta": 3})
    mcp_b.call_tool(
        "conflux_task_update",
        {"task_id": "audit-1", "status": "done", "result": {"passed": True}},
    )

    # Gossip merges everything — preserving the MVRegister conflict!
    cluster.gossip_round()

    conflict_report = mcp_a.call_tool("conflux_conflicts", {})
    assert conflict_report["conflict_count"] == 1
    assert "swarm.api.version" in conflict_report["conflicts"]

    # Prompt renders live conflict payload for the LLM to synthesize
    prompt_msg = mcp_b.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/get",
            "params": {"name": "resolve_state_conflicts"},
        }
    )
    prompt_text = prompt_msg["result"]["messages"][0]["content"]["text"]
    assert "v2-graphql" in prompt_text and "v2-rest" in prompt_text

    # Resolve conflict via MCP tool and gossip
    resolve_res = mcp_b.call_tool(
        "conflux_resolve_conflict",
        {"key": "api.version", "resolved_value": "v2-rest-with-graphql-gateway"},
    )
    assert resolve_res["status"] == "resolved"
    cluster.gossip_round()

    assert mcp_a.call_tool("conflux_conflicts", {})["conflict_count"] == 0
    assert mcp_a.call_tool("conflux_read", {"key": "api.version"})["value"] == "v2-rest-with-graphql-gateway"

    # 4. Advisory Resource/File Claim Leases prevent edit collisions
    claim_1 = mcp_a.call_tool(
        "conflux_claim_resource",
        {"resource": "conflux/server.py", "ttl_seconds": 60, "intent": "refactoring gossip loop"},
    )
    assert claim_1["acquired"] is True
    cluster.gossip_round()

    # cursor-b tries to claim the same file while claude-a's lease is active -> conflict!
    claim_2 = mcp_b.call_tool("conflux_claim_resource", {"resource": "conflux/server.py"})
    assert claim_2["acquired"] is False
    assert claim_2["status"] == "conflict"
    assert claim_2["held_by"] == "claude-a"

    # cursor-b cannot release claude-a's lease
    bad_release = mcp_b.call_tool("conflux_release_resource", {"resource": "conflux/server.py"})
    assert bad_release["released"] is False

    # claude-a releases -> cursor-b can now claim cleanly
    ok_release = mcp_a.call_tool("conflux_release_resource", {"resource": "conflux/server.py"})
    assert ok_release["released"] is True
    cluster.gossip_round()

    claim_3 = mcp_b.call_tool("conflux_claim_resource", {"resource": "conflux/server.py"})
    assert claim_3["acquired"] is True
    cluster.gossip_round()

    # 5. Resources & URI Templates (`conflux://state/snapshot`, `conflux://state/conflicts`, `conflux://key/{key}`)
    tmpl_resp = mcp_a.handle_message({"jsonrpc": "2.0", "id": 5, "method": "resources/templates/list"})
    assert tmpl_resp["result"]["resourceTemplates"][0]["uriTemplate"] == "conflux://key/{key}"

    key_res = mcp_a.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "resources/read",
            "params": {"uri": "conflux://key/bugs_fixed"},
        }
    )
    assert json.loads(key_res["result"]["contents"][0]["text"])["value"] == 5

    assert state_hash(cluster.node("claude-a").replica.state) == state_hash(
        cluster.node("cursor-b").replica.state
    )


def test_mcp_watch_wait_and_stdio_framing():
    server = Server("mcp-hub", ("127.0.0.1", 0)).start()
    try:
        mcp = ConfluxMCPServer(server)

        # Background writer triggers watch_wait early wakeup
        def delayed_write():
            time.sleep(0.12)
            mcp.call_tool("conflux_register_set", {"key": "handoff.ready", "value": True})

        t = threading.Thread(target=delayed_write)
        t.start()
        waited = mcp.call_tool(
            "conflux_watch_wait",
            {"key": "handoff.ready", "since_value": None, "timeout_seconds": 3.0},
        )
        t.join()
        assert waited["changed"] is True
        assert waited["value"] is True

        # Content-Length framing test
        req_obj = {"jsonrpc": "2.0", "id": 42, "method": "ping"}
        req_bytes = json.dumps(req_obj).encode("utf-8")
        framed_in = io.BytesIO(f"Content-Length: {len(req_bytes)}\r\n\r\n".encode("ascii") + req_bytes)
        framed_out = io.BytesIO()
        mcp.serve_stdio(stdin=framed_in, stdout=framed_out)
        raw_reply = framed_out.getvalue()
        assert raw_reply.startswith(b"Content-Length:")
        body = json.loads(raw_reply.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
        assert body == {"jsonrpc": "2.0", "id": 42, "result": {}}

        # Subprocess CLI check
        host, port = server.node_address
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "conflux_mv_write",
                        "arguments": {"key": "mcp.mode", "value": "clustered"},
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
        assert out_lines[1]["result"]["structuredContent"]["winner"] == "clustered"
        assert server.read("mcp.mode") == "clustered"
    finally:
        server.stop()
