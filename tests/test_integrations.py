"""Comprehensive tests for Conflux framework integrations (LangGraph, CrewAI, AutoGen)."""

from __future__ import annotations

import asyncio
import json
import operator
import os
import subprocess
import sys
from typing import Annotated, TypedDict

import pytest

from conflux import AgentRegistry, Client, Cluster, InMemoryTransport, Replica, Server, state_hash
from conflux.integrations import (
    BackendAdapter,
    ConfluxAutoGenMemory,
    ConfluxCheckpointSaver,
    ConfluxCrewStorage,
    ConfluxFlowPersistence,
    ConfluxStore,
    FallbackMemoryRecord,
    create_autogen_tools,
    create_crewai_tools,
)
from conflux.integrations.autogen import MemoryContent, MemoryMimeType

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _NativeGraphState(TypedDict):
    log: Annotated[list[str], operator.add]
    done: bool


# ----------------------------------------------------------------------
# BackendAdapter parity across Replica, DistributedApp, Server, Client
# ----------------------------------------------------------------------
def test_backend_adapter_across_all_targets():
    replica = Replica("r1")
    b_rep = BackendAdapter(replica)
    b_rep.counter_inc("k.cnt", 4)
    b_rep.register_set("k.reg", {"ok": True})
    b_rep.set_add("k.set", "alpha")
    assert b_rep.read("k.cnt") == 4
    assert b_rep.read("k.reg") == {"ok": True}
    assert b_rep.read_set("k.set") == frozenset({"alpha"})
    assert len(b_rep.hash()) == 64

    server = Server("srv-1", ("127.0.0.1", 0)).start()
    try:
        b_srv = BackendAdapter(server)
        b_srv.counter_inc("s.cnt", 2)
        b_srv.register_set("s.reg", "live")
        b_srv.set_add("s.set", "m1")

        with Client("127.0.0.1", server.node_address[1], agent_id="cli-1") as client:
            b_cli = BackendAdapter(client)
            assert b_cli.read("s.cnt") == 2
            assert b_cli.read("s.reg") == "live"
            assert b_cli.read_set("s.set") == frozenset({"m1"})

            b_cli.counter_inc("s.cnt", 3)
            b_cli.set_add("s.set", "m2")
            assert b_srv.read("s.cnt") == 5
            assert b_srv.read_set("s.set") == frozenset({"m1", "m2"})
            assert b_cli.hash() == b_srv.hash()
    finally:
        server.stop()


# ----------------------------------------------------------------------
# LangGraph: ConfluxCheckpointSaver + ConfluxStore
# ----------------------------------------------------------------------
def test_langgraph_checkpoint_saver_lifecycle_and_convergence():
    cluster = Cluster({"lg-1", "lg-2"}, transport=InMemoryTransport())
    saver_1 = ConfluxCheckpointSaver(cluster.node("lg-1"))
    saver_2 = ConfluxCheckpointSaver(cluster.node("lg-2"))

    cfg = {"configurable": {"thread_id": "thread-alpha", "checkpoint_ns": ""}}
    ckpt_1 = {
        "v": 1,
        "id": "0001",
        "ts": "2026-11-13T01:00:00+00:00",
        "channel_values": {"messages": ["hello"], "count": 1},
        "channel_versions": {"messages": 1, "count": 1},
        "versions_seen": {},
    }
    cfg_after_1 = saver_1.put(cfg, ckpt_1, {"source": "input", "step": 0}, {"messages": 1, "count": 1})
    saver_1.put_writes(cfg_after_1, [("messages", ["world"])], task_id="task-a")

    ckpt_2 = {
        "v": 1,
        "id": "0002",
        "ts": "2026-11-13T01:00:01+00:00",
        "channel_values": {"messages": ["hello", "world"], "count": 2},
        "channel_versions": {"messages": 2, "count": 1},
        "versions_seen": {},
    }
    saver_1.put(cfg_after_1, ckpt_2, {"source": "loop", "step": 1}, {"messages": 2})

    cluster.gossip_round()

    # Latest checkpoint retrieved on peer node lg-2
    latest = saver_2.get_tuple({"configurable": {"thread_id": "thread-alpha"}})
    assert latest is not None
    assert latest.checkpoint["id"] == "0002"
    assert latest.checkpoint["channel_values"]["messages"] == ["hello", "world"]
    assert latest.checkpoint["channel_values"]["count"] == 2
    assert latest.parent_config["configurable"]["checkpoint_id"] == "0001"

    # Specific checkpoint + pending writes retrieved on peer node lg-2
    first = saver_2.get_tuple(cfg_after_1)
    assert first is not None
    assert first.pending_writes == [("task-a", "messages", ["world"])]

    # Listing with metadata filter & limit
    listed = list(saver_2.list({"configurable": {"thread_id": "thread-alpha"}}))
    assert [t.checkpoint["id"] for t in listed] == ["0002", "0001"]
    filtered = list(saver_2.list(None, filter={"source": "input"}))
    assert len(filtered) == 1 and filtered[0].checkpoint["id"] == "0001"

    # Delete thread propagates cleanly
    saver_2.delete_thread("thread-alpha")
    cluster.gossip_round()
    assert saver_1.get_tuple({"configurable": {"thread_id": "thread-alpha"}}) is None


def test_langgraph_store_and_compiled_stategraph():
    cluster = Cluster({"node-a", "node-b"}, transport=InMemoryTransport())
    store_a = ConfluxStore(cluster.node("node-a"))
    store_b = ConfluxStore(cluster.node("node-b"))

    store_a._do_put(("org", "eng"), "doc-1", {"topic": "crdt lattice", "tier": 1})
    store_a._do_put(("org", "sec"), "doc-2", {"topic": "ed25519 signatures", "tier": 2})
    store_a.counter_inc(("org", "metrics"), "writes", by=4)
    store_b.counter_inc(("org", "metrics"), "writes", by=6)

    cluster.gossip_round()

    assert store_a.counter_read(("org", "metrics"), "writes") == 10
    assert store_b.counter_read(("org", "metrics"), "writes") == 10

    got = store_b._do_get(("org", "eng"), "doc-1")
    assert got is not None and got.value["topic"] == "crdt lattice"

    hits = store_b._do_search(("org",), query="ed25519")
    assert len(hits) == 1 and hits[0].key == "doc-2"

    namespaces = store_b._do_list_namespaces(prefix=("org",))
    assert namespaces == [("org", "eng"), ("org", "sec")]

    # If native langgraph is installed in the test venv, verify end-to-end StateGraph execution
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph

    saver_a = ConfluxCheckpointSaver(cluster.node("node-a"))
    saver_b = ConfluxCheckpointSaver(cluster.node("node-b"))

    builder = StateGraph(_NativeGraphState)
    builder.add_node("step_one", lambda s: {"log": ["one"], "done": False})
    builder.add_node("step_two", lambda s: {"log": ["two"], "done": True})
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)

    g_a = builder.compile(checkpointer=saver_a, store=store_a, interrupt_before=["step_two"])
    cfg = {"configurable": {"thread_id": "native-graph-1"}}
    res_1 = g_a.invoke({"log": [], "done": False}, config=cfg)
    assert res_1 == {"log": ["one"], "done": False}

    cluster.gossip_round()

    g_b = builder.compile(checkpointer=saver_b, store=store_b)
    res_2 = g_b.invoke(None, config=cfg)
    assert res_2 == {"log": ["one", "two"], "done": True}


# ----------------------------------------------------------------------
# CrewAI: ConfluxCrewStorage, ConfluxFlowPersistence, create_crewai_tools
# ----------------------------------------------------------------------
def test_crewai_storage_flow_and_tools_convergence():
    cluster = Cluster({"crew-1", "crew-2"}, transport=InMemoryTransport())
    storage_1 = ConfluxCrewStorage(cluster.node("crew-1"))
    storage_2 = ConfluxCrewStorage(cluster.node("crew-2"))

    storage_1.save(
        [
            FallbackMemoryRecord(
                id="r1",
                content="Deploy canary to us-west first",
                scope="/ops/deploy",
                categories=["runbook", "ops"],
                metadata={"env": "prod"},
                importance=0.9,
                embedding=[1.0, 0.0, 0.0],
            ),
            FallbackMemoryRecord(
                id="r2",
                content="Rotate journal HMAC keys quarterly",
                scope="/ops/security",
                categories=["security"],
                metadata={"env": "prod"},
                importance=0.8,
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )

    flow_1 = ConfluxFlowPersistence(cluster.node("crew-1"))
    flow_2 = ConfluxFlowPersistence(cluster.node("crew-2"))
    flow_1.save_state("flow-99", "kickoff", {"step": "canary", "healthy": True})

    tools_1 = {t.name: t for t in create_crewai_tools(cluster.node("crew-1"))}
    tools_2 = {t.name: t for t in create_crewai_tools(cluster.node("crew-2"))}
    tools_1["conflux_increment_counter"].run("jobs", 3)
    tools_2["conflux_increment_counter"].run("jobs", 4)
    tools_1["conflux_add_to_set"].run("online", "agent-1")
    tools_2["conflux_add_to_set"].run("online", "agent-2")

    cluster.gossip_round()

    assert storage_2.count() == 2
    assert storage_2.list_scopes("/ops") == ["/ops/deploy", "/ops/security"]
    assert storage_2.list_categories("/ops") == {"runbook": 1, "ops": 1, "security": 1}
    info = storage_2.get_scope_info("/ops")
    assert info.record_count == 2
    assert info.child_scopes == ["/ops/deploy", "/ops/security"]

    vec_hits = storage_2.search([0.95, 0.05, 0.0], scope_prefix="/ops")
    assert vec_hits[0][0].id == "r1" and vec_hits[0][1] > 0.95

    text_hits = storage_2.search("HMAC quarterly", scope_prefix="/ops")
    assert len(text_hits) == 1 and text_hits[0][0].id == "r2"

    assert flow_2.load_state("flow-99") == {"step": "canary", "healthy": True}

    jobs_payload = json.loads(tools_2["conflux_read_state"].run("jobs"))
    assert jobs_payload["value"] == 7
    online_payload = json.loads(tools_1["conflux_read_state"].run("online"))
    assert online_payload["value"] == ["agent-1", "agent-2"]

    deleted = storage_2.delete(record_ids=["r1"])
    assert deleted == 1
    cluster.gossip_round()
    assert storage_1.get_record("r1") is None
    assert storage_1.count() == 1


# ----------------------------------------------------------------------
# AutoGen: ConfluxAutoGenMemory + create_autogen_tools
# ----------------------------------------------------------------------
def test_autogen_memory_and_tools_convergence():
    class DummyContext:
        def __init__(self):
            self.messages = []

        async def add_message(self, msg):
            self.messages.append(msg)

    async def _exercise():
        cluster = Cluster({"ag-1", "ag-2"}, transport=InMemoryTransport())
        mem_1 = ConfluxAutoGenMemory(cluster.node("ag-1"), state_watch_prefix="team.")
        mem_2 = ConfluxAutoGenMemory(cluster.node("ag-2"), state_watch_prefix="team.")

        tools_1 = create_autogen_tools(cluster.node("ag-1"), namespace="team.")
        tools_2 = create_autogen_tools(cluster.node("ag-2"), namespace="team.")

        def call_tool(tools, name, **kw):
            for t in tools:
                tname = getattr(t, "name", None) or getattr(t, "__name__", "")
                if tname == name:
                    return t._func(**kw) if hasattr(t, "_func") else t(**kw)
            raise KeyError(name)

        await mem_1.add(
            MemoryContent(content="Service A latency spiked to 420ms", mime_type=MemoryMimeType.TEXT)
        )
        await mem_2.add(
            MemoryContent(
                content="Cache hit ratio recovered after warmup", mime_type=MemoryMimeType.TEXT
            )
        )
        call_tool(tools_1, "conflux_increment_counter", key="checks", by=2)
        call_tool(tools_2, "conflux_increment_counter", key="checks", by=5)
        call_tool(tools_1, "conflux_set_register", key="leader", value='"ag-1"')

        cluster.gossip_round()

        q = await mem_1.query("latency")
        assert len(q.results) == 1 and "420ms" in q.results[0].content

        ctx = DummyContext()
        await mem_2.update_context(ctx)
        assert len(ctx.messages) == 1
        body = ctx.messages[0].content
        assert "420ms" in body and "Cache hit ratio" in body
        assert "team.checks: 7" in body

        await mem_1.clear()
        cluster.gossip_round()
        assert len(mem_2.content) == 0

        assert state_hash(cluster.node("ag-1").replica.state) == state_hash(
            cluster.node("ag-2").replica.state
        )

    asyncio.run(_exercise())


# ----------------------------------------------------------------------
# Runnable framework demo scripts
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "script",
    [
        "langgraph_shared_state.py",
        "crewai_shared_memory.py",
        "autogen_shared_state.py",
    ],
)
def test_framework_integration_demos_run_clean(script):
    proc = subprocess.run(
        [sys.executable, os.path.join("examples", script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "demo passed:" in proc.stdout
