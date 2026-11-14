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
        "silent_overwrite_benchmark.py",
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
    assert "passed:" in proc.stdout


def test_real_session_hazards_thread_safety_batch_tags_and_large_checkpoints():
    """Regression suite covering real-world multi-agent session failure modes:
    1. Multi-threaded parallel Client I/O (LangGraph/CrewAI/AutoGen asyncio.to_thread).
    2. Multiple set_add() + mv_set() inside a single Client.batch() without tag collision.
    3. Restarted Client with identical agent_id overwriting prior LWW values cleanly.
    4. Large LangGraph checkpoints (>100 KB) & deep (>256-char) subgraph namespaces over TCP.
    5. Conflict-preserving MVRegister tools across LangGraph, CrewAI, and AutoGen adapters.
    """
    import concurrent.futures
    from conflux import Client, Server
    from conflux.integrations.autogen import create_autogen_tools
    from conflux.integrations.crewai import create_crewai_tools

    server = Server("session-hub", ("127.0.0.1", 0)).start()
    port = server.node_address[1]
    try:
        # 1. Parallel threads hammering a single shared Client instance simultaneously
        with Client("127.0.0.1", port, agent_id="parallel-agent") as shared_client:
            def worker(idx: int):
                shared_client.counter_inc("session.parallel_ops", 1)
                shared_client.register_set(f"session.slot.{idx % 5}", {"worker": idx})
                return shared_client.read("session.parallel_ops")

            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(worker, range(48)))
            assert len(results) == 48
            assert shared_client.read("session.parallel_ops") == 48

            # 2. Multiple set_add() + mv_set() in one atomic batch must NOT collide tags
            with shared_client.batch() as b:
                b.set_add("session.artifacts", "plan.md")
                b.set_add("session.artifacts", "schema.sql")
                b.set_add("session.artifacts", "tests.py")
                b.mv_set("session.spec", {"v": 1})
            assert sorted(shared_client.read("session.artifacts")) == [
                "plan.md",
                "schema.sql",
                "tests.py",
            ]
            assert shared_client.read("session.spec") == {"v": 1}

            # Write initial status before simulating agent restart
            shared_client.register_set("session.lifecycle", "phase-1-before-restart")

        # 3. Restarted process reconnecting with the SAME agent_id must win LWW & generate fresh action_id
        with Client("127.0.0.1", port, agent_id="parallel-agent") as restarted_client:
            restarted_client.register_set("session.lifecycle", "phase-2-after-restart")
            assert restarted_client.read("session.lifecycle") == "phase-2-after-restart"

            # 4. Large LangGraph checkpoint (>120 KB) + deep nested subgraph namespace (>260 chars) over live TCP Client
            saver = ConfluxCheckpointSaver(restarted_client)
            deep_ns = "/".join(f"subgraph_stage_{i}_agent_executor_node" for i in range(10))
            assert len(deep_ns) > 260
            cfg = {"configurable": {"thread_id": "prod-session-thread-9999", "checkpoint_ns": deep_ns}}
            big_messages = [{"role": "assistant", "content": "reasoning " * 1500} for _ in range(8)]
            ckpt = {
                "v": 1,
                "id": "1ef99999-0001-6000-8000-000000000001",
                "ts": "2026-11-13T12:00:00+00:00",
                "channel_values": {"messages": big_messages, "step": 4},
                "channel_versions": {"messages": "4", "step": "4"},
                "versions_seen": {},
                "pending_sends": [],
            }
            saved_cfg = saver.put(cfg, ckpt, {"source": "loop", "step": 4}, {"messages": "4", "step": "4"})
            loaded = saver.get_tuple(saved_cfg)
            assert loaded is not None
            assert len(loaded.checkpoint["channel_values"]["messages"]) == 8

            # 5. MVRegister conflict workflow across LangGraph Store, CrewAI tools, and AutoGen tools
            store_a = ConfluxStore(restarted_client)
            with Client("127.0.0.1", port, agent_id="peer-agent-b") as client_b:
                # Concurrent MV writes (peer-agent-b supersedes=[], simulating offline/unsynced proposal)
                store_a.mv_put(("arch", "rfcs"), "db_choice", {"engine": "postgres"})
                client_b.mv_set("lg.store/mv/arch/rfcs:db_choice", {"engine": "sqlite-crdt"}, supersedes=[])
                conflicts = store_a.mv_conflicts(("arch",))
                assert len(conflicts) == 1
                resolved = store_a.mv_resolve(("arch", "rfcs"), "db_choice", {"engine": "conflux+postgres"})
                assert resolved == {"engine": "conflux+postgres"}
                assert len(store_a.mv_conflicts(("arch",))) == 0

                crew_tools = {t.name: t for t in create_crewai_tools(restarted_client, namespace="crew.")}
                assert "conflux_mv_write" in crew_tools and "conflux_resolve_conflict" in crew_tools
                crew_tools["conflux_mv_write"].run("plan", {"owner": "researcher"})

                ag_tools = {
                    (getattr(t, "name", None) or t.__name__): t
                    for t in create_autogen_tools(client_b, namespace="ag.")
                }
                assert "conflux_mv_write" in ag_tools and "conflux_conflicts" in ag_tools
    finally:
        server.stop()


def test_five_technical_risks_lamport_clock_async_client_alias_and_compaction(tmp_path):
    """Verifies the 5 technical audit pillars:
    1. Pure Lamport clock sync (zero wall-clock / time.time() drift sensitivity).
    2. Non-blocking asyncio methods on Client & BackendAdapter (`aread`, `acounter_inc`, `async with`).
    3. Package alias (`import conflux_state`) + PEP 561 `py.typed` markers.
    4. Strict JSON-only payload validation (arbitrary Python objects rejected).
    5. Tombstone compaction (`Server.compact_state()`) & bounded memory journal on durable servers.
    """
    import conflux
    import conflux_state
    from conflux import Client, Server, validate_action, Action, ValidationError

    # Risk 4: Package alias & PEP 561 py.typed markers
    assert conflux_state.__version__ == conflux.__version__
    assert conflux_state.Server is conflux.Server
    assert os.path.exists(os.path.join(REPO_ROOT, "conflux", "py.typed"))
    assert os.path.exists(os.path.join(REPO_ROOT, "conflux_state", "py.typed"))

    # Risk 5a: Arbitrary non-JSON Python objects strictly rejected
    with pytest.raises(ValidationError):
        validate_action(
            Action("act-a-1", "a", 1, "lww_set", "bad", {"value": object()})
        )

    # Risk 3: Pure Lamport logical clock sync — even if a client's local tick starts at 0
    # and another node wrote at tick 5000, connecting & reading advances the causal clock!
    data_dir = str(tmp_path / "durable")
    srv = Server("lamport-hub", ("127.0.0.1", 0), data_dir=data_dir, snapshot_threshold=5).start()
    port = srv.node_address[1]
    try:
        # Simulate an agent with a high logical tick (5000) writing first
        high_tick_action = Action("act-fast-5000", "fast-clock-agent", 5000, "lww_set", "leader", {"value": "old"})
        srv._absorb([high_tick_action])
        assert srv.read("leader") == "old"

        # Now a brand-new client connects (zero wall-clock reliance!), observes the key via read(),
        # and causally supersedes it without depending on system clocks!
        with Client("127.0.0.1", port, agent_id="fresh-agent") as c:
            assert c.read("leader") == "old"
            assert c.replica.timestamp() >= 5000
            c.register_set("leader", "new-causal-winner")
            assert c.read("leader") == "new-causal-winner"

            # Add & remove ORSet tags, then run tombstone compaction
            c.set_add("jobs", "j1", tag="t-j1")
            c.set_add("jobs", "j2", tag="t-j2")
            c.set_remove("jobs", "t-j1")
            assert c.read("jobs") == ["j2"]
            pruned = srv.compact_state()
            assert pruned == 1
            assert c.read("jobs") == ["j2"]

        # Risk 2: Non-blocking asyncio Client & BackendAdapter (`async with`, `acounter_inc`, `aread`)
        async def _async_workout():
            async with Client("127.0.0.1", port, agent_id="async-fastapi") as ac:
                await asyncio.gather(
                    ac.acounter_inc("async.hits", 1),
                    ac.acounter_inc("async.hits", 2),
                    ac.aregister_set("async.status", "non-blocking"),
                )
                assert await ac.aread("async.hits") == 3
                assert await ac.aread("async.status") == "non-blocking"

        asyncio.run(_async_workout())
    finally:
        srv.stop()

    # Restart durable server from compacted snapshot and verify Lamport clock + compacted state survive!
    srv2 = Server("lamport-hub", ("127.0.0.1", 0), data_dir=data_dir).start()
    try:
        assert srv2.replica.timestamp() >= 5001
        assert srv2.read("leader") == "new-causal-winner"
        assert srv2.read("jobs") == frozenset({"j2"})
        assert len(srv2.replica.state._map.get("jobs").tombstones) == 0
    finally:
        srv2.stop()


