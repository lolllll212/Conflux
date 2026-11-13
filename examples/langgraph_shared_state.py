"""LangGraph + Conflux shared state integration demo.

Demonstrates two LangGraph workflows running on SEPARATE Conflux nodes in a
gossip cluster:
  1. Node A executes a multi-step research workflow backed by
     ``ConfluxCheckpointSaver`` + ``ConfluxStore``.
  2. Node B receives the replicated state over gossip, resumes the exact same
     LangGraph ``thread_id`` from Node A's checkpoint, and aggregates shared
     CRDT counters and findings with zero locks.

Works both when ``langgraph`` is installed (compiles a real ``StateGraph``)
and in zero-dependency environments (exercises the exact checkpoint + store
adapter lifecycle).

Run from the repo root:
    python examples/langgraph_shared_state.py
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import Cluster, InMemoryTransport, state_hash
from conflux.integrations.langgraph import ConfluxCheckpointSaver, ConfluxStore


class ResearchState(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    stage: str


def _run_with_langgraph(cluster: Cluster) -> None:
    from langgraph.graph import END, START, StateGraph

    store_a = ConfluxStore(cluster.node("planner-node"))
    saver_a = ConfluxCheckpointSaver(cluster.node("planner-node"))

    store_b = ConfluxStore(cluster.node("reviewer-node"))
    saver_b = ConfluxCheckpointSaver(cluster.node("reviewer-node"))

    def gather_sources(state: ResearchState) -> dict:
        store_a.counter_inc(("metrics", "research"), "queries_executed", by=3)
        store_a.put(
            ("findings", "crdt"),
            "paper-1",
            {"title": "Shapiro et al. 2011 — CRDTs", "confidence": 0.98},
        )
        store_a.set_add(("fleet", "contributors"), "active", "planner-node")
        return {
            "notes": [f"Collected 3 primary sources on '{state['topic']}'"],
            "stage": "sources_gathered",
        }

    def synthesize(state: ResearchState) -> dict:
        store_b.counter_inc(("metrics", "research"), "queries_executed", by=2)
        store_b.set_add(("fleet", "contributors"), "active", "reviewer-node")
        item = store_b.get(("findings", "crdt"), "paper-1")
        cited = item.value["title"] if item else "unknown"
        return {
            "notes": [f"Synthesized report citing: {cited}"],
            "stage": "complete",
        }

    # Node A runs step 1 and interrupts before synthesis
    builder_a = StateGraph(ResearchState)
    builder_a.add_node("gather_sources", gather_sources)
    builder_a.add_node("synthesize", synthesize)
    builder_a.add_edge(START, "gather_sources")
    builder_a.add_edge("gather_sources", "synthesize")
    builder_a.add_edge("synthesize", END)
    graph_a = builder_a.compile(
        checkpointer=saver_a,
        store=store_a,
        interrupt_before=["synthesize"],
    )

    thread_cfg = {"configurable": {"thread_id": "research-thread-42"}}
    partial = graph_a.invoke(
        {"topic": "Deterministic Multi-Agent State", "notes": [], "stage": "init"},
        config=thread_cfg,
    )
    print(f"[planner-node] paused at checkpoint stage={partial['stage']!r}")

    # Replicate across the Conflux mesh — no central coordinator or DB lock
    cluster.gossip_round()

    # Node B compiles the same graph against ITS OWN Conflux replica and resumes!
    graph_b = builder_a.compile(checkpointer=saver_b, store=store_b)
    resumed_state = graph_b.get_state(thread_cfg)
    assert resumed_state.values["stage"] == "sources_gathered"
    assert resumed_state.next == ("synthesize",)

    final = graph_b.invoke(None, config=thread_cfg)
    cluster.gossip_round()

    assert final["stage"] == "complete"
    assert len(final["notes"]) == 2
    assert store_a.counter_read(("metrics", "research"), "queries_executed") == 5
    assert store_b.counter_read(("metrics", "research"), "queries_executed") == 5
    assert store_a.set_members(("fleet", "contributors"), "active") == frozenset(
        {"planner-node", "reviewer-node"}
    )


def _run_standalone(cluster: Cluster) -> None:
    saver_a = ConfluxCheckpointSaver(cluster.node("planner-node"))
    store_a = ConfluxStore(cluster.node("planner-node"))
    saver_b = ConfluxCheckpointSaver(cluster.node("reviewer-node"))
    store_b = ConfluxStore(cluster.node("reviewer-node"))

    cfg = {"configurable": {"thread_id": "research-thread-42", "checkpoint_ns": ""}}
    ckpt = {
        "v": 1,
        "id": "0001-gather",
        "ts": "2026-11-13T00:00:00+00:00",
        "channel_values": {"topic": "CRDTs", "stage": "sources_gathered"},
        "channel_versions": {"topic": 1, "stage": 1},
        "versions_seen": {},
    }
    saver_a.put(cfg, ckpt, {"source": "loop", "step": 1}, {"topic": 1, "stage": 1})
    store_a.put(("findings", "crdt"), "paper-1", {"title": "Shapiro et al. 2011"})
    store_a.counter_inc(("metrics", "research"), "queries_executed", by=3)

    cluster.gossip_round()

    loaded = saver_b.get_tuple(cfg)
    assert loaded is not None
    assert loaded.checkpoint["channel_values"]["stage"] == "sources_gathered"
    store_b.counter_inc(("metrics", "research"), "queries_executed", by=2)
    cluster.gossip_round()
    assert store_a.counter_read(("metrics", "research"), "queries_executed") == 5


def main() -> None:
    cluster = Cluster({"planner-node", "reviewer-node"}, transport=InMemoryTransport())
    try:
        import langgraph  # noqa: F401

        mode = "native LangGraph StateGraph"
        _run_with_langgraph(cluster)
    except ImportError:
        mode = "standalone Conflux LangGraph adapter"
        _run_standalone(cluster)

    hash_a = state_hash(cluster.node("planner-node").replica.state)
    hash_b = state_hash(cluster.node("reviewer-node").replica.state)
    assert hash_a == hash_b
    print(f"demo passed: LangGraph ({mode}) converged to state_hash {hash_a[:16]}...")


if __name__ == "__main__":
    main()
