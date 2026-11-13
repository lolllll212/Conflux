"""CrewAI + Conflux shared state & memory integration demo.

Demonstrates a 3-agent CrewAI deployment sharing state over a 2-node signed
Conflux TCP mesh:
  1. ``ConfluxCrewStorage`` replicates hierarchical agent memories (``/research``,
     ``/security``) across nodes with cosine-similarity + scope recall.
  2. ``ConfluxFlowPersistence`` checkpoints CrewAI Flow state so any node can
     recover and resume an in-flight Flow.
  3. ``create_crewai_tools`` equips CrewAI agents with lock-free CRDT tools
     (counters, LWW registers, ORSets) signed per-agent via ``AgentRegistry``.

Run from the repo root:
    python examples/crewai_shared_memory.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import AgentRegistry, Client, Server
from conflux.integrations.crewai import (
    ConfluxCrewStorage,
    ConfluxFlowPersistence,
    FallbackMemoryRecord,
    create_crewai_tools,
)


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def main() -> None:
    registry = AgentRegistry()
    registry.register("researcher", "res-secret", namespaces=("crew.",))
    registry.register("auditor", "aud-secret", namespaces=("crew.",))

    node_1 = Server("crew-node-1", ("127.0.0.1", 0), registry=registry, gossip_interval=0.08).start()
    node_2 = Server("crew-node-2", ("127.0.0.1", 0), registry=registry, gossip_interval=0.08).start()
    node_1.add_peer(node_2.node_address)

    try:
        with (
            Client("127.0.0.1", node_1.node_address[1], agent_id="researcher", secret="res-secret") as c_res,
            Client("127.0.0.1", node_2.node_address[1], agent_id="auditor", secret="aud-secret") as c_aud,
        ):
            # 1. Shared CrewAI Memory Storage across two networked nodes
            mem_res = ConfluxCrewStorage(c_res, prefix="crew.mem/")
            mem_aud = ConfluxCrewStorage(c_aud, prefix="crew.mem/")

            mem_res.save(
                [
                    FallbackMemoryRecord(
                        id="mem-1",
                        content="CRDT join-semilattices guarantee lock-free convergence.",
                        scope="/engineering/architecture",
                        categories=["crdt", "architecture"],
                        metadata={"verified": True},
                        importance=0.95,
                        embedding=[0.9, 0.1, 0.0],
                    ),
                    FallbackMemoryRecord(
                        id="mem-2",
                        content="HMAC-SHA256 and Ed25519 envelopes guard the agent submit boundary.",
                        scope="/engineering/security",
                        categories=["security", "signing"],
                        metadata={"verified": True},
                        importance=0.9,
                        embedding=[0.1, 0.9, 0.0],
                    ),
                ]
            )

            # 2. CrewAI Flow persistence across nodes
            flow_res = ConfluxFlowPersistence(c_res, prefix="crew.flow/")
            flow_aud = ConfluxFlowPersistence(c_aud, prefix="crew.flow/")
            flow_res.save_state(
                "release-flow-01",
                "draft_architecture",
                {"phase": "security_review", "approved_components": ["crdts", "storage"]},
            )

            # 3. CrewAI Agent CRDT tools writing concurrently on different nodes
            tools_res = {t.name: t for t in create_crewai_tools(c_res, namespace="crew.")}
            tools_aud = {t.name: t for t in create_crewai_tools(c_aud, namespace="crew.")}

            tools_res["conflux_increment_counter"].run(key="tasks_completed", by=2)
            tools_res["conflux_add_to_set"].run(key="active_agents", item="researcher")

            tools_aud["conflux_increment_counter"].run(key="tasks_completed", by=1)
            tools_aud["conflux_add_to_set"].run(key="active_agents", item="auditor")
            tools_aud["conflux_write_register"].run(key="release_gate", value="approved")

            # Wait for gossip convergence across TCP nodes
            assert _wait_until(lambda: node_1.hash() == node_2.hash() and node_1.read("crew.tasks_completed") == 3)

            # Auditor on node_2 recalls memories saved by Researcher on node_1
            matches = mem_aud.search([0.85, 0.15, 0.0], scope_prefix="/engineering")
            assert len(matches) == 2
            top_record, top_score = matches[0]
            assert top_record.id == "mem-1" and top_score > 0.9

            # Auditor resumes the persisted CrewAI Flow on node_2
            restored_flow = flow_aud.load_state("release-flow-01")
            assert restored_flow == {
                "phase": "security_review",
                "approved_components": ["crdts", "storage"],
            }

            gate = json.loads(tools_res["conflux_read_state"].run(key="release_gate"))
            assert gate["value"] == "approved"

            print(
                f"demo passed: CrewAI shared memory, flow persistence & CRDT tools "
                f"converged (state_hash={node_1.hash()[:16]}...)"
            )
    finally:
        node_1.stop()
        node_2.stop()


if __name__ == "__main__":
    main()
