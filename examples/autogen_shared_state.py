"""AutoGen + Conflux shared state & memory integration demo.

Demonstrates AutoGen agents coordinating across a Conflux gossip cluster:
  1. ``ConfluxAutoGenMemory`` implements ``autogen_core.memory.Memory`` so
     converged facts and live CRDT state keys are injected directly into each
     agent's ``ChatCompletionContext``.
  2. ``create_autogen_tools`` produces native ``FunctionTool`` instances (when
     ``autogen-core`` is installed) so concurrent AutoGen agents mutate shared
     CRDT counters, registers, and sets without central locking.

Works out of the box both when ``autogen-agentchat`` / ``autogen-core`` are
installed and in zero-dependency environments.

Run from the repo root:
    python examples/autogen_shared_state.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import Cluster, InMemoryTransport, state_hash
from conflux.integrations.autogen import (
    ConfluxAutoGenMemory,
    MemoryContent,
    MemoryMimeType,
    create_autogen_tools,
)


class _SimpleContext:
    """Fallback ChatCompletionContext used only when autogen-core is not installed."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    async def add_message(self, msg: object) -> None:
        self.messages.append(msg)

    async def get_messages(self) -> list[object]:
        return list(self.messages)


async def _run() -> None:
    cluster = Cluster({"coder-agent", "verifier-agent"}, transport=InMemoryTransport())

    mem_coder = ConfluxAutoGenMemory(
        cluster.node("coder-agent"),
        name="incident_war_room",
        state_watch_prefix="team.",
    )
    mem_verifier = ConfluxAutoGenMemory(
        cluster.node("verifier-agent"),
        name="incident_war_room",
        state_watch_prefix="team.",
    )

    coder_tools = create_autogen_tools(cluster.node("coder-agent"), namespace="team.")
    verifier_tools = create_autogen_tools(cluster.node("verifier-agent"), namespace="team.")

    def _invoke(tool_list, name: str, **kwargs):
        for t in tool_list:
            t_name = getattr(t, "name", None) or getattr(t, "__name__", "")
            if t_name == name:
                if hasattr(t, "_func"):
                    return t._func(**kwargs)
                return t(**kwargs)
        raise KeyError(name)

    # Coder agent records findings in AutoGen Memory and updates CRDT tools
    await mem_coder.add(
        MemoryContent(
            content="Root cause isolated: retry storm on upstream payment gateway.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"severity": "sev-2"},
        )
    )
    _invoke(coder_tools, "conflux_increment_counter", key="patches_proposed", by=2)
    _invoke(coder_tools, "conflux_add_set_member", key="mitigations", member="circuit-breaker")
    _invoke(coder_tools, "conflux_set_register", key="incident_status", value='"mitigating"')

    # Verifier agent concurrently mutates its own replica BEFORE gossip!
    await mem_verifier.add(
        MemoryContent(
            content="Canary metrics clean at 10% traffic; error rate < 0.01%.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"canary": "us-east"},
        )
    )
    _invoke(verifier_tools, "conflux_increment_counter", key="patches_proposed", by=1)
    _invoke(verifier_tools, "conflux_add_set_member", key="mitigations", member="rate-limiter")

    # Anti-entropy gossip merges both agents' memories, counters, and sets
    cluster.gossip_round()

    try:
        from autogen_core.model_context import BufferedChatCompletionContext

        ctx = BufferedChatCompletionContext(buffer_size=10)
        mode = "native autogen-core Memory + FunctionTool"
    except ImportError:
        ctx = _SimpleContext()
        mode = "standalone Conflux AutoGen adapter"

    update_res = await mem_verifier.update_context(ctx)
    assert len(update_res.memories.results) == 2

    messages = await ctx.get_messages()
    assert len(messages) == 1
    injected_text = messages[0].content
    assert "retry storm" in injected_text
    assert "Canary metrics clean" in injected_text
    assert "team.patches_proposed: 3" in injected_text

    snap = json.loads(_invoke(coder_tools, "conflux_snapshot"))
    assert snap["state"]["team.patches_proposed"] == 3
    assert snap["state"]["team.mitigations"] == ["circuit-breaker", "rate-limiter"]

    hash_coder = state_hash(cluster.node("coder-agent").replica.state)
    hash_verifier = state_hash(cluster.node("verifier-agent").replica.state)
    assert hash_coder == hash_verifier
    print(f"demo passed: AutoGen ({mode}) converged to state_hash {hash_coder[:16]}...")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
