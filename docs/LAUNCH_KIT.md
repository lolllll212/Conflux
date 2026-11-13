# Conflux (`conflux-state`) — Launch & Community Distribution Kit

Ready-to-post technical announcements for **Hacker News (`Show HN`)**, **Reddit**, **X (Twitter) / LinkedIn**, and **Framework Discords**.

---

## 1. Hacker News (`Show HN`)

**Title:**
`Show HN: Conflux – Lock-free CRDT shared state for concurrent AI agents (zero deps, Python)`

**URL:**
`https://github.com/lolllll212/Conflux`

**First Comment (Author Context):**
> Hey HN — I built Conflux (`pip install conflux-state`), a deterministic, lock-free shared state protocol for small crews of autonomous AI agents.
>
> **The problem:** When you run 10–30 concurrent agents (parallel coding agents, research swarms, or multi-node LangGraph/CrewAI workers) that all update shared counters, task queues, capability sets, or hand-off registers, classic row-locked databases either serialize your agents or overwrite each other's state dictionaries. Worse, one hallucinating agent can accidentally clobber keys outside its scope.
>
> **How Conflux works:**
> - **State as a Lattice Join:** Instead of mutating rows, agents emit immutable `Action`s over mathematically convergent CRDTs (`GCounter`, `PNCounter`, `LWWRegister`, `GSet`, `ORSet`, `ORMap`). Merging is commutative, associative, and idempotent (`a ⨆ b = b ⨆ a`), so every node that sees the same set of actions converges to the exact same 64-hex SHA-256 `state_hash`, regardless of arrival order.
> - **Signed Identity & Namespace Sandboxing:** Every agent write can be signed with HMAC-SHA256 (stdlib, zero deps) or Ed25519, with per-agent op allowlists, write quotas, and key-prefix namespaces enforced *before* merge.
> - **Durable & Crash-Recoverable:** Append-only fsynced journals (optional HMAC-signed lines to prevent tampering) + cumulative lattice-merged snapshots survive `kill -9` and torn tails.
> - **Zero-Dependency Core + Drop-In Ecosystem Adapters:** Framed TCP and RFC6455 WebSocket transports (+ TLS and Prometheus `/metrics`) are implemented purely on the Python stdlib (~168k actions/s local replay, ~0.13ms TCP round-trip). On top of that, `conflux.mcp` provides a stdio MCP server for Claude Code / Cursor, and `conflux.integrations` ships drop-in adapters for LangGraph (`BaseCheckpointSaver` + `BaseStore`), CrewAI (`StorageBackend` + `FlowPersistence`), and AutoGen (`Memory` + `FunctionTool`).
>
> Protocol spec (`docs/PROTOCOL.md`) and security threat model (`docs/SECURITY.md`) map every claim directly to conformance tests. Would love feedback on the protocol design, failure model, and CRDT trade-offs!

---

## 2. Reddit (`r/AI_Agents`, `r/LangChain`, `r/LocalLLaMA`, `r/Python`)

**Title:**
`I open-sourced Conflux: a lock-free CRDT shared state backend + MCP server for parallel LangGraph, CrewAI, AutoGen, and Claude Code agents`

**Body:**
Ever had two parallel LangGraph threads, CrewAI workers, or Claude Code instances overwrite each other's shared state or hit SQLite `database is locked` errors?

I built **Conflux** (`pip install conflux-state`) to solve multi-agent shared state without central coordinators or database locks:

- **Conflict-Free Convergence (CRDTs):** Agents write concurrently to shared counters (`PNCounter`), registers (`LWWRegister`), and sets (`ORSet`). Every replica merges updates deterministically to the exact same `state_hash`.
- **Drop-In Adapters Included:**
  - **MCP Server (`conflux-mcp`):** Connect multiple Claude Code / Cursor / Windsurf sessions to one convergent state mesh.
  - **LangGraph:** `ConfluxCheckpointSaver` (pause on Node A, resume on Node B over gossip!) + `ConfluxStore`.
  - **CrewAI:** `ConfluxCrewStorage` (`StorageBackend` protocol) + `ConfluxFlowPersistence` + CRDT tools.
  - **AutoGen:** `ConfluxAutoGenMemory` + `FunctionTool`s.
- **Per-Agent Security:** HMAC-SHA256 / Ed25519 signed actions + namespace sandboxing + write quotas so a hallucinating agent can't corrupt another team's keys.
- **Zero Core Dependencies:** Pure Python stdlib core with TCP, WebSocket, TLS, crash-safe journals, and Prometheus metrics.

- **GitHub:** https://github.com/lolllll212/Conflux
- **PyPI:** `pip install conflux-state`
- **Interactive Colab Quickstart:** `examples/quickstart_colab.ipynb`

Happy to answer any questions about CRDTs in agent orchestration!

---

## 3. X (Twitter) / LinkedIn Launch Thread

**Post 1:**
Multi-agent AI breaks down the moment 10 parallel agents try to mutate the same shared state:
- SQLite locks up
- Dict merges silently clobber updates
- Hallucinating agents overwrite keys they shouldn't touch

Meet **Conflux** (`pip install conflux-state`) — a deterministic, lock-free CRDT state fabric for concurrent AI agents.

GitHub: https://github.com/lolllll212/Conflux

**Post 2:**
Instead of row locks, Conflux models agent state as a join-semilattice (`PNCounter`, `LWWRegister`, `ORSet`, `ORMap`).

Agents write concurrently with zero coordination. Every node that absorbs the same actions converges to the exact same SHA-256 `state_hash` — in any delivery order, even across partitions and `kill -9` crashes.

**Post 3:**
Works out of the box with your existing stack (zero core dependencies):
- `conflux-mcp` -> Shared CRDT state for parallel Claude Code & Cursor agents
- LangGraph -> `ConfluxCheckpointSaver` + `ConfluxStore`
- CrewAI -> `ConfluxCrewStorage` + `ConfluxFlowPersistence`
- AutoGen -> `ConfluxAutoGenMemory` + `FunctionTool`s

Try it in 60 seconds on Colab or install now:
`pip install conflux-state`

---

## 4. LangGraph / CrewAI / AutoGen Discord (`#share-your-work`)

> Built a zero-dependency, CRDT-backed shared state & checkpointing integration for **[LangGraph / CrewAI / AutoGen]**: **Conflux (`pip install conflux-state`)**.
>
> It lets parallel agents across different processes or machines share counters, sets, checkpoints, and scoped memories over TCP/WebSocket gossip with **zero locks**, cryptographic per-agent namespace guards, and deterministic replay.
>
> - Repo & runnable examples: https://github.com/lolllll212/Conflux
> - PyPI: https://pypi.org/project/conflux-state/
