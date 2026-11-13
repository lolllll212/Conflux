# Conflux (`conflux-state`) — Step 4 Full Launch, Benchmark & Community Playbook

This playbook contains the **complete, copy-paste execution guide** for launching Conflux across **Technical Blogs (Dev.to / Hashnode / Medium / Substack)**, **Hacker News (`Show HN`)**, **Reddit (`r/ClaudeCode`, `r/cursor`, `r/LangChain`, `r/AI_Agents`, `r/Python`)**, and **X / LinkedIn**—backed by the reproducible **`examples/silent_overwrite_benchmark.py`** script.

---

## 0. Run the Reproducible Benchmark First (Your Viral Proof)

Anyone can verify the benchmark in **1.1 seconds** using nothing but the Python standard library:

```bash
pip install conflux-state
python examples/silent_overwrite_benchmark.py
```

### Actual Benchmark Output (8 Parallel Agents × 25 Ops = 200 Concurrent Updates)

```text
==============================================================================
CONFLUX MULTI-AGENT CONCURRENCY BENCHMARK (8 parallel agents x 25 ops)
==============================================================================

[1] Shared Counter & Artifact Set (Read-Modify-Write JSON vs CRDTs)
  Naive JSON KV (Redis/DB LWW) : 25/200 increments kept (87.5% SILENT DATA LOSS), 25/200 files kept
  Conflux PNCounter + ORSet    : 200/200 increments kept (0.0% loss), 200/200 files kept

[2] Parallel Architectural Spec Proposals (8 Partitioned Agents)
  Naive Last-Write-Wins        : 1/8 kept (7 agent proposals silently destroyed!)
  Conflux MVRegister           : 8/8 candidates preserved with causal dots -> resolved cleanly to 0 conflicts

[3] Durable Disk Contention (SQLite Shared File vs Conflux Journal)
  SQLite Shared State Table    : 129/200 committed, 71 'database is locked' OperationalErrors
  Conflux Durable Server       : 200/200 committed, 0 errors
==============================================================================
```

---

## 1. Full Technical Blog Post (Publish on Dev.to, Medium, or Personal Blog *Before* HN)

**Title:**
`Why Last-Write-Wins Breaks Multi-Agent Swarms (and How Pure-Python CRDTs Fix It)`

**Subtitle:**
`When 8 parallel AI agents update shared JSON state, 87.5% of updates vanish silently. Here is how join-semilattice CRDTs and Multi-Value Registers solve multi-agent state without Redis or Postgres.`

**Full Markdown Article Body:**

```markdown
Everyone is running parallel AI agents now: three Claude Code terminals across git worktrees, a Cursor frontend agent alongside a Codex backend agent, or an 8-worker LangGraph / CrewAI / AutoGen swarm.

And almost everyone hits the same invisible bug in production: **silent state loss.**

No exception is thrown. Every LLM call returns `200 OK`. Yet at the end of the run, 7 out of 8 architectural findings are gone, shared token counters are wrong by 80%, or local SQLite state explodes with `sqlite3.OperationalError: database is locked`.

### Reproducing the Bug in 20 Lines of Python

Suppose 8 parallel agents share a JSON state blob in Redis, Postgres JSONB, or a shared file (`{"tokens_used": 0, "files_audited": []}`). Each agent reads the state, appends the file it just audited, and writes the state back:

```python
state = kv.get("swarm_state")
state["tokens_used"] += 1
state["files_audited"].append(f"agent{idx}_file{step}.py")
kv.set("swarm_state", state)  # <-- Silently overwrites 7 peer agents!
```

When you run 8 agents × 25 updates (`examples/silent_overwrite_benchmark.py`), **175 out of 200 updates (87.5%) are silently destroyed**. Worse, if two offline or partitioned agents propose competing API schemas (`{"auth": "oauth2"}` vs `{"auth": "jwt-mTLS"}`), Last-Write-Wins (LWW) picks whichever network packet arrived a millisecond later and **erases the other agent's reasoning forever**.

### Why Vector Memory (Mem0 / Zep) and Row Locks Aren't the Answer

- **Vector / Graph Memory layers (Mem0, Zep, Letta)** solve *long-term semantic recall* ("what did the user tell me last week?"). They require embeddings + external databases and aren't built for deterministic, millisecond concurrent state coordination.
- **Pessimistic locks / SQLite transactions** serialize parallel agents or fail with `database is locked` under bursty tool calls—and they don't work at all when agents are partitioned across containers, laptops, or git worktrees.

### Modeling Agent State as a Join-Semilattice (CRDTs)

Distributed systems solved this decades ago—think **Amazon Dynamo's shopping cart** and **Riak**—using **Conflict-free Replicated Data Types (CRDTs)**.

I built **[Conflux (`conflux-state`)](https://github.com/lolllll212/Conflux)** (`pip install conflux-state`), a **zero-dependency pure-Python CRDT state protocol + MCP server** purpose-built for concurrent AI agents.

Instead of mutating JSON blobs in place, agents emit immutable, optionally signed `Action`s over four composable CRDTs inside an `ORMap`:

1. **`PNCounter`** — Lock-free increments/decrements (token budgets, progress counters). `200/200` concurrent increments preserved with zero locks.
2. **`ORSet` (Observed-Remove Set)** — Add-wins shared sets (discovered URLs, audited files, active workers) without lost appends.
3. **`LWWRegister`** — Deterministic Last-Writer-Wins registers ordered by Lamport clock + agent ID.
4. **`MVRegister` (Multi-Value Register)** — **Causal conflict preservation.** Sequential updates cleanly replace prior values, while *true concurrent writes* from un-synced agents are **all preserved side-by-side** with causal `(tick, agent)` dots (`conflux_conflicts`) until a lead agent or human synthesizes them via `mv_resolve()`!

Because every CRDT forms a mathematical join-semilattice (`a ⨆ b = b ⨆ a`, associative, and idempotent), **any set of nodes that sees the same actions—in any order, across retries or network partitions—is guaranteed to converge to the exact same 64-hex SHA-256 `state_hash`.**

### Zero-Install Multi-Agent Coordination Over MCP (Claude Code + Cursor)

You don't need Docker, Redis, or Postgres. Point Claude Code, Cursor, Windsurf, or Cline at the same local or networked Conflux mesh via `uvx`:

```json
{
  "mcpServers": {
    "conflux": {
      "command": "uvx",
      "args": [
        "--from", "conflux-state",
        "conflux-mcp",
        "--agent-id", "claude-backend",
        "--listen", "127.0.0.1:7001",
        "--data", "./.conflux-state"
      ]
    }
  }
}
```

A second IDE (e.g., Cursor) connects with `"--connect", "127.0.0.1:7001", "--agent-id", "cursor-frontend"`. Out of the box, both agents get 14 MCP tools + resources + prompts:
- **Advisory TTL File/Resource Leases (`conflux_claim_resource`)** backed by `MVRegister` so two agents never edit `server.py` at the same time.
- **Conflict-Safe Shared Specs (`conflux_mv_write` -> `conflux_conflicts` -> `conflux_resolve_conflict`)**.
- **Reactive Barriers (`conflux_watch_wait`)** so the frontend agent wakes up within 40ms of the backend agent publishing an OpenAPI schema.

### Drop-In Python Adapters for LangGraph, CrewAI, and AutoGen

Already using an orchestration framework? Swap one import:

- **LangGraph:** `ConfluxCheckpointSaver` (checkpoint on Node A, resume on Node B over TCP/WebSocket gossip!) + `ConfluxStore` (with `.counter_inc()`, `.set_add()`, `.mv_put()`, `.mv_resolve()`).
- **CrewAI:** `ConfluxCrewStorage` (`StorageBackend` protocol) + `ConfluxFlowPersistence` + `create_crewai_tools()`.
- **AutoGen:** `ConfluxAutoGenMemory` + `create_autogen_tools()`.
- **Untrusted Swarms:** Sign every agent action with stdlib `HMAC-SHA256` or `Ed25519` and enforce per-agent key namespaces and write quotas *before* merge.

### Try It / Break It

- **GitHub:** https://github.com/lolllll212/Conflux
- **PyPI:** `pip install conflux-state`
- **Run the 1-second benchmark yourself:** `python examples/silent_overwrite_benchmark.py`
```

---

## 2. Hacker News (`Show HN`) — Exact Timing, Title & Comment Rules

### Timing & Mechanics That Maximize Front-Page Odds
1. **Best Window:** **Tuesday, Wednesday, or Thursday between 12:30–14:30 UTC (8:30–10:30 AM US Eastern)**.
2. **Submit URL:** Submit `https://github.com/lolllll212/Conflux` directly (HN engineers prefer landing straight on clean code + README + `docs/PROTOCOL.md` over marketing pages).
3. **Never ask friends to upvote from direct links** (HN's voting ring detector penalizes direct-URL upvote spikes). Share for *discussion* only after it is organically on `/newest` or `/shownew`.
4. **Reply to every technical comment within 10 minutes for the first 3 hours**—especially questions about CRDT tombstones, clock skew, `MVRegister` causal dots vs vector clocks, and why pure Python stdlib was chosen.

### Show HN Title (Under 80 chars, crisp engineering hook)
`Show HN: Conflux – Pure-Python CRDT shared state and MCP server for AI agents`

### Show HN Opening Comment (Paste immediately after submitting)
> Hi HN — I built Conflux (`pip install conflux-state`), a zero-dependency Python CRDT state engine and MCP server for coordinating concurrent AI agents without Redis, Postgres, or row locks.
>
> **Why I built it:** When running parallel agents (e.g. Claude Code + Cursor across worktrees, or 8 parallel LangGraph/CrewAI workers), standard read-modify-write state stores fail silently. In `examples/silent_overwrite_benchmark.py` (8 agents × 25 ops), naive JSON KV loses 87.5% of counter/list updates, Last-Write-Wins destroys 7 of 8 concurrent spec proposals, and shared SQLite throws `database is locked`.
>
> **How Conflux solves it:**
> - **Join-Semilattice CRDTs + Causal Conflict Preservation:** State is an `ORMap` over `GCounter`, `PNCounter`, `ORSet`, `LWWRegister`, and **`MVRegister` (Multi-Value Register)** (inspired by Dynamo). Sequential `mv_set` writes overwrite cleanly, while *concurrent* writes from un-synced replicas preserve *all* competing values with `(tick, agent)` causal dots until explicitly merged via `mv_resolve`. Every node absorbing the same actions converges to the identical 64-hex SHA-256 `state_hash`.
> - **Per-Agent Cryptographic Sandboxing:** Actions can be signed with HMAC-SHA256 (stdlib) or Ed25519, enforcing per-agent op allowlists, key-prefix namespaces, and write quotas prior to merge—so a rogue/hallucinating agent can't clobber keys outside its scope.
> - **Pure Python Stdlib Transport & Durability:** Framed TCP + RFC6455 WebSocket + TLS + fsynced HMAC-signed append-only journals + Prometheus `/metrics` with **zero core runtime dependencies** (~168k actions/s local replay).
> - **MCP + Framework Adapters:** Includes `conflux-mcp` (`uvx --from conflux-state conflux-mcp`) with conflict-aware file leases (`conflux_claim_resource`) and reactive watches, plus drop-in adapters for LangGraph (`BaseCheckpointSaver` + `BaseStore`), CrewAI, and AutoGen.
>
> Known trade-offs we're tackling next for `v1.0`: causal-watermark truncation for the in-memory peer catch-up journal on million-op daemons, and `ORSet` tombstone GC. Protocol spec (`docs/PROTOCOL.md`) and threat model (`docs/SECURITY.md`) are in the repo—would love critical feedback from distributed systems & agent builders!

### Pre-Written Answers to the 3 Toughest HN Questions You Will Get
- **Q1: *"Why not just use Redis (`INCR`/`SADD`) or Postgres?"***
  > *"If all your agents live inside one cloud VPC and always have connectivity, Redis `INCR`/`SADD` works great for counters/sets. Where Redis/Postgres struggle is (1) local multi-IDE workflows (`Claude Code` + `Cursor` via `uvx` where nobody wants to spin up Docker containers per repo), (2) offline/partitioned agents that reconnect and gossip later, and (3) `MVRegister` conflict preservation—Redis `SET` overwrites the prior JSON value, whereas `MVRegister` automatically detects concurrent causal branches and keeps both proposals alive until resolved."*
- **Q2: *"Doesn't `MVRegister` with `(tick, agent)` dots instead of full version vectors miss some transitive causality?"***
  > *"Great catch—each `mv_set` carries an explicit `supersedes: list[dot]` listing the exact candidate dots observed on that replica when the write was authored (effectively an explicit causal context over active dots on that key, similar to Dotted Version Vectors scoped per register). Any dot in `supersedes` is permanently tombstoned on merge so delayed/replayed packets can't resurrect resolved conflicts, while any dot *not* in `supersedes` survives as a concurrent sibling."*
- **Q3: *"What about CRDT metadata / tombstone bloat in long-running sessions?"***
  > *"On disk, `JournalStore` already compacts journals into `snapshot.json`. In memory, `ORSet` tombstones and `MVRegister.superseded` sets grow with distinct removals/supersessions—totally fine for tens of thousands of agent coordination actions per session, and for `v1.0` we're adding causal-watermark GC once all registered cluster peers ack a snapshot epoch."*

---

## 3. Subreddit-Specific Posts (Tailored to Each Community's Pain)

> **Golden Reddit Rule:** Space posts 24 hours apart, lead with the problem + benchmark numbers (not marketing language), and put the GitHub link at the bottom.

### A. `r/ClaudeCode` & `r/cursor` (Highest Immediate Conversion)
**Title:**
`I built a zero-dependency CRDT MCP server so parallel Claude Code & Cursor agents stop overwriting each other's files and specs`

**Body:**
```markdown
If you run 2–4 Claude Code terminals (or Claude Code + Cursor) in parallel on the same codebase, you've probably hit two annoying walls:
1. Two agents edit the same file at the same time and create a mess.
2. You use a shared `state.json` / SQLite MCP server, and one agent's update silently overwrites another's (or SQLite throws `database is locked`).

I open-sourced **Conflux (`conflux-state`)** — a pure-Python CRDT shared state engine + MCP server (`conflux-mcp`) with zero external dependencies (no Docker, no Redis, no API keys).

**One-liner to add to Claude Code / Cursor MCP config:**
- Terminal 1 (hosts embedded mesh + durable journal):
  `uvx --from conflux-state conflux-mcp --agent-id claude-api --listen 127.0.0.1:7001 --data ./.conflux`
- Terminal 2 / Cursor (joins mesh):
  `uvx --from conflux-state conflux-mcp --agent-id cursor-ui --connect 127.0.0.1:7001`

**What your agents get:**
- **`conflux_claim_resource` / `conflux_release_resource`**: TTL file/module leases backed by Multi-Value Registers (`MVRegister`). If Cursor tries to touch `conflux/server.py` while Claude holds it, it immediately sees `"held_by": "claude-api", "intent": "refactoring gossip loop"` and works on another task.
- **`conflux_mv_write` + `conflux_conflicts` + `conflux_resolve_conflict`**: Like Git for agent working memory—concurrent spec writes are *both* preserved instead of Last-Write-Wins clobbering.
- **`conflux_watch_wait`**: Agents can block up to 30s waiting for a peer agent to publish a schema/task output instead of burning tokens polling.

Repo (MIT, zero deps): https://github.com/lolllll212/Conflux
PyPI: `pip install conflux-state`
```

### B. `r/LangChain` & `r/AI_Agents`
**Title:**
`Benchmark: 8 parallel agents lose 87.5% of shared JSON updates — so I built a zero-infra CRDT Checkpointer & Store for LangGraph / CrewAI / AutoGen`

**Body:**
```markdown
When moving multi-agent graphs out of a single notebook, `MemorySaver` breaks across processes, shared SQLite hits `database is locked` under parallel fan-out, and spinning up Postgres + Redis just for local/edge agent swarms is heavy.

Even worse: if parallel workers read-modify-write shared state keys without distributed locks, **87.5% of updates vanish silently** (reproducible benchmark in `examples/silent_overwrite_benchmark.py`).

I built **Conflux (`pip install conflux-state`)** — a pure-Python CRDT shared state fabric with drop-in adapters for:
- **LangGraph:** `ConfluxCheckpointSaver` (checkpoint a graph on Process A, resume seamlessly on Process B over TCP/WS gossip!) + `ConfluxStore` (adds lock-free `.counter_inc()`, `.set_add()`, and conflict-preserving `.mv_put()` / `.mv_resolve()`).
- **CrewAI:** `ConfluxCrewStorage` (`StorageBackend`) + `ConfluxFlowPersistence` + 7 CRDT Crew tools.
- **AutoGen:** `ConfluxAutoGenMemory` + 8 `FunctionTool`s.
- **MCP (`conflux-mcp`):** Connect Claude Code / Cursor directly to the same state mesh your Python agents write to.

Zero core dependencies (pure Python stdlib), optional HMAC/Ed25519 per-agent signing + namespace sandboxing, and 125 conformance tests.

GitHub + runnable examples: https://github.com/lolllll212/Conflux
```

---

## 4. 45-Second Terminal Demo Script (Record with `vhs` or `asciinema` for X/README)

Create a file `demo.tape` locally if you have Charm's [`vhs`](https://github.com/charmbracelet/vhs) (`brew install vhs`) to generate a crisp `demo.gif` for your README and X launch thread:

```text
Output conflux-demo.gif
Set FontSize 16
Set Width 1100
Set Height 620
Set Theme "Catppuccin Mocha"

Type "# 8 parallel AI agents updating shared state: Naive JSON LWW vs Conflux CRDTs"
Enter
Sleep 800ms
Type "python examples/silent_overwrite_benchmark.py"
Enter
Sleep 4s
Type "# Zero-install MCP server for Claude Code + Cursor:"
Enter
Type "conflux-mcp --help"
Enter
Sleep 3s
```

---

## 5. 7-Day Launch Sequence Checklist

- [ ] **Day 0 (Prep, 5 mins):** Merge PR #6 (`v0.9.1`), publish `0.9.1` to PyPI via GitHub Actions, and add GitHub repository topics (`mcp`, `mcp-server`, `crdt`, `multi-agent`, `langgraph`, `crewai`, `autogen`, `claude-code`).
- [ ] **Day 1 (Directories):** Run `mcp-publisher publish` (uses root `server.json` -> Official MCP Registry) and open 1-line PR to `punkpeye/awesome-mcp-servers` + submit to `smithery.ai` and `mcp.so`.
- [ ] **Day 2 (Blog + X):** Publish the Section 1 blog post on Dev.to / personal blog + post the benchmark output screenshot + `conflux-demo.gif` on X and LinkedIn.
- [ ] **Day 3 (Show HN):** Submit `Show HN` at **13:00 UTC (Tue/Wed/Thu)** with the Section 2 comment and stay active in the thread for 3 hours.
- [ ] **Day 4 (`r/ClaudeCode` + `r/cursor`):** Post the Section 3A MCP workflow guide.
- [ ] **Day 5 (`r/LangChain` + `r/AI_Agents`):** Post the Section 3B benchmark + checkpointer/store guide.
- [ ] **Day 7 (Ecosystem PRs):** Open docs/community showcase PRs to `kyrolabs/awesome-agents`, `e2b-dev/awesome-ai-agents`, and `crdt.tech`.
