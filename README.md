# Conflux

A deterministic state protocol for autonomous multi-agent systems. State is not a
set of rows — it is the join of an evolving stream of signed, replayable actions.
Agents write concurrently to the same shared state with no table locks, no
two-phase commit, and no central coordinator; every node that has seen the same
actions converges to the identical state by construction.

## Core use case: shared agent state

The headline problem conflux solves is **shared agent state**: a few dozen
agents of one application reading and mutating the same counters, registers,
and sets — fleet progress, job counters, leader/slot election, capability
flags, hand-off pointers — with signed writes, durable replay, and crash
recovery, on a mesh of commodity TCP/WebSocket nodes. Two adjacent use cases
ride on the same fabric: **workflow state** (task queues and step status
through `TaskStore` and deterministic `Workflow`) and a **replicated memory /
knowledge layer** (an agent's shared facts as register/set/string slots).
All three are one storage geometry; the product pitches the agent-state head.

This is not a blockchain and not a distributed SQL system. It is a replicated
state fabric built on mathematically convergent data structures (CRDTs / lattice
theory) — a coordination layer for distributed AI workflows.

## Product definition and scope

Conflux is a **small-crew deterministic state fabric** for a few dozen agents
of a single application: signed, replayable, eventually-consistent shared
agent state with durable journaling and crash recovery. The protocol is
specified in [`docs/PROTOCOL.md`](docs/PROTOCOL.md); every claim in it maps to
a test (conformance index at the end of the spec). Conflux is released under
the [MIT License](LICENSE).

Scope — what it is:

- deterministic convergence: same action set ⇒ same `state_hash`, in any order;
- signed identity: HMAC (zero deps) or Ed25519, plus op/namespace/quota rules;
- durability: fsynced journal + cumulative snapshots that survive a hard crash;
- TCP and WebSocket transports, TLS on both, gossip between peer nodes.

Explicitly out of scope (backlog, not promised):

- linearizable reads, multi-region federation,
  general-scale KV, blockchain-style global ordering.

## Problem

Classic distributed databases serialize writes: a crew of agents updating the
same operation counters, queue pointers, capability flags, or hand-off state
either blocks itself (latency) or deadlocks. Locks scale poorly once agents
outnumber transactions — and agent crews are exactly that regime: many writers,
one small shared state.

## Approach

Conflux treats shared state not as static rows but as the join of an evolving stream
of probabilistic actions. Every value living in the store is an element of a
join-semilattice, so any two replicas can merge their states with a single
commutative, associative, idempotent operation — the lattice join. Because a join is
order-independent, every replica that has seen the same set of actions converges to
the identical state, no matter the delivery order.

```
state  =  join(contribution(action_1), ..., contribution(action_n))
```

## Core concepts

### The lattice

`Lattice` is the abstract contract: every CRDT implements `merge(other)` (the join),
`read()`, and `canonical()` (a deterministic fingerprint used for hashing). The join
laws are verified directly in `tests/test_crdts.py`:

- idempotent: `a ⨆ a = a`
- commutative: `a ⨆ b = b ⨆ a`
- associative: `(a ⨆ b) ⨆ c = a ⨆ (b ⨆ c)`

### CRDT types

| Type            | Field         | Meaning                                             |
| --------------- | ------------- | --------------------------------------------------- |
| `GCounter`      | count per key | grow-only counter, merge = elementwise max          |
| `PNCounter`     | (p, n) slots  | counters with increments and decrements             |
| `LWWRegister`   | value, stamp  | last-writer-wins register, stamp = (tick, agent)    |
| `GSet`          | set           | grows only, merge = union                           |
| `ORSet`         | tags, values  | add/remove with tombstones, merge = union           |
| `ORMap`         | key -> CRDT    | grows-only keys over any of the above               |

Register values are ordinary JSON-ish values; equal stamps (or equal set tags, an
invalid protocol usage) are resolved deterministically by canonical order so every
replica still picks the same winner.

### Actions and determinism

Agents do not mutate shared rows. They emit immutable `Action`s, each with a globally
unique `action_id`, an `agent`, a logical `tick`, an operation, and params:

```python
Action(action_id="act-agent-1-42", agent="agent-1", tick=42,
       op="counter_inc", key="orders", params={"by": 5})
```

Probabilistic actions carry a `weight` param. Whether an action "fires" is decided by
`SHA-256(action_id, key)` projected onto a fixed-point 32-bit scale — a pure function
every replica can compute, so probabilistic inference produces identical state
everywhere without any coordination.

### Convergence protocol

- Each agent owns a `Replica` — its local join of the actions it has seen. Its
  journal is append-only and de-duplicated by `action_id`.
- `reconcile(a, b)`: the two replicas exchange journals and absorb what they have not
  seen. Absorption is exactly the lattice join.
- `converge_all(replicas)`: full-mesh anti-entropy sweep (for tests and small crews).
- `converge_by_gossip(replicas, rounds)`: randomized pairwise anti-entropy (for
  large, partial meshes).
- `Cluster` / `DistributedApp` / `InMemoryTransport`: a thin network abstraction to
  demonstrate the protocol over message passing.

Any two replicas that have both absorbed the union of available actions are
provably identical: identical action sets + identical join = identical state. A
consensus fingerprint, `state_hash(state)`, is a SHA-256 over the canonical
serialization.

## Quick start

```python
from conflux import Cluster, InMemoryTransport

cluster = Cluster({"agent-a", "agent-b", "agent-c"}, transport=InMemoryTransport())
cluster.node("agent-a").counter_inc("orders", 5)
cluster.node("agent-b").counter_inc("orders", 3)
cluster.node("agent-c").register_set_weighted("risk.appetite", "low", weight=0.6)
cluster.gossip_round()

assert cluster.read("agent-a", "orders") == 8
assert cluster.read("agent-b", "orders") == 8
```

Deterministic replay from a journal:

```python
from conflux import state_hash, replay
from conflux.protocol import journal_from_json, journal_to_json

blob = journal_to_json(cluster.node("agent-a"))
actions = journal_from_json(blob)
# fold in any order, every time, always the same state:
assert state_hash(replay(actions).state) == state_hash(cluster.node("agent-b").state)
```

## Protocol hardening: versioning, signatures, authority

Actions travel as envelopes that any node can verify before they touch state.

```json
{
  "action": { "action_id": "act-a-1", "agent": "a", "tick": 1, "op": "counter_inc",
              "key": "orders", "params": {"by": 3}, "version": 1 },
  "signed_by": "a",
  "signature": "<signature over canonical action bytes>",
  "alg": "hmac-sha256"
}
```

- **Versioning.** Every action carries `version`; the validator rejects unknown
  versions, so protocol changes are explicit (current: `1`).
- **Signatures.** `sign_action(action, secret)` produces an HMAC-SHA256 over the
  canonical (key-sorted) JSON of the action (zero dependencies). For asymmetric
  identity, pass `alg="ed25519"` with a raw private key — `generate_ed25519_keypair()`
  returns `(private, public)`, `verify_action(envelope, public, "ed25519")` checks it.
  Ed25519 requires the optional `cryptography` package; HMAC needs nothing.
- **Identity + authority.** `AgentRegistry` maps agent ids to a secret (HMAC) *or* a
  public key (Ed25519), plus allowed op sets and per-agent write quotas.
  `validate_envelope` checks signer identity, signature, authorization, and quota
  before a node ever merges, and refuses envelopes whose algorithm does not match
  the agent's registered key.
- **Schema.** `validate_action` enforces op-specific payload rules (counter `by`,
  JSON-safe register/set values, non-empty tags, `weight` in `[0, 1]`).

A server equipped with a registry refuses unverified, unauthorized, or
quota-exhausted submits (`{"type": "error", "reason": "..."}`); a server without
one still schema-validates but runs in trust-less dev mode.

```python
from conflux import AgentRegistry, Client, Server

registry = AgentRegistry()
registry.register("driver", "s3cr3t", ops={"counter_inc", "lww_set"}, quota=1000)
server = Server("n1", ("127.0.0.1", 7001), registry=registry)
server.start()

client = Client("127.0.0.1", 7001, agent_id="driver", secret="s3cr3t")
client.counter_inc("orders", 5)   # signed, authorized, quota-counted
```

## Running the tests

```bash
python -m pytest
```

The suite covers join laws, threaded lock-free convergence, merge-order
determinism, probabilistic determinism, journal round-tripping, the
network-framework gossip path, the networked backend (client round-trip,
two-node gossip convergence, restart persistence), signed-envelope authority,
namespaces, the WebSocket transport (RFC6455 framing, WS node convergence,
discovery, batching, backoff), the application layer (events, watchers,
task state, deterministic workflows), and the operational surface (TLS framing
over tcp + ws, signed/verified journals, membership + partition heal, and
cumulative snapshot compaction).

## Application layer: workflows, tasks, events, watchers

Everything sits on the same replicated fabric, so application state converges
and replays exactly like raw CRDT state.

- **`EventBus`** — subscribe to the action stream by op, exact key, or key
  prefix; handlers get `(action, origin)` where origin is `local` (this node's
  own emit) or `absorb` (received from a peer). `Server.subscribe(...)` wires it
  onto the live wire path.
- **`Watcher`** — value-level snapshots: `watch(key, handler)` fires when a
  key's effective value changes, with the previous value.
- **`TaskStore`** — replicated task execution state under `tasks/<id>`
  (created → running → step → done/failed), each transition a signed,
  deterministic `lww_set`.
- **`Workflow` / `WorkflowRunner`** — ordered steps (`counter_inc`, `lww_set`,
  `orset_add`, `orset_remove`, `noop`) against keys, with optional
  `weight`-gated steps decided by the same deterministic hash every replica can
  reproduce. Identical inputs always emit identical action logs.

```python
from conflux import Server, Workflow, WorkflowRunner, WorkflowStep, TaskStore

server = Server("n1", ("127.0.0.1", 7001))
server.start()
server.subscribe(lambda action, origin: print("event:", origin, action.op),
                 op="counter_inc")
server.watch("wf/shipments", lambda value, key, old: print("watched:", value))

workflow = Workflow("fulfill", [
    WorkflowStep("reserve", "counter_inc", key="wf/capacity", params={"by": 1}),
    WorkflowStep("ship", "orset_add", key="wf/shipments", params={"value": "carrier-17"}),
])
task = WorkflowRunner(server.replica, workflow).run("fulfill-1", "fulfill",
                                                    inputs={"sku": "A-1"})
```

## Operations: metrics, admin stats, benchmarks

Every server exposes a live `metrics()` dict — absorbed/submitted actions,
journal length, links, outbound dials, discovered peers, known members, uptime,
current `state_hash` — and answers the wire query `{"type": "admin", "what": "stats"}`
with the same payload, so monitoring can poll nodes without a driver agent.

For Prometheus, the same data is rendered in text exposition format — no
dependencies — and served three ways: `Server.prometheus()`, the wire query
`{"type": "admin", "what": "prometheus"}` (returns the body over the framed
connection), and an embedded stdlib HTTP listener:

```bash
python -m conflux.server --id alpha --listen 127.0.0.1:7001 --metrics-http
# -> metrics http on http://127.0.0.1:<ephemeral>/metrics
```

```python
server = Server("alpha", ("127.0.0.1", 7001)).start()
server.start_metrics_http("0.0.0.0", 9101)   # scrape http://<node>:9101/metrics
```

Counters (`conflux_actions_absorbed_total`, `..._submitted_total`) and gauges
(links, outbound, peers, discovered, members, journal_actions, uptime,
state-hash info label) are labeled with `node_id`:

```bash
python benchmarks/bench.py            # replay, cluster, tcp + ws round-trips
python benchmarks/bench.py --quick    # small sizes for CI
```

Example (previous run, this machine): ~168k actions/s journal replay,
~38k actions/s full-mesh cluster convergence, ~0.13 ms single-agent
submit→ack round-trip over TCP and ~0.17 ms over WebSocket.

The real agent-state app, in one file: `examples/agent_crew_demo.py` — six
signed agents share state (`crew.work_done`, check-in set, leader slot) across
a 3-node mesh, then put the runtime through the failure model it was hardened
for: a hard crash, continued writes through the partition, restart from the
signed snapshot + journal tail, keyless reload refused, and an out-of-namespace
write refused with no state mutation. `examples/durable_crew_demo.py` shows the
same crash-recovery cycle on a single agent. Supporting demos:
`examples/workflow_demo.py` (tasks/events/watchers over a cluster) and
`examples/deploy_multi_node.py` (signed Ed25519 crews over TCP and WS meshes).

## Backend: networked, persistent nodes

The reference implementation ships with a multi-node backend so agents in
different processes (or different machines) can converge. Two transports are
built in: the default framed TCP and `--transport ws`, a dependency-free
RFC6455 WebSocket transport (`conflux.websocket`). Set `transport="ws"` on the
`Server`/`Client` to use it.

### Run a node

```bash
python -m conflux.server --id alpha --listen 127.0.0.1:7001 --data ./data/alpha
python -m conflux.server --id beta --listen 127.0.0.1:7002 --peers 127.0.0.1:7001 --data ./data/beta
```

Use `--peers host:port,host:port` to list neighbors. Gossip links are
bidirectional once established, and each node replays its persisted journal on
startup.

### TLS framing (tls:// and wss://)

Pass a PEM certificate to encrypt every socket with the stdlib `ssl` module —
both the framed TCP transport and the RFC6455 WebSocket transport, for agent
clients and peer-to-peer gossip alike:

```bash
python -m conflux.server --id alpha --listen 127.0.0.1:7001 \
    --cert ./cert.pem --key ./key.pem --data ./data/alpha
```

```python
server = Server("alpha", ("127.0.0.1", 7001), tls=("cert.pem", "key.pem")).start()
client = Client("127.0.0.1", 7001, tls=("cert.pem", "key.pem", "cert.pem"))  # ca = 3rd element
```

Omitting the CA in the client's `tls=` tuple disables peer verification (self-signed
clusters); `tls=True` uses the system CAs, and `tls="ca.pem"` verifies against that CA.

### Membership and partitions

Peers announce their node id, address, peer list, and member view on
handshake; every server maintains a `members` view (see `metrics()["members"]`
and `members_view()`). When a neighbor dies, the receive loop drops the link
and the gossip loop re-dials it with exponential backoff. Links that go silent
without closing (a hung or partitioned peer) are liveness-checked and dropped
after `--stale-after` seconds (default 3× `gossip_interval`), then re-dialed —
so a healed partition re-converges automatically once the peer is reachable
again (see `tests/test_phase4.py::test_partition_reconnect`).

### Signed, compacted journals

```bash
python -m conflux.server --id alpha --listen 127.0.0.1:7001 \
    --cert ./cert.pem --key ./key.pem --journal-key "$(openssl rand -hex 16)" --data ./data/alpha
```

Every journal line carries `"sig"` — an HMAC-SHA256 over the canonical action
bytes — and every snapshot carries the same over its `{seq, root}` payload.
Replay verifies both; a tampered log or snapshot raises instead of silently
restoring a corrupt state. Snapshots are **cumulative**: each compaction folds
the current root into the previous generation via the lattice join and emits a
single merged summary, pruning older generations, so a crash mid-history never
loses committed state.

### Write and read from an agent

```python
from conflux import Client

client = Client("127.0.0.1", 7001, agent_id="agent-42")
client.counter_inc("bookings.seats", 2)
client.register_set("flight.eta", "14:05")
client.set_add("tickets", "TK-100", tag="tk-100")
print(client.read("bookings.seats"))          # 2
print(client.hash())                          # 64-hex SHA-256 of canonical state

# Batch multiple mutations in a single atomic round-trip:
with client.batch() as b:
    b.counter_inc("bookings.seats", 1)
    b.register_set("flight.status", "boarding")

client.close()
```

### Durability

Each absorbed action is appended to `data/<node>/journal.jsonl` immediately.
When the journal passes the snapshot threshold, the current CRDT root is merged
(lattice join) into the previous generation's root and written atomically —
both as `snapshot-<seq>.json` (generation) and `snapshot.json` (latest alias) —
after which the log is truncated and older generations are pruned. A node never
replays a tail older than what its cumulative snapshot already captures, and a
crash mid-rotation degrades to the newest untampered generation. Optional
`--journal-key` HMAC-signs every line and snapshot, and replay refuses
tampered files. State values travel on the wire as JSON; register/set values
must be JSON-serializable.

## Guarantees and limits

Conflux promises, and the test suite verifies:

- **Eventually consistent convergence**: every node that has absorbed the union of
  available actions shows identical state (same `state_hash`), regardless of
  delivery order.
- **No global lock, no blocking writes**: agents write to their own replicas
  concurrently; merging is a join, so writers never contend.
- **Deterministic merges**: identical action sets always yield identical state,
  including tie-breaks (equal stamps/tags) resolved by canonical order.
- **Replayability**: any subset of the journal rebuilds exactly the same state.
- **Durable ack**: a submit is acknowledged only after the accepting node has
  absorbed the actions and fsynced them to its journal; an acked action
  survives that node's process crash (torn-tail recovery).

The exact limits — schema bounds, frame size, snapshot/gossip/stale defaults,
backoff schedule, quota rules, and measured throughput (≈24k actions/s local
replay, ≈2.8k actions/s across an 8-agent TCP cluster) — are pinned in
[`docs/PROTOCOL.md` §8](docs/PROTOCOL.md); re-run `benchmarks/bench.py` for
host-local figures.

Honest limits:

- **Stale reads**: a read observes the join of everything that node has gossiped
  so far. Convergence is eventual, not linearizable; workflows that need
  strictly ordered reads need an external sequencing channel.
- **One type per key**: merging a key's existing CRDT with a different CRDT type
  raises `TypeError` — a deterministic schema contract, not an ambiguity.
- **Unique IDs are load-bearing**: `action_id` and set tags are event identities;
  reuse silently merges (so agent ids must be process-unique and ticks monotonic).
- **Fixed-point weights**: probabilistic decisions use a fixed 32-bit scale over
  the same serialized `weight`, so all replicas agree as long as `weight` is a
  JSON number.

## Roadmap -> status

Phase 1 (protocol foundations), Phase 2 (cluster runtime), Phase 3 (application
layer), and the Phase 4 operational surface (metrics/admin probe, Prometheus
exporter, benchmarks, TLS, signed journals, membership/partition handling,
cumulative compaction) are complete.

The remaining ideas in each phase are **backlog, deliberately out of scope**
under the current product definition (§"Product definition and scope"): secure
framing beyond TLS, multi-cluster federation, chunked cross-peer snapshot
exchange, OpenTelemetry exporters, and workflow orchestration expansion. Each
is a candidate to revisit only after the core fabric — determinism, durability,
crash recovery — is proven by adopters.

## License

Conflux is released under the [MIT License](LICENSE), as declared in
`pyproject.toml`.