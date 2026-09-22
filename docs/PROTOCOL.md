# Conflux Protocol Specification

Version: 1 (action envelope) · Package version: 0.7.3

This is the authoritative description of the conflux wire format, state model,
determinism rules, persistence format, and hardening guarantees. Every
numbered claim has a conformance test; the index is at the end.

## 1. Product scope

Conflux is a **small-crew deterministic state fabric**: signed, replayable,
eventually-consistent shared agent state for a few dozen agents of a single
application. Workflow state (`TaskStore` task queues, `Workflow` step views)
and a replicated memory/knowledge layer (per-agent fact slots) are supported
applications of the same storage geometry. It is deliberately **not**:

- a linearizable KV store (reads observe the join of gossiped state; §7);
- a multi-region/geo-distributed system (lead times are for one process mesh layer);
- a general relational store (one CRDT per key, §2);
- a blockchain (no global ordering, no total order, no chain).

## 2. State model

Shared state is a lattice: a map of keys, one convergent data structure per key.

| Type        | Key/slot      | Op(s)              | Semantics                              |
| ----------- | ------------- | ------------------ | -------------------------------------- |
| `PNCounter` | per count key | `counter_inc`, `counter_dec` | signed arithmetic, `Δ{by}` |
| `LWWRegister`| per key      | `lww_set`          | `(tick, agent)` stamp, higher wins, ties by canonical bytes |
| `ORSet`     | per key       | `orset_add`, `orset_remove` | add/remove by unique `tag`, tombstones |

Merge is the lattice join: commutative, associative, idempotent. Any two
replicas that have absorbed the same action set are identical.

**Claim C1 (join laws).** `merge` is commutative, associative, idempotent.
— `tests/test_crdts.py`

**Claim C2 (one type per key).** Merging a key across CRDT types raises
`TypeError`. — `tests/test_crdts.py`

## 3. Determinism rules

1. **Canonical serialization.** Any byte produced from an action or envelope is
   `json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")`.
2. **Hash decisions.** Probabilistic `weight ∈ [0,1]` fires iff
   `H(weight ∥ action_id ∥ key)` projected onto the fixed 32-bit scale lands in
   range. It is a pure function, identical on every replica.
3. **Deterministic tie-breaks.** Equal LWW stamps or (invalidly) equal ORSet
   tags resolve by canonical bytes of the full action, identically everywhere.

**Claim C3 (state_hash).** `state_hash(state)` is SHA-256 over the canonical
state; two replicas with equal hash have identical state. — `tests/test_convergence.py`

**Claim C4 (deterministic replay).** Folding any permutation of a journal
reproduces the same state. — `tests/test_replay.py`

## 4. Actions and envelopes

### 4.1 The action

```json
{
  "action": {
    "version": 1,
    "action_id": "act-<agent>-<tick>",
    "agent": "driver",
    "tick": 5,
    "op": "counter_inc",
    "key": "orders.total",
    "params": {"by": 1}
  }
}
```

Field rules (enforced by `validate_action`, `conflux/validate.py`):

| Field      | Rule                                                             |
| ---------- | ---------------------------------------------------------------- |
| `version`  | must equal `1`                                                    |
| `action_id`| non-empty identifier, ≤512 chars, charset `[A-Za-z0-9._:\-]`      |
| `agent`    | non-empty identifier, ≤128 chars, same charset                     |
| `tick`     | positive integer                                                   |
| `key`      | non-empty, ≤256 chars, no control characters; dot/namespace by convention |
| `op`       | one of `counter_inc`, `counter_dec`, `lww_set`, `orset_add`, `orset_remove` |
| `params`   | JSON object, serialized size ≤4096 bytes, values JSON-safe         |
| `weight`   | optional; numeric, `0 ≤ w ≤ 1`                                    |

### 4.2 The envelope

```json
{
  "action": {...},
  "signed_by": "driver",
  "signature": "<hex HMAC-SHA256 or base64 Ed25519>",
  "alg": "hmac-sha256" | "ed25519"
}
```

Signatures cover the canonical action bytes (§3.1). `signed_by` must equal
`action.agent`; `alg` must match the agent's registered key kind.

**Claim H1.** Malformed or out-of-bounds fields are rejected with
`ValidationError`, never merged. — `tests/test_validate.py`, `tests/test_spec_conformance.py`

## 5. Identity and authority

An `AgentRegistry` maps `agent_id → (key kind, key, ops, quota, namespaces)`:

- **HMAC** (`alg: hmac-sha256`, default): shared-secret, zero dependencies.
- **Ed25519** (`alg: ed25519`): asymmetric; requires optional `cryptography`.
  `generate_ed25519_keypair()` → `(seed, public_key)`.
- **Authorization**: `ops` allowlist; absent = any of the standard ops.
- **Quota**: per-agent write budget; enforced at absorption.
- **Namespaces**: prefix allowlist over keys (`"orders."` permits `"orders.total"`).

A registered server refuses unverified / unauthorized / out-of-namespace /
quota-exhausted submits with `{"type":"error","reason":...}`.

**Claim H2.** Envelopes whose algorithm, signer, signature, authorization,
namespace, or quota fails are rejected before any state mutation.
— `tests/test_validate.py`, `tests/test_namespace.py`

## 6. Wire protocol

### 6.1 Framing

- **TCP**: 4-byte big-endian unsigned length prefix, then UTF-8 JSON payload
  (max frame 64 MiB).
- **WebSocket**: dependency-free RFC6455 subset — text/binary frames, masking
  (client→server), ping/pong, close handshake, fragmentation reassembly.
- **TLS**: both transports wrap sockets with the stdlib `ssl` module when
  `--cert/--key` (or `Server(tls=...)`) is configured; peers/clients verify
  against the provided CA or system CAs.

**Claim H3.** Framing round-trips and rejects overlong/malformed frames.
— `tests/test_websocket.py`, `tests/test_phase4.py::test_tls_convergence`

### 6.2 Messages (agent)

| Request                          | Response                                  |
| -------------------------------- | ----------------------------------------- |
| `{"type":"hello","role":"agent"}`| `{"type":"welcome","node_id":...,"peers":[...]}` |
| `{"type":"submit","actions":[...]}` | `{"type":"ack","accepted":n}` / `{"type":"error","reason":...}` |
| `{"type":"read","key":...}`      | `{"type":"read_resp","key":...,"value":...}` |
| `{"type":"read_all"}`            | `{"type":"read_resp","value":{...}}`     |
| `{"type":"hash"}`                | `{"type":"hash_resp","hash":"<sha256>"}` |
| `{"type":"sync"}`                | `{"type":"journal","actions":[...]}`     |

### 6.3 Messages (peer-to-peer)

- `hello` (role `peer`) carries `node_id`, listen `address`, `peers`, and
  `members`; answered by `hello_ack` with the same. Each side sends its full
  journal (`journal`) on link setup and re-sends gaps on gossip ticks
  (optionally batched). `discover`/`peers` accelerates mesh wiring;
  `auto_discover` promotes advertised addresses to dialed peers.
- **Membership**: node ids + addresses are recorded on handshake
  (`metrics()["members"]`, `members_view()`).
- **Liveness**: links idle longer than `stale_after` (default 3× gossip
  interval) are dropped and re-dialed with exponential backoff
  (`2^min(failures, 6)` tick skip). Dropped links re-converge automatically on
  reconnect because reconciliation is a re-join.
- **Backpressure, not blocking**: each link owns a bounded (256-message)
  outbound queue drained by its own sender thread, and dialing runs in
  per-address workers with a dedup guard. One stalled or maliciously silent
  peer can never freeze the gossip loop for the rest of the mesh; when a queue
  is full, further frames are dropped and re-sent next round (safe because the
  receiver dedups by `action_id` and the join is idempotent).

**Claim H4.** Two peers converge to one `state_hash`; a partitioned and healed
peer re-converges without operator action. — `tests/test_server.py`,
`tests/test_phase4.py::test_partition_reconnect`
**Claim H7.** A stalled peer doesn't block other links; reconnect re-sends do
not double-apply (`action_id` dedup). — `tests/test_prod_hardening.py`

### 6.4 Admin

`{"type":"admin","what":"stats"}` → `{"type":"stats",...}` (metrics dict);
`{"type":"admin","what":"prometheus"}` → `{"type":"prometheus","body":...}`
(Prometheus text). Metrics are also served over HTTP `/metrics`
(indicator: `--metrics-http`, `Server.start_metrics_http()`).

### 6.5 Failure semantics

Conflux fails safe. Every failure mode below leaves the survivor replicas
computable and the crashed replica able to rejoin exactly.

| Failure                        | Behavior |
| ------------------------------ | -------- |
| Node crash mid-write           | fsync'd append means no acknowledged action is lost; an unacknowledged torn tail is skipped on load (§7) |
| Silent tamper vs. corruption   | signature mismatch on any verifiable line raises on load; nothing is silently replayed |
| Invalid/unframed/invalid-JSON message | peer link closed; re-dial per liveness rules |
| Unverified / unauthorized submit | refused with `{"type":"error"}`; no state mutation (§5) |
| Peer down / partition          | other peers keep converging; stale link dropped after `stale_after`, re-dialed with backoff; reconcile-on-reconnect restores convergence without operator action |
| Clock skew between agents      | convergence does not depend on wall-clock time anywhere in the state machine; receipt mismatch can delay session messages but not merge correctness |

**Claim H5.** The failure semantics above are exercised by the conformance
index: partition + tamper in `tests/test_phase4.py`, torn tail in
`tests/test_spec_conformance.py`, refusals in `tests/test_validate.py`, and
the full crash cycle in `examples/durable_crew_demo.py`.

## 7. Persistence and crash recovery

- **Journal**: `journal.jsonl`, one envelope-entry per line:
  `{"action":{...},"seq":N}` plus `"sig"` when a journal key is configured —
  HMAC-SHA256 over the canonical action bytes. Lines are flushed **and fsynced**
  per append.
- **Cumulative snapshots**: when the post-snapshot log exceeds
  `snapshot_threshold`, the current root is lattice-joined into the latest
  generation (`snapshot-<seq>.json`) and atomically published as
  `snapshot.json`; the log is truncated and older generations pruned. Newer is
  always cumulative, so replay never requires a missing generation.
- **Recovery rules**:
  - A valid signature mismatch anywhere → raise (tamper), refuse to replay.
  - A **torn final line** (unparseable tail from a mid-write crash) → skipped;
    earlier corrupt lines → raise.
  - A log that carries signatures but is loaded **without the signing key** →
    raise (silent verification downgrade is refused; `journal_secret` required).
  - A node refuses to start from a journal that fails any of the above.
- **Snapshots are signed** under the same key when configured.

**Claim D1.** A tampered journal or snapshot raises on load.
— `tests/test_phase4.py`
**Claim D2.** A torn tail is skipped; an early corrupt line is refused.
— `tests/test_spec_conformance.py`
**Claim D3.** Restart after crash reproduces the pre-crash state and
re-converges with peers. — `examples/durable_crew_demo.py` (e2e)
**Claim D4.** Cumulative generations rebuild the full state after multiple
rotations. — `tests/test_phase4.py::test_cumulative_compaction`

## 8. Guarantees and limits

### 8.1 What is guaranteed

- **Eventual consistency**: every node that absorbed the union of available
  actions holds identical state (equal `state_hash`), regardless of delivery
  order. Convergence is eventual, not linearizable — cross-node reads can be
  stale until gossip catches up.
- **Deterministic execution**: same action set ⇒ same state, including
  weighted decisions and tie-breaks. No wall-clock time is consulted by any
  state transition (session timeouts live in the transport layer only).
- **Durable ack**: a `submit` gets `ack` only after the accepting node has
  absorbed the actions, written them to the journal, and fsynced them. An
  acknowledged action is never silently lost by that node's process crash
  (torn-tail recovery, §7). Crashes before the ack leave the write in limbo —
  resubmit is idempotent by `action_id`.
- **Read-your-writes on the node that acks**: the value a client just wrote is
  readable back from that same node; other nodes catch up by gossip.
- **Refusal on violation**: schema, signature, authorization, namespace, or
  quota violations are refused with `{"type":"error"}` and mutate nothing.
- **Unique IDs are load-bearing**: `action_id` collisions and equal set tags
  merge as the same event; agent ids must be process-unique, ticks monotonic.

### 8.2 Exact limits and defaults

| Item | Value |
| ---- | ----- |
| `action_id` length | ≤ 512 chars, charset `[A-Za-z0-9._:\-]` |
| `agent` length | ≤ 128 chars, same charset |
| `key` length | ≤ 256 chars, no control characters |
| set `tag` length | ≤ 512 chars, no control characters |
| `params` size | ≤ 4096 UTF-8 bytes, JSON-safe values |
| frame / message size | ≤ 64 MiB (TCP length prefix and WS frame) |
| journal line | `{"action":..., "seq":N}` + `"sig"` when a key is configured |
| snapshot threshold | 1000 actions (configurable) |
| gossip interval | 1.0 s (configurable) |
| stale-after (link drop) | `max(3.0, 3 × gossip_interval)` s |
| dial backoff | `2^min(failures, 6)` gossip ticks, failures reset on success |
| dial / recv timeouts | 3 s connect, 10 s socket read |
| per-link outbound queue | bounded at 256 messages; overflow dropped, re-sent next round |
| listen backlog | 32 |
| quota | `None` = unlimited; charged once per **validated submit envelope** (redelivery of an already-applied `action_id` counts again); enforcement is atomic under concurrent submits |
| TLS | stdlib `ssl`; server cert required to enable, peer/client CA verification |
| determinism clock | none — no wall clock in merge, hashing, or decisions |

### 8.3 Where the fabric stops being a good fit

- **More than a few dozen active nodes**: gossip is O(journal) per link per
  round; the mesh re-advertises itself only on handshake.
- **Tombstone / applied-set growth**: ORSet tombstones and the de-dup
  `action_id` set grow with distinct writes; compaction bounds disk (journal
  truncation) but not memory. Snapshots bound replay cost, not RAM.
- **Linearizable or ordered semantics**: none; every read is a lattice join of
  local + gossiped state.
- **Offline-first mobile/CBDB clients**: agents must hold an open connection to
  reach a node; there is no client-side offline session replay.
- **Measured throughput (current dev machine, `benchmarks/bench.py`)**: ≈24k
  actions/s single-node replay (20k actions), ≈2.8k actions/s absorbed across an
  8-agent TCP cluster, ≈0.18 ms TCP and ≈0.24 ms WS agent submit round-trips.
  Re-run `python benchmarks/bench.py` on the target host for local numbers;
  these are indicative, not a promise.

## 9. Conformance index

| # | Claim                                    | Where it's proven                       |
|---|------------------------------------------|-----------------------------------------|
| C1| join laws                                | `tests/test_crdts.py`                   |
| C2| one type per key                         | `tests/test_crdts.py`                   |
| C3| state_hash equality ⇒ identical state    | `tests/test_convergence.py`             |
| C4| permutation-independent replay           | `tests/test_replay.py`                  |
| H1| schema bounds + charset enforcement       | `tests/test_validate.py`, `tests/test_spec_conformance.py` |
| H2| signature / authz / namespace / quota    | `tests/test_validate.py`, `tests/test_namespace.py` |
| H3| TCP + WS framing, TLS on both            | `tests/test_websocket.py`, `tests/test_phase4.py` |
| H4| peer convergence, partition heal         | `tests/test_server.py`, `tests/test_phase4.py` |
| H5| failure semantics (crash/tamper/refusal)  | `tests/test_phase4.py`, `tests/test_validate.py`, `examples/durable_crew_demo.py` |
| D1| tamper detection on journal + snapshot   | `tests/test_phase4.py`, `tests/test_prod_hardening.py` |
| D2| torn tail vs. corrupt interior           | `tests/test_spec_conformance.py`        |
| D3| crash → recover → re-converge (e2e)      | `examples/durable_crew_demo.py`         |
| D4| cumulative generation rebuild            | `tests/test_phase4.py`                  |
| D5| signed store refuses unsigned lines/snaps| `tests/test_prod_hardening.py`          |
| D6| snapshot fsync survives rotation         | `tests/test_prod_hardening.py`          |
| D7| whole-node kill -9 and mid-rotation restart | `tests/test_prod_hardening.py`        |
| D8| signed log refuses reload without key; server refuses corrupt start | `tests/test_spec_conformance.py` |
| H6| no self-dial, peer-journal validation, client timeout surface | `tests/test_prod_hardening.py` |
| H7| async per-link sends + dial workers; stalled peer can't block gossip; reconnect dedup | `tests/test_prod_hardening.py`, `tests/test_websocket.py` |