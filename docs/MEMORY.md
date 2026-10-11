# Conflux Memory: a self-pruning memory layer for agent facts

Version 1 · Package version 0.7.3 · Module: `conflux.memory`

The memory layer gives an agent (or a crew of them) one place to write what it
knows, and a pass that keeps that place from silting up: restated facts get
folded, dead facts get retired, and anything the machine is not certain about
goes into a queue for a human instead of into the void. It is a SQLite database
with an audit trail, a policy engine, a review queue, a CLI, and an optional
HTTP service. Nothing in it needs a model, a network, or a server.

It is deliberately **not**:

- a vector database (there is no ANN index; similarity is computed over rows a
  blocking pass selects, and optional local embeddings only re-rank them);
- a source of truth for Conflux state (the CRDT fabric in `conflux/` stays
  authoritative for shared agent state; §11 mirrors memories onto it, not the
  reverse);
- a knowledge graph (a memory is one statement plus provenance, not entities
  and edges);
- a remote service (the only network calls in this package are to `127.0.0.1`
  model servers the operator started, and those are refused otherwise, §6).

Every numbered claim below is pinned by a test; the file is named with it.

## 1. The five things a pass may do

| Action    | What changes                                        | Reversible |
| --------- | --------------------------------------------------- | ---------- |
| `touch`   | `access_count`, `last_seen_at`, provenance union     | n/a        |
| `merge`   | N sources → `state='merged'`, one new folded record  | `revert_merge` |
| `archive` | `state='archived'` (default retirement target)       | `set_state('active')` |
| `prune`   | `state='pruned'` (opt-in retirement target)          | `set_state('active')` |
| `delete`  | `DELETE FROM memories`, only with two explicit opts  | audit trail keeps the text |

**Claim M1 (no silent deletion).** A cleanup pass never issues a `DELETE`. Hard
deletion requires `allow_hard_delete=True` **and** an explicit `older_than`
cutoff, and every deleted row is written to `memory_events` first with its full
record. — `tests/test_memory_store.py::test_hard_delete_requires_cutoff_and_keeps_the_record_in_the_trail`,
`tests/test_memory_api.py::test_purge_is_refused_until_it_is_switched_on`

**Claim M2 (transitions are compare-and-set).** `set_state(expect_revision=…)`
and `apply_merge(expected=…)` fail rather than overwrite a row someone else
moved; a merge that cannot claim all of its sources rolls back whole. —
`tests/test_memory_cleanup.py::test_apply_skips_a_cluster_whose_rows_moved`,
`::test_a_merge_is_atomic_when_one_source_is_gone`

## 2. Data model

One file, ten tables, `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, foreign
keys on, `schema_version` in `PRAGMA user_version` (currently 2) with
per-step migrations that run in their own transaction.

| Table             | Role                                                             |
| ----------------- | ---------------------------------------------------------------- |
| `memories`        | one row per statement; PK `(namespace, id)`; `revision`, `state`, `body_norm`, `fingerprint`, `content` provenance, embedding BLOB |
| `memory_events`   | append-only trail: `add`, `update`, `touch`, `seen`, `state:*`, `merge_*`, `run_*`, `hard_delete`, each with before/after JSON |
| `merges`          | a proposal: merged body, confidence, `cluster_key`, per-source snapshot (`sources`) |
| `merge_links`     | proposal ↔ memory join, so a memory's merge history is queryable |
| `reviews`         | the human queue: `kind` (`merge`, `contradiction`, `prune`, …), `state`, detail JSON |
| `runs`            | one ledger row per pass: trigger, mode, policy snapshot, `plan_hash`, `plan_json`, stats |
| `settings`        | the per-namespace policy document                                 |
| `run_lock`        | one row used as a lease so two processes cannot clean at once      |
| `review_cursor`   | `memory_id → fingerprint` of the last verdict, to skip re-judging  |
| `scheduler_state` | last fired slot, run id, status, next fire                         |

**Claim M3 (provenance survives a fold).** A merged record carries
`provenance.origin_ids`, `metadata.merged_from` and every source's original
text inside the proposal, so nothing a pass did is unexplainable after the fact.
— `tests/test_memory_cleanup.py::test_exact_duplicates_are_folded_with_full_provenance`,
`::test_merge_proposal_keeps_source_bodies_and_revisions`

**Claim M4 (namespaces are hard walls).** Every read and write is filtered by
namespace; a memory id that exists in `team-b` is a 404 in `default`. —
`tests/test_memory_store.py::test_policy_settings_roundtrip_and_namespaces_are_isolated`

**Claim M5 (migrations preserve data).** Opening an older database upgrades it in
place, one transaction per step; an unknown `user_version` is refused instead of
guessed. — `tests/test_memory_store.py::test_migrations_are_incremental_and_preserve_data`

## 3. Writes are idempotent by construction

`add()` normalizes (NFKC casefold, punctuation trimmed, whitespace collapsed),
computes a `content_fingerprint(kind, topic, body_norm)` and:

- on a fingerprint collision (default `on_duplicate="touch"`): unions provenance,
  bumps `access_count`/`last_seen_at`, returns `created=False,
  reason="exact_duplicate"` and **does not** move `revision` — so an open plan
  stays valid;
- with `idempotency_key`: replays the original receipt inside the same
  `BEGIN IMMEDIATE` transaction, whatever the body says;
- `ttl_days=N` is resolved to an absolute `expires_at` **inside `add()`**, so the
  library, the CLI and HTTP cannot mean different things by an expiry. A
  non-positive TTL is refused (it would store something already retired), as is
  setting `ttl_days` and `expires_at` together;
- `on_duplicate="insert"` keeps both rows for a pass to reason about;
  `"error"` raises `DuplicateMemory`.

**Claim M6 (a re-stated fact is not a new fact).** —
`tests/test_memory_store.py::test_exact_duplicate_touch_is_idempotent`,
`::test_idempotency_key_replays_without_new_rows`

## 4. How a pass decides

1. **Block.** An inverted index over word/char shingles plus topic buckets;
   a pair is compared only if it shares ≥ `min_shared_shingles` keys, or the two
   rows share a topic. Identical fingerprints are related by construction.
2. **Score.** `text_weight` × a lexical blend (Jaccard over shingles, containment
   for subset restatements, reorder tolerance) + `vector_weight` × cosine of the
   stored embeddings, when `use_embeddings` and vectors exist.
3. **Cluster.** Two-stage single/complete linkage: `near` when the score clears
   `duplicate_similarity` (0.78), `related` at `related_similarity` (0.45) — and
   complete linkage means **every** pair inside a cluster clears the bar, so
   chains cannot swallow a corpus. `max_cluster_size` splits anything huge.
4. **Propose.** Per cluster: a merge candidate plus a similarity verdict. Any
   contradiction in a cluster is sticky: the cluster goes to review.
5. **Judge staleness.** `expired_ttl` is the only rule that acts alone; age past
   `stale_after_days`, `unused_for_days` with a low `access_count`, and
   `min_importance` each only *raise* a candidate. Protection vetoes (`protect_kinds`,
   `protect_topics`, `protect_recent_days`, `pinned`) can only downgrade.
6. **Apply the floors** (§5), then, if not a dry run, apply under CAS.

Measured calibration (heuristic scorer, no embeddings): 0.793 *"Alice joined the
crew in March"* vs *"Bob joined…"* (contradiction, review), 0.750 *timeout 30*
vs *45* (contradiction), 0.692 *"never commit .env"* vs *"do not commit .env"*
(restatement), 0.481 unrelated-but-same-topic, 0.032 unrelated. A paraphrase
that shares no content words (0.179 in the calibration set) is a known miss of a
lexical scorer; storing embeddings lifts it.

**Claim M7 (clusters are deterministic).** Same rows, same policy → same
clusters, same ids, same scores, in any insertion order. —
`tests/test_memory_analysis.py::test_cluster_flags_contradictions_and_is_deterministic`

**Claim M8 (a contradiction is never folded by a machine).** —
`tests/test_memory_cleanup.py::test_contradictions_are_parked_for_review_and_both_survive`

## 5. The floors, and why a plan can be smaller than the analysis

| Knob                  | Default | Effect                                                  |
| --------------------- | ------- | ------------------------------------------------------- |
| `min_active_memories` | 8       | a pass may not leave fewer than this many active (0 = no floor) |
| `max_prune_fraction`  | 0.25    | share of the store one pass may retire, merges included  |
| `max_merge_fraction`  | 0.5     | share of the store whose rows one pass may fold          |
| `protect_kinds`       | policy, safety, instruction, credential | never pruned or merged away       |
| `prune_alone_in_topic`| False   | never retire the last active memory of a topic            |
| `allow_auto_prune`    | False   | staleness may only *recommend* until this is on          |
| `allow_hard_delete`   | False   | `DELETE` is refused at every entry point                 |

A guard only ever *downgrades* an action to `review` and reports itself; it
never drops an item from the plan silently.

**Claim M9 (no pass empties the store).** —
`tests/test_memory_cleanup.py::test_a_pass_never_removes_everything`,
`::test_max_prune_fraction_caps_one_run`, `::test_topics_are_never_emptied_by_one_pass`

**Claim M10 (a plan is inert the second time).** Each run stores `plan_hash`;
re-applying a plan that a completed `apply` run already executed returns
`already_applied` and changes nothing. — `tests/test_memory_cleanup.py::test_applying_the_same_plan_twice_is_a_no_op`

## 6. Models are optional, local, and never required

`policy.model` selects how a merged body is *phrased*. `null` (the default) and
`heuristic` need nothing at all: the heuristic is a deterministic sentence-level
union that keeps every distinct sentence and reports `dropped_facts` when
`max_merged_chars` forces a cut. An exact cluster is provably lossless, so it is
scored 1.0; a lossy fold is capped at 0.6 — below `auto_apply_confidence` (0.85)
— which means truncation always reaches a human.

`ollama[:model]`, `local[:model@base_url]` and `command:<argv>` are for a server
on this machine. `assert_local_endpoint` refuses a non-loopback URL unless
`allow_remote=True`; no code here reads, requests or sends an API key (an
operator may pass one explicitly for a local server that insists). A model that
is down, slow, or replies with prose instead of JSON yields `None`, the engine
falls back to the heuristic, and the run completes with a note.

**Claim M11 (a dead model costs nothing).** Every model path is exercised against
a stub HTTP server on loopback, and every failure mode falls back. —
`tests/test_memory_model.py`

## 7. Reviewing

`reviews` is the only queue a human needs: `kind` says whether a machine wanted
to fold (`merge`), disagreed with itself (`contradiction`), or wanted to retire
something (`prune`), and `detail` carries the proposed text, the rules that
fired, and the ids involved. Approving a `merge`/`contradiction` review executes
that proposal; approving a `prune` review moves the memory to its target state.
Rejecting leaves the memories exactly as they are — and the *next* pass proposes
a fresh merge row rather than resurrecting the rejected one.

`review_unchanged` (on) makes a pass skip records whose content fingerprint it
already judged; any edit to the row invalidates that, and `force=True` ignores
the cursor entirely.

**Claim M12 (resolving a review is one call, from CLI or API).** —
`tests/test_memory_api.py::test_review_queue_round_trip`,
`tests/test_memory_cli.py::test_reviews_merges_and_decisions`

## 8. Scheduling

`CleanupScheduler` is a daemon thread that calls `tick()` — the same function a
test can call with a stand-in clock. A slot is identified in UTC
(`2026-10-11T00:00`, or `interval:<epoch bucket>`) and claimed in
`scheduler_state`, so N processes sharing a database fire **once** per slot; a
skipped-but-not-run pass (the lease was busy) releases the slot for retry.

`schedule_grace_minutes` (120) is why a service started at 23:00 waits for
midnight instead of immediately replaying the slot it missed 23 hours ago, and
why a box that was down at 00:00 catches up at 01:30. DST uses real
`zoneinfo` rules: Berlin's spring-forward has no 02:30 on that day, and the slot
resolves to one UTC instant, fired once.

**Claim M13 (once per slot, in every clock direction).** —
`tests/test_memory_scheduler.py`

## 9. HTTP service

`python -m conflux.memory serve --db memories.db` (extras: `pip install 'conflux[memory]'`).
26 routes / 29 operations in five OpenAPI tags, docs at `/docs`; dry-run by
default everywhere, errors are `{"detail": …}` problems, and no CORS unless the
operator adds it. Reads are open; writes need
`Authorization: Bearer $CONFLUX_MEMORY_TOKEN` when a token is configured (401 +
`WWW-Authenticate`, never 403, because the caller may retry with credentials).
Authentication is checked before the body is parsed, so a stranger cannot make
the service validate a 600-item batch it was never going to accept.

Three response codes are the operational contract rather than decoration:
`413` for a body over `--max-body-bytes` (1 MiB default, refused before parse),
`503` + `Retry-After` for anything the store itself reported — `5` when it is
busy or the volume went read-only, `1` for a lock contention worth retrying,
`0` when the file will not open at all — and `400` for a request the policy
refused (a hard delete without the switch, an apply over a stale plan).
`GET /metrics` is Prometheus text and needs no token: a scraper must be able to
tell a degraded service from a stopped one.

| Route | Notes |
| ----- | ----- |
| `GET /health`, `GET /status`, `GET /safety` | integrity, counts, lock, schedule, the floors in force |
| `GET/PUT /settings`, `GET /settings/schema`, `POST /settings/reset` | PUT validates and coerces (`"75"` → 75.0); `dry_run:true` previews with `{"valid": false, "errors": [...]}` instead of raising; unknown keys are 400 |
| `POST /memories` | 201 new, 200 folded; extra keys are 422 (`extra="forbid"`); `on_duplicate` is a documented enum; `ttl_days` or `expires_at` (400 if both, or if `ttl_days <= 0`) |
| `POST /memories/batch` | up to 500 memories in **one transaction** with per-row receipts: `{written, folded, receipts}`. A batch is not a licence to hide a fold, and half a batch is worse than none |
| `GET /memories`, `GET /memories/{id}`, `GET …/events`, `POST …/seen`, `POST …/restore` | `state=all` (or empty) means every state; an unknown state is a 400 naming the valid ones, not an empty list |
| `POST /runs` | `{}` is a dry run; `{"dry_run": false}` mutates; `{"wait": false}` returns at once and the ledger tells the rest; contention is 409 |
| `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/items`, `POST /runs/{id}/apply` | a dry run's plan stays applicable |
| `GET /reviews`, `POST /reviews/{id}` | `{decision: approve|reject, note}`; re-deciding a decided review is refused, not repeated |
| `GET /merges[/{id}]`, `POST /merges/{id}/revert` | `state=open` covers proposed + needs_review |
| `POST /backups`, `POST /maintenance/purge` | `VACUUM INTO` snapshot; purge needs the hard-delete switch |
| `POST /maintenance/checkpoint`, `POST /maintenance/events` | fold the WAL; trim observation-only audit rows (`touch`/`seen`/`replay`), dry run by default, and state changes are unreachable through it |

**Claim M14 (the API is thin over the engine).** Every write path funnels
through `CleanupService`, so the CLI, the scheduler and HTTP cannot disagree —
and the response codes above are asserted, not assumed. — `tests/test_memory_api.py`

## 10. Command line

```
python -m conflux.memory add      --db m.db "cache TTL is 30s" --kind note \
                            --topic cache --ttl-days 30       # expires on its own
python -m conflux.memory sample   --db m.db --embeddings   # the 20-row teaching set
python -m conflux.memory run      --db m.db --explain      # dry run: the whole plan
python -m conflux.memory run      --db m.db --apply
python -m conflux.memory reviews  --db m.db --json
python -m conflux.memory review   --db m.db 4 --decision approve --note "same fact"
python -m conflux.memory merges   --db m.db --state applied
python -m conflux.memory revert   --db m.db 3 --note "they are different facts"
python -m conflux.memory restore  --db m.db mem-sample-11
python -m conflux.memory settings --db m.db --set stale_after_days=90 --dry-run
python -m conflux.memory status   --db m.db --json
python -m conflux.memory verify   --db m.db --mirror --fabric-dir ./fabric
python -m conflux.memory backup   --db m.db --dest snapshots/m.db
python -m conflux.memory checkpoint --db m.db             # fold the WAL (add --no-truncate to leave -wal)
python -m conflux.memory serve    --db m.db --port 8765 \
                            --durability balanced --max-body-bytes 262144
```

`--durability durable|balanced` is a store property, not a policy key: `durable`
is `synchronous=FULL` (one fsync per commit, the default), `balanced` is `NORMAL`
(a killed process loses at most the last transactions, a power loss can lose more)
— and `verify` reports which one the store was *opened* under next to the
integrity check, the read-only state and any interrupted runs, because
`synchronous` is a connection setting and a file cannot tell you how its writer
was configured. `serve` passes the byte budget to uvicorn's own
header/body limits too, so "refused at 1 MiB" is true before the request is
buffered, not after.

Exit codes: `0` success, `1` a typed failure (one line on stderr, no traceback),
`2` a usage error or a run that was refused because another cleanup holds the
lease. `--json` is available on every command that has output worth parsing.

**Claim M15 (the CLI is a wrapper, not a second implementation).** It calls
`CleanupService` for every mutation and `main(argv)` is importable, so the tests
below drive the real code path. — `tests/test_memory_cli.py`

## 11. Mirroring onto the Conflux fabric

`conflux.memory.bridge.MemoryFabric` projects one namespace as **one
LWW register per memory** at `memory/<namespace>/<id>`, carrying body, kind,
topic, provenance, `state` and `revision`. Retiring a memory is just a newer
register whose state is not `active` — no tombstone bookkeeping, no index set to
keep in sync.

Because registers order by the replica's `(tick, agent)` stamp and not by the
memory `revision`, a mirror can transiently show an older revision until the
next sync; `put()` therefore refuses to re-publish something older than what it
already carries (`force=True` overrides), `sync_store()` reports what moved, and
`verify_against()` names every difference instead of fixing one. The database
stays authoritative; the fabric gives you a `state_hash` a run can record, a
journal to replay, and gossip that converges.

**Claim M16 (order never matters, and drift is loud).** Replaying a journal
backwards, or flooding five replicas in random gossip order, yields the same
`state_hash` as the source; `verify_against` distinguishes missing, stale and
differing. — `tests/test_memory_bridge.py`

## 12. Tuning, and what to leave alone

The full list of 43 knobs with types, defaults, groups and one-line help is
`GET /settings/schema`, `conflux.memory.CleanupPolicy().schema()`, or
`python -m conflux.memory settings --db m.db`. The three that change behaviour
most:

- `duplicate_similarity` / `related_similarity` — lower them and a pass sees more
  clusters (and more review work); the gap between the two is what separates
  "fold it" from "group it".
- `allow_auto_prune` — off means nothing is ever retired by a machine. This is
  the recommended production setting until you have read a few runs.
- `min_active_memories` — the "small store" guard: with 3 rows and a floor of 8,
  every action downgrades to review, which is correct and looks like a bug until
  you read the guard line in the output.

Store-side defaults that are load-bearing and should not be relaxed casually:
`review_unchanged`, `prune_alone_in_topic`, `require_provenance_for_prune`,
`protect_recent_days`, `archive_instead_of_prune`, `allow_hard_delete=False`.

## 13. Known limits

- Lexical similarity misses a paraphrase that shares no content words; embeddings
  (or a model that rewrites titles) recover most of it, and the calibration set in
  `tests/test_memory_analysis.py` pins exactly where the line sits.
- `reviews(memory_id=…)` matches inside a JSON blob with `LIKE` — fine for a
  queue of thousands, not a search index.
- The fabric mirror is a projection, not a second store: §11 explains the one
  ordering caveat that follows from the register rule.
- `serve` is a single-process service: the run lease serializes passes across
  processes, but the HTTP layer adds no authentication beyond the bearer token and
  no rate limiting. Put it behind what you already use.
- A non-ASCII bearer token cannot survive an HTTP header (headers are decoded as
  latin-1), so a token with a stray accented character in it is unusable *by
  construction*; the guard answers 401 for it rather than raising out of a
  constant-time comparison. Generate tokens with `openssl rand -hex 32`.
- `compact_events` bounds the audit trail by age and action class only; there is
  no retention policy engine and no `TTL` on events. A store whose *adds* outpace
  its compaction still grows.
- The HTTP budget (`--max-body-bytes`) is per request, and a batch is one request:
  500 items × 100 000 characters is far over the default 1 MiB, so a client that
  wants to send a huge batch sends several smaller ones — which is also the
  behaviour you want after a crash, since each commit is independently durable.

## 14. Fault behaviour

Everything above is about what a pass *does*; this section is about what happens
when the box, the disk or the client misbehaves. The rule throughout: a fault is
classified, counted and recoverable, and nothing already committed is lost.

| Situation | Behaviour |
| --------- | --------- |
| process killed mid-write | SQLite replays its own journal; no committed row is lost. A run left `running` by a dead process is reaped on the next start and its ledger row reads `aborted` — it does not pretend to have finished |
| volume mounted read-only | the store reopens `file:…?mode=ro&immutable=1`, sets `read_only`, and says so in `info()`; reads serve, writes raise `StoreUnavailable`, HTTP answers 503, and the scheduler stays down instead of retrying into a log |
| disk full, `SQLITE_BUSY`, locked by another process | `classify_sqlite` maps each to a type with the retry hint that is true for it (`Retry-After: 5`, `1`, `0`); nothing retries in a tight loop |
| the file will not open / fails `PRAGMA integrity_check` | `StoreCorrupt` — reported, never silently recreated; `GET /health` carries the check output |
| a body with a sliced emoji, control characters, or 12 MB of text | U+FFFD substitution (a lone surrogate is a client bug, not a server error), a named refusal for characters SQLite would silently truncate, and 413 before parsing for size |
| two processes cleaning at once | the `run_lock` lease refuses the second one with 409; a lease whose owner died expires and is recorded |
| audit trail growth | reads and re-statements are most of `memory_events`; `POST /maintenance/events` trims exactly `touch`/`seen`/`replay`, older than N days, dry run first, and writes an `events_compacted` row into the trail it trims |
| restart after a long session | `close()` folds the WAL (best-effort: a failure there is logged, never raised out of a shutdown), and `checkpoint --truncate` does it on demand |

**Claim M17 (a batch is one commit and one decision).** `add_many` /
`POST /memories/batch` write N memories under a single transaction, so an agent
recording 200 facts pays one fsync and gets all-or-nothing durability; every row
still returns its own receipt, so batching never conceals a fold. Measured on this
box with `durability=durable`: 1.08 ms per single write vs 0.22 ms per write in a
batch of 200. The gap is fsync latency, so re-measure on your disk — the point is
that the batch path does one commit, not that this machine is fast.
— `tests/test_memory_faults.py`

**Claim M18 (compaction cannot rewrite history).** `compact_events` deletes only
rows whose action is in its drop list, whose default is `touch`/`seen`/`replay`;
an empty list is refused, a cutoff must be in the past, dry run is the default,
and the rows that changed a memory's state are not reachable through it. —
`tests/test_memory_faults.py`

**Claim M19 (degradation is a mode, not a crash).** The read-only, busy,
disk-full and corrupt cases are four types with four answers, all of them
observable without a traceback: `info()`/`health()` report `read_only`,
`degraded` and `durability`, HTTP answers 503 with a `Retry-After`, and the
scheduler refuses to fire rather than spinning. — `tests/test_memory_faults.py`

**Claim M20 (the wire in front of the store is supervised too).** Every frame
a peer or agent can send — oversized, non-UTF-8, non-object, over-deep,
truncated, or one that asks for more work than `max_actions_per_frame` — is
refused before it touches state, counted, and answered; per-link state is a
journal cursor, so a 20 000-action journal gossips at the cost of what it
delivered. — `tests/test_fault_transport.py`, `docs/PROTOCOL.md` §6,
`docs/SECURITY.md` S7–S8

## 15. Where to read next

- `examples/self_pruning_memory_demo.py` — the whole story in one file,
  asserted by `tests/test_demos.py`.
- `conflux/memory/policy.py` — the knobs, validation and coercion, in one place.
- `conflux/memory/cleanup.py` — plan, floors, apply; ~600 lines with no I/O
  beyond the store.
- `docs/PROTOCOL.md` — the fabric the mirror plugs into.
