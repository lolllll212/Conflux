# Conflux Security Review

Package version: 0.7.3 · Protocol version: 1 (action envelope) ·
Reviewed at commit `689e62e` · Review date: 2026-09-22

This document is the formal review of Conflux's cryptographic signing,
identity verification, and storage subsystems. It records what the
implementation enforces, **where** (module and function), and **which test
proves it**, in the same claim → test style as
[`PROTOCOL.md`](PROTOCOL.md). It also states the trust boundary plainly and
lists the findings that remain open. Every property below was re-verified
against the code at the commit above, not transcribed from prior docs.

Modules in scope: `conflux/validate.py` (envelopes, `AgentRegistry`),
`conflux/storage.py` (`JournalStore`), `conflux/state.py` (`Replica` dedup),
`conflux/actions.py` (canonical action shape), `conflux/net.py` (framing,
TLS), and the `submit` / `journal` paths of `conflux/server.py`.

## 1. Summary

| # | Property | Status | Enforced in |
|---|----------|--------|-------------|
| S1 | Constant-time signature comparison (wire + disk) | ✅ holds | `validate.verify_action`, `storage.JournalStore._verify` |
| S2 | Algorithm binding — no confusion, no downgrade, no `alg: "none"` | ✅ holds | `validate._envelope_algorithm`, `validate.validate_envelope`, `AgentRegistry.register` |
| S3 | Canonical serialization invariant | ✅ holds | `validate.canonical_action_bytes`, `storage.JournalStore._sign`, `protocol.state_hash` |
| S4 | Signer identity and scope binding; verification precedes ACL/quota | ✅ holds | `validate.validate_envelope` |
| S5 | Replay and malleability defenses | ✅ holds | `state.Replica.apply` / `absorb`, lattice join laws |
| S6 | Keyless reload refusal; per-entry journal/snapshot integrity | ✅ holds | `storage.JournalStore.load` / `_verify` |

Open findings (§5): **one High** (F1 — the peer gossip path is
unauthenticated and bypasses the registry), one Low (F2), four
Informational (F3–F6). None of the open findings weakens S1–S6 as stated;
F1 limits *where* S2/S4 apply (the agent `submit` path only).

## 2. Threat model and trust boundary

Conflux is a small-crew state fabric for a few dozen agents of one
application ([`PROTOCOL.md` §1](PROTOCOL.md)). The security model is built
around that scope.

**Actors**

| Actor | Trust | Notes |
|-------|-------|-------|
| Registered agent | Trusted for its own writes, within its `ops` / `namespaces` / `quota` | Holds an HMAC secret or an Ed25519 private key |
| Peer node | **Fully trusted** | Anything a peer sends as a `journal` frame is absorbed after schema validation only (F1) |
| Operator | Fully trusted | Holds the `journal_secret`, TLS keys, and the registry |
| Network attacker | Partly mitigated | TLS gives confidentiality; peer authenticity depends on how TLS is configured (F4) |
| Disk attacker (offline access to `data_dir`, no key) | Mitigated for authenticity, not for completeness | Per-entry HMAC (S6); deletion/rollback not detected (F3) |

**What is authenticated today**

- Agent `submit` frames, **when** the `Server` is constructed with an
  `AgentRegistry`. Each envelope is schema-checked, signature-verified
  against the signer's registered key, then checked for op allowlist,
  namespace prefix, and quota — in that order (S4). A server without a
  registry is explicitly a trust-less dev mode (README, "Protocol
  hardening").
- Journal lines and snapshots on disk, **when** a `journal_secret` is
  configured (S6).

**What is not authenticated today**

- The peer protocol. A `hello` with `"role": "peer"` is accepted from any
  connection, and `journal` frames are absorbed from any link after
  `validate_action` only — no signature, no registry (F1).
- Reads (`read`, `read_all`, `hash`), `sync` (returns the full journal), and
  `admin` (F5).
- Inbound TLS connections: the server never requests a client certificate.
  Outbound dials verify the far node only when a CA is configured via
  `Server(tls=(cert, key, ca))`; the CLI's `--cert/--key` form disables
  verification (F4).

**Consequence.** Every host that can open a TCP/WebSocket connection to a
node's listen port must be treated as a trusted peer. Under that boundary
the registry provides per-agent accounting and accident prevention against
*well-behaved-but-misconfigured* agents, and adversarial protection only
against parties who cannot reach the port at all.

**Assumptions the guarantees rest on**

1. Secrets are high-entropy (≥ 32 random bytes; see §6) and are not shared
   across independent deployments — an envelope is valid forever, so a
   secret reused elsewhere lets a captured envelope be replayed into the
   other deployment.
2. `action_id`s are unique per event and honest agents mint ids inside
   their own prefix (`act-<agent>-<tick>`), per
   [`PROTOCOL.md` §8.1](PROTOCOL.md) ("unique IDs are load-bearing"); see F6.
3. The optional `cryptography` package (OpenSSL) is trusted for Ed25519.

## 3. Cryptographic inventory

| Purpose | Primitive | Implementation | Key material |
|---------|-----------|----------------|--------------|
| Envelope MAC (default) | HMAC-SHA256, hex digest | stdlib `hmac` / `hashlib` — `validate.sign_action`, `validate.verify_action` | `str` secret, UTF-8 encoded; per agent in `AgentRegistry` |
| Envelope signature (asymmetric) | Ed25519 (RFC 8032), base64 signature | `cryptography` (optional extra) — `validate.sign_ed25519`, `validate.verify_ed25519` | 32-byte seed / 32-byte raw public key; `generate_ed25519_keypair()` draws the seed from `os.urandom(32)` |
| Journal line + snapshot MAC | HMAC-SHA256, hex digest | stdlib — `storage.JournalStore._sign` / `_verify` | `journal_secret` (`str` or `bytes`), one per node |
| State fingerprint | SHA-256 over canonical state | `protocol.state_hash` | none — a convergence check, not an authenticator |
| Weighted decisions | SHA-256 as a deterministic PRF over `(action_id, key)` | `actions.decide` | none — determinism only, no security claim |
| Transport | TLS via stdlib `ssl` (`PROTOCOL_TLS_SERVER`; `create_default_context(SERVER_AUTH)` for dials) | `net.TLSConfig` | PEM cert/key, optional CA |

No custom primitives, no home-grown padding, no key derivation. HMAC keys
are used as given; there is no stretching, so low-entropy secrets are
brute-forceable offline from a single captured plaintext envelope (§6).

## 4. Reviewed properties

Each property lists the enforcing code, how it was verified, and any caveat
found during review.

### S1 — Side-channel timing protection

**Property.** Every comparison of a received MAC against a recomputed one
is constant-time, so an attacker cannot learn a valid tag byte-by-byte from
response latency.

**Enforced in.**

- Wire: `validate.verify_action` recomputes
  `hmac.new(secret, canonical_action_bytes(action_dict), sha256).hexdigest()`
  and returns `hmac.compare_digest(expected, signature)`.
- Disk: `storage.JournalStore._verify` returns only after
  `hmac.compare_digest(self._sign(payload), str(sig))`; a missing tag is
  rejected before any comparison.
- Ed25519: verification is delegated to `Ed25519PublicKey.verify`
  (OpenSSL); it is a public-key operation with no secret-dependent
  comparison on the verifier side. All failures (bad base64, wrong length,
  `InvalidSignature`) are collapsed into `False` by `verify_ed25519`.

**Verified by.** `tests/test_security_review.py::test_wire_and_journal_verification_use_constant_time_compare`
(spies on `hmac.compare_digest`; a regression to `==` fails the test),
`tests/test_validate.py::test_signature_round_trip_and_tamper_detection`,
`tests/test_phase4.py::test_signed_journal_tamper`.

**Caveat.** `hmac.compare_digest` raises `TypeError` for `str` operands that
contain non-ASCII characters. `verify_action` checks that `signature` is a
`str` but not that it is ASCII, so a non-ASCII signature raises instead of
returning `False`. The request is still refused (F2).

### S2 — Algorithm binding: no confusion, no downgrade

**Property.** An identity is bound to exactly one key type at registration
time, an envelope must name that type, and no unauthenticated or unknown
algorithm is ever accepted.

**Enforced in.**

- `AgentRegistry.register(agent_id, secret=None, public_key=None, ...)`
  stores `kind = ALG_HMAC` when a `secret` is given, `kind = ALG_ED25519`
  when a `public_key` is given, and raises `ValidationError` if neither.
  `AgentRegistry.secret()` returns `None` for non-HMAC identities.
- `validate._envelope_algorithm` maps `envelope["alg"]` through an
  allowlist `{"hmac-sha256", "ed25519"}`; anything else — including
  `"none"`, `"HS256"`, or `""` — becomes `None` and `validate_envelope`
  raises `"envelope uses an unsupported signature algorithm"`. An absent
  `alg` defaults to `hmac-sha256` for backward compatibility, which is safe
  because of the next check.
- `validate_envelope` then requires `kind == alg` for the registered
  `signed_by`, raising `"envelope algorithm does not match the agent's
  registered key"`. Consequences: an HMAC envelope cannot be verified
  against an Ed25519 identity, an Ed25519 public key can never serve as an
  HMAC secret (even with `alg` omitted), and an Ed25519 identity cannot be
  downgraded to HMAC.
- Defense in depth: `verify_action` returns `False` for any non-`str`
  secret, so a raw 32-byte Ed25519 public key is unusable as an HMAC key
  even when the function is called directly.

**Verified by.**
`tests/test_security_review.py::test_alg_none_and_unknown_algorithms_are_rejected`,
`::test_algorithm_must_match_registered_key_kind`,
`::test_ed25519_public_key_cannot_be_used_as_hmac_secret`,
`tests/test_validate.py::test_ed25519_registry_and_server_reject_wrong_key`.

### S3 — Canonical serialization invariant

**Property.** Every byte that is signed, MAC'd, or hashed is produced by one
recipe: `json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")`.
Two nodes therefore compute identical bytes for identical logical content
regardless of dict insertion order or platform.

**Enforced in.** `validate.canonical_action_bytes` (used by `sign_action`,
`verify_action`, `sign_ed25519`, `verify_ed25519`), `storage.JournalStore._sign`
(journal lines and snapshots), `protocol.state_hash`, and the `params` size
check in `validate.validate_action`.

The signer MACs `action.export()` (all seven fields, `params` copied to a
plain `dict`); the verifier MACs the received `envelope["action"]` dict as
is. The two agree only if the received object canonicalizes to the same
bytes, so any inserted, removed, or mutated field invalidates the signature.

**Verified by.**
`tests/test_security_review.py::test_canonical_bytes_are_key_order_independent`,
`tests/test_validate.py::test_signature_round_trip_and_tamper_detection`,
`tests/test_convergence.py::test_merge_order_is_irrelevant` (C3).

**Interoperability notes.** The recipe inherits Python's `json.dumps`
defaults: separators `", "` / `": "`, shortest round-trip `repr` for floats,
and `allow_nan=True` (a `NaN` value would be emitted as the non-standard
token `NaN`). A non-Python implementation must reproduce these exactly to
verify Python-produced signatures.

### S4 — Signer identity and scope binding

**Property.** A signature is always checked against the key of the agent
that claims to have produced the action, and only a verified envelope can
reach authorization, namespace, or quota logic.

**Enforced in.** `validate.validate_envelope`, whose checks run in this
fixed order; each failure raises `ValidationError` and stops:

1. envelope is a mapping with an `action` mapping;
2. `validate_action(Action.import_action(action))` — schema, bounds, charset;
3. `signed_by` is a `str` **and equals `action.agent`**;
4. `alg` is allowlisted (S2);
5. `signed_by` is registered (`registry.verifier`), else `unknown agent`;
6. registered key kind equals `alg` (S2);
7. signature verifies against **`signed_by`'s** registered key;
8. `registry.authorize(signed_by, action.op)` — op allowlist;
9. `registry.namespace_allows(signed_by, action.key)` — key prefix allowlist;
10. `registry.consume(signed_by)` — quota, taken under the registry lock.

Consequences: an agent cannot sign an action attributed to another agent
(steps 3 + 7); an unauthenticated envelope cannot consume quota or probe
namespace policy (7 precedes 9–10); the fields outside the signed bytes
(`signed_by`, `alg`) are each pinned by an independent check, so flipping
either on a captured envelope is rejected.

**Verified by.**
`tests/test_security_review.py::test_signed_by_must_equal_action_agent`,
`::test_bad_signature_is_rejected_before_quota_or_namespace`,
`tests/test_validate.py::test_envelope_validation_integration`,
`::test_rejected_submits_do_not_mutate_state`, `::test_concurrent_quota_is_not_overdrawn`,
`tests/test_namespace.py::test_validate_envelope_enforces_namespace`.

**Note.** Error reasons are stage-specific (`unknown agent` vs.
`signature verification failed`), which lets a caller learn whether an agent
id is registered. Agent ids are not treated as secrets, so this is recorded
for completeness only.

### S5 — Replay and malleability defenses

**Property.** Re-delivering an already-applied action — by a retrying
client, a reconnecting peer, a journal replay, or an attacker replaying a
captured envelope — has no effect on state.

**Enforced in.**

- `Replica.apply` / `Replica.absorb` (`conflux/state.py`) keep the set of
  applied ids in `Replica._applied` (exposed as `ids_applied()`) and skip
  any `action_id` already present. This is the single dedup point for
  local emits, `submit`, peer `journal` frames, and startup replay from
  `JournalStore`.
- Even without dedup, every contribution is folded with the lattice join,
  which is idempotent (`a ⊔ a = a`), commutative, and associative (C1); a
  duplicate application yields the same state.
- Each action carries a unique `action_id` and a signer-asserted logical
  `tick` (≥ 1). Ticks order LWW registers via `Stamp(tick, agent)`; they are
  not required to be monotonic by the validator — monotonicity is the
  emitting `Replica`'s convention and a spec assumption (§8.1).
- Malleability: the MAC/signature covers the entire canonical action; the
  envelope fields outside it are bound by S4. Ed25519 as implemented by
  OpenSSL rejects non-canonical signatures, and dedup is keyed on
  `action_id`, not on signature bytes, so signature malleability cannot
  create a "new" event.

**Verified by.**
`tests/test_security_review.py::test_replayed_envelope_does_not_change_state`,
`tests/test_prod_hardening.py::test_reconnect_duplicates_do_not_double_apply` (H7),
`tests/test_spec_conformance.py::test_restart_replay_has_no_reapplication`,
`tests/test_crdts.py` (C1 join laws), `tests/test_replay.py` (C4).

**Caveats.** (a) Quota is charged per validated envelope, so replaying a
captured envelope burns the victim agent's quota even though state is
unchanged ([`PROTOCOL.md` §8.2](PROTOCOL.md)); TLS prevents capture.
(b) Envelopes carry no expiry or nonce beyond `action_id`; replay into a
deployment that has not seen the id is *intended* (it is the recovery
model), which is why secrets must not be shared across deployments (§2).
(c) Because dedup is by id alone, the first body seen under an id wins —
see F6.

### S6 — Keyless reload refusal and journal integrity

**Property.** A node cannot silently downgrade to unverified replay: signed
on-disk state refuses to load unless the signing key is supplied, and any
verifiable line or snapshot that fails verification aborts the load.

**Enforced in** `storage.JournalStore`:

- `append` writes `{"action", "seq", "sig"}` when `journal_secret` is set,
  with `sig = HMAC-SHA256(journal_secret, canonical(action))`, then
  `flush()` + `os.fsync`.
- `rotate` signs the snapshot as `HMAC-SHA256(journal_secret, canonical({"seq", "root"}))`
  and publishes it atomically (`os.replace`, directory fsync).
- `load` — snapshot: if `"sig" in snapshot` and `journal_secret is None`,
  raise `"snapshot is signed but no journal_secret was provided"`; otherwise
  `_verify({"seq", "root"}, sig)`.
- `load` — journal: every parsed line goes through `_verify(entry["action"], entry.get("sig"))`,
  which raises on a missing tag (`"journal signature missing"`) or a
  mismatched tag (`"journal signature mismatch"`); after the scan, if any
  line carried a `sig` and no key is configured, raise
  `"journal lines are signed but no journal_secret was provided"`. `load`
  therefore never returns actions from a signed log without verifying them.
- A key configured against an *unsigned* store is also refused (missing
  tag), so an attacker cannot strip signatures to disable verification.
- Interior corruption (unparseable non-final line) raises; only an
  unparseable **final** line is skipped, as a mid-write torn tail (D2).

**Verified by.**
`tests/test_security_review.py::test_signed_journal_lines_refuse_reload_without_key`,
`tests/test_spec_conformance.py::test_signed_log_refuses_to_reload_without_key` (D8, snapshot path),
`tests/test_prod_hardening.py::test_signed_journal_refuses_unsigned_line`,
`::test_signed_journal_refuses_unsigned_snapshot` (D5),
`tests/test_phase4.py::test_signed_journal_tamper`, `::test_signed_snapshot_tamper` (D1),
`tests/test_spec_conformance.py::test_truncated_early_line_is_corruption` (D2).

**Caveats.** The journal-line tag covers `action` only, not `seq`
(sequence numbers are unauthenticated but unused for state). Tags are
independent per entry — there is no hash chain — so deletion, truncation,
or reordering of whole lines is not detected (F3). Reordering is harmless
(the join is order-independent, C4); deletion is a rollback.

## 5. Open findings

Severity reflects impact under the trust boundary of §2.

### F1 — High: the peer protocol is unauthenticated and bypasses the registry

**Description.** `Server._handle_message` applies `validate_envelope` only
to `submit` frames. `journal` frames are absorbed after `validate_action`
(schema only) from **any** link: an `agent`-role connection can send a
`journal` frame directly, and any connection can obtain a peer link simply
by sending `{"type": "hello", "role": "peer"}`, since peer hellos carry no
credential and TLS never requests client certificates.

**Impact.** Any party that can reach a node's listen port can inject
arbitrary actions — any `agent`, any `key`, any `op`, unlimited — into a
registry-protected node. Injected actions are journaled (and signed by the
node's own `journal_secret`) and gossiped to every peer, so the write is
durable and cluster-wide. This voids the adversarial value of `ops`,
`namespaces`, `quota`, and signatures for any attacker with network reach.
It does not affect S1–S6 as stated, but it means they protect the
`submit` path only.

**Reproduction (verified during review).** Open a socket, send
`{"type":"hello","role":"agent"}`, then
`{"type":"journal","actions":[{...unsigned action as agent "mallory"...}]}`;
the value is readable back and appears in `Server.view()`. The same works
with `"role":"peer"` in the hello.

**Recommended fix.** Authenticate the peer layer explicitly: a cluster-wide
peer secret (`Server(peer_secret=...)`, `--peer-key`) with an HMAC over each
peer frame and a challenge in the hello, or mutual TLS
(`ssl.CERT_REQUIRED` with a CA on the server context). Independently,
refuse `journal` frames on `agent`-role links (`source is None`). Until
then, place the listen port on a private network and treat every host that
can reach it as a fully trusted peer.

**Tracking.** [lolllll212/Conflux#2](https://github.com/lolllll212/Conflux/issues/2).

### F2 — Low: malformed envelopes raise `TypeError` (fail-closed, not fail-clean)

**Description.** Three inputs escape `validate_envelope` as `TypeError`
rather than `ValidationError`: a non-ASCII `signature` string
(`hmac.compare_digest`), and an `action` object with extra or missing keys
(`Action(**data)` in `Action.import_action`). `_handle_message` catches
only `ValidationError` on the `submit` path, so the connection thread
raises, a traceback is logged, and the socket is closed without the
documented `{"type":"error"}` reply. No state is mutated and other
connections are unaffected. (`read` with a missing `key` is the same class:
`KeyError`.)

**Recommended fix.** In `validate_envelope`, wrap `Action.import_action` in
`try/except TypeError → ValidationError` and reject non-ASCII signatures
before comparison; additionally catch `(TypeError, KeyError, ValueError)`
around the `submit` handler, as the peer `journal` branch already does.

### F3 — Informational: per-entry journal MAC provides authenticity, not completeness

Each journal line is MAC'd independently and the tag excludes `seq`; there
is no chain or running digest across lines, and the torn-tail rule accepts
a truncated final line by design. An attacker with write access to
`data_dir` but no key can therefore delete lines (or the whole journal) to
roll a node back without detection. Snapshots partially bound the damage
(their `{seq, root}` is signed). **Fix if required:** include `seq` and the
previous line's tag in each line's MAC, and record the expected journal
length in the snapshot.

### F4 — Informational: TLS provides confidentiality; peer authentication depends on configuration

`net.normalize_tls((cert, key))` builds a `TLSConfig` with `verify=False`
(`CERT_NONE`, no hostname check) for outbound dials, and the CLI exposes
only `--cert/--key` (no `--ca`), so CLI-launched nodes accept any far-end
certificate. Server contexts never request client certificates, so inbound
connections are never authenticated by TLS (see F1). `Server(tls=(cert, key, ca))`
enables verification of dialed peers. **Fix:** add `--ca`, default to
verifying when one is supplied, and offer mutual TLS.

### F5 — Informational: reads, `sync`, and `admin` are unauthenticated

`read`, `read_all`, `hash`, `sync` (which returns the complete journal —
every action of every agent) and `admin` require no identity. Confidentiality
of state rests on TLS and network access control. Gate these behind the
registry (signed requests or an authenticated session) if state is
sensitive.

### F6 — Informational: `action_id` is the identity of an event; an authorized agent can squat ids

Dedup is keyed on `action_id` alone and the validator does not require the
id to lie within the signer's namespace. If two *different* bodies are
submitted under one id (a malicious or buggy authorized agent minting
`act-<victim>-<n>` first), nodes that receive them in different orders
apply different bodies and **diverge permanently** — later gossip is
deduped on both sides (verified during review: two replicas end with
different `state_hash`). On a single ingress node the effect is instead a
silent drop: the second submit is acknowledged (`accepted` counts validated
envelopes) but never applied. [`PROTOCOL.md` §8.1](PROTOCOL.md) declares
unique ids load-bearing, so this is a documented assumption rather than a
defect; it is listed here because it is the one way an *authorized* agent
can break convergence. **Fix if required:** require
`action_id.startswith(f"{agent}:")` (or a similar signer-bound prefix) in
`validate_envelope`, or key dedup on `(agent, action_id)`.

## 6. Deployment checklist

- [ ] Always construct the `Server` with an `AgentRegistry`; a server
      without one accepts unsigned submits (dev mode).
- [ ] Generate secrets with ≥ 32 bytes of entropy
      (`openssl rand -hex 32`); never reuse a secret across deployments.
- [ ] Prefer Ed25519 (`pip install "conflux[crypto]"`) where the party that
      verifies is not the party that signs; an HMAC secret held by the node
      can also *produce* valid envelopes for that agent.
- [ ] Set `journal_secret` (`--journal-key`) on every node with a
      `data_dir`, and keep the key out of the data directory. Store it
      where the operator — not the node's disk — controls it; without it a
      signed store cannot be loaded (S6, by design).
- [ ] Enable TLS on both transports and pass a CA to dialing nodes
      (`Server(tls=(cert, key, ca))`) so peers verify each other (F4).
- [ ] Until F1 is fixed, bind listen ports to a private network and
      firewall them to known peers and agent hosts; treat every host that
      can reach the port as a trusted peer.
- [ ] Give each agent the narrowest `ops` and `namespaces` that fit, and a
      `quota` sized to its workload; both are enforced only on `submit`.
- [ ] Mint `action_id`s inside the agent's own prefix and keep ticks
      monotonic (F6, `PROTOCOL.md` §8.1).
- [ ] Do not expose `start_metrics_http` or the `admin` probe beyond the
      operations network (F5).

## 7. Reporting a vulnerability

Open a GitHub issue on `lolllll212/Conflux` for hardening findings. For a
vulnerability that is exploitable in a deployment, contact the maintainer
privately first (the repository owner's GitHub profile) and allow a
reasonable window before public disclosure. Please include the package
version, the transport (`tcp`/`ws`, TLS on or off), whether a registry and
`journal_secret` were configured, and a minimal reproduction.

## 8. Verification index

| # | Claim | Where it's proven |
|---|-------|-------------------|
| S1 | constant-time compare on wire and disk | `tests/test_security_review.py`, `tests/test_validate.py`, `tests/test_phase4.py` |
| S2 | `alg` allowlist; key kind ↔ `alg` binding; no Ed25519-as-HMAC | `tests/test_security_review.py`, `tests/test_validate.py` |
| S3 | canonical bytes are order-independent; any mutation breaks the MAC | `tests/test_security_review.py`, `tests/test_validate.py`, `tests/test_convergence.py` |
| S4 | `signed_by == agent`; signature before ACL/namespace/quota | `tests/test_security_review.py`, `tests/test_validate.py`, `tests/test_namespace.py` |
| S5 | replayed action is a no-op on state | `tests/test_security_review.py`, `tests/test_prod_hardening.py`, `tests/test_spec_conformance.py`, `tests/test_crdts.py` |
| S6 | signed lines and snapshots refuse keyless / tampered / stripped reload | `tests/test_security_review.py`, `tests/test_spec_conformance.py`, `tests/test_prod_hardening.py`, `tests/test_phase4.py` |

Open findings F1–F6 are intentionally **not** pinned by tests, so that
fixing them does not require rewriting this document's evidence; each fix
should add its own conformance test and move the finding to §4.
