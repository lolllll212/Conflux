"""Conformance tests for the properties documented in docs/SECURITY.md (S1-S6).

Each test pins one reviewed property to the code path that enforces it, in
the same spirit as PROTOCOL.md's conformance index. Open findings (F1-F6 in
the review) are deliberately not pinned here.
"""
import hashlib
import hmac
import json

import pytest

from conflux import Action, AgentRegistry, Replica, ValidationError, sign_action, state_hash
from conflux.storage import JournalStore
from conflux.validate import (
    ALG_ED25519,
    ALG_HMAC,
    canonical_action_bytes,
    validate_envelope,
    verify_action,
)


def make_action(**overrides):
    fields = {"action_id": "act-a-1", "agent": "a", "tick": 1, "op": "counter_inc",
              "key": "orders.total", "params": {"by": 1}}
    fields.update(overrides)
    return Action(**fields)


def hmac_registry(agent="a", secret="secret-a", **kwargs):
    registry = AgentRegistry()
    registry.register(agent, secret, **kwargs)
    return registry


# --- S1: constant-time comparison ------------------------------------------------


def test_wire_and_journal_verification_use_constant_time_compare(monkeypatch, tmp_path):
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)

    envelope = sign_action(make_action(), "secret-a")
    assert verify_action(envelope, "secret-a")
    assert len(calls) == 1, "wire verification must go through hmac.compare_digest"

    store = JournalStore(str(tmp_path / "ct"), journal_secret="journal-key")
    store.append(make_action())
    store.close()
    calls.clear()
    root, actions = JournalStore(str(tmp_path / "ct"), journal_secret="journal-key").load()
    assert len(actions) == 1
    assert len(calls) == 1, "journal verification must go through hmac.compare_digest"


# --- S2: algorithm binding --------------------------------------------------------


def test_alg_none_and_unknown_algorithms_are_rejected():
    registry = hmac_registry()
    envelope = sign_action(make_action(), "secret-a")
    assert validate_envelope(envelope, registry).op == "counter_inc"

    for bogus in ("none", "None", "HS256", "", "hmac-sha1"):
        forged = dict(envelope, alg=bogus)
        with pytest.raises(ValidationError, match="unsupported signature algorithm"):
            validate_envelope(forged, registry)

    # an absent alg defaults to hmac-sha256 and still verifies against the HMAC key
    legacy = {k: v for k, v in envelope.items() if k != "alg"}
    assert validate_envelope(legacy, registry).op == "counter_inc"


def test_algorithm_must_match_registered_key_kind():
    pytest.importorskip("cryptography")
    from conflux import generate_ed25519_keypair

    private, public = generate_ed25519_keypair()
    registry = AgentRegistry()
    registry.register("h", "secret-h")
    registry.register("e", public_key=public)

    hmac_env = sign_action(make_action(agent="h", action_id="act-h-1"), "secret-h")
    with pytest.raises(ValidationError, match="does not match the agent's registered key"):
        validate_envelope(dict(hmac_env, alg=ALG_ED25519), registry)

    ed_env = sign_action(make_action(agent="e", action_id="act-e-1"), private, ALG_ED25519)
    assert validate_envelope(ed_env, registry).agent == "e"
    with pytest.raises(ValidationError, match="does not match the agent's registered key"):
        validate_envelope(dict(ed_env, alg=ALG_HMAC), registry)
    # an Ed25519 identity cannot be downgraded by omitting alg (defaults to HMAC)
    with pytest.raises(ValidationError, match="does not match the agent's registered key"):
        validate_envelope({k: v for k, v in ed_env.items() if k != "alg"}, registry)


def test_ed25519_public_key_cannot_be_used_as_hmac_secret():
    pytest.importorskip("cryptography")
    from conflux import generate_ed25519_keypair

    _, public = generate_ed25519_keypair()
    registry = AgentRegistry()
    registry.register("e", public_key=public)

    action = make_action(agent="e", action_id="act-e-1")
    # attacker knows the public key and MACs the action with it
    forged_sig = hmac.new(public, canonical_action_bytes(action.export()),
                          hashlib.sha256).hexdigest()
    forged = {"action": action.export(), "signed_by": "e", "signature": forged_sig,
              "alg": ALG_HMAC}
    with pytest.raises(ValidationError):
        validate_envelope(forged, registry)
    with pytest.raises(ValidationError):
        validate_envelope({k: v for k, v in forged.items() if k != "alg"}, registry)
    # defense in depth: a bytes key is never accepted by the HMAC verifier directly
    assert verify_action(forged, public) is False


# --- S3: canonical serialization --------------------------------------------------


def test_canonical_bytes_are_key_order_independent():
    a = {"b": 1, "a": {"y": "ü", "x": [1, 2.5, None, True]}}
    b = {"a": {"x": [1, 2.5, None, True], "y": "ü"}, "b": 1}
    assert canonical_action_bytes(a) == canonical_action_bytes(b)
    assert canonical_action_bytes(a) == json.dumps(
        a, sort_keys=True, ensure_ascii=False).encode("utf-8")
    # the same recipe covers an Action and its exported dict
    action = make_action(params={"by": 2, "weight": 0.5})
    assert canonical_action_bytes(action) == canonical_action_bytes(action.export())


# --- S4: signer binding and check order -----------------------------------------


def test_signed_by_must_equal_action_agent():
    registry = AgentRegistry()
    registry.register("a", "secret-a")
    registry.register("b", "secret-b")
    envelope = sign_action(make_action(agent="a"), "secret-a")
    # b re-labels a's envelope as its own: refused before any key lookup
    with pytest.raises(ValidationError, match="signed_by does not match"):
        validate_envelope(dict(envelope, signed_by="b"), registry)
    # b signs an action attributed to a with b's own key: refused at verification
    forged = sign_action(make_action(agent="a", action_id="act-a-2"), "secret-b")
    with pytest.raises(ValidationError, match="signature verification failed"):
        validate_envelope(forged, registry)


def test_bad_signature_is_rejected_before_quota_or_namespace():
    registry = AgentRegistry()
    registry.register("q", "secret-q", quota=1, namespaces=("orders.",))

    bad_sig = sign_action(make_action(agent="q", action_id="act-q-1", key="secret.flag"), "WRONG")
    with pytest.raises(ValidationError, match="signature verification failed"):
        validate_envelope(bad_sig, registry)
    assert registry.quota_left("q") == 1, "an unverified envelope must not consume quota"

    out_of_scope = sign_action(make_action(agent="q", action_id="act-q-1", key="secret.flag"),
                               "secret-q")
    with pytest.raises(ValidationError, match="cannot write keys outside"):
        validate_envelope(out_of_scope, registry)
    assert registry.quota_left("q") == 1, "namespace refusal precedes quota consumption"

    ok = sign_action(make_action(agent="q", action_id="act-q-1"), "secret-q")
    assert validate_envelope(ok, registry).key == "orders.total"
    assert registry.quota_left("q") == 0


# --- S5: replay idempotence -------------------------------------------------------


def test_replayed_envelope_does_not_change_state():
    registry = hmac_registry()
    replica = Replica("node")
    envelope = sign_action(make_action(params={"by": 3}), "secret-a")

    replica.absorb([validate_envelope(envelope, registry)])
    before = state_hash(replica.state)
    assert replica.read("orders.total") == 3

    for _ in range(5):
        replayed = validate_envelope(envelope, registry)
        assert replica.absorb([replayed]) == []
        replica.apply(replayed)
    assert state_hash(replica.state) == before
    assert replica.read("orders.total") == 3
    assert replica.ids_applied() == frozenset({"act-a-1"})


# --- S6: keyless reload refusal ---------------------------------------------------


def test_signed_journal_lines_refuse_reload_without_key(tmp_path):
    d = str(tmp_path / "signed-lines")
    store = JournalStore(d, journal_secret="journal-key")
    store.append(make_action())
    store.append(make_action(action_id="act-a-2", tick=2))
    store.close()  # no rotation: only journal lines exist, no snapshot

    with pytest.raises(ValueError, match="journal_secret"):
        JournalStore(d).load()
    with pytest.raises(ValueError, match="mismatch"):
        JournalStore(d, journal_secret="wrong-key").load()
    root, actions = JournalStore(d, journal_secret="journal-key").load()
    assert root is None and [a.action_id for a in actions] == ["act-a-1", "act-a-2"]
