import threading

import pytest

from conflux import (
    ACTION_VERSION,
    Action,
    AgentRegistry,
    Client,
    Server,
    ValidationError,
    generate_ed25519_keypair,
    replay,
    sign_action,
    state_hash,
    validate_action,
    validate_envelope,
    verify_action,
)


def raw_action(op="counter_inc", params=None, **overrides):
    fields = {"action_id": "act-a-1", "agent": "a", "tick": 1, "op": op, "key": "k",
              "params": params or {}}
    fields.update(overrides)
    return Action(**fields)


def test_action_schema_validation():
    validate_action(raw_action(params={"by": 3}))
    with pytest.raises(ValidationError):
        validate_action(raw_action(tick=0))
    with pytest.raises(ValidationError):
        validate_action(raw_action(op="nope"))
    with pytest.raises(ValidationError):
        validate_action(raw_action("counter_inc", params={"by": 0}))
    with pytest.raises(ValidationError):
        validate_action(raw_action("counter_inc", params={"by": "3"}))
    with pytest.raises(ValidationError):
        validate_action(raw_action("lww_set", params={"value": object()}))
    with pytest.raises(ValidationError):
        validate_action(raw_action("orset_add", params={"value": "x"}))
    validate_action(raw_action("orset_add", params={"tag": "t1", "value": "x"}))
    with pytest.raises(ValidationError):
        validate_action(raw_action("orset_remove", params={"tag": ""}))
    with pytest.raises(ValidationError):
        validate_action(raw_action("lww_set", params={"value": 1, "weight": 1.5}))
    with pytest.raises(ValidationError):
        validate_action(raw_action(version=2))


def test_signature_round_trip_and_tamper_detection():
    action = raw_action(params={"by": 2})
    envelope = sign_action(action, "secret-key")
    assert verify_action(envelope, "secret-key")
    assert not verify_action(envelope, "wrong-key")

    tampered = dict(envelope)
    tampered["action"] = dict(tampered["action"])
    tampered["action"]["params"] = {"by": 999}
    assert not verify_action(tampered, "secret-key")

    registry = AgentRegistry()
    registry.register("a", "secret-key")
    forged = dict(envelope)
    forged["signed_by"] = "mallory"
    with pytest.raises(ValidationError):
        validate_envelope(forged, registry)


def test_registry_authority_and_quota():
    registry = AgentRegistry()
    registry.register("a", "secret-a", ops={"counter_inc"}, quota=2)
    registry.register("b", "secret-b")

    assert registry.authorize("a", "counter_inc")
    assert not registry.authorize("a", "lww_set")
    assert registry.authorize("b", "orset_add")
    assert not registry.authorize("unknown", "counter_inc")

    assert registry.consume("a")
    assert registry.consume("a")
    assert not registry.consume("a")


def test_envelope_validation_integration():
    registry = AgentRegistry()
    registry.register("a", "secret-a", ops={"counter_inc", "lww_set"}, quota=5)
    registry.register("b", "secret-b", ops={"lww_set"})

    ok = sign_action(raw_action(params={"by": 1}), "secret-a")
    assert validate_envelope(ok, registry).op == "counter_inc"

    forbidden = sign_action(raw_action("orset_add", params={"tag": "t", "value": "x"}), "secret-a")
    with pytest.raises(ValidationError):
        validate_envelope(forbidden, registry)

    unknown = sign_action(raw_action(), "no-such-secret")
    with pytest.raises(ValidationError):
        validate_envelope(unknown, registry)


def test_server_accepts_signed_submits_and_rejects_unknown_agents():
    registry = AgentRegistry()
    registry.register("driver", "s3cr3t", ops={"counter_inc", "lww_set"})
    server = Server("n1", ("127.0.0.1", 0), registry=registry)
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="s3cr3t") as c:
            c.counter_inc("orders", 4)
            c.register_set("status", "open")
            assert c.read("orders") == 4
            assert c.read("status") == "open"

        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="wrong") as c:
            with pytest.raises(ConnectionError):
                c.counter_inc("orders", 1)

        with Client("127.0.0.1", server.node_address[1], agent_id="stranger", secret="s3cr3t") as c:
            with pytest.raises(ConnectionError):
                c.register_set("hack", "x")
    finally:
        server.stop()


def test_signed_actions_replay_deterministically():
    registry = AgentRegistry()
    registry.register("a", "sa")
    registry.register("b", "sb")
    actions = [
        raw_action("counter_inc", {"by": 1}, action_id="a1", agent="a", tick=1, key="orders"),
        raw_action("counter_inc", {"by": 2}, action_id="b1", agent="b", tick=1, key="orders"),
        raw_action("lww_set", {"value": "open"}, action_id="a2", agent="a", tick=2, key="status"),
    ]
    verified = [validate_envelope(sign_action(a, registry.secret(a.agent)), registry) for a in actions]
    assert state_hash(replay(verified).state) == state_hash(replay(list(reversed(verified))).state)
    rebuilt = replay(verified)
    assert rebuilt.read("orders") == 3
    assert rebuilt.read("status") == "open"


def test_ed25519_round_trip():
    pytest.importorskip("cryptography")
    private, public = generate_ed25519_keypair()
    action = raw_action(params={"by": 2})
    envelope = sign_action(action, private, "ed25519")
    assert verify_action(envelope, public, "ed25519")
    assert not verify_action(envelope, public, "hmac-sha256")
    _, other_public = generate_ed25519_keypair()
    assert not verify_action(envelope, other_public, "ed25519")

    tampered = dict(envelope)
    tampered["action"] = dict(tampered["action"])
    tampered["action"]["params"] = {"by": 999}
    assert not verify_action(tampered, public, "ed25519")


def test_rejected_submits_do_not_mutate_state():
    registry = AgentRegistry()
    registry.register("driver", "s3cr3t", ops={"counter_inc"}, namespaces=("orders.",))
    server = Server("n1", ("127.0.0.1", 0), registry=registry)
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="s3cr3t") as c:
            c.counter_inc("orders.total", 1)
            assert c.read("orders.total") == 1
        before = server.hash()

        # bad signature -> refused, state untouched
        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="wrong") as c:
            with pytest.raises(ConnectionError):
                c.counter_inc("orders.total", 100)
        # unauthorized op -> refused, state untouched
        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="s3cr3t") as c:
            with pytest.raises(ConnectionError):
                c.register_set("orders.status", "x")
        # out of namespace -> refused, state untouched
        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="s3cr3t") as c:
            with pytest.raises(ConnectionError):
                c.counter_inc("billing.total", 100)

        assert server.hash() == before
        assert server.read("orders.total") == 1
    finally:
        server.stop()


def test_quota_exhaustion_over_wire():
    registry = AgentRegistry()
    registry.register("driver", "s3cr3t", ops={"counter_inc"}, quota=2)
    server = Server("n1", ("127.0.0.1", 0), registry=registry)
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="s3cr3t") as c:
            c.counter_inc("orders.total", 1)
            c.counter_inc("orders.total", 1)
            assert registry.quota_left("driver") == 0
            with pytest.raises(ConnectionError):
                c.counter_inc("orders.total", 1)
        assert server.read("orders.total") == 2
        assert registry.quota_left("driver") == 0
    finally:
        server.stop()


def test_concurrent_quota_is_not_overdrawn():
    registry = AgentRegistry()
    registry.register("a", "sa", quota=4)
    results = []

    def hammer():
        results.append(registry.consume("a"))

    threads = [threading.Thread(target=hammer) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 4
    assert registry.quota_left("a") == 0


def test_ed25519_registry_and_server_reject_wrong_key():
    pytest.importorskip("cryptography")
    private, public = generate_ed25519_keypair()
    registry = AgentRegistry()
    registry.register("agent", public_key=public, ops={"counter_inc"})

    envelope = sign_action(raw_action(params={"by": 4}, agent="agent"), private, "ed25519")
    assert validate_envelope(envelope, registry).op == "counter_inc"

    server = Server("n1", ("127.0.0.1", 0), registry=registry)
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="agent",
                    secret=private, signature_alg="ed25519") as c:
            c.counter_inc("orders", 4)
            assert c.read("orders") == 4

        wrong = AgentRegistry()
        wrong.register("agent", public_key=generate_ed25519_keypair()[1], ops={"counter_inc"})
        other_server = Server("n2", ("127.0.0.1", 0), registry=wrong)
        other_server.start()
        try:
            with Client("127.0.0.1", other_server.node_address[1], agent_id="agent",
                        secret=private, signature_alg="ed25519") as c:
                with pytest.raises(ConnectionError):
                    c.counter_inc("orders", 1)
        finally:
            other_server.stop()
    finally:
        server.stop()