import pytest

from conflux import (
    Action,
    AgentRegistry,
    Client,
    Server,
    ValidationError,
    converge_all,
    sign_action,
    validate_envelope,
)


def raw_action(key="orders.draft", op="counter_inc", params=None, agent="a", **overrides):
    fields = {"action_id": "act-a-1", "agent": agent, "tick": 1, "op": op, "key": key,
              "params": params or {}}
    fields.update(overrides)
    return Action(**fields)


def test_namespace_policy_prefix_matching():
    registry = AgentRegistry()
    registry.register("orders_svc", "s1", namespaces=("orders.", "shipping."))
    registry.register("global", "s2")
    registry.register("exact", "s3", namespaces=("beta",))

    assert registry.namespace_allows("orders_svc", "orders.123")
    assert registry.namespace_allows("orders_svc", "shipping.label")
    assert not registry.namespace_allows("orders_svc", "inventory.stock")
    assert not registry.namespace_allows("orders_svc", "xorders.1")
    assert registry.namespace_allows("global", "anything.here")
    assert registry.namespace_allows("exact", "beta")
    assert registry.namespace_allows("exact", "beta.sub")  # prefix semantics
    assert not registry.namespace_allows("exact", "gamma")


def test_validate_envelope_enforces_namespace():
    registry = AgentRegistry()
    registry.register("a", "secret-a", namespaces=("orders.",))
    allowed = sign_action(raw_action(key="orders.1"), "secret-a")
    assert validate_envelope(allowed, registry).key == "orders.1"
    denied = sign_action(raw_action(key="kill.all"), "secret-a")
    with pytest.raises(ValidationError):
        validate_envelope(denied, registry)


def test_server_rejects_out_of_namespace_submit():
    registry = AgentRegistry()
    registry.register("driver", "s3cr3t", namespaces=("orders.",), ops={"counter_inc"})
    server = Server("n1", ("127.0.0.1", 0), registry=registry)
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="driver", secret="s3cr3t") as c:
            c.counter_inc("orders.count", 1)
            assert c.read("orders.count") == 1
            with pytest.raises(ConnectionError):
                c.counter_inc("billing.total", 1)
            assert c.read("billing.total") is None
    finally:
        server.stop()


def test_namespace_policy_is_replicated_policy_only():
    registry = AgentRegistry()
    registry.register("a", "sa", namespaces=("a.",))
    registry.register("b", "sb", namespaces=("b.",))
    a = Action(f"act-a-1", "a", 1, "counter_inc", "a.counter", {"by": 1})
    b = Action(f"act-b-1", "b", 1, "counter_inc", "b.counter", {"by": 1})
    replicas = []
    for action in (a, b):
        envelope = sign_action(action, registry.secret(action.agent))
        validate_envelope(envelope, registry)  # both in their own scope -> valid
        replicas.append(action)
    from conflux import Replica

    r1, r2 = Replica("x"), Replica("y")
    r1.absorb([replicas[0]])
    r2.absorb([replicas[0]])
    r1.absorb([replicas[1]])
    r2.absorb([replicas[1]])
    assert r1.read("a.counter") == 1
    assert r2.read("b.counter") == 1