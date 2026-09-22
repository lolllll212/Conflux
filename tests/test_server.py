import time

import pytest

from conflux import Client, Server
from conflux.crdts import GCounter, LWWRegister, ORMap, ORSet, PNCounter, Stamp, from_dict


def wait_until(fn, timeout=6.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if fn():
                return
        except Exception:
            pass
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def test_crdt_round_trip_serialization():
    root = ORMap(
        {
            "orders": PNCounter({"slot-a:x": (3, 1)}),
            "note": LWWRegister("hi", Stamp(1, "a")),
            "name": LWWRegister("alice", Stamp(2, "b")),
            "pool": ORSet({"t1": "x", "t2": "y"}, frozenset({"t1"})),
        }
    )
    root = root.put("grow", GCounter({"a": 7}))
    restored = from_dict(root.to_dict())
    assert restored == root
    assert restored.read("orders") == 2
    assert restored.read("note") == "hi"
    assert restored.read("pool") == frozenset({"y"})


def test_client_round_trip():
    server = Server("n1", ("127.0.0.1", 0))
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="alice") as client:
            client.counter_inc("orders", 3)
            client.register_set("profile.name", "alice")
            client.set_add("pool", "x", tag="t1")
            assert client.read("orders") == 3
            assert client.read("profile.name") == "alice"
            assert client.read("pool") == ["x"]
    finally:
        server.stop()


def test_two_nodes_gossip_converge():
    alpha = Server("alpha", ("127.0.0.1", 0), gossip_interval=0.2)
    beta = Server("beta", ("127.0.0.1", 0), gossip_interval=0.2)
    alpha.start()
    beta.start()
    try:
        alpha.add_peer(beta.node_address)
        beta.add_peer(alpha.node_address)

        with Client("127.0.0.1", alpha.node_address[1], agent_id="c-a") as ca:
            ca.counter_inc("orders", 5)
            ca.register_set("status", "open")
            with Client("127.0.0.1", beta.node_address[1], agent_id="c-b") as cb:
                cb.counter_inc("orders", 3)
                wait_until(lambda: cb.read("orders") == 8)
                wait_until(lambda: cb.read("status") == "open")
                wait_until(lambda: alpha.hash() == beta.hash())
                assert ca.read("orders") == 8
                assert alpha.read("orders") == 8
                assert beta.read("orders") == 8
    finally:
        alpha.stop()
        beta.stop()


def test_persistence_across_restart(tmp_path):
    data_dir = str(tmp_path)
    server = Server("n1", ("127.0.0.1", 0), data_dir=data_dir, snapshot_threshold=2)
    server.start()
    with Client("127.0.0.1", server.node_address[1], agent_id="writer") as client:
        client.counter_inc("orders", 5)
        client.register_set("name", "bob")
        client.set_add("pool", "x", tag="t1")
        client.set_remove("pool", "t1")
    server.stop()

    assert (tmp_path / "snapshot.json").exists()

    restarted = Server("n1", ("127.0.0.1", 0), data_dir=data_dir, snapshot_threshold=2)
    restarted.start()
    try:
        with Client("127.0.0.1", restarted.node_address[1], agent_id="reader") as client:
            assert client.read("orders") == 5
            assert client.read("name") == "bob"
            assert client.read("pool") == []
            client.counter_inc("orders", 2)
            assert client.read("orders") == 7
    finally:
        restarted.stop()


def test_sync_pulls_journal_to_client():
    server = Server("n1", ("127.0.0.1", 0))
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="writer") as writer:
            writer.register_set("k", "v1")
        with Client("127.0.0.1", server.node_address[1], agent_id="reader") as reader:
            assert reader.replica.read("k") is None
            reader.sync()
            assert reader.replica.read("k") == "v1"
    finally:
        server.stop()


def test_client_batch_submit():
    server = Server("n1", ("127.0.0.1", 0))
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="batcher") as client:
            with client.batch() as b:
                b.counter_inc("batch_orders", 10)
                b.register_set("batch_status", "in_progress")
                b.set_add("batch_items", "item1", tag="it-1")
            assert client.read("batch_orders") == 10
            assert client.read("batch_status") == "in_progress"
            assert client.read("batch_items") == ["item1"]
    finally:
        server.stop()