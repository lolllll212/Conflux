import json
import os
import socket
import threading
import time

import pytest

from conflux import AgentRegistry, Client, Server
from conflux.actions import Action
from conflux.client import Client
from conflux.server import Server
from conflux.state import Replica
from conflux.storage import JournalStore
from conflux.validate import ValidationError

HMAC_SECRET = "node-secret"


def _torn_write(path):
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"action": {"torn": true')


def _wait_converged(servers, key, value, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(s.read(key) == value for s in servers):
            return True
        time.sleep(0.02)
    return False


def test_signed_journal_refuses_unsigned_line(tmp_path):
    d = str(tmp_path / "unsigned-line")
    store = JournalStore(d, journal_secret=HMAC_SECRET)
    r = Replica("gen")
    r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[-1])
    store.close()

    # an attacker appends a well-formed action WITHOUT a signature
    with open(os.path.join(d, "journal.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"action": {"action_id": "forged", "agent": "gen", "tick": 99, '
                '"op": "counter_inc", "key": "orders", "params": {"by": 1000}, '
                '"version": 1}, "seq": 2}\n')

    recovered = JournalStore(d, journal_secret=HMAC_SECRET)
    with pytest.raises(ValueError):
        recovered.load()


def test_signed_journal_refuses_unsigned_snapshot(tmp_path):
    d = str(tmp_path / "unsigned-snap")
    store = JournalStore(d, snapshot_threshold=1, journal_secret=HMAC_SECRET)
    r = Replica("gen")
    r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[-1])
    store.rotate(r.state._map)
    store.close()

    recovered = JournalStore(d, journal_secret=HMAC_SECRET)
    assert recovered.load()[0].read("orders") == 1
    # strip the signature from snapshot.json -> must now be refused
    with open(os.path.join(d, "snapshot.json"), encoding="utf-8") as f:
        payload = __import__("json").load(f)
    payload.pop("sig", None)
    with open(os.path.join(d, "snapshot.json"), "w", encoding="utf-8") as f:
        __import__("json").dump(payload, f)
    refused = JournalStore(d, journal_secret=HMAC_SECRET)
    with pytest.raises(ValueError):
        refused.load()


def test_snapshot_files_survive_rotation(tmp_path):
    d = str(tmp_path / "fsync")
    store = JournalStore(d, snapshot_threshold=1)
    r = Replica("gen")
    for i in range(1, 4):
        r._emit("counter_inc", "orders", {"by": 1})
        store.append(r.journal[-1])
        store.rotate(r.state._map)
    store.close()
    assert os.path.exists(os.path.join(d, "snapshot.json"))
    root, actions = JournalStore(d).load()
    assert root.read("orders") == 3
    assert actions == []


def test_server_will_not_dial_itself():
    server = Server("solo", ("127.0.0.1", 0), peers=[("127.0.0.1", 12345)]).start()
    try:
        # the guard must ignore its own listen address at the dial site
        before = server.metrics()["outbound"]
        server._ensure_peer(server.node_address)
        assert server.metrics()["outbound"] == before
    finally:
        server.stop()


def test_client_surfaces_timeout_as_connection_error():
    server = Server("t", ("127.0.0.1", 0)).start()
    try:
        client = Client("127.0.0.1", server.node_address[1], agent_id="tester")
        client._recv = lambda: (_ for _ in ()).throw(socket.timeout("forced"))
        with pytest.raises(ConnectionError, match="timed out"):
            client._recv_safe()
        client.close()
    finally:
        server.stop()


def test_peer_journal_validates_actions():
    server = Server("v", ("127.0.0.1", 0), registry=None).start()
    try:
        bad = {"actions": [{"action_id": "x/../broken", "agent": "a", "tick": 1,
                            "op": "counter_inc", "key": "k", "params": {"by": 1},
                            "version": 1}]}
        with pytest.raises(ValueError):
            server._handle_message({"type": "journal", "actions": bad["actions"]},
                                   send=lambda m: None, source=None)
    finally:
        server.stop()


def test_node_recovers_after_kill9_and_torn_tail(tmp_path):
    secret = "s"
    registry = AgentRegistry()
    registry.register("driver", secret, ops={"counter_inc"})
    d = str(tmp_path / "crew-a")
    a = Server("alpha", ("127.0.0.1", 0), registry=registry, data_dir=d,
               snapshot_threshold=2, gossip_interval=0.05).start()
    oracle = Server("oracle", ("127.0.0.1", 0), registry=registry,
                    gossip_interval=0.05).start()
    a.add_peer(oracle.node_address)
    oracle.add_peer(a.node_address)
    try:
        with Client("127.0.0.1", a.node_address[1], agent_id="driver",
                    secret=secret) as c:
            for _ in range(5):
                c.counter_inc("orders.total", 1)
        assert _wait_converged([a, oracle], "orders.total", 5)
        pre = oracle.hash()

        # hard kill: clean stop (fd close), then tear the journal tail and
        # remove the alias so recovery must use the generation files
        a.stop()
        jar = os.path.join(d, "journal.jsonl")
        _torn_write(jar)
        os.unlink(os.path.join(d, "snapshot.json"))

        # restart on the same data dir: cumulative snapshot + journal tail
        a2 = Server("alpha", ("127.0.0.1", 0), registry=registry, data_dir=d,
                    snapshot_threshold=2, gossip_interval=0.05).start()
        oracle.add_peer(a2.node_address)
        assert a2.read("orders.total") == 5
        assert _wait_converged([a2, oracle], "orders.total", 5)
        assert a2.hash() == pre
    finally:
        a.stop()
        oracle.stop()


def test_node_recovers_from_mid_rotation_crash(tmp_path):
    """Crash after the alias rename but with a torn NEWEST generation file and
    a leftover .tmp: recovery must fall back to the older cumulative
    generation (or the full journal), never lose state."""
    secret = "s"
    registry = AgentRegistry()
    registry.register("writer", secret, ops={"counter_inc"})
    d = str(tmp_path / "crew-b")
    s = Server("beta", ("127.0.0.1", 0), registry=registry, data_dir=d,
               snapshot_threshold=2, gossip_interval=0.05).start()
    try:
        with Client("127.0.0.1", s.node_address[1], agent_id="writer",
                    secret=secret) as c:
            for i in range(6):
                c.counter_inc("pings", 1)
        expect = s.read("pings")
        s.stop()

        gens = sorted(p for p in os.listdir(d) if p.startswith("snapshot-"))
        newest = os.path.join(d, gens[-1])
        with open(newest, "w", encoding="utf-8") as f:
            f.write('{"seq": 99, "root": {"pings": {"p": 1}')  # torn mid-object
        tmp = newest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write('{"partial": true')
        _torn_write(os.path.join(d, "journal.jsonl"))

        s2 = Server("beta", ("127.0.0.1", 0), registry=registry, data_dir=d,
                    snapshot_threshold=2, gossip_interval=0.05).start()
        assert s2.read("pings") == expect
    finally:
        s.stop()


def test_slow_peer_does_not_block_gossip():
    """A peer that accepts but never reads (socket buffers fill, sends stall)
    must not stall convergence between healthy nodes: dials run off the gossip
    loop and each link has its own bounded sender queue."""
    blackhole = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blackhole.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blackhole.bind(("127.0.0.1", 0))
    blackhole.listen(8)

    def hold():
        while True:
            try:
                conn, _ = blackhole.accept()
            except OSError:
                return
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512)

    threading.Thread(target=hold, daemon=True).start()

    secret = "s"
    registry = AgentRegistry()
    registry.register("driver", secret, ops={"counter_inc"})
    stall = ("127.0.0.1", blackhole.getsockname()[1])
    a = Server("alpha", ("127.0.0.1", 0), peers=[stall], registry=registry,
               gossip_interval=0.05, stale_after=60).start()
    b = Server("beta", ("127.0.0.1", 0), registry=registry,
               gossip_interval=0.05).start()
    a.add_peer(b.node_address)
    b.add_peer(a.node_address)
    try:
        with Client("127.0.0.1", a.node_address[1], agent_id="driver",
                    secret=secret) as c:
            c.counter_inc("orders.total", 1)
        assert _wait_converged([a, b], "orders.total", 1, timeout=3.0), (
            "healthy peers must converge despite a stalled link")
    finally:
        a.stop()
        b.stop()
        blackhole.close()


def test_reconnect_duplicates_do_not_double_apply():
    secret = "s"
    secret2 = "s2"
    registry = AgentRegistry()
    registry.register("driver", secret, ops={"counter_inc"})
    registry.register("driver2", secret2, ops={"counter_inc"})
    a = Server("alpha", ("127.0.0.1", 0), registry=registry,
               gossip_interval=0.05).start()
    b = Server("beta", ("127.0.0.1", 0), registry=registry,
               gossip_interval=0.05).start()
    a.add_peer(b.node_address)
    b.add_peer(a.node_address)
    try:
        with Client("127.0.0.1", a.node_address[1], agent_id="driver",
                    secret=secret) as c:
            for _ in range(3):
                c.counter_inc("orders.total", 1)
        assert _wait_converged([a, b], "orders.total", 3)

        # partition; a distinct agent process writes while b is down
        b.stop()
        with Client("127.0.0.1", a.node_address[1], agent_id="driver2",
                    secret=secret2) as c:
            c.counter_inc("orders.total", 1)
        b2 = Server("beta", ("127.0.0.1", 0), registry=registry,
                    gossip_interval=0.05).start()
        a.add_peer(b2.node_address)
        b2.add_peer(a.node_address)
        assert _wait_converged([a, b2], "orders.total", 4), (
            "duplicate journal frames on reconnect must not inflate the counter")
    finally:
        a.stop()
        b.stop()