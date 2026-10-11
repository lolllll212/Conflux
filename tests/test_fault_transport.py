"""Transport faults: frames that must not kill a node, and a journal that must stay joinable.

Every test here starts from something a real peer or agent can do over a socket -
send garbage, dribble bytes, hang up mid-frame, join a node with a big journal -
and asserts the node degrades in a *counted, recoverable* way instead of printing a
traceback from a thread nobody is watching.
"""

import json
import socket
import struct
import threading
import time

import pytest

from conflux import Client, Server
from conflux.net import FrameTooLarge, ProtocolError, dumps_frame, recv_msg, send_msg
from conflux.state import Replica
from conflux.server import Server as _Server

_NODE = iter(range(1_000_000))


def frame(obj_or_bytes):
    payload = obj_or_bytes if isinstance(obj_or_bytes, bytes) else json.dumps(obj_or_bytes).encode()
    return struct.pack(">I", len(payload)) + payload


class ScriptedSocket:
    """A socket that replays a fixed set of read sizes, and records what was written."""

    def __init__(self, data, chunk=64 * 1024):
        self._view = memoryview(data)
        self.chunk = chunk
        self.written = bytearray()
        self.reads = 0

    def recv_into(self, buffer, size=None):
        self.reads += 1
        size = size if size else len(buffer)
        take = min(size, self.chunk, len(self._view))
        if take <= 0:
            return 0
        buffer[:take] = self._view[:take]
        self._view = self._view[take:]
        return take

    def sendall(self, data):
        self.written += data


# --- frame decoding ----------------------------------------------------------


def test_every_kind_of_bad_frame_is_one_typed_error():
    """Six ways a frame can be wrong, one exception class that says "drop the peer"."""
    too_deep = b"[" * 5000 + b"]" * 5000
    cases = {
        "zero length": frame(b""),
        "over the cap": struct.pack(">I", 10_000) + b"x" * 10,
        "not utf-8": struct.pack(">I", 4) + b"\xff\xfe\xfd\xfc",
        "not json": frame(b"this is not json at all"),
        "json array, not object": frame(b"[1, 2, 3]"),
        "nests too deeply": frame(too_deep),
    }
    for label, raw in cases.items():
        sock = ScriptedSocket(raw, chunk=7)  # dribbled reads, like a slow peer
        with pytest.raises(ProtocolError):
            recv_msg(sock, max_frame=1024)


def test_a_clean_close_during_the_header_is_eof_not_an_error():
    assert recv_msg(ScriptedSocket(b"")) is None


def test_a_frame_that_ends_mid_payload_is_reported_as_truncated():
    """A peer that hangs up half-way used to look like a closed socket; it is a fault."""
    raw = struct.pack(">I", 64) + b"x" * 10
    with pytest.raises(ProtocolError, match="truncated"):
        recv_msg(ScriptedSocket(raw))


def test_reads_use_recv_into_so_a_big_frame_costs_its_size():
    """``buf += chunk`` over a 64 MB frame is quadratic and keeps every copy alive."""
    body = json.dumps({"type": "ping"}).encode()
    sock = ScriptedSocket(struct.pack(">I", len(body)) + body, chunk=4)
    assert recv_msg(sock) == {"type": "ping"}
    assert sock.reads >= 3  # it read in pieces, through recv_into, without reallocating


def test_send_refuses_a_frame_the_peer_is_allowed_to_reject():
    """Writing a frame the receiver will drop means a link that never converges."""
    sock = ScriptedSocket(b"")
    with pytest.raises(FrameTooLarge, match="over the"):
        send_msg(sock, {"type": "journal", "actions": ["x" * 200]}, max_frame=128)
    assert bytes(sock.written) == b""  # nothing half-written on the wire


def test_dumps_frame_rejects_an_unserialisable_message():
    with pytest.raises(ProtocolError, match="not serialisable"):
        dumps_frame({"bad": object()})


# --- the live node -----------------------------------------------------------


def wait_until(predicate, timeout=8.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_a_hostile_frame_is_counted_and_leaves_nothing_behind():
    """Deep JSON used to die inside json.loads and print a traceback from a thread.

    Now: the peer is dropped, the reason is counted, and neither a link entry nor a
    sender thread outlives the connection - which matters because a leaked link with
    no address was never reaped, so each attempt cost a thread for the life of the
    process.
    """
    server = Server_fixture(gossip_interval=0.05, stale_after=0.3)
    port = server.node_address[1]
    threads_before = threading.active_count()

    # 1. register a link, then feed it a frame that cannot be decoded
    sock = socket.create_connection(("127.0.0.1", port), timeout=3)
    sock.sendall(frame({"role": "peer", "node_id": "attacker", "members": []}))
    time.sleep(0.15)
    assert len(server._links) == 1
    sock.sendall(frame(b"[" * 60_000 + b"]" * 60_000))
    time.sleep(0.2)  # close after the frame is read: an RST mid-frame is a hang-up, not a fault
    sock.close()

    assert wait_until(lambda: server.metrics()["malformed_frames"] >= 1), server.metrics()
    assert wait_until(lambda: len(server._links) == 0), "address-less link was never reaped"
    assert wait_until(lambda: threading.active_count() <= threads_before), (
        threading.active_count())
    # and the node is still a node
    with Client("127.0.0.1", port, agent_id="probe") as client:
        client.counter_inc("still-working", 2)
        assert client.read("still-working") == 2
    server.stop()


def test_a_malformed_submit_gets_an_answer_not_a_disconnection():
    """An agent posting a half-built action is a client bug; hanging up helps nobody."""
    server = Server_fixture()
    sock = socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=5)
    sock.sendall(frame({"type": "hello", "role": "agent", "node_id": "clumsy-agent"}))
    assert recv_msg(sock)["type"] == "welcome"  # an agent connection is a handshake first
    sock.sendall(frame({"type": "submit", "actions": [{"nonsense": True}]}))
    reply = recv_msg(sock)
    assert reply and reply["type"] == "error" and "malformed action" in reply["reason"]
    # the same connection still serves a valid submit afterwards
    # built by a real Replica so the action is schema-valid by construction
    action = Replica("agent-x")._emit("counter_inc", "orders", {"by": 3})
    sock.sendall(frame({"type": "submit", "actions": [action.export()]}))
    ack = recv_msg(sock)
    assert ack["type"] == "ack" and ack["accepted"] == 1
    assert server.read("orders") == 3
    sock.close()
    server.stop()


def test_a_read_that_cannot_be_answered_is_answered_anyway():
    """A `read` with no key (or a key of the wrong type) is a client bug.

    Before this the handler did msg["key"] and the KeyError fell out of the receive
    loop, which closed a working connection and reported nothing.
    """
    server = Server_fixture()
    sock = socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=5)
    sock.sendall(frame({"type": "hello", "role": "agent", "node_id": "curious"}))
    assert recv_msg(sock)["type"] == "welcome"
    for bad in ({"type": "read"}, {"type": "read", "key": None}, {"type": "read", "key": [1]}):
        sock.sendall(frame(bad))
        reply = recv_msg(sock)
        assert reply["type"] == "error" and "key" in reply["reason"], reply
    # the same connection still answers a well-formed read
    sock.sendall(frame({"type": "read", "key": "nothing-here"}))
    assert recv_msg(sock) == {"type": "read_resp", "key": "nothing-here", "value": None}
    sock.close()
    server.stop()


def test_one_frame_cannot_ask_for_unbounded_work():
    """A valid JSON array of a million actions is affordable to decode and not to absorb.

    The byte cap and the action cap are different guards: this one is what stops a
    peer inside the trust boundary - or a misconfigured one with a huge batch size -
    from turning a gossip round into a minute of CPU on every node it can reach.
    """
    server = Server_fixture(gossip_interval=0.05, stale_after=0.3, max_actions_per_frame=3)
    sock = socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=3)
    sock.sendall(frame({"role": "peer", "node_id": "greedy", "members": []}))
    time.sleep(0.15)
    assert len(server._links) == 1
    sock.sendall(frame({"type": "journal", "actions": [{"action_id": f"a{i}", "agent": "x",
                                                        "tick": i, "op": "counter_inc",
                                                        "key": "k", "params": {}}
                                                       for i in range(4)]}))
    assert wait_until(lambda: server.metrics()["oversize_frames"] >= 1), server.metrics()
    assert wait_until(lambda: len(server._links) == 0), "the greedy link was kept"
    with Client("127.0.0.1", server.node_address[1], agent_id="probe-2") as client:
        client.counter_inc("after", 1)
        assert client.read("after") == 1
    sock.close()
    server.stop()


def test_a_submit_over_the_cap_is_answered_and_the_agent_keeps_working():
    """An agent batching 100k actions is a client bug, so the reply names the remedy."""
    server = Server_fixture(max_actions_per_frame=5)
    sock = socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=5)
    sock.sendall(frame({"type": "hello", "role": "agent", "node_id": "overreacher"}))
    assert recv_msg(sock)["type"] == "welcome"
    big = [{"action_id": f"b{i}", "agent": "overreacher", "tick": i + 1, "op": "counter_inc",
            "key": "k", "params": {"by": 1}} for i in range(6)]
    sock.sendall(frame({"type": "submit", "actions": big}))
    reply = recv_msg(sock)
    assert reply["type"] == "error" and "split it" in reply["reason"], reply
    assert server.read("k") is None, "a refused frame must not half-apply"
    sock.sendall(frame({"type": "submit", "actions": big[:5]}))
    ack = recv_msg(sock)
    assert ack["type"] == "ack" and ack["accepted"] == 5
    assert server.read("k") == 5
    sock.close()
    server.stop()


def test_actions_that_is_not_a_list_is_a_refused_frame():
    """`{"type":"journal","actions": 5}` used to be a TypeError handled as a hang-up."""
    server = Server_fixture(gossip_interval=0.05, stale_after=0.3)
    sock = socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=3)
    sock.sendall(frame({"role": "peer", "node_id": "wrong-shape", "members": []}))
    time.sleep(0.15)
    sock.sendall(frame({"type": "journal", "actions": 5}))
    time.sleep(0.2)  # let the frame be read before hanging up
    sock.close()
    assert wait_until(lambda: server.metrics()["malformed_frames"] >= 1), server.metrics()
    assert wait_until(lambda: len(server._links) == 0)
    server.stop()


def test_gossip_round_failures_are_survived_not_fatal():
    """One poisoned link entry must not end this node's ability to gossip, ever."""
    server = Server_fixture(gossip_interval=0.05)
    real = server._gossip_round
    state = {"calls": 0}

    def flaky():
        state["calls"] += 1
        if state["calls"] == 1:
            raise TypeError("simulated bug in one round")
        return real()

    server._gossip_round = flaky
    assert wait_until(lambda: server.metrics()["gossip_errors"] >= 1)
    tick_after_error = server._gossip_tick
    server._gossip_round = real
    assert wait_until(lambda: server._gossip_tick > tick_after_error), "the loop died"
    server.stop()


def test_inbound_connections_are_bounded():
    """Each accepted link costs a thread plus receive buffer: a limit has to exist."""
    server = Server_fixture(max_connections=2)
    open_socks = [socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=3)
                  for _ in range(6)]
    assert wait_until(lambda: server.metrics()["rejected_connections"] >= 1), server.metrics()
    for sock in open_socks:
        sock.close()
    server.stop()


def test_a_node_with_a_long_journal_is_still_joinable():
    """The bug this whole drill was for: one frame per sync, over the peer's cap.

    Before chunking, a new peer dialled a busy node, was refused, redialled, and was
    refused again - forever, with the journal-growing node looking perfectly healthy.
    """
    busy = Server_fixture(gossip_interval=0.05, max_frame=8192, frame_batch_bytes=4096)
    with Client("127.0.0.1", busy.node_address[1], agent_id="writer") as client:
        for index in range(600):
            client.counter_inc(f"orders:{index}", 1)
    time.sleep(0.2)
    assert len(busy.replica.journal) == 600

    joiner = Server_fixture(node_id="late-joiner", peers=[busy.node_address],
                            gossip_interval=0.05, max_frame=8192, frame_batch_bytes=4096)
    try:
        assert wait_until(lambda: joiner.replica.state.view().get("orders:599") is not None), \
            "the new peer never caught up"
    finally:
        joiner.stop()
        busy.stop()
    assert busy.hash() == joiner.hash()
    assert busy.metrics()["oversize_frames"] == 0


def test_the_gossip_cursor_replaces_a_per_peer_set_of_every_action():
    """Memory per link must not grow with the journal, and push must stay O(new)."""
    busy = Server_fixture(gossip_interval=0.05, frame_batch_bytes=8192)
    joiner = Server_fixture(peers=[busy.node_address], gossip_interval=0.05, stale_after=30)
    assert wait_until(lambda: next(iter(busy._links), None) is not None), "the join never linked"
    key = next(iter(busy._links))
    with Client("127.0.0.1", busy.node_address[1], agent_id="w") as client:
        # 1200 actions is ~110 KB: far over the 8 KiB chunk budget, so the push has
        # to be chunked and the cursor has to walk 1200 times without growing sets
        for index in range(1200):
            client.counter_inc(f"orders:{index}", 1)
    assert wait_until(lambda: wait_until_idle(busy, joiner), timeout=20)
    info = busy._links[key]
    assert info["cursor"] == 1200 and info["acked"] == 1200
    assert "sent" not in info and "queued" not in info  # the unbounded sets are gone
    start = time.perf_counter()
    for _ in range(20):
        busy._push_to_peers()
    assert (time.perf_counter() - start) / 20 < 0.01  # nothing pending: a no-op
    joiner.stop()
    busy.stop()


def wait_until_idle(busy, joiner):
    busy._push_to_peers()
    time.sleep(0.2)
    return busy.hash() == joiner.hash()


def Server_fixture(node_id=None, **kwargs):
    """A started Server, so each test can stop it without repeating the dance."""
    server = _Server(node_id or f"node-{next(_NODE)}", ("127.0.0.1", 0), **kwargs)
    return server.start()


# --- prometheus --------------------------------------------------------------


def test_converged_idle_nodes_stop_churning_their_links():
    """Two caught-up peers used to reap each other as idle and re-dial, forever.

    Every cycle cost a thread, a registration, a send and a warning line - on a
    twelve-node mesh that is a permanent background storm whose only output is log
    noise saying the cluster is healthy. The link now carries its state hash, so it
    is never idle when it is alive, and is still reaped when it is not.
    """
    a = Server_fixture(gossip_interval=0.1, stale_after=0.5)
    b = Server_fixture(gossip_interval=0.1, stale_after=0.5, peers=[a.node_address])
    assert wait_until(lambda: len(a._links) == 1 and len(b._links) == 1)
    key = next(iter(a._links))
    time.sleep(2.5)                      # five stale windows, nothing to deliver
    assert list(a._links) == [key], "the live link was reaped despite its heartbeats"
    assert a.metrics()["link_drops"] == 0 == b.metrics()["link_drops"]
    assert a.hash() == b.hash()
    b.stop()
    a.stop()


def test_a_link_that_missed_gossip_heals_without_an_operator():
    """The heartbeat is also anti-entropy: a lost frame does not mean lost forever.

    Here the delivery is suppressed by hand - the same hole backpressure under a
    full queue can open - so the only thing that can notice is the hash a peer
    carries. Divergence must repair itself, quietly, and say so once.
    """
    a = Server_fixture(gossip_interval=0.1, stale_after=30)
    b = Server_fixture(gossip_interval=0.1, stale_after=30, peers=[a.node_address])
    assert wait_until(lambda: len(a._links) == 1 and a.hash() == b.hash())
    with Client("127.0.0.1", a.node_address[1], agent_id="late") as client:
        for index in range(25):
            client.counter_inc(f"missed:{index}", 1)
    key = next(iter(a._links))
    a._links[key]["cursor"] = len(a.replica.journal)      # "I already sent all of this"
    a._links[key]["acked"] = len(a.replica.journal)
    assert a.hash() != b.hash()
    assert wait_until(lambda: b.hash() == a.hash(), timeout=15), (a.metrics(), b.metrics())
    resyncs = a.metrics()["resyncs"] + b.metrics()["resyncs"]
    assert resyncs >= 1, "the repair happened without ever being noticed, which is the bug"
    assert b.read("missed:1") == 1
    time.sleep(0.4)                      # and it settles instead of stampeding
    settled = a.metrics()["resyncs"] + b.metrics()["resyncs"]
    assert settled - resyncs <= 2, (settled, resyncs)
    b.stop()
    a.stop()


def test_liveness_still_means_what_it_said_it_meant():
    """A heartbeat keeps a live link up; it cannot keep a dead one pretending.

    Two ways a peer goes away matter differently: a clean shutdown closes the socket
    and the reader notices immediately, while a hung or partitioned peer leaves the
    socket open and is only caught by the idle timer - which is exactly the counter
    an operator alerts on.
    """
    a = Server_fixture(gossip_interval=0.1, stale_after=0.4)
    b = Server_fixture(gossip_interval=0.1, stale_after=0.4, peers=[a.node_address])
    assert wait_until(lambda: len(a._links) == 1 and len(b._links) == 1)
    b.stop()
    assert wait_until(lambda: len(a._links) == 0, timeout=6), a.metrics()

    # now a peer that says hello and then goes silent: link registered, never traffic
    sock = socket.create_connection(("127.0.0.1", a.node_address[1]), timeout=3)
    sock.sendall(frame({"role": "peer", "node_id": "muted", "address": ["127.0.0.1", 9],
                        "members": []}))
    assert wait_until(lambda: "muted" in {i.get("peer_id") for i in a._links.values()})
    assert a.metrics()["link_drops"] == 0
    assert wait_until(lambda: len(a._links) == 0, timeout=6), a.metrics()
    assert a.metrics()["link_drops"] >= 1, "an idle link was reaped without being counted"
    sock.close()
    a.stop()


def test_reads_are_never_torn_by_a_snapshot_rotation(tmp_path):
    """A compaction swaps and rewrites structures an agent is concurrently reading.

    The guarantee is the boring one an operator actually needs: a read is either the
    state or an exception that never happens. This is also what the gossip heartbeat
    must not do - hash the live map without the lock and it becomes the thread that
    raises.
    """
    import threading

    from conflux.state import Replica

    server = Server("reader", ("127.0.0.1", 0), data_dir=str(tmp_path / "d"),
                    snapshot_threshold=5, gossip_interval=100).start()
    stop = threading.Event()
    problems = []
    seen = []

    def writer():
        rep = Replica("agent-w")
        round_no = 0
        while not stop.is_set():
            for i in range(20):
                rep._emit("counter_inc", f"k{(round_no * 20 + i) % 40}", {"by": 1})
            server._absorb(rep.full_journal())
            round_no += 1

    def reader():
        while not stop.is_set():
            try:
                view = server.view()
                seen.append(len(view))
                digest = server.hash()
                server._heartbeat_links(time.time())
                assert isinstance(digest, str) and len(digest) == 64
            except Exception as exc:                      # noqa: BLE001 - the test is the point
                problems.append(f"{type(exc).__name__}: {exc}")
            time.sleep(0.002)

    def churner():
        while not stop.is_set():
            server._gossip_round()
            time.sleep(0.01)

    threads = [threading.Thread(target=writer, daemon=True), threading.Thread(target=reader, daemon=True),
               threading.Thread(target=churner, daemon=True)]
    for thread in threads:
        thread.start()
    time.sleep(0.7)
    stop.set()
    for thread in threads:
        thread.join(timeout=5)
    server.stop()
    assert not problems, problems[:3]
    assert seen and max(seen) == 40, (min(seen), max(seen))
    assert len(server.view()) == 40


def test_the_new_series_are_exposed_for_alerting():
    server = Server_fixture()
    text = server.prometheus()
    for series in ("malformed_frames_total", "rejected_connections_total", "links_dropped_total",
                   "gossip_errors_total", "handler_errors_total", "journal_actions",
                   "oversize_frames_total", "max_connections", "max_actions_per_frame"):
        assert f"conflux_{series}" in text, series
        assert f"# TYPE conflux_{series} " in text, series
    # a counter that is absent at zero is indistinguishable from a scrape that failed,
    # so the failure series must always be present
    assert text.count("# TYPE conflux_link_cursor ") == 0
    assert "conflux_link_cursor" not in text
    server.stop()


def test_link_cursors_are_one_sample_per_link_with_one_header_block():
    server = Server_fixture(gossip_interval=0.05, stale_after=30)
    peers = [Server_fixture(gossip_interval=0.05, stale_after=30) for _ in range(2)]
    for peer in peers:
        peer.add_peer(server.node_address)
    assert wait_until(lambda: len(server._links) >= 2), server.metrics()
    text = server.prometheus()
    header = "# TYPE conflux_link_cursor gauge"
    assert text.count(header) == 1, "repeating HELP/TYPE per sample breaks the format"
    samples = [line for line in text.splitlines() if line.startswith("conflux_link_cursor{")]
    assert len(samples) == len(server._links) and all("link=" in line for line in samples)
    server.stop()
    for peer in peers:
        peer.stop()
