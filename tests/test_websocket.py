import threading
import time

import pytest

from conflux import Action, Client, Server
from conflux.websocket import (
    WebSocket,
    WebSocketServer,
    _build_frame,
    accept_nonce,
    connect_websocket,
)


def test_accept_nonce_rfc_vector():
    assert accept_nonce("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_frame_codec_shapes():
    unmasked = _build_frame(0x2, b"hello")
    assert unmasked[:2] == b"\x82\x05"
    assert unmasked[2:] == b"hello"
    masked = _build_frame(0x1, b"ab", b"\x00\x01\x02\x03")
    assert masked[0] == 0x81
    assert masked[1] == 0x82
    assert masked[2:6] == b"\x00\x01\x02\x03"


def _spawn_echo_server():
    received = []

    def handler(ws):
        while True:
            data = ws.recv()
            if data is None:
                return
            received.append(data)
            ws.send(data)

    server = WebSocketServer("127.0.0.1", 0, handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_websocket_echo_round_trip():
    server, received = _spawn_echo_server()
    try:
        with connect_websocket("127.0.0.1", server.address[1]) as ws:
            for payload in (b"", b"x" * 125, b"y" * 1000, b"z" * 70000):
                ws.send(payload)
                echoed = ws.recv()
                assert echoed == payload
    finally:
        server.shutdown()
        server.server_close()


def test_websocket_json_messages():
    server, _ = _spawn_echo_server()
    try:
        with connect_websocket("127.0.0.1", server.address[1]) as ws:
            for obj in ({"type": "hello", "node_id": "x"}, {"type": "read", "key": "orders"}):
                ws.send_json(obj)
                assert ws.recv_json() == obj
    finally:
        server.shutdown()
        server.server_close()


def test_websocket_handshake_rejects_non_upgrade():
    server = WebSocketServer("127.0.0.1", 0, lambda ws: None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import socket as _socket

        with _socket.create_connection(("127.0.0.1", server.address[1]), timeout=3) as raw:
            raw.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            resp = raw.recv(1024)
        assert b"HTTP/1.1 400" in resp
    finally:
        server.shutdown()
        server.server_close()


def test_server_ws_two_nodes_converge():
    alpha = Server("alpha", ("127.0.0.1", 0), transport="ws", gossip_interval=0.05).start()
    beta = Server("beta", ("127.0.0.1", 0), peers=[alpha.node_address],
                  transport="ws", gossip_interval=0.05).start()
    try:
        with Client("127.0.0.1", beta.node_address[1], agent_id="wb", transport="ws") as c:
            for _ in range(5):
                c.counter_inc("orders")

        deadline = time.time() + 5
        while time.time() < deadline:
            if alpha.hash() == beta.hash() and alpha.read("orders") == 5:
                break
            time.sleep(0.05)
        assert alpha.read("orders") == 5
        assert alpha.hash() == beta.hash()
    finally:
        alpha.stop()
        beta.stop()


def test_client_ws_ed25519_submit():
    pytest.importorskip("cryptography")
    from conflux import AgentRegistry, generate_ed25519_keypair

    private, public = generate_ed25519_keypair()
    registry = AgentRegistry()
    registry.register("driver", public_key=public, ops={"counter_inc", "lww_set"})
    server = Server("n1", ("127.0.0.1", 0), transport="ws", registry=registry)
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="driver",
                    secret=private, signature_alg="ed25519", transport="ws") as c:
            c.counter_inc("orders", 7)
            c.register_set("status", "open")
            assert c.read("orders") == 7
            assert c.read("status") == "open"
    finally:
        server.stop()


def test_gossip_batching_chunks_by_size():
    server = Server("b", ("127.0.0.1", 0), batch_size=3)
    server.start()
    try:
        calls = []

        def fake_send(obj):
            calls.append(obj)

        key = server._register_link(fake_send, lambda: None, close=lambda: None)
        actions = [Action(f"a{i}", "x", 1, "counter_inc", f"k{i % 3}", {"by": 1})
                   for i in range(7)]
        server._absorb(actions)
        server._push_to_peers()
        deadline = time.time() + 2
        while time.time() < deadline and len(calls) < 3:
            time.sleep(0.01)
        server._drop_link(key)
        assert len(calls) == 3
        assert sum(len(msg["actions"]) for msg in calls) == 7
    finally:
        server.stop()


def test_dial_backoff_skips_failed_peers():
    attempts = [0]
    server = Server("b", ("127.0.0.1", 0), peers=[("127.0.0.1", 9)], gossip_interval=0.01)
    server._dial = lambda address: (attempts.__setitem__(0, attempts[0] + 1), None)[1]
    server.start()
    try:
        time.sleep(0.5)
        assert attempts[0] >= 2
        assert server._peer_failures[("127.0.0.1", 9)] == attempts[0]
        first = attempts[0]
        deadline = time.time() + 2
        while time.time() < deadline and attempts[0] == first:
            time.sleep(0.05)
        assert attempts[0] > first
    finally:
        server.stop()


def test_peer_discovery_through_mesh():
    c_addr = [None]

    alpha = Server("alpha", ("127.0.0.1", 0), transport="ws", gossip_interval=0.05,
                   auto_discover=True).start()
    beta = Server("beta", ("127.0.0.1", 0), peers=[alpha.node_address], transport="ws",
                  gossip_interval=0.05, auto_discover=True).start()
    gamma = Server("gamma", ("127.0.0.1", 0), peers=[beta.node_address], transport="ws",
                   gossip_interval=0.05, auto_discover=True).start()
    c_addr[0] = gamma.node_address
    try:
        deadline = time.time() + 6
        while time.time() < deadline:
            if gamma.node_address in alpha.peers or gamma.node_address in alpha.discovered:
                break
            time.sleep(0.05)
        assert gamma.node_address in alpha.peers or gamma.node_address in alpha.discovered
        assert beta.node_address in alpha.peers or beta.node_address in alpha.discovered
    finally:
        alpha.stop()
        beta.stop()
        gamma.stop()


def test_tcp_and_ws_nodes_can_mesh_only_on_same_transport():
    tcp_node = Server("t", ("127.0.0.1", 0), transport="tcp").start()
    try:
        ws_node = Server("w", ("127.0.0.1", 0), peers=[tcp_node.node_address], transport="ws")
        ws_node.start()
        try:
            time.sleep(0.2)
            assert tcp_node.node_address not in ws_node._outbound
        finally:
            ws_node.stop()
    finally:
        tcp_node.stop()