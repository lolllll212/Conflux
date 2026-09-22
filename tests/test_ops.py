import time

from conflux import Client, Server
from conflux.net import recv_msg, send_msg


def test_server_metrics_after_workload():
    server = Server("n1", ("127.0.0.1", 0), gossip_interval=0.05)
    server.start()
    try:
        before = server.metrics()
        assert before["node_id"] == "n1"
        assert before["absorbed"] == 0

        with Client("127.0.0.1", server.node_address[1], agent_id="c") as c:
            c.counter_inc("orders", 3)
            c.register_set("status", "open")

        metrics = server.metrics()
        assert metrics["absorbed"] == 2
        assert metrics["submitted"] == 2
        assert metrics["journal"] == 2
        assert metrics["links"] == 0
        assert metrics["state_hash"] == server.hash()
        assert metrics["uptime_s"] >= 0
        assert len(metrics["state_hash"]) == 64
    finally:
        server.stop()


def test_admin_stats_wire_op():
    server = Server("n1", ("127.0.0.1", 0))
    server.start()
    try:
        import socket

        with socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=5) as conn:
            send_msg(conn, {"type": "hello", "role": "agent", "node_id": "ops"})
            resp = recv_msg(conn)
            assert resp["type"] == "welcome"
            send_msg(conn, {"type": "admin", "what": "stats"})
            stats = recv_msg(conn)
            assert stats["type"] == "stats"
            assert stats["node_id"] == "n1"
            assert stats["absorbed"] == 0
            assert stats["uptime_s"] >= 0
    finally:
        server.stop()


def test_events_and_watchers_fire_on_server_wire_submit():
    server = Server("n1", ("127.0.0.1", 0))
    server.start()
    events = []
    server.subscribe(lambda action, origin: events.append((action.op, action.key)), op="counter_inc")
    changes = []
    server.watch("orders", lambda value, key, old: changes.append((value, old)))
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="c") as c:
            c.counter_inc("orders", 5)
            c.counter_inc("orders", 1)
        deadline = time.time() + 2
        while time.time() < deadline and server.read("orders") != 6:
            time.sleep(0.02)
        assert server.read("orders") == 6
        assert ("counter_inc", "orders") in events
        assert (6, 5) in changes
    finally:
        server.stop()