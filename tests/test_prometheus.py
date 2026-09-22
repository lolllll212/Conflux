import json
import socket
import urllib.request

from conflux import Client, Server
from conflux.net import recv_msg, send_msg


def _scrape(server):
    text = server.prometheus()
    body = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        body[name] = value
    return text, body


def test_prometheus_text_matches_metrics():
    server = Server("n1", ("127.0.0.1", 0), gossip_interval=0.05)
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="c") as c:
            c.counter_inc("orders", 3)
            c.register_set("status", "open")

        text, body = _scrape(server)
        assert "# HELP conflux_actions_absorbed_total" in text
        assert "# TYPE conflux_actions_absorbed_total counter" in text
        assert '# TYPE conflux_journal_actions gauge' in text
        assert body['conflux_actions_absorbed_total{node_id="n1"}'] == "2"
        assert body['conflux_actions_submitted_total{node_id="n1"}'] == "2"
        assert body['conflux_journal_actions{node_id="n1"}'] == "2"
        assert "state_hash_info" in text
        assert server.hash() in text
        assert abs(float(body['conflux_uptime_seconds{node_id="n1"}'])
                   - server.metrics()["uptime_s"]) < 0.05
    finally:
        server.stop()


def test_prometheus_admin_wire_op():
    server = Server("n1", ("127.0.0.1", 0))
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.node_address[1]), timeout=5) as conn:
            send_msg(conn, {"type": "hello", "role": "agent", "node_id": "ops"})
            resp = recv_msg(conn)
            assert resp["type"] == "welcome"
            send_msg(conn, {"type": "admin", "what": "prometheus"})
            reply = recv_msg(conn)
            assert reply["type"] == "prometheus"
            assert 'conflux_state_hash_info{node_id="n1"' in reply["body"]
            assert "# TYPE conflux_actions_absorbed_total counter" in reply["body"]
    finally:
        server.stop()


def test_prometheus_http_scrape():
    server = Server("n1", ("127.0.0.1", 0))
    server.start()
    try:
        with Client("127.0.0.1", server.node_address[1], agent_id="c") as c:
            c.counter_inc("orders", 1)

        addr = server.start_metrics_http()
        with urllib.request.urlopen(f"http://{addr[0]}:{addr[1]}/metrics",
                                    timeout=5) as resp:
            body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "text/plain" in resp.headers["Content-Type"]
        assert f'conflux_actions_absorbed_total{{node_id="n1"}} 1' in body
        assert server.start_metrics_http() == addr  # idempotent
    finally:
        server.stop()