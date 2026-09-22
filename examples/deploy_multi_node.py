"""Deployment example: a signed agent crew over real sockets.

Starts a 3-node TCP mesh plus a 2-node WebSocket mesh inside one process
(threads), registers agents with namespaces + quotas, signs submits with
Ed25519, converges every node, and reports per-node admin stats.

Real multi-process deployment is the same code with one Server per process —
see `python -m conflux.server --help`.

Run from the repo root:  python examples/deploy_multi_node.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import AgentRegistry, Client, Server, generate_ed25519_keypair


def wait_converged(nodes, key, value, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(n.hash() == nodes[0].hash() and n.read(key) == value for n in nodes):
            return True
        time.sleep(0.05)
    return False


def main():
    private, public = generate_ed25519_keypair()
    registry = AgentRegistry()
    registry.register(
        "driver",
        public_key=public,
        ops={"counter_inc", "lww_set"},
        namespaces=("orders.", "status."),
        quota=10_000,
    )

    crew_tcp = [
        Server("alpha", ("127.0.0.1", 0), registry=registry,
               gossip_interval=0.1).start(),
        Server("beta", ("127.0.0.1", 0),
               registry=registry, gossip_interval=0.1).start(),
        Server("gamma", ("127.0.0.1", 0),
               registry=registry, gossip_interval=0.1).start(),
    ]
    crew_ws = [
        Server("ws1", ("127.0.0.1", 0), transport="ws", registry=registry,
               gossip_interval=0.1).start(),
        Server("ws2", ("127.0.0.1", 0), transport="ws",
               registry=registry, gossip_interval=0.1).start(),
    ]

    crew_tcp[1].add_peer(crew_tcp[0].node_address)
    crew_tcp[2].add_peer(crew_tcp[1].node_address)
    crew_ws[1].add_peer(crew_ws[0].node_address)

    try:
        with Client("127.0.0.1", crew_tcp[0].node_address[1], agent_id="driver",
                    secret=private, signature_alg="ed25519") as c:
            c.counter_inc("orders.total", 5)
            c.register_set("status.region", "emea")

        # a second submit into the independent ws crew shows both crews converge
        with Client("127.0.0.1", crew_ws[0].node_address[1], agent_id="driver",
                    secret=private, signature_alg="ed25519", transport="ws") as c2:
            c2.counter_inc("orders.total", 2)

        print("tcp crew converged:", wait_converged(crew_tcp, "orders.total", 5),
              "->", {n.node_id: n.read("orders.total") for n in crew_tcp})
        print(" ws crew converged:", wait_converged(crew_ws, "orders.total", 2),
              "->", {n.node_id: n.read("orders.total") for n in crew_ws})
        print()
        for n in crew_tcp + crew_ws:
            print(f"  {n.node_id:5s} {n.metrics()}")
    finally:
        for n in crew_tcp + crew_ws:
            n.stop()


if __name__ == "__main__":
    main()