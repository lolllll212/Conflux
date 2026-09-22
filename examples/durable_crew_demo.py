"""The one end-to-end demo: shared agent state that survives a crash.

A 3-node crew shares the state of one agent ("driver"): a live counter and a
status register. Signed writes (HMAC), gossip convergence, durable journals +
cumulative snapshots, and recovery from a simulated hard crash (a torn journal
tail) onto the exact same state. Workflow state (task queues) and a shared
knowledge layer run on the same fabric; this demo shows the agent-state head.

Run from the repo root:  python examples/durable_crew_demo.py
"""

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import AgentRegistry, Client, Server
from conflux.storage import JournalStore


def wait_converged(servers, key, value, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(s.read(key) == value for s in servers):
            return True
        time.sleep(0.05)
    return False


def crash_server(node, data_dir):
    """Hard-kill simulation: stop cleanly, then tear the tail of the journal
    the way an OS-level crash would (a partial write buffered mid-append)."""
    node.stop()
    with open(os.path.join(data_dir, "journal.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"action": {"torn": true')


def main():
    root = tempfile.mkdtemp(prefix="conflux-demo-")
    secret = "crew-secret"
    registry = AgentRegistry()
    registry.register("driver", secret, ops={"counter_inc", "lww_set"},
                      namespaces=("orders.", "status."), quota=1000)

    data = {n: os.path.join(root, n) for n in ("alpha", "beta", "gamma")}
    nodes = [
        Server("alpha", ("127.0.0.1", 0), registry=registry, data_dir=data["alpha"],
               gossip_interval=0.1, snapshot_threshold=3).start(),
        Server("beta", ("127.0.0.1", 0), registry=registry, data_dir=data["beta"],
               gossip_interval=0.1, snapshot_threshold=3).start(),
        Server("gamma", ("127.0.0.1", 0), registry=registry, data_dir=data["gamma"],
               gossip_interval=0.1, snapshot_threshold=3).start(),
    ]
    alpha, beta, gamma = nodes
    alpha.add_peer(beta.node_address)
    beta.add_peer(gamma.node_address)
    print("shared agent state: crew of 3 nodes up (tcp, HMAC-signed, durable journals)\n")

    try:
        with Client("127.0.0.1", alpha.node_address[1], agent_id="driver",
                    secret=secret) as c:
            for _ in range(4):
                c.counter_inc("orders.total", 1)
            c.register_set("status.region", "emea")

            assert wait_converged(nodes, "orders.total", 4)
            assert wait_converged(nodes, "status.region", "emea")
            hashes = {n.node_id: n.hash() for n in nodes}
            assert len(set(hashes.values())) == 1, "all nodes must converge to one state_hash"
            print("converged: every node has the identical state")
            print("  state_hash:", hashes["alpha"])
            print("  gamma journal actions:", gamma.metrics()["journal"])
            print("  membership: alpha=%d beta=%d gamma=%d" % (
                alpha.metrics()["members"], beta.metrics()["members"],
                gamma.metrics()["members"]))

            # -- hard crash on gamma: stop, tear a journal tail --
            crash_server(gamma, data["gamma"])
            print("\ncrash: gamma stopped, journal tail torn (simulated kill -9)")

            # the same agent keeps writing while the partition is up
            c.register_set("status.region", "izada")   # a later tick, so LWW wins
            assert wait_converged([alpha, beta], "status.region", "izada")

            # -- restart gamma on the same data dir: must replay + heal --
            gamma2 = Server("gamma", gamma.node_address, registry=registry,
                            data_dir=data["gamma"], gossip_interval=0.1,
                            snapshot_threshold=3).start()
            beta.add_peer(gamma2.node_address)
            print("restart: gamma replays its journal + cumulative snapshot")

            assert wait_converged([alpha, beta, gamma2], "orders.total", 4)
            assert wait_converged([alpha, beta, gamma2], "status.region", "izada")
            assert gamma2.hash() == alpha.hash() == beta.hash()
            assert gamma2.read("orders.total") == 4
            print("recovered: gamma is identical to its peers again")
            print("  orders.total:", gamma2.read("orders.total"),
                  "status.region:", gamma2.read("status.region"))

        # the torn line was refused, not blindly replayed
        recovered = JournalStore(data["gamma"], journal_secret=None)
        snap_root, actions = recovered.load()
        source = "cumulative snapshot (journal tail below threshold)" if snap_root else "journal replay"
        print(f"  recovery source: {source}; {len(actions)} tail action(s) replayed, torn line dropped")

        metrics_http = alpha.start_metrics_http()
        print("\nmetrics: prometheus http smoke at http://%s:%d/metrics" % metrics_http)
        print("  prometheus probe:", alpha.prometheus().splitlines()[-1])
        print("\ndemo passed: shared agent state - converge -> crash -> recover -> converge")
    finally:
        for n in nodes:
            try:
                n.stop()
            except (AttributeError, OSError):
                pass
        try:
            gamma2.stop()
        except (AttributeError, UnboundLocalError):
            pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()