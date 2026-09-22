"""The real agent-state demo: a coordinated agent crew on the hardened runtime.

Six worker agents share signed state across a 3-node TCP mesh:
  - `crew.work_done`     shared counter   (total units of work, PNCounter)
  - `crew.online`        capability set   (workers check themselves in/out)
  - `crew.leader`        slot register    (deterministic canonical tie-break)
  - `crew.worker.<w>.status`  per-worker registers

Then the runtime is put through the failure model it was hardened for:
  - gamma crashes hard (torn journal tail),
  - the crew keeps working during the partition,
  - gamma restarts on the same signed, snapshotted data dir and re-converges,
  - a signed log reloaded with the WRONG key is refused (silent-downgrade guard),
  - an out-of-namespace agent is refused and mutates nothing.

No frameworks, no abstractions: Server / Client / AgentRegistry / JournalStore.

Run from the repo root:  python examples/agent_crew_demo.py
"""

import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import AgentRegistry, Client, Server
from conflux.storage import JournalStore

NODE_KEY = "node-journal-secret"


def wait_converged(servers, key, value, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if all(s.read(key) == value for s in servers):
                return True
        except Exception:
            pass
        time.sleep(0.05)
    return False


def worker_job(host, port, worker, secret):
    """One agent's unit of work: a status flip, a counted action, a check-in,
    and a vote for the leader slot."""
    with Client(host, port, agent_id=worker, secret=secret) as c:
        c.register_set(f"crew.worker.{worker}.status", "working")
        c.counter_inc("crew.work_done", 1)
        c.set_add("crew.online", worker, tag=f"online:{worker}")
        c.register_set("crew.leader", worker)
        c.register_set(f"crew.worker.{worker}.status", "done")
        return c.read("crew.work_done")


def main():
    root = tempfile.mkdtemp(prefix="conflux-crew-")
    registry = AgentRegistry()
    nodenames = ("alpha", "beta", "gamma")
    data = {n: os.path.join(root, n) for n in nodenames}
    for i in range(1, 7):
        registry.register(f"w{i}", f"w{i}-secret",
                          ops={"counter_inc", "lww_set", "orset_add", "orset_remove"},
                          namespaces=("crew.",), quota=100)
    registry.register("supervisor", "sup-secret",
                      ops={"counter_inc", "lww_set"}, namespaces=("crew.",))
    registry.register("outsider", "evil-secret", namespaces=("external.",))

    nodes = [Server(n, ("127.0.0.1", 0), registry=registry, data_dir=data[n],
                    gossip_interval=0.1, snapshot_threshold=3,
                    journal_secret=NODE_KEY) for n in nodenames]
    alpha, beta, gamma = nodes
    for s in nodes:
        s.start()
    alpha.add_peer(beta.node_address)
    beta.add_peer(gamma.node_address)
    print("agent crew up: 3 signed, durable nodes + 6 agents\n")

    try:
        host, port = "127.0.0.1", alpha.node_address[1]

        # six agents write concurrently; no locks, no serialization, no order
        threads = [threading.Thread(target=worker_job, args=(
            host, port, f"w{i}", f"w{i}-secret")) for i in range(1, 7)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert wait_converged(nodes, "crew.work_done", 6)
        assert wait_converged(nodes, "crew.leader", "w6"), "equal stamps must tie-break canonically"
        expected_online = frozenset(f"w{i}" for i in range(1, 7))
        assert {n.read("crew.online") for n in nodes} == {expected_online}
        hashes = {n.node_id: n.hash() for n in nodes}
        assert len(set(hashes.values())) == 1, "crew must hold one state"
        print("converged: identical state_hash %s" % hashes["alpha"])
        print("  work_done=6 online=6 leader=%s (deterministic tie-break)"
              % alpha.read("crew.leader"))

        # trust boundary: an out-of-namespace agent is refused before merge
        with Client(host, port, agent_id="outsider", secret="evil-secret") as c:
            try:
                c.counter_inc("crew.poison", 100)
                raise SystemExit("outsider write was NOT refused")
            except ConnectionError:
                pass
        assert alpha.read("crew.poison") is None
        assert alpha.hash() == hashes["alpha"], "refused write must not move state"
        print("  trust: out-of-namespace write refused, state unchanged")

        # -- crash: gamma stops, journal tail torn (simulated kill -9) --
        gamma.stop()
        with open(os.path.join(data["gamma"], "journal.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write('{"action": {"torn": true')
        print("\ncrash: gamma down; the crew keeps working through the partition")

        # the crew (actually one supervisor) keeps writing while gamma is out
        with Client(host, port, agent_id="supervisor", secret="sup-secret") as c:
            c.counter_inc("crew.work_done", 1)
            c.register_set("crew.quality", "good")
        assert wait_converged([alpha, beta], "crew.quality", "good")

        # -- restart gamma on the same signed data dir --
        gamma2 = Server("gamma", gamma.node_address, registry=registry,
                        data_dir=data["gamma"], gossip_interval=0.1,
                        snapshot_threshold=3, journal_secret=NODE_KEY).start()
        beta.add_peer(gamma2.node_address)
        print("restart: gamma replays its signed snapshot + journal tail")

        assert wait_converged([alpha, beta, gamma2], "crew.work_done", 7)
        assert wait_converged([alpha, beta, gamma2], "crew.quality", "good")
        assert gamma2.hash() == alpha.hash() == beta.hash()
        print("healed: gamma identical again  work_done=7  quality=%r"
              % gamma2.read("crew.quality"))

        # silent-downgrade guard: a signed log refuses a keyless load
        try:
            JournalStore(data["gamma"], journal_secret=None).load()
            raise SystemExit("keyless load of a signed log was NOT refused")
        except ValueError:
            print("  durability: signed log refused without the journal key")

        print("\ndemo passed: signed agent crew - converge -> crash -> heal -> reconverge")
    finally:
        for n in nodes:
            try:
                n.stop()
            except Exception:
                pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()