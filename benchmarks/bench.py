"""Standard-library microbenchmarks for the conflux protocol.

Run with:
    python benchmarks/bench.py            # default size
    python benchmarks/bench.py --quick    # small sizes for CI sanity
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import Client, Cluster, Replica, Server, replay


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start


def bench_replay(n):
    source = Replica("gen")
    for i in range(n):
        source.counter_inc("orders", 1)
    journal = source.full_journal()
    with Timer() as t:
        rebuilt = replay(journal)
        _ = rebuilt.state.canonical()
    rate = n / t.elapsed
    return {"actions": n, "seconds": t.elapsed, "rate_s": round(rate),
            "canonical": rebuilt.read("orders")}


def bench_cluster_convergence(agents, per_node):
    cluster = Cluster({f"agent-{i}" for i in range(agents)})
    for i in range(agents):
        node = cluster.node(f"agent-{i}")
        for _ in range(per_node):
            node.counter_inc("orders", 1)
    with Timer() as t:
        cluster.gossip_round()
    total = agents * per_node
    return {"agents": agents, "actions": total, "seconds": t.elapsed,
            "rate_s": round(total / t.elapsed), "final": cluster.read("agent-0", "orders")}


def bench_server_roundtrip(transport, n, host="127.0.0.1"):
    server = Server("bench", (host, 0), transport=transport).start()
    try:
        with Client(host, server.node_address[1], agent_id="bench", transport=transport) as c:
            times = []
            for _ in range(n):
                with Timer() as t:
                    c.counter_inc("orders", 1)
                times.append(t.elapsed)
            final = c.read("orders")
        return {"transport": transport, "submits": n,
                "avg_ms": round(statistics.mean(times) * 1000, 3),
                "p50_ms": round(statistics.median(times) * 1000, 3),
                "p95_ms": round(sorted(times)[int(n * 0.95)] * 1000, 3),
                "final": final}
    finally:
        server.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="conflux microbenchmarks")
    parser.add_argument("--quick", action="store_true", help="small sizes for CI sanity")
    args = parser.parse_args(argv)
    n_replay = 2_000 if args.quick else 20_000
    n_cluster = (4, 500) if args.quick else (8, 2_000)
    n_submit = 50 if args.quick else 200

    print(f"bench replications: replay={n_replay}, cluster={n_cluster}, submits={n_submit}\n")

    print("== replay throughput (fold journal into fresh state) ==")
    r = bench_replay(n_replay)
    print(f"  {r['rate_s']:,} actions/s ({r['actions']:,} actions in {r['seconds']:.2f}s, "
          f"final counter={r['canonical']})")

    print("\n== in-memory cluster convergence (full-mesh gossip) ==")
    c = bench_cluster_convergence(*n_cluster)
    print(f"  {c['rate_s']:,} actions/s ({c['agents']} agents x {c['actions'] / c['agents']:.0f} each "
          f"in {c['seconds']:.2f}s, converged={c['final']})")

    for transport in ("tcp", "ws"):
        print(f"\n== {transport} single-agent submit round-trip (client -> ack) ==")
        s = bench_server_roundtrip(transport, n_submit)
        print(f"  avg={s['avg_ms']}ms p50={s['p50_ms']}ms p95={s['p95_ms']}ms "
              f"({s['submits']} submits, final={s['final']})")

    print("\ndone.")


if __name__ == "__main__":
    main()