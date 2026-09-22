import random
import threading

from conflux.actions import OP_COUNTER_DEC, OP_COUNTER_INC, OP_LWW_SET, OP_ORSET_ADD, OP_ORSET_REMOVE
from conflux.protocol import converge_all, converge_by_gossip, replay, state_hash
from conflux.state import Replica


def run_agents(n, ops_per_agent, seed):
    replicas = [Replica(f"agent-{i}") for i in range(n)]
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()
        replica = replicas[i]
        prng = random.Random(seed + i * 117)
        local_tags = []
        tag_counter = 0
        for _ in range(ops_per_agent):
            choice = prng.choices(
                ["inc", "dec", "reg", "add", "rm"], weights=[30, 10, 30, 20, 10]
            )[0]
            if choice == "inc":
                replica.counter_inc("orders", prng.randint(1, 5))
            elif choice == "dec":
                replica.counter_dec("orders", prng.randint(1, 3))
            elif choice == "reg":
                replica.register_set("bid", prng.randint(1, 1000))
            elif choice == "add":
                tag = f"tag-{i}-{tag_counter}"
                tag_counter += 1
                local_tags.append(tag)
                replica.set_add("allocations", prng.randint(1, 50), tag=tag)
            else:
                if not local_tags:
                    continue
                tag = prng.choice(local_tags)
                local_tags.remove(tag)
                replica.set_remove("allocations", tag)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return replicas


def full_expected(replicas):
    orders = 0
    best = None
    elements = {}
    tombstones = set()
    seen = set()
    for replica in replicas:
        for action in replica.journal:
            if action.action_id in seen:
                continue
            seen.add(action.action_id)
            if action.op == OP_COUNTER_INC:
                orders += action.params["by"]
            elif action.op == OP_COUNTER_DEC:
                orders -= action.params["by"]
            elif action.op == OP_LWW_SET and action.fired():
                key = (action.tick, action.agent)
                if best is None or key > best[0]:
                    best = (key, action.params["value"])
            elif action.op == OP_ORSET_ADD and action.fired():
                elements[action.params["tag"]] = action.params["value"]
            elif action.op == OP_ORSET_REMOVE:
                tombstones.add(action.params["tag"])
    allocations = frozenset(
        value for tag, value in elements.items() if tag not in tombstones
    )
    return orders, best[1] if best is not None else None, allocations


def test_threaded_agents_converge_without_locking():
    replicas = run_agents(n=8, ops_per_agent=300, seed=1234)
    converge_all(replicas)

    assert len({state_hash(r.state) for r in replicas}) == 1

    orders, bid, allocations = full_expected(replicas)
    for replica in replicas:
        assert replica.read("orders") == orders
        assert replica.read("bid") == bid
        assert replica.read("allocations") == allocations


def test_merge_order_is_irrelevant():
    replicas = run_agents(n=5, ops_per_agent=150, seed=99)
    full = []
    for replica in replicas:
        full.extend(replica.journal)

    base = state_hash(replay(full).state)
    for seed in range(20):
        rng = random.Random(seed)
        order = list(full)
        rng.shuffle(order)
        assert state_hash(replay(order).state) == base


def test_gossip_based_convergence():
    replicas = run_agents(n=6, ops_per_agent=100, seed=7)
    converge_by_gossip(replicas, rounds=10, seed=1)
    assert len({state_hash(r.state) for r in replicas}) == 1