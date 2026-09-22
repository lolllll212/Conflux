import random

from conflux.protocol import replay, state_hash
from conflux.state import Replica


def test_weighted_decisions_are_deterministic():
    replica = Replica("dist")
    fired_first = []
    for i in range(2000):
        action = replica.register_set_weighted("k", i, weight=0.5)
        fired_first.append(action.fired())
    fired_again = [action.fired() for action in replica.journal]
    assert fired_first == fired_again


def test_partial_logs_converge_to_full_replay():
    rng = random.Random(5)
    agents = [Replica(f"agent-{k}") for k in range(4)]
    actions = []
    for i in range(2000):
        action = agents[i % 4].register_set_weighted("k", rng.randint(0, 10**6), weight=0.5)
        actions.append(action)

    left = Replica("left")
    right = Replica("right")
    left.absorb(actions[::2])
    right.absorb(actions[1::2])
    left.absorb(right.journal)
    right.absorb(left.journal)

    reference = replay(actions)
    assert state_hash(left.state) == state_hash(reference.state)
    assert state_hash(right.state) == state_hash(reference.state)


def test_weight_distribution_is_roughly_balanced():
    rng = random.Random(11)
    replica = Replica("dist")
    fired = 0
    total = 10000
    for i in range(total):
        action = replica.set_add_weighted("pool", i, weight=0.5, tag=f"t{i}")
        if action.fired():
            fired += 1
    ratio = fired / total
    assert 0.45 < ratio < 0.55