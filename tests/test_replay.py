import random

from conflux.actions import Action
from conflux.protocol import converge_all, journal_from_json, journal_to_json, replay, state_hash
from conflux.state import Replica


def test_journal_round_trip_preserves_state():
    replicas = [Replica("r0"), Replica("r1"), Replica("r2")]
    replicas[0].counter_inc("orders", 3)
    replicas[0].register_set("profile.name", "alice")
    replicas[0].set_add("tokens", "t1", tag="tag-1")
    replicas[0].register_set_weighted("risk", "high", weight=0.5)
    replicas[1].counter_inc("orders", 4)
    replicas[1].set_remove("tokens", "tag-1")
    converge_all(replicas)

    reference = state_hash(replicas[0].state)
    actions = journal_from_json(journal_to_json(replicas[0]))

    for action in actions:
        assert isinstance(action, Action)
    assert state_hash(replay(actions).state) == reference

    rng = random.Random(0)
    shuffled = list(actions)
    rng.shuffle(shuffled)
    assert state_hash(replay(shuffled).state) == reference


def test_replay_from_scratch_matches_merged_state():
    left = Replica("l")
    right = Replica("r")
    left.counter_inc("views", 7)
    right.register_set("status", "online")
    left.set_add("pool", "x", tag="x1")
    right.counter_inc("views", 2)
    converge_all([left, right])

    actions = left.journal
    rebuilt = replay(actions)
    assert rebuilt.read("views") == 9
    assert rebuilt.read("status") == "online"
    assert rebuilt.read("pool") == frozenset({"x"})