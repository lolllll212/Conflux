import hashlib
import json
import random

from .actions import Action
from .state import Replica, State


def reconcile(a, b):
    a.absorb(b.journal)
    b.absorb(a.journal)
    return a, b


def converge_all(replicas):
    for i in range(len(replicas)):
        for j in range(i + 1, len(replicas)):
            reconcile(replicas[i], replicas[j])
    return replicas


def gossip_round(replicas, rng=None):
    rng = rng or random
    order = list(replicas)
    rng.shuffle(order)
    for i in range(len(order)):
        reconcile(order[i], order[(i + 1) % len(order)])
    return replicas


def converge_by_gossip(replicas, rounds, seed=None):
    rng = random.Random(seed)
    for _ in range(rounds):
        gossip_round(replicas, rng)
    return replicas


def replay(actions, agent_id="replay"):
    replica = Replica(agent_id)
    for action in sorted(actions, key=lambda a: (a.tick, a.agent, a.action_id)):
        replica.apply(action)
    return replica


def journal_to_json(replica):
    return "\n".join(json.dumps(a.export()) for a in replica.journal)


def journal_from_json(text):
    return [Action.import_action(json.loads(line)) for line in text.splitlines() if line.strip()]


def state_hash(state):
    raw = json.dumps(state.canonical(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()