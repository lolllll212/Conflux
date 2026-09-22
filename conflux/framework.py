from .actions import Action, OP_COUNTER_DEC, OP_COUNTER_INC, OP_LWW_SET, OP_ORSET_ADD, OP_ORSET_REMOVE
from .state import Replica

OP_ALIASES = {
    "set": OP_LWW_SET,
}


class InMemoryTransport:
    def __init__(self):
        self._boxes = {}

    def send(self, to, message):
        self._boxes.setdefault(to, []).append(message)

    def pending(self, to):
        return list(self._boxes.get(to, []))

    def drain(self, to):
        messages = self._boxes.pop(to, [])
        return messages


class DistributedApp:
    def __init__(self, node_id, transport=None, peers=()):
        self.node_id = node_id
        self.transport = transport
        self.peers = set(peers)
        self.replica = Replica(node_id)
        self._broadcasted = set()

    def emit(self, op, key, params):
        op = OP_ALIASES.get(op, op)
        return self.replica._emit(op, key, params)

    def counter_inc(self, key, by=1):
        return self.emit(OP_COUNTER_INC, key, {"by": by})

    def counter_dec(self, key, by=1):
        return self.emit(OP_COUNTER_DEC, key, {"by": by})

    def register_set(self, key, value):
        return self.emit(OP_LWW_SET, key, {"value": value})

    def register_set_weighted(self, key, value, weight):
        return self.emit(OP_LWW_SET, key, {"value": value, "weight": weight})

    def set_add(self, key, value, tag=None):
        return self.emit(OP_ORSET_ADD, key, {"value": value, "tag": tag or self._next_tag()})

    def set_add_weighted(self, key, value, weight, tag=None):
        return self.emit(
            OP_ORSET_ADD, key, {"value": value, "weight": weight, "tag": tag or self._next_tag()}
        )

    def set_remove(self, key, tag):
        return self.emit(OP_ORSET_REMOVE, key, {"tag": tag})

    def read(self, key):
        return self.replica.read(key)

    def _next_tag(self):
        return f"{self.node_id}:{self.replica.timestamp() + 1}"

    def broadcast(self):
        journal = self.replica.journal
        to_send = [action for action in journal if action.action_id not in self._broadcasted]
        if not to_send:
            return []
        for action in to_send:
            self._broadcasted.add(action.action_id)
        message = {"from": self.node_id, "actions": [action.export() for action in to_send]}
        for peer in self.peers:
            self.transport.send(peer, message)
        return [action.action_id for action in to_send]

    def consume_pending(self):
        consumed = 0
        for message in self.transport.drain(self.node_id):
            actions = [Action.import_action(data) for data in message["actions"]]
            self.replica.absorb(actions)
            consumed += len(actions)
        return consumed


class Cluster:
    def __init__(self, node_ids, transport=None):
        self.transport = transport if transport is not None else InMemoryTransport()
        self.apps = {}
        node_ids = set(node_ids)
        for node_id in node_ids:
            peers = node_ids - {node_id}
            self.apps[node_id] = DistributedApp(node_id, transport=self.transport, peers=peers)

    def node(self, node_id):
        return self.apps[node_id]

    def read(self, node_id, key):
        return self.apps[node_id].read(key)

    def gossip_round(self):
        for app in self.apps.values():
            app.broadcast()
        for app in self.apps.values():
            app.consume_pending()
        return self