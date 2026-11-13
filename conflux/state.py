from .actions import (
    Action,
    OP_COUNTER_DEC,
    OP_COUNTER_INC,
    OP_LWW_SET,
    OP_MV_SET,
    OP_ORSET_ADD,
    OP_ORSET_REMOVE,
)
from .crdts import MVRegister, ORMap


class State:
    def __init__(self, root=None):
        self._map = root if root is not None else ORMap()

    def put(self, key, crdt):
        self._map = self._map.put(key, crdt)

    def merge(self, other):
        self._map = self._map.merge(other._map)
        return self

    def read(self, key):
        return self._map.read(key)

    def candidates(self, key):
        crdt = self._map.get(key)
        if isinstance(crdt, MVRegister):
            return crdt.candidates()
        return []

    def conflicts(self, prefix=None):
        """Return a mapping of keys to concurrent candidate lists for any MVRegister with 2+ distinct values."""
        out = {}
        for key in sorted(self._map.keys()):
            if prefix is not None and not key.startswith(prefix):
                continue
            crdt = self._map.get(key)
            if isinstance(crdt, MVRegister) and crdt.has_conflict():
                out[key] = crdt.candidates()
        return out

    def view(self):
        return {key: self._map.read(key) for key in sorted(self._map.keys())}

    def keys(self):
        return self._map.keys()

    def canonical(self):
        return self._map.canonical()


class Replica:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.state = State()
        self.journal = []
        self._tick = 0
        self._applied = set()
        self.on_absorb = None

    def counter_inc(self, key, by=1):
        return self._emit(OP_COUNTER_INC, key, {"by": by})

    def counter_dec(self, key, by=1):
        return self._emit(OP_COUNTER_DEC, key, {"by": by})

    def register_set(self, key, value):
        return self._emit(OP_LWW_SET, key, {"value": value})

    def register_set_weighted(self, key, value, weight):
        return self._emit(OP_LWW_SET, key, {"value": value, "weight": weight})

    def mv_set(self, key, value, supersedes=None):
        """Write to a conflict-preserving Multi-Value Register.

        If ``supersedes`` is omitted, automatically supersedes the dots this
        replica has ALREADY observed locally on ``key`` — so sequential local
        writes replace prior values cleanly, whereas concurrent writes from
        un-synced peers are BOTH preserved as conflicts until resolved!
        """
        if supersedes is None:
            existing = self.state._map.get(key)
            supersedes = list(existing.active_dots()) if isinstance(existing, MVRegister) else []
        else:
            supersedes = [str(s) for s in supersedes]
        return self._emit(OP_MV_SET, key, {"value": value, "supersedes": supersedes})

    def mv_resolve(self, key, resolved_value):
        """Causally resolve all currently observed concurrent candidates on ``key`` into ``resolved_value``."""
        existing = self.state._map.get(key)
        dots = list(existing.active_dots()) if isinstance(existing, MVRegister) else []
        return self._emit(OP_MV_SET, key, {"value": resolved_value, "supersedes": dots})

    def conflicts(self, prefix=None):
        return self.state.conflicts(prefix=prefix)

    def candidates(self, key):
        return self.state.candidates(key)

    def set_add(self, key, value, tag=None):
        return self._emit(OP_ORSET_ADD, key, {"value": value, "tag": tag or self._next_tag()})

    def set_add_weighted(self, key, value, weight, tag=None):
        return self._emit(
            OP_ORSET_ADD, key, {"value": value, "weight": weight, "tag": tag or self._next_tag()}
        )

    def set_remove(self, key, tag):
        return self._emit(OP_ORSET_REMOVE, key, {"tag": tag})

    def read(self, key):
        return self.state.read(key)

    def timestamp(self):
        return self._tick

    def _next_tag(self):
        return f"{self.agent_id}:{self._tick}"

    def _emit(self, op, key, params):
        self._tick += 1
        action = Action(f"act-{self.agent_id}-{self._tick}", self.agent_id, self._tick, op, key, params)
        self.apply(action)
        self.journal.append(action)
        if self.on_absorb is not None:
            self.on_absorb([action], "local")
        return action

    def absorb(self, actions, origin="absorb"):
        absorbed = []
        for action in actions:
            if action.action_id in self._applied:
                continue
            self.apply(action)
            self.journal.append(action)
            absorbed.append(action)
        if absorbed and self.on_absorb is not None:
            self.on_absorb(absorbed, origin)
        return absorbed

    def apply(self, action):
        if action.action_id in self._applied:
            return
        self._applied.add(action.action_id)
        contribution = action.contribution()
        if contribution is not None:
            self.state.put(action.key, contribution)

    def ids_applied(self):
        return frozenset(self._applied)

    def full_journal(self):
        by_id = {a.action_id: a for a in self.journal}
        return [by_id[k] for k in sorted(by_id)]