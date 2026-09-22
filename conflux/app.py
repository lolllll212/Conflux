"""Multi-agent application layer on top of the replicated state fabric.

Everything here is expressed as deterministic Actions, so task and workflow
state is replicated like any other state: every node that has seen the same
actions reconstructs the same tasks, and subscriptions fire identically.
"""

import threading

from .actions import decide

TASK_CREATED = "created"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"

TASK_KEY_PREFIX = "tasks/"


def attach_listeners(replica, *listeners):
    """Route every absorbed action to a set of objects exposing ``on_absorb``
    (or plain callables invoked with action list and origin)."""
    active = [ln for ln in listeners if ln is not None]
    if not active:
        replica.on_absorb = None
        return
    def dispatch(actions, origin):
        for ln in active:
            if hasattr(ln, "on_absorb"):
                ln.on_absorb(actions, origin)
            else:
                ln(actions, origin)
    replica.on_absorb = dispatch


class EventBus:
    """Subscription layer over the action stream.

    A subscription matches by operation, exact key, and/or key prefix. Handlers
    receive ``(action, origin)`` and run synchronously in a worker thread of the
    absorbing replica; one handler's failure never stops the others.
    """

    def __init__(self):
        self._subs = {}
        self._next = 0
        self._lock = threading.RLock()

    def subscribe(self, handler, op=None, key=None, key_prefix=None):
        with self._lock:
            token = self._next
            self._next += 1
            self._subs[token] = {
                "handler": handler, "op": op, "key": key, "key_prefix": key_prefix,
            }
        return token

    def unsubscribe(self, token):
        with self._lock:
            self._subs.pop(token, None)

    def subscriptions(self):
        with self._lock:
            return len(self._subs)

    def publish(self, action, origin="local"):
        matched = 0
        with self._lock:
            subs = [dict(s) for s in self._subs.values()]
        for sub in subs:
            if sub["op"] is not None and sub["op"] != action.op:
                continue
            if sub["key"] is not None and sub["key"] != action.key:
                continue
            if sub["key_prefix"] is not None and not action.key.startswith(sub["key_prefix"]):
                continue
            matched += 1
            try:
                sub["handler"](action, origin)
            except Exception:
                pass
        return matched

    def on_absorb(self, actions, origin="absorb"):
        for action in actions:
            self.publish(action, origin=origin)


class Watcher:
    """Value-level watchers: fire when a watched key's effective value changes.

    Compare is by effective value (register value, counter total, set members),
    not by action. Watches replayed state, so it converges on every replica.
    """

    def __init__(self, replica, fire_initial=True):
        self.replica = replica
        self.fire_initial = fire_initial
        self._watches = {}
        self._values = {}
        self._lock = threading.RLock()

    def watch(self, key, handler):
        with self._lock:
            current = self.replica.read(key)
            self._watches[key] = handler
            self._values[key] = current
        if self.fire_initial and current is not None:
            handler(current, key, None)
        return key

    def unwatch(self, key):
        with self._lock:
            self._watches.pop(key, None)
            self._values.pop(key, None)

    def watched_keys(self):
        with self._lock:
            return sorted(self._watches)

    def check(self, keys=None):
        with self._lock:
            targets = list(keys) if keys is not None else list(self._watches)
            snapshots = {key: self.replica.read(key) for key in targets}
            fired = []
            for key in targets:
                current = snapshots[key]
                previous = self._values.get(key)
                if current != previous:
                    self._values[key] = current
                    fired.append((key, current, previous))
        for key, current, previous in fired:
            handler = self._watches.get(key)
            if handler is not None:
                try:
                    handler(current, key, previous)
                except Exception:
                    pass

    def on_absorb(self, actions, origin="absorb"):
        self.check()


class TaskStore:
    """Replicated task execution state.

    A task is a last-writer-wins record under ``tasks/<task_id>`` updated by the
    single agent that owns it, so status transitions are deterministic and any
    node can read or reconstruct the same task from the action log.
    """

    def __init__(self, replica, prefix=TASK_KEY_PREFIX):
        self.replica = replica
        self.prefix = prefix

    def _key(self, task_id):
        return f"{self.prefix}{task_id}"

    def _write(self, task_id, **fields):
        record = dict(self.read(task_id) or {})
        record.update(fields)
        self.replica.register_set(self._key(task_id), record)

    def create(self, task_id, kind, agent=None, inputs=None):
        self._write(task_id, id=task_id, kind=kind,
                    agent=agent or self.replica.agent_id, status=TASK_CREATED,
                    step=None, inputs=dict(inputs or {}), result=None, error=None)
        return self.read(task_id)

    def start(self, task_id):
        self._write(task_id, status=TASK_RUNNING)
        return self.read(task_id)

    def advance(self, task_id, step):
        self._write(task_id, status=TASK_RUNNING, step=step)
        return self.read(task_id)

    def complete(self, task_id, result=None):
        self._write(task_id, status=TASK_DONE, result=result, step=None)
        return self.read(task_id)

    def fail(self, task_id, error):
        self._write(task_id, status=TASK_FAILED, error=str(error), step=None)
        return self.read(task_id)

    def read(self, task_id):
        return self.replica.read(self._key(task_id))

    def list(self, status=None, kind=None):
        out = []
        for key in self.replica.state.keys():
            if not key.startswith(self.prefix):
                continue
            record = self.replica.read(key)
            if record is None:
                continue
            if status is not None and record.get("status") != status:
                continue
            if kind is not None and record.get("kind") != kind:
                continue
            out.append(record)
        return sorted(out, key=lambda r: r.get("id", ""))


class WorkflowStep:
    def __init__(self, name, kind, key=None, params=None, weight=None):
        self.name = name
        self.kind = kind
        self.params = dict(params or {})
        self.weight = weight
        self.key = key

    def _ops(self, replica, task_id, step_index):
        """Return (key, apply) pairs; ``apply`` emits the state mutation."""
        key = self.key or f"wf/{task_id}/{self.name}"
        if self.kind == "counter_inc":
            return [(key, lambda: replica.counter_inc(key, int(self.params.get("by", 1))))]
        if self.kind == "counter_dec":
            return [(key, lambda: replica.counter_dec(key, int(self.params.get("by", 1))))]
        if self.kind == "lww_set":
            return [(key, lambda: replica.register_set(key, self.params["value"]))]
        if self.kind == "orset_add":
            tag = self.params.get("tag") or f"{task_id}:{step_index}"
            return [(key, lambda: replica.set_add(key, self.params["value"], tag=tag))]
        if self.kind == "orset_remove":
            return [(key, lambda: replica.set_remove(key, self.params["tag"]))]
        if self.kind == "noop":
            return []
        raise ValueError(f"unknown workflow step kind {self.kind!r}")

    def fired(self, task_id, step_index):
        if self.weight is None:
            return True
        return decide((task_id, step_index, self.name), self.weight)


class Workflow:
    def __init__(self, workflow_id, steps):
        self.workflow_id = workflow_id
        self.steps = list(steps)

    @property
    def step_names(self):
        return [step.name for step in self.steps]


class WorkflowRunner:
    """Deterministic step executor: the same task inputs always produce the
    same action sequence, so every node replays the same workflow."""

    def __init__(self, replica, workflow, store=None):
        self.replica = replica
        self.workflow = workflow
        self.store = store or TaskStore(replica)

    def run(self, task_id, kind, inputs=None):
        self.store.create(task_id, kind, agent=self.replica.agent_id, inputs=inputs)
        self.store.start(task_id)
        result = {}
        for index, step in enumerate(self.workflow.steps):
            if not step.fired(task_id, index):
                continue
            self.store.advance(task_id, step.name)
            observed = {}
            for key, apply in step._ops(self.replica, task_id, index):
                apply()
                value = self.replica.read(key)
                if isinstance(value, (set, frozenset)):
                    value = sorted(value)
                observed[key] = value
            if observed:
                result[step.name] = observed if len(observed) > 1 else next(iter(observed.values()))
        self.store.complete(task_id, result)
        return self.store.read(task_id)