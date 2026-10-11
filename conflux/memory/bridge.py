"""Optional bridge to the Conflux fabric: memories as replicated, replayable state.

SQLite stays the source of truth for memory content. This module only *mirrors*
it, so a memory namespace can be shared between two agents, archived as a
journal, or fingerprinted without inventing a second storage engine:

* every memory becomes a last-writer-wins register at
  ``memory/<namespace>/<id>`` carrying body, kind, topic, provenance and
  ``state``. Retiring a memory is just a newer register whose state is not
  ``active``, so no lock, delete or tombstone bookkeeping is involved;
* registers order by the replica's ``(tick, agent)`` stamp, which is Conflux's
  rule and not the memory ``revision``: a mirror can therefore transiently show
  an older revision until the next sync. ``put()`` refuses to publish a payload
  older than the one already mirrored (locally that ordering is honoured), and
  ``verify_against`` reports any drift, because the database stays authoritative;
* ``export_journal`` / ``import_journal`` round-trip through Conflux ``Action``
  records, and ``hash()`` is the library's ordinary ``state_hash`` — a
  deterministic fingerprint a run can record for later verification.

Nothing here talks to a network: transport is the caller's problem, and
``replay``/gossip order never changes the result.
"""

import json

from ..actions import Action
from ..protocol import state_hash
from ..state import Replica
from ..storage import JournalStore
from .store import ACTIVE

MEMORY_PREFIX = "memory/"
BRIDGE_VERSION = 1


def _revision(value):
    """Revision as an int, defaulting to 1 without swallowing 0 or negatives."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def memory_key(namespace, memory_id):
    return f"{MEMORY_PREFIX}{namespace}/{memory_id}"


class MemoryFabric:
    """Project one namespace's memories onto a Conflux replica."""

    def __init__(self, agent_id, namespace="default", data_dir=None, snapshot_threshold=1000,
                 journal_secret=None):
        self.agent_id = agent_id
        self.namespace = namespace or "default"
        self.replica = Replica(agent_id)
        self.prefix = MEMORY_PREFIX + self.namespace + "/"
        self.store = JournalStore(data_dir, snapshot_threshold, journal_secret) if data_dir else None
        if self.store is not None:
            root, actions = self.store.load()
            if root is not None:
                self.replica.state._map = root
            for action in actions:
                self.replica.apply(action)

    # -- projection ---------------------------------------------------------

    @staticmethod
    def payload_for(record):
        """The JSON a memory projects onto the fabric (works for dicts too)."""
        def pick(name, default=None):
            value = getattr(record, name, None)
            if value is None and isinstance(record, dict):
                value = record.get(name, default)
            return value
        return {
            "id": pick("id"), "kind": pick("kind", "note") or "note", "topic": pick("topic"),
            "body": pick("body", "") or "", "fingerprint": pick("fingerprint"),
            "state": pick("state", ACTIVE) or ACTIVE,
            "revision": _revision(pick("revision")),
            "importance": pick("importance", 1.0), "confidence": pick("confidence"),
            "created_at": pick("created_at"), "updated_at": pick("updated_at"),
            "provenance": pick("provenance", {}) or {}, "v": BRIDGE_VERSION,
        }

    def put(self, record, *, force=False):
        """Mirror one memory (or its retired state) as a replicated register.

        A payload older than what is already mirrored is skipped rather than
        published, because register order is the replica stamp and not the
        memory revision; ``force=True`` re-publishes anyway (useful after a
        manual correction to the row).
        """
        payload = self.payload_for(record)
        if not payload["id"]:
            raise ValueError("fabric projection needs a memory id")
        key = memory_key(self.namespace, payload["id"])
        current = self.replica.read(key)
        if not force and current:
            if int(current.get("revision") or 0) > payload["revision"]:
                return dict(payload, mirrored=False, reason="older than the mirrored revision",
                            mirrored_revision=current.get("revision"))
            if current == payload:
                # re-syncing an unchanged row is the common case, and a no-op here
                # keeps the journal from growing one action per memory per pass
                return dict(payload, mirrored=False, reason="unchanged",
                            mirrored_revision=current.get("revision"))
        action = self.replica.register_set(key, payload)
        if self.store is not None and self.store.append(action):
            self.store.rotate(self.replica.state._map)
        return dict(payload, mirrored=True, reason=None)

    def retire(self, memory_id, state="pruned", *, body=None, revision=1):
        """Mark a memory inactive in the shared view without deleting anything."""
        if state == ACTIVE:
            raise ValueError("use put() to publish an active memory")
        payload = {"id": memory_id, "state": state, "revision": int(revision), "body": body or "",
                   "kind": "note", "topic": None, "fingerprint": None, "provenance": {},
                   "importance": 1.0, "confidence": None, "created_at": None, "updated_at": None,
                   "v": BRIDGE_VERSION}
        return self.put(payload)

    def sync_store(self, store, *, namespace=None, limit=100000, force=False):
        """Mirror a whole namespace from the database.

        Re-running it is free: an unchanged row re-publishes the same payload and
        a row the mirror already carries at a newer revision is skipped, so the
        report distinguishes what actually moved.
        """
        ns = namespace or self.namespace
        mirrored = skipped = unchanged = 0
        for memory in store.list(state=None, namespace=ns, limit=limit):
            receipt = self.put(memory, force=force)
            if receipt["mirrored"]:
                mirrored += 1
            elif receipt["reason"] == "unchanged":
                unchanged += 1
            else:
                skipped += 1
        return {"scanned": mirrored + unchanged + skipped, "mirrored": mirrored,
                "unchanged": unchanged, "skipped_older": skipped, "hash": self.hash()}

    # -- reads ---------------------------------------------------------------

    def view(self, *, active_only=True):
        """Memories this replica has seen, keyed by id, oldest id first."""
        out = {}
        for key in self.replica.state.keys():
            if not key.startswith(self.prefix):
                continue
            record = self.replica.read(key)
            if not record:
                continue
            if active_only and record.get("state") != ACTIVE:
                continue
            out[record["id"]] = record
        return {k: out[k] for k in sorted(out)}

    def read(self, memory_id):
        return self.replica.read(memory_key(self.namespace, memory_id))

    def keys(self):
        return sorted(k for k in self.replica.state.keys() if k.startswith(self.prefix))

    def hash(self):
        return state_hash(self.replica.state)

    def reconcile(self, other):
        """Two-way gossip with another fabric, labelled from each side's view."""
        theirs = other.replica.absorb(self.replica.journal)
        mine = self.replica.absorb(other.replica.journal)
        return {"received_by_us": len(mine), "received_by_peer": len(theirs)}

    # -- portability ---------------------------------------------------------

    def export_actions(self):
        return [action.export() for action in self.replica.full_journal()]

    def export_journal(self):
        return "\n".join(json.dumps(a, sort_keys=True, ensure_ascii=False)
                         for a in self.export_actions())

    def import_actions(self, actions):
        """Absorb Conflux actions, keeping only this namespace's memory keys."""
        incoming = []
        for item in actions or ():
            action = item if isinstance(item, Action) else Action.import_action(item)
            if action.key.startswith(self.prefix):
                incoming.append(action)
        absorbed = self.replica.absorb(incoming)
        if self.store is not None:
            for action in absorbed:
                if self.store.append(action):
                    self.store.rotate(self.replica.state._map)
        return len(absorbed)

    def import_journal(self, text):
        records = [json.loads(line) for line in str(text or "").splitlines() if line.strip()]
        return self.import_actions(records)

    def snapshot(self):
        if self.store is not None:
            self.store.rotate(self.replica.state._map)

    def close(self):
        if self.store is not None:
            self.store.close()

    # -- verification --------------------------------------------------------

    def verify_against(self, store, *, namespace=None):
        """Compare the mirrored view with the database. Reports, never mutates."""
        ns = namespace or self.namespace
        expected = {m.id: m for m in store.list(state=ACTIVE, namespace=ns, limit=100000)}
        mirrored = self.view()
        missing = sorted(set(expected) - set(mirrored))
        stale = sorted(set(mirrored) - set(expected))
        differing = sorted(mid for mid, record in mirrored.items()
                           if mid in expected and record.get("body") != expected[mid].body)
        return {"ok": not (missing or stale or differing), "missing": missing, "stale": stale,
                "differing": differing, "mirrored": len(mirrored), "in_database": len(expected)}


def replay_memories(actions, agent_id="replay", namespace="default"):
    """Rebuild a mirrored view from a raw action log — order never matters."""
    fabric = MemoryFabric(agent_id, namespace=namespace)
    fabric.import_actions(actions)
    return fabric
