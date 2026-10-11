import json
import random

import pytest

from conflux.memory.bridge import BRIDGE_VERSION, MemoryFabric, memory_key, replay_memories
from conflux.memory.fixtures import load_samples
from conflux.memory.store import ACTIVE, MemoryStore


@pytest.fixture
def store(tmp_path):
    handle = MemoryStore(tmp_path / "m.db")
    load_samples(handle)
    yield handle
    handle.close()


def test_projection_is_one_register_per_memory(store):
    fabric = MemoryFabric("agent-a", namespace="default")
    written = store.list()[0]
    payload = fabric.put(written)
    assert payload["id"] == written.id
    assert fabric.keys() == [memory_key("default", written.id)]
    record = fabric.read(written.id)
    assert record["body"] == written.body
    assert record["state"] == ACTIVE and record["revision"] == written.revision
    assert record["fingerprint"] == written.fingerprint
    assert record["v"] == BRIDGE_VERSION
    # the key is namespaced, so two namespaces never collide
    other = MemoryFabric("agent-a", namespace="team-b")
    assert other.keys() == []
    other.put(written)
    assert other.keys() == [memory_key("team-b", written.id)]
    assert other.read(written.id)["body"] == written.body


def test_sync_store_reports_what_moved_and_is_free_to_repeat(store):
    fabric = MemoryFabric("agent-a", namespace="default")
    report = fabric.sync_store(store)
    assert report == {"scanned": 20, "mirrored": 20, "unchanged": 0, "skipped_older": 0,
                      "hash": fabric.hash()}
    assert len(fabric.view()) == 20
    again = fabric.sync_store(store)
    # nothing new to say: the journal is not padded with repeat actions
    assert again == {"scanned": 20, "mirrored": 0, "unchanged": 20, "skipped_older": 0,
                     "hash": fabric.hash()}
    assert len(fabric.replica.journal) == 20
    forced = fabric.sync_store(store, force=True)
    assert forced["mirrored"] == 20 and len(fabric.replica.journal) == 40


def test_retiring_a_memory_takes_it_out_of_the_active_view(store):
    fabric = MemoryFabric("agent-a", namespace="default")
    fabric.sync_store(store)
    target = store.list()[0]
    fabric.retire(target.id, "archived", body=target.body, revision=target.revision + 1)
    view = fabric.view()
    assert target.id not in view
    assert len(view) == 19
    assert fabric.read(target.id)["state"] == "archived"  # nothing is deleted
    assert len(fabric.view(active_only=False)) == 20


def test_the_newest_revision_wins_whatever_the_arrival_order(store):
    a, b = MemoryFabric("agent-a"), MemoryFabric("agent-b")
    base = {"id": "m1", "kind": "note", "topic": "t", "fingerprint": "fp", "state": ACTIVE,
            "importance": 1.0, "confidence": None, "created_at": "2026-01-01T00:00:00+00:00",
            "provenance": {}, "v": BRIDGE_VERSION}
    a.put({**base, "body": "first draft", "revision": 1, "updated_at": "2026-01-01T00:00:00+00:00"})
    b.put({**base, "body": "second draft", "revision": 2, "updated_at": "2026-01-02T00:00:00+00:00"})
    # whichever direction the gossip travels, revision 2 is the winner
    for first, second in ((a, b), (b, a)):
        second.reconcile(first)
        assert second.read("m1")["body"] == "second draft", second.read("m1")
    a.reconcile(b)
    b.reconcile(a)
    assert a.hash() == b.hash()
    assert a.view() == b.view()


def test_concurrent_editors_merge_without_losing_the_other_key(store):
    a, b = MemoryFabric("agent-a"), MemoryFabric("agent-b")
    rows = store.list()[:6]
    for row in rows[:4]:
        a.put(row)
    for row in rows[2:6]:
        b.put(row)
    a.reconcile(b)
    assert sorted(a.view()) == sorted(b.view())
    assert len(a.view()) == 6  # the overlap is not duplicated
    b.reconcile(a)
    assert a.hash() == b.hash()


def test_a_gossip_flood_converges_for_every_replica(store):
    rows = store.list()[:8]
    fabrics = [MemoryFabric(f"agent-{index}") for index in range(5)]
    for index, fabric in enumerate(fabrics):
        for row in rows[index:index + 3]:
            fabric.put(row)
    rng = random.Random(7)
    order = list(range(5))
    for _ in range(40):
        rng.shuffle(order)
        for left, right in zip(order, order[1:]):
            fabrics[left].reconcile(fabrics[right])
    hashes = {f.hash() for f in fabrics}
    assert len(hashes) == 1, hashes
    # rows[0:3], rows[1:4] ... rows[4:7] - row 7 is in no shard, and that is the
    # point: gossip only ever shows you what someone actually wrote
    assert len(fabrics[0].view()) == 7


def test_reconcile_reports_what_each_side_took(store):
    a, b = MemoryFabric("agent-a"), MemoryFabric("agent-b")
    rows = store.list()[:3]
    a.put(rows[0])
    b.put(rows[1])
    b.put(rows[2])
    result = b.reconcile(a)
    assert result == {"received_by_us": 1, "received_by_peer": 2}
    assert len(b.view()) == 3


def test_a_journal_round_trip_rebuilds_the_same_view(store):
    source = MemoryFabric("agent-a", namespace="default")
    source.sync_store(store)
    text = source.export_journal()
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 20
    assert all(json.loads(line)["key"].startswith("memory/default/") for line in lines)

    rebuilt = MemoryFabric("agent-c", namespace="default")
    assert rebuilt.import_journal(text) == 20
    assert rebuilt.hash() == source.hash()
    assert rebuilt.view() == source.view()
    # importing twice changes nothing
    assert rebuilt.import_journal(text) == 0
    assert rebuilt.hash() == source.hash()


def test_imported_actions_from_another_namespace_are_ignored(store):
    other = MemoryFabric("agent-a", namespace="team-b")
    other.sync_store(store, namespace="default")
    target = MemoryFabric("agent-b", namespace="default")
    assert target.import_actions(other.export_actions()) == 0
    assert target.view() == {}


def test_replay_is_order_independent(store):
    fabric = MemoryFabric("agent-a")
    fabric.sync_store(store)
    actions = fabric.export_actions()
    shuffled = list(actions)
    random.Random(11).shuffle(shuffled)
    replayed = replay_memories(shuffled, agent_id="replay", namespace="default")
    assert replayed.hash() == fabric.hash()
    assert replayed.view() == fabric.view()


def test_a_persisted_fabric_survives_a_restart_and_snapshots(tmp_path, store):
    data_dir = tmp_path / "fabric"
    fabric = MemoryFabric("agent-a", data_dir=str(data_dir), snapshot_threshold=5)
    assert fabric.sync_store(store)["mirrored"] == 20  # snapshots rotate well before closing
    target = store.list()[0]
    fabric.retire(target.id, "pruned", body=target.body, revision=2)
    expected_hash, expected_view = fabric.hash(), fabric.view()
    fabric.close()
    assert any(path.suffix in (".jsonl", ".json", ".snapshot") or path.stat().st_size
               for path in data_dir.glob("**/*") if path.is_file())

    reopened = MemoryFabric("agent-a", data_dir=str(data_dir))
    # snapshot plus journal replays to exactly the same state, not a fresh start
    assert reopened.hash() == expected_hash
    assert reopened.view() == expected_view
    assert target.id not in reopened.view()
    reopened.close()

    latecomer = MemoryFabric("agent-b", data_dir=str(tmp_path / "fabric-b"))
    assert latecomer.sync_store(store)["mirrored"] == 20
    assert target.id in latecomer.view()  # it never saw the retire, and that is honest
    latecomer.close()


def test_verification_catches_every_kind_of_drift(store, tmp_path):
    fabric = MemoryFabric("agent-a")
    fabric.sync_store(store)
    assert fabric.verify_against(store) == {"ok": True, "missing": [], "stale": [], "differing": [],
                                           "mirrored": 20, "in_database": 20}
    # a memory retired in the database but never mirrored
    retired = store.list()[0]
    store.set_state(retired.id, "archived", actor="test")
    check = fabric.verify_against(store)
    assert check["ok"] is False and retired.id in check["stale"]
    # a memory added to the database that the fabric has never seen
    added = store.add("Brand new fact nobody gossiped.", topic="drift")["id"]
    check = fabric.verify_against(store)
    assert check["missing"] == [added]
    fabric.put(store.get(added))
    fabric.retire(retired.id, "archived", body=retired.body, revision=retired.revision)
    assert fabric.verify_against(store)["ok"] is True


def test_the_fabric_tracks_a_cleanup_run(store, tmp_path):
    from conflux.memory.cleanup import CleanupEngine
    from conflux.memory.policy import CleanupPolicy

    fabric = MemoryFabric("agent-a")
    fabric.sync_store(store)
    before = fabric.view()
    pair_a = store.add("The edge proxy is envoy.", topic="edge", on_duplicate="insert")["id"]
    pair_b = store.add("the edge proxy is envoy", topic="edge", on_duplicate="insert")["id"]
    report = CleanupEngine(store, policy=CleanupPolicy(model="heuristic", min_active_memories=0,
                                                        max_merge_fraction=1.0,
                                                        max_prune_fraction=1.0)).run(dry_run=False)
    assert report.outcome.merges_applied
    fabric.sync_store(store)
    view = fabric.view()
    assert pair_a not in view and pair_b not in view  # folded away in the shared view too
    assert len(view) == len(before) + 1
    assert store.get(pair_a).state == "merged"
    # and the mirror agrees with the database
    assert fabric.verify_against(store)["ok"] is True


def test_payload_for_accepts_rows_and_dicts_and_needs_an_id(store):
    row = store.list()[0]
    payload = MemoryFabric.payload_for(row)
    assert payload["id"] == row.id and payload["body"] == row.body
    from_dict = MemoryFabric.payload_for({"id": "x", "body": "text"})
    assert from_dict["kind"] == "note" and from_dict["state"] == ACTIVE and from_dict["revision"] == 1
    fabric = MemoryFabric("agent-a")
    with pytest.raises(ValueError, match="needs a memory id"):
        fabric.put({"body": "no id"})
    with pytest.raises(ValueError, match="use put()"):
        fabric.retire("m1", ACTIVE)
    assert fabric.keys() == []  # nothing half-written


def test_an_older_revision_is_not_republished_over_a_newer_one(store):
    row = store.list()[0]
    fabric = MemoryFabric("agent-a")
    assert fabric.put(row)["mirrored"] is True
    stale = dict(MemoryFabric.payload_for(row), body="stale rewrite", revision=row.revision - 1)
    refused = fabric.put(stale)
    assert refused["mirrored"] is False and "older" in refused["reason"]
    assert fabric.put(row)["reason"] == "unchanged"
    assert fabric.read(row.id)["body"] == row.body  # the register is untouched
    fresh = dict(MemoryFabric.payload_for(row), body="fresh rewrite", revision=row.revision + 1)
    assert fabric.put(fresh)["mirrored"] is True
    assert fabric.read(row.id)["body"] == "fresh rewrite"
    # force is the escape hatch for a manual correction
    assert fabric.put(dict(MemoryFabric.payload_for(row), body="manual override",
                          revision=row.revision), force=True)["mirrored"] is True
    assert fabric.read(row.id)["body"] == "manual override"


def test_revision_is_never_invented_for_a_bare_payload():
    assert MemoryFabric.payload_for({"id": "m"} )["revision"] == 1
    assert MemoryFabric.payload_for({"id": "m", "revision": 0})["revision"] == 0
    assert MemoryFabric.payload_for({"id": "m", "revision": "7"})["revision"] == 7
