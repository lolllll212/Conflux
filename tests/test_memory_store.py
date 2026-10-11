import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from conflux.memory.policy import CleanupPolicy
from conflux.memory.schema import MIGRATIONS, SCHEMA_VERSION, apply_migrations, user_version
from conflux.memory.similar import hash_embed
from conflux.memory.store import (ACTIVE, DuplicateMemory, MemoryNotFound, MemoryStore,
                                  age_days, iso, parse_ts, utcnow)


# --- schema and migrations ---------------------------------------------------


def test_open_creates_current_schema(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.schema_version == SCHEMA_VERSION
    assert store.integrity_check() == ["ok"]
    tables = {r["name"] for r in store.query(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"memories", "memory_events", "merges", "merge_links", "runs", "settings",
            "run_lock", "review_cursor", "reviews", "scheduler_state"} <= tables
    store.close()


def test_migrations_are_incremental_and_preserve_data(tmp_path):
    path = str(tmp_path / "m.db")
    conn = sqlite3.connect(path, isolation_level=None)
    apply_migrations(conn, target_version=SCHEMA_VERSION - 1)
    assert user_version(conn) == SCHEMA_VERSION - 1
    conn.execute("INSERT INTO memories (id, namespace, body, body_norm, fingerprint, created_at, "
                 "updated_at) VALUES ('seed', 'default', 'kept across the upgrade', "
                 "'kept across the upgrade', 'fp', '2026-01-01T00:00:00+00:00', "
                 "'2026-01-01T00:00:00+00:00')")
    conn.close()

    store = MemoryStore(path)  # opens and upgrades the older file
    assert store.schema_version == SCHEMA_VERSION
    assert store.get("seed").body == "kept across the upgrade"
    # v2 objects exist only after the upgrade step
    assert store.query("SELECT * FROM review_cursor") == []
    store.close()


def test_migration_steps_are_each_a_transaction(tmp_path):
    # every migration is applied in its own BEGIN IMMEDIATE, and the runner is a
    # no-op when the database is already current
    store = MemoryStore(tmp_path / "m.db")
    assert apply_migrations(store._conn) == []
    assert len(MIGRATIONS) >= 2
    store.close()


def test_unknown_schema_version_is_refused(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "m.db"), isolation_level=None)
    with pytest.raises(ValueError, match="unknown schema version"):
        apply_migrations(conn, target_version=99)
    conn.close()


# --- writes, idempotency, provenance ----------------------------------------


def test_add_and_get_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    receipt = store.add("Deploy the api before the dashboard.", kind="runbook", topic="deploy",
                        source_agent="agent-a", uri="chat://room/12")
    assert receipt["created"] is True
    memory = store.get(receipt["id"])
    assert memory.body == "Deploy the api before the dashboard."
    assert memory.body_norm == "deploy the api before the dashboard"
    assert memory.state == ACTIVE
    assert memory.revision == 1
    assert memory.provenance["agent"] == "agent-a"
    assert memory.provenance["uri"] == "chat://room/12"
    store.close()


def test_ttl_days_is_the_relative_form_of_an_expiry(tmp_path):
    """One rule for "valid for N days", so CLI, HTTP and library cannot drift.

    Expiry is the only staleness rule that can retire a memory on its own, so it
    has to be settable without computing a timestamp - and a zero or negative
    TTL must be refused instead of storing something already retired.
    """
    store = MemoryStore(tmp_path / "m.db")
    added = store.add("cache TTL is 30s", kind="fact", topic="cache", ttl_days=1)
    expiry = datetime.fromisoformat(store.get(added["id"]).expires_at)
    assert abs((expiry - (datetime.now(timezone.utc) + timedelta(days=1))).total_seconds()) < 60

    for bad in (0, -2, "soon"):
        with pytest.raises(ValueError, match="ttl_days"):
            store.add("nope", kind="fact", ttl_days=bad)
    with pytest.raises(ValueError, match="not both"):
        store.add("nope", kind="fact", ttl_days=1,
                  expires_at="2030-01-01T00:00:00+00:00")
    # nothing above wrote a row
    assert store.counts()["active"] == 1
    store.close()


def test_exact_duplicate_touch_is_idempotent(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    first = store.add("Staging box is 10.0.0.5.", source_agent="agent-a")
    again = store.add("  STAGING BOX IS 10.0.0.5! ", source_agent="agent-b")
    assert again["created"] is False
    assert again["reason"] == "exact_duplicate"
    assert again["id"] == first["id"]
    memory = store.get(first["id"])
    assert memory.access_count == 2
    # revision does NOT move: a duplicate touch must not invalidate open plans
    assert memory.revision == 1
    # ...and the second speaker is recorded rather than dropped
    assert memory.provenance["also_reported_by"] == ["agent-b"]
    assert store.counts()["active"] == 1
    store.close()


def test_on_duplicate_insert_and_error(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.add("same text", on_duplicate="insert")
    assert store.add("same text", on_duplicate="insert")["created"] is True
    assert store.counts()["active"] == 2
    with pytest.raises(DuplicateMemory):
        store.add("same text", on_duplicate="error")
    store.close()


def test_idempotency_key_replays_without_new_rows(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    first = store.add("retry budget is 3 attempts", idempotency_key="job-77")
    replay = store.add("retry budget is 3 attempts", idempotency_key="job-77")
    assert replay["created"] is False and replay["reason"] == "idempotency_key"
    assert replay["id"] == first["id"]
    assert store.counts()["active"] == 1
    # a different body under the same key is also a no-op, never an overwrite
    clash = store.add("totally different", idempotency_key="job-77")
    assert clash["created"] is False
    assert store.get(first["id"]).body == "retry budget is 3 attempts"
    store.close()


def test_empty_and_oversized_bodies_are_refused(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    with pytest.raises(ValueError, match="must not be empty"):
        store.add("   ")
    with pytest.raises(ValueError, match="too long"):
        store.add("x" * 100_001)
    store.close()


def test_kind_is_part_of_the_fingerprint(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    note = store.add("rotate the token", kind="note")
    policy = store.add("rotate the token", kind="policy")
    assert note["id"] != policy["id"]
    assert store.counts()["active"] == 2
    store.close()


def test_embedding_blob_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    vector = list(hash_embed("some text to embed", dim=64))
    receipt = store.add("some text to embed", embedding=vector, embedding_model="test-64")
    memory = store.get(receipt["id"], with_vector=True)
    assert memory.embedding_dim == 64
    assert len(memory.vector) == 64
    assert abs(sum(memory.vector) - sum(vector)) < 1e-5
    store.close()


# --- transactions ------------------------------------------------------------


def test_transaction_rolls_back_every_row_including_events(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    before = store.counts()
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.add("doomed record", source_agent="x")
            store.event("doomed-note", memory_id="none")
            raise RuntimeError("boom")
    assert store.counts() == before
    assert store.query("SELECT * FROM memory_events WHERE action='add'") == []
    assert store.list() == []
    store.close()


def test_nested_transaction_uses_savepoints(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.add("outer survives? no", memory_id="outer")
            try:
                with store.transaction():
                    store.add("inner dies", memory_id="inner")
                    raise RuntimeError("inner boom")
            except RuntimeError:
                pass
            raise RuntimeError("outer boom")
    assert store.query("SELECT id FROM memories") == []
    store.close()


def test_nested_savepoint_partial_rollback_keeps_the_rest(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    with store.transaction():
        store.add("kept", memory_id="keep-me")
        try:
            with store.transaction():
                store.add("dropped", memory_id="drop-me")
                raise RuntimeError("rollback the inner unit only")
        except RuntimeError:
            pass
    ids = {r["id"] for r in store.query("SELECT id FROM memories")}
    assert ids == {"keep-me"}
    store.close()


def test_concurrent_writers_are_serialized(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    errors = []

    def writer(worker):
        try:
            for index in range(20):
                store.add(f"worker {worker} note {index}", topic=f"t{worker}")
        except Exception as exc:  # pragma: no cover - would be a real bug
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert store.counts()["active"] == 80
    assert store.integrity_check() == ["ok"]
    store.close()


# --- audit trail -------------------------------------------------------------


def test_audit_trail_records_provenance_for_every_transition(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    receipt = store.add("the api is rate limited to 100 rpm", topic="limits")
    store.touch_body(receipt["id"], body="the api is rate limited to 120 rpm")
    store.set_state(receipt["id"], "archived", run_id="run-1", actor="cleanup:test")

    events = store.timeline(receipt["id"])
    assert [e["action"] for e in events] == ["add", "update", "state:archived"]
    assert events[0]["after"]["body"].endswith("100 rpm")
    assert events[1]["before"]["body"].endswith("100 rpm")
    assert events[1]["after"]["body"].endswith("120 rpm")
    assert events[2]["detail"]["to"] == "archived"
    assert events[2]["before"]["state"] == "active"
    # the record itself is still there, just retired - never deleted
    assert store.get(receipt["id"]).state == "archived"
    store.close()


def test_state_transitions_are_compare_and_set(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    receipt = store.add("revision guard", topic="guards")
    assert store.set_state(receipt["id"], "archived", expect_revision=99) is False
    assert store.get(receipt["id"]).state == ACTIVE
    assert store.set_state(receipt["id"], "archived", expect_revision=1) is True
    assert store.get(receipt["id"]).state == "archived"
    with pytest.raises(ValueError, match="unknown memory state"):
        store.set_state(receipt["id"], "deleted")
    store.close()


def test_content_update_bumps_revision(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    receipt = store.add("before edit", topic="t")
    assert store.get(receipt["id"]).revision == 1
    store.touch_body(receipt["id"], body="after edit")
    assert store.get(receipt["id"]).revision == 2
    with pytest.raises(MemoryNotFound):
        store.get("nope")
    store.close()


def test_hard_delete_requires_cutoff_and_keeps_the_record_in_the_trail(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    receipt = store.add("gone from the table, not from history", on_duplicate="insert")
    store.set_state(receipt["id"], "pruned")
    with pytest.raises(ValueError, match="cutoff"):
        store.delete_rows(state="pruned")
    removed = store.delete_rows(state="pruned", older_than=iso(utcnow() + timedelta(minutes=1)))
    assert removed == 1
    assert store.query("SELECT id FROM memories") == []
    deleted = store.events(action="hard_delete")
    assert len(deleted) == 1
    assert deleted[0]["before"]["body"] == "gone from the table, not from history"
    store.close()


# --- run lock ----------------------------------------------------------------


def test_run_lock_is_exclusive_and_expires(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.try_lock(owner="a")["acquired"] is True
    busy = store.try_lock(owner="b")
    assert busy["acquired"] is False and busy["owner"] == "a"
    assert store.lock_info()["held"] is True
    assert store.unlock(owner="a") is True
    assert store.try_lock(owner="b", lease_seconds=-1)["acquired"] is True  # already expired lease
    # a dead holder cannot wedge the database forever
    assert store.try_lock(owner="c")["acquired"] is True
    store.close()


def test_run_ledger_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    run_id = store.create_run(trigger="manual", mode="dry_run", policy={"a": 1}, model="null")
    store.update_run(run_id, status="completed", stats={"considered": 3}, plan_hash="deadbeef",
                      plan_json={"merges": []}, finished=True)
    run = store.get_run(run_id)
    assert run["status"] == "completed" and run["stats"]["considered"] == 3
    assert run["plan"] == {"merges": []} and run["policy"] == {"a": 1}
    assert store.find_completed_plan("deadbeef") is None  # only 'apply' mode counts
    store._execute("UPDATE runs SET mode='apply' WHERE id=?", (run_id,))
    assert store.find_completed_plan("deadbeef") == run_id
    assert store.list_runs()[0]["id"] == run_id
    store.close()


def test_review_cursor_skips_unchanged_records(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    receipt = store.add("already judged", topic="cursor")
    memory = store.get(receipt["id"])
    assert store.reviewed_fingerprints([memory.id]) == {}
    store.record_review(memory.id, memory.fingerprint, run_id="r1", verdict="retained")
    assert store.reviewed_fingerprints([memory.id]) == {memory.id: memory.fingerprint}
    store.touch_body(memory.id, body="content changed")
    # the cursor is keyed on the fingerprint, so edited content is judged again
    assert store.query_one("SELECT verdict FROM review_cursor WHERE memory_id=?",
                           (memory.id,))["verdict"] == "retained"
    assert store.clear_review_cursor(namespace="default") == 1
    store.close()


def test_policy_settings_roundtrip_and_namespaces_are_isolated(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.get_policy().duplicate_similarity == CleanupPolicy().duplicate_similarity
    store.set_policy({"duplicate_similarity": 0.61, "protect_kinds": ["policy"]}, updated_by="me")
    policy = store.get_policy()
    assert policy.duplicate_similarity == 0.61 and policy.protect_kinds == ("policy",)
    stored = store.query_one("SELECT updated_by FROM settings")
    assert stored["updated_by"] == "me"

    store.add("namespaced fact", namespace="other")
    assert store.counts(namespace="other")["active"] == 1
    assert store.counts()["active"] == 0
    with pytest.raises(MemoryNotFound):
        store.get(store.list(namespace="other")[0].id)
    store.close()


def test_timestamp_helpers():
    assert iso(None) is None
    parsed = parse_ts("2026-03-01T12:00:00Z")
    assert parsed.isoformat().startswith("2026-03-01T12:00:00+00:00")
    assert parse_ts(parsed) == parsed
    with pytest.raises(ValueError, match="invalid timestamp"):
        parse_ts("not a date")
    assert age_days(None) is None
    assert round(age_days(iso(utcnow() - timedelta(days=1)))) == 1
    store_time = json.dumps({"t": iso(parse_ts(1_700_000_000))})
    assert "2023-11-14" in store_time


def test_store_rejects_unknown_merge_and_review(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    with pytest.raises(MemoryNotFound):
        store.get_merge(4242)
    with pytest.raises(MemoryNotFound):
        store.resolve_review(4242, "approved")
    with pytest.raises(ValueError, match="decision must be"):
        store.resolve_review(1, "maybe")
    store.close()


def test_open_without_create_requires_the_file(tmp_path):
    missing = str(tmp_path / "absent.db")
    with pytest.raises(FileNotFoundError):
        MemoryStore(missing, create=False)
    store = MemoryStore(missing, create=True)
    store.close()
    assert MemoryStore(missing, create=False).path == missing


def test_review_queue_dedupes_equivalent_items(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    receipt = store.add("uncertain thing", topic="review")
    first = store.open_review(kind="prune", memory_ids=[receipt["id"]], reason="because", run_id="r1")
    assert store.find_open_review("prune", [receipt["id"]])["id"] == first
    # A review is keyed per run, so a later run can record its own view of the
    # same disagreement; the engine dedupes by calling find_open_review() first.
    again = store.open_review(kind="prune", memory_ids=[receipt["id"]], reason="because", run_id="r2")
    assert again != first
    assert len(store.reviews(state="open")) == 2
    resolved = store.resolve_review(first, "rejected", actor="ops", note="nope")
    assert resolved["state"] == "rejected" and resolved["changed"] is True
    assert store.resolve_review(first, "rejected")["changed"] is False  # already decided
    store.close()
