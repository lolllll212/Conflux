import threading
from datetime import timedelta

import pytest

from conflux.memory.cleanup import CleanupEngine, PlanStale
from conflux.memory.fixtures import load_samples, sample_dataset
from conflux.memory.policy import CleanupPolicy
from conflux.memory.service import CleanupService
from conflux.memory.store import (ACTIVE, MemoryNotFound, MemoryStore, RunInProgress, iso, utcnow)

OLD = "2020-01-01T00:00:00+00:00"

# Deliberately unrelated bodies: near-identical rows would cluster and merge, which
# hides the counting these tests are about.
UNRELATED = [
    "The deploy window is Tuesday afternoons.",
    "Redis needs an allkeys-lru eviction policy.",
    "Grafana alerts go to the ops channel.",
    "The certificate authority rotates every 90 days.",
    "Backups are verified by a nightly restore.",
    "The on-call handover happens at 17:00.",
    "Terraform state lives in the shared bucket.",
    "The load balancer health check hits /healthz.",
    "Log retention is capped at fourteen days.",
    "The staging VPS reboots on the first Sunday.",
]


@pytest.fixture
def store(tmp_path):
    handle = MemoryStore(tmp_path / "m.db")
    yield handle
    handle.close()


def engine_for(store, **overrides):
    # Floors are switched off here so each test exercises one mechanism; the guard
    # tests below turn them back on and assert the exact capping.
    settings = {"model": "heuristic", "min_active_memories": 0, "max_prune_fraction": 1.0,
                "max_merge_fraction": 1.0}
    settings.update(overrides)
    return CleanupEngine(store, policy=CleanupPolicy(**settings))


def aged(store, body, *, day_offset=0, **kwargs):
    stamp = (utcnow() + timedelta(days=day_offset)).isoformat(timespec="microseconds")
    kwargs.setdefault("created_at", stamp)
    kwargs.setdefault("updated_at", stamp)
    kwargs.setdefault("last_seen_at", stamp)
    return store.add(body, **kwargs)


def memory_rows(store):
    return store.query("SELECT id, body, state, revision, superseded_by, merge_id FROM memories "
                       "ORDER BY id")


# --- duplicate detection and merging -----------------------------------------


def test_exact_duplicates_are_folded_with_full_provenance(store):
    a = store.add("The staging box is at 10.0.0.5.", kind="fact", topic="hosts",
                  source_agent="agent-a", on_duplicate="insert")["id"]
    b = store.add("the staging box is at 10.0.0.5", kind="fact", topic="hosts",
                  source_agent="agent-b", uri="chat://room/9", on_duplicate="insert")["id"]
    report = engine_for(store).run(dry_run=False)
    assert report.outcome.merges_applied, report.outcome
    merged_id = store.get(a).superseded_by
    assert merged_id == store.get(b).superseded_by
    assert merged_id not in (a, b) or store.get(merged_id).metadata["merged_from"]
    survivor = store.get(merged_id)
    assert survivor.state == ACTIVE
    assert sorted(survivor.metadata["merged_from"]) == sorted([a, b])
    # provenance survives the fold: both sources are named on the new record
    assert sorted(survivor.provenance["origin_ids"]) == sorted([a, b])
    assert survivor.kind == "fact" and survivor.topic == "hosts"
    # and the originals are still readable, only retired
    assert store.get(a).state == "merged"
    assert store.get(b).state == "merged"
    # the originals keep their own text, so nothing was rewritten in place
    assert store.get(a).body == "The staging box is at 10.0.0.5."
    assert store.get(b).body == "the staging box is at 10.0.0.5"


def test_merge_proposal_keeps_source_bodies_and_revisions(store):
    for index, body in enumerate(["Marta owns billing.", "Marta owns the billing service."]):
        store.add(body, topic="owners", kind="fact", memory_id=f"src{index}", on_duplicate="insert")
    plan = engine_for(store).plan(dry_run=True)
    assert len(plan.merges) == 1
    merge = store.get_merge(plan.merges[0].merge_id)
    assert merge["state"] in ("proposed", "needs_review")
    bodies = sorted(source["body"] for source in merge["sources"])
    assert bodies == sorted(["Marta owns billing.", "Marta owns the billing service."])
    assert {source["id"] for source in merge["sources"]} == {"src0", "src1"}
    assert {source["revision"] for source in merge["sources"]} == {1}
    assert [link["memory_id"] for link in merge["links"]] == ["src0", "src1"]


def test_related_but_distinct_memories_are_proposed_not_applied(store):
    # the second row is a subset of the first: related, not provably identical
    for index, body in enumerate(["Always restart nginx after a deploy. Nginx needs 2 workers.",
                                  "Always restart nginx after a deploy."]):
        store.add(body, topic="deploy", kind="runbook", memory_id=f"r{index}", on_duplicate="insert")
    plan = engine_for(store).plan(dry_run=True)
    assert [m.decision for m in plan.merges] == ["review"]
    assert plan.merges[0].conflict is False
    assert plan.merges[0].similarity < 1.0
    report = engine_for(store).run(dry_run=False)
    assert report.outcome.merges_applied == ()
    assert store.counts()["active"] == 2
    assert len(store.reviews(state="open")) == 1


def test_contradictions_are_parked_for_review_and_both_survive(store):
    store.add("Session timeout is 30 minutes.", topic="limits", kind="policy", on_duplicate="insert")
    store.add("Session timeout is 45 minutes.", topic="limits", kind="policy", on_duplicate="insert")
    plan = engine_for(store).plan(dry_run=True)
    assert plan.clusters[0].conflicts is True
    assert [m.decision for m in plan.merges] == ["review"]
    assert [r.kind for r in plan.reviews] == ["contradiction"]
    engine_for(store).run(dry_run=False)
    assert {m.state for m in store.list(state=None)} == {ACTIVE, ACTIVE}
    review = store.reviews(state="open")[0]
    assert review["kind"] == "contradiction"
    assert "contradicts" in review["reason"] or "review" in review["reason"]


def test_approving_a_review_applies_the_merge(store):
    a = store.add("The api rate limit is 100 rpm.", topic="limits", on_duplicate="insert")["id"]
    b = store.add("the api rate limit is 100 rpm", topic="limits", on_duplicate="insert")["id"]
    policy = CleanupPolicy(model="heuristic", auto_apply_merges=False, auto_merge_exact=False)
    engine = CleanupEngine(store, policy=policy)
    engine.run(dry_run=True)
    review = store.reviews(state="open")[0]
    assert review["kind"] == "merge"
    result = engine.approve_review(review["id"], actor="ops")
    assert result["applied"], result
    assert store.get(a).state == "merged" and store.get(b).state == "merged"
    assert store.get(a).superseded_by == store.get(b).superseded_by
    assert store.reviews(state="open") == []


def test_rejecting_a_review_keeps_both_memories_and_the_proposal(tmp_path):
    service = CleanupService(tmp_path / "svc.db",
                            policy={"model": "heuristic", "auto_apply_merges": False,
                                    "auto_merge_exact": False})
    service.store.add("Deploy with make release.", topic="deploy", on_duplicate="insert")
    service.store.add("Deploy with make release", topic="deploy", on_duplicate="insert")
    service.run(dry_run=True)
    review = service.store.reviews(state="open")[0]
    result = service.review(review["id"], "rejected", actor="ops", note="both are useful")
    assert result["state"] == "rejected"
    assert service.store.counts()["active"] == 2
    assert service.store.merges(state="rejected"), service.store.merges(state="all")
    assert service.store.reviews(state="open") == []
    service.store.close()


def test_reverting_a_merge_restores_every_source(store):
    ids = [store.add("The build takes 12 minutes.", topic="build", memory_id=f"v{i}",
                     on_duplicate="insert")["id"] for i in range(2)]
    report = engine_for(store, auto_merge_exact=True).run(dry_run=False)
    merge_id = report.outcome.merges_applied[0]
    merged_id = store.get(ids[0]).superseded_by
    result = store.revert_merge(merge_id, actor="ops", note="bad merge")
    assert result["reverted"] is True
    assert sorted(result["restored"]) == sorted(ids)
    for memory_id in ids:
        assert store.get(memory_id).state == ACTIVE
        assert store.get(memory_id).superseded_by is None
    assert store.get(merged_id).state == "superseded"
    assert store.get_merge(merge_id)["state"] == "reverted"


# --- staleness and retention --------------------------------------------------


def test_stale_memories_are_only_pruned_when_the_policy_allows_it(store):
    aged(store, "We used bower for assets.", topic="tooling", day_offset=-400, kind="note")
    engine = engine_for(store, stale_after_days=30, allow_auto_prune=False)
    plan = engine.plan(dry_run=True)
    assert [p.decision for p in plan.prunes] == ["review"]
    assert store.counts()["active"] == 1
    report = engine.run(dry_run=False)
    assert report.outcome.prunes_applied == ()
    assert store.counts()["active"] == 1
    # and the reason is visible in the queue, not swallowed
    assert store.reviews(state="open")[0]["kind"] == "prune"


def test_auto_prune_archives_rather_than_deletes(store):
    memory_id = aged(store, "Try Redis 7 for the cache spike.", topic="experiments",
                     day_offset=-400, kind="note", provenance={"agent": "a"})["id"]
    report = engine_for(store, stale_after_days=30, allow_auto_prune=True).run(dry_run=False)
    assert report.outcome.prunes_applied == (memory_id,)
    retired = store.get(memory_id)
    assert retired.state == "archived"
    assert retired.revision == 2
    rows = store.query("SELECT COUNT(*) AS n FROM memories")
    assert rows[0]["n"] == 1  # still on disk
    # restoring is one call
    assert store.set_state(memory_id, ACTIVE, actor="ops", action="restore") is True
    assert store.get(memory_id).state == ACTIVE


def test_expired_ttl_is_pruned_even_without_age(store):
    store.add("Short-lived experiment note.", topic="experiments",
              expires_at=iso(utcnow() - timedelta(days=3)), provenance={"agent": "a"})
    # a brand-new row sits inside the protect_recent_days window, so the default
    # pass does not even propose retiring it
    careful = engine_for(store, allow_auto_prune=True, archive_instead_of_prune=False)
    assert careful.plan(dry_run=True).prunes == ()
    # ...and an expired TTL outranks it once the caller clears the recency window
    report = engine_for(store, allow_auto_prune=True, archive_instead_of_prune=False,
                        protect_recent_days=0).run(dry_run=False)
    assert len(report.outcome.prunes_applied) == 1
    assert store.list(state=None)[0].state == "pruned"


def test_protected_kinds_survive_an_aggressive_pass(store):
    aged(store, "Never commit .env files.", day_offset=-900, kind="policy",
         provenance={"agent": "a"})
    aged(store, "Session notes from the old vendor.", day_offset=-900, kind="note",
         provenance={"agent": "a"})
    report = engine_for(store, stale_after_days=1, allow_auto_prune=True,
                        protect_kinds=("policy",)).run(dry_run=False)
    assert len(report.outcome.prunes_applied) == 1
    kept = [m for m in store.list(state=None) if m.state == ACTIVE]
    assert [m.kind for m in kept] == ["policy"]


# --- safety floors ------------------------------------------------------------


def test_a_pass_never_removes_everything(store):
    for index in range(6):
        aged(store, UNRELATED[index], topic=f"t{index}", day_offset=-500,
             provenance={"agent": "a"})
    engine = engine_for(store, stale_after_days=1, allow_auto_prune=True, max_prune_fraction=1.0,
                        min_active_memories=4)
    report = engine.run(dry_run=False)
    assert len(report.outcome.prunes_applied) == 2  # 6 active - floor of 4
    assert store.counts()["active"] == 4
    assert any("min_active_memories=4" in g for g in report.plan.guards)
    assert "parked for review" in report.plan.guards[0]


def test_max_prune_fraction_caps_one_run(store):
    for index in range(8):
        aged(store, UNRELATED[index], topic=f"t{index}", day_offset=-500,
             provenance={"agent": "a"})
    report = engine_for(store, stale_after_days=1, allow_auto_prune=True,
                        max_prune_fraction=0.25).run(dry_run=False)
    assert len(report.outcome.prunes_applied) == 2  # int(8 * 0.25)
    assert any("max_prune_fraction=0.25" in g for g in report.plan.guards)
    assert store.counts()["active"] == 6
    # the parked ones are visible as reviews, not silently dropped
    assert len(store.reviews(state="open")) == 6
    assert {r["kind"] for r in store.reviews(state="open")} == {"prune"}


def test_topics_are_never_emptied_by_one_pass(store):
    # two unrelated facts that share a topic: retiring both would erase the concept
    aged(store, "Marta handles billing escalations.", topic="billing", day_offset=-500,
         provenance={"agent": "a"})
    aged(store, "The billing schema lives on cluster 3.", topic="billing", day_offset=-500,
         provenance={"agent": "a"})
    report = engine_for(store, stale_after_days=1, allow_auto_prune=True).run(dry_run=False)
    assert len(report.outcome.prunes_applied) == 1
    assert any("topic:billing" in guard for guard in report.plan.guards)
    assert store.counts()["active"] == 1


# --- dry run, transactions, idempotency --------------------------------------


def test_dry_run_leaves_memories_untouched_but_records_the_plan(store):
    for index, body in enumerate(["The registry host is reg.internal.",
                                  "the registry host is reg.internal"]):
        store.add(body, topic="registry", memory_id=f"d{index}", on_duplicate="insert")
    before = memory_rows(store)
    report = engine_for(store).run(dry_run=True)
    assert memory_rows(store) == before
    assert report.stats["merges_proposed"] == 1
    assert report.outcome is None
    run = store.get_run(report.run_id)
    assert run["mode"] == "dry_run" and run["status"] == "completed"
    assert run["plan"]["merges"][0]["source_ids"] == ["d0", "d1"] or \
        sorted(run["plan"]["merges"][0]["source_ids"]) == ["d0", "d1"]
    # the proposal is inspectable and later applicable
    applied_run, outcome = engine_for(store).apply_run(report.run_id)
    assert outcome.merges_applied
    assert memory_rows(store) != before
    assert store.get_run(applied_run)["stats"]["source_run"] == report.run_id


def test_applying_the_same_plan_twice_is_a_no_op(store):
    store.add("Kafka is on port 9092.", topic="brokers", on_duplicate="insert")
    store.add("kafka is on port 9092", topic="brokers", on_duplicate="insert")
    engine = engine_for(store)
    first = engine.run(dry_run=False)
    assert len(first.outcome.merges_applied) == 1
    second = engine.run(dry_run=False)
    assert second.outcome.merges_applied == ()
    assert second.outcome.prunes_applied == ()
    assert store.counts()["active"] == 1


def test_review_cursor_skips_records_that_have_not_changed(store):
    memory_id = aged(store, "Long forgotten fact.", topic="tooling", day_offset=-500,
                     provenance={"agent": "a"})["id"]
    # protect_recent_days=0: editing a row must not make it "fresh" and hide the
    # thing under test, which is whether the verdict is reused
    engine = engine_for(store, stale_after_days=1, allow_auto_prune=True, review_unchanged=True,
                        protect_recent_days=0)
    assert len(engine.run(dry_run=False).outcome.prunes_applied) == 1
    # put the row back exactly as it was, timestamps included, as a restore tool would
    store._execute("UPDATE memories SET state='active', revision=revision+1, created_at=?, "
                   "updated_at=? WHERE id=?", (OLD, OLD, memory_id))
    again = engine.run(dry_run=True)
    # the verdict stands: same content fingerprint, already judged by this policy
    assert again.stats["skipped_unchanged"] == 1
    assert again.stats["prunes_proposed"] == 0
    forced = engine.plan(dry_run=True, force=True)
    assert forced.stats["skipped_unchanged"] == 0
    assert forced.stats["prunes_proposed"] == 1
    # edited content re-enters the judged set without needing a force flag
    before = store.get(memory_id)
    edited = store.touch_body(memory_id, body=before.body + " Verified twice.")
    assert edited.fingerprint != before.fingerprint
    plan = engine.plan(dry_run=True)
    assert plan.stats["skipped_unchanged"] == 0
    assert plan.stats["considered"] == 1


def test_apply_skips_a_cluster_whose_rows_moved(store):
    a = store.add("The gateway listens on 8443.", topic="edge", on_duplicate="insert")["id"]
    store.add("the gateway listens on 8443", topic="edge", on_duplicate="insert")["id"]
    engine = engine_for(store)
    plan = engine.plan(dry_run=True)
    assert len(plan.merges) == 1
    store.touch_body(a, body="The gateway listens on 8443 for mTLS only.")  # revision moves
    outcome = engine.apply(plan, run_id=None)
    assert outcome.merges_applied == ()
    assert any("revision" in note for note in outcome.merges_skipped)
    # nothing half-applied: no source is merged, no orphan merged record exists
    assert {m.state for m in store.list(state=None)} == {ACTIVE}
    assert store.counts()["total"] == 2


def test_a_merge_is_atomic_when_one_source_is_gone(store):
    store.add("Vault is at 10.0.0.9:8200.", topic="secrets", memory_id="keep1", on_duplicate="insert")
    store.add("vault is at 10.0.0.9:8200", topic="secrets", memory_id="keep2", on_duplicate="insert")
    engine = engine_for(store)
    plan = engine.plan(dry_run=True)
    merge = plan.merges[0]
    expected = dict(merge.expected)
    expected["keep1"] = 99  # pretend the row moved
    broken = plan.merges[0].__class__(**{**merge.to_dict(), "expected": tuple(expected.items())})
    with pytest.raises(PlanStale):
        store.apply_merge(broken.merge_id, source_ids=list(broken.source_ids),
                         merged_body=broken.merged_body, expected=expected, run_id="r-x")
    assert {m.state for m in store.list(state=None)} == {ACTIVE}


def test_run_id_is_recorded_and_observable(store):
    for index in range(3):
        store.add(f"note {index}", topic=f"t{index}")
    report = engine_for(store).run(dry_run=True)
    events = store.events(run_id=report.run_id)
    assert [e["action"] for e in events][-1] == "run_completed"
    assert "run_started" in [e["action"] for e in events]
    listed = store.list_runs()
    assert listed[0]["id"] == report.run_id
    assert listed[0]["stats"]["considered"] == 3


# --- fixtures ----------------------------------------------------------------


def test_sample_dataset_covers_the_interesting_cases(tmp_path):
    store = MemoryStore(tmp_path / "sample.db")
    loaded = load_samples(store)
    assert loaded == len(sample_dataset()) == 20
    assert load_samples(store) == 0  # seeding twice is a no-op
    kinds = store.counts()["kinds"]
    assert kinds["policy"] == 2 and kinds["fact"] == 10
    report = engine_for(store, min_active_memories=4, allow_auto_prune=True,
                        max_merge_fraction=0.6, max_prune_fraction=0.25,
                        stale_after_days=180).run(dry_run=False)
    # the protected instructions survive, the stale pytest note does not
    assert store.get("mem-sample-15").state == ACTIVE
    assert store.get("mem-sample-14").state == ACTIVE
    assert "mem-sample-11" in report.outcome.prunes_applied
    assert report.plan.considered == 20
    store.close()


def test_service_wraps_every_run_in_the_lease(tmp_path):
    service = CleanupService(tmp_path / "svc.db",
                            policy={"model": "heuristic", "min_active_memories": 1})
    for index in range(4):
        service.store.add(f"duplicate free note {index}", topic=f"t{index}")
    service.store.try_lock(owner="someone-else", lease_seconds=60)
    with pytest.raises(RunInProgress):
        service.run(dry_run=True)
    service.store.unlock(owner="someone-else")
    assert service.run(dry_run=True).status == "completed"
    # the lease is released even when the pass explodes
    class Boom(CleanupEngine):
        def plan(self, **kwargs):
            raise RuntimeError("kaboom")

    service.engine = lambda *a, **k: Boom(service.store, policy=service.policy())
    with pytest.raises(RuntimeError):
        service.run(dry_run=False)
    assert service.store.lock_info()["held"] is False
    run = service.store.list_runs()[0]
    assert run["status"] == "failed" and "kaboom" in run["error"]
    service.store.close()
