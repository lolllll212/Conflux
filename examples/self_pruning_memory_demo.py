"""One agent's memory, growing back on itself: dedupe, staleness, review, undo.

A 20-row SQLite memory store written by three agents over a year: restated facts,
a typo-level near duplicate, a contradiction (two "truths" about the same host),
an expired experiment, a 400-day-old tooling note, and two protected safety
instructions. A cleanup pass reads it, proposes folds and retirements, opens a
review for everything it is not certain about, and never deletes a row. Then the
same plan is applied, inspected, approved from the queue, reverted, mirrored onto
a Conflux fabric, and verified.

No model, no server, no network: stdlib only, deterministic.

Run from the repo root:  python examples/self_pruning_memory_demo.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux.memory import (CleanupEngine, CleanupPolicy, MemoryFabric, MemoryStore,
                            load_samples, sample_dataset)

DEMO_POLICY = {
    "model": "heuristic",
    "min_active_memories": 4,
    "max_merge_fraction": 0.6,
    "max_prune_fraction": 0.25,
    "allow_auto_prune": True,
    "auto_apply_merges": False,
    "auto_merge_exact": True,
    "schedule_enabled": False,
}


def show(title):
    print(f"\n== {title} " + "=" * max(0, 66 - len(title)))


def main():
    root = tempfile.mkdtemp(prefix="conflux-memory-demo-")
    path = os.path.join(root, "memories.db")
    try:
        return run(path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run(path):
    # --- the store arrives the way an agent writes it: one call per fact ---------
    show("seed")
    with MemoryStore(path) as store:
        loaded = load_samples(store)
        counts = store.counts()
        print(f"  {loaded} memories from {len(sample_dataset())} rows, "
              f"{counts['active']} active across {len(counts['kinds'])} kinds")
        for kind, size in sorted(counts["kinds"].items()):
            print(f"    {kind:10s} {size}")
        assert loaded == 20 and counts["open_reviews"] == 0

        policy = CleanupPolicy.from_dict(DEMO_POLICY)
        store.set_policy(policy)
        engine = CleanupEngine(store, policy=policy)

        # --- a dry run is a report, not a mutation ---------------------------------
        show("dry run")
        report = engine.run(dry_run=True, trigger="demo", actor="demo")
        stats = report.stats
        print(f"  run {report.run_id[:28]}  plan {report.plan_hash[:12]}")
        print(f"  considered {stats['considered']}  clusters {stats['clusters']}  "
              f"merges {stats['merges_proposed']} (auto {stats['merges_auto']})  "
              f"prunes {stats['prunes_proposed']} (auto {stats['prunes_auto']})  "
              f"reviews {stats['reviews']}  retained {stats['retained']}")
        for cluster in report.plan.clusters:
            print(f"    cluster[{cluster.kind}] {cluster.score:.3f} "
                  f"conflict={str(cluster.conflicts):5s} {', '.join(cluster.ids)}")
        for guard in report.plan.guards:
            print(f"    guard: {guard}")
        assert store.counts()["active"] == 20, "a dry run must not touch a memory"

        # --- what a review is for: the machine declines to guess -------------------
        show("the contradiction nobody should auto-fold")
        reviews = store.reviews(state="open")
        clash = next(r for r in reviews if r["kind"] == "contradiction")
        print(f"  #{clash['id']} [{clash['kind']}] conf={clash['confidence']} "
              f"ids={', '.join(clash['memory_ids'])}")
        print(f"  why: {clash['reason'][:96]}")
        print(f"  both bodies are kept, side by side, in the proposal:")
        merge = store.get_merge(clash["merge_id"])
        for source in merge["sources"]:
            print(f"    {source['id']}: {source['body'][:64]}")
        print(f"    merged: {merge['merged_body'][:88]}")
        assert len(merge["sources"]) == 4  # the whole hosts cluster, nothing dropped

        # --- applying: state transitions plus an audit trail -----------------------
        show("apply")
        applied = engine.run(dry_run=False, trigger="demo", actor="demo")
        outcome = applied.outcome
        print(f"  merges applied: {len(outcome.merges_applied)}  "
              f"prunes applied: {len(outcome.prunes_applied)}  "
              f"errors: {len(outcome.errors)}")
        after = store.counts()
        print(f"  by state: {after['by_state']}   (total rows {after['total']}, nothing deleted)")
        assert after["total"] == 20 + len(outcome.merges_applied)
        assert outcome.prunes_applied, "the stale tooling note should have been archived"
        assert not outcome.merges_applied, "the sample policy never auto-folds a contradiction"

        archived = store.get("mem-sample-11")
        print(f"  archived {archived.id}: state={archived.state} rev={archived.revision}")
        trail = store.timeline("mem-sample-11")
        for event in trail:
            before = (event["before"] or {}).get("state")
            after = (event["after"] or {}).get("state")
            print(f"    {event['seq']:>3} {event['ts'][:19]}  {event['action']:18s} "
                  f"{event['actor'] or '-':18s} {before or '-'} -> {after or '-'}")
        assert [e["action"] for e in trail][0] == "add"
        assert trail[-1]["action"].startswith("state:archived")
        assert trail[-1]["after"]["state"] == "archived" and trail[-1]["before"]["state"] == "active"

        # --- the queue is actionable, not advisory ---------------------------------
        show("resolve one review by hand")
        pending = store.reviews(state="open")
        merge_review = next((r for r in pending if r["kind"] == "merge"), None)
        assert merge_review is not None
        result = engine.approve_review(merge_review["id"], actor="ops", note="same fact, fold them")
        print(f"  approved #{merge_review['id']} -> {result['applied']}")
        survivor = store.get(store.get(merge_review["memory_ids"][0]).superseded_by)
        print(f"  the folded record carries its own provenance:")
        print(f"    body: {survivor.body[:80]}")
        print(f"    kind={survivor.kind} topic={survivor.topic} "
              f"origins={len(survivor.provenance['origin_ids'])} "
              f"merged_from={len(survivor.metadata['merged_from'])}")
        assert sorted(survivor.provenance["origin_ids"]) == sorted(merge_review["memory_ids"])
        assert store.get(merge_review["memory_ids"][0]).state == "merged"
        assert store.get(merge_review["memory_ids"][0]).body  # the original is still readable

        # --- undo ------------------------------------------------------------------
        show("revert the merge")
        merge_id = merge_review["merge_id"]
        reverted = store.revert_merge(merge_id, actor="ops", note="actually these are different")
        print(f"  merge #{merge_id} reverted, restored {len(reverted['restored'])} records")
        for memory_id in reverted["restored"]:
            print(f"    {memory_id}: {store.get(memory_id).state} rev={store.get(memory_id).revision}")
        assert all(store.get(m).state == "active" for m in reverted["restored"])

        # --- re-applying the same plan is a no-op, and a fresh look is honest -------
        show("the same plan, twice")
        before = store.counts()
        stale_replay = engine.apply(applied.plan, run_id=None)
        note = stale_replay.errors[0] if stale_replay.errors else "nothing to do"
        print(f"  re-applying the plan this run just executed: already_applied="
              f"{stale_replay.already_applied} ({note})")
        assert stale_replay.already_applied is True and stale_replay.merges_applied == ()
        assert store.counts() == before, "a duplicate apply must be inert"
        look = engine.plan(dry_run=True)
        print(f"  a fresh pass proposes {len(look.merges)} merges and {len(look.prunes)} prunes "
              f"again - the revert put those rows back in play, and the queue (not the store) "
              f"remembers intent")
        print(f"  skipped_unchanged {look.stats['skipped_unchanged']} of {look.considered} rows "
              f"were already judged by this policy")

        # --- settings are validated, coerced, and reversible ----------------------
        show("policy")
        tuned = policy.merged({"stale_after_days": "75", "protect_kinds": "policy, safety"})
        print(f"  '75' became {tuned.stale_after_days!r}, protect_kinds={tuned.protect_kinds}")
        assert tuned.stale_after_days == 75.0
        try:
            policy.merged({"nonsense_knob": 1})
        except ValueError as exc:
            print(f"  an unknown knob is refused: {exc}")
        else:  # pragma: no cover - would mean silent typos are accepted
            raise AssertionError("unknown policy keys must raise")
        print(f"  safety floors in force: min_active={policy.min_active_memories}, "
              f"max_prune_fraction={policy.max_prune_fraction}, "
              f"allow_hard_delete={policy.allow_hard_delete}")

        # --- the same memories, projected onto the Conflux fabric ------------------
        show("fabric mirror")
        fabric = MemoryFabric("demo-agent", namespace="default")
        synced = fabric.sync_store(store)
        print(f"  mirrored {synced['mirrored']} of {synced['scanned']} rows "
              f"(unchanged={synced['unchanged']}), state_hash={fabric.hash()[:16]}")
        peer = MemoryFabric("peer-agent", namespace="default")
        print(f"  a second replica absorbs the journal: "
              f"{peer.import_actions(fabric.export_actions())} actions, "
              f"hash {'matches' if peer.hash() == fabric.hash() else 'DIFFERS'}")
        assert peer.hash() == fabric.hash()
        replayed = MemoryFabric("replay", namespace="default")
        shuffled = list(fabric.export_actions())
        shuffled.reverse()
        replayed.import_actions(shuffled)
        print(f"  replayed in reverse order: {'same state' if replayed.hash() == fabric.hash() else 'DRIFT'}")
        assert replayed.hash() == fabric.hash()
        drift = fabric.verify_against(store)
        print(f"  mirror vs database: ok={drift['ok']} missing={len(drift['missing'])} "
              f"stale={len(drift['stale'])} differing={len(drift['differing'])}")
        assert drift["ok"] is True, json.dumps(drift)

        # --- and nothing here ever left the machine --------------------------------
        show("hard limits")
        try:
            store.delete_rows(state="archived")  # a cutoff is mandatory, by design
        except ValueError as exc:
            print(f"  a purge with no cutoff is refused: {exc}")
        else:  # pragma: no cover - would mean DELETE is one careless call away
            raise AssertionError("delete_rows must require an older_than cutoff")
        retired = store.delete_rows(state="archived", older_than="2027-01-01T00:00:00+00:00",
                                   actor="demo")
        print(f"  with an explicit cutoff it removed {retired} row(s); the audit trail still "
              f"holds the text")
        assert retired == len(outcome.prunes_applied)
        assert store.query("SELECT * FROM memories WHERE id='mem-sample-11'") == []
        deleted = store.events(action="hard_delete")
        print(f"    {deleted[0]['before']['body'][:64]}")
        assert store.integrity_check() == ["ok"]
        final = store.counts()

        print(f"\n  final counts: {final['by_state']}, total rows {final['total']}, "
              f"open reviews {final['open_reviews']}")
        print("demo passed: self-pruning memory "
              f"({final['total']} rows, {len(outcome.prunes_applied)} retired, "
              f"{len(outcome.merges_applied) + 1} folded, 0 deleted by the pass)")


if __name__ == "__main__":
    main()
