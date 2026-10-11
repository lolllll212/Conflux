"""Command line interface: ``python -m conflux.memory <command>``.

Same shape as ``python -m conflux.server``: one ``main(argv=None)`` with
argparse, no framework, and a ``--db`` that points at a file you can copy away.
Every command that changes data prints what it did, and ``run`` defaults to a
dry run so the surprising thing is the exception rather than the rule.
"""

import argparse
import json
import logging
import sqlite3
import sys

from .fixtures import build_sample_db
from .service import CleanupService
from .store import STATES, MemoryStore, RunInProgress, StoreError


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    handler = globals().get("_run_" + str(args.command))
    if handler is None:
        parser.error(f"no handler for {args.command}")
    try:
        return handler(args)
    except (ValueError, StoreError, sqlite3.Error, OSError) as exc:
        # a typed failure is information for the operator, so it prints one line
        # and exits 1; anything else is a bug and keeps its traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


# --- parser ------------------------------------------------------------------


def _global_options():
    """Shared flags, repeated per subcommand so ``--db x --log-level debug`` works."""
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--log-level", default="info", choices=("debug", "info", "warning", "error"),
                        help="logging verbosity (default: info)")
    return shared


def _parser():
    parser = argparse.ArgumentParser(prog="conflux.memory",
                                     description="self-pruning memory layer for AI agents (local only)")
    sub = parser.add_subparsers(dest="command", required=True)
    shared = _global_options()

    def add(name, **kwargs):
        return sub.add_parser(name, parents=[shared], **kwargs)

    def with_db(sp):
        sp.add_argument("--db", required=True, help="path to the SQLite memory database")
        sp.add_argument("--namespace", default="default", help="memory namespace (default: default)")
        return sp

    run = add("run", help="plan (default) or apply one cleanup pass")
    with_db(run)
    run.add_argument("--apply", action="store_true", help="mutate the database (default: dry run)")
    run.add_argument("--dry-run", action="store_true", help="explicit dry run (the default)")
    run.add_argument("--force", action="store_true", help="ignore the review cursor, rejudge everything")
    run.add_argument("--model", default=None, help="override policy.model for this run")
    run.add_argument("--limit", type=int, default=None, help="max memories to consider")
    run.add_argument("--explain", action="store_true", help="print every action in the plan")
    run.add_argument("--json", action="store_true", help="machine-readable output")

    serve = add("serve", help="run the FastAPI service (loopback by default)")
    with_db(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-scheduler", action="store_true", help="disable the nightly pass")
    serve.add_argument("--dry-run-schedule", action="store_true",
                       help="scheduled passes only propose; they never mutate")
    serve.add_argument("--reload", action="store_true")

    status = add("status", help="counts, schedule state, last runs")
    with_db(status)
    status.add_argument("--json", action="store_true")

    sample = add("sample", help="create a sample database to play with")
    with_db(sample)
    sample.add_argument("--policy", default=json.dumps(_SAMPLE_POLICY, sort_keys=True),
                       help="JSON policy to store with the samples")
    sample.add_argument("--embeddings", action="store_true", help="fill local hash embeddings")

    add_cmd = add("add", help="write one memory")
    with_db(add_cmd)
    add_cmd.add_argument("body")
    add_cmd.add_argument("--kind", default="note")
    add_cmd.add_argument("--topic", default=None)
    add_cmd.add_argument("--agent", default=None)
    add_cmd.add_argument("--uri", default=None)
    add_cmd.add_argument("--importance", type=float, default=1.0)
    add_cmd.add_argument("--idempotency-key", default=None)
    add_cmd.add_argument("--ttl-days", type=float, default=None,
                         help="retire this memory after N days (stored as expires_at)")
    add_cmd.add_argument("--json", action="store_true", help="print the write receipt as JSON")

    listing = add("list", help="list memories")
    with_db(listing)
    listing.add_argument("--state", default="active")
    listing.add_argument("--kind", default=None)
    listing.add_argument("--topic", default=None)
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true")

    reviews = add("reviews", help="show the review queue")
    with_db(reviews)
    reviews.add_argument("--all", action="store_true", help="include resolved items")
    reviews.add_argument("--json", action="store_true", help="print the queue as JSON")

    review = add("review", help="approve or reject one queued item")
    with_db(review)
    review.add_argument("id", type=int)
    review.add_argument("--decision", required=True, choices=("approve", "reject"))
    review.add_argument("--note", default=None)

    merges = add("merges", help="show merge proposals with provenance")
    with_db(merges)
    merges.add_argument("--state", default="open",
                        help="open (default), proposed, needs_review, applied, rejected, all")
    merges.add_argument("--limit", type=int, default=20)
    merges.add_argument("--json", action="store_true", help="print the proposals as JSON")

    revert = add("revert", help="undo an applied merge")
    with_db(revert)
    revert.add_argument("merge_id", type=int)
    revert.add_argument("--note", default=None)

    settings = add("settings", help="show or change the cleanup policy")
    with_db(settings)
    settings.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                          help="set a policy field (repeatable)")
    settings.add_argument("--dry-run", action="store_true", help="validate without writing")

    apply_cmd = add("apply", help="apply the plan stored by an earlier run")
    with_db(apply_cmd)
    apply_cmd.add_argument("run_id")

    backup = add("backup", help="VACUUM INTO a timestamped snapshot file")
    with_db(backup)
    backup.add_argument("--dest", default=None)

    verify = add("verify", help="integrity check plus a mirror comparison")
    with_db(verify)
    verify.add_argument("--mirror", action="store_true",
                        help="re-project the database onto a fabric and compare (checks the "
                             "projection itself)")
    verify.add_argument("--fabric-dir", default=None,
                        help="directory of a persisted mirror to compare against; drift here "
                             "means the gossiped view is behind the database")

    restore = add("restore", help="bring one memory back to active")
    with_db(restore)
    restore.add_argument("memory_id")
    restore.add_argument("--json", action="store_true")
    backup.add_argument("--json", action="store_true")
    revert.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    settings.add_argument("--json", action="store_true", help="print the resulting policy as JSON")
    return parser


_SAMPLE_POLICY = {
    "model": "heuristic",
    "min_active_memories": 4,
    "max_merge_fraction": 0.6,
    "max_prune_fraction": 0.25,
    "allow_auto_prune": True,
    "auto_apply_merges": False,
    "schedule_enabled": False,
}


# --- commands ----------------------------------------------------------------


def _service(args, **kwargs):
    return CleanupService(args.db, namespace=args.namespace, **kwargs)


def _parse_assignment(text):
    key, sep, raw = text.partition("=")
    if not sep:
        raise SystemExit(f"--set expects KEY=VALUE, got {text!r}")
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    return key.strip(), value


def _run_run(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    # --model is a one-run override: it is layered on the stored policy instead of
    # being written into settings, where it would silently outlive the invocation
    policy = store.get_policy(namespace=args.namespace)
    service = CleanupService(store=store, policy=policy.merged({"model": args.model})
                             if args.model else policy)
    try:
        dry_run = not args.apply or args.dry_run
        report = service.run(dry_run=dry_run, trigger="cli", actor="cli", force=args.force,
                             limit=args.limit)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        plan = report.plan
        print(f"run {report.run_id}  mode={'dry-run' if dry_run else 'apply'}  "
              f"model={plan.model}  plan={plan.plan_hash[:12]}")
        print(f"  considered {plan.considered}  clusters {len(plan.clusters)}  "
              f"merges {len(plan.merges)} (auto {plan.stats.get('merges_auto')})  "
              f"prunes {len(plan.prunes)} (auto {plan.stats.get('prunes_auto')})  "
              f"reviews {len(plan.reviews)}  retained {len(plan.retained)}")
        if plan.guards:
            print("  guards:")
            for guard in plan.guards:
                print(f"    ! {guard}")
        if plan.notes:
            print("  notes:")
            for note in plan.notes:
                print(f"    - {note}")
        if args.explain:
            for cluster in plan.clusters:
                print(f"  cluster[{cluster.kind}] score={cluster.score:.3f} "
                      f"conflict={cluster.conflicts}: {', '.join(cluster.ids)}")
            for merge in plan.merges:
                print(f"  merge #{merge.merge_id} [{merge.decision}] conf={merge.confidence:.2f} "
                      f"sources={list(merge.source_ids)}")
                print(f"    -> {merge.merged_body[:200]}")
            for prune in plan.prunes:
                print(f"  prune {prune.memory_id} [{prune.decision}] -> {prune.target_state}: "
                      f"{'; '.join(prune.reasons)}")
        if not dry_run and report.outcome is not None:
            applied = report.outcome.to_dict()
            print(f"  applied: {len(applied['merges_applied'])} merges, "
                  f"{len(applied['prunes_applied'])} prunes, "
                  f"skipped {len(applied['merges_skipped']) + len(applied['prunes_skipped'])}")
            for note in applied["merges_skipped"] + applied["prunes_skipped"]:
                print(f"    ~ skipped: {note}")
            for error in applied["errors"]:
                print(f"    x {error}")
        if dry_run:
            print(f"  memories untouched; to apply this plan later: "
                  f"python -m conflux.memory apply --db {args.db} {report.run_id}")
        return 0
    except RunInProgress as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        service.store.close()


def _run_serve(args):
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - depends on the extras
        raise SystemExit("serve needs the memory extras: pip install 'conflux[memory]'")
    from .api import create_app
    service = _service(args)
    app = create_app(service, start_scheduler=not args.no_scheduler,
                     scheduler=None)
    app.state.scheduler.dry_run = bool(args.dry_run_schedule)
    print(f"conflux memory api on http://{args.host}:{args.port}  db={args.db}  "
          f"namespace={args.namespace}  scheduler={'on' if not args.no_scheduler else 'off'}")
    print("docs at /docs; token auth " +
          ("REQUIRED (CONFLUX_MEMORY_TOKEN set)" if _token_set() else "off (loopback only)"))
    try:
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
    finally:
        service.store.close()
    return 0


def _token_set():
    import os
    return bool(os.environ.get("CONFLUX_MEMORY_TOKEN"))


def _run_status(args):
    service = _service(args)
    try:
        data = service.status()
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        counts = data["counts"]
        print(f"db {args.db}  namespace {args.namespace}  schema {data['database']['schema_version']}")
        print(f"  active {counts['active']}  total {counts['total']}  by state {counts['by_state']}")
        print(f"  open reviews {counts['open_reviews']}  proposed merges {counts['proposed_merges']}")
        sched = data["scheduler"]
        print(f"  scheduler enabled={sched.get('enabled')} running={sched.get('running')} "
              f"tz={sched.get('timezone')} at={sched.get('at')} next={sched.get('next_run_at')}")
        print(f"  model {data['model']}")
        print(f"  lock {data['lock']}")
        for run in data["last_runs"]:
            print(f"  run {run['id']}  {run['mode']}  {run['status']}  {run['started_at']}  "
                  f"{ {k: v for k, v in (run['stats'] or {}).items() if k in ('considered', 'merges_proposed', 'prunes_auto')} }")
        return 0
    finally:
        service.store.close()


def _run_sample(args):
    store = build_sample_db(args.db, namespace=args.namespace,
                            policy=json.loads(args.policy), with_embedding=args.embeddings)
    counts = store.counts()
    store.close()
    print(f"sample database at {args.db}: {counts['active']} active memories, "
          f"{len(counts['kinds'])} kinds")
    print("try: python -m conflux.memory run --db " + args.db + " --explain")
    return 0


def _run_add(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    try:
        receipt = store.add(args.body, kind=args.kind, topic=args.topic, source_agent=args.agent,
                            uri=args.uri, importance=args.importance, ttl_days=args.ttl_days,
                            idempotency_key=args.idempotency_key, actor="cli")
        if getattr(args, "json", False):
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        verb = "added" if receipt["created"] else "matched existing"
        print(f"{verb} {receipt['id']} ({receipt.get('reason', 'new')})")
        return 0
    finally:
        store.close()


def _run_list(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    try:
        state = None if args.state in ("all", "") else args.state
        if state is not None and state not in STATES:
            raise ValueError(f"unknown state {args.state!r}; use 'all' or one of "
                             f"{', '.join(sorted(STATES))}")
        rows = store.list(state=state, kind=args.kind, topic=args.topic,
                          limit=args.limit)
        if args.json:
            print(json.dumps([m.to_dict() for m in rows], indent=2, sort_keys=True))
            return 0
        for memory in rows:
            print(f"{memory.id}  {memory.state:9s}  {memory.kind:10s}  "
                  f"rev={memory.revision} seen={str(memory.last_seen_at)[:10]}  "
                  f"{memory.body[:72]}")
        print(f"({len(rows)} shown, state={args.state})")
        return 0
    finally:
        store.close()


def _run_reviews(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    try:
        rows = store.reviews(state=None if args.all else "open", limit=100)
        if getattr(args, "json", False):
            print(json.dumps({"items": rows}, indent=2, sort_keys=True, default=str))
            return 0
        if not rows:
            print("review queue is empty")
            return 0
        for review in rows:
            print(f"#{review['id']}  [{review['state']}]  {review['kind']}  "
                  f"conf={review['confidence']}")
            print(f"   why: {review['reason']}")
            print(f"   ids: {', '.join(review['memory_ids'])}")
            if review.get("merge_id"):
                print(f"   merge: {review['merge_id']}")
        print(f"\nresolve with: python -m conflux.memory review --db {args.db} <id> "
              f"--decision approve|reject")
        return 0
    finally:
        store.close()


def _run_review(args):
    service = _service(args)
    try:
        result = service.review(args.id, "approved" if args.decision == "approve" else "rejected",
                               actor="cli", note=args.note)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    finally:
        service.store.close()


def _run_merges(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    try:
        rows = store.merges(state=None if args.state == "all" else args.state, limit=args.limit)
        if getattr(args, "json", False):
            print(json.dumps({"items": rows}, indent=2, sort_keys=True, default=str))
            return 0
        if not rows:
            print(f"no merges with state={args.state}")
            return 0
        for merge in rows:
            print(f"#{merge['id']}  [{merge['state']}]  conf={merge['confidence']}  "
                  f"sim={merge['similarity']}  model={merge['model']}")
            print(f"   reason: {merge['reason']}")
            print(f"   merged: {merge['merged_body'][:200]}")
            print(f"   sources: {', '.join(link['memory_id'] for link in store.get_merge(merge['id'])['links'])}")
        return 0
    finally:
        store.close()


def _run_revert(args):
    service = _service(args)
    try:
        print(json.dumps(service.store.revert_merge(args.merge_id, actor="cli", note=args.note),
                         indent=2, sort_keys=True))
        return 0
    finally:
        service.store.close()


def _run_settings(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    try:
        policy = store.get_policy(namespace=args.namespace)
        if not args.set:
            print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
            return 0
        updates = dict(_parse_assignment(item) for item in args.set)
        candidate = policy.merged(updates)
        candidate.validate()
        changes = {k: v for k, v in candidate.to_dict().items() if policy.to_dict().get(k) != v}
        if args.dry_run:
            if getattr(args, "json", False):
                print(json.dumps({"dry_run": True, "valid": True, "changes": changes},
                                 indent=2, sort_keys=True))
            else:
                print("valid; nothing written")
                print(json.dumps(changes, indent=2, sort_keys=True))
            return 0
        store.set_policy(candidate, namespace=args.namespace, updated_by="cli")
        if getattr(args, "json", False):
            print(json.dumps({"updated": True, "changes": changes, "policy": candidate.to_dict()},
                             indent=2, sort_keys=True))
        else:
            print("updated " + ", ".join(sorted(updates)))
        return 0
    finally:
        store.close()


def _run_apply(args):
    service = _service(args)
    try:
        run_id, outcome = service.apply_run(args.run_id, actor="cli")
        print(f"applied {args.run_id} as run {run_id}")
        print(json.dumps(outcome.to_dict(), indent=2, sort_keys=True))
        return 0
    except RunInProgress as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        service.store.close()


def _run_backup(args):
    service = _service(args)
    try:
        path = service.backup(args.dest)
        print(json.dumps({"path": str(path)}) if getattr(args, "json", False) else path)
        return 0
    finally:
        service.store.close()


def _run_verify(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    try:
        report = {"integrity": store.integrity_check(), "counts": store.counts(),
                  "schema": store.schema_version}
        if args.mirror:
            from .bridge import MemoryFabric
            fabric = MemoryFabric("verify", namespace=args.namespace)
            fabric.sync_store(store, namespace=args.namespace)
            report["mirror"] = fabric.verify_against(store, namespace=args.namespace)
            report["fabric_hash"] = fabric.hash()
        if args.fabric_dir:
            from .bridge import MemoryFabric
            # a persisted mirror is deliberately NOT re-synced first: the point of
            # this check is to see whether what has been gossiped lags the database
            fabric = MemoryFabric("verify", namespace=args.namespace, data_dir=args.fabric_dir)
            report["fabric"] = {"dir": args.fabric_dir,
                                **fabric.verify_against(store, namespace=args.namespace),
                                "hash": fabric.hash()}
            fabric.close()
            if not report["fabric"]["ok"]:
                print("drift: the mirrored view does not match the database "
                      f"(missing={report['fabric']['missing']}, "
                      f"stale={report['fabric']['stale']})", file=sys.stderr)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        clean = report["integrity"] == ["ok"]
        if args.fabric_dir:
            clean = clean and report["fabric"]["ok"]
        return 0 if clean else 1
    finally:
        store.close()


def _run_restore(args):
    store = MemoryStore(args.db, namespace=args.namespace)
    try:
        record = store.get(args.memory_id)
        if record.state == "active":
            if getattr(args, "json", False):
                print(json.dumps({"restored": False, "reason": "already active", "id": record.id}))
            else:
                print(f"{record.id} is already active")
            return 0
        ok = store.set_state(record.id, "active", actor="cli", action="restore",
                            detail={"from": record.state})
        if getattr(args, "json", False):
            print(json.dumps({"restored": bool(ok), "id": record.id, "from": record.state,
                              "state": "active" if ok else record.state}))
        else:
            print(f"restored {record.id} from {record.state}: {ok}")
        return 0 if ok else 1
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
