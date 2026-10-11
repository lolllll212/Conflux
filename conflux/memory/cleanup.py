"""The cleanup pass: consider → propose → guard → apply, with no silent edits.

A run has two halves and either can be stopped.

``plan()``
    reads the active memories, groups duplicates and related sets, asks the
    local model for a concise merge, scores staleness, applies the policy's
    safety floors, and reduces everything to a :class:`CleanupPlan` of explicit
    actions. It persists proposals, review requests and the run ledger row —
    and never touches a ``memories`` row.

``apply()``
    executes a plan with one transaction per action, re-checking every target
    row's ``revision`` against the value captured at plan time. A record that
    moved in between is skipped as stale instead of guessed at, and a merge
    whose cluster half-moved rolls back whole. The safety floors are recomputed
    here as well, so a plan that would empty the store is blocked even if it was
    written yesterday.

Idempotency comes from three places: the plan fingerprint (``runs.plan_hash`` —
an identical plan that already applied is not applied twice), per-row
compare-and-set, and the ``review_cursor`` that keeps a nightly pass from
re-litigating records whose content has not changed.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, replace

from .dedupe import build_clusters
from .model import HeuristicModel, MergeRequest, build_model
from .policy import CleanupPolicy
from .staleness import ARCHIVE, PRUNE, RETAIN, evaluate
from .similar import fingerprint
from .store import ACTIVE, PlanStale

log = logging.getLogger("conflux.memory.cleanup")


# --- actions ----------------------------------------------------------------


@dataclass(frozen=True)
class MergeAction:
    """A proposal to fold ``source_ids`` into one new memory."""

    merge_id: int
    source_ids: tuple
    merged_body: str
    kind: str = "merged"
    topic: str = None
    confidence: float = 0.0
    similarity: float = 0.0
    reason: str = ""
    model: str = "none"
    decision: str = "review"            # 'auto' (may apply) | 'review' (park it)
    conflict: bool = False
    importance: float = 1.0
    expected: tuple = field(default_factory=tuple)   # ((memory_id, revision), ...)
    dropped_facts: tuple = field(default_factory=tuple)

    @property
    def will_apply(self):
        return self.decision == "auto"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PruneAction:
    """A proposal to retire one memory to ``archived`` or ``pruned`` state."""

    memory_id: str
    target_state: str = "archived"
    decision: str = "review"
    confidence: float = 0.0
    reasons: tuple = field(default_factory=tuple)
    rules: tuple = field(default_factory=tuple)
    expected_revision: int = 1
    guarded_by: tuple = field(default_factory=tuple)

    @property
    def will_apply(self):
        return self.decision == "auto"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ReviewAction:
    kind: str                            # 'merge' | 'prune' | 'contradiction'
    memory_ids: tuple
    reason: str
    confidence: float = 0.0
    merge_id: int = None
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CleanupPlan:
    run_id: str = ""
    namespace: str = "default"
    dry_run: bool = True
    plan_hash: str = ""
    considered: int = 0
    skipped_unchanged: int = 0
    clusters: tuple = field(default_factory=tuple)
    merges: tuple = field(default_factory=tuple)
    prunes: tuple = field(default_factory=tuple)
    reviews: tuple = field(default_factory=tuple)
    retained: tuple = field(default_factory=tuple)
    guards: tuple = field(default_factory=tuple)
    notes: tuple = field(default_factory=tuple)
    stats: dict = field(default_factory=dict)
    model: str = "null"

    @property
    def empty(self):
        return not self.merges and not self.prunes

    def to_dict(self):
        return {"run_id": self.run_id, "namespace": self.namespace, "dry_run": self.dry_run,
                "plan_hash": self.plan_hash, "considered": self.considered,
                "skipped_unchanged": self.skipped_unchanged,
                "clusters": [c.to_dict() for c in self.clusters],
                "merges": [m.to_dict() for m in self.merges],
                "prunes": [p.to_dict() for p in self.prunes],
                "reviews": [r.to_dict() for r in self.reviews],
                "retained": list(self.retained), "guards": list(self.guards),
                "notes": list(self.notes), "stats": dict(self.stats), "model": self.model}

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        data.pop("clusters", None)  # clusters are an analysis artifact, not an action
        return cls(run_id=data.get("run_id", ""), namespace=data.get("namespace", "default"),
                   dry_run=bool(data.get("dry_run", True)), plan_hash=data.get("plan_hash", ""),
                   considered=int(data.get("considered", 0)),
                   skipped_unchanged=int(data.get("skipped_unchanged", 0)),
                   merges=tuple(MergeAction(**_clean(MergeAction, m)) for m in data.get("merges", [])),
                   prunes=tuple(PruneAction(**_clean(PruneAction, p)) for p in data.get("prunes", [])),
                   reviews=tuple(ReviewAction(**_clean(ReviewAction, r)) for r in data.get("reviews", [])),
                   retained=tuple(data.get("retained", [])), guards=tuple(data.get("guards", [])),
                   notes=tuple(data.get("notes", [])), stats=dict(data.get("stats", {})),
                   model=data.get("model", "null"))


@dataclass(frozen=True)
class ApplyOutcome:
    merges_applied: tuple = field(default_factory=tuple)
    merges_skipped: tuple = field(default_factory=tuple)
    prunes_applied: tuple = field(default_factory=tuple)
    prunes_skipped: tuple = field(default_factory=tuple)
    reviews_opened: int = 0
    guards: tuple = field(default_factory=tuple)
    already_applied: bool = False
    errors: tuple = field(default_factory=tuple)

    def to_dict(self):
        return {"merges_applied": list(self.merges_applied), "merges_skipped": list(self.merges_skipped),
                "prunes_applied": list(self.prunes_applied), "prunes_skipped": list(self.prunes_skipped),
                "reviews_opened": self.reviews_opened, "guards": list(self.guards),
                "already_applied": self.already_applied, "errors": list(self.errors)}


@dataclass(frozen=True)
class RunReport:
    run_id: str
    namespace: str = "default"
    dry_run: bool = True
    status: str = "completed"
    plan_hash: str = ""
    stats: dict = field(default_factory=dict)
    outcome: ApplyOutcome = None
    plan: CleanupPlan = None

    def to_dict(self, include_plan=True):
        return {"run_id": self.run_id, "namespace": self.namespace, "dry_run": self.dry_run,
                "status": self.status, "plan_hash": self.plan_hash, "stats": dict(self.stats),
                "applied": self.outcome.to_dict() if self.outcome else None,
                "plan": self.plan.to_dict() if (include_plan and self.plan) else None}


# --- engine -----------------------------------------------------------------


class CleanupEngine:
    """Turns the store's contents into a guarded, reviewable cleanup plan."""

    def __init__(self, store, *, policy=None, model=None, logger=None):
        self.store = store
        self.policy = policy if isinstance(policy, CleanupPolicy) else CleanupPolicy.from_dict(
            policy or store.get_policy(), known_only=True)
        self.model = model if model is not None else build_model(self.policy.model,
                                                                 dict(self.policy.model_options))
        self.log = logger or log

    # -- planning ------------------------------------------------------------

    def plan(self, *, run_id=None, dry_run=True, namespace=None, limit=None, force=False):
        ns = namespace or self.store.namespace
        policy = self.policy
        active = self.store.list(state=ACTIVE, namespace=ns, order="created_at",
                                 limit=int(limit or policy.max_memories_per_run), with_vector=True)
        notes = []
        clusters, _singletons = build_clusters(active, policy)
        by_id = {m.id: m for m in active}

        cursors = {} if force or not policy.review_unchanged else \
            self.store.reviewed_fingerprints([m.id for m in active], namespace=ns)
        targets = [m for m in active if cursors.get(m.id) != m.fingerprint]
        skipped = len(active) - len(targets)

        merges, merge_notes = self._plan_merges(clusters, by_id, ns, run_id, policy)
        prunes, prune_notes = self._plan_staleness(targets, clusters, policy, ns, notes=False)
        merges, prunes, guards = self._apply_guards(active, merges, prunes, policy)
        reviews = self._review_actions(merges, prunes)

        touched = {mid for m in merges for mid in m.source_ids} | \
                  {p.memory_id for p in prunes}
        retained = tuple(sorted(m.id for m in active if m.id not in touched))
        stats = {
            "considered": len(active),
            "clusters": len(clusters),
            "exact_clusters": sum(1 for c in clusters if c.kind == "exact"),
            "near_clusters": sum(1 for c in clusters if c.kind == "near"),
            "conflict_clusters": sum(1 for c in clusters if c.conflicts),
            "merges_proposed": len(merges),
            "merges_auto": sum(1 for m in merges if m.will_apply),
            "sources_to_merge": sum(len(m.source_ids) for m in merges if m.will_apply),
            "prunes_proposed": len(prunes),
            "prunes_auto": sum(1 for p in prunes if p.will_apply),
            "reviews": len(reviews),
            "retained": len(retained),
            "skipped_unchanged": skipped,
            "model": self.model.name,
            "dry_run": bool(dry_run),
        }
        plan = CleanupPlan(run_id=run_id or "", namespace=ns, dry_run=bool(dry_run),
                           considered=len(active), skipped_unchanged=skipped,
                           clusters=tuple(clusters), merges=tuple(merges), prunes=tuple(prunes),
                           reviews=tuple(reviews), retained=retained, guards=tuple(guards),
                           notes=tuple(notes + merge_notes + prune_notes), stats=stats,
                           model=self.model.name)
        return replace(plan, plan_hash=_plan_hash(plan))

    def _plan_merges(self, clusters, by_id, ns, run_id, policy):
        merges, notes = [], []
        for cluster in clusters:
            members = [by_id[i] for i in cluster.ids if i in by_id]
            if len(members) < 2:
                continue
            request = MergeRequest(
                topic=cluster.topic, similarity=cluster.mean_score, conflicts=cluster.conflicts,
                max_chars=policy.max_merged_chars, reason=cluster.kind,
                members=tuple({"id": m.id, "body": m.body, "kind": m.kind,
                               "importance": m.importance, "created_at": m.created_at,
                               "updated_at": m.updated_at} for m in members))
            suggestion = None
            try:
                suggestion = self.model.merge(request)
            except Exception as exc:  # a broken model must never cost data
                self.log.warning("model merge failed for %s: %r", cluster.ids, exc)
                notes.append(f"model error on cluster {members[0].id}: {exc!r}")
            if suggestion is None:
                suggestion = HeuristicModel().merge(request)
                if suggestion is None:
                    notes.append(f"no merge suggestion for {cluster.ids}, left alone")
                    continue
                notes.append(f"heuristic merge used for {members[0].id} (no or failing model)")
            body = str(suggestion.body).strip()
            if not body:
                notes.append(f"empty merge body for {cluster.ids}, left alone")
                continue
            conflict = bool(cluster.conflicts or suggestion.contradiction)
            # An "exact" cluster is provably lossless: every member normalizes to
            # the same text, so the merge cannot invent or drop a statement.
            lossless = cluster.kind == "exact"
            confidence = 1.0 if lossless else max(0.0, min(1.0, float(suggestion.confidence)))
            duplicate_grade = lossless or cluster.score >= policy.duplicate_similarity
            allowed_by_model = policy.auto_apply_merges or (lossless and policy.auto_merge_exact)
            auto = bool(allowed_by_model and duplicate_grade and not conflict
                        and confidence >= policy.auto_apply_confidence)
            sources = [{"id": m.id, "body": m.body, "kind": m.kind, "topic": m.topic,
                        "created_at": m.created_at, "updated_at": m.updated_at,
                        "importance": m.importance, "revision": m.revision,
                        "provenance": m.provenance} for m in members]
            kinds = {m.kind for m in members}
            cluster_key = fingerprint(",".join(sorted(m.id for m in members)),
                                      fingerprint(body))
            merge_id = self.store.propose_merge(
                cluster_key=cluster_key,
                merged_body=body, source_ids=[m.id for m in members], run_id=run_id,
                topic=cluster.topic, confidence=confidence, similarity=cluster.mean_score,
                reason=(suggestion.reason or ("contradictory statements" if conflict else cluster.kind))
                       + (" (exact duplicate fold)" if lossless else ""),
                model=suggestion.model or self.model.name, namespace=ns,
                state="proposed" if auto else "needs_review", sources=sources)
            merges.append(MergeAction(
                merge_id=merge_id, source_ids=tuple(m.id for m in members), merged_body=body,
                kind=next(iter(kinds)) if len(kinds) == 1 else "merged", topic=cluster.topic,
                confidence=confidence, similarity=cluster.mean_score,
                reason=suggestion.reason or cluster.kind, model=suggestion.model or self.model.name,
                decision="auto" if auto else "review", conflict=conflict,
                importance=max(float(m.importance) for m in members),
                expected=tuple((m.id, m.revision) for m in members),
                dropped_facts=tuple(suggestion.dropped_facts or ())))
            self.log.info("cluster %s -> %s merge (confidence %.2f, similarity %.2f, %d sources)",
                          ",".join(cluster.ids), "auto" if auto else "review", confidence,
                          cluster.mean_score, len(members))
        return merges, notes

    def _plan_staleness(self, targets, clusters, policy, ns, notes=True):
        conflicted = {mid for cluster in clusters if cluster.conflicts for mid in cluster.ids}
        actions, log_lines = [], []
        for memory in targets:
            verdict = evaluate(memory, policy, conflict=memory.id in conflicted)
            if verdict.decision == RETAIN:
                continue
            decision = "auto" if (policy.allow_auto_prune and verdict.decision in (PRUNE, ARCHIVE)
                                  and not verdict.guarded_by) else "review"
            if verdict.guarded_by:
                log_lines.append(f"{memory.id}: retained by {','.join(verdict.guarded_by)}")
            target_state = _target_state(verdict.decision, policy)
            actions.append(PruneAction(memory_id=memory.id, target_state=target_state,
                                       decision=decision, confidence=verdict.confidence,
                                       reasons=verdict.reasons, rules=verdict.rules,
                                       expected_revision=memory.revision,
                                       guarded_by=verdict.guarded_by))
            self.log.info("stale %s: %s -> %s (%s)", memory.id, verdict.decision, decision,
                          "; ".join(verdict.reasons) or "no rule fired")
        return actions, (log_lines if notes else [])

    def _review_actions(self, merges, prunes):
        reviews = []
        for merge in merges:
            if merge.will_apply:
                continue
            reviews.append(ReviewAction(
                kind="contradiction" if merge.conflict else "merge", memory_ids=merge.source_ids,
                reason=f"merge #{merge.merge_id} needs review: {merge.reason}",
                confidence=merge.confidence, merge_id=merge.merge_id,
                detail={"merged_body": merge.merged_body, "similarity": round(merge.similarity, 4),
                        "dropped_facts": list(merge.dropped_facts), "conflict": merge.conflict,
                        "model": merge.model}))
        for prune in prunes:
            if prune.will_apply:
                continue
            reviews.append(ReviewAction(
                kind="prune", memory_ids=(prune.memory_id,),
                reason=f"{prune.target_state} proposed: {'; '.join(prune.reasons)}",
                confidence=prune.confidence,
                detail={"rules": list(prune.rules), "guarded_by": list(prune.guarded_by),
                        "target_state": prune.target_state}))
        return tuple(reviews)

    # -- safety floors -------------------------------------------------------

    def _apply_guards(self, active, merges, prunes, policy):
        """Cap how much one pass may remove. This only ever downgrades a decision."""
        guards = []
        active_count = len(active)
        if not active_count:
            return (), (), ("nothing active",)
        min_active = int(policy.min_active_memories)
        budget = int(active_count * float(policy.max_prune_fraction))
        merge_budget = int(active_count * float(policy.max_merge_fraction))

        if not policy.allow_auto_prune:
            if any(p.will_apply for p in prunes):
                guards.append("allow_auto_prune=False: every prune parked for review")
            prunes = [replace(p, decision="review") for p in prunes]

        auto_prunes = [p for p in prunes if p.will_apply]
        auto_merges = [m for m in merges if m.will_apply]
        merge_removals = sum(len(m.source_ids) - 1 for m in auto_merges)
        removals = len(auto_prunes) + merge_removals
        floor_room = max(0, active_count - min_active)

        # A merge and a prune both shrink the store, so they share the removal budget.
        # The floor is what it leaves room for *after* the merges, and a guard is only
        # reported when it actually cost something - a warning about a cap that
        # changed nothing reads like a bug report.
        if removals > floor_room:
            allowed = max(0, floor_room - merge_removals)
            source = f"min_active_memories={min_active} with {active_count} active"
        elif removals > budget:
            allowed = max(0, budget - merge_removals)
            source = f"max_prune_fraction={policy.max_prune_fraction} with {active_count} active"
        else:
            allowed, source = None, None
        if source is not None:
            keep = {p.memory_id for p in _ranked(auto_prunes)[:allowed]}
            blocked = [p for p in prunes if p.will_apply and p.memory_id not in keep]
            if blocked:
                guards.append(f"{source}: {len(blocked)} prune(s) parked for review "
                              f"(removals {removals}, room {allowed + merge_removals})")
                prunes = _downgrade(prunes, lambda p: p.memory_id not in keep)

        auto_merges = [m for m in merges if m.will_apply]
        if auto_merges and sum(len(m.source_ids) for m in auto_merges) > merge_budget:
            guards.append(f"max_merge_fraction={policy.max_merge_fraction}: merge sources capped at "
                          f"{merge_budget}")
            used, allowed = 0, set()
            for merge in _ranked_merges(auto_merges):
                if used + len(merge.source_ids) > merge_budget:
                    continue
                allowed.add(merge.merge_id)
                used += len(merge.source_ids)
            merges = _downgrade(merges, lambda m: m.merge_id not in allowed)

        if not policy.prune_alone_in_topic:
            # Never erase a concept a crew still relies on: when a pass would
            # prune every active memory under a topic, keep the freshest one.
            by_topic = {}
            for memory in active:
                if memory.topic:
                    by_topic.setdefault(memory.topic, []).append(memory)
            merge_sources = {sid for m in merges if m.will_apply for sid in m.source_ids}
            for topic, members in sorted(by_topic.items(), key=lambda kv: str(kv[0])):
                if len(members) < 2:
                    continue
                doomed = [m for m in members if any(p.memory_id == m.id and p.will_apply
                                                     for p in prunes) and m.id not in merge_sources]
                if len(doomed) == len(members):
                    keep = max(doomed, key=lambda m: (m.updated_at or "", m.id))
                    guards.append(f"topic:{topic}: kept {keep.id} so the concept survives")
                    prunes = _downgrade(prunes, lambda p: p.memory_id == keep.id)
        return tuple(merges), tuple(prunes), tuple(guards)

    # -- applying ------------------------------------------------------------

    def apply(self, plan, *, run_id=None, namespace=None, force=False, reopen_reviews=True):
        """Execute a plan under compare-and-set. Skips anything that moved."""
        ns = namespace or plan.namespace or self.store.namespace
        if plan.plan_hash and not force:
            prior = self.store.find_completed_plan(plan.plan_hash, namespace=ns)
            if prior and prior != run_id:
                self.log.info("plan %s already applied by run %s", plan.plan_hash[:12], prior)
                return ApplyOutcome(already_applied=True, errors=(f"already applied by run {prior}",))
        current = {m.id: m for m in self.store.list(state=ACTIVE, namespace=ns, order="id",
                                                    limit=self.policy.max_memories_per_run)}
        merges, prunes, guards = self._apply_guards(list(current.values()), list(plan.merges),
                                                    list(plan.prunes), self.policy)
        applied_merges, skipped_merges, errors = [], [], []
        for merge in merges:
            if not merge.will_apply:
                continue
            try:
                self.store.apply_merge(merge.merge_id, source_ids=list(merge.source_ids),
                                       merged_body=merge.merged_body, kind=merge.kind,
                                       topic=merge.topic, confidence=merge.confidence,
                                       importance=merge.importance,
                                       metadata={"cluster_similarity": round(merge.similarity, 4),
                                                 "dropped_facts": list(merge.dropped_facts),
                                                 "model": merge.model, "plan_hash": plan.plan_hash},
                                       reason=merge.reason, model=merge.model,
                                       expected=dict(merge.expected), run_id=run_id,
                                       actor=f"cleanup:{merge.model}", namespace=ns)
            except PlanStale as exc:
                skipped_merges.append(f"merge {merge.merge_id}: {exc}")
                self.log.warning("merge %s skipped: %s", merge.merge_id, exc)
                continue
            except Exception as exc:
                errors.append(f"merge {merge.merge_id}: {exc!r}")
                self.log.exception("merge %s failed", merge.merge_id)
                continue
            applied_merges.append(merge.merge_id)
            for source_id in merge.source_ids:
                row = current.get(source_id)
                if row is not None:
                    self.store.record_review(source_id, row.fingerprint, namespace=ns, run_id=run_id,
                                             verdict=f"merged via merge {merge.merge_id}")
        applied_prunes, skipped_prunes = [], []
        for prune in prunes:
            if not prune.will_apply:
                continue
            row = current.get(prune.memory_id)
            if row is None:
                skipped_prunes.append(f"{prune.memory_id}: no longer active")
                continue
            moved = self.store.set_state(prune.memory_id, prune.target_state, namespace=ns,
                                         run_id=run_id, actor=f"cleanup:{self.model.name}",
                                         action=f"state:{prune.target_state}:auto",
                                         detail={"reasons": list(prune.reasons),
                                                 "rules": list(prune.rules),
                                                 "confidence": prune.confidence,
                                                 "plan_hash": plan.plan_hash},
                                         expect_revision=prune.expected_revision)
            if not moved:
                skipped_prunes.append(f"{prune.memory_id}: revision moved past {prune.expected_revision}")
                continue
            applied_prunes.append(prune.memory_id)
            self.store.record_review(prune.memory_id, row.fingerprint, namespace=ns, run_id=run_id,
                                    verdict=prune.target_state)
        opened = self._open_reviews(plan, run_id=run_id, namespace=ns) if reopen_reviews else 0
        outcome = ApplyOutcome(merges_applied=tuple(applied_merges), merges_skipped=tuple(skipped_merges),
                               prunes_applied=tuple(applied_prunes), prunes_skipped=tuple(skipped_prunes),
                               reviews_opened=opened, guards=tuple(guards), errors=tuple(errors))
        self.log.info("apply: %d merges, %d prunes, %d skipped, %d reviews opened",
                      len(applied_merges), len(applied_prunes),
                      len(skipped_merges) + len(skipped_prunes), opened)
        return outcome

    # -- run orchestration ---------------------------------------------------

    def run(self, *, dry_run=True, trigger="manual", namespace=None, force=False, limit=None,
             actor=None):
        """One complete pass: plan, persist, and apply unless it is a dry run."""
        ns = namespace or self.store.namespace
        run_id = self.store.create_run(namespace=ns, trigger=trigger,
                                       mode="dry_run" if dry_run else "apply", actor=actor,
                                       policy=self.policy.to_dict(), model=self.model.name)
        status, error = "completed", None
        try:
            plan = self.plan(run_id=run_id, dry_run=dry_run, namespace=ns, limit=limit, force=force)
            plan = replace(plan, run_id=run_id)
            outcome = None
            if dry_run:
                self._open_reviews(plan, run_id=run_id, namespace=ns)
            else:
                outcome = self.apply(plan, run_id=run_id, namespace=ns, force=force)
            stats = dict(plan.stats)
            stats["applied"] = outcome.to_dict() if outcome else None
            self.store.update_run(run_id, status=status, stats=stats, plan_hash=plan.plan_hash,
                                  plan_json=json.dumps(_plan_payload(plan), sort_keys=True,
                                                        ensure_ascii=False),
                                  namespace=ns, finished=True)
            self.store.event("run_completed", namespace=ns, target=run_id, run_id=run_id, actor=actor,
                             detail={"dry_run": bool(dry_run), "plan_hash": plan.plan_hash,
                                     "merges": len(plan.merges), "prunes": len(plan.prunes)})
            return RunReport(run_id=run_id, namespace=ns, dry_run=bool(dry_run), status=status,
                             plan_hash=plan.plan_hash, stats=stats, outcome=outcome, plan=plan)
        except Exception as exc:
            error = repr(exc)
            self.store.update_run(run_id, status="failed", error=error, namespace=ns, finished=True,
                                  stats={"error": error})
            self.store.event("run_failed", namespace=ns, target=run_id, run_id=run_id,
                             detail={"error": error})
            self.log.exception("cleanup run %s failed", run_id)
            raise

    def rehydrate(self, run_id, *, namespace=None):
        """Load a stored plan so an earlier dry run can be applied later."""
        run = self.store.get_run(run_id, namespace=namespace or self.store.namespace)
        if not run.get("plan"):
            raise StoreError(f"run {run_id} stored no plan; was it a dry run?")
        return CleanupPlan.from_dict(run["plan"])

    def apply_run(self, run_id, *, namespace=None, actor=None, force=False):
        """Apply the plan recorded by an earlier run, revalidated row by row."""
        ns = namespace or self.store.namespace
        plan = self.rehydrate(run_id, namespace=ns)
        plan = replace(plan, dry_run=False)
        new_run = self.store.create_run(namespace=ns, trigger="manual", mode="apply", actor=actor,
                                       policy=self.policy.to_dict(), model=self.model.name)
        outcome = self.apply(plan, run_id=new_run, namespace=ns, force=force)
        stats = dict(plan.stats)
        stats["applied"] = outcome.to_dict()
        stats["source_run"] = run_id
        self.store.update_run(new_run, status="completed", stats=stats, plan_hash=plan.plan_hash,
                              namespace=ns, finished=True)
        self.store.event("run_applied", namespace=ns, target=new_run, run_id=new_run, actor=actor,
                         detail={"source_run": run_id, "merges_applied": len(outcome.merges_applied),
                                 "prunes_applied": len(outcome.prunes_applied)})
        return new_run, outcome

    def approve_review(self, review_id, *, actor=None, note=None, namespace=None):
        """Resolve a review; an approved merge/prune is executed immediately."""
        ns = namespace or self.store.namespace
        result = self.store.resolve_review(review_id, "approved", note=note, actor=actor,
                                            namespace=ns)
        if not result.get("changed"):
            return {"review_id": review_id, "applied": False, "reason": f"already {result.get('state')}"}
        run_id = self.store.create_run(namespace=ns, trigger="review", mode="apply", actor=actor,
                                       policy=self.policy.to_dict(), model=self.model.name)
        applied, skipped, errors = [], [], []
        detail, review_row = _review_detail(self.store, review_id, namespace=ns)
        # merge_id lives on the review row itself; older rows kept it only inside
        # the JSON detail, so accept either.
        merge_id = detail.get("merge_id") or (review_row or {}).get("merge_id")
        if merge_id and result.get("kind") in ("merge", "contradiction"):
            merge = self.store.get_merge(merge_id, namespace=ns)
            if merge and merge.get("sources"):
                try:
                    merged_id = self.store.apply_merge(
                        merge["id"], source_ids=[s["id"] for s in merge["sources"]],
                        merged_body=merge["merged_body"], kind="merged", topic=merge.get("topic"),
                        confidence=merge.get("confidence"),
                        importance=max([float(s.get("importance") or 1.0) for s in merge["sources"]]),
                        metadata={"approved_by": actor, "review_id": review_id,
                                  "cluster_similarity": merge.get("similarity")},
                        reason=merge.get("reason"), model=merge.get("model"),
                        expected={s["id"]: s.get("revision") for s in merge["sources"] if s.get("revision")},
                        run_id=run_id, actor=actor or "review", namespace=ns)
                    applied.append(f"merge:{merged_id}")
                except PlanStale as exc:
                    skipped.append(f"merge: {exc}")
                except Exception as exc:
                    errors.append(f"merge: {exc!r}")
        elif result["kind"] == "prune":
            for memory_id in result.get("memory_ids") or []:
                memory = self.store.get(memory_id, namespace=ns)
                target = (detail.get("target_state") if detail.get("target_state") in
                          ("pruned", "archived") else "archived")
                if self.store.set_state(memory_id, target, namespace=ns, run_id=run_id, actor=actor,
                                       action=f"state:{target}:approved",
                                       detail={"review_id": review_id, "note": note},
                                       expect_revision=memory.revision):
                    applied.append(f"{memory_id}:{target}")
                else:
                    skipped.append(f"{memory_id}: revision moved")
        self.store.update_run(run_id, status="completed", namespace=ns, finished=True,
                              stats={"review_id": review_id, "applied": applied,
                                     "skipped": skipped, "errors": errors})
        return {"review_id": review_id, "run_id": run_id, "applied": applied, "skipped": skipped,
                "errors": errors}

    def _open_reviews(self, plan, *, run_id, namespace):
        opened = 0
        for review in plan.reviews:
            if self.store.find_open_review(review.kind, review.memory_ids, namespace=namespace):
                continue
            self.store.open_review(kind=review.kind, memory_ids=list(review.memory_ids),
                                   reason=review.reason, run_id=run_id, confidence=review.confidence,
                                   merge_id=review.merge_id, detail=review.detail, namespace=namespace)
            opened += 1
        return opened


# --- helpers ----------------------------------------------------------------


def _clean(cls, data):
    """Keep only the fields a frozen action dataclass declares (forward compat)."""
    names = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in dict(data or {}).items() if k in names}


def _plan_payload(plan):
    return {"merges": [m.to_dict() for m in plan.merges], "prunes": [p.to_dict() for p in plan.prunes],
            "reviews": [r.to_dict() for r in plan.reviews], "notes": list(plan.notes),
            "considered": plan.considered, "skipped_unchanged": plan.skipped_unchanged,
            "namespace": plan.namespace, "model": plan.model}


def _plan_hash(plan):
    core = {"merges": [{k: v for k, v in m.to_dict().items() if k != "merge_id"} for m in plan.merges],
            "prunes": [p.to_dict() for p in plan.prunes]}
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _review_detail(store, review_id, *, namespace):
    for review in store.reviews(state=None, namespace=namespace, limit=1000):
        if review["id"] == int(review_id):
            return dict(review.get("detail") or {}), review
    return {}, None


def _downgrade(items, predicate):
    """Move matching actions to ``review``. Never drops an action from the plan."""
    return [replace(item, decision="review") if (item.will_apply and predicate(item)) else item
            for item in items]


def _target_state(decision, policy):
    """Where a retired memory lands: archived by default, ``pruned`` on request."""
    if policy.archive_instead_of_prune or decision != PRUNE:
        return "archived"
    return "pruned"


def _ranked(prunes):
    return sorted(prunes, key=lambda p: (-round(float(p.confidence), 4), p.memory_id))


def _ranked_merges(merges):
    return sorted(merges, key=lambda m: (-round(float(m.confidence), 4), -len(m.source_ids), m.merge_id))
