"""Staleness rules: which memories look old enough to retire, and how surely.

Every rule returns a reason string and a confidence, and the verdict is a
*draft*, not an order: a memory is only prunable when its strongest rule is
confident, nothing in its cluster contradicts it, and the policy's safety floors
still hold after the prune (that arithmetic lives in ``cleanup.py``, because it
needs the whole-run view). Anything short of that becomes ``review`` — the
record stays active and an operator decides.

Two classes of rule matter:

* *hard* rules (``final=True``) come from the record itself — an explicit TTL
  that has passed. They decide on their own.
* *veto* rules (``veto=True``) — a contradicting sibling, missing provenance —
  can only downgrade, never upgrade. Uncertainty is never resolved silently.
"""

from dataclasses import dataclass, field

from .store import age_days, parse_ts, utcnow

RETAIN = "retain"
REVIEW = "review"
ARCHIVE = "archive"
PRUNE = "prune"

_ORDER = {RETAIN: 0, REVIEW: 1, ARCHIVE: 2, PRUNE: 3}


@dataclass(frozen=True)
class RuleHit:
    rule: str
    decision: str
    confidence: float
    reason: str
    age_days: float = None
    final: bool = False
    veto: bool = False


@dataclass(frozen=True)
class StaleVerdict:
    memory_id: str
    decision: str = RETAIN
    confidence: float = 0.0
    hits: tuple = field(default_factory=tuple)
    guarded_by: tuple = field(default_factory=tuple)

    @property
    def reasons(self):
        return tuple(h.reason for h in self.hits)

    @property
    def rules(self):
        return tuple(h.rule for h in self.hits)

    @property
    def actionable(self):
        return self.decision in (PRUNE, ARCHIVE)

    def to_dict(self):
        return {"memory_id": self.memory_id, "decision": self.decision,
                "confidence": round(self.confidence, 4), "reasons": list(self.reasons),
                "rules": list(self.rules), "guarded_by": list(self.guarded_by),
                "hits": [{"rule": h.rule, "decision": h.decision, "confidence": round(h.confidence, 4),
                          "reason": h.reason} for h in self.hits]}


def is_protected(memory, policy, now=None):
    """Which protection rules cover this memory (empty tuple = unprotected).

    ``now`` is the run's reference time: judging "recently written" against the
    wall clock instead would make a backfill over old data see every record as
    fresh (or stale) in the wrong way.
    """
    guards = []
    if memory.kind in tuple(policy.protect_kinds):
        guards.append(f"kind:{memory.kind}")
    if memory.topic and memory.topic in tuple(policy.protect_topics):
        guards.append(f"topic:{memory.topic}")
    metadata = dict(memory.metadata or {})
    for flag in ("pinned", "protect", "immutable"):
        if metadata.get(flag):
            guards.append(f"metadata:{flag}")
    recent = age_days(memory.created_at, now=now)
    if recent is not None and recent < float(policy.protect_recent_days):
        guards.append(f"younger_than:{policy.protect_recent_days}d")
    return tuple(guards)


def has_provenance(memory):
    prov = dict(memory.provenance or {})
    return bool(prov.get("origin_ids") or prov.get("uri") or prov.get("source")
                or prov.get("agent") or memory.source_agent)


def evaluate(memory, policy, *, now=None, conflict=False, protected=None):
    """Score one memory against every staleness rule."""
    reference = parse_ts(now) if now is not None else utcnow()
    hits = []
    guards = tuple(protected if protected is not None else is_protected(memory, policy,
                                                                        now=reference))

    if memory.state != "active":
        hits.append(RuleHit("already_inactive", RETAIN, 1.0, f"state={memory.state}"))

    if memory.expires_at:
        expires = parse_ts(memory.expires_at)
        if expires is not None:
            overdue = (reference.timestamp() - expires.timestamp()
                       - float(policy.ttl_grace_days) * 86400.0) / 86400.0
            if overdue > 0:
                hits.append(RuleHit("expired_ttl", PRUNE, min(0.99, 0.9 + 0.02 * overdue),
                                    f"ttl expired {overdue:.1f}d ago (grace {policy.ttl_grace_days}d)",
                                    overdue, final=True))

    freshest = max((s for s in (memory.last_seen_at, memory.updated_at, memory.created_at) if s),
                   default=None)
    age = age_days(freshest, now=reference)
    window = policy.days_for_kind(memory.kind)
    if age is not None and window > 0 and age > window:
        overshoot = age / window
        if overshoot >= 2.0:
            hits.append(RuleHit("stale_by_age", ARCHIVE, min(0.9, 0.6 + 0.1 * (overshoot - 2.0)),
                               f"untouched {age:.0f}d, {window:.0f}d window (x{overshoot:.1f})", age))
        else:
            hits.append(RuleHit("stale_by_age", REVIEW, 0.45,
                               f"untouched {age:.0f}d past {window:.0f}d window", age))

    unused = age_days(freshest, now=reference)
    if (unused is not None and unused > float(policy.unused_for_days)
            and memory.access_count <= int(policy.low_access_count)):
        weak = float(memory.importance) < float(policy.min_importance)
        hits.append(RuleHit("unused_low_value" if weak else "unused",
                            ARCHIVE if weak else REVIEW, 0.8 if weak else 0.4,
                            f"unseen {unused:.0f}d, access_count={memory.access_count}, "
                            f"importance={memory.importance:.2f}", unused))

    # Vetoes only ever downgrade a proposed action; on their own they must not
    # manufacture work for the reviewer queue on every untouched record.
    if any(h.decision in (PRUNE, ARCHIVE) for h in hits):
        if conflict:
            hits.append(RuleHit("conflicting_siblings", REVIEW, 0.95,
                                "a related memory contradicts this one (numbers or polarity differ)",
                                veto=True))
        if not has_provenance(memory):
            hits.append(RuleHit("no_provenance", REVIEW, 0.6,
                                "no provenance to justify discarding", veto=True))

    return decide(memory.id, hits, guards)


def decide(memory_id, hits, guards=()):
    """Collapse rule hits into one verdict, respecting protections and vetoes."""
    hits = tuple(hits)
    if not hits:
        return StaleVerdict(memory_id, RETAIN, 0.0, (), tuple(guards))
    if any(h.decision == RETAIN for h in hits):
        return StaleVerdict(memory_id, RETAIN, max(h.confidence for h in hits), hits, tuple(guards))
    if guards:
        return StaleVerdict(memory_id, RETAIN, 1.0, hits, tuple(guards))
    vetoed = [h for h in hits if h.veto]
    if vetoed:
        worst = max(vetoed, key=lambda h: h.confidence)
        return StaleVerdict(memory_id, REVIEW, worst.confidence, hits, tuple(guards))
    hard = [h for h in hits if h.final]
    pool = hard or [h for h in hits if h.decision != RETAIN] or list(hits)
    best = max(pool, key=lambda h: (_ORDER[h.decision], h.confidence))
    return StaleVerdict(memory_id, best.decision, best.confidence, hits, tuple(guards))
