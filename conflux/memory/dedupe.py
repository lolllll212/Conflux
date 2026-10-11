"""Duplicate/near-duplicate detection: blocking, clustering, conflict flags.

O(n²) comparisons are avoided with an inverted index over shingle keys (two
memories that share no shingle cannot be similar), so a pass over a few thousand
memories stays quick and, more importantly, deterministic: candidates are
enumerated in sorted id order and every score is a pure function of the text.

Clusters are built from *near-duplicate* edges only. "Related" pairs (above
``related_similarity`` but below ``duplicate_similarity``) are reported inside
the cluster so a merge proposal can see the context, but they never merge on
their own. A cluster whose members contradict each other (different numbers,
opposite polarity) is marked ``conflicts`` and must be reviewed, not folded.
"""

from dataclasses import dataclass, field

from .similar import conflicts, similarity


@dataclass(frozen=True)
class PairScore:
    left: str
    right: str
    score: float
    duplicate: bool
    related: bool
    conflict: bool


@dataclass(frozen=True)
class Cluster:
    """A group of memories talking about the same thing, newest member first."""

    ids: tuple
    kind: str = "near"
    score: float = 0.0
    mean_score: float = 0.0
    conflicts: bool = False
    pairs: tuple = field(default_factory=tuple)
    topic: str = None

    @property
    def size(self):
        return len(self.ids)

    def contains(self, memory_id):
        return memory_id in self.ids

    def to_dict(self):
        return {"ids": list(self.ids), "kind": self.kind, "score": round(self.score, 4),
                "mean_score": round(self.mean_score, 4), "conflicts": self.conflicts,
                "topic": self.topic,
                "pairs": [[p.left, p.right, round(p.score, 4), p.conflict] for p in self.pairs]}


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Deterministic root: lexicographically smallest id.
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return True


def exact_groups(by_fingerprint):
    """``{fingerprint: [ids]}`` restricted to genuine repeats (>1 member)."""
    return {fp: sorted(ids) for fp, ids in by_fingerprint.items() if len(ids) > 1}


def candidate_pairs(ids, signatures, topics=None, min_shared=2, max_bucket=64,
                    max_pairs=200_000, topic_only=False):
    """Candidate pairs from a shared-shingle inverted index.

    A pair only becomes a candidate when the two memories share at least
    ``min_shared`` shingles, and buckets bigger than ``max_bucket`` (a word
    shared by half the corpus) are skipped rather than expanded quadratically.
    Same-topic groups are always compared, which is what a "related concepts"
    pass needs.
    """
    order = {memory_id: index for index, memory_id in enumerate(sorted(ids))}
    index = {}
    for memory_id in ids:
        for key in signatures[memory_id].blocking_keys():
            index.setdefault(key, set()).add(memory_id)
    shared = {}
    for _key, members in index.items():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        members = sorted(members)
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                pair = (left, right) if order[left] <= order[right] else (right, left)
                shared[pair] = shared.get(pair, 0) + 1
        if len(shared) > max_pairs:
            break
    pairs = {pair for pair, count in shared.items() if count >= max(1, int(min_shared))}
    if topics:
        groups = {}
        for memory_id, topic in topics.items():
            if topic:
                groups.setdefault(topic, []).append(memory_id)
        for members in groups.values():
            members = sorted(members)
            if len(members) > max_bucket:
                continue
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    if signatures[left].norm == signatures[right].norm:
                        continue
                    pairs.add((left, right) if order[left] <= order[right] else (right, left))
    return sorted(pairs, key=lambda p: (order[p[0]], order[p[1]]))


def score_pairs(pairs, signatures, policy):
    out = []
    for left, right in pairs:
        score = similarity(signatures[left], signatures[right],
                           text_weight=policy.text_weight, vector_weight=policy.vector_weight)
        out.append(PairScore(
            left=left, right=right, score=score,
            duplicate=score >= policy.duplicate_similarity,
            related=score >= policy.related_similarity,
            conflict=conflicts(signatures[left], signatures[right]),
        ))
    return out


def build_clusters(memories, policy, *, signatures=None, pair_scores=None):
    """Group memories into duplicate and related clusters.

    ``memories`` is a sequence of :class:`~conflux.memory.store.Memory` records.
    Returns ``(clusters, singleton_ids)``; every cluster with more than one
    member becomes a merge proposal, but only ``exact``/``near`` clusters are
    eligible for an automatic merge — a ``related`` cluster is always reviewed,
    because grouping "talking about the same thing" is not the same as "safe to
    fold".

    Two linkage stages, in a fixed order so the result never depends on
    dictionary iteration:

    1. union-find over edges at or above ``duplicate_similarity``;
    2. related edges (``related_similarity`` .. ``duplicate_similarity``) are
       merged into a group only under complete linkage — every cross pair must
       clear ``related_similarity`` and the group must stay under
       ``max_cluster_size``. That is what stops a chain of merely-similar
       neighbours from swallowing the corpus.
    """
    ids = [m.id for m in memories]
    if not ids:
        return [], []
    if signatures is None:
        signatures = {m.id: m.signature() for m in memories}
    topics = {m.id: m.topic for m in memories}
    pairs = candidate_pairs(ids, signatures, topics=topics, min_shared=policy.min_shared_shingles,
                            max_pairs=max(1000, int(policy.max_memories_per_run) ** 2),
                            topic_only=policy.compare_within_topic_only)
    scored = score_pairs(pairs, signatures, policy) if pair_scores is None else list(pair_scores)
    by_key = {(p.left, p.right): p for p in scored}

    # Same content fingerprint: the strongest possible duplicate edge, added
    # even when blocking skipped the pair.
    by_fingerprint = {}
    for memory in memories:
        by_fingerprint.setdefault(memory.fingerprint, []).append(memory.id)
    for members in by_fingerprint.values():
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                key = (left, right) if left <= right else (right, left)
                by_key.setdefault(key, PairScore(left=key[0], right=key[1], score=1.0,
                                                 duplicate=True, related=True, conflict=False))
    scored = sorted(by_key.values(), key=lambda p: (p.left, p.right))

    def pair_score(a, b):
        key = (a, b) if a <= b else (b, a)
        known = by_key.get(key)
        if known is not None:
            return known.score, known.conflict
        score = similarity(signatures[a], signatures[b], text_weight=policy.text_weight,
                           vector_weight=policy.vector_weight)
        return score, conflicts(signatures[a], signatures[b])

    uf = _UnionFind(ids)
    members = {i: [i] for i in ids}

    def unite(a, b):
        """Deterministic union: the smallest id stays the group root."""
        root_a, root_b = uf.find(a), uf.find(b)
        if root_a == root_b:
            return root_a
        combined = sorted(set(members.get(root_a, [root_a])) | set(members.get(root_b, [root_b])))
        root, other = sorted((root_a, root_b))
        uf.parent[root_a] = root
        uf.parent[root_b] = root
        members[root] = combined
        members.pop(other, None)
        return root

    for pair in scored:
        if pair.duplicate:
            unite(pair.left, pair.right)

    cap = max(2, int(policy.max_cluster_size))
    for pair in scored:
        if pair.duplicate or not pair.related:
            continue
        root_a, root_b = uf.find(pair.left), uf.find(pair.right)
        if root_a == root_b:
            continue
        group_a, group_b = members.get(root_a, [root_a]), members.get(root_b, [root_b])
        if len(group_a) + len(group_b) > cap:
            continue
        if any(pair_score(a, b)[0] < policy.related_similarity
               for a in group_a for b in group_b):
            continue
        unite(pair.left, pair.right)

    stamps = {m.id: (m.updated_at or "", m.created_at or "") for m in memories}
    topic_of = {m.id: m.topic for m in memories}
    final = {}
    for memory_id in ids:
        final.setdefault(uf.find(memory_id), []).append(memory_id)
    clusters, singletons = [], []
    for group in final.values():
        group = sorted(group, key=lambda i: (stamps.get(i, ("", ""))[0],
                                            stamps.get(i, ("", ""))[1], i), reverse=True)
        if len(group) == 1:
            singletons.append(group[0])
            continue
        for start in range(0, len(group), cap):
            chunk = group[start:start + cap]
            if len(chunk) == 1:
                singletons.append(chunk[0])
                continue
            clusters.append(_make_cluster(chunk, by_key, signatures, topic_of, policy))
    clusters.sort(key=lambda c: (-c.score, c.ids[0]))
    return clusters, sorted(singletons)


def _make_cluster(members, by_key, signatures, topic_of, policy):
    scores, pair_rows, conflicted = [], [], False
    for i, left in enumerate(members):
        for right in members[i + 1:]:
            key = (left, right) if left <= right else (right, left)
            pair = by_key.get(key)
            if pair is None:
                score = similarity(signatures[left], signatures[right], text_weight=policy.text_weight,
                                   vector_weight=policy.vector_weight)
                is_conflict = conflicts(signatures[left], signatures[right])
            else:
                score, is_conflict = pair.score, pair.conflict
            scores.append(score)
            conflicted = conflicted or is_conflict
            pair_rows.append(PairScore(left=key[0], right=key[1], score=score,
                                       duplicate=score >= policy.duplicate_similarity,
                                       related=score >= policy.related_similarity, conflict=is_conflict))
    exact = len({signatures[m].norm for m in members}) == 1
    weakest = min(scores) if scores else 1.0
    mean = sum(scores) / len(scores) if scores else 1.0
    kind = "exact" if exact else ("near" if weakest >= policy.duplicate_similarity else "related")
    return Cluster(ids=tuple(members), kind=kind, score=weakest, mean_score=mean,
                   conflicts=conflicted, pairs=tuple(pair_rows), topic=_common_topic(members, topic_of))


def _common_topic(members, topic_of):
    values = {topic_of.get(m) for m in members} - {None}
    return next(iter(values)) if len(values) == 1 else None
