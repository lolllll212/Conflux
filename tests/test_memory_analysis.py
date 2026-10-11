import math

from conflux.memory.policy import CleanupPolicy
from conflux.memory.dedupe import build_clusters, candidate_pairs, exact_groups
from conflux.memory.similar import (blend, char_shingles, cosine, conflicts, content_fingerprint,
                                    fold_unique_sentences, hash_embed, jaccard, normalize_text,
                                    numbers_in, pack_vector, polarity, signature, split_sentences,
                                    substitutions, text_similarity, unpack_vector, word_shingles)
from conflux.memory.store import Memory, MemoryStore
from conflux.memory.staleness import ARCHIVE, PRUNE, RETAIN, REVIEW, evaluate, has_provenance, is_protected


def memory(body, *, kind="note", topic=None, state="active", created_at=None, updated_at=None,
           access_count=1, importance=1.0, provenance=None, expires_at=None, last_seen_at=None,
           memory_id=None, revision=1):
    return Memory(id=memory_id or body[:8], body=body, kind=kind, topic=topic, state=state,
                  body_norm=normalize_text(body), fingerprint=content_fingerprint(body, kind, topic),
                  created_at=created_at, updated_at=updated_at, last_seen_at=last_seen_at,
                  access_count=access_count, importance=importance,
                  provenance=provenance or {}, expires_at=expires_at, revision=revision)


# --- normalization and fingerprints ------------------------------------------


def test_normalize_text_is_aggressive_but_stable():
    assert normalize_text("  Deploy   the API, FIRST! ") == "deploy the api first"
    # straight double quotes count as punctuation and are dropped; apostrophes stay
    assert normalize_text("\u201cQuotes\u201d and \u2019apostrophes\u2019") == "quotes and 'apostrophes'"
    assert normalize_text("caf\u00e9") == normalize_text("cafe\u0301")  # NFC folds the accent
    assert normalize_text(None) == ""
    # casefold, not lower: German sharp s maps to 'ss'
    assert normalize_text("STRASSE") == "strasse"


def test_fingerprint_collapses_punctuation_but_not_kind():
    a = content_fingerprint("The box is at 10.0.0.5.", "fact", "hosts")
    b = content_fingerprint("the box IS at 10.0.0.5", "fact", "hosts")
    assert a == b
    assert a != content_fingerprint("The box is at 10.0.0.5.", "note", "hosts")
    assert a != content_fingerprint("The box is at 10.0.0.5.", "fact", "prod")


def test_shingles_and_jaccard():
    grams = word_shingles("alpha beta gamma")
    assert {"alpha", "alpha beta", "alpha beta gamma"} <= grams
    chars = char_shingles("deploy")
    assert "depl" in chars and len(chars) == 3
    assert jaccard(frozenset({1, 2}), frozenset({1, 2})) == 1.0
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({1}), frozenset()) == 0.0


# --- similarity ---------------------------------------------------------------


def test_identical_text_scores_one_and_unrelated_scores_low():
    a = signature("The build takes 12 minutes.")
    b = signature("the build takes 12 minutes")
    c = signature("Marta owns the billing service.")
    assert text_similarity(a, b) == text_similarity(a, a) == 1.0
    assert text_similarity(a, c) < 0.2


def test_reworded_pair_beats_unrelated_pairs():
    reworded = text_similarity(signature("Use postgres 15 for the primary database."),
                              signature("The primary database is postgres 15."))
    unrelated = text_similarity(signature("Use postgres 15 for the primary database."),
                                signature("The build takes 12 minutes."))
    assert reworded > 0.6 > unrelated


def test_containment_rescues_a_subset_restatement():
    short = signature("Restart nginx after a deploy.")
    long = signature("Always restart nginx after a deploy. Nginx needs 2 workers.")
    assert text_similarity(short, long) > text_similarity(short, signature("Nginx is a proxy."))


def test_numbers_and_polarity_helpers():
    assert numbers_in("100 requests/min, 1,500 total, 3.5 days") == ["100", "1500", "3.5"]
    assert polarity("do not commit .env") == -1
    assert polarity("always commit .env") == 1
    assert polarity("commit .env sometimes") == 0


def test_conflicts_flags_the_false_duplicate_traps():
    assert conflicts(signature("Session timeout is 30 minutes."),
                     signature("Session timeout is 45 minutes.")) is True
    assert conflicts(signature("Alice joined the crew in March."),
                     signature("Bob joined the crew in March.")) is True
    assert conflicts(signature("Never restart the box."),
                     signature("Always restart the box.")) is True
    # a restatement is not a contradiction
    assert conflicts(signature("Marta owns the billing service."),
                     signature("Billing is owned by Marta.")) is False
    # ...nor is a longer record that merely adds a measurement
    assert conflicts(signature("Restart nginx after a deploy."),
                     signature("Restart nginx after a deploy. Nginx needs 2 workers.")) is False
    # a length-changing difference is a restatement, not a substitution
    assert substitutions(signature("marta owns billing"),
                         signature("marta owns the billing")) is None


def test_hash_embedding_is_deterministic_and_unit_length():
    a = hash_embed("the api is rate limited to 100 rpm", dim=64)
    b = hash_embed("the api is rate limited to 100 rpm", dim=64)
    assert list(a) == list(b)
    assert abs(math.sqrt(sum(v * v for v in a)) - 1.0) < 1e-4
    near = hash_embed("the api is rate limited to 100 rpm per caller", dim=64)
    far = hash_embed("the user prefers dark mode", dim=64)
    assert cosine(a, near) > cosine(a, far)
    assert cosine(a, hash_embed("same text", dim=32)) is None  # dimension mismatch


def test_vectors_pack_and_unpack_through_sqlite():
    vector = hash_embed("round trip", dim=32)
    blob = pack_vector(vector)
    assert len(blob) == 32 * 4
    restored = unpack_vector(blob, 32)
    assert list(restored) == list(vector)
    assert pack_vector(None) is None and unpack_vector(None) is None


def test_blend_renormalizes_and_ignores_missing_vectors():
    assert blend(0.8, None, 0.75, 0.25) == 0.8
    assert abs(blend(0.8, 0.8, 0.75, 0.25) - 0.8) < 1e-9
    assert blend(1.0, 0.0, 0.5, 0.5) == 0.5
    assert blend(1.0, 0.0, 0.0, 0.0) == 1.0


# --- the model-free merge -----------------------------------------------------


def test_fold_unique_sentences_dedupes_and_keeps_boundaries():
    merged = fold_unique_sentences(["Restart nginx after a deploy. Nginx needs 2 workers.",
                                    "Restart nginx after a deploy.",
                                    "Nginx needs 2 workers."], prefix="")
    assert merged.count("Restart nginx") == 1
    assert len(split_sentences(merged)) == 2
    with_prefix = fold_unique_sentences(["a", "b"], max_chars=700, prefix="[merged]")
    assert with_prefix.startswith("[merged] ")


def test_fold_respects_the_character_budget_without_losing_order():
    bodies = [f"sentence number {i}." for i in range(40)]
    merged = fold_unique_sentences(bodies, max_chars=120, prefix="")
    assert len(merged) <= 120
    assert merged.startswith("sentence number 0.")


# --- staleness ----------------------------------------------------------------


def test_staleness_uses_the_newest_signal_available():
    policy = CleanupPolicy(stale_after_days=30)
    provenance = {"agent": "seed"}
    old = memory("stale fact", created_at="2020-01-01T00:00:00+00:00",
                 updated_at="2020-01-01T00:00:00+00:00", provenance=provenance)
    verdict = evaluate(old, policy, now="2026-01-01T00:00:00+00:00")
    assert verdict.decision == ARCHIVE
    assert "stale_by_age" in verdict.rules
    # but a memory that was merely *seen* recently is not stale
    seen = memory("stale fact", created_at="2020-01-01T00:00:00+00:00",
                  updated_at="2020-01-01T00:00:00+00:00", last_seen_at="2025-12-20T00:00:00+00:00",
                  provenance=provenance)
    assert evaluate(seen, policy, now="2026-01-01T00:00:00+00:00").decision == RETAIN


def test_expired_ttl_is_a_hard_signal():
    policy = CleanupPolicy(stale_after_days=10_000)
    expired = memory("temporary", created_at="2026-01-01T00:00:00+00:00",
                     updated_at="2026-01-01T00:00:00+00:00", expires_at="2026-02-01T00:00:00+00:00",
                     provenance={"agent": "a"})
    verdict = evaluate(expired, policy, now="2026-03-01T00:00:00+00:00")
    assert verdict.decision == PRUNE and verdict.confidence > 0.9
    assert "expired_ttl" in verdict.rules
    # the grace window defers it
    assert evaluate(expired, policy.merged({"ttl_grace_days": 90}),
                     now="2026-03-01T00:00:00+00:00").decision == RETAIN


def test_protected_kinds_and_fresh_records_are_never_prunable():
    policy = CleanupPolicy(protect_kinds=("policy",), protect_recent_days=30)
    protected = memory("never commit .env", kind="policy", created_at="2020-01-01T00:00:00+00:00",
                       updated_at="2020-01-01T00:00:00+00:00", provenance={"agent": "a"})
    verdict = evaluate(protected, policy, now="2026-01-01T00:00:00+00:00")
    assert verdict.decision == RETAIN and "kind:policy" in verdict.guarded_by
    fresh = memory("just written", created_at="2025-12-25T00:00:00+00:00",
                   updated_at="2025-12-25T00:00:00+00:00")
    guards = is_protected(fresh, policy, now="2026-01-01T00:00:00+00:00")
    assert guards == ("younger_than:30d",)
    assert is_protected(fresh, policy, now="2027-01-01T00:00:00+00:00") == ()


def test_pinned_and_protected_metadata_flags_are_honoured():
    policy = CleanupPolicy(stale_after_days=1)
    row = Memory(id="pinned-1", body="x", body_norm="x", fingerprint="f", state="active",
                 created_at="2020-01-01T00:00:00+00:00", updated_at="2020-01-01T00:00:00+00:00",
                 metadata={"pinned": True}, provenance={"agent": "a"})
    assert evaluate(row, policy, now="2026-01-01T00:00:00+00:00").decision == RETAIN


def test_missing_provenance_downgrades_a_prune_to_review():
    policy = CleanupPolicy(stale_after_days=1)
    old = "2020-01-01T00:00:00+00:00"
    bare = memory("no provenance", created_at=old, updated_at=old)
    verdict = evaluate(bare, policy, now="2026-01-01T00:00:00+00:00")
    assert verdict.decision == REVIEW and "no_provenance" in verdict.rules
    assert has_provenance(memory("with", provenance={"uri": "chat://1"}, created_at=old)) is True


def test_conflict_in_a_cluster_forces_review():
    policy = CleanupPolicy(stale_after_days=1)
    old = "2020-01-01T00:00:00+00:00"
    row = memory("timeout is 30", created_at=old, updated_at=old, provenance={"agent": "a"})
    assert evaluate(row, policy, now="2026-01-01T00:00:00+00:00").decision == ARCHIVE
    assert evaluate(row, policy, now="2026-01-01T00:00:00+00:00",
                    conflict=True).decision == REVIEW


def test_a_veto_alone_must_not_create_review_work():
    policy = CleanupPolicy(stale_after_days=10_000)
    fresh = memory("no provenance and nothing stale", created_at="2026-01-01T00:00:00+00:00",
                   updated_at="2026-01-01T00:00:00+00:00")
    verdict = evaluate(fresh, policy, now="2026-01-02T00:00:00+00:00")
    assert verdict.decision == RETAIN and verdict.reasons == ()


# --- clustering ---------------------------------------------------------------


def _cluster_of(bodies, **policy_kwargs):
    rows = [memory(body, memory_id=f"m{i}") for i, body in enumerate(bodies)]
    return build_clusters(rows, CleanupPolicy(**policy_kwargs))


def test_clusters_group_rewordings_and_leave_other_topics_alone():
    clusters, singles = _cluster_of([
        "The staging box is at 10.0.0.5.",
        "the staging box is at 10.0.0.5",
        "The staging box sits at 10.0.0.5.",
        "Marta owns the billing service.",
    ])
    assert [c.ids for c in clusters] == [("m2", "m1", "m0")]
    assert clusters[0].kind == "near"
    # members are ordered newest-first by the caller's timestamp ordering, and the
    # cluster score is its weakest link
    assert clusters[0].score >= 0.78
    assert singles == ["m3"]


def test_identical_bodies_always_group_even_when_blocking_misses():
    rows = [memory("alpha beta gamma delta", memory_id="x1", topic=None,
                  updated_at="2026-01-02T00:00:00+00:00"),
            memory("alpha beta gamma delta", memory_id="x2", topic=None,
                   updated_at="2026-01-01T00:00:00+00:00")]
    clusters, singles = build_clusters(rows, CleanupPolicy())
    assert [c.ids for c in clusters] == [("x1", "x2")]  # newest member first
    assert clusters[0].kind == "exact" and clusters[0].score == 1.0
    assert singles == []


def test_unrelated_memories_produce_no_clusters():
    rows = [memory("the build takes 12 minutes", memory_id="a"),
            memory("Marta owns billing", memory_id="b"),
            memory("prefer dark mode in the editor", memory_id="c")]
    clusters, singles = build_clusters(rows, CleanupPolicy())
    assert clusters == []
    assert singles == ["a", "b", "c"]


def test_related_pairs_cannot_chain_into_a_giant_cluster():
    # c0-c1 and c1-c2 share words, but the ends are unrelated: complete linkage
    # refuses a transitive merge, so nothing swallows the whole corpus
    bodies = ["alpha one beta two gamma three", "alpha one beta two gamma four",
              "delta five beta two gamma three", "delta five epsilon six zeta seven"]
    rows = [memory(b, memory_id=f"c{i}") for i, b in enumerate(bodies)]
    clusters, singles = build_clusters(rows, CleanupPolicy(related_similarity=0.4))
    assert all("c0" not in c.ids or "c3" not in c.ids for c in clusters)
    for cluster in clusters:
        # every member pair of a cluster clears the related bar
        assert cluster.score >= 0.4
    assert "c3" in singles


def test_cluster_size_cap_splits_large_groups():
    rows = [memory("the staging box is at 10.0.0.5", memory_id=f"z{i}") for i in range(7)]
    clusters, singles = build_clusters(rows, CleanupPolicy(max_cluster_size=3))
    assert {len(c.ids) for c in clusters} <= {3}
    assert sum(len(c.ids) for c in clusters) + len(singles) == 7


def test_cluster_flags_contradictions_and_is_deterministic():
    rows = [memory("Session timeout is 30 minutes.", memory_id="s1"),
            memory("Session timeout is 45 minutes.", memory_id="s2")]
    first, _ = build_clusters(rows, CleanupPolicy(related_similarity=0.3, duplicate_similarity=0.3))
    again, _ = build_clusters(list(reversed(rows)),
                              CleanupPolicy(related_similarity=0.3, duplicate_similarity=0.3))
    assert first[0].conflicts is True
    assert sorted(first[0].ids) == sorted(again[0].ids)


def test_candidate_pairs_use_the_inverted_index():
    rows = [memory("alpha beta", memory_id="a"), memory("gamma delta", memory_id="b")]
    sigs = {r.id: r.signature() for r in rows}
    # no shared shingle => not a candidate
    assert candidate_pairs(["a", "b"], sigs, topics={}, min_shared=2) == []
    # ...unless they share a topic, which forces the comparison
    assert candidate_pairs(["a", "b"], sigs, topics={"a": "t", "b": "t"}, min_shared=2) == [("a", "b")]
    same = [memory("alpha beta", memory_id="a"), memory("alpha beta", memory_id="b")]
    pairs = candidate_pairs(["a", "b"], {r.id: r.signature() for r in same}, min_shared=2)
    assert pairs == [("a", "b")]


def test_exact_groups_reports_only_repeats():
    groups = {"fp1": ["a", "b"], "fp2": ["c"]}
    assert exact_groups(groups) == {"fp1": ["a", "b"]}
