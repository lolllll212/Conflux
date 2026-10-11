"""A small, honest sample corpus for demos and tests.

The shapes here are the ones a cleanup pass has to tell apart: restatements of
the same fact, a typo-level near duplicate, a contradiction (same sentence,
different number), a stale record nobody has touched in a year, an expired
preference, and a protected instruction that must never be pruned.
"""

from datetime import timedelta

from .store import MemoryStore, utcnow

# (body, kind, topic, age_days, importance, access_count, expires_in_days, note)
SAMPLE_MEMORIES = (
    ("The staging box is at 10.0.0.5.", "fact", "hosts", 2, 1.0, 4, None, "fresh"),
    ("the staging box is at 10.0.0.5", "fact", "hosts", 9, 0.8, 1, None, "exact restatement"),
    ("Staging box: 10.0.0.5.", "fact", "hosts", 30, 0.7, 2, None, "punctuation-only variant"),
    ("The staging box used to be at 10.0.0.4.", "fact", "hosts", 210, 0.4, 1, None,
     "same shape, different number - must be reviewed, not merged"),
    ("Always restart nginx after a deploy.", "runbook", "deploy", 5, 1.0, 12, None, "fresh"),
    ("Restart nginx after every deploy.", "runbook", "deploy", 60, 0.9, 3, None, "near duplicate"),
    ("Builds are run with `make release` on Fridays.", "runbook", "deploy", 12, 0.6, 2, None,
     "related, not a duplicate"),
    ("Marta owns the billing service.", "fact", "owners", 3, 1.0, 5, None, "owner"),
    ("Billing is owned by Marta.", "fact", "owners", 45, 0.9, 1, None, "passive restatement"),
    ("Bob joined the crew in March.", "fact", "people", 40, 0.5, 1, None, "distinct person"),
    ("Alice joined the crew in March.", "fact", "people", 41, 0.5, 1, None,
     "same shape, different name - a false duplicate trap"),
    ("We use pytest for tests.", "note", "tooling", 400, 0.3, 1, None, "stale: untouched for a year"),
    ("Unit tests are run with pytest.", "note", "tooling", 120, 0.5, 2, None, "stale-ish paraphrase"),
    ("The user prefers dark mode in the editor.", "preference", "user", 300, 0.4, 1, None,
     "stale preference"),
    ("Session timeout must stay at 30 minutes.", "policy", "safety", 500, 1.0, 9, None,
     "protected kind: never pruned"),
    ("Never commit .env files.", "policy", "safety", 600, 1.0, 15, None, "protected instruction"),
    ("The API rate limit is 100 requests per minute.", "fact", "limits", 15, 0.8, 6, None, "fresh"),
    ("Rate limit for the API: 100 req/min.", "fact", "limits", 200, 0.6, 1, None, "abbreviated form"),
    ("Try Redis 7 for the cache spike.", "note", "experiments", 1, 0.5, 1,
     -3, "expired experiment (ttl already passed)"),
    ("Remember to rotate the deploy token.", "note", None, 3, 0.2, 1, None, "no provenance agent"),
)


def sample_dataset(now=None):
    """The corpus as plain dicts, with timestamps relative to ``now``."""
    reference = now or utcnow()
    rows = []
    for index, (body, kind, topic, age_days, importance, accesses, ttl, note) in enumerate(
            SAMPLE_MEMORIES):
        stamp = reference - timedelta(days=float(age_days))
        row = {
            "id": f"mem-sample-{index:02d}",
            "body": body, "kind": kind, "topic": topic, "importance": importance,
            "created_at": stamp.isoformat(timespec="microseconds"),
            "updated_at": stamp.isoformat(timespec="microseconds"),
            "last_seen_at": stamp.isoformat(timespec="microseconds"),
            "metadata": {"access_count_hint": accesses},
            "provenance": {"source": "sample", "agent": "seed-agent", "note": note},
            "access_count": accesses,
            "confidence": 0.9,
        }
        if ttl is not None:
            row["expires_at"] = (reference + timedelta(days=float(ttl))).isoformat(timespec="microseconds")
        else:
            row["expires_at"] = None
        rows.append(row)
    return tuple(rows)


def load_samples(store, *, namespace="default", rows=None, with_embedding=False, policy=None):
    """Write the corpus into ``store`` (idempotent per memory id)."""
    from .similar import hash_embed
    rows = rows if rows is not None else sample_dataset()
    known = {r["id"] for r in store.query("SELECT id FROM memories WHERE namespace=?", (namespace,))}
    loaded = 0
    for row in rows:
        if row["id"] in known:  # seeding twice must not fight over the primary key
            continue
        embedding = None
        if with_embedding:
            embedding = list(hash_embed(row["body"], dim=64))
        store.add(row["body"], kind=row["kind"], topic=row["topic"], namespace=namespace,
                  metadata=row["metadata"], provenance={k: v for k, v in row["provenance"].items()},
                  source_agent=row["provenance"].get("agent"), created_at=row["created_at"],
                  updated_at=row["updated_at"], last_seen_at=row["last_seen_at"],
                  expires_at=row["expires_at"], importance=row["importance"],
                  confidence=row["confidence"], memory_id=row["id"], on_duplicate="insert",
                  access_count=row["access_count"], embedding=embedding,
                  embedding_model="local-hash-64" if embedding else None, actor="seed")
        loaded += 1
    if policy is not None:
        store.set_policy(policy, namespace=namespace, updated_by="seed")
    return loaded


def build_sample_db(path, *, namespace="default", policy=None, with_embedding=False):
    """Create a ready-to-inspect database at ``path`` and return an open store."""
    store = MemoryStore(path, namespace=namespace)
    load_samples(store, namespace=namespace, with_embedding=with_embedding, policy=policy)
    return store


def conflict_pair():
    """Two memories whose only difference is a number - the classic trap."""
    return ("The retry budget is 3 attempts.", "The retry budget is 5 attempts.")


__all__ = ["SAMPLE_MEMORIES", "sample_dataset", "load_samples", "build_sample_db", "conflict_pair"]
