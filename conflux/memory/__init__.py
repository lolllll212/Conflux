"""Self-pruning memory layer: local-first memory hygiene for AI agents.

Memories live in SQLite (see :mod:`conflux.memory.store`), are reviewed by a
periodic pass (:mod:`conflux.memory.cleanup`), and are never silently destroyed:
a duplicate becomes a *merge proposal* with the provenance of every source, and
anything uncertain becomes an entry in a *review queue*. The pass is driven by a
nightly scheduler and an HTTP surface that are both optional —
``conflux.memory.api`` needs FastAPI, this package needs nothing but the
standard library.

Typical use::

    from conflux.memory import CleanupService, MemoryStore, CleanupPolicy

    store = MemoryStore("memories.db")
    store.add("The staging box is at 10.0.0.5.", topic="hosts", source_agent="agent-a")
    service = CleanupService(store=store, policy=CleanupPolicy(model="heuristic"))
    report = service.run(dry_run=True)      # proposes, writes nothing to memories
    print(report.stats)

The HTTP service and the nightly worker::

    python -m conflux.memory serve --db memories.db
    python -m conflux.memory run --db memories.db --explain

``FastAPI`` is only imported by :mod:`conflux.memory.api`, so importing this
package never requires it. See ``docs/MEMORY.md`` for policy keys, the local
model contract, and the recovery procedure.
"""

from .cleanup import ApplyOutcome, CleanupEngine, CleanupPlan, MergeAction, PruneAction, RunReport
from .dedupe import Cluster, build_clusters
from .fixtures import build_sample_db, load_samples, sample_dataset
from .model import (CommandModel, HeuristicModel, LocalChatModel, MemoryModel, MergeRequest,
                    MergeSuggestion, NullModel, OllamaModel, SummaryRequest, build_model,
                    register_model, registered_models)
from .policy import CleanupPolicy
from .scheduler import CleanupScheduler
from .service import CleanupService
from .similar import content_fingerprint, fingerprint, hash_embed, normalize_text, signature, \
    text_similarity
from .staleness import ARCHIVE, PRUNE, RETAIN, REVIEW, StaleVerdict, evaluate
from .store import (ACTIVE, DuplicateMemory, Memory, MemoryNotFound, MemoryStore, PlanStale,
                    PolicyViolation, RunInProgress, StoreError)

__all__ = [
    "ACTIVE",
    "ARCHIVE",
    "ApplyOutcome",
    "CleanupEngine",
    "CleanupPolicy",
    "CleanupPlan",
    "CleanupScheduler",
    "CleanupService",
    "Cluster",
    "CommandModel",
    "DuplicateMemory",
    "HeuristicModel",
    "LocalChatModel",
    "Memory",
    "MemoryFabric",
    "MemoryModel",
    "MemoryNotFound",
    "MemoryStore",
    "MergeAction",
    "MergeRequest",
    "MergeSuggestion",
    "NullModel",
    "OllamaModel",
    "PRUNE",
    "PlanStale",
    "PolicyViolation",
    "PruneAction",
    "RETAIN",
    "REVIEW",
    "RunInProgress",
    "RunReport",
    "StaleVerdict",
    "StoreError",
    "SummaryRequest",
    "build_clusters",
    "build_model",
    "build_sample_db",
    "content_fingerprint",
    "evaluate",
    "fingerprint",
    "hash_embed",
    "load_samples",
    "normalize_text",
    "register_model",
    "registered_models",
    "sample_dataset",
    "signature",
    "text_similarity",
]

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy access to the two entry points that need extra packages."""
    if name == "MemoryFabric":
        from .bridge import MemoryFabric
        return MemoryFabric
    if name == "create_app":
        from .api import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
