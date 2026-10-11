"""SQLite schema for the self-pruning memory layer, plus a migration runner.

The database is the source of truth for memory content; Conflux's replicated
state is a derived view (see ``bridge.py``). Two rules shape the layout:

* nothing is destroyed by a cleanup pass — ``memories`` rows are only ever
  transitioned between states, and every transition lands in the append-only
  ``memory_events`` trail with a ``before``/``after`` snapshot so a merge or a
  prune can be replayed or reverted;
* a run is a record, not a side effect — ``runs`` holds the plan and its
  fingerprint, so a dry run can be inspected and applied later, and an
  identical plan is never applied twice.

Migrations are an ordered tuple of ``(version, sql_statements)`` applied
inside one transaction each. ``PRAGMA user_version`` is the current version.
"""

SCHEMA_DDL_V1 = (
    """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT NOT NULL,
        namespace TEXT NOT NULL DEFAULT 'default',
        kind TEXT NOT NULL DEFAULT 'note',
        topic TEXT,
        body TEXT NOT NULL,
        body_norm TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}',
        provenance TEXT NOT NULL DEFAULT '{}',
        source_agent TEXT,
        uri TEXT,
        confidence REAL,
        importance REAL NOT NULL DEFAULT 1.0,
        embedding BLOB,
        embedding_model TEXT,
        embedding_dim INTEGER,
        state TEXT NOT NULL DEFAULT 'active',
        superseded_by TEXT,
        merge_id INTEGER,
        idempotency_key TEXT,
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_seen_at TEXT,
        expires_at TEXT,
        access_count INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (namespace, id),
        CHECK (state IN ('active', 'merged', 'pruned', 'superseded', 'archived')),
        CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
        CHECK (length(body) > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS memories_fingerprint
        ON memories (namespace, fingerprint, state)
    """,
    """
    CREATE INDEX IF NOT EXISTS memories_state_updated
        ON memories (namespace, state, updated_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS memories_topic ON memories (namespace, topic)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS memories_idempotency
        ON memories (namespace, idempotency_key)
        WHERE idempotency_key IS NOT NULL
    """,

    # Append-only audit trail: one row per observation about a memory. Merges
    # and prunes keep the full pre-transition record in ``before`` so nothing
    # a cleanup pass touched is unrecoverable.
    """
    CREATE TABLE IF NOT EXISTS memory_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        namespace TEXT NOT NULL DEFAULT 'default',
        action TEXT NOT NULL,
        memory_id TEXT,
        target TEXT,
        run_id TEXT,
        actor TEXT,
        detail TEXT NOT NULL DEFAULT '{}',
        before TEXT,
        after TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS memory_events_memory ON memory_events (namespace, memory_id, seq)",
    "CREATE INDEX IF NOT EXISTS memory_events_run ON memory_events (run_id, seq)",

    # A proposed/applied merge: the merged body plus the provenance union of
    # every source memory. ``sources`` is the durable provenance document;
    # ``merge_links`` is the queryable projection of it.
    """
    CREATE TABLE IF NOT EXISTS merges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace TEXT NOT NULL DEFAULT 'default',
        run_id TEXT,
        created_at TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'proposed',
        topic TEXT,
        merged_body TEXT NOT NULL,
        merged_norm TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        merged_memory_id TEXT,
        confidence REAL,
        similarity REAL,
        reason TEXT,
        model TEXT,
        note TEXT,
        sources TEXT NOT NULL DEFAULT '[]',
        applied_at TEXT,
        reverted_at TEXT,
        cluster_key TEXT,
        CHECK (state IN ('proposed', 'applied', 'rejected', 'needs_review', 'reverted')),
        CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0))
    )
    """,
    # One open proposal per cluster: re-running a pass refreshes it instead of
    # stacking up duplicate suggestions.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS merges_open_cluster
        ON merges (namespace, cluster_key) WHERE state IN ('proposed', 'needs_review')
    """,
    """
    CREATE TABLE IF NOT EXISTS merge_links (
        merge_id INTEGER NOT NULL REFERENCES merges (id) ON DELETE CASCADE,
        namespace TEXT NOT NULL,
        memory_id TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'source',
        PRIMARY KEY (merge_id, memory_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS merge_links_memory ON merge_links (namespace, memory_id)",

    # Run ledger: the plan, its fingerprint and its outcome.
    """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL DEFAULT 'default',
        trigger TEXT NOT NULL DEFAULT 'manual',
        mode TEXT NOT NULL DEFAULT 'dry_run',
        status TEXT NOT NULL DEFAULT 'running',
        started_at TEXT NOT NULL,
        finished_at TEXT,
        actor TEXT,
        model TEXT,
        policy TEXT NOT NULL DEFAULT '{}',
        stats TEXT NOT NULL DEFAULT '{}',
        error TEXT,
        CHECK (mode IN ('dry_run', 'apply')),
        CHECK (status IN ('running', 'completed', 'failed', 'aborted'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS runs_namespace ON runs (namespace, started_at)",

    # Versioned policy, one row per namespace.
    """
    CREATE TABLE IF NOT EXISTS settings (
        namespace TEXT PRIMARY KEY,
        policy TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        updated_by TEXT
    )
    """,

    # Cross-process run lock with a lease: a crashed holder stops blocking
    # once the lease expires.
    """
    CREATE TABLE IF NOT EXISTS run_lock (
        namespace TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        run_id TEXT,
        acquired_at TEXT NOT NULL,
        lease_expires_at TEXT NOT NULL
    )
    """,
)

# v2 adds the two mechanisms that make repeated passes cheap and uncertain
# material safe to park: a review cursor (so a record whose fingerprint was
# already reviewed and unchanged is skipped instead of reprocessed) and a
# durable review queue (so nothing uncertain is decided silently). A plan
# fingerprint on ``runs`` lets an apply be replayed safely.
SCHEMA_DDL_V2 = (
    """
    CREATE TABLE IF NOT EXISTS review_cursor (
        namespace TEXT NOT NULL,
        memory_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,
        run_id TEXT,
        verdict TEXT,
        PRIMARY KEY (namespace, memory_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace TEXT NOT NULL DEFAULT 'default',
        run_id TEXT,
        created_at TEXT NOT NULL,
        kind TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'open',
        confidence REAL,
        reason TEXT,
        memory_ids TEXT NOT NULL DEFAULT '[]',
        merge_id INTEGER,
        detail TEXT NOT NULL DEFAULT '{}',
        resolved_at TEXT,
        resolved_by TEXT,
        resolution_note TEXT,
        CHECK (kind IN ('merge', 'prune', 'archive', 'contradiction', 'retain')),
        CHECK (state IN ('open', 'approved', 'rejected', 'expired')),
        UNIQUE (namespace, run_id, kind, memory_ids)
    )
    """,
    "CREATE INDEX IF NOT EXISTS reviews_state ON reviews (namespace, state, created_at)",
    """
    CREATE TABLE IF NOT EXISTS scheduler_state (
        namespace TEXT PRIMARY KEY,
        enabled INTEGER NOT NULL DEFAULT 1,
        timezone TEXT NOT NULL DEFAULT 'UTC',
        hour INTEGER NOT NULL DEFAULT 0,
        minute INTEGER NOT NULL DEFAULT 0,
        last_fired_slot TEXT,
        last_run_id TEXT,
        last_run_at TEXT,
        last_status TEXT,
        next_run_at TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    "ALTER TABLE runs ADD COLUMN plan_hash TEXT",
    "ALTER TABLE runs ADD COLUMN plan_json TEXT",
    "CREATE INDEX IF NOT EXISTS runs_plan_hash ON runs (namespace, plan_hash)",
    # Retention: pruned/archived rows stay in the database until an explicit
    # hard-delete pass clears them, which is never automatic by default.
    """
    CREATE TABLE IF NOT EXISTS retention (
        namespace TEXT PRIMARY KEY,
        hard_deleted_at TEXT,
        hard_deleted_count INTEGER NOT NULL DEFAULT 0,
        last_report TEXT
    )
    """,
)

MIGRATIONS = ((1, SCHEMA_DDL_V1), (2, SCHEMA_DDL_V2))

SCHEMA_VERSION = MIGRATIONS[-1][0]

PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=30000",
    "PRAGMA synchronous=FULL",
)


def user_version(conn):
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def set_user_version(conn, version):
    conn.execute(f"PRAGMA user_version={int(version)}")


def apply_migrations(conn, target_version=None):
    """Bring ``conn`` up to ``target_version`` (default: latest).

    Every step runs in its own transaction, so a failure mid-upgrade leaves
    the database at the last fully applied version. Returns the list of
    versions applied.
    """
    target = SCHEMA_VERSION if target_version is None else int(target_version)
    if target > SCHEMA_VERSION:
        raise ValueError(f"unknown schema version {target} (latest is {SCHEMA_VERSION})")
    applied = []
    current = user_version(conn)
    for version, statements in MIGRATIONS:
        if version <= current or version > target:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                conn.execute(statement)
            set_user_version(conn, version)
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        applied.append(version)
    return applied
