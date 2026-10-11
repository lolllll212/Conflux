"""Versioned, validated cleanup/retention policy.

A policy is a frozen value object so a run is reproducible: the exact dict is
snapshotted into ``runs.policy`` before the pass starts, and the same snapshot
can be replayed later. Everything the pass may decide is here — duplicate
thresholds, staleness rules, retention windows, safety floors and schedule
knobs — so no call site needs ad-hoc magic numbers.
"""

import dataclasses
import json
from dataclasses import dataclass, field, replace
from zoneinfo import available_timezones, ZoneInfo, ZoneInfoNotFoundError

_PROTECTED_KINDS = ("policy", "safety", "instruction", "credential")


@dataclass(frozen=True)
class CleanupPolicy:
    """Knobs for one cleanup pass."""

    # --- scope / model -----------------------------------------------------
    model: str = "null"
    model_options: dict = field(default_factory=dict)

    # --- duplicate detection ----------------------------------------------
    duplicate_similarity: float = 0.78
    related_similarity: float = 0.45
    min_shared_shingles: int = 2
    use_embeddings: bool = True
    embedding_dim: int = 256
    text_weight: float = 0.75
    vector_weight: float = 0.25
    max_cluster_size: int = 12
    compare_within_topic_only: bool = False

    # --- merging ------------------------------------------------------------
    auto_apply_merges: bool = False
    auto_merge_exact: bool = True
    auto_apply_confidence: float = 0.85
    max_merged_chars: int = 700
    merged_prefix: str = "[merged]"

    # --- staleness ----------------------------------------------------------
    stale_after_days: float = 180.0
    unused_for_days: float = 90.0
    low_access_count: int = 1
    min_importance: float = 0.2
    ttl_grace_days: float = 0.0
    stale_after_days_by_kind: dict = field(default_factory=dict)
    protect_recent_days: float = 1.0

    # --- pruning / retention -----------------------------------------------
    allow_auto_prune: bool = False
    archive_instead_of_prune: bool = True
    require_provenance_for_prune: bool = True
    prune_requires_no_conflict: bool = True
    prune_alone_in_topic: bool = False
    hard_delete_after_days: float = 90.0
    allow_hard_delete: bool = False

    # --- safety floors ------------------------------------------------------
    min_active_memories: int = 8
    max_prune_fraction: float = 0.25
    max_merge_fraction: float = 0.5
    protect_kinds: tuple = _PROTECTED_KINDS
    protect_topics: tuple = ()
    review_unchanged: bool = True
    max_memories_per_run: int = 5000

    # --- background schedule ------------------------------------------------
    schedule_enabled: bool = True
    schedule_timezone: str = "UTC"
    schedule_hour: int = 0
    schedule_minute: int = 0
    schedule_interval_minutes: float = 0.0
    schedule_grace_minutes: float = 120.0

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def field_names(cls):
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls, data, known_only=False):
        """Build a policy from a mapping, coercing types and rejecting junk.

        ``known_only`` drops unknown keys instead of raising — used when a
        stored policy was written by a newer version of this package.
        """
        if data is None:
            return cls()
        if isinstance(data, CleanupPolicy):
            return data
        if not isinstance(data, dict):
            raise TypeError("policy must be a mapping")
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = [k for k in data if k not in fields]
        if unknown and not known_only:
            raise ValueError(f"unknown policy keys: {', '.join(sorted(unknown))}")
        values = {}
        for key, raw in data.items():
            if key not in fields:
                continue
            values[key] = _coerce(key, raw, fields[key])
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid policy: {exc}") from None

    @classmethod
    def from_json(cls, text, known_only=False):
        return cls.from_dict(json.loads(text), known_only=known_only)

    def merged(self, updates, known_only=False):
        """Return a copy with ``updates`` applied.

        Unknown keys are an error by default: a typo in a submitted setting must
        not read as "accepted, and silently did nothing".
        """
        fields = {f.name: f for f in dataclasses.fields(self)}
        submitted = dict(updates or {})
        unknown = [k for k in submitted if k not in fields]
        if unknown and not known_only:
            raise ValueError(f"unknown policy keys: {', '.join(sorted(unknown))}")
        clean = {}
        for key, value in submitted.items():
            if key not in fields:
                continue
            clean[key] = _coerce(key, value, fields[key])
        return replace(self, **clean) if clean else self

    def validate(self):
        """Raise ``ValueError`` if the policy cannot be used for a run."""
        problems = []
        for name in ("duplicate_similarity", "related_similarity", "auto_apply_confidence",
                     "text_weight", "vector_weight", "max_prune_fraction", "max_merge_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{name} must be numeric")
            elif not 0.0 <= float(value) <= 1.0:
                problems.append(f"{name} must be within [0, 1] (got {value!r})")
        if self.duplicate_similarity < self.related_similarity:
            problems.append("duplicate_similarity must be >= related_similarity")
        if self.text_weight + self.vector_weight <= 0:
            problems.append("text_weight + vector_weight must be > 0")
        for name in ("stale_after_days", "unused_for_days", "ttl_grace_days", "protect_recent_days",
                     "hard_delete_after_days"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                problems.append(f"{name} must be a non-negative number of days")
        for name in ("max_cluster_size", "max_merged_chars", "max_memories_per_run",
                     "min_shared_shingles", "low_access_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                problems.append(f"{name} must be a positive integer")
        if not isinstance(self.min_active_memories, int) or isinstance(self.min_active_memories, bool) \
                or self.min_active_memories < 0:
            # 0 means "no floor": the pass may retire every memory it can prove
            # is redundant. Only a deliberate choice, hence the explicit value.
            problems.append("min_active_memories must be a non-negative integer")
        if self.max_merged_chars < 40:
            problems.append("max_merged_chars must be at least 40")
        if not 0 <= int(self.schedule_hour) <= 23:
            problems.append("schedule_hour must be within [0, 23]")
        if not 0 <= int(self.schedule_minute) <= 59:
            problems.append("schedule_minute must be within [0, 59]")
        if self.schedule_interval_minutes and self.schedule_interval_minutes < 0:
            problems.append("schedule_interval_minutes must be >= 0")
        if self.schedule_grace_minutes <= 0:
            problems.append("schedule_grace_minutes must be > 0")
        try:
            ZoneInfo(self.schedule_timezone)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            known = len(available_timezones())
            problems.append(f"unknown timezone {self.schedule_timezone!r} ({known} zones available)")
        if not isinstance(self.model, str) or not self.model:
            problems.append("model must be a non-empty string")
        if not isinstance(self.model_options, dict):
            problems.append("model_options must be a mapping")
        for name in ("protect_kinds", "protect_topics"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
                problems.append(f"{name} must be a list of strings")
        for key, value in dict(self.stale_after_days_by_kind or {}).items():
            if not isinstance(key, str) or not isinstance(value, (int, float)) or value < 0:
                problems.append(f"stale_after_days_by_kind[{key!r}] must be a non-negative number")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def days_for_kind(self, kind):
        """Staleness window for ``kind``, honouring per-kind overrides."""
        overrides = dict(self.stale_after_days_by_kind or {})
        if kind in overrides:
            return float(overrides[kind])
        return float(self.stale_after_days)

    def schema(self):
        """Every knob with its type, default, group and one-line help."""
        return field_schema()

    def summary(self):
        """The knobs an operator actually reads in ``/status`` output."""
        return {
            "model": self.model,
            "duplicate_similarity": self.duplicate_similarity,
            "related_similarity": self.related_similarity,
            "use_embeddings": self.use_embeddings,
            "auto_apply_merges": self.auto_apply_merges,
            "auto_merge_exact": self.auto_merge_exact,
            "auto_apply_confidence": self.auto_apply_confidence,
            "allow_auto_prune": self.allow_auto_prune,
            "archive_instead_of_prune": self.archive_instead_of_prune,
            "stale_after_days": self.stale_after_days,
            "unused_for_days": self.unused_for_days,
            "min_active_memories": self.min_active_memories,
            "max_prune_fraction": self.max_prune_fraction,
            "allow_hard_delete": self.allow_hard_delete,
            "review_unchanged": self.review_unchanged,
            "schedule": {
                "enabled": self.schedule_enabled,
                "timezone": self.schedule_timezone,
                "at": f"{int(self.schedule_hour):02d}:{int(self.schedule_minute):02d}",
                "interval_minutes": self.schedule_interval_minutes,
                "grace_minutes": self.schedule_grace_minutes,
            },
        }


def _coerce(name, raw, info):
    """Coerce a stored/submitted value to the type of the field's default."""
    if info.default is not None:
        proto = info.default
    elif info.default_factory is not None:
        proto = info.default_factory()
    else:
        proto = None
    if raw is None:
        return proto
    if isinstance(proto, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(proto, float):
        if isinstance(raw, bool) or isinstance(raw, (list, tuple, dict)):
            raise ValueError(f"{name} must be a number")
        return float(raw)
    if isinstance(proto, int):
        if isinstance(raw, bool) or isinstance(raw, (list, tuple, dict)):
            raise ValueError(f"{name} must be an integer")
        return int(raw)
    if isinstance(proto, tuple):
        if isinstance(raw, str):
            return tuple(x for x in (part.strip() for part in raw.split(",")) if x)
        return tuple(raw)
    if isinstance(proto, dict):
        if isinstance(raw, str):
            parsed = json.loads(raw or "{}")
            if not isinstance(parsed, dict):
                raise ValueError(f"{name} must be a mapping")
            return parsed
        return dict(raw)
    if isinstance(proto, str):
        return str(raw)
    return raw

# One line per knob, grouped the way the dataclass is. A settings endpoint that
# returns bare defaults makes every caller read the source; this is the doc that
# travels with the API, the CLI and /status.
FIELD_HELP = {
    "model": ("scope", "Merge-suggestion model: null, heuristic, ollama[:name], "
              "local[:name@url] or module.path:Class. Loopback only."),
    "model_options": ("scope", "Extra kwargs handed to the model constructor (timeout, temperature...)."),
    "duplicate_similarity": ("duplicates", "Blend score at or above which two memories are the same fact "
                            "worth folding."),
    "related_similarity": ("duplicates", "Blend score at or above which memories are grouped into one "
                           "cluster but never folded automatically."),
    "min_shared_shingles": ("duplicates", "How many shingles (or a shared topic) two records must have in "
                            "common before they are even compared."),
    "use_embeddings": ("duplicates", "Blend the lexical score with cosine similarity when vectors are stored."),
    "embedding_dim": ("duplicates", "Dimension of the built-in hashing embedding; stored vectors must match."),
    "text_weight": ("duplicates", "Weight of the lexical score in the blend."),
    "vector_weight": ("duplicates", "Weight of the embedding score in the blend."),
    "max_cluster_size": ("duplicates", "Largest cluster one pass will consider; larger groups are split."),
    "compare_within_topic_only": ("duplicates", "Never compare memories from different topics."),
    "auto_apply_merges": ("merging", "Apply merge proposals without a human when every other guard passes."),
    "auto_merge_exact": ("merging", "Auto-fold provably identical text even with auto_apply_merges off."),
    "auto_apply_confidence": ("merging", "Minimum model confidence before anything is applied automatically."),
    "max_merged_chars": ("merging", "Character cap on a merged body; sentences past it are reported as dropped."),
    "merged_prefix": ("merging", "Prefix on a merged body, or empty string for none."),
    "stale_after_days": ("staleness", "Days without a content change before a memory is a retirement candidate."),
    "unused_for_days": ("staleness", "Days without being read or touched before it counts as unused."),
    "low_access_count": ("staleness", "access_count at or below this is treated as hardly used at all."),
    "min_importance": ("staleness", "Importance at or below this makes retirement likelier; 0 disables."),
    "ttl_grace_days": ("staleness", "Extra days after expires_at before the TTL is acted on."),
    "stale_after_days_by_kind": ("staleness", "Per-kind overrides of stale_after_days, e.g. "
                                 "{\"experiment\": 14}."),
    "protect_recent_days": ("staleness", "Days a freshly written memory is off limits to any pass."),
    "allow_auto_prune": ("retention", "Allow a pass to retire memories on staleness grounds at all."),
    "archive_instead_of_prune": ("retention", "Write 'archived' rather than 'pruned' - both are recoverable."),
    "require_provenance_for_prune": ("retention", "Never retire a record whose sources are unknown."),
    "prune_requires_no_conflict": ("retention", "A memory inside a contradictory cluster is reviewed, not retired."),
    "prune_alone_in_topic": ("retention", "Allow retiring the last active memory in a topic (default: never)."),
    "hard_delete_after_days": ("retention", "Age a record must reach before purge will consider it, "
                               "only with allow_hard_delete."),
    "allow_hard_delete": ("retention", "Permit DELETE at all. Off by default; nothing is ever "
                          "unrecoverable by accident."),
    "min_active_memories": ("floors", "Never leave fewer than this many active memories (0 disables the floor)."),
    "max_prune_fraction": ("floors", "Share of the store one pass may retire, merges included."),
    "max_merge_fraction": ("floors", "Share of the store whose rows one pass may fold."),
    "protect_kinds": ("floors", "Kinds that are never pruned or merged away (safety and policy by default)."),
    "protect_topics": ("floors", "Topics that are never pruned or merged away."),
    "review_unchanged": ("floors", "Skip records judged by an earlier run whose content has not changed."),
    "max_memories_per_run": ("floors", "Upper bound on rows considered by one pass."),
    "schedule_enabled": ("schedule", "Run the pass in the background on a schedule."),
    "schedule_timezone": ("schedule", "IANA zone the scheduled hour is expressed in."),
    "schedule_hour": ("schedule", "Local hour the scheduled pass fires at."),
    "schedule_minute": ("schedule", "Local minute the scheduled pass fires at."),
    "schedule_interval_minutes": ("schedule", "If positive, run this often instead of at a daily hour."),
    "schedule_grace_minutes": ("schedule", "Window after the scheduled slot in which a missed fire still runs."),
}


def field_schema():
    """Type, default, group and help for every policy knob (for settings UIs)."""
    out = {}
    for name, info in _field_infos().items():
        # A default_factory field carries dataclasses.MISSING, which is neither
        # None nor JSON-serialisable: call the factory instead, so the schema is
        # something a settings UI (or json.dumps) can actually consume.
        if info.default is dataclasses.MISSING:
            default = info.default_factory() if info.default_factory is not None else None
        else:
            default = info.default
        if isinstance(default, (list, set)):
            default = tuple(default)
        elif isinstance(default, dict):
            default = dict(default)
        group, help_text = FIELD_HELP.get(name, ("advanced", ""))
        out[name] = {"type": type(default).__name__ if default is not None else "any",
                     "default": default, "group": group, "help": help_text}
    return out


def _field_infos():
    return {f.name: f for f in dataclasses.fields(CleanupPolicy)}
