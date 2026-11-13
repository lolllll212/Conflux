from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


def _canonical_value(v):
    if v is None:
        return ["none", 0]
    if isinstance(v, bool):
        return ["bool", v]
    if isinstance(v, int):
        return ["int", v]
    if isinstance(v, float):
        return ["float", repr(v)]
    if isinstance(v, str):
        return ["str", v]
    if isinstance(v, (tuple, list)):
        return ["seq", [_canonical_value(x) for x in v]]
    if isinstance(v, dict):
        return ["map", sorted([[_canonical_value(k), _canonical_value(x)] for k, x in v.items()],
                              key=repr)]
    if isinstance(v, frozenset):
        return ["set", sorted([_canonical_value(x) for x in v], key=repr)]
    raise TypeError(f"cannot canonicalize {type(v)}")


@dataclass(frozen=True)
class Stamp:
    tick: int
    agent: str

    def __lt__(self, other):
        return (self.tick, self.agent) < (other.tick, other.agent)

    def __le__(self, other):
        return (self.tick, self.agent) <= (other.tick, other.agent)


class Lattice(ABC):
    @abstractmethod
    def merge(self, other):
        raise NotImplementedError

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def canonical(self):
        raise NotImplementedError


@dataclass(frozen=True)
class GCounter(Lattice):
    counts: Mapping[str, int] = field(default_factory=dict)

    def inc(self, agent, by=1):
        merged = dict(self.counts)
        merged[agent] = merged.get(agent, 0) + by
        return GCounter(merged)

    def merge(self, other):
        merged = dict(self.counts)
        for agent, value in other.counts.items():
            merged[agent] = max(merged.get(agent, 0), value)
        return GCounter(merged)

    def read(self):
        return sum(self.counts.values())

    def canonical(self):
        return ["gc", sorted((a, v) for a, v in self.counts.items())]

    def to_dict(self):
        return {"t": "gc", "counts": dict(self.counts)}


@dataclass(frozen=True)
class PNCounter(Lattice):
    pairs: Mapping[str, tuple] = field(default_factory=dict)

    def inc(self, agent, by=1):
        p, n = self.pairs.get(agent, (0, 0))
        return PNCounter({**self.pairs, agent: (p + by, n)})

    def dec(self, agent, by=1):
        p, n = self.pairs.get(agent, (0, 0))
        return PNCounter({**self.pairs, agent: (p, n + by)})

    def merge(self, other):
        merged = dict(self.pairs)
        for agent, (p2, n2) in other.pairs.items():
            p1, n1 = merged.get(agent, (0, 0))
            merged[agent] = (max(p1, p2), max(n1, n2))
        return PNCounter(merged)

    def read(self):
        return sum(p - n for p, n in self.pairs.values())

    def canonical(self):
        return ["pn", sorted((a, p, n) for a, (p, n) in self.pairs.items())]

    def to_dict(self):
        return {"t": "pn", "pairs": {k: [p, n] for k, (p, n) in self.pairs.items()}}


@dataclass(frozen=True)
class LWWRegister(Lattice):
    value: Any = None
    stamp: Stamp | None = None

    def set(self, new_value, stamp):
        if self.stamp is not None and self.stamp > stamp:
            return self
        if self.stamp == stamp and _canonical_value(self.value) <= _canonical_value(new_value):
            return self
        return LWWRegister(new_value, stamp)

    def merge(self, other):
        if other.stamp is None:
            return self
        if self.stamp is None:
            return other
        if self.stamp > other.stamp:
            return self
        if self.stamp < other.stamp:
            return other
        if _canonical_value(self.value) <= _canonical_value(other.value):
            return self
        return other

    def read(self):
        return self.value

    def canonical(self):
        if self.stamp is None:
            return ["lww", None]
        return ["lww", self.stamp.tick, self.stamp.agent, _canonical_value(self.value)]

    def to_dict(self):
        stamp = [self.stamp.tick, self.stamp.agent] if self.stamp is not None else None
        return {"t": "lww", "value": self.value, "stamp": stamp}


@dataclass(frozen=True)
class GSet(Lattice):
    elements: frozenset = frozenset()

    def add(self, value):
        return GSet(self.elements | {value})

    def merge(self, other):
        return GSet(self.elements | other.elements)

    def read(self):
        return self.elements

    def canonical(self):
        return ["gset", sorted((_canonical_value(v) for v in self.elements), key=repr)]

    def to_dict(self):
        return {"t": "gset", "elements": list(self.elements)}


@dataclass(frozen=True)
class ORSet(Lattice):
    elements: Mapping[str, Any] = field(default_factory=dict)
    tombstones: frozenset = frozenset()

    def add(self, tag, value):
        return ORSet({**self.elements, tag: value}, self.tombstones)

    def remove(self, tag):
        return ORSet(self.elements, self.tombstones | {tag})

    def merge(self, other):
        merged = dict(self.elements)
        for tag, value in other.elements.items():
            if tag not in merged or _canonical_value(value) <= _canonical_value(merged[tag]):
                merged[tag] = value
        return ORSet(merged, self.tombstones | other.tombstones)

    def read(self):
        return frozenset(
            value for tag, value in self.elements.items() if tag not in self.tombstones
        )

    def visible_tags(self):
        return frozenset(
            tag for tag in self.elements if tag not in self.tombstones
        )

    def compact(self):
        """Return a compacted ORSet where tombstoned tags and their associated elements are pruned."""
        active = {t: v for t, v in self.elements.items() if t not in self.tombstones}
        return ORSet(active, frozenset())

    def canonical(self):
        elems = sorted(
            ((t, _canonical_value(v)) for t, v in self.elements.items()), key=repr
        )
        return ["orset", elems, sorted(self.tombstones)]

    def to_dict(self):
        return {
            "t": "orset",
            "elements": dict(self.elements),
            "tombstones": list(self.tombstones),
        }


@dataclass(frozen=True)
class MVRegister(Lattice):
    """Multi-Value Register (Causal CRDT) for conflict-preserving concurrent writes.

    Each write records a ``(value, stamp, supersedes)`` causal entry. When two
    agents write concurrently without having observed each other's stamp, BOTH
    concurrent candidate values survive the lattice join in ``conflicts()`` /
    ``candidates()``, while ``read()`` returns the deterministic canonical winner.
    Once an agent issues a write whose ``supersedes`` contains those prior stamps,
    the superseded entries are pruned cleanly across every replica.
    """

    entries: Mapping[str, Any] = field(default_factory=dict)  # "tick:agent" -> value
    superseded: frozenset = frozenset()  # frozenset of "tick:agent" dot strings

    @staticmethod
    def dot_of(stamp: Stamp) -> str:
        return f"{stamp.tick}:{stamp.agent}"

    @staticmethod
    def parse_dot(dot: str) -> Stamp:
        tick_s, _, agent = dot.partition(":")
        return Stamp(int(tick_s), agent)

    def write(self, value: Any, stamp: Stamp, supersedes: frozenset | set | list | tuple = ()):
        dot = self.dot_of(stamp)
        sup = self.superseded | frozenset(str(d) for d in supersedes)
        merged_entries = dict(self.entries)
        if dot not in merged_entries or _canonical_value(merged_entries[dot]) < _canonical_value(value):
            merged_entries[dot] = value
        active = {d: v for d, v in merged_entries.items() if d not in sup}
        return MVRegister(active, sup)

    def merge(self, other):
        sup = self.superseded | other.superseded
        merged_entries: dict[str, Any] = {}
        for dot, val in self.entries.items():
            if dot not in sup:
                merged_entries[dot] = val
        for dot, val in other.entries.items():
            if dot in sup:
                continue
            if dot not in merged_entries or _canonical_value(merged_entries[dot]) < _canonical_value(val):
                merged_entries[dot] = val
        return MVRegister(merged_entries, sup)

    def candidates(self) -> list[dict[str, Any]]:
        """Return all active concurrent candidate entries ordered deterministically (winner first)."""
        active = [
            (self.parse_dot(dot), dot, val)
            for dot, val in self.entries.items()
            if dot not in self.superseded
        ]
        active.sort(key=lambda item: (item[0], _canonical_value(item[2])), reverse=True)
        return [
            {
                "dot": dot,
                "tick": st.tick,
                "agent": st.agent,
                "value": val,
            }
            for st, dot, val in active
        ]

    def has_conflict(self) -> bool:
        """True when 2+ distinct concurrent values exist that have not been causally resolved."""
        distinct = {repr(_canonical_value(c["value"])) for c in self.candidates()}
        return len(distinct) > 1

    def active_dots(self) -> tuple[str, ...]:
        return tuple(sorted(d for d in self.entries if d not in self.superseded))

    def read(self):
        cands = self.candidates()
        return cands[0]["value"] if cands else None

    def canonical(self):
        elems = sorted(
            ((d, _canonical_value(v)) for d, v in self.entries.items() if d not in self.superseded),
            key=repr,
        )
        return ["mv", elems, sorted(self.superseded)]

    def to_dict(self):
        return {
            "t": "mv",
            "entries": dict(self.entries),
            "superseded": sorted(self.superseded),
        }


@dataclass(frozen=True)
class ORMap(Lattice):
    fields: Mapping[str, Lattice] = field(default_factory=dict)

    def put(self, key, value):
        existing = self.fields.get(key)
        if existing is None:
            return ORMap({**self.fields, key: value})
        if type(existing) is not type(value):
            raise TypeError(
                f"key {key!r} holds {type(existing).__name__}, cannot absorb {type(value).__name__}"
            )
        return ORMap({**self.fields, key: existing.merge(value)})

    def merge(self, other):
        out = self
        for key, value in other.fields.items():
            out = out.put(key, value)
        return out

    def get(self, key):
        return self.fields.get(key)

    def keys(self):
        return frozenset(self.fields)

    def read(self, key):
        value = self.fields.get(key)
        return value.read() if value is not None else None

    def canonical(self):
        rows = [[k, type(c).__name__, c.canonical()] for k, c in self.fields.items()]
        rows.sort(key=repr)
        return ["ormap", rows]

    def to_dict(self):
        return {"t": "ormap", "fields": {k: v.to_dict() for k, v in self.fields.items()}}


def from_dict(data):
    kind = data["t"]
    if kind == "gc":
        return GCounter(data["counts"])
    if kind == "pn":
        return PNCounter({k: (v[0], v[1]) for k, v in data["pairs"].items()})
    if kind == "lww":
        stamp = Stamp(data["stamp"][0], data["stamp"][1]) if data["stamp"] is not None else None
        return LWWRegister(data["value"], stamp)
    if kind == "gset":
        return GSet(frozenset(data["elements"]))
    if kind == "orset":
        return ORSet(data["elements"], frozenset(data["tombstones"]))
    if kind == "mv":
        return MVRegister(dict(data["entries"]), frozenset(data.get("superseded", ())))
    if kind == "ormap":
        return ORMap({k: from_dict(v) for k, v in data["fields"].items()})
    raise ValueError(f"unknown crdt {kind!r}")