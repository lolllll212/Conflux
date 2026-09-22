import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from .crdts import LWWRegister, ORSet, PNCounter, Stamp

OP_COUNTER_INC = "counter_inc"
OP_COUNTER_DEC = "counter_dec"
OP_LWW_SET = "lww_set"
OP_ORSET_ADD = "orset_add"
OP_ORSET_REMOVE = "orset_remove"

ACTION_VERSION = 1

SCALE_BITS = 32
SCALE = 1 << SCALE_BITS
MASK = (1 << SCALE_BITS) - 1


def digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8", "surrogatepass"))
    return h.digest()


def decide(seed_parts, weight):
    if weight is None:
        return True
    weight = float(weight)
    threshold = int(weight * SCALE)
    threshold = max(0, min(SCALE, threshold))
    if threshold <= 0:
        return False
    if threshold >= SCALE:
        return True
    top_bits = int.from_bytes(digest(*seed_parts), "big") >> (256 - SCALE_BITS)
    return (top_bits & MASK) < threshold


@dataclass(frozen=True)
class Action:
    action_id: str
    agent: str
    tick: int
    op: str
    key: str
    params: Mapping[str, Any] = field(default_factory=dict)
    version: int = ACTION_VERSION

    def seed_parts(self):
        return (self.action_id, self.key)

    def fired(self):
        return decide(self.seed_parts(), self.params.get("weight"))

    def contribution(self):
        if not self.fired():
            return None
        if self.op in (OP_COUNTER_INC, OP_COUNTER_DEC):
            by = int(self.params.get("by", 1))
            slot = f"{self.agent}:{self.action_id}"
            if self.op == OP_COUNTER_INC:
                return PNCounter({slot: (by, 0)})
            return PNCounter({slot: (0, by)})
        if self.op == OP_LWW_SET:
            return LWWRegister(self.params["value"], Stamp(self.tick, self.agent))
        if self.op == OP_ORSET_ADD:
            return ORSet({self.params["tag"]: self.params["value"]}, frozenset())
        if self.op == OP_ORSET_REMOVE:
            return ORSet({}, frozenset({self.params["tag"]}))
        raise ValueError(f"unknown op {self.op!r}")

    def export(self):
        return {
            "action_id": self.action_id,
            "agent": self.agent,
            "tick": self.tick,
            "op": self.op,
            "key": self.key,
            "params": dict(self.params),
            "version": self.version,
        }

    @classmethod
    def import_action(cls, data):
        return cls(**data)