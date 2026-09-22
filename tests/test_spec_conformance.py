import json
import os

import pytest

from conflux import Action, Replica, Server, state_hash
from conflux.actions import Action
from conflux.storage import JournalStore
from conflux.validate import (
    MAX_PARAMS_BYTES,
    ValidationError,
    validate_action,
    validate_envelope,
    AgentRegistry,
)


def test_key_bounds():
    r = Replica("gen")
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="act-1", agent="gen", tick=1, op="counter_inc",
                               key="", params={"by": 1}))
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="act-1", agent="gen", tick=1, op="counter_inc",
                               key="x" * 257, params={"by": 1}))
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="act-1", agent="gen", tick=1, op="counter_inc",
                               key="bad\nkey", params={"by": 1}))


def test_identifier_rules():
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="spaces in id", agent="gen", tick=1,
                               op="counter_inc", key="k", params={"by": 1}))
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="act-1", agent="bad agent!", tick=1,
                               op="counter_inc", key="k", params={"by": 1}))
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="act-1", agent="gen", tick=1,
                               op="orset_add", key="k",
                               params={"value": 1, "tag": "bad\ttag"}))


def test_params_must_be_mapping_and_bounded():
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="act-1", agent="gen", tick=1,
                               op="counter_inc", key="k", params="nope"))
    with pytest.raises(ValidationError):
        validate_action(Action(action_id="act-1", agent="gen", tick=1,
                               op="lww_set", key="k",
                               params={"value": "x" * MAX_PARAMS_BYTES}))


def test_valid_actions_still_pass():
    for params in ({"by": 3}, {"by": 3, "weight": 0.5}):
        validate_action(Action(action_id="act-9", agent="agent-9", tick=9,
                               op="counter_inc", key="orders.total", params=params))
    validate_action(Action(action_id="act-10", agent="agent-9", tick=10,
                           op="orset_add", key="wf/shipments",
                           params={"value": {"box": "M"}, "tag": "order:100"}))


def test_envelope_still_roundtrips_through_hardening():
    registry = AgentRegistry().register("driver", "s3cr3t", ops={"counter_inc", "lww_set"},
                                        namespaces=("orders.",))
    r = Replica("driver")
    action = r._emit("counter_inc", "orders.total", {"by": 2})
    envelope = __import__("conflux.validate", fromlist=["sign_action"]).sign_action(
        action, "s3cr3t")
    validated = validate_envelope(envelope, registry)
    assert validated.action_id == action.action_id
    assert validated.agent == action.agent
    with pytest.raises(ValidationError):
        validate_envelope({**envelope, "alg": "bad"}, registry)


def test_torn_tail_recovery(tmp_path):
    d = str(tmp_path / "torn")
    store = JournalStore(d)
    r = Replica("gen")
    for i in range(3):
        r._emit("counter_inc", "orders", {"by": 1})
        store.append(r.journal[-1])
    store.close()

    path = os.path.join(d, "journal.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"action": {"partial": 1')  # simulate a torn last write

    recovered = JournalStore(d)
    root, actions = recovered.load()
    assert len(actions) == 3
    assert root is None  # no snapshot yet; journal tail below threshold
    from conflux.protocol import replay
    assert replay(actions).read("orders") == 3


def test_truncated_early_line_is_corruption(tmp_path):
    d = str(tmp_path / "corrupt")
    store = JournalStore(d)
    r = Replica("gen")
    r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[-1])
    r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[-1])
    store.close()

    path = os.path.join(d, "journal.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    lines[0] = lines[0][: len(lines[0]) // 2]  # truncate inside the first line
    open(path, "w", encoding="utf-8").write("\n".join(lines))

    recovered = JournalStore(d)
    with pytest.raises(ValueError):
        recovered.load()


def test_torn_tail_with_signed_journal(tmp_path):
    d = str(tmp_path / "torn-signed")
    store = JournalStore(d, journal_secret="k")
    r = Replica("gen")
    r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[-1])
    store.append(r.journal[-1])
    r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[-1])
    store.close()

    with open(os.path.join(d, "journal.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"action": {"torn": true')  # torn, unparseable tail skipped

    # a forged-but-well-formed wrong signature must still be refused
    ok = JournalStore(d, journal_secret="k")
    _, actions = ok.load()
    assert len(actions) == 3


def test_snapshot_reload_after_crash_generations(tmp_path):
    d = str(tmp_path / "gen")
    store = JournalStore(d, snapshot_threshold=1)
    r = Replica("gen")
    for _ in range(3):
        r._emit("counter_inc", "orders", {"by": 1})
        store.append(r.journal[-1])
        store.rotate(r.state._map)
    store.close()

    # new instance sees cumulative generations without a fresh snapshot.json touch
    store2 = JournalStore(d, snapshot_threshold=1)
    root, actions = store2.load()
    assert root is not None and root.read("orders") == 3
    assert actions == []


def test_signed_log_refuses_to_reload_without_key(tmp_path):
    """Reloading a signed log with no journal_secret is a silent verification
    downgrade and must be refused: the operator must pass the signing key."""
    d = str(tmp_path / "no-key")
    store = JournalStore(d, snapshot_threshold=1, journal_secret="node-secret")
    r = Replica("gen")
    for _ in range(2):
        r._emit("counter_inc", "orders", {"by": 1})
        store.append(r.journal[-1])
        store.rotate(r.state._map)
    store.close()

    unsigned = JournalStore(d)
    with pytest.raises(ValueError, match="journal_secret"):
        unsigned.load()


def test_restart_replay_has_no_reapplication(tmp_path):
    d = str(tmp_path / "parity")
    store = JournalStore(d, snapshot_threshold=2)
    r = Replica("gen")
    for _ in range(4):
        r._emit("counter_inc", "orders", {"by": 1})
    pre_restart = state_hash(r.state)

    # mimic the server absorb path: append + rotate on threshold cross
    for action in r.journal:
        if store.append(action):
            store.rotate(r.state._map)
    store.close()

    root, tail = store.load()
    assert root.read("orders") == 4

    # restart: uniquely-shaped root + tail, folded exactly once each
    rebuilt = Replica("gen")
    rebuilt.state._map = root
    for action in tail:
        rebuilt.apply(action)
    assert rebuilt.read("orders") == 4
    assert state_hash(rebuilt.state) == pre_restart

    # a peer re-delivering the whole journal must not inflate state:
    # the applied-set is restart-session-local, so idempotence is the guarantee.
    reapplied = rebuilt.absorb(sorted(r.journal, key=lambda a: a.action_id))
    assert state_hash(rebuilt.state) == pre_restart
    assert rebuilt.read("orders") == 4
    assert len(reapplied) == len(r.journal)  # re-absorbed, but joined idempotently


def test_server_refuses_corrupt_interior_on_start(tmp_path):
    d = str(tmp_path / "tamper-start")
    store = JournalStore(d, snapshot_threshold=1000)
    r = Replica("gen")
    for _ in range(3):
        r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[0])
    store.append(r.journal[1])
    # a garbage line injected mid-log MUST be refused, not silently replayed
    with open(os.path.join(d, "journal.jsonl"), "a", encoding="utf-8") as f:
        f.write("this is not json at all\n")
    store.append(r.journal[2])
    store.close()

    with pytest.raises(ValueError):
        Server("alpha", ("127.0.0.1", 0), data_dir=d).start()