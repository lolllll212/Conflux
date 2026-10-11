"""Fault drills: what happens when the box, the disk or a client misbehaves.

These are the scenarios a unit test of "does add() return the right dict" never
touches: a process killed mid-transaction, a volume that turns read-only under a
running service, a client that sends a sliced emoji, a peer that stops replying.
Each test asserts the *failure* is classified, bounded and recoverable - and that
nothing already committed is lost.
"""

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from conflux.memory.api import create_app
from conflux.memory.service import CleanupService as Service
from conflux.memory.store import (MemoryStore, StoreBusy, StoreCorrupt, StoreError,
                                  StoreUnavailable, classify_sqlite, sanitize_text)

fast = {"durability": "balanced"}


# --- hostile input -----------------------------------------------------------


def test_a_sliced_emoji_is_substituted_not_fatal(tmp_path):
    """A lone surrogate is a valid Python string and an invalid SQLite value.

    Clients produce them constantly by slicing a string in the middle of an
    emoji; the driver used to raise UnicodeEncodeError while binding the
    parameter, which in an HTTP process is a 500 with no explanation.
    """
    store = MemoryStore(tmp_path / "m.db", **fast)
    receipt = store.add("deploy the \ud800 api first", kind="fact", topic="deploy")
    record = store.get(receipt["id"])
    assert "\ufffd" in record.body and "deploy the" in record.body
    # the replacement is stable, so the fingerprint still collides on a re-statement
    again = store.add("deploy the \ud800 api first", kind="fact", topic="deploy")
    assert again["id"] == receipt["id"] and again["created"] is False
    store.close()


def test_control_characters_are_refused_because_they_silently_truncate(tmp_path):
    """An embedded NUL survives SQLite and then vanishes from every tool after it."""
    store = MemoryStore(tmp_path / "m.db", **fast)
    for bad in ("a\x00b", "a\x07b", "bell\x08"):
        with pytest.raises(ValueError, match="control characters"):
            store.add(bad, kind="note")
    assert store.counts()["total"] == 0
    # tabs and newlines are legitimate content for a runbook
    ok = store.add("steps:\n\t1. restart\n", kind="runbook")
    assert "\n" in store.get(ok["id"]).body
    store.close()


def test_kind_topic_and_agent_are_sanitised_too(tmp_path):
    store = MemoryStore(tmp_path / "m.db", **fast)
    receipt = store.add("body", kind="\ud800", topic="as\udbff-is", source_agent="\udc00x")
    record = store.get(receipt["id"])
    assert record.kind and record.topic and record.provenance["agent"]
    assert "\ud800" not in json.dumps(record.to_dict(), ensure_ascii=False)
    store.close()


def test_oversize_metadata_is_refused_with_an_alternative(tmp_path):
    """"Keep the payload elsewhere" is the whole answer; a 2 MB blob per write is not."""
    store = MemoryStore(tmp_path / "m.db", **fast)
    added = store.add("short fact", kind="note")
    with pytest.raises(ValueError, match="too large"):
        store.add("another", kind="note", metadata={"blob": "y" * 200_000})
    with pytest.raises(ValueError, match="provenance"):
        store.add("third", kind="note", provenance={"raw": "z" * 200_000})
    assert store.counts()["active"] == 1
    assert not store.get(added["id"]).metadata
    store.close()


def test_sanitize_text_keeps_everything_printable():
    assert sanitize_text("plain") == "plain"
    assert sanitize_text("  spaced  out ") == "spaced  out"
    assert sanitize_text("one\ntwo\tthree") == "one\ntwo\tthree"
    assert sanitize_text(b"\xff bytes")  # undecodable bytes become U+FFFD, not a crash
    with pytest.raises(ValueError, match="too long"):
        sanitize_text("x" * 50, field_name="merged body", limit=10)


# --- the volume going away ---------------------------------------------------


def _lock_down(directory, *, read_only=True):
    mode = 0o555 if read_only else 0o755
    os.chmod(directory, mode)
    for entry in os.scandir(directory):
        if entry.is_file():
            os.chmod(entry.path, 0o444 if read_only else 0o644)


def test_a_read_only_volume_degrades_instead_of_crashing(tmp_path):
    """Disk full or remounted ro: reads keep working, writes are refused loudly.

    The alternative - refusing to open - turns a broken volume into a service that
    answers nothing, including the health check that would have explained it.
    """
    store = MemoryStore(tmp_path / "m.db", **fast)
    store.add("written before the volume froze", kind="fact", topic="safety")
    store.close()
    _lock_down(tmp_path)
    try:
        reopened = MemoryStore(tmp_path / "m.db", **fast)
        assert reopened.read_only is True
        assert "read-only" in reopened.degraded or "writable" in reopened.degraded
        rows = reopened.list()
        assert [r.body for r in rows] == ["written before the volume froze"]
        with pytest.raises(StoreUnavailable, match="cannot"):
            reopened.add("this must not be accepted", kind="note")
        assert reopened.counts()["active"] == 1  # nothing half-written
        info = reopened.info()
        assert info["read_only"] is True
        reopened.close()
    finally:
        _lock_down(tmp_path, read_only=False)


def test_a_read_only_volume_still_serves_reads_and_answers_503(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    service = Service(tmp_path / "m.db", **fast)
    service.store.add("fact before the freeze", kind="fact")
    service.close()
    _lock_down(tmp_path)
    try:
        service = Service(tmp_path / "m.db", **fast)
        client = TestClient(create_app(service, scheduler=None))
        assert client.get("/memories").json()["count"] == 1
        health = client.get("/health").json()
        assert health["status"] == "degraded" and health["writable"] is False
        assert health["read_only"] is True and health["degraded_reason"]
        refused = client.post("/memories", json={"body": "after the freeze", "kind": "note"})
        assert refused.status_code == 503, refused.text
        assert refused.headers["retry-after"] == "5"
        assert refused.headers["x-conflux-store"] == "unavailable"
        assert "read-only" in refused.json()["detail"] or "writable" in refused.json()["detail"]
        assert client.get("/memories").json()["count"] == 1
        service.close()
    finally:
        _lock_down(tmp_path, read_only=False)


def test_a_missing_parent_directory_is_not_reported_as_broken_hardware(tmp_path):
    """"The path is wrong" must not look like "the disk is full"."""
    missing = tmp_path / "nope" / "deeper" / "m.db"
    with pytest.raises(Exception) as caught:
        MemoryStore(missing)
    assert not isinstance(caught.value, StoreCorrupt)
    assert "nope" in str(caught.value)  # the message names the path that failed


@pytest.mark.parametrize("message,expected", [
    ("database is locked", StoreBusy),
    ("database or disk is full", StoreUnavailable),
    ("attempt to write a readonly database", StoreUnavailable),
    ("disk I/O error", StoreUnavailable),
    ("unable to open database file", StoreUnavailable),
    ("database disk image is malformed", StoreCorrupt),
    ("file is not a database", StoreCorrupt),
])
def test_sqlite_failures_are_classified_by_what_the_caller_should_do(message, expected):
    """A retryable environment fault and a bad query are different answers."""
    assert isinstance(classify_sqlite(sqlite3.OperationalError(message)), expected)


def test_a_programming_error_stays_a_programming_error():
    """Nothing here should swallow a bug in our own SQL."""
    exc = sqlite3.OperationalError('near "SELEC": syntax error')
    assert classify_sqlite(exc) is exc


def test_the_api_maps_a_full_disk_to_503_not_400(tmp_path, monkeypatch):
    """The client did nothing wrong; the box is out of space.

    Retrying is correct for one and pointless for the other, which is the only
    reason this distinction earns its keep in the HTTP layer.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    service = Service(tmp_path / "m.db", **fast)
    client = TestClient(create_app(service, scheduler=None))

    real_raw = MemoryStore._raw

    def fake_raw(self, sql, params=()):
        if sql.strip().upper().startswith("INSERT"):
            raise StoreUnavailable("database or disk is full - the store cannot be written")
        return real_raw(self, sql, params)

    monkeypatch.setattr(MemoryStore, "_raw", fake_raw)
    response = client.post("/memories", json={"body": "won't fit", "kind": "note"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    # reads are unaffected: the process is healthy, the volume is not
    assert client.get("/health").status_code == 200
    service.close()


def test_a_corrupt_file_says_restore_rather_than_traceback(tmp_path):
    path = tmp_path / "m.db"
    store = MemoryStore(path, **fast)
    store.add("doomed", kind="note")
    store.close()
    data = bytearray(path.read_bytes())
    for offset in range(64, min(len(data), 4096)):
        data[offset] = 0  # smash the first page's payload, keep the header
    path.write_bytes(bytes(data))
    for candidate in (path.with_name("m.db-wal"), path.with_name("m.db-shm")):
        if candidate.exists():
            candidate.unlink()
    with pytest.raises(StoreError) as caught:
        MemoryStore(path, **fast)
    text = str(caught.value).casefold()
    assert "restore" in text or "verify" in text or "malformed" in text, text


# --- dying processes ---------------------------------------------------------

CHILD = '''
import sys, time
sys.path.insert(0, {repo!r})
from conflux.memory.store import MemoryStore
store = MemoryStore({db!r}, durability="balanced")
for i in range(400):
    store.add(f"durable row {{i}}", kind="note", topic="drill")
    with store.transaction():
        store.add(f"in-flight row {{i}}", kind="note", topic="drill")
        print(f"OPEN {{i}}", flush=True)
        time.sleep(0.5)          # the commit below has not happened yet
    print(f"COMMIT {{i}}", flush=True)
print("DONE", flush=True)
'''


def test_a_killed_writer_loses_only_what_it_had_not_committed(tmp_path):
    """SIGKILL in the middle of a transaction: the row vanishes, the file does not.

    This is the property the whole store is built on - a cleanup pass that dies
    half-way must leave either cluster committed or absent, never half-applied.
    """
    db = tmp_path / "m.db"
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   db=str(db)), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    last_open = None
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            # wait until the child is a few transactions in, then stop it while it
            # is inside one: everything before it is committed, it is not
            if line.startswith("OPEN ") and int(line.split()[1]) >= 2:
                last_open = int(line.split()[1])
                time.sleep(0.1)  # the child holds the transaction for 0.5s
                break
        assert last_open is not None, "child never opened a transaction"
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()

    store = MemoryStore(db, **fast)
    try:
        assert store.integrity_check() == ["ok"]
        bodies = {row.body for row in store.list(limit=10_000, state=None)}
        # the row written inside the killed transaction is gone, the one written
        # before it opened is not, and every earlier transaction is intact
        assert f"in-flight row {last_open}" not in bodies
        assert f"durable row {last_open}" in bodies
        for index in range(last_open):
            assert f"in-flight row {index}" in bodies
            assert f"durable row {index}" in bodies
        assert store.counts()["active"] == last_open * 2 + 1
    finally:
        store.close()


def test_a_run_left_running_by_a_dead_process_is_reaped(tmp_path):
    """A crashed pass is survivable but must not be reported as still running."""
    store = MemoryStore(tmp_path / "m.db", **fast)
    stranded = store.create_run(trigger="cli", mode="apply", actor="cli")
    store.close()

    reopened = MemoryStore(tmp_path / "m.db", **fast)
    assert reopened.interrupted_runs == (stranded,)
    row = reopened.query_one("SELECT status, error, finished_at FROM runs WHERE id=?", (stranded,))
    assert row["status"] == "aborted"
    assert "exited before it finished" in row["error"] and row["finished_at"]
    event = reopened.query_one("SELECT action, actor FROM memory_events WHERE action='run_aborted'")
    assert event == {"action": "run_aborted", "actor": "recovery"}
    reopened.close()


def test_reaping_never_interrupts_a_pass_that_is_actually_running(tmp_path):
    store = MemoryStore(tmp_path / "m.db", **fast)
    live = store.create_run(trigger="cli", mode="apply", actor="cli")
    store.try_lock(owner="other-process", run_id=live, lease_seconds=600)
    store.close()

    reopened = MemoryStore(tmp_path / "m.db", **fast)
    assert reopened.interrupted_runs == ()
    assert reopened.query_one("SELECT status FROM runs WHERE id=?", (live,))["status"] == "running"
    reopened.close()


def test_reopening_a_healthy_database_is_cheap_and_idempotent(tmp_path):
    store = MemoryStore(tmp_path / "m.db", **fast)
    store.add("nothing to reap", kind="note")
    assert store.reap_interrupted_runs() == ()
    assert store.reap_interrupted_runs() == ()
    store.close()


# --- growth and the write-ahead log -----------------------------------------


def test_checkpointing_bounds_the_wal(tmp_path):
    store = MemoryStore(tmp_path / "m.db", **fast)
    for i in range(3000):
        store.add(f"bulk row {i} about the cache", kind="note", topic="cache")
    wal = tmp_path / "m.db-wal"
    assert wal.exists() and wal.stat().st_size > 100_000
    busy, _log_pages, _framed = store.checkpoint(truncate=True)
    assert busy == 0
    assert wal.stat().st_size < 100_000
    assert store.counts()["active"] == 3000  # checkpointing must not lose rows
    store.close()


def test_durability_is_an_explicit_choice(tmp_path):
    """"durable" fsyncs every commit; "balanced" trusts the WAL. Both are honest."""
    durable = MemoryStore(tmp_path / "d.db")
    balanced = MemoryStore(tmp_path / "b.db", durability="balanced")
    bogus = MemoryStore(tmp_path / "x.db", durability="paranoid")  # unknown -> the safe one
    assert durable._execute("PRAGMA synchronous").fetchone()[0] == 2
    assert balanced._execute("PRAGMA synchronous").fetchone()[0] == 1
    assert bogus.durability == "durable"
    assert durable.durability == "durable" and balanced.durability == "balanced"
    # a bounded -wal is set on every open, whatever the durability choice
    for store in (durable, balanced, bogus):
        assert store._execute("PRAGMA journal_size_limit").fetchone()[0] == 67108864
        store.close()


def test_the_service_checkpoints_after_an_applied_pass(tmp_path, monkeypatch):
    calls = []
    service = Service(tmp_path / "m.db", policy={"allow_auto_prune": True,
                                                "min_active_memories": 0}, **fast)
    monkeypatch.setattr(service.store, "checkpoint",
                        lambda *, truncate=True: calls.append(truncate) or (0, 0, 0))
    for i in range(12):
        service.store.add(f"throwaway note number {i}", kind="note", topic="junk")
    service.run(dry_run=True)
    assert calls == []                      # a dry run writes nothing to fold
    service.run(dry_run=False)
    assert calls == [True]
    service.close()


def test_the_scheduler_stays_down_on_a_read_only_store(tmp_path):
    from conflux.memory.scheduler import CleanupScheduler

    service = Service(tmp_path / "m.db", **fast)
    service.store.add("readable but not writable", kind="fact")
    # pretend the volume went read-only between opening and arming
    service.store.read_only = True
    service.store.degraded = "the database is not writable; opened read-only"
    scheduler = CleanupScheduler(service, enabled=True, timezone_name="UTC", hour=3, minute=30)
    assert scheduler.start() is None
    assert scheduler._thread is None
    # the row it could not write is not half-written anywhere
    assert service.store.counts()["active"] == 1
    service.store.read_only = False
    service.close()


# --- concurrency ------------------------------------------------------------


def test_many_threads_writing_at_once_never_lose_or_duplicate(tmp_path):
    """The store's one connection is shared by every request thread; prove it holds."""
    store = MemoryStore(tmp_path / "m.db", **fast)
    failures = []

    def worker(index):
        try:
            for i in range(25):
                store.add(f"thread {index} row {i}", kind="note", topic="threads")
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            failures.append((index, repr(exc)))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert store.counts()["active"] == 250
    # and a restatement from another thread still folds instead of racing a twin
    receipt = store.add("thread 3 row 7", kind="note", topic="threads")
    assert receipt["created"] is False
    store.close()


def test_a_cleanup_pass_and_a_writer_do_not_deadlock(tmp_path):
    """One thread cleans while another writes: the lock must serialise, not hang."""
    service = Service(tmp_path / "m.db", policy={"allow_auto_prune": True,
                                                "min_active_memories": 0}, **fast)
    for i in range(40):
        service.store.add(f"note {i} that may be stale", kind="note", topic="cache",
                          created_at=iso_days(400))
    stop = threading.Event()
    errors = []

    def writer():
        while not stop.is_set():
            try:
                service.store.add("live write during a pass", kind="note")
                time.sleep(0.001)
            except StoreBusy:
                pass  # waiting for the write lock is the expected outcome
            except Exception as exc:
                errors.append(repr(exc))

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        report = service.run(dry_run=False)
        assert report.status == "completed"
    finally:
        stop.set()
        thread.join(timeout=30)
    assert errors == []
    assert not thread.is_alive()
    service.close()


def iso_days(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


# --- operational surface ----------------------------------------------------


def test_metrics_expose_the_series_an_operator_alerts_on(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    service = Service(tmp_path / "m.db", **fast)
    service.store.add("a counted fact", kind="fact")
    client = TestClient(create_app(service, scheduler=None))
    body = client.get("/metrics")
    assert body.status_code == 200
    assert body.headers["content-type"].startswith("text/plain")
    text = body.text
    for series in ("conflux_memory_up", "conflux_memory_writable", "conflux_memory_rows",
                   "conflux_memory_reviews_open", "conflux_memory_db_bytes",
                   "conflux_memory_wal_bytes", "conflux_memory_run_lock_held",
                   "conflux_memory_events", "conflux_memory_http_problems_total"):
        assert re.search(rf"^{series}(?P<labels>\{{[^}}]*\}})? \S+$", text, re.M), series
    assert 'conflux_memory_memories{state="active"}' in text
    assert "# TYPE conflux_memory_http_problems_total counter" in text
    before = _gauge(text, "conflux_memory_rows")
    service.store.add("a new fact", kind="fact")
    after = _gauge(client.get("/metrics").text, "conflux_memory_rows")
    assert after == before + 1
    # and a 404 moves the problem counter, which is the point of exposing it
    client.get("/memories/does-not-exist")
    assert _gauge(client.get("/metrics").text, "conflux_memory_http_problems_total") >= 1
    service.close()


def _gauge(metrics_text, name):
    match = re.search(rf"^{name}(?:\{{[^}}]*\}})? (\S+)$", metrics_text, re.M)
    assert match, name
    value = match.group(1)
    return int(float(value))


def test_an_oversized_request_is_refused_before_it_is_parsed(tmp_path):
    """The store caps a body at 100k chars, but that check runs on the parsed value."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    service = Service(tmp_path / "m.db", **fast)
    client = TestClient(create_app(service, scheduler=None, max_body_bytes=4096))
    huge = json.dumps({"body": "x" * 20_000, "kind": "note"}).encode()
    refused = client.post("/memories", content=huge,
                         headers={"content-type": "application/json",
                                  "content-length": str(len(huge))})
    assert refused.status_code == 413
    assert refused.json()["max_body_bytes"] == 4096
    assert client.get("/memories").json()["count"] == 0  # nothing was written
    # a normal write still fits, so the bound is not a functional regression
    assert client.post("/memories", json={"body": "fits fine", "kind": "note"}).status_code == 201
    service.close()


def test_a_filtered_listing_tells_the_truth_about_paging(tmp_path):
    """``?q=&limit=5`` used to filter *after* the limit: zero matches, hundreds exist."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    service = Service(tmp_path / "m.db", **fast)
    for i in range(200):
        suffix = " NEEDLE" if i % 20 == 0 else ""
        service.store.add(f"row {i} about nginx tuning{suffix}", kind="note")
    client = TestClient(create_app(service, scheduler=None))
    page = client.get("/memories", params={"q": "NEEDLE", "limit": 5}).json()
    assert page["count"] == 5 and page["total"] == 10 and page["has_more"] is True
    second = client.get("/memories", params={"q": "NEEDLE", "limit": 5, "offset": 5}).json()
    assert {m["id"] for m in page["items"]}.isdisjoint({m["id"] for m in second["items"]})
    last = client.get("/memories", params={"q": "NEEDLE", "limit": 5, "offset": 15}).json()
    assert last["count"] == 0 and last["has_more"] is False
    # wildcards in a needle are literal, not operators
    assert client.get("/memories", params={"q": "NEED%"}).json()["total"] == 0
    assert client.get("/memories", params={"q": "needle"}).json()["total"] == 10  # case-free
    service.close()


# --- CLI --------------------------------------------------------------------


def test_the_cli_reports_a_broken_volume_in_one_line(tmp_path, capsys):
    from conflux.memory.cli import main

    db = str(tmp_path / "m.db")
    assert main(["sample", "--db", db]) == 0
    capsys.readouterr()
    _lock_down(tmp_path)
    try:
        code = main(["list", "--db", db, "--json"])
        # listing is a read, and reads still work - that is the point of degrading
        assert code == 0, capsys.readouterr().err
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 20
        assert "Traceback" not in capsys.readouterr().err  # a degraded read, not a crash
    finally:
        _lock_down(tmp_path, read_only=False)


def test_checkpoint_command_runs_clean(tmp_path, capsys):
    from conflux.memory.cli import main

    db = str(tmp_path / "m.db")
    assert main(["sample", "--db", db]) == 0
    capsys.readouterr()
    assert main(["checkpoint", "--db", db, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["busy"] is False and report["truncate"] is True
    assert report["db_bytes"] > 0
    assert main(["checkpoint", "--db", db, "--no-truncate"]) == 0
    assert "checkpointed" in capsys.readouterr().out
    broken = tmp_path / "nope" / "deeper" / "absent.db"
    assert main(["checkpoint", "--db", str(broken)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "absent.db" in err and "Traceback" not in err


def test_verify_reports_the_environment(tmp_path, capsys):
    from conflux.memory.cli import main

    db = str(tmp_path / "m.db")
    assert main(["sample", "--db", db]) == 0
    capsys.readouterr()
    assert main(["verify", "--db", db]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["integrity"] == ["ok"]
    assert report["environment"] == {"read_only": False, "durability": "durable",
                                    "interrupted_runs": [], "bytes": report["environment"]["bytes"]}
    assert report["environment"]["bytes"] > 0


def test_durability_flag_is_accepted_by_every_command(tmp_path, capsys):
    from conflux.memory.cli import main

    db = str(tmp_path / "m.db")
    for command in (["sample", "--db", db],
                    ["status", "--db", db, "--durability", "balanced"],
                    ["run", "--db", db, "--durability", "balanced"],
                    ["list", "--db", db, "--durability", "balanced"],
                    ["verify", "--db", db, "--durability", "balanced"]):
        assert main(command) == 0, " ".join(command)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_info:  # argparse owns a bad flag, so exit 2
        main(["status", "--db", db, "--durability", "paranoid"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# --- bulk writes and the growth of the audit trail --------------------------


def test_a_batch_writes_in_one_transaction_and_reports_each_row(tmp_path):
    """200 facts after a session should cost one fsync, not two hundred.

    The batch is also the place where "all or nothing" is easiest to get wrong, so
    the receipt per row is the point of the API, not a nicety: folding a duplicate
    must still be visible to whoever sent it.
    """
    store = MemoryStore(tmp_path / "m.db", **fast)
    receipts = store.add_many([f"fact {i}" for i in range(200)])
    assert len(receipts) == 200 and all(r["created"] for r in receipts)
    # a re-statement inside the batch folds and says so
    again = store.add_many([{"body": "fact 3"}, {"body": "fact 200"}])
    assert again[0]["created"] is False and again[0]["reason"] == "exact_duplicate"
    assert again[1]["created"] is True
    store.close()


def test_a_bad_row_rolls_the_whole_batch_back(tmp_path):
    """Half a batch is worse than no batch: the caller cannot tell what landed."""
    store = MemoryStore(tmp_path / "m.db", **fast)
    before = store.counts()["total"]
    with pytest.raises(ValueError, match="missing a body"):
        store.add_many([{"body": "good one"}, {"topic": "oops"}])
    assert store.counts()["total"] == before
    assert store.add_many([]) == []
    store.close()


def test_batching_is_materially_cheaper_than_a_loop_of_writes(tmp_path):
    """The reason the route exists, measured rather than asserted in prose."""
    store = MemoryStore(tmp_path / "m.db")            # durable: real fsync per commit
    n = 60
    start = time.perf_counter()
    for i in range(n):
        store.add(f"single {i}")
    solo = time.perf_counter() - start
    start = time.perf_counter()
    store.add_many([f"batched {i}" for i in range(n)])
    batched = time.perf_counter() - start
    assert batched < solo
    store.close()


def test_audit_compaction_clears_observations_but_never_a_state_change(tmp_path):
    """Touch/seen/replay rows repeat what the surviving record says; the rest is permanent.

    Without this, memory_events grows roughly linearly with *reads*, which is a
    surprising way to run out of disk on a store whose writes all deduplicated.
    """
    store = MemoryStore(tmp_path / "m.db", **fast)
    ids = [r["id"] for r in store.add_many([f"fact {i}" for i in range(5)])]
    for i in range(3):
        store.add(f"fact {i}")                        # exact duplicates -> touch
    store.mark_seen(ids[4], actor="reader")
    stale = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store._execute("UPDATE memory_events SET ts=? WHERE action IN ('touch','seen')", (stale,))
    # the window is "older than", so a cutoff a month ago covers them and today does not
    older_than = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    plan = store.compact_events(older_than=older_than, dry_run=True)
    assert plan["dry_run"] is True and plan["deleted"] == 0
    assert plan["candidates"] == 4                    # 3 touch + 1 seen
    assert set(plan["droppable"]) == {"touch", "seen", "replay"}
    applied = store.compact_events(older_than=older_than, dry_run=False)
    assert applied["deleted"] == 4
    left = [row["action"] for row in store.query("SELECT DISTINCT action FROM memory_events")]
    assert "add" in left and "touch" not in left and "seen" not in left
    # the state of the world the trail describes is untouched
    assert store.counts()["total"] == 5
    assert store.get(ids[0]).last_seen_at
    # rows written today are outside the window, and a second pass finds nothing
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert store.compact_events(older_than=today, dry_run=True)["candidates"] == 0
    assert store.compact_events(older_than=older_than, dry_run=True)["candidates"] == 0
    store.close()


def test_compaction_refuses_an_empty_and_an_unbounded_window(tmp_path):
    store = MemoryStore(tmp_path / "m.db", **fast)
    store.add("something worth keeping")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with pytest.raises(ValueError, match="at least one action"):
        store.compact_events(older_than=now, drop_actions=[])
    # a cutoff in the past must not delete the rows written a moment ago
    future = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert store.compact_events(older_than=future, dry_run=True)["candidates"] == 0
    store.close()


def test_the_batch_and_compaction_routes_are_bounded_and_token_guarded(tmp_path):
    from fastapi.testclient import TestClient

    store = MemoryStore(tmp_path / "m.db", **fast)
    service = Service(store=store)
    app = create_app(service, start_scheduler=False, scheduler=None, token="sekrit")
    client = TestClient(app)
    headers = {"authorization": "Bearer sekrit"}
    with client:
        # unauthenticated first, parsed second: a stranger cannot make the box spend
        # effort on a 600-item body just to learn it was rejected
        assert client.post("/memories/batch", json={"items": [{"body": "one fact"}]}
                           ).status_code == 401
        big = [{"body": f"fact {i}", "topic": "t"} for i in range(600)]
        assert client.post("/memories/batch", json={"items": big}, headers=headers).status_code == 422
        assert client.post("/memories/batch", json={"items": [{"body": "x", "nope": 1}]},
                           headers=headers).status_code == 422
        headers = {"authorization": "Bearer sekrit"}
        out = client.post("/memories/batch", json={"items": [{"body": "one fact"},
                                                             {"body": "one fact"}]},
                          headers=headers)
        assert out.status_code == 200 and out.json() == {"written": 1, "folded": 1,
                                                         "receipts": out.json()["receipts"]}
        assert client.post("/maintenance/events", json={"older_than_days": 0},
                           headers=headers).status_code == 422
        assert client.post("/maintenance/events", json={"older_than_days": 90}
                           ).status_code == 401
        dry = client.post("/maintenance/events", json={"older_than_days": 90}, headers=headers)
        assert dry.status_code == 200 and dry.json()["dry_run"] is True


def test_closing_the_service_folds_the_wal_so_a_restart_is_cheap(tmp_path):
    """A restart that replays a 40 MiB WAL is a slow restart with a scary log line.

    The checkpoint on close is best-effort on purpose: failing to fold the WAL is
    not a reason to raise out of a shutdown handler and hide the real error.
    """
    db = tmp_path / "m.db"
    service = Service(str(db), namespace="default", **fast)
    service.store.add_many([f"fact {i}" for i in range(400)])
    service.close()
    wal = Path(str(db) + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    # and the data is all there on the next open
    again = Service(str(db), namespace="default", **fast)
    assert again.store.counts()["total"] == 400
    again.close()


def test_a_failed_shutdown_checkpoint_does_not_raise(tmp_path):
    from conflux.memory import service as service_module

    service = Service(str(tmp_path / "m.db"), namespace="default", **fast)
    service.store.add("one fact")

    def explode(**kwargs):
        raise service_module.StoreError("volume is gone")

    service.store.checkpoint = explode
    service.close()                      # no exception escapes a shutdown handler
    reopened = Service(str(tmp_path / "m.db"), namespace="default", **fast)
    assert reopened.store.counts()["total"] == 1
    reopened.close()


def test_a_token_that_is_not_ascii_is_rejected_in_cleanly(tmp_path):
    """compare_digest raises TypeError on non-ASCII strings, and a header is text.

    Every client that has ever pasted a token with a stray character got a 500 and a
    stack trace out of a security check; the answer to a wrong token is 401 whatever
    bytes it happens to contain.
    """
    from fastapi.testclient import TestClient

    store = MemoryStore(tmp_path / "m.db", **fast)
    client = TestClient(create_app(Service(store=store), start_scheduler=False,
                                   scheduler=None, token="s3cr3t"))
    # sent as bytes because an HTTP client may refuse to encode non-ASCII text into a
    # header at all, which is the client's business and not the behaviour under test
    with client:
        for header in (b"Bearer caf\xc3\xa9", b"Bearer \xff\xfe", b"Bearer ",
                        b"Bearer " + b"x" * 8192, b"Bearer s3cr3t; charset=caf\xc3\xa9"):
            answer = client.post("/memories", json={"body": "one fact"},
                                 headers={"authorization": header})
            assert answer.status_code == 401, header
            assert "WWW-Authenticate" in answer.headers
        assert client.post("/memories", json={"body": "one fact"},
                           headers={"authorization": "Bearer s3cr3t"}).status_code == 201
        assert client.get("/health").status_code == 200
        # a non-ASCII token is still answerable: everything is refused, nothing crashes
        odd = TestClient(create_app(Service(store=MemoryStore(tmp_path / "o.db", **fast)),
                                    start_scheduler=False, scheduler=None, token="caf\u00e9"))
        with odd:
                # header bytes are decoded per RFC 9110 (latin-1), so a token with
                # non-ASCII in it is unusable by construction - but the answer to it is
                # still a 401 from a guard, never a TypeError from a comparison
                assert odd.post("/memories", json={"body": "x"},
                                headers={"authorization": b"Bearer caf\xc3\xa9"}).status_code == 401
    store.close()
