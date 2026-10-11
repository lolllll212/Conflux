import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

from conflux.memory.cli import _SAMPLE_POLICY, main
from conflux.memory.store import MemoryStore


@pytest.fixture
def db(tmp_path):
    yield str(tmp_path / "cli.db")


@pytest.fixture
def seeded(db, capsys):
    assert main(["sample", "--db", db, "--embeddings", "--log-level", "warning"]) == 0
    capsys.readouterr()
    return db


def payload(text):
    start = min(index for index in (text.find("{"), text.find("[")) if index >= 0)
    return json.loads(text[start:])


def policy_defaults():
    from conflux.memory.policy import CleanupPolicy
    return CleanupPolicy().to_dict()


# --- plumbing ----------------------------------------------------------------


def test_missing_db_is_a_usage_error_not_a_traceback(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["status"])
    assert exit_info.value.code == 2
    assert "--db" in capsys.readouterr().err


def test_unknown_command_and_bad_flag_are_usage_errors(capsys):
    with pytest.raises(SystemExit) as first:
        main(["defenestrate", "--db", "x.db"])
    assert first.value.code == 2
    with pytest.raises(SystemExit):
        main(["list", "--db", "x.db", "--not-a-flag"])
    err = capsys.readouterr().err
    assert "invalid choice" in err or "unrecognized arguments" in err


def test_a_typed_failure_is_one_line_and_exit_one(db, capsys):
    # an unusable file is a real condition the operator can act on, not a traceback
    with open(db, "w") as handle:
        handle.write("this is not a database\n" * 30)
    assert main(["status", "--db", db, "--log-level", "error"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for command in ("run", "serve", "status", "sample", "add", "list", "reviews", "review", "merges",
                    "revert", "settings", "apply", "backup", "verify", "restore"):
        assert command in text


def test_log_level_works_after_the_subcommand(seeded, capsys):
    assert main(["status", "--db", seeded, "--log-level", "error"]) == 0
    capsys.readouterr()
    assert main(["status", "--db", seeded, "--log-level", "debug"]) == 0


def test_module_entry_point_is_wired():
    from conflux.memory import __main__ as entry
    assert entry.main is main


# --- sample data -------------------------------------------------------------


def test_sample_seeds_a_teachable_database(db, capsys):
    assert main(["sample", "--db", db, "--embeddings", "--log-level", "warning"]) == 0
    out = capsys.readouterr().out
    assert "20 active memories" in out and "5 kinds" in out
    assert "--explain" in out  # the next step is printed
    seeded = db
    with MemoryStore(seeded) as store:
        counts = store.counts()
        assert counts["active"] == 20 and counts["total"] == 20
        assert counts["kinds"] == {"fact": 10, "note": 4, "runbook": 3, "policy": 2,
                                   "preference": 1}
        policy = store.get_policy()
        assert policy.to_dict() == {**policy_defaults(), **_SAMPLE_POLICY}
        assert store.query_one("SELECT embedding_model FROM memories "
                               "WHERE embedding IS NOT NULL")["embedding_model"]


def test_sample_refuses_to_lose_an_existing_database(seeded, capsys):
    # seeding twice is a no-op: ids are skipped, not overwritten
    assert main(["sample", "--db", seeded, "--log-level", "warning"]) == 0
    assert "20 active memories" in capsys.readouterr().out
    with MemoryStore(seeded) as store:
        assert store.counts()["total"] == 20
        store.touch_body("mem-sample-00", body="A correction a human made.")
    assert main(["sample", "--db", seeded, "--log-level", "warning"]) == 0
    with MemoryStore(seeded) as store:
        assert store.get("mem-sample-00").body == "A correction a human made."


def test_sample_rejects_an_unknown_policy_key(seeded, capsys):
    assert main(["sample", "--db", seeded, "--policy", '{"nope": 1}', "--log-level", "error"]) == 1
    assert "nope" in capsys.readouterr().err


# --- the pass ----------------------------------------------------------------


def test_run_previews_then_applies(seeded, capsys):
    assert main(["run", "--db", seeded, "--explain", "--log-level", "warning"]) == 0
    out = capsys.readouterr().out
    assert "mode=dry-run" in out and "considered 20" in out
    assert "mem-sample-09" in out  # the Alice/Bob contradiction is explained
    assert "python -m conflux.memory apply" in out  # the next step is printed
    with MemoryStore(seeded) as store:
        assert store.counts()["active"] == 20
        assert store.list_runs()[0]["mode"] == "dry_run"
        # the sample policy never auto-applies merges, so proposals wait in the queue
        assert len(store.merges(state="open")) == 3
        assert len(store.reviews(state="open")) == 6

    assert main(["run", "--db", seeded, "--apply", "--log-level", "warning"]) == 0
    out = capsys.readouterr().out
    assert "applied: 0 merges, 2 prunes" in out
    with MemoryStore(seeded) as store:
        assert store.counts()["active"] == 18
        assert {m.id for m in store.list(state="archived")} == {"mem-sample-11", "mem-sample-18"}
        assert store.list_runs()[0]["mode"] == "apply"


def test_run_json_output_is_machine_readable(seeded, capsys):
    assert main(["run", "--db", seeded, "--json", "--log-level", "warning"]) == 0
    report = payload(capsys.readouterr().out)
    assert report["dry_run"] is True and report["status"] == "completed"
    assert report["run_id"].startswith("run-")
    assert report["applied"] is None
    assert report["stats"]["considered"] == 20 and report["stats"]["prunes_auto"] == 2
    cluster = report["plan"]["clusters"][0]
    assert {"kind", "ids", "score", "conflicts", "topic"} <= set(cluster)
    assert report["plan"]["merges"][0]["source_ids"]


def test_run_can_override_the_model_without_rewriting_the_policy(seeded, capsys):
    assert main(["run", "--db", seeded, "--model", "null", "--json", "--log-level", "warning"]) == 0
    report = payload(capsys.readouterr().out)
    assert report["stats"]["model"] == "null"
    assert any("heuristic merge used" in note for note in report["plan"]["notes"])
    with MemoryStore(seeded) as store:
        assert store.get_policy().model == "heuristic"  # the override did not persist
    assert main(["run", "--db", seeded, "--model", "ollama", "--json",
                 "--log-level", "warning"]) == 0
    unreachable = payload(capsys.readouterr().out)
    assert unreachable["stats"]["model"].startswith("ollama")
    assert unreachable["stats"]["considered"] == 20  # a dead model never costs a run


def test_run_reports_contention_as_a_retryable_exit_code(seeded, capsys):
    with MemoryStore(seeded) as store:
        assert store.try_lock(owner="another-process", lease_seconds=120)["acquired"] is True
    assert main(["run", "--db", seeded, "--apply", "--log-level", "error"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("refused:") and "already running" in err


def test_apply_replays_a_stored_plan(seeded, capsys):
    assert main(["run", "--db", seeded, "--json", "--log-level", "warning"]) == 0
    run_id = payload(capsys.readouterr().out)["run_id"]
    assert main(["apply", run_id, "--db", seeded, "--log-level", "warning"]) == 0
    out = capsys.readouterr().out
    assert f"applied {run_id} as run " in out
    assert "prunes_applied" in out
    with MemoryStore(seeded) as store:
        assert store.counts()["active"] == 18
        assert store.list_runs()[0]["stats"]["source_run"] == run_id
    assert main(["apply", "run-does-not-exist", "--db", seeded, "--log-level", "error"]) == 1
    assert "run-does-not-exist" in capsys.readouterr().err


def test_apply_of_a_dry_run_only_run_explains_itself(seeded, capsys):
    assert main(["run", "--db", seeded, "--apply", "--json", "--log-level", "warning"]) == 0
    applied = payload(capsys.readouterr().out)["run_id"]
    # a plan already consumed by an apply run is reported, not executed twice
    assert main(["apply", applied, "--db", seeded, "--log-level", "warning"]) == 0
    capsys.readouterr()
    with MemoryStore(seeded) as store:
        assert store.counts()["active"] == 18


# --- browsing ----------------------------------------------------------------


def test_list_refuses_an_unknown_state(seeded, capsys):
    assert main(["list", "--db", seeded, "--state", "banana", "--log-level", "error"]) == 1
    err = capsys.readouterr().err
    assert "unknown state" in err and "archived" in err


def test_list_renders_rows_and_json(seeded, capsys):
    assert main(["list", "--db", seeded, "--kind", "policy", "--log-level", "warning"]) == 0
    out = capsys.readouterr().out
    assert "policy" in out and "rev=1" in out and "(2 shown, state=active)" in out
    assert main(["list", "--db", seeded, "--state", "all", "--limit", "5", "--json",
                 "--log-level", "warning"]) == 0
    rows = payload(capsys.readouterr().out)
    assert isinstance(rows, list) and len(rows) == 5
    assert {"id", "body", "state", "revision", "kind", "topic"} <= set(rows[0])
    assert main(["list", "--db", seeded, "--topic", "hosts", "--json",
                 "--log-level", "warning"]) == 0
    assert len(payload(capsys.readouterr().out)) == 4


def test_status_reports_the_whole_picture(seeded, capsys):
    assert main(["status", "--db", seeded, "--json", "--log-level", "warning"]) == 0
    status = payload(capsys.readouterr().out)
    assert status["counts"]["active"] == 20
    assert status["database"]["schema_version"] == 2
    assert status["lock"]["held"] is False
    assert status["model"]["spec"] == "heuristic"
    assert status["open_reviews"] == []
    assert status["scheduler"]["enabled"] is False  # the sample policy turns it off
    assert main(["status", "--db", seeded, "--log-level", "warning"]) == 0
    out = capsys.readouterr().out
    assert "active 20" in out and "scheduler enabled=False" in out


def test_add_writes_once_and_folds_on_repetition(seeded, capsys):
    body = "The metrics retention window is 30 days."
    assert main(["add", body, "--db", seeded, "--topic", "metrics", "--kind", "fact", "--agent",
                 "cli-user", "--importance", "1.5", "--idempotency-key", "k1", "--json",
                 "--log-level", "warning"]) == 0
    first = payload(capsys.readouterr().out)
    assert first["created"] is True
    assert main(["add", body, "--db", seeded, "--topic", "metrics", "--kind", "fact",
                 "--idempotency-key", "k1", "--json", "--log-level", "warning"]) == 0
    again = payload(capsys.readouterr().out)
    assert again["id"] == first["id"] and again["created"] is False
    assert again["reason"] == "idempotency_key"
    assert main(["add", body, "--db", seeded, "--topic", "metrics", "--kind", "fact",
                 "--log-level", "warning"]) == 0
    assert "matched existing" in capsys.readouterr().out
    with MemoryStore(seeded) as store:
        assert store.counts()["active"] == 21
        record = store.get(first["id"])
        assert record.importance == 1.5 and record.provenance["agent"] == "cli-user"
        # the idempotent replay is a no-op (no bump), the un-keyed restatement touches
        access_count = record.access_count
        assert access_count == 2
    # --ttl-days is the only way to state an expiry without computing a date
    assert main(["add", "a temporary note", "--db", seeded, "--ttl-days", "1",
                 "--json", "--log-level", "warning"]) == 0
    receipt = payload(capsys.readouterr().out)  # the receipt carries no timestamps...
    assert main(["list", "--db", seeded, "--json", "--log-level", "warning"]) == 0
    rows = payload(capsys.readouterr().out)
    stored = [row for row in rows if row["body"] == "a temporary note"][0]
    assert stored["id"] == receipt["id"]
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    assert datetime.fromisoformat(stored["expires_at"]).date() == tomorrow  # ...the row does
    assert main(["add", "bad ttl", "--db", seeded, "--ttl-days", "0",
                 "--log-level", "warning"]) == 1
    assert "ttl_days must be positive" in capsys.readouterr().err

    assert main(["add", "   ", "--db", seeded, "--log-level", "error"]) == 1
    assert "must not be empty" in capsys.readouterr().err


# --- the queue ---------------------------------------------------------------


def test_reviews_merges_and_decisions(seeded, capsys):
    assert main(["run", "--db", seeded, "--log-level", "warning"]) == 0  # dry run opens the queue
    capsys.readouterr()
    assert main(["reviews", "--db", seeded, "--json", "--log-level", "warning"]) == 0
    items = payload(capsys.readouterr().out)["items"]
    assert {item["kind"] for item in items} >= {"merge", "prune", "contradiction"}
    contradiction = next(i for i in items if i["kind"] == "contradiction")
    assert contradiction["detail"]["conflict"] is True

    assert main(["merges", "--db", seeded, "--json", "--log-level", "warning"]) == 0
    merges = payload(capsys.readouterr().out)["items"]
    assert merges and {m["state"] for m in merges} <= {"proposed", "needs_review"}
    assert len(merges[0]["sources"]) >= 2

    assert main(["review", str(contradiction["id"]), "--decision", "reject", "--note", "keep both",
                 "--db", seeded, "--log-level", "warning"]) == 0
    resolved = payload(capsys.readouterr().out)
    assert resolved["state"] == "rejected" and resolved["changed"] is True
    assert main(["reviews", "--db", seeded, "--json", "--log-level", "warning"]) == 0
    assert contradiction["id"] not in {i["id"] for i in payload(capsys.readouterr().out)["items"]}
    assert main(["reviews", "--db", seeded, "--all", "--log-level", "warning"]) == 0
    assert "[rejected]" in capsys.readouterr().out

    assert main(["review", "9999", "--decision", "approve", "--db", seeded,
                 "--log-level", "error"]) == 1
    assert "9999" in capsys.readouterr().err


def test_empty_queues_say_so(seeded, capsys):
    assert main(["reviews", "--db", seeded, "--log-level", "warning"]) == 0
    assert "review queue is empty" in capsys.readouterr().out
    assert main(["merges", "--db", seeded, "--log-level", "warning"]) == 0
    assert "no merges with state=open" in capsys.readouterr().out


def test_restore_and_revert_round_trip(seeded, capsys):
    assert main(["run", "--db", seeded, "--apply", "--log-level", "warning"]) == 0
    capsys.readouterr()
    with MemoryStore(seeded) as store:
        victim = store.list(state="archived")[0].id
    assert main(["restore", victim, "--db", seeded, "--json", "--log-level", "warning"]) == 0
    assert payload(capsys.readouterr().out)["restored"] is True
    with MemoryStore(seeded) as store:
        assert store.get(victim).state == "active"
    assert main(["restore", victim, "--db", seeded, "--log-level", "warning"]) == 0
    assert "already active" in capsys.readouterr().out
    assert main(["restore", "mem-nope", "--db", seeded, "--log-level", "error"]) == 1

    # an exact duplicate pair can be folded and then pulled apart again
    a = store_pair(seeded)
    assert main(["run", "--db", seeded, "--apply", "--json", "--log-level", "warning"]) == 0
    applied = payload(capsys.readouterr().out)
    merge_ids = applied["applied"]["merges_applied"]
    assert merge_ids, applied
    assert main(["merges", "--db", seeded, "--state", "applied", "--json",
                 "--log-level", "warning"]) == 0
    listed = payload(capsys.readouterr().out)["items"]
    assert [m["id"] for m in listed] == merge_ids
    assert main(["revert", str(merge_ids[0]), "--note", "wrong merge", "--db", seeded,
                 "--log-level", "warning"]) == 0
    revert_report = payload(capsys.readouterr().out)
    assert revert_report["reverted"] is True
    with MemoryStore(seeded) as store:
        assert store.get(a).state == "active"
        assert store.get(a).superseded_by is None
    assert main(["revert", str(merge_ids[0]), "--db", seeded, "--log-level", "warning"]) == 0
    assert payload(capsys.readouterr().out)["reason"] == "already reverted"


def store_pair(db):
    with MemoryStore(db) as store:
        first = store.add("The graph endpoint is graph.internal:8182.", topic="graphs",
                         memory_id="pair-a", on_duplicate="insert")["id"]
        store.add("the graph endpoint is graph.internal:8182", topic="graphs", memory_id="pair-b",
                 on_duplicate="insert")
        return first


# --- configuration and ops ---------------------------------------------------


def test_settings_reads_previews_and_writes(seeded, capsys):
    assert main(["settings", "--db", seeded, "--log-level", "warning"]) == 0
    policy = payload(capsys.readouterr().out)
    assert policy["stale_after_days"] == _SAMPLE_POLICY.get("stale_after_days", 180.0)

    assert main(["settings", "--db", seeded, "--set", "stale_after_days=90", "--set",
                 "allow_auto_prune=false", "--dry-run", "--json", "--log-level", "warning"]) == 0
    preview = payload(capsys.readouterr().out)
    assert preview == {"dry_run": True, "valid": True,
                       "changes": {"stale_after_days": 90.0, "allow_auto_prune": False}}
    with MemoryStore(seeded) as store:
        assert store.get_policy().stale_after_days != 90.0  # a preview stores nothing

    assert main(["settings", "--db", seeded, "--set", "stale_after_days=90", "--json",
                 "--log-level", "warning"]) == 0
    written = payload(capsys.readouterr().out)
    assert written["updated"] is True and written["changes"]["stale_after_days"] == 90.0
    with MemoryStore(seeded) as store:
        assert store.get_policy().stale_after_days == 90.0

    assert main(["settings", "--db", seeded, "--set", "max_prune_fraction=9", "--log-level",
                 "error"]) == 1
    assert "max_prune_fraction" in capsys.readouterr().err
    assert main(["settings", "--db", seeded, "--set", "not_a_setting=1", "--log-level",
                 "error"]) == 1
    assert "not_a_setting" in capsys.readouterr().err
    # a malformed assignment is a usage error, and argparse's convention is to exit
    with pytest.raises(SystemExit) as usage:
        main(["settings", "--db", seeded, "--set", "no_equals_sign"])
    assert "KEY=VALUE" in str(usage.value)


def test_backup_and_verify(seeded, tmp_path, capsys):
    destination = str(tmp_path / "copy.db")
    assert main(["backup", "--db", seeded, "--dest", destination, "--log-level", "warning"]) == 0
    assert tmp_path.joinpath("copy.db").exists()
    assert capsys.readouterr().out.strip() == destination
    other = str(tmp_path / "copy2.db")
    assert main(["backup", "--db", seeded, "--dest", other, "--json",
                 "--log-level", "warning"]) == 0
    assert payload(capsys.readouterr().out)["path"] == other
    # a directory is not a database file: one clean line, no traceback
    assert main(["backup", "--db", seeded, "--dest", str(tmp_path), "--log-level", "error"]) == 1
    assert capsys.readouterr().err.startswith("error:")

    assert main(["verify", "--db", seeded, "--log-level", "warning"]) == 0
    report = payload(capsys.readouterr().out)
    assert report["integrity"] == ["ok"] and report["schema"] == 2
    assert report["counts"]["active"] == 20 and "mirror" not in report and "fabric_hash" not in report

    assert main(["verify", "--db", seeded, "--mirror", "--log-level", "warning"]) == 0
    mirrored = payload(capsys.readouterr().out)
    assert mirrored["mirror"]["ok"] is True
    assert mirrored["mirror"]["mirrored"] == 20 == mirrored["counts"]["active"]
    assert mirrored["fabric_hash"]

    # a persisted mirror that was never re-synced lags the database, and verify says so
    fabric_dir = str(tmp_path / "fabric")
    with MemoryStore(seeded) as store:
        from conflux.memory.bridge import MemoryFabric
        armed = MemoryFabric("verify", data_dir=fabric_dir)
        assert armed.sync_store(store)["mirrored"] == 20
        armed.close()
        retired = store.list()[0]
        store.set_state(retired.id, "archived", actor="test")
    assert main(["verify", "--db", seeded, "--fabric-dir", fabric_dir, "--log-level",
                 "warning"]) == 1
    captured = capsys.readouterr()  # readonce: capsys buffers are consumed on read
    report = payload(captured.out)
    assert report["fabric"]["ok"] is False and report["fabric"]["stale"] == [retired.id]
    assert "drift:" in captured.err
    assert report["integrity"] == ["ok"]  # the database itself is fine, only the mirror lags
    with MemoryStore(seeded) as store:
        store.set_state(retired.id, "active", actor="test")
    assert main(["verify", "--db", seeded, "--fabric-dir", fabric_dir, "--log-level",
                 "warning"]) == 0


def test_serve_explains_a_missing_dependency(db, capsys, monkeypatch):
    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "uvicorn":
                raise ImportError("blocked for the test")
            return None

        def find_module(self, name, path=None):  # pragma: no cover - legacy hook
            return None

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(SystemExit) as exit_info:
            main(["serve", "--db", db, "--log-level", "error"])
    finally:
        sys.meta_path.remove(blocker)
    assert "conflux[memory]" in str(exit_info.value.code)


def test_serve_help_documents_the_flags(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["serve", "--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--host", "--port", "--no-scheduler", "--dry-run-schedule"):
        assert flag in text


def test_every_command_works_against_a_fresh_database(db, capsys):
    # an empty database is a normal state, not an error
    for argv in (["status", "--db", db], ["list", "--db", db], ["reviews", "--db", db],
                 ["merges", "--db", db], ["settings", "--db", db], ["run", "--db", db],
                 ["verify", "--db", db]):
        assert main(argv + ["--log-level", "error"]) == 0, argv
    out = capsys.readouterr().out
    assert "active 0" in out
