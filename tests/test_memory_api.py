import time

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from conflux.memory.api import create_app  # noqa: E402
from conflux.memory.fixtures import load_samples  # noqa: E402
from conflux.memory.service import CleanupService  # noqa: E402


class FakeScheduler:
    """Stands in for the real one so no test starts a background thread."""

    enabled = False
    running = False

    def __init__(self):
        self.started = self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self, timeout=5.0):
        self.stopped += 1

    def status(self):
        return {"enabled": False, "running": False, "next_fire": None, "last_fire": None,
                "tick_seconds": 60.0, "timezone": "UTC", "spec": "00:00", "interval_minutes": None,
                "last_slot": None, "dry_run": False}


@pytest.fixture
def service(tmp_path):
    # floors off so the plan actually folds in a two-row store; /safety covers the floors
    handle = CleanupService(tmp_path / "api.db",
                           policy={"model": "heuristic", "min_active_memories": 0,
                                   "max_prune_fraction": 1.0, "max_merge_fraction": 1.0})
    yield handle
    handle.stop()
    handle.store.close()


@pytest.fixture
def client(service):
    app = create_app(service, scheduler=FakeScheduler())
    with TestClient(app) as test_client:
        yield test_client


def dup(client, body, **extra):
    """Insert a second row that the store would otherwise fold at write time."""
    payload = {"body": body, "on_duplicate": "insert"}
    payload.update(extra)
    response = client.post("/memories", json=payload)
    assert response.status_code in (200, 201), response.text
    return response.json()["id"]


# --- lifecycle ---------------------------------------------------------------


def test_health_reports_integrity_and_schema(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["integrity"] == ["ok"]
    assert health["writable"] is True
    assert health["schema_version"] == health["latest_schema_version"] == 2
    assert health["running"] is False
    assert client.get("/health").status_code == 200


def test_openapi_documents_every_route(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert len(paths) == 26, sorted(paths)
    for path in ("/memories", "/memories/batch", "/runs", "/runs/{run_id}/apply",
                 "/reviews/{review_id}", "/settings", "/safety", "/maintenance/purge",
                 "/maintenance/events", "/maintenance/checkpoint", "/metrics", "/backups",
                 "/merges/{merge_id}/revert"):
        assert path in paths, path
    # the operational surface must stay readable without the write token, or a
    # scraper cannot tell a degraded service from a stopped one
    assert "get" in paths["/metrics"] and "get" in paths["/health"]
    assert "Nothing leaves this machine" in schema["info"]["description"]
    # write paths carry the bearer security hint through the guard dependency
    assert paths["/runs"]["post"]["responses"]["422"]


def test_app_can_build_its_own_service(tmp_path):
    app = create_app(db_path=str(tmp_path / "owned.db"), policy={"stale_after_days": 9},
                     scheduler=FakeScheduler())
    assert isinstance(app, FastAPI)
    with TestClient(app) as owned:
        assert owned.get("/health").json()["status"] == "ok"
        assert owned.get("/settings").json()["policy"]["stale_after_days"] == 9
    assert (tmp_path / "owned.db").exists()
    with pytest.raises(ValueError, match="needs a service or a db_path"):
        create_app()


# --- auth --------------------------------------------------------------------


def test_writes_are_open_without_a_token_and_guarded_with_one(service):
    with TestClient(create_app(service, token="s3cret", scheduler=FakeScheduler())) as guarded:
        refused = guarded.post("/memories", json={"body": "no token"})
        assert refused.status_code == 401
        assert refused.headers["www-authenticate"] == "Bearer"
        assert guarded.put("/settings", json={"policy": {}}).status_code == 401
        assert guarded.get("/status").status_code == 200  # reads stay open
        assert guarded.post("/memories", json={"body": "with a token"},
                            headers={"Authorization": "Bearer s3cret"}).status_code == 201
        assert guarded.post("/memories", json={"body": "wrong token"},
                            headers={"Authorization": "Bearer nope"}).status_code == 401


def test_token_can_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFLUX_MEMORY_TOKEN", "from-env")
    own = CleanupService(tmp_path / "env.db")
    with TestClient(create_app(own, scheduler=FakeScheduler())) as env_client:
        assert env_client.post("/memories", json={"body": "x"}).status_code == 401
        assert env_client.post("/memories", json={"body": "x"},
                               headers={"Authorization": "Bearer from-env"}).status_code == 201
    own.store.close()


# --- memories ----------------------------------------------------------------


def test_memory_endpoints_cover_the_whole_lifecycle(client):
    created = client.post("/memories", json={"body": "The staging box is at 10.0.0.5.", "kind": "fact",
                                             "topic": "hosts", "source_agent": "agent-a",
                                             "uri": "chat://room/3", "confidence": 0.9})
    assert created.status_code == 201
    memory_id = created.json()["id"]
    assert created.json()["memory"]["revision"] == 1
    assert created.json()["memory"]["provenance"]["agent"] == "agent-a"
    assert created.json()["memory"]["created_at"]
    assert created.json()["memory"]["id"] == memory_id

    duplicate = client.post("/memories", json={"body": " THE staging box IS at 10.0.0.5. ",
                                               "kind": "fact", "topic": "hosts"})
    assert duplicate.status_code == 200  # folded, not created
    assert duplicate.json()["id"] == memory_id
    assert duplicate.json()["reason"] == "exact_duplicate"
    assert duplicate.json()["created"] is False

    fetched = client.get(f"/memories/{memory_id}").json()
    assert fetched["state"] == "active" and fetched["access_count"] == 2
    assert fetched["confidence"] == 0.9
    assert fetched["namespace"] == "default" and fetched["topic"] == "hosts"

    seen = client.post(f"/memories/{memory_id}/seen", json={"count": 3})
    assert seen.json()["updated"] is True
    assert client.get(f"/memories/{memory_id}").json()["access_count"] == 5
    assert client.post("/memories/ghost/seen", json={}).status_code == 404

    events = client.get(f"/memories/{memory_id}/events").json()
    assert events["memory_id"] == memory_id
    assert "add" in {e["action"] for e in events["events"]}

    listed = client.get("/memories", params={"topic": "hosts", "state": "active"}).json()
    assert listed["count"] == 1 and listed["state"] == "active" and listed["items"][0]["id"] == memory_id
    assert client.get("/memories", params={"kind": "fact"}).json()["count"] == 1
    assert client.get("/memories", params={"kind": "note"}).json()["count"] == 0
    assert client.get("/memories", params={"q": "staging"}).json()["count"] == 1
    assert client.get("/memories", params={"q": "nothing-matches"}).json()["count"] == 0

    assert client.get("/memories/does-not-exist").status_code == 404
    assert client.get("/memories/does-not-exist").json()["detail"]
    assert client.post("/memories", json={}).status_code == 422          # body required
    assert client.post("/memories", json={"body": "   "}).status_code == 400
    # unknown keys are refused rather than silently dropped, and the enum is
    # documented in the schema so a client can see the choices without reading source
    assert client.post("/memories", json={"body": "x", "nope": 1}).status_code == 422
    assert client.post("/memories", json={"body": "x", "on_duplicate": "sideways"}).status_code == 422
    enum = client.get("/openapi.json").json()["components"]["schemas"]["MemoryCreate"]["properties"]
    assert enum["on_duplicate"]["enum"] == ["touch", "insert", "error"]


def test_restore_endpoint_brings_a_retired_memory_back(client):
    memory_id = dup(client, "Retire me then bring me back.", topic="restore")
    client.app.state.service.store.set_state(memory_id, "archived", actor="api")
    assert client.get("/memories", params={"state": "active"}).json()["count"] == 0
    restored = client.post(f"/memories/{memory_id}/restore")
    assert restored.status_code == 200 and restored.json()["restored"] is True
    assert client.get(f"/memories/{memory_id}").json()["state"] == "active"
    assert client.post(f"/memories/{memory_id}/restore").json()["reason"] == "already active"
    assert client.post("/memories/ghost/restore").status_code == 404


def test_idempotency_key_is_honoured_over_http(client):
    first = client.post("/memories", json={"body": "Retries must not double-write.",
                                          "idempotency_key": "job-1"})
    second = client.post("/memories", json={"body": "Retries must not double-write.",
                                           "idempotency_key": "job-1"})
    assert first.status_code == 201 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert client.get("/memories").json()["count"] == 1


# --- runs --------------------------------------------------------------------


def test_listing_refuses_an_unknown_state(client):
    """``state=banana`` used to answer "no memories", which reads as an empty store.

    A typo in a filter should say so; ``all`` - and the empty string, which
    clients send when they build query strings blindly - mean every state.
    """
    bad = client.get("/memories", params={"state": "banana"})
    assert bad.status_code == 400
    assert "unknown state" in bad.json()["detail"]
    assert "archived" in bad.json()["detail"]  # the message names what is valid
    for state in ("all", "", "active"):
        ok = client.get("/memories", params={"state": state})
        assert ok.status_code == 200, ok.text
        assert "count" in ok.json()


def test_ttl_days_is_the_relative_form_of_an_expiry(client):
    """A caller can say "valid for a day" without computing a timestamp."""
    from datetime import datetime, timedelta, timezone

    created = client.post("/memories", json={"body": "cache TTL is 30s", "kind": "note",
                                             "topic": "cache", "ttl_days": 1})
    assert created.status_code == 201
    expiry = datetime.fromisoformat(created.json()["memory"]["expires_at"])
    assert abs((expiry - (datetime.now(timezone.utc) + timedelta(days=1))).total_seconds()) < 120

    # the two spellings of an expiry are mutually exclusive, and a non-positive
    # one would retire the memory as it is written
    both = client.post("/memories", json={"body": "y", "ttl_days": 1,
                                          "expires_at": "2030-01-01T00:00:00+00:00"})
    assert both.status_code == 400 and "not both" in both.json()["detail"]
    zero = client.post("/memories", json={"body": "z", "ttl_days": 0})
    assert zero.status_code == 400 and "positive" in zero.json()["detail"]


def test_dry_run_is_the_default_and_changes_nothing(client):
    dup(client, "The api rate limit is 100 rpm.", topic="limits")
    dup(client, "the api rate limit is 100 rpm", topic="limits")
    before = client.get("/memories").json()["items"]

    run = client.post("/runs", json={})
    assert run.status_code == 200
    payload = run.json()
    assert payload["dry_run"] is True and payload["status"] == "completed"
    assert payload["stats"]["merges_proposed"] == 1
    assert payload["stats"]["merges_auto"] == 1
    assert payload["applied"] is None
    assert client.get("/memories").json()["items"] == before

    detailed = client.get(f"/runs/{payload['run_id']}").json()
    assert detailed["plan_hash"] == payload["plan_hash"]
    assert sorted(detailed["plan"]["merges"][0]["source_ids"]) == sorted(
        [m["id"] for m in before])
    assert client.get(f"/runs/{payload['run_id']}?include_plan=false").json().get("plan") is None
    items = client.get(f"/runs/{payload['run_id']}/items").json()
    assert items["mode"] == "dry_run" and len(items["merges"]) == 1
    assert items["merges"][0]["merged_body"]


def test_a_dry_run_plan_can_be_applied_later(client):
    dup(client, "Redis uses allkeys-lru.", topic="cache")
    dup(client, "redis uses allkeys-lru", topic="cache")
    run_id = client.post("/runs", json={"dry_run": True}).json()["run_id"]
    applied = client.post(f"/runs/{run_id}/apply").json()
    assert len(applied["applied"]["merges_applied"]) == 1
    assert applied["run_id"] != run_id
    assert client.get("/memories").json()["count"] == 1
    # the applied run says which plan it came from
    assert client.get(f"/runs/{applied['run_id']}").json()["stats"]["source_run"] == run_id
    assert client.post("/runs/ghost/apply").status_code == 404


def test_apply_mode_reports_the_outcome_inline(client):
    dup(client, "Backups run nightly at 02:00.", topic="backups")
    dup(client, "backups run nightly at 02:00", topic="backups")
    run = client.post("/runs", json={"dry_run": False}).json()
    assert run["stats"]["applied"] is not None
    assert len(run["applied"]["merges_applied"]) == 1


def test_contention_is_409_for_waited_runs(client):
    service = client.app.state.service
    assert service.store.try_lock(owner="whoever", lease_seconds=30)["acquired"] is True
    busy = client.post("/runs", json={"dry_run": True})
    assert busy.status_code == 409
    assert "already running" in busy.json()["detail"]
    service.store.unlock(owner="whoever")
    assert client.post("/runs", json={"dry_run": True}).status_code == 200


def test_async_run_returns_immediately_and_the_ledger_tells_the_rest(client):
    service = client.app.state.service
    for index in range(3):
        service.store.add(f"unique note number {index}", topic=f"t{index}")
    started = client.post("/runs", json={"wait": False})
    assert started.status_code == 200
    body = started.json()
    assert body["accepted"] is True and body["finished"] in (True, False)
    assert "poll /runs" in body["hint"]
    for _ in range(400):
        runs = client.get("/runs").json()["items"]
        if runs and runs[0]["status"] in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert runs[0]["status"] == "completed"
    assert runs[0]["trigger"] == "manual"


def test_async_run_reports_a_busy_database_through_the_error_field(client):
    service = client.app.state.service
    assert service.store.try_lock(owner="holder", lease_seconds=30)["acquired"] is True
    started = client.post("/runs", json={"wait": False}).json()
    deadline = time.time() + 5
    while time.time() < deadline and not started["finished"] and "error" not in started:
        time.sleep(0.02)
    # the trigger was accepted; the pass inside it was refused, and that is visible
    service.store.unlock(owner="holder")
    assert started["accepted"] is True


def test_run_listing_and_status_filter(client):
    client.post("/runs", json={})
    client.post("/runs", json={"dry_run": False})
    listing = client.get("/runs", params={"limit": 5}).json()
    assert len(listing["items"]) >= 2
    assert {r["mode"] for r in listing["items"]} == {"dry_run", "apply"}
    assert {r["status"] for r in client.get("/runs", params={"status": "completed"}).json()["items"]} \
        == {"completed"}
    assert client.get("/runs", params={"status": "running"}).json()["items"] == []
    assert client.get("/runs", params={"limit": 1}).json()["items"].__len__() == 1
    assert client.get("/runs", params={"limit": 0}).status_code == 422


# --- settings ----------------------------------------------------------------


def test_settings_read_write_validate_and_reset(client):
    settings = client.get("/settings").json()
    assert settings["namespace"] == "default"
    assert settings["policy"]["duplicate_similarity"] == 0.78
    assert settings["summary"]["min_active_memories"] == 0
    assert settings["policy"]["allow_hard_delete"] is False
    assert settings["stored"] is None
    assert settings["model"] == {"spec": "heuristic", "available": True,
                                 "fallback": "heuristic merge (no model needed)"} or \
        settings["model"]["spec"] == "heuristic"

    updated = client.put("/settings", json={"policy": {"duplicate_similarity": 0.66,
                                                       "allow_auto_prune": True}})
    assert updated.status_code == 200
    assert updated.json()["updated"] is True
    assert updated.json()["policy"]["duplicate_similarity"] == 0.66
    assert client.get("/settings").json()["policy"]["allow_auto_prune"] is True
    assert client.get("/settings").json()["stored"]["updated_by"] == "api"

    # strings and lists are coerced, because form posts send strings
    coerced = client.put("/settings", json={"policy": {"stale_after_days": "30",
                                                      "allow_auto_prune": "no",
                                                      "protect_kinds": "policy, safety"}}).json()
    assert coerced["policy"]["stale_after_days"] == 30.0
    assert coerced["policy"]["allow_auto_prune"] is False
    assert coerced["policy"]["protect_kinds"] == ["policy", "safety"]

    assert client.put("/settings", json={"policy": {"nope": 1}}).status_code == 400
    assert "nope" in client.put("/settings", json={"policy": {"nope": 1}}).json()["detail"]
    out_of_range = client.put("/settings", json={"policy": {"max_prune_fraction": 5}})
    assert out_of_range.status_code == 400
    assert "max_prune_fraction" in out_of_range.json()["detail"]
    preview = client.put("/settings", json={"policy": {"max_prune_fraction": 5}, "dry_run": True})
    assert preview.status_code == 200 and preview.json()["valid"] is False
    assert "max_prune_fraction" in preview.json()["errors"][0]
    assert preview.json()["submitted"] == {"max_prune_fraction": 5}
    good = client.put("/settings", json={"policy": {"stale_after_days": 7}, "dry_run": True})
    assert good.json()["valid"] is True and good.json()["changes"] == {"stale_after_days": 7.0}
    assert client.get("/settings").json()["policy"]["stale_after_days"] == 30.0  # nothing stored by a preview
    untouched = client.get("/settings").json()["policy"]
    assert client.put("/settings", json={"policy": {}}).json()["policy"] == untouched
    assert client.put("/settings", json={"policy": {"duplicate_similarity": 0.78},
                                         "replace": True}).json()["policy"]["allow_auto_prune"] is False
    reset = client.post("/settings/reset").json()
    assert reset["reset"] is True
    assert reset["policy"]["min_active_memories"] == 8  # back to the shipped floor
    assert client.get("/settings").json()["policy"]["duplicate_similarity"] == 0.78


def test_settings_schema_is_self_describing(client):
    schema = client.get("/settings/schema").json()
    field = schema["fields"]["stale_after_days"]
    assert field["type"] == "float" and field["default"] == 180.0
    assert "days" in field["help"].lower()
    assert field["group"] == "staleness"
    assert schema["groups"] == ["duplicates", "floors", "merging", "retention", "schedule", "scope",
                                "staleness"]
    assert all(v["help"] for v in schema["fields"].values())
    assert schema["defaults"]["auto_apply_confidence"] == 0.85
    assert "loopback" in schema["model_options_hint"].lower()
    assert "docs/MEMORY.md" in schema["model_options_hint"]


def test_the_python_schema_is_serialisable_on_its_own(client):
    """The HTTP route sanitises, but a settings UI may call ``schema()`` directly.

    Fields defined with ``default_factory`` carry ``dataclasses.MISSING`` rather
    than a value, and that sentinel used to leak into the payload - making
    json.dumps() raise and reporting a type name of "_MISSING_TYPE".
    """
    import json

    from conflux.memory.policy import CleanupPolicy

    schema = CleanupPolicy().schema()
    assert len(schema) == 43
    round_tripped = json.loads(json.dumps(schema))  # no sentinel, no datetime
    for name, entry in round_tripped.items():
        assert entry["type"] not in {"NoneType", "any", "_MISSING_TYPE"}, name
        assert entry["help"], name
        assert entry["group"] in {"duplicates", "floors", "merging", "retention",
                                  "schedule", "scope", "staleness"}, name
    assert round_tripped["model_options"]["default"] == {}
    assert round_tripped["stale_after_days_by_kind"]["type"] == "dict"
    # sequence defaults travel as tuples in Python and as lists over JSON
    assert schema["protect_topics"]["default"] == ()
    assert round_tripped["protect_topics"]["default"] == []
    assert "credential" in schema["protect_kinds"]["default"]


def test_a_tuned_policy_changes_the_plan(client):
    dup(client, "Alice joined the crew in March.", topic="crew")
    dup(client, "Bob joined the crew in March.", topic="crew")
    strict = client.post("/runs", json={}).json()
    # 0.793 similarity clears the 0.78 bar, so they cluster - but a contradiction is
    # never folded by a machine, whatever the threshold
    assert strict["stats"]["clusters"] == 1 and strict["stats"]["merges_auto"] == 0
    assert strict["stats"]["merges_proposed"] == 1
    # push both bars above 0.793 and the pair is not even worth a look
    client.put("/settings", json={"policy": {"related_similarity": 0.9, "duplicate_similarity": 0.9,
                                             "min_active_memories": 0}})
    distant = client.post("/runs", json={}).json()
    assert distant["stats"]["clusters"] == 0 and distant["stats"]["merges_proposed"] == 0
    client.put("/settings", json={"policy": {"related_similarity": 0.6, "duplicate_similarity": 0.78}})
    loose = client.post("/runs", json={}).json()
    assert loose["stats"]["clusters"] == 1
    assert loose["stats"]["conflict_clusters"] == 1
    assert loose["stats"]["merges_auto"] == 0  # a contradiction is never folded by a machine
    reviews = client.get("/reviews").json()["items"]
    assert reviews[0]["kind"] == "contradiction"
    assert reviews[0]["detail"]["conflict"] is True


# --- review queue ------------------------------------------------------------


def test_review_queue_round_trip(client):
    dup(client, "The vendor contract renews 2027-04-01.", topic="contracts")
    dup(client, "the vendor contract renews 2027-04-01", topic="contracts")
    client.put("/settings", json={"policy": {"auto_merge_exact": False, "auto_apply_merges": False,
                                             "min_active_memories": 0}})
    client.post("/runs", json={})
    review = client.get("/reviews").json()["items"][0]
    assert review["state"] == "open" and review["kind"] == "merge"
    assert review["detail"]["merged_body"]
    assert len(review["memory_ids"]) == 2 and review["confidence"] > 0
    merge = client.get(f"/merges/{review['merge_id']}").json()
    assert merge["state"] == "needs_review"
    assert sorted(source["body"] for source in merge["sources"]) == sorted(
        ["The vendor contract renews 2027-04-01.", "the vendor contract renews 2027-04-01"])
    # the compact list and the detailed row agree on who is involved
    assert sorted(source["id"] for source in merge["sources"]) == sorted(
        link["memory_id"] for link in merge["links"])

    rejected = client.post(f"/reviews/{review['id']}", json={"decision": "rejected",
                                                            "note": "keep both"}).json()
    assert rejected["state"] == "rejected" and rejected["changed"] is True
    assert client.get("/reviews").json()["items"] == []
    assert [r["state"] for r in client.get("/reviews", params={"state": "resolved"}).json()["items"]] \
        == ["rejected"]
    assert [m["state"] for m in client.get("/merges", params={"state": "rejected"}).json()["items"]] \
        == ["rejected"]

    # the next pass proposes a fresh merge (the rejected one stays rejected), and
    # approving this review folds it
    client.post("/runs", json={})
    assert client.get("/merges", params={"state": "open"}).json()["items"][0]["state"] == "needs_review"
    again = client.get("/reviews").json()["items"][0]
    assert again["merge_id"] != review["merge_id"]
    approved = client.post(f"/reviews/{again['id']}", json={"decision": "approved",
                                                           "note": "fold them"}).json()
    assert approved["applied"], approved
    assert client.get("/memories").json()["count"] == 1
    assert client.get("/merges", params={"state": "applied"}).json()["items"][0]["merged_body"]

    # a second decision on a decided review is refused without redoing the work
    repeat = client.post(f"/reviews/{again['id']}", json={"decision": "approved"}).json()
    assert repeat["applied"] is False and "already" in repeat["reason"]
    assert client.post("/reviews/1", json={"decision": "maybe"}).status_code == 400
    assert client.post("/reviews/9999", json={"decision": "approved"}).status_code == 404


def test_merges_can_be_looked_up_by_memory(client):
    for body in ("The forge endpoint is forge.internal.", "the forge endpoint is forge.internal"):
        dup(client, body, topic="forge")
    client.put("/settings", json={"policy": {"auto_merge_exact": False, "auto_apply_merges": False}})
    client.post("/runs", json={})
    merge = client.get("/merges", params={"state": "open"}).json()["items"][0]
    assert len(merge["sources"]) == 2  # the list form carries sources, not links
    for memory_id in [source["id"] for source in merge["sources"]]:
        found = client.get("/merges", params={"state": "open", "memory_id": memory_id}).json()["items"]
        assert [m["id"] for m in found] == [merge["id"]]
    assert client.get("/merges", params={"memory_id": "unrelated"}).json()["items"] == []


def test_merge_listing_and_revert(client):
    dup(client, "Vault is at 10.0.0.9:8200.", topic="secrets")
    dup(client, "vault is at 10.0.0.9:8200", topic="secrets")
    report = client.post("/runs", json={"dry_run": False}).json()
    merge_id = report["applied"]["merges_applied"][0]
    assert client.get("/merges", params={"state": "applied"}).json()["items"]
    reverted = client.post(f"/merges/{merge_id}/revert", params={"note": "undo"})
    assert reverted.json()["reverted"] is True
    assert sorted(reverted.json()["restored"]) and client.get("/memories").json()["count"] == 2
    assert client.post(f"/merges/{merge_id}/revert").json()["reason"] == "already reverted"
    assert client.get("/merges/4242").status_code == 404
    assert client.get("/merges", params={"state": "bogus"}).json()["items"] == []
    assert client.get("/merges", params={"state": "all", "limit": 1}).json()["items"]


# --- ops ---------------------------------------------------------------------


def test_purge_is_refused_until_it_is_switched_on(client):
    refused = client.post("/maintenance/purge")
    assert refused.status_code == 409
    assert "allow_hard_delete" in refused.json()["detail"]
    client.put("/settings", json={"policy": {"allow_hard_delete": True}})
    purged = client.post("/maintenance/purge", params={"retain_days": 0})
    assert purged.status_code == 200
    assert purged.json()["deleted"] == 0 and "cutoff" in purged.json()
    assert client.post("/maintenance/purge", params={"retain_days": -1}).status_code == 422


def test_retired_rows_are_purgeable_once_the_policy_allows_it(client):
    memory_id = dup(client, "A note that will be archived then deleted.", topic="purge")
    client.app.state.service.store.set_state(memory_id, "pruned", actor="api")
    assert client.get("/memories", params={"state": "pruned"}).json()["count"] == 1
    client.put("/settings", json={"policy": {"allow_hard_delete": True}})
    deleted = client.post("/maintenance/purge", params={"retain_days": 0}).json()
    assert deleted["deleted"] == 1
    assert client.get("/memories", params={"state": "pruned"}).json()["count"] == 0
    # the audit trail still knows what was there
    assert client.get("/memories", params={"state": "all"}).status_code == 200
    deleted_event = client.app.state.service.store.events(action="hard_delete")[0]
    assert deleted_event["before"]["body"] == "A note that will be archived then deleted."


def test_backup_endpoint_writes_a_consistent_snapshot(client):
    service = client.app.state.service
    dup(client, "backed up text", topic="t")
    result = client.post("/backups").json()
    assert result["path"].endswith(".sqlite")
    import sqlite3
    conn = sqlite3.connect(f"file:{result['path']}?mode=ro", uri=True)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("SELECT body FROM memories").fetchone()[0] == "backed up text"
    conn.close()
    restored = CleanupService(result["path"])
    assert restored.store.query_one("SELECT body FROM memories")["body"] == "backed up text"
    restored.store.close()


def test_safety_summary_names_the_floors(client):
    safety = client.get("/safety").json()
    assert safety["min_active_memories"] == 0
    assert safety["allow_hard_delete"] is False
    assert safety["auto_apply_merges"] is False
    assert safety["allow_auto_prune"] is False
    assert safety["dry_run_by_default"] is True
    assert safety["active"] == 0
    assert safety["would_block_total_removal"] is True
    assert safety["policy"]["max_prune_fraction"] == 1.0


def test_status_gathers_everything_a_dashboard_needs(client):
    load_samples(client.app.state.service.store)
    client.post("/runs", json={})
    status = client.get("/status").json()
    assert status["healthy"] is True
    assert status["counts"]["active"] == 20
    assert status["counts"]["kinds"]["fact"] == 10
    assert status["database"]["schema_version"] == 2
    assert status["database"]["path"].endswith("api.db")
    assert status["running"] is False and status["lock"]["held"] is False
    assert status["model"]["spec"] == "heuristic"
    assert status["scheduler"]["enabled"] is False
    assert status["last_runs"][0]["stats"]
    assert set(status["policy"]) >= {"allow_auto_prune", "allow_hard_delete", "min_active_memories"}
    assert status["proposed_merges"]
    assert "server_time" in status
