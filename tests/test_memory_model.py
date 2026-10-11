import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conflux.memory.cleanup import CleanupEngine
from conflux.memory.model import (CommandModel, HeuristicModel, LocalChatModel, MergeRequest,
                                  ModelError, NullModel, OllamaModel, RemoteModelRefused,
                                  SummaryRequest, assert_local_endpoint, build_model, parse_json_object,
                                  register_model, registered_models)
from conflux.memory.policy import CleanupPolicy
from conflux.memory.store import MemoryStore

REQUEST = MergeRequest(topic="limits", similarity=0.82, conflicts=False, max_chars=900,
                      reason="near", members=({"id": "m1", "body": "Session timeout is 30 minutes.",
                                              "kind": "fact", "importance": 1.0,
                                              "created_at": "2026-01-01T00:00:00+00:00",
                                              "updated_at": "2026-01-02T00:00:00+00:00"},
                                             {"id": "m2", "body": "Session timeout: 30 min.",
                                              "kind": "fact", "importance": 1.0,
                                              "created_at": "2026-01-03T00:00:00+00:00",
                                              "updated_at": "2026-01-03T00:00:00+00:00"}))


# --- a stub server so the HTTP models are covered without a network ----------


class StubServer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, payload, code=200):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                outer.seen.append(("GET", self.path, None, dict(self.headers)))
                self._send({"models": [{"name": "stub"}]})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                outer.seen.append(("POST", self.path, json.loads(raw or "null"), dict(self.headers)))
                reply = outer.replies.pop(0) if outer.replies else {"error": "exhausted"}
                if isinstance(reply, tuple):
                    self._send(reply[1], reply[0])
                elif isinstance(reply, Exception):
                    self.close_connection = True
                    raise reply
                else:
                    self._send(reply)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    server = StubServer([])
    yield server
    server.close()


# --- the no-model defaults ---------------------------------------------------


def test_null_model_is_the_default_and_is_never_unavailable_by_accident():
    policy = CleanupPolicy()
    assert policy.model == "null"
    model = build_model(policy.model)
    assert isinstance(model, NullModel)
    assert model.available() is False
    assert model.merge(REQUEST) is None
    assert model.summarize(SummaryRequest(body="anything", max_chars=200)) is None


def test_heuristic_model_unions_sentences_without_inventing():
    suggestion = HeuristicModel().merge(REQUEST)
    assert suggestion is not None
    assert "30 minutes" in suggestion.body and "30 min" in suggestion.body
    assert suggestion.body.count("Session timeout") == 2  # dedupe is per sentence
    assert suggestion.dropped_facts == ()
    assert suggestion.contradiction is False
    assert suggestion.confidence == 0.7  # the extractive cap, under auto_apply_confidence
    # deterministic: the same request folds the same way twice
    again = HeuristicModel().merge(REQUEST)
    assert again.body == suggestion.body
    folded = HeuristicModel().summarize(SummaryRequest(body="First sentence. Second one.",
                                                        max_chars=200))
    assert folded == "First sentence. Second one."


def test_heuristic_model_reports_what_the_budget_cuts():
    long_request = MergeRequest(topic="t", similarity=0.9, conflicts=False, max_chars=40, reason="near",
                                members=tuple({"id": f"m{i}", "body": f"fact number {i} about the queue",
                                              "kind": "fact", "importance": 1.0, "created_at": None,
                                              "updated_at": None} for i in range(6)))
    suggestion = HeuristicModel().merge(long_request)
    assert len(suggestion.body) <= 40
    assert suggestion.dropped_facts  # the cut is recorded, not hidden
    # ...and a lossy fold is held below the auto-apply bar on purpose
    assert suggestion.confidence < CleanupPolicy().auto_apply_confidence


def test_engine_uses_the_heuristic_without_any_model_and_needs_none(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.add("Session timeout is 30 minutes.", topic="limits", on_duplicate="insert")
    store.add("session timeout is 30 minutes", topic="limits", on_duplicate="insert")
    report = CleanupEngine(store, policy=CleanupPolicy(model="null", min_active_memories=0,
                                                        max_merge_fraction=1.0,
                                                        max_prune_fraction=1.0)).run(dry_run=False)
    assert len(report.outcome.merges_applied) == 1  # exact fold, lossless
    assert "heuristic merge used" in " ".join(report.plan.notes)
    store.close()


# --- transport safety --------------------------------------------------------


def test_non_local_endpoints_are_refused_unless_explicitly_allowed():
    assert assert_local_endpoint("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert assert_local_endpoint("http://localhost:8080/v1")
    with pytest.raises(RemoteModelRefused, match="non-local endpoint"):
        assert_local_endpoint("https://api.inference.example/v1")
    assert assert_local_endpoint("https://api.inference.example/v1", allow_remote=True)
    with pytest.raises(RemoteModelRefused):
        OllamaModel(endpoint="http://example.com")
    with pytest.raises(RemoteModelRefused):
        LocalChatModel(base_url="http://attacker.example:9/v1")


def test_no_api_key_is_invented_and_headers_are_opt_in(stub):
    stub.replies.append({"message": {"content": json.dumps({"merged_body": "ok", "confidence": 0.6})}})
    model = OllamaModel(model="stub", endpoint=stub.url, timeout=5.0)
    assert model.merge(REQUEST) is not None
    headers = stub.seen[-1][3]
    assert "Authorization" not in headers
    assert "Proxy-Authorization" not in headers
    keyed = LocalChatModel(base_url=stub.url + "/v1", api_key="local-test-key", timeout=5.0)
    assert keyed.headers["Authorization"] == "Bearer local-test-key"
    assert LocalChatModel(base_url=stub.url + "/v1").headers == {}


def test_json_is_parsed_out_of_prose_and_code_fences():
    assert parse_json_object('```json\n{"merged_body": "x"}\n```') == {"merged_body": "x"}
    assert parse_json_object('Sure! {"merged_body": "y", "confidence": 0.4} hope that helps') == \
        {"merged_body": "y", "confidence": 0.4}
    assert parse_json_object("no json here") is None
    assert parse_json_object(None) is None
    assert parse_json_object('{"broken": ') is None


# --- HTTP models -------------------------------------------------------------


def test_ollama_round_trip_uses_the_documented_shape(stub):
    stub.replies.append({"message": {"content": json.dumps(
        {"merged_body": "Session timeout is 30 minutes.", "confidence": 0.93,
         "dropped_facts": ["Session timeout: 30 min."], "contradiction": False})}})
    model = OllamaModel(model="qwen2.5", endpoint=stub.url, timeout=5.0)
    assert model.available() is True
    suggestion = model.merge(REQUEST)
    assert suggestion.body == "Session timeout is 30 minutes."
    assert suggestion.confidence == 0.93
    assert suggestion.dropped_facts == ("Session timeout: 30 min.",)
    assert suggestion.model == "ollama:qwen2.5"
    verb, path, payload, _ = stub.seen[-1]
    assert (verb, path) == ("POST", "/api/chat")
    assert payload["format"] == "json" and payload["stream"] is False
    assert payload["model"] == "qwen2.5"
    assert REQUEST.members[0]["body"] in payload["messages"][1]["content"]
    assert "Never invent facts" in payload["messages"][0]["content"]


def test_local_chat_model_reads_the_openai_shape_and_flags_contradictions(stub):
    stub.replies.append({"choices": [{"message": {"content": json.dumps(
        {"merged_body": "Timeout is 30 min, not 45.", "confidence": 1.7, "contradiction": True})}}]})
    model = LocalChatModel(base_url=stub.url + "/v1", timeout=5.0)
    suggestion = model.merge(REQUEST)
    assert suggestion.contradiction is True
    assert suggestion.confidence == 1.0  # clamped into range
    assert stub.seen[-1][1] == "/v1/chat/completions"
    assert stub.seen[-1][2]["response_format"] == {"type": "json_object"}


def test_model_outage_is_a_fallback_not_a_failure(stub):
    for replies, label in ([(500, {"error": "overloaded"})], "http_error"), \
                          ([{"unexpected": "shape"}], "bad_shape"), \
                          ([{"message": {"content": "sorry, I cannot"}}], "no_json"):
        stub.replies.extend(replies)
        model = OllamaModel(endpoint=stub.url, timeout=5.0)
        assert model.merge(REQUEST) is None, label
        stub.seen.clear()


def test_summarize_is_optional_and_returns_none_on_garbage(stub):
    stub.replies.append({"message": {"content": '{"summary": "Timeout is 30 minutes."}'}})
    model = OllamaModel(endpoint=stub.url, timeout=5.0)
    assert model.summarize(SummaryRequest(body="whatever")) == "Timeout is 30 minutes."
    stub.replies.append({"message": {"content": "{}"}})
    assert model.summarize(SummaryRequest(body="whatever")) is None


# --- command model ------------------------------------------------------------


def test_command_model_exchanges_json_over_stdin_and_stdout(tmp_path):
    script = tmp_path / "merger.py"
    script.write_text("import json,sys\n"
                      "req = json.load(sys.stdin)\n"
                      "print(json.dumps({'merged_body': 'Merged: ' + req['members'][0]['body'], "
                      "'confidence': 0.91}))\n")
    import sys
    model = CommandModel([sys.executable, str(script)], timeout=20.0)
    assert model.available() is True
    suggestion = model.merge(REQUEST)
    assert suggestion.body == "Merged: Session timeout is 30 minutes."
    assert suggestion.confidence == 0.91
    failing = CommandModel([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=20.0)
    assert failing.merge(REQUEST) is None
    missing = CommandModel([str(tmp_path / "not-a-binary")])
    assert missing.available() is False and missing.merge(REQUEST) is None
    with pytest.raises(ValueError, match="non-empty argv"):
        CommandModel([])


def test_command_model_drives_a_real_cleanup(tmp_path):
    script = tmp_path / "merger.py"
    script.write_text("import json,sys\n"
                      "req = json.load(sys.stdin)\n"
                      "print(json.dumps({'merged_body': 'Consolidated limit: 30 minutes.', "
                      "'confidence': 0.99}))\n")
    import sys
    store = MemoryStore(tmp_path / "m.db")
    store.add("Session timeout is 30 minutes.", topic="limits", on_duplicate="insert")
    store.add("The session timeout should be 30 minutes.", topic="limits", on_duplicate="insert")
    policy = CleanupPolicy(model=f"command:{sys.executable} {script}", auto_apply_merges=True,
                          auto_apply_confidence=0.9, duplicate_similarity=0.6,
                          min_active_memories=0, max_merge_fraction=1.0, max_prune_fraction=1.0)
    report = CleanupEngine(store, policy=policy).run(dry_run=False)
    assert len(report.outcome.merges_applied) == 1
    retired = store.list(state="merged")
    assert len(retired) == 2
    assert store.counts()["active"] == 1  # the survivor
    survivor = store.get(retired[0].superseded_by)
    # the model's prose is what landed in the store, and the survivor inherits the
    # cluster's kind rather than inventing a new one
    assert survivor.body == "Consolidated limit: 30 minutes."
    assert survivor.kind == "note"
    assert sorted(survivor.provenance["origin_ids"]) == sorted(m.id for m in retired)
    store.close()


# --- registry ----------------------------------------------------------------


def test_build_model_accepts_shortcuts_dotted_paths_and_registered_names():
    assert isinstance(build_model(""), NullModel)
    assert build_model("heuristic").name == "heuristic"
    assert build_model("extractive").name == "heuristic"
    assert build_model("ollama:llama3.1", {"timeout": 3}).model == "llama3.1"
    routed = build_model("local:phi@http://127.0.0.1:9099/v1")
    assert routed.base_url == "http://127.0.0.1:9099/v1" and routed.model == "phi"
    assert build_model("conflux.memory.model:HeuristicModel").name == "heuristic"
    register_model("test-stub", lambda **kwargs: NullModel())
    assert "test-stub" in registered_models()
    assert isinstance(build_model("test-stub"), NullModel)
    with pytest.raises(ModelError, match="unknown model"):
        build_model("does-not-exist")
    with pytest.raises(ModelError, match="cannot import model"):
        build_model("conflux.memory.model:NotAClass")
    with pytest.raises(ModelError, match="cannot build model"):
        build_model("command", {"argv": []})


def test_options_are_filtered_to_what_the_constructor_takes():
    model = build_model("ollama:llama3.1", {"timeout": 7.5, "junk": "ignored"})
    assert model.timeout == 7.5 and not hasattr(model, "junk")
    assert model.name == "ollama:llama3.1"


def test_service_describes_the_model_without_requiring_one(tmp_path):
    from conflux.memory.service import CleanupService
    service = CleanupService(tmp_path / "s.db")
    described = service.describe_model()
    assert described["spec"] == "null" and described["available"] is False
    assert "fallback" in described
    service.store.close()
