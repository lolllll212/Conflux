"""The local model interface: summarize and merge memories, offline or not at all.

Design rules, in order of importance:

1. **Never required.** The default is :class:`NullModel`, and with it the whole
   pipeline still detects exact and near duplicates (pure text scoring in
   ``similar.py``) and writes deterministic merge proposals
   (:class:`HeuristicModel`). Nothing else in the package imports an LLM.
2. **Never remote.** :func:`assert_local_endpoint` refuses to build a client
   whose base URL is not loopback unless the caller passes ``allow_remote=True``.
   No API key is read, requested or sent by anything here.
3. **Never fatal.** A model that is down, slow or chatty returns ``None``; the
   engine falls back to the heuristic merge and records why. A cleanup pass must
   still finish.

To plug in a runtime, point ``policy.model`` at it — e.g. ``ollama:llama3.1``,
``local:<model>@http://127.0.0.1:8080/v1`` (llama.cpp / LM Studio / vLLM all
speak this), or ``command`` with an argv that reads a JSON request on stdin and
writes a JSON answer on stdout. Custom implementations can be registered with
:func:`register_model` or named directly as ``my_pkg.my_mod:MyModel``.
"""

import inspect
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .similar import fold_unique_sentences, normalize_text, split_sentences

JSON_START = re.compile(r"\{")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::", "unix", ""}
MAX_BODY_CHARS = 1200


class ModelError(RuntimeError):
    pass


class RemoteModelRefused(ModelError):
    """A model endpoint outside the loopback was requested (off by default)."""


@dataclass(frozen=True)
class MergeRequest:
    """One cluster of memories offered to the model, as plain JSON-ish data."""

    topic: str = None
    members: tuple = field(default_factory=tuple)
    similarity: float = 0.0
    conflicts: bool = False
    max_chars: int = 700
    reason: str = ""

    def to_dict(self):
        return {"task": "merge", "topic": self.topic, "members": [dict(m) for m in self.members],
                "similarity": round(float(self.similarity), 4), "conflicts": bool(self.conflicts),
                "max_chars": int(self.max_chars), "reason": self.reason}


@dataclass(frozen=True)
class SummaryRequest:
    memory_id: str = ""
    body: str = ""
    kind: str = "note"
    max_chars: int = 200

    def to_dict(self):
        return {"task": "summarize", "memory_id": self.memory_id, "body": self.body,
                "kind": self.kind, "max_chars": int(self.max_chars)}


@dataclass(frozen=True)
class MergeSuggestion:
    body: str
    confidence: float = 0.0
    reason: str = ""
    model: str = "none"
    dropped_facts: tuple = ()
    contradiction: bool = False

    def to_dict(self):
        return {"body": self.body, "confidence": round(float(self.confidence), 4),
                "reason": self.reason, "model": self.model,
                "dropped_facts": list(self.dropped_facts), "contradiction": self.contradiction}


def assert_local_endpoint(url, allow_remote=False):
    """Return ``url`` if it points at this machine; raise otherwise."""
    if allow_remote:
        return url
    parsed = urllib.parse.urlsplit(str(url))
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https", "unix") or host not in LOCAL_HOSTS:
        raise RemoteModelRefused(
            f"refusing to send memory text to a non-local endpoint {url!r}; "
            f"pass allow_remote=True only if you mean it")
    return url


def http_json(url, payload, *, timeout=60.0, headers=None, method="POST", allow_remote=False):
    """POST/GET JSON over stdlib urllib. No auth header is ever invented."""
    assert_local_endpoint(url, allow_remote=allow_remote)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, str(value))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise ModelError(f"{url} replied {exc.code}: {exc.read(400).decode('utf-8', 'replace')!r}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ModelError(f"cannot reach {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ModelError(f"{url} returned non-JSON: {raw[:200]!r}") from exc


def parse_json_object(text):
    """Pull the first JSON object out of a model reply (fences, prose, all)."""
    if not text:
        return None
    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", candidate).strip()
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    start = JSON_START.search(candidate)
    if not start:
        return None
    depth = 0
    for index in range(start.start(), len(candidate)):
        char = candidate[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start.start():index + 1])
                except ValueError:
                    return None
    return None


class MemoryModel(ABC):
    """The whole contract a local model has to satisfy. Two methods, both optional."""

    name = "abstract"

    def available(self):
        """Cheap readiness probe; ``False`` means "fall back, don't try"."""
        return True

    def describe(self):
        return {"name": self.name, "available": self.available()}

    @abstractmethod
    def merge(self, request):
        """Return a :class:`MergeSuggestion` or ``None`` to leave the cluster alone."""

    def summarize(self, request):
        """Optional shortening of one memory; ``None`` = no opinion."""
        return None


class NullModel(MemoryModel):
    """The default: no model at all. Duplicate detection still works."""

    name = "null"

    def available(self):
        return False

    def merge(self, request):
        return None


class HeuristicModel(MemoryModel):
    """Deterministic extractive merge — the fallback, and a fine default.

    Sentence-level union over the cluster (newest source first), capped at
    ``max_chars``. Nothing is invented, so its confidence is capped at the
    cluster's own similarity; dropped sentences are reported as ``dropped_facts``
    so the caller can see what the merge left out.
    """

    name = "heuristic"

    def available(self):
        return True

    def merge(self, request):
        bodies = [m.get("body", "") for m in request.members]
        if len(bodies) < 2:
            return None
        max_chars = max(40, int(request.max_chars))
        merged = fold_unique_sentences(bodies, max_chars=max_chars, prefix="")
        kept = {normalize_text(sentence) for sentence in split_sentences(merged)}
        dropped = []
        for member in request.members:
            for sentence in split_sentences(member.get("body", "")):
                key = normalize_text(sentence)
                if key and key not in kept and sentence not in dropped:
                    dropped.append(sentence.strip())
        confidence = min(0.7, max(0.3, float(request.similarity)))
        if dropped:
            # A fold that left sentences on the floor is not something to apply
            # without a look, so it stays under the auto-apply bar by construction.
            confidence = min(confidence, 0.6)
        if request.conflicts:
            confidence = min(confidence, 0.35)
        reason = "deterministic sentence union (no model configured)"
        return MergeSuggestion(body=merged, confidence=confidence, reason=reason, model=self.name,
                              dropped_facts=tuple(sorted(set(dropped)))[:10])

    def summarize(self, request):
        return fold_unique_sentences([request.body], max_chars=int(request.max_chars), prefix="")


class OllamaModel(MemoryModel):
    """Ollama chat model on loopback. No key, no cloud, optional by construction."""

    name = "ollama"

    def __init__(self, model="llama3.1", endpoint="http://127.0.0.1:11434", *, timeout=60.0,
                 temperature=0.0, options=None, allow_remote=False):
        self.model = model
        self.endpoint = assert_local_endpoint(endpoint.rstrip("/"), allow_remote=allow_remote)
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.options = dict(options or {})
        self.allow_remote = bool(allow_remote)
        self.name = f"ollama:{model}"

    def _chat(self, prompt, system):
        payload = {"model": self.model, "stream": False, "format": "json",
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}],
                   "options": {"temperature": self.temperature, **self.options}}
        reply = http_json(f"{self.endpoint}/api/chat", payload, timeout=self.timeout,
                          allow_remote=self.allow_remote)
        return (reply.get("message") or {}).get("content") or ""

    def available(self):
        try:
            http_json(f"{self.endpoint}/api/tags", None, timeout=min(5.0, self.timeout),
                       method="GET", allow_remote=self.allow_remote)
            return True
        except (ModelError, RemoteModelRefused):
            return False

    def merge(self, request):
        system = ("You consolidate an AI agent's memories. Merge the listed statements into one "
                  "short paragraph that keeps every distinct fact, number and negation. Never "
                  "invent facts. If two statements contradict, keep both and say so. Reply with "
                  'JSON only: {"merged_body": str, "confidence": number, '
                  '"dropped_facts": [str], "contradiction": bool}.')
        prompt = _merge_prompt(request)
        try:
            content = self._chat(prompt, system)
        except ModelError:
            return None
        data = parse_json_object(content)
        if not data or not str(data.get("merged_body", "")).strip():
            return None
        return MergeSuggestion(
            body=str(data["merged_body"]).strip(),
            confidence=_clamp(data.get("confidence"), 0.5),
            reason=f"ollama {self.model}",
            model=self.name,
            dropped_facts=tuple(str(x) for x in (data.get("dropped_facts") or [])[:10]),
            contradiction=bool(data.get("contradiction")),
        )

    def summarize(self, request):
        system = ("Compress the agent memory to one factual sentence. Keep every number, date and "
                  'negation. Reply with JSON only: {"summary": str}.')
        try:
            content = self._chat(f"Memory:\n{request.body[:MAX_BODY_CHARS]}", system)
        except ModelError:
            return None
        data = parse_json_object(content) or {}
        summary = str(data.get("summary") or "").strip()
        return summary or None


class LocalChatModel(MemoryModel):
    """OpenAI-compatible ``/chat/completions`` on loopback (llama.cpp, vLLM, LM Studio)."""

    name = "local"

    def __init__(self, model="local", base_url="http://127.0.0.1:8080/v1", *, timeout=60.0,
                 temperature=0.0, headers=None, api_key=None, allow_remote=False):
        self.model = model
        self.base_url = assert_local_endpoint(base_url.rstrip("/"), allow_remote=allow_remote)
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.allow_remote = bool(allow_remote)
        self.name = f"local:{model}"
        # An API key is never required; if the local server insists, the caller
        # supplies it explicitly and it stays in this process only.
        self.headers = dict(headers or {})
        if api_key:
            self.headers.setdefault("Authorization", f"Bearer {api_key}")

    def _chat(self, prompt, system):
        payload = {"model": self.model, "stream": False, "temperature": self.temperature,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}]}
        reply = http_json(f"{self.base_url}/chat/completions", payload, timeout=self.timeout,
                          headers=self.headers, allow_remote=self.allow_remote)
        try:
            return reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"unexpected reply shape from {self.base_url}: {exc}") from exc

    def available(self):
        try:
            http_json(f"{self.base_url}/models", None, timeout=min(5.0, self.timeout), method="GET",
                      headers=self.headers, allow_remote=self.allow_remote)
            return True
        except (ModelError, RemoteModelRefused):
            return False

    def merge(self, request):
        system = ("Consolidate an AI agent's memories into one paragraph keeping every distinct "
                  "fact, number and negation; never invent. If statements contradict, keep both and "
                  'flag it. Reply with JSON only: {"merged_body": str, "confidence": number, '
                  '"dropped_facts": [str], "contradiction": bool}.')
        try:
            content = self._chat(_merge_prompt(request), system)
        except ModelError:
            return None
        data = parse_json_object(content)
        if not data or not str(data.get("merged_body", "")).strip():
            return None
        return MergeSuggestion(body=str(data["merged_body"]).strip(),
                             confidence=_clamp(data.get("confidence"), 0.5),
                             reason=f"{self.base_url} {self.model}", model=self.name,
                             dropped_facts=tuple(str(x) for x in (data.get("dropped_facts") or [])[:10]),
                             contradiction=bool(data.get("contradiction")))

    def summarize(self, request):
        system = ('Compress the memory to one sentence, keeping every number and negation. '
                  'Reply with JSON only: {"summary": str}.')
        try:
            content = self._chat(request.body[:MAX_BODY_CHARS], system)
        except ModelError:
            return None
        data = parse_json_object(content) or {}
        return str(data.get("summary") or "").strip() or None


class CommandModel(MemoryModel):
    """Any local binary: one JSON request in on stdin, one JSON answer out."""

    name = "command"

    def __init__(self, argv=(), *, timeout=60.0, cwd=None, env=None):
        self.argv = list(argv)
        if not self.argv:
            raise ValueError("CommandModel needs a non-empty argv")
        self.timeout = float(timeout)
        self.cwd = cwd
        self.env = env
        self.name = f"command:{self.argv[0]}"

    def available(self):
        return shutil.which(self.argv[0]) is not None or os.path.exists(self.argv[0])

    def _run(self, payload):
        try:
            proc = subprocess.run(self.argv, input=json.dumps(payload).encode("utf-8"),
                                  capture_output=True, timeout=self.timeout, cwd=self.cwd, env=self.env)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        return parse_json_object(proc.stdout.decode("utf-8", "replace"))

    def merge(self, request):
        data = self._run(request.to_dict())
        if not data or not str(data.get("merged_body", "")).strip():
            return None
        return MergeSuggestion(body=str(data["merged_body"]).strip(),
                             confidence=_clamp(data.get("confidence"), 0.5),
                             reason="external command", model=self.name,
                             dropped_facts=tuple(str(x) for x in (data.get("dropped_facts") or [])[:10]),
                             contradiction=bool(data.get("contradiction")))

    def summarize(self, request):
        data = self._run(request.to_dict()) or {}
        return str(data.get("summary") or "").strip() or None


# --- registry ---------------------------------------------------------------

_BUILTIN = {
    "null": NullModel,
    "none": NullModel,
    "heuristic": HeuristicModel,
    "extractive": HeuristicModel,
    "ollama": OllamaModel,
    "local": LocalChatModel,
    "openai-local": LocalChatModel,
    "command": CommandModel,
}

_REGISTRY = dict(_BUILTIN)


def register_model(name, factory):
    """Make ``policy.model = name`` resolve to ``factory(**options)``."""
    if not name or not callable(factory):
        raise ValueError("register_model needs a name and a callable factory")
    _REGISTRY[name] = factory
    return factory


def registered_models():
    return sorted(_REGISTRY)


def _import_dotted(path):
    module_path, _, attr = path.partition(":")
    if not attr:
        module_path, _, attr = module_path.rpartition(".")
    if not module_path or not attr:
        raise ModelError(f"cannot import model {path!r} (expected module.path:Class)")
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise ModelError(f"cannot import model {path!r}: {exc}") from exc


def build_model(spec, options=None, allow_remote=False):
    """Turn ``policy.model`` into a model instance; never raises on a dead model.

    Accepts ``null``, ``heuristic``, ``ollama``, ``ollama:llama3.1``,
    ``local:<model>@<base_url>``, ``command``, any registered name, or a dotted
    ``module.path:Class``. ``options`` overrides constructor kwargs.
    """
    options = dict(options or {})
    options.setdefault("allow_remote", allow_remote)
    spec = (spec or "null").strip()
    name, _, tail = spec.partition(":")
    if name in ("", "null", "none"):
        return NullModel()
    if tail:
        if name in ("ollama", "local", "openai-local"):
            model, _, url = tail.partition("@")
            options.setdefault("model", model)
            if url:
                key = "endpoint" if name == "ollama" else "base_url"
                options.setdefault(key, url)
        elif name == "command":
            options.setdefault("argv", shlex_split(tail))
        else:
            options.setdefault("model", tail)
    factory = _REGISTRY.get(name)
    if factory is None and "." in spec:
        factory = _import_dotted(spec)
    if factory is None:
        raise ModelError(f"unknown model {spec!r}; known: {', '.join(registered_models())}")
    accepted = _accepted_kwargs(factory)
    kwargs = {k: v for k, v in options.items() if accepted is None or k in accepted}
    try:
        return factory(**kwargs)
    except RemoteModelRefused:
        raise
    except (TypeError, ValueError) as exc:
        # one exception type for "this model cannot be built", so callers that
        # treat a bad configuration as a fallbackable condition can catch one thing
        raise ModelError(f"cannot build model {spec!r}: {exc}") from exc


def _accepted_kwargs(factory):
    """Parameter names a factory accepts, or None if it takes **kwargs."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return None
    names = set()
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.name != "self":
            names.add(parameter.name)
    return names


def shlex_split(text):
    import shlex
    return shlex.split(text) if isinstance(text, str) else list(text)


def _clamp(value, default):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


def _merge_prompt(request):
    lines = [f"Topic: {request.topic or '(none)'}",
             f"These {len(request.members)} memories were grouped as "
             f"{'duplicates' if request.similarity >= 0.9 else 'related'} "
             f"(similarity {request.similarity:.2f}).", ""]
    for index, member in enumerate(request.members, start=1):
        stamp = member.get("updated_at") or member.get("created_at") or "?"
        lines.append(f"[{index}] id={member.get('id')} kind={member.get('kind', 'note')} "
                     f"importance={member.get('importance', 1.0)} at={stamp}")
        lines.append(f"    {str(member.get('body', ''))[:MAX_BODY_CHARS]}")
    lines += ["", f"Reply in at most {request.max_chars} characters total."]
    if request.conflicts:
        lines.append("Note: these statements contradict; keep both claims and flag contradiction=true.")
    return "\n".join(lines)
