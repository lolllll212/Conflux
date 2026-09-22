import base64
import hashlib
import hmac
import json
import re
import threading

from .actions import (
    ACTION_VERSION,
    OP_COUNTER_DEC,
    OP_COUNTER_INC,
    OP_LWW_SET,
    OP_ORSET_ADD,
    OP_ORSET_REMOVE,
    Action,
)

ALL_OPS = {OP_COUNTER_INC, OP_COUNTER_DEC, OP_LWW_SET, OP_ORSET_ADD, OP_ORSET_REMOVE}
COUNTER_OPS = {OP_COUNTER_INC, OP_COUNTER_DEC}

ALG_HMAC = "hmac-sha256"
ALG_ED25519 = "ed25519"

MAX_KEY_LENGTH = 256
MAX_AGENT_LENGTH = 128
MAX_ID_LENGTH = 512
MAX_TAG_LENGTH = 512
MAX_PARAMS_BYTES = 4096
_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]+$")


class ValidationError(Exception):
    pass


def _is_json_safe(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False


def _check_identifier(value, field, maxlen):
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be a non-empty string")
    if len(value) > maxlen:
        raise ValidationError(f"{field} too long (max {maxlen} chars)")
    if not _ID_RE.match(value):
        raise ValidationError(f"{field} contains invalid characters "
                              f"(allowed: letters, digits, . _ : -)")


def _check_key(value):
    if not isinstance(value, str) or not value:
        raise ValidationError("key must be a non-empty string")
    if len(value) > MAX_KEY_LENGTH:
        raise ValidationError(f"key too long (max {MAX_KEY_LENGTH} chars)")
    if any(ord(ch) < 32 for ch in value):
        raise ValidationError("key must not contain control characters")


def _check_tag(value):
    if not isinstance(value, str) or not value:
        raise ValidationError("tag must be a non-empty string")
    if len(value) > MAX_TAG_LENGTH:
        raise ValidationError(f"tag too long (max {MAX_TAG_LENGTH} chars)")
    if any(ord(ch) < 32 for ch in value):
        raise ValidationError("tag must not contain control characters")


def validate_action(action):
    if not isinstance(action, Action):
        raise ValidationError(f"expected Action, got {type(action).__name__}")
    if action.version != ACTION_VERSION:
        raise ValidationError(f"unsupported action version {action.version}")
    _check_identifier(action.action_id, "action_id", MAX_ID_LENGTH)
    _check_identifier(action.agent, "agent", MAX_AGENT_LENGTH)
    if not isinstance(action.tick, int) or action.tick < 1:
        raise ValidationError("tick must be a positive integer")
    _check_key(action.key)
    if action.op not in ALL_OPS:
        raise ValidationError(f"unknown op {action.op!r}")
    if not isinstance(action.params, dict):
        raise ValidationError("params must be a mapping")
    try:
        params_size = len(json.dumps(action.params, sort_keys=True, ensure_ascii=False))
    except (TypeError, ValueError):
        raise ValidationError("params must be JSON-serializable")
    if params_size > MAX_PARAMS_BYTES:
        raise ValidationError(f"params exceed {MAX_PARAMS_BYTES} bytes")
    _validate_params(action.op, action.params)
    return action


def _validate_params(op, params):
    if op in COUNTER_OPS:
        by = params.get("by", 1)
        if not isinstance(by, int) or by < 1:
            raise ValidationError("counter op requires int by >= 1")
    elif op == OP_LWW_SET:
        if "value" not in params or not _is_json_safe(params["value"]):
            raise ValidationError("lww_set requires a JSON-safe value")
    elif op == OP_ORSET_ADD:
        tag = params.get("tag", "")
        _check_tag(tag)
        if "value" not in params or not _is_json_safe(params["value"]):
            raise ValidationError("orset_add requires a JSON-safe value")
    elif op == OP_ORSET_REMOVE:
        tag = params.get("tag", "")
        _check_tag(tag)
    weight = params.get("weight")
    if weight is not None:
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValidationError("weight must be numeric")
        if not 0 <= float(weight) <= 1:
            raise ValidationError("weight must be within [0, 1]")


def canonical_action_bytes(action_or_dict):
    payload = action_or_dict.export() if not isinstance(action_or_dict, dict) else action_or_dict
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


class AgentRegistry:
    def __init__(self):
        self._agents = {}
        self._lock = threading.Lock()

    def register(self, agent_id, secret=None, public_key=None, ops=None, quota=None,
                 namespaces=None):
        if secret is not None:
            kind, key = ALG_HMAC, secret
        elif public_key is not None:
            kind, key = ALG_ED25519, public_key
        else:
            raise ValidationError("register requires a secret or a public_key")
        with self._lock:
            self._agents[agent_id] = {
                "kind": kind,
                "key": key,
                "ops": frozenset(ops) if ops is not None else None,
                "quota": quota,
                "used": 0,
                "namespaces": tuple(namespaces) if namespaces is not None else None,
            }
        return self

    def secret(self, agent_id):
        agent = self._agents.get(agent_id)
        if agent is None or agent["kind"] != ALG_HMAC:
            return None
        return agent["key"]

    def verifier(self, agent_id):
        agent = self._agents.get(agent_id)
        if agent is None:
            return None
        return agent["kind"], agent["key"]

    def authorize(self, agent_id, op):
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        if agent["ops"] is not None and op not in agent["ops"]:
            return False
        return True

    def consume(self, agent_id, n=1):
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            if agent["quota"] is not None:
                if agent["used"] + n > agent["quota"]:
                    return False
                agent["used"] += n
            return True

    def quota_left(self, agent_id):
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None or agent["quota"] is None:
                return None
            return max(0, agent["quota"] - agent["used"])

    def namespace_allows(self, agent_id, key):
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        if agent["namespaces"] is None:
            return True
        return any(key == prefix or key.startswith(prefix) for prefix in agent["namespaces"])


def action_bytes(action):
    return canonical_action_bytes(action.export())


def sign_action(action, secret, alg=ALG_HMAC):
    if alg == ALG_ED25519:
        return sign_ed25519(action, secret)
    if alg != ALG_HMAC:
        raise ValidationError(f"unsupported signature algorithm {alg!r}")
    signature = hmac.new(secret.encode("utf-8"), action_bytes(action), hashlib.sha256).hexdigest()
    return {"action": action.export(), "signed_by": action.agent, "signature": signature,
            "alg": ALG_HMAC}


def verify_action(envelope, secret, alg=ALG_HMAC):
    if alg == ALG_ED25519:
        return verify_ed25519(envelope, secret)
    if not isinstance(secret, str):
        return False
    action_dict = envelope.get("action")
    signed_by = envelope.get("signed_by")
    signature = envelope.get("signature")
    if not isinstance(action_dict, dict) or not isinstance(signed_by, str) or not isinstance(
        signature, str
    ):
        return False
    expected = hmac.new(secret.encode("utf-8"), canonical_action_bytes(action_dict),
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def generate_ed25519_keypair():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    except ImportError:
        raise ImportError("Ed25519 requires the optional 'cryptography' package")
    import os
    seed = os.urandom(32)
    private = Ed25519PrivateKey.from_private_bytes(seed)
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return seed, public


def _ed25519_sign(data, private_key_bytes):
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        raise ImportError("Ed25519 requires the optional 'cryptography' package")
    private = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return private.sign(data)


def _ed25519_verify(data, signature, public_key_bytes):
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        raise ImportError("Ed25519 requires the optional 'cryptography' package")
    public = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    public.verify(signature, data)


def sign_ed25519(action, private_key_bytes):
    signature = _ed25519_sign(action_bytes(action), private_key_bytes)
    return {"action": action.export(), "signed_by": action.agent,
            "signature": base64.b64encode(signature).decode("ascii"), "alg": ALG_ED25519}


def verify_ed25519(envelope, public_key_bytes):
    action_dict = envelope.get("action")
    signature = envelope.get("signature")
    if not isinstance(action_dict, dict) or not isinstance(signature, str):
        return False
    try:
        _ed25519_verify(canonical_action_bytes(action_dict), base64.b64decode(signature),
                        public_key_bytes)
        return True
    except Exception:
        return False


def _envelope_algorithm(envelope):
    alg = envelope.get("alg", ALG_HMAC)
    return alg if alg in (ALG_HMAC, ALG_ED25519) else None


def validate_envelope(envelope, registry):
    if not isinstance(envelope, dict):
        raise ValidationError("envelope must be a mapping")
    action_data = envelope.get("action")
    if not isinstance(action_data, dict):
        raise ValidationError("envelope has no action")
    action = validate_action(Action.import_action(action_data))
    signed_by = envelope.get("signed_by")
    if not isinstance(signed_by, str) or signed_by != action.agent:
        raise ValidationError("signed_by does not match action agent")
    alg = _envelope_algorithm(envelope)
    if alg is None:
        raise ValidationError("envelope uses an unsupported signature algorithm")
    verifier = registry.verifier(signed_by)
    if verifier is None:
        raise ValidationError(f"unknown agent {signed_by!r}")
    kind, key = verifier
    if kind != alg:
        raise ValidationError("envelope algorithm does not match the agent's registered key")
    if alg == ALG_HMAC:
        ok = verify_action(envelope, key)
    else:
        ok = verify_ed25519(envelope, key)
    if not ok:
        raise ValidationError("signature verification failed")
    if not registry.authorize(signed_by, action.op):
        raise ValidationError(f"agent {signed_by!r} is not authorized for {action.op!r}")
    if not registry.namespace_allows(signed_by, action.key):
        raise ValidationError(f"agent {signed_by!r} cannot write keys outside the '{signed_by}' scope")
    if not registry.consume(signed_by):
        raise ValidationError(f"agent {signed_by!r} has exhausted its quota")
    return action