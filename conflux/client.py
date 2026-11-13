import socket

from .actions import Action
from .net import normalize_tls, recv_msg, send_msg
from .state import Replica
from .validate import sign_action
from .websocket import connect_websocket


class Client:
    def __init__(self, host, port, agent_id="client", secret=None, signature_alg="hmac-sha256",
                 transport="tcp", tls=None, auto_reconnect=True):
        import threading
        import time

        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.secret = secret
        self.signature_alg = signature_alg
        self.transport = transport
        self._tls_config = normalize_tls(tls)
        self.auto_reconnect = auto_reconnect
        self._io_lock = threading.RLock()
        self._tag_seq = 0
        self._closed_explicitly = False
        # Seed initial monotonic tick from epoch microseconds so restarted agents
        # with the same agent_id never reuse old action_ids or lose LWW ordering!
        self.replica = Replica(agent_id)
        self.replica._tick = int(time.time() * 1_000_000)
        self._connect()

    def _connect(self):
        if self.transport == "ws":
            ws = connect_websocket(self.host, self.port, ssl_ctx=self._tls_config)
            self._send = ws.send_json
            self._recv = ws.recv_json
            self._close = ws.close
        elif self.transport == "tcp":
            sock = socket.create_connection((self.host, self.port), timeout=10)
            if self._tls_config is not None:
                sock = self._tls_config.client_wrap(sock, server_hostname=self.host)
            self._send = lambda m: send_msg(sock, m)
            self._recv = lambda: recv_msg(sock)
            self._close = sock.close
        else:
            raise ValueError(f"unknown transport {self.transport!r}")
        try:
            self._welcome()
        except Exception:
            self._close()
            raise

    def _request(self, payload, *, retryable=True):
        with self._io_lock:
            try:
                self._send(payload)
                resp = self._recv_safe()
                if resp is None:
                    raise ConnectionError("connection closed by server")
                return resp
            except (ConnectionError, OSError) as exc:
                if not retryable or not self.auto_reconnect or self._closed_explicitly:
                    raise ConnectionError(str(exc)) from None
                try:
                    self._close()
                except OSError:
                    pass
                self._connect()
                self._send(payload)
                resp = self._recv_safe()
                if resp is None:
                    raise ConnectionError("connection closed by server after reconnect")
                return resp

    def _recv_safe(self):
        try:
            return self._recv()
        except socket.timeout:
            raise ConnectionError("timed out waiting for server") from None

    def _welcome(self):
        self._send({"type": "hello", "role": "agent", "node_id": self.agent_id})
        resp = self._recv_safe()
        if not resp or resp.get("type") != "welcome":
            raise ConnectionError("server did not accept connection")
        self.server_id = resp.get("node_id")

    def counter_inc(self, key, by=1):
        return self._submit("counter_inc", key, {"by": by})

    def counter_dec(self, key, by=1):
        return self._submit("counter_dec", key, {"by": by})

    def register_set(self, key, value):
        return self._submit("lww_set", key, {"value": value})

    def register_set_weighted(self, key, value, weight):
        return self._submit("lww_set", key, {"value": value, "weight": weight})

    def mv_set(self, key, value, supersedes=None):
        if supersedes is None:
            cands = self.candidates(key)
            supersedes = [c["dot"] for c in cands]
        else:
            supersedes = [str(s) for s in supersedes]
        return self._submit("mv_set", key, {"value": value, "supersedes": supersedes})

    def mv_resolve(self, key, resolved_value):
        cands = self.candidates(key)
        dots = [c["dot"] for c in cands]
        return self._submit("mv_set", key, {"value": resolved_value, "supersedes": dots})

    def set_add(self, key, value, tag=None):
        return self._submit("orset_add", key, {"value": value, "tag": tag or self._next_tag()})

    def set_add_weighted(self, key, value, weight, tag=None):
        return self._submit(
            "orset_add", key, {"value": value, "weight": weight, "tag": tag or self._next_tag()}
        )

    def set_remove(self, key, tag):
        return self._submit("orset_remove", key, {"tag": tag})

    def _next_tag(self):
        with self._io_lock:
            self._tag_seq += 1
            return f"{self.agent_id}:{self.replica.timestamp() + self._tag_seq}"

    def submit_batch(self, op_specs):
        """Submit multiple operations in a single atomic network round-trip.

        Each spec is a tuple of (op, key, params).
        """
        if not op_specs:
            return []
        with self._io_lock:
            actions = [self.replica._emit(op, key, params) for op, key, params in op_specs]
            if self.secret is not None:
                payload = {"type": "submit",
                           "actions": [sign_action(a, self.secret, self.signature_alg) for a in actions]}
            else:
                payload = {"type": "submit", "actions": [a.export() for a in actions]}
            # Actions are idempotent by action_id on the server, so retry on broken idle socket is safe
            resp = self._request(payload, retryable=True)
            if resp.get("type") == "error":
                raise ConnectionError(resp.get("reason", "submit rejected"))
            if resp.get("type") != "ack":
                raise ConnectionError("server did not acknowledge submit")
            return actions

    def batch(self):
        """Context manager to batch multiple mutations into a single network frame:

            with client.batch() as b:
                b.counter_inc("orders", 1)
                b.register_set("status", "ready")
        """
        return _ClientBatch(self)

    def _submit(self, op, key, params):
        with self._io_lock:
            action = self.replica._emit(op, key, params)
            if self.secret is not None:
                payload = {"type": "submit",
                           "actions": [sign_action(action, self.secret, self.signature_alg)]}
            else:
                payload = {"type": "submit", "actions": [action.export()]}
            resp = self._request(payload, retryable=True)
            if resp.get("type") == "error":
                raise ConnectionError(resp.get("reason", "submit rejected"))
            if resp.get("type") != "ack":
                raise ConnectionError("server did not acknowledge submit")
            return action

    def read(self, key):
        resp = self._request({"type": "read", "key": key})
        if resp.get("type") != "read_resp":
            raise ConnectionError("failed to read")
        return resp.get("value")

    def read_all(self):
        resp = self._request({"type": "read_all"})
        if resp.get("type") != "read_resp":
            raise ConnectionError("failed to read_all")
        return resp.get("value")

    def conflicts(self, prefix=None):
        resp = self._request({"type": "conflicts", "prefix": prefix})
        if resp.get("type") != "conflicts_resp":
            raise ConnectionError("failed to read conflicts")
        return resp.get("value") or {}

    def candidates(self, key):
        resp = self._request({"type": "candidates", "key": key})
        if resp.get("type") != "candidates_resp":
            raise ConnectionError("failed to read candidates")
        return resp.get("value") or []

    def hash(self):
        resp = self._request({"type": "hash"})
        if resp.get("type") != "hash_resp":
            raise ConnectionError("failed to read hash")
        return resp.get("hash")

    def sync(self):
        msg = self._request({"type": "sync"})
        if msg.get("type") != "journal":
            raise ConnectionError("failed to sync")
        actions = [Action.import_action(d) for d in msg.get("actions", [])]
        return self.replica.absorb(actions)

    def close(self):
        self._closed_explicitly = True
        try:
            self._close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _ClientBatch:
    def __init__(self, client):
        self._client = client
        self._specs = []
        self.actions = []

    def counter_inc(self, key, by=1):
        self._specs.append(("counter_inc", key, {"by": by}))
        return self

    def counter_dec(self, key, by=1):
        self._specs.append(("counter_dec", key, {"by": by}))
        return self

    def register_set(self, key, value):
        self._specs.append(("lww_set", key, {"value": value}))
        return self

    def register_set_weighted(self, key, value, weight):
        self._specs.append(("lww_set", key, {"value": value, "weight": weight}))
        return self

    def mv_set(self, key, value, supersedes=None):
        if supersedes is None:
            cands = self._client.candidates(key)
            supersedes = [c["dot"] for c in cands]
        else:
            supersedes = [str(s) for s in supersedes]
        self._specs.append(("mv_set", key, {"value": value, "supersedes": supersedes}))
        return self

    def mv_resolve(self, key, resolved_value):
        cands = self._client.candidates(key)
        dots = [c["dot"] for c in cands]
        self._specs.append(("mv_set", key, {"value": resolved_value, "supersedes": dots}))
        return self

    def set_add(self, key, value, tag=None):
        tag = tag or self._client._next_tag()
        self._specs.append(("orset_add", key, {"value": value, "tag": tag}))
        return self

    def set_add_weighted(self, key, value, weight, tag=None):
        tag = tag or self._client._next_tag()
        self._specs.append(("orset_add", key, {"value": value, "weight": weight, "tag": tag}))
        return self

    def set_remove(self, key, tag):
        self._specs.append(("orset_remove", key, {"tag": tag}))
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self._specs:
            self.actions = self._client.submit_batch(self._specs)
        return False