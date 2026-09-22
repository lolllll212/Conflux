import socket

from .actions import Action
from .net import normalize_tls, recv_msg, send_msg
from .state import Replica
from .validate import sign_action
from .websocket import connect_websocket


class Client:
    def __init__(self, host, port, agent_id="client", secret=None, signature_alg="hmac-sha256",
                 transport="tcp", tls=None):
        self.agent_id = agent_id
        self.secret = secret
        self.signature_alg = signature_alg
        self.replica = Replica(agent_id)
        tls_config = normalize_tls(tls)
        if transport == "ws":
            ws = connect_websocket(host, port, ssl_ctx=tls_config)
            self._send = ws.send_json
            self._recv = ws.recv_json
            self._close = ws.close
        elif transport == "tcp":
            sock = socket.create_connection((host, port), timeout=10)
            if tls_config is not None:
                sock = tls_config.client_wrap(sock, server_hostname=host)
            self._send = lambda m: send_msg(sock, m)
            self._recv = lambda: recv_msg(sock)
            self._close = sock.close
        else:
            raise ValueError(f"unknown transport {transport!r}")
        try:
            self._welcome()
        except Exception:
            self._close()
            raise

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

    def set_add(self, key, value, tag=None):
        return self._submit("orset_add", key, {"value": value, "tag": tag or self._next_tag()})

    def set_add_weighted(self, key, value, weight, tag=None):
        return self._submit(
            "orset_add", key, {"value": value, "weight": weight, "tag": tag or self._next_tag()}
        )

    def set_remove(self, key, tag):
        return self._submit("orset_remove", key, {"tag": tag})

    def _next_tag(self):
        return f"{self.agent_id}:{self.replica.timestamp() + 1}"

    def _submit(self, op, key, params):
        action = self.replica._emit(op, key, params)
        if self.secret is not None:
            payload = {"type": "submit",
                       "actions": [sign_action(action, self.secret, self.signature_alg)]}
        else:
            payload = {"type": "submit", "actions": [action.export()]}
        self._send(payload)
        resp = self._recv_safe()
        if not resp or resp.get("type") == "error":
            raise ConnectionError(resp.get("reason", "submit rejected") if resp else "submit rejected")
        if resp.get("type") != "ack":
            raise ConnectionError("server did not acknowledge submit")
        return action

    def read(self, key):
        self._send({"type": "read", "key": key})
        resp = self._recv_safe()
        if not resp or resp.get("type") != "read_resp":
            raise ConnectionError("failed to read")
        return resp.get("value")

    def read_all(self):
        self._send({"type": "read_all"})
        resp = self._recv_safe()
        if not resp or resp.get("type") != "read_resp":
            raise ConnectionError("failed to read_all")
        return resp.get("value")

    def hash(self):
        self._send({"type": "hash"})
        resp = self._recv_safe()
        if not resp or resp.get("type") != "hash_resp":
            raise ConnectionError("failed to read hash")
        return resp.get("hash")

    def sync(self):
        self._send({"type": "sync"})
        msg = self._recv_safe()
        if not msg or msg.get("type") != "journal":
            raise ConnectionError("failed to sync")
        actions = [Action.import_action(d) for d in msg.get("actions", [])]
        return self.replica.absorb(actions)

    def close(self):
        try:
            self._close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False