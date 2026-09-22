import json
import ssl
import struct

MAX_FRAME = 64 * 1024 * 1024


class TLSConfig:
    """Thin ssl wrapper so TCP and WebSocket transports share one knob.

    ``certfile``/(``keyfile``) turn the server side on; ``ca`` and ``verify``
    control how outbound (client) connections authenticate the far node.
    """

    def __init__(self, certfile=None, keyfile=None, ca=None, verify=True):
        self._server_ctx = None
        if certfile is not None:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile, keyfile)
            self._server_ctx = ctx
        self._client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if ca is not None:
            self._client_ctx.load_verify_locations(ca)
        if not verify:
            self._client_ctx.check_hostname = False
            self._client_ctx.verify_mode = ssl.CERT_NONE

    @property
    def server_ready(self):
        return self._server_ctx is not None

    def server_wrap(self, sock):
        return self._server_ctx.wrap_socket(sock, server_side=True)

    def client_wrap(self, sock, server_hostname=None):
        return self._client_ctx.wrap_socket(sock, server_hostname=server_hostname)


def normalize_tls(tls, server_side=False):
    """Accept None | TLSConfig | True | cert path | (cert, key[, ca])."""
    if tls is None:
        return None
    if isinstance(tls, TLSConfig):
        return tls
    if isinstance(tls, bool):
        return TLSConfig(verify=True) if tls else None
    if isinstance(tls, str):
        return TLSConfig(ca=tls, verify=True)
    if isinstance(tls, (tuple, list)):
        if len(tls) >= 3:
            return TLSConfig(tls[0], tls[1], ca=tls[2], verify=bool(tls[2]))
        return TLSConfig(tls[0], tls[1] if len(tls) > 1 else None, verify=False)
    raise TypeError(f"cannot interpret tls config {tls!r}")


def send_msg(sock, obj):
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_msg(sock):
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > MAX_FRAME:
        raise ValueError(f"invalid frame length {length}")
    body = _recv_exact(sock, length)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def to_wire(value):
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value