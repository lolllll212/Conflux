import json
import ssl
import struct

MAX_FRAME = 64 * 1024 * 1024
# A frame's nesting depth, not its size, is what kills a json decoder: the C
# scanner recurses per level and raises RecursionError part way through. Conflux
# frames are flat objects of flat actions, so 128 is generous for real traffic
# and far below the point where the decoder starts failing.
MAX_DEPTH = 128
_DEEP_RUN = max(24, min(64, MAX_DEPTH // 2))


class ProtocolError(ValueError):
    """A frame this node will not accept: oversized, truncated, or not a JSON object.

    Subclasses ``ValueError`` deliberately - every caller already treats a bad
    frame as "drop this peer" by catching ValueError, and an unclassified
    ``RecursionError`` from the decoder is what used to escape that net and print
    a traceback from a worker thread.
    """


class FrameTooLarge(ProtocolError):
    """The payload exceeds the frame cap, so the *sender* must chunk it."""


def _reject_nesting(raw):
    """Cheap pre-scan: only pathological input has long runs of openers."""
    if b"[" * _DEEP_RUN in raw or b"{" * _DEEP_RUN in raw:
        raise ProtocolError(f"frame nests more than {_DEEP_RUN} levels deep; "
                            f"refusing to decode it")


def dumps_frame(obj, *, max_frame=MAX_FRAME):
    """Serialise a frame, refusing one the peer is contractually allowed to reject."""
    try:
        payload = json.dumps(obj).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not serialisable: {exc}") from exc
    if max_frame and len(payload) > max_frame:
        raise FrameTooLarge(f"frame is {len(payload)} bytes, over the {max_frame} byte cap; "
                            f"split the payload (batch_size/frame_batch_bytes)")
    return payload




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


def send_msg(sock, obj, *, max_frame=MAX_FRAME):
    """Write one length-prefixed frame. Returns the payload size.

    The size is checked *before* anything is written: a frame the peer would reject
    on the floor used to be written anyway, so the link closed with no clue on this
    side and the caller retried the same oversize payload forever.
    """
    payload = dumps_frame(obj, max_frame=max_frame)
    sock.sendall(struct.pack(">I", len(payload)) + payload)
    return len(payload)


def recv_msg(sock, *, max_frame=MAX_FRAME):
    """Read one frame into a dict, or raise ProtocolError. ``None`` means EOF."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > max_frame:
        raise ProtocolError(f"invalid frame length {length} (cap {max_frame})")
    body = _recv_exact(sock, length)
    if body is None:
        raise ProtocolError(f"peer closed after {len(header)}/1+{length} bytes: truncated frame")
    _reject_nesting(body)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"frame is not valid utf-8: {exc}") from exc
    try:
        msg = json.loads(text)
    except RecursionError as exc:
        raise ProtocolError(f"frame nests too deeply to decode: {exc}") from exc
    except ValueError as exc:
        raise ProtocolError(f"frame is not JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise ProtocolError(f"frame must be a JSON object, got {type(msg).__name__}")
    return msg


def _recv_exact(sock, n):
    """Read exactly ``n`` bytes into one buffer.

    ``buf += chunk`` over a 64 MB frame is quadratic and keeps every partial copy
    alive, which turns a slow or dribbling peer into a memory amplifier; a fixed
    bytearray read into in place costs what the frame costs.
    """
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        try:
            count = sock.recv_into(view[got:], n - got)
        except AttributeError:  # a socket-like without recv_into (test doubles)
            chunk = sock.recv(n - got)
            if not chunk:
                return None
            count = len(chunk)
            buf[got:got + count] = chunk
        if not count:
            return None
        got += count
    return bytes(buf)




def to_wire(value):
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value