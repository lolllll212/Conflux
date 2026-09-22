import base64
import hashlib
import json
import os
import socket
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .net import MAX_FRAME

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUE = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


class WebSocketError(ConnectionError):
    pass


def accept_nonce(nonce):
    return base64.b64encode(hashlib.sha1((nonce + WS_GUID).encode("ascii")).digest()).decode(
        "ascii"
    )


def _build_frame(opcode, payload, mask_key=None, fin=True):
    first = (0x80 if fin else 0) | opcode
    blen = len(payload)
    mask_bit = 0x80 if mask_key is not None else 0
    if blen < 126:
        header = struct.pack(">BB", first, mask_bit | blen)
    elif blen < 65536:
        header = struct.pack(">BBH", first, mask_bit | 126, blen)
    else:
        header = struct.pack(">BBQ", first, mask_bit | 127, blen)
    if mask_key is not None:
        mask = mask_key[:4]
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return header + mask + body
    return header + payload


class _BufferedReader:
    def __init__(self, sock):
        self._sock = sock
        self._buf = b""
        self._pos = 0

    def read(self, n):
        while len(self._buf) - self._pos < n:
            chunk = self._sock.recv(max(n - (len(self._buf) - self._pos), 65536))
            if not chunk:
                return None
            self._buf += chunk
        data = self._buf[self._pos:self._pos + n]
        self._pos += n
        if self._pos > 65536 and self._pos == len(self._buf):
            self._buf = b""
            self._pos = 0
        return data

    def read_until(self, delim):
        while True:
            idx = self._buf.find(delim, self._pos)
            if idx >= 0:
                end = idx + len(delim)
                data = self._buf[self._pos:end]
                self._pos = end
                return data
            chunk = self._sock.recv(65536)
            if not chunk:
                return None
            self._buf += chunk


class WebSocket:
    """A single RFC6455 duplex connection.

    ``client`` selects framing rules: clients must mask outgoing frames,
    servers must not. Message APIs speak bytes on the wire and JSON here.
    """

    def __init__(self, sock, reader=None, client=False):
        self._sock = sock
        self._reader = reader or _BufferedReader(sock)
        self._client = client
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def send(self, payload, binary=True):
        if self._closed:
            raise WebSocketError("websocket is closed")
        mask_key = os.urandom(4) if self._client else None
        opcode = OP_BINARY if binary else OP_TEXT
        self._sock.sendall(_build_frame(opcode, bytes(payload), mask_key))

    def send_json(self, obj):
        self.send(json.dumps(obj).encode("utf-8"))

    def recv(self):
        """Return the next complete message payload, or None on close."""
        while not self._closed:
            frame = self._read_frame()
            if frame is None:
                self._closed = True
                return None
            opcode, fin, payload = frame
            if opcode == OP_CLOSE:
                self.close(echo=True)
                return None
            if opcode == OP_PING:
                self._sock.sendall(_build_frame(OP_PONG, payload, None))
                continue
            if opcode == OP_PONG:
                continue
            if opcode in (OP_TEXT, OP_BINARY):
                if fin:
                    return payload
                parts = [payload]
                while True:
                    next_frame = self._read_frame()
                    if next_frame is None:
                        self._closed = True
                        return None
                    nxt, is_fin, data = next_frame
                    if nxt == OP_CONTINUE:
                        parts.append(data)
                        if is_fin:
                            return b"".join(parts)
                    elif nxt in (OP_TEXT, OP_BINARY, OP_CLOSE):
                        raise WebSocketError("interleaved data frame during fragmentation")
        return None

    def recv_json(self):
        data = self.recv()
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))

    def _read_frame(self):
        header = self._reader.read(1)
        if header is None:
            return None
        b0 = header[0]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        b1 = self._reader.read(1)
        if b1 is None:
            return None
        masked = bool(b1[0] & 0x80)
        length = b1[0] & 0x7F
        if length == 126:
            ext = self._reader.read(2)
            if ext is None:
                return None
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            ext = self._reader.read(8)
            if ext is None:
                return None
            length = struct.unpack(">Q", ext)[0]
        if length > MAX_FRAME:
            raise WebSocketError(f"frame too large ({length} bytes)")
        if masked:
            mask = self._reader.read(4)
            if mask is None:
                return None
        else:
            mask = None
        payload = self._reader.read(length)
        if payload is None:
            return None
        if mask is not None:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, fin, payload

    def close(self, echo=False):
        if self._closed:
            return
        self._closed = True
        try:
            if not echo:
                self._sock.sendall(_build_frame(OP_CLOSE, struct.pack(">H", 1000), None))
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _WSHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        nonce = self.headers.get("Sec-WebSocket-Key")
        upgrade = (self.headers.get("Upgrade") or "").lower()
        connection = (self.headers.get("Connection") or "").lower()
        if nonce is None or upgrade != "websocket" or "upgrade" not in connection:
            self.send_error(400, "expected a websocket upgrade")
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_nonce(nonce))
        self.end_headers()
        self.connection.settimeout(10)
        ws = WebSocket(self.connection, client=False)
        try:
            self.server.on_ws(ws)
        finally:
            ws.close()
            self.close_connection = True

    def log_message(self, *args):
        pass


class WebSocketServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        super().server_bind()

    def __init__(self, host, port, on_ws, ssl_ctx=None):
        self.on_ws = on_ws
        super().__init__((host, port), _WSHandler)
        if ssl_ctx is not None:
            self.socket = ssl_ctx.server_wrap(self.socket)

    @property
    def address(self):
        return self.server_address


def connect_websocket(host, port, path="/", timeout=10, ssl_ctx=None):
    sock = socket.create_connection((host, port), timeout=timeout)
    if ssl_ctx is not None:
        sock = ssl_ctx.client_wrap(sock, server_hostname=host)
    try:
        sock.settimeout(timeout)
        nonce = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {nonce}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(request.encode("ascii"))
        reader = _BufferedReader(sock)
        status = reader.read_until(b"\r\n")
        if status is None:
            raise WebSocketError("connection closed during handshake")
        headers = {}
        while True:
            line = reader.read_until(b"\r\n")
            if line is None or line == b"\r\n":
                break
            name, _, value = line.partition(b":")
            headers[name.strip().decode("ascii").lower()] = value.strip().decode("ascii")
        if b" 101 " not in status:
            raise WebSocketError(f"handshake failed: {status.decode('ascii', 'replace').strip()}")
        if headers.get("sec-websocket-accept") != accept_nonce(nonce):
            raise WebSocketError("handshake accept mismatch")
        return WebSocket(sock, reader=reader, client=True)
    except Exception:
        sock.close()
        raise


class WSMessenger:
    """JSON-message adapter so a WebSocket behaves like the TCP framed path."""

    def __init__(self, ws):
        self._ws = ws

    def send(self, obj):
        self._ws.send_json(obj)

    def recv(self):
        return self._ws.recv_json()

    def close(self):
        self._ws.close()

    @property
    def closed(self):
        return self._ws.closed