import argparse
import queue
import socket
import ssl
import threading
import time

from .actions import Action
from .app import EventBus, Watcher, attach_listeners
from .net import normalize_tls, recv_msg, send_msg, to_wire
from .protocol import state_hash
from .state import Replica
from .storage import JournalStore
from .validate import ValidationError, validate_action, validate_envelope
from .websocket import WebSocketServer, connect_websocket


class Server:
    def __init__(self, node_id, address, peers=(), data_dir=None, gossip_interval=1.0,
                 snapshot_threshold=1000, registry=None, transport="tcp", batch_size=0,
                 auto_discover=False, tls=None, journal_secret=None, stale_after=None):
        self.node_id = node_id
        self.address = tuple(address)
        self.peers = [tuple(p) for p in peers]
        self.gossip_interval = gossip_interval
        self.node_address = None
        self.registry = registry
        self.transport = transport
        self.batch_size = int(batch_size)
        self.auto_discover = auto_discover
        self._tls = normalize_tls(tls)
        self.stale_after = stale_after if stale_after is not None else max(3.0, gossip_interval * 3)
        self.replica = Replica(node_id)
        self.events = EventBus()
        self.watchers = Watcher(self.replica)
        attach_listeners(self.replica, self.events, self.watchers, self._count_absorbed)
        self.lock = threading.RLock()
        self._stopped = threading.Event()
        self._started_at = None
        self._metrics = {"absorbed": 0, "submitted": 0}
        self._socket = None
        self._ws_server = None
        self._metrics_http = None
        self._links = {}
        self._outbound = {}
        self._peer_failures = {}
        self._dialing = set()
        self._gossip_tick = 0
        self.discovered = set()
        self.members = {}
        self._link_by_address = {}

        self.store = JournalStore(data_dir, snapshot_threshold, journal_secret) if data_dir else None
        if self.store:
            root, actions = self.store.load()
            if root is not None:
                self.replica.state._map = root
            for action in actions:
                self.replica.apply(action)

    def start(self):
        self._started_at = time.time()
        if self.transport == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(self.address)
            sock.listen(32)
            if self._tls is not None and self._tls.server_ready:
                sock = self._tls.server_wrap(sock)
            self._socket = sock
            self.node_address = sock.getsockname()
            threading.Thread(target=self._accept_loop, daemon=True).start()
        elif self.transport == "ws":
            ssl_ctx = None
            if self._tls is not None and self._tls.server_ready:
                ssl_ctx = self._tls
            self._ws_server = WebSocketServer(self.address[0], self.address[1],
                                              self._handle_ws_conn, ssl_ctx=ssl_ctx)
            threading.Thread(target=self._ws_server.serve_forever, daemon=True).start()
            self.node_address = self._ws_server.address
        else:
            raise ValueError(f"unknown transport {self.transport!r}")
        threading.Thread(target=self._gossip_loop, daemon=True).start()
        return self

    def add_peer(self, address):
        address = tuple(address)
        if address not in self.peers and address != self.node_address:
            self.peers.append(address)
        self._ensure_peer(address)

    def stop(self):
        self._stopped.set()
        for info in list(self._links.values()):
            try:
                info["outgoing"].put_nowait(None)
            except (queue.Full, AttributeError):
                pass
            try:
                info["close"]()
            except OSError:
                pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._ws_server is not None:
            try:
                self._ws_server.shutdown()
                self._ws_server.server_close()
            except OSError:
                pass
        if self._metrics_http is not None:
            try:
                self._metrics_http.shutdown()
                self._metrics_http.server_close()
            except OSError:
                pass
        self._links.clear()
        self._outbound.clear()
        if self.store is not None:
            self.store.close()

    def read(self, key):
        with self.lock:
            return self.replica.read(key)

    def subscribe(self, handler, op=None, key=None, key_prefix=None):
        return self.events.subscribe(handler, op=op, key=key, key_prefix=key_prefix)

    def watch(self, key, handler):
        return self.watchers.watch(key, handler)

    def metrics(self):
        with self.lock:
            return {
                "node_id": self.node_id,
                "transport": self.transport,
                "uptime_s": round(time.time() - self._started_at, 3) if self._started_at else 0,
                "absorbed": self._metrics["absorbed"],
                "submitted": self._metrics["submitted"],
                "journal": len(self.replica.journal),
                "links": len(self._links),
                "outbound": len(self._outbound),
                "peers": len(self.peers),
                "discovered": len(self.discovered),
                "members": len(self.members),
                "gossip_interval": self.gossip_interval,
                "state_hash": state_hash(self.replica.state),
            }

    def prometheus(self, namespace="conflux"):
        """Expose metrics in Prometheus text exposition format (no deps)."""
        m = self.metrics()
        labels = f'node_id="{self.node_id}"'
        counters = {
            "actions_absorbed_total": ("Total actions absorbed by this node.", m["absorbed"]),
            "actions_submitted_total": ("Total submit ops accepted from agents.", m["submitted"]),
        }
        gauges = {
            "journal_actions": ("Actions held in the journal.", m["journal"]),
            "links": ("Open peer links.", m["links"]),
            "outbound": ("Outbound dialed peer links.", m["outbound"]),
            "peers": ("Configured peer addresses.", m["peers"]),
            "discovered": ("Advertised addresses discovered via peers.", m["discovered"]),
            "members": ("Peers known in the membership view.", m["members"]),
            "uptime_seconds": ("Seconds since this node started.", m["uptime_s"]),
            "gossip_interval_seconds": ("Gossip round interval.", m["gossip_interval"]),
        }
        lines = []
        for name, (help_text, value) in counters.items():
            lines += [f"# HELP {namespace}_{name} {help_text}",
                      f"# TYPE {namespace}_{name} counter",
                      f"{namespace}_{name}{{{labels}}} {value}"]
        for name, (help_text, value) in gauges.items():
            lines += [f"# HELP {namespace}_{name} {help_text}",
                      f"# TYPE {namespace}_{name} gauge",
                      f"{namespace}_{name}{{{labels}}} {value}"]
        lines += [f'# HELP {namespace}_state_hash_info Current canonical state hash.',
                  f'# TYPE {namespace}_state_hash_info gauge',
                  f'{namespace}_state_hash_info{{node_id="{self.node_id}",hash="{m["state_hash"]}"}} 1']
        return "\n".join(lines) + "\n"

    def start_metrics_http(self, host="127.0.0.1", port=0):
        """Serve Prometheus text on HTTP GET /metrics (stdlib http.server)."""
        if self._metrics_http is not None:
            return self._metrics_http.server_address
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?")[0] != "/metrics":
                    self.send_error(404)
                    return
                if self.server.get_metrics is None:
                    self.send_error(404)
                    return
                body = self.server.get_metrics().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        httpd = _Server((host, port), _Handler)
        httpd.get_metrics = self.prometheus
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._metrics_http = httpd
        return httpd.server_address

    def _count_absorbed(self, actions, origin="absorb"):
        with self.lock:
            self._metrics["absorbed"] += len(actions)

    def view(self):
        with self.lock:
            return self.replica.state.view()

    def hash(self):
        with self.lock:
            return state_hash(self.replica.state)

    def discovery(self):
        with self.lock:
            return sorted(self.discovered)

    def _accept_loop(self):
        while not self._stopped.is_set():
            try:
                conn, _ = self._socket.accept()
            except OSError:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn):
        conn.settimeout(10)

        def send(obj):
            send_msg(conn, obj)

        def recv():
            return recv_msg(conn)

        try:
            self._conn_loop(send, recv, conn.close)
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _handle_ws_conn(self, ws):
        try:
            self._conn_loop(ws.send_json, lambda: ws.recv_json(), ws.close)
        except (ConnectionError, OSError, ValueError):
            pass

    def _conn_loop(self, send, recv, close):
        hello = recv()
        if not hello:
            return
        if hello.get("role") == "peer":
            send({"type": "hello_ack", "node_id": self.node_id,
                  "peers": self._peer_list(), "members": self._member_list()})
            peer_id = hello.get("node_id")
            far_address = tuple(hello["address"]) if hello.get("address") else None
            key = self._register_link(send, recv, close=close,
                                      address=far_address, peer_id=peer_id)
            if hello.get("members"):
                self._member_see_all(hello["members"])
            candidates = [far_address] + list(hello.get("peers", []))
            self._discover([c for c in candidates if c])
            try:
                self._send_all(key)
            except (ConnectionError, OSError, ValueError):
                self._drop_link(key)
                return
            self._receive_loop(recv, send, key)
        else:
            send({"type": "welcome", "node_id": self.node_id,
                  "peers": self._peer_list(), "members": self._member_list()})
            if hello.get("members"):
                self._member_see_all(hello["members"])
            self._receive_loop(recv, send, None)

    def _receive_loop(self, recv, send, key):
        while not self._stopped.is_set():
            try:
                msg = recv()
            except (ConnectionError, OSError, ValueError):
                break
            if msg is None:
                break
            self._touch_link(key)
            try:
                self._handle_message(msg, send, key)
            except (ConnectionError, OSError, ValueError):
                break
        if key is not None:
            self._drop_link(key)

    def _handle_message(self, msg, send, source):
        if msg.get("type") == "submit":
            try:
                if self.registry is not None:
                    actions = [validate_envelope(env, self.registry) for env in msg.get("actions", [])]
                else:
                    actions = [Action.import_action(d) for d in msg.get("actions", [])]
                    for action in actions:
                        validate_action(action)
            except ValidationError as exc:
                send({"type": "error", "reason": str(exc)})
                return
            self._absorb(actions)
            with self.lock:
                self._metrics["submitted"] += len(actions)
            send({"type": "ack", "node_id": self.node_id, "accepted": len(actions)})
        elif msg.get("type") == "read":
            send({"type": "read_resp", "key": msg["key"], "value": to_wire(self.read(msg["key"]))})
        elif msg.get("type") == "read_all":
            send({"type": "read_resp", "value": to_wire(self.view())})
        elif msg.get("type") == "hash":
            send({"type": "hash_resp", "hash": self.hash()})
        elif msg.get("type") == "sync":
            self._send_journal(send)
        elif msg.get("type") == "journal":
            try:
                actions = [validate_action(Action.import_action(d))
                           for d in msg.get("actions", [])]
            except (TypeError, ValueError, KeyError, ValidationError) as exc:
                raise ValueError(f"malformed peer journal: {exc}") from exc
            self._absorb(actions)
        elif msg.get("type") == "hello_ack":
            self._discover(msg.get("peers", []))
            if msg.get("members"):
                self._member_see_all(msg["members"])
        elif msg.get("type") == "discover":
            send({"type": "peers", "peers": self._peer_list()})
        elif msg.get("type") == "admin":
            if msg.get("what") == "stats":
                send({"type": "stats", **self.metrics()})
            elif msg.get("what") == "prometheus":
                send({"type": "prometheus", "body": self.prometheus()})

    def _absorb(self, actions):
        if not actions:
            return
        with self.lock:
            absorbed = self.replica.absorb(actions)
            if absorbed and self.store is not None:
                rotate = False
                for action in absorbed:
                    rotate = self.store.append(action) or rotate
                if rotate:
                    self.store.rotate(self.replica.state._map)

    def _send_journal(self, send):
        with self.lock:
            actions = [a.export() for a in self.replica.journal]
        send({"type": "journal", "actions": actions})

    def _peer_list(self):
        with self.lock:
            known = {tuple(a) for a in self.peers}
            if self.node_address is not None:
                known.discard(self.node_address)
            known.update(self.discovered)
            return [list(a) for a in sorted(known)]

    def _discover(self, addresses):
        if not addresses:
            return
        with self.lock:
            ours = self.node_address
            for address in addresses:
                address = tuple(address)
                if address == ours or address in self._outbound or address in self.peers:
                    continue
                self.discovered.add(address)

    def _register_link(self, send, recv, close=None, address=None, peer_id=None):
        key = id(send)
        with self.lock:
            self._links[key] = {"send": send, "recv": recv, "close": close or (lambda: None),
                                "sent": set(), "queued": set(), "address": address,
                                "peer_id": peer_id, "last_seen": time.time(),
                                "outgoing": queue.Queue(maxsize=256)}
            if address is not None:
                self._link_by_address[address] = key
        if peer_id:
            self._member_see(peer_id, address=address)
        threading.Thread(target=self._link_sender, args=(key,), daemon=True).start()
        return key

    def _link_sender(self, key):
        info = self._links.get(key)
        if info is None:
            return
        while not self._stopped.is_set():
            msg = info["outgoing"].get()
            if msg is None or self._stopped.is_set():
                return
            try:
                info["send"](msg)
            except (ConnectionError, OSError, ValueError):
                self._drop_link(key)
                return
            with self.lock:
                for export in msg.get("actions", ()):
                    info["sent"].add(export["action_id"])
                    info["queued"].discard(export["action_id"])

    def _enqueue(self, key, payload, actions):
        info = self._links.get(key)
        if info is None:
            return
        try:
            info["outgoing"].put_nowait(payload)
        except queue.Full:
            return
        with self.lock:
            info["queued"].update(a["action_id"] for a in actions)

    def _touch_link(self, key):
        if key is not None:
            info = self._links.get(key)
            if info is not None:
                info["last_seen"] = time.time()

    def _member_see(self, node_id, address=None):
        with self.lock:
            entry = self.members.setdefault(node_id, {"address": None, "last_seen": time.time()})
            if address is not None:
                entry["address"] = address
            entry["last_seen"] = time.time()

    def _member_see_all(self, members):
        for item in members:
            if isinstance(item, dict):
                self._member_see(item.get("node_id"), tuple(item["address"]) if item.get("address") else None)
            else:
                self._member_see(item)

    def _member_list(self):
        with self.lock:
            return [{"node_id": n, "address": list(e["address"])} if e["address"] else n
                    for n, e in sorted(self.members.items())]

    def members_view(self):
        with self.lock:
            return {n: dict(info, **{"address": list(info["address"]) if info["address"] else None})
                    for n, info in self.members.items()}

    def _send_all(self, key):
        info = self._links.get(key)
        if info is None:
            return
        with self.lock:
            actions = [a.export() for a in self.replica.journal]
        self._enqueue(key, {"type": "journal", "actions": actions}, actions)

    def _drop_link(self, key):
        with self.lock:
            info = self._links.get(key)
            if info is not None:
                self._links.pop(key, None)
                if info.get("address"):
                    self._link_by_address.pop(info["address"], None)
            for address, link_key in list(self._outbound.items()):
                if link_key == key:
                    self._outbound.pop(address, None)
                    self._link_by_address.pop(tuple(address), None)
        if info is not None:
            try:
                info["outgoing"].put_nowait(None)
            except (queue.Full, AttributeError):
                pass
            try:
                info["close"]()
            except OSError:
                pass

    def _gossip_loop(self):
        while not self._stopped.is_set():
            now = time.time()
            for key, info in list(self._links.items()):
                if info.get("address") and now - info.get("last_seen", now) > self.stale_after:
                    with self.lock:
                        self._peer_failures[info["address"]] = self._peer_failures.get(
                            info["address"], 0) + 1
                    self._drop_link(key)
            for address in list(self.peers):
                address = tuple(address)
                if address in self._outbound or address in self._dialing:
                    continue
                failures = self._peer_failures.get(address, 0)
                if failures and self._gossip_tick % (2 ** min(failures, 6)) != 0:
                    continue
                with self.lock:
                    if address in self._dialing:
                        continue
                    self._dialing.add(address)
                threading.Thread(target=self._dial_worker, args=(address,), daemon=True).start()
            if self.auto_discover:
                self._merge_discovery()
            self._push_to_peers()
            self._gossip_tick += 1
            self._stopped.wait(self.gossip_interval)

    def _dial_worker(self, address):
        try:
            self._ensure_peer(address)
        finally:
            with self.lock:
                self._dialing.discard(address)

    def _merge_discovery(self):
        with self.lock:
            lookups = [tuple(a) for a in self.discovered if tuple(a) not in self.peers]
        for address in lookups:
            if address not in self.peers:
                self.peers.append(address)

    def _ensure_peer(self, address):
        address = tuple(address)
        if address == self.node_address or address in self._outbound:
            return
        link = self._dial(address)
        if link is None:
            with self.lock:
                self._peer_failures[address] = self._peer_failures.get(address, 0) + 1
            return
        key = self._register_link(link["send"], link["recv"], close=link["close"],
                                  address=address, peer_id=link.get("node_id"))
        with self.lock:
            self._outbound[address] = key
            self._peer_failures[address] = 0
        try:
            self._send_all(key)
        except (ConnectionError, OSError, ValueError):
            self._drop_link(key)
            return
        threading.Thread(target=self._receive_loop,
                         args=(link["recv"], link["send"], key), daemon=True).start()

    def _dial(self, address):
        hello = {"type": "hello", "role": "peer", "node_id": self.node_id,
                 "address": list(self.node_address), "peers": self._peer_list(),
                 "members": self._member_list()}
        try:
            if self.transport == "ws":
                ssl_ctx = None
                if self._tls is not None and self._tls.server_ready:
                    ssl_ctx = self._tls
                ws = connect_websocket(address[0], address[1], ssl_ctx=ssl_ctx)
                ws.send_json(hello)
                ack = ws.recv_json()
                if not ack or ack.get("type") != "hello_ack":
                    ws.close()
                    return None
                self._discover(ack.get("peers", []))
                if ack.get("members"):
                    self._member_see_all(ack["members"])
                return {"send": ws.send_json, "recv": ws.recv_json, "close": ws.close,
                        "node_id": ack.get("node_id")}
            sock = socket.create_connection(address, timeout=3)
            sock.settimeout(10)
            if self._tls is not None and self._tls.server_ready:
                try:
                    sock = self._tls.client_wrap(sock, server_hostname=address[0])
                except (ConnectionError, OSError, ValueError, ssl.SSLError):
                    sock.close()
                    return None
            send_msg(sock, hello)
            ack = recv_msg(sock)
            if not ack or ack.get("type") != "hello_ack":
                sock.close()
                return None
            self._discover(ack.get("peers", []))
            if ack.get("members"):
                self._member_see_all(ack["members"])
            return {"send": lambda m: send_msg(sock, m), "recv": lambda: recv_msg(sock),
                    "close": sock.close, "node_id": ack.get("node_id")}
        except (ConnectionError, OSError, ValueError):
            return None

    def _push_to_peers(self):
        for key, info in list(self._links.items()):
            try:
                with self.lock:
                    pending = [a.export() for a in self.replica.journal
                               if a.action_id not in info["sent"]
                               and a.action_id not in info["queued"]]
                if not pending:
                    continue
                chunks = [pending[i:i + self.batch_size] for i in range(0, len(pending),
                          self.batch_size)] if self.batch_size > 0 else [pending]
                for chunk in chunks:
                    self._enqueue(key, {"type": "journal", "actions": chunk}, chunk)
            except (KeyError, AttributeError):
                continue


def main(argv=None):
    parser = argparse.ArgumentParser(prog="conflux.server", description="conflux node (CRDT backend)")
    parser.add_argument("--id", required=True, help="unique node id")
    parser.add_argument("--listen", required=True, help="bind address, e.g. 127.0.0.1:7001")
    parser.add_argument("--peers", default="", help="comma-separated node addresses to gossip with")
    parser.add_argument("--data", default=None, help="data directory for journal + snapshots")
    parser.add_argument("--gossip-interval", type=float, default=1.0)
    parser.add_argument("--snapshot-threshold", type=int, default=1000)
    parser.add_argument("--transport", choices=("tcp", "ws"), default="tcp")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="max journal actions per gossip frame (0 = unlimited)")
    parser.add_argument("--auto-discover", action="store_true",
                        help="wire peers advertised by linked nodes")
    parser.add_argument("--cert", default=None, help="TLS certificate (PEM); enables TLS framing")
    parser.add_argument("--key", default=None, help="TLS private key (PEM), if not in --cert")
    parser.add_argument("--journal-key", default=None,
                        help="HMAC key to sign journal lines; verified on replay")
    parser.add_argument("--stale-after", type=float, default=None,
                        help="seconds without link traffic before a peer is dropped (default 3x gossip)")
    parser.add_argument("--metrics-http", action="store_true",
                        help="expose Prometheus text on 127.0.0.1:<ephemeral>/metrics")
    args = parser.parse_args(argv)

    def split_address(text):
        host, _, port = text.rpartition(":")
        if not host or not port:
            raise SystemExit(f"invalid address {text!r}, expected host:port")
        return host, int(port)

    peers = [split_address(p) for p in args.peers.split(",") if p.strip()]
    server = Server(
        args.id,
        split_address(args.listen),
        peers=peers,
        data_dir=args.data,
        gossip_interval=args.gossip_interval,
        snapshot_threshold=args.snapshot_threshold,
        transport=args.transport,
        batch_size=args.batch_size,
        auto_discover=args.auto_discover,
        tls=(args.cert, args.key) if args.cert else None,
        journal_secret=args.journal_key,
        stale_after=args.stale_after,
    )
    server.start()
    print(f"node {args.id} listening on {server.node_address[0]}:{server.node_address[1]} "
          f"peers={peers} data={args.data}")
    if args.metrics_http:
        addr = server.start_metrics_http()
        print(f"metrics http on http://{addr[0]}:{addr[1]}/metrics")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.stop()


if __name__ == "__main__":
    main()