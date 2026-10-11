import argparse
import json
import logging
import queue
import socket
import ssl
import threading
import time

from .actions import Action
from .app import EventBus, Watcher, attach_listeners
from .net import MAX_FRAME, ProtocolError, normalize_tls, recv_msg, send_msg, to_wire
from .protocol import state_hash
from .state import Replica
from .storage import JournalStore
from .validate import ValidationError, validate_action, validate_envelope
from .websocket import WebSocketServer, connect_websocket


class Server:
    def __init__(self, node_id, address, peers=(), data_dir=None, gossip_interval=1.0,
                 snapshot_threshold=1000, registry=None, transport="tcp", batch_size=0,
                 auto_discover=False, tls=None, journal_secret=None, stale_after=None,
                 max_frame=MAX_FRAME, max_connections=64, frame_batch_bytes=1 << 20,
                 max_actions_per_frame=100_000):
        """Three knobs exist for the world outside a test: ``max_frame`` is the
        largest frame a peer may hand us, ``max_connections`` how many inbound links
        one process will hold open at all (each costs a thread plus receive buffer),
        and ``frame_batch_bytes`` the budget a journal chunk is built to fit inside
        so a node with a long journal can still be joined. ``max_actions_per_frame``
        caps how much work one inbound frame may ask for: a frame inside the byte
        cap can still hold a million tiny actions, and "valid JSON" is not the same
        statement as "affordable to absorb"."""
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
        self.max_frame = int(max_frame)
        self.max_connections = max(1, int(max_connections))
        self.frame_batch_bytes = int(frame_batch_bytes)
        self.max_actions_per_frame = max(1, int(max_actions_per_frame))
        self._connection_slots = threading.BoundedSemaphore(self.max_connections)
        self._metrics = {"absorbed": 0, "submitted": 0, "malformed_frames": 0,
                         "oversize_frames": 0, "rejected_connections": 0, "link_drops": 0,
                         "handler_errors": 0, "gossip_errors": 0, "resyncs": 0}
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

        self.log = logging.getLogger(f"conflux.server.{node_id}")
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
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
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
            addr = self.node_address
            try:
                self._socket.close()
            except OSError:
                pass
            if addr and self.transport == "tcp":
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as wake_sock:
                        wake_sock.settimeout(0.1)
                        wake_sock.connect(addr)
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
                "malformed_frames": self._metrics["malformed_frames"],
                "oversize_frames": self._metrics["oversize_frames"],
                "rejected_connections": self._metrics["rejected_connections"],
                "link_drops": self._metrics["link_drops"],
                "handler_errors": self._metrics["handler_errors"],
                "gossip_errors": self._metrics["gossip_errors"],
                "resyncs": self._metrics["resyncs"],
                "max_connections": self.max_connections,
                "max_actions_per_frame": self.max_actions_per_frame,
                "link_cursors": {str(k): int(v.get("cursor", 0))
                                 for k, v in list(self._links.items())[:8]},
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
            "max_connections": ("Inbound links this node holds open at once.", m["max_connections"]),
            "max_actions_per_frame": ("Actions one inbound frame may ask this node to absorb.",
                                      m["max_actions_per_frame"]),
            "links_dropped_total": ("Links dropped for idleness or a failed push.",
                                    m["link_drops"]),
            "malformed_frames_total": ("Frames refused before they could touch state.",
                                       m["malformed_frames"]),
            "rejected_connections_total": ("Inbound connections refused at the limit.",
                                          m["rejected_connections"]),
            "gossip_errors_total": ("Gossip rounds that raised and were survived.",
                                    m["gossip_errors"]),
            "handler_errors_total": ("Connection handlers that raised.", m["handler_errors"]),
            "resyncs_total": ("Links asked to resend their journal after a heartbeat showed "
                              "differing state roots.", m["resyncs"]),
            "oversize_frames_total": ("Frames this node refused to emit because a peer "
                                      "would be entitled to reject them.", m["oversize_frames"]),
        }
        lines = []
        counters.update({k: v for k, v in gauges.items() if k.endswith("_total")})
        gauges = {k: v for k, v in gauges.items() if not k.endswith("_total")}
        for name, (help_text, value) in counters.items():
            lines += [f"# HELP {namespace}_{name} {help_text}",
                      f"# TYPE {namespace}_{name} counter",
                      f"{namespace}_{name}{{{labels}}} {value}"]
        for name, (help_text, value) in gauges.items():
            lines += [f"# HELP {namespace}_{name} {help_text}",
                      f"# TYPE {namespace}_{name} gauge",
                      f"{namespace}_{name}{{{labels}}} {value}"]
        cursors = m["link_cursors"]
        if cursors:
            # one sample per link, one header block: repeating HELP per series is
            # not legal exposition format and a scraper drops the type
            lines += [f"# HELP {namespace}_link_cursor Actions of the journal delivered to a link.",
                      f"# TYPE {namespace}_link_cursor gauge"]
            lines += [f'{namespace}_link_cursor{{node_id="{self.node_id}",link="{key}"}} {value}'
                      for key, value in cursors.items()]
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

    def _note(self, metric, message, *args):
        """Count and log an anomaly once, instead of letting it vanish with a thread.

        Every one of these is a thing a node can get wrong quietly: a peer that
        sends garbage, a link dropped for idleness, a connection refused because the
        node is at its limit. Without a counter each looks identical to "fine".
        """
        with self.lock:
            self._metrics[metric] = self._metrics.get(metric, 0) + 1
        self.log.warning(message, *args)

    def _accept_loop(self):
        # One accept loop per node: an unexpected error here is the difference
        # between a cluster that heals and one that quietly stops being joinable.
        while not self._stopped.is_set():
            try:
                conn, _ = self._socket.accept()
            except OSError:
                break
            if self._stopped.is_set():
                try:
                    conn.close()
                except OSError:
                    pass
                break
            if not self._connection_slots.acquire(blocking=False):
                self._note("rejected_connections",
                           "at %d inbound links; refusing another connection "
                           "(raise max_connections if that is too low)", self.max_connections)
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True,
                             name="conflux-link").start()

    def _handle_conn(self, conn):
        conn.settimeout(10)
        registered = []
        write_lock = threading.Lock()

        def send(obj):
            # a link's outbound queue is drained by its own thread while this thread
            # answers hello_ack, journal pushes and sync requests: two sendall calls
            # on one socket interleave as soon as a frame needs more than one write,
            # and the peer then reads a length prefix belonging to the other frame
            with write_lock:
                send_msg(conn, obj, max_frame=self.max_frame)

        def recv():
            return recv_msg(conn, max_frame=self.max_frame)

        try:
            self._conn_loop(send, recv, conn.close, registered)
        except ProtocolError as exc:
            # a peer that sends garbage is disconnected, not argued with - but the
            # reason is counted, so a noisy neighbour is distinguishable from a bug
            self._note("malformed_frames", "dropping a link after a bad frame: %s", exc)
        except (ConnectionError, OSError):
            pass
        except Exception as exc:
            # anything else is our bug: report it and still close cleanly, because
            # a traceback printed from a dying thread is not an audit trail
            self._note("handler_errors", "connection handler failed: %s: %s",
                       type(exc).__name__, exc)
        finally:
            for key in registered:
                # a link registered before its reader died must not linger: with no
                # address for the staleness reaper to key on, it kept a sender
                # thread alive for the life of the process
                self._drop_link(key)
            try:
                conn.close()
            except OSError:
                pass
            self._connection_slots.release()

    def _handle_ws_conn(self, ws):
        registered = []
        try:
            self._conn_loop(ws.send_json, lambda: ws.recv_json(), ws.close, registered)
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            for key in registered:
                self._drop_link(key)
            try:
                ws.close()
            except (ConnectionError, OSError):
                pass

    def _conn_loop(self, send, recv, close, registered=None):
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
            if registered is not None:
                registered.append(key)
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
            except ProtocolError as exc:
                # a peer that hands us an unreadable frame is a different incident
                # from a peer that hangs up, and only one of them wants a human
                self._note("malformed_frames", "peer %s sent an unusable frame: %s",
                           self._links.get(key, {}).get("peer_id"), exc)
                break
            except (ConnectionError, OSError, ValueError):
                break
            if msg is None:
                break
            self._touch_link(key)
            try:
                self._handle_message(msg, send, key)
            except ProtocolError as exc:
                self._note("malformed_frames", "rejecting a message from a peer: %s", exc)
                break
            except (ConnectionError, OSError, ValueError):
                break
        if key is not None:
            self._drop_link(key)

    def _consider_resync(self, msg, key):
        """A heartbeat whose hash is not ours: ask that peer to resend to us.

        Rate-limited by one outstanding request per link (`resyncing`), because a
        pair that is genuinely converging by other means must not stampede each
        other with full journal resends every round.
        """
        if key is None:
            return  # an agent link has no journal to resync
        peer_hash = msg.get("hash")
        if not isinstance(peer_hash, str) or not peer_hash:
            return
        with self.lock:
            info = self._links.get(key)
            if info is None or info.get("resyncing"):
                return
            mine = state_hash(self.replica.state)
            if peer_hash == mine:
                return
            info["resyncing"] = time.time()
        self._note("resyncs", "peer %s reports a different state root; asking it to resend",
                   msg.get("node_id") or info.get("peer_id"))
        self._enqueue(key, {"type": "resync", "node_id": self.node_id})

    def _frame_actions(self, msg):
        """The action list of a frame, or ``None`` if the frame asks for too much.

        A non-list is a malformed frame, so it raises; an over-long list is a
        *budget*, so the caller decides whether to answer (agent) or hang up (peer).
        """
        raw = msg.get("actions", [])
        if not isinstance(raw, list):
            raise ProtocolError(f"'actions' must be a list, not {type(raw).__name__}")
        if len(raw) > self.max_actions_per_frame:
            self._note("oversize_frames", "a frame carried %d actions (cap %d)",
                       len(raw), self.max_actions_per_frame)
            return None
        return raw

    def _handle_message(self, msg, send, source):
        if msg.get("type") == "submit":
            incoming = self._frame_actions(msg)
            if incoming is None:
                send({"type": "error", "reason": f"submit carries more than "
                       f"{self.max_actions_per_frame} actions; split it"})
                return
            try:
                if self.registry is not None:
                    actions = [validate_envelope(env, self.registry) for env in incoming]
                else:
                    actions = [Action.import_action(d) for d in incoming]
                    for action in actions:
                        validate_action(action)
            except ValidationError as exc:
                send({"type": "error", "reason": str(exc)})
                return
            except (TypeError, KeyError, AttributeError) as exc:
                # an agent that posts a half-built action is a client bug, not a
                # peer to disconnect: answer with the reason and keep serving them
                self._note("malformed_frames", "rejected a submit: %s: %s",
                           type(exc).__name__, exc)
                send({"type": "error", "reason": f"malformed action: {exc}"})
                return
            self._absorb(actions)
            with self.lock:
                self._metrics["submitted"] += len(actions)
            send({"type": "ack", "node_id": self.node_id, "accepted": len(actions)})
        elif msg.get("type") == "read":
            key = msg.get("key")
            if not isinstance(key, str) or not key:
                self._note("malformed_frames", "rejected a read: 'key' must be a non-empty string")
                send({"type": "error", "reason": "read needs a string 'key'"})
                return
            send({"type": "read_resp", "key": key, "value": to_wire(self.read(key))})
        elif msg.get("type") == "read_all":
            send({"type": "read_resp", "value": to_wire(self.view())})
        elif msg.get("type") == "hash":
            send({"type": "hash_resp", "hash": self.hash()})
        elif msg.get("type") == "sync":
            self._send_journal(send)
        elif msg.get("type") == "journal":
            incoming = self._frame_actions(msg)
            if incoming is None:
                # a peer that asks for more than one frame's worth of work is not
                # having a bad day, it is either misconfigured or hostile: say so,
                # count it, and hang up rather than grinding through the list
                raise ProtocolError(f"peer sent more than {self.max_actions_per_frame} "
                                    f"actions in one frame; refusing")
            try:
                actions = [validate_action(Action.import_action(d)) for d in incoming]
            except (TypeError, ValueError, KeyError, ValidationError) as exc:
                raise ProtocolError(f"malformed peer journal: {exc}") from exc
            self._absorb(actions)
        elif msg.get("type") == "heartbeat":
            self._consider_resync(msg, source)
        elif msg.get("type") == "resync":
            # the peer's state root differs from ours and it asked us to speak: send
            # everything, from index 0. At-least-once plus action_id dedup means a
            # needless resend costs bandwidth and nothing else
            if source is not None:
                info = self._links.get(source)
                if info is not None:
                    info["cursor"] = 0
                    self._push_link(source, info, start=0)
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

    def _export_range(self, start=0):
        """Exported journal actions from ``start`` on, under the node lock."""
        with self.lock:
            return [action.export() for action in self.replica.journal[start:]]

    def _frame_budget(self):
        """Bytes one gossip frame may take: the batch budget, and never the cap."""
        budget = int(self.frame_batch_bytes or 0) or (1 << 20)
        if self.max_frame:
            budget = min(budget, int(self.max_frame))
        return max(4096, budget)

    @staticmethod
    def chunk_exported(actions, *, max_actions=0, max_bytes=1 << 20):
        """Group exported actions into frames that fit a count *and* a byte budget.

        A frame over the peer's cap is not merely late, it is refused: the link
        closes, the peer redials, and the same oversize frame is refused again. That
        is how a node whose journal outgrew one frame becomes permanently
        unjoinable, so the sender chunks before it ever builds the frame.
        """
        chunk, size = [], 0
        for export in actions:
            # measured per action, so one enormous action still ships (it has to)
            # and the running sum stays an upper bound on the frame
            weight = len(json.dumps(export)) + 1
            if chunk and (size + weight > max_bytes
                          or (max_actions and len(chunk) >= max_actions)):
                yield chunk, size
                chunk, size = [], 0
            chunk.append(export)
            size += weight
        if chunk:
            yield chunk, size

    def _send_journal(self, send):
        # chunked, never one frame per sync: this is the path a joining agent uses
        # to catch up, and a single frame of the whole journal is what an oversized
        # peer rejects
        for chunk, size in self.chunk_exported(self._export_range(), max_actions=self.batch_size,
                                                max_bytes=self._frame_budget()):
            send({"type": "journal", "actions": chunk})

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
            # ``cursor`` is an index into the append-only journal, not a set of
            # action ids: a per-peer set of everything ever gossiped grows for the
            # life of the process (one entry per action per peer) and turns every
            # gossip round into a full journal scan, which is exactly how a node
            # that ran for a month starts lagging.
            self._links[key] = {"send": send, "recv": recv, "close": close or (lambda: None),
                                "cursor": 0, "acked": 0, "address": address,
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
            except (ConnectionError, OSError, ValueError) as exc:
                self.log.info("link %s closed after a send failure: %s", info.get("peer_id"), exc)
                self._drop_link(key)
                return
            # only now is the peer's cursor allowed to move: what is queued but
            # unwritten stays owed, and the next round retries from here
            info["acked"] = max(int(info.get("acked", 0)), int(msg.get("through", 0) or 0))

    def _enqueue(self, key, payload):
        """Hand one frame to a link's sender. False means "not now" (backpressure).

        The per-link queue is bounded on purpose: a peer that cannot drain its own
        inbox must not turn this process into its buffer. The caller leaves the
        cursor where it is, so nothing is marked delivered and the next round
        retries - delivery is at-least-once, and absorb dedupes by action id.
        """
        info = self._links.get(key)
        if info is None:
            return False
        try:
            info["outgoing"].put_nowait(payload)
        except queue.Full:
            return False
        return True

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
        """Catch a brand-new link up, in frames the peer will accept."""
        info = self._links.get(key)
        if info is None:
            return
        self._push_link(key, info, start=0)

    def _push_link(self, key, info, start):
        """Deliver journal actions from ``start`` on, advancing the cursor as we go."""
        journal_len = len(self.replica.journal)
        if start > journal_len:
            start = 0  # the journal shrank under us (a future compaction): restart
        delivered = start
        for chunk, size in self.chunk_exported(self._export_range(start),
                                                max_actions=self.batch_size,
                                                max_bytes=self._frame_budget()):
            if self.max_frame and size > self.max_frame:
                # a single action bigger than the cap: nothing we can chunk about
                self._note("oversize_frames",
                           "journal frame of %d bytes exceeds the %d byte cap; the peer "
                           "will refuse it - shrink the action or raise max_frame",
                           size, self.max_frame)
            if not self._enqueue(key, {"type": "journal", "actions": chunk,
                                       "through": delivered + len(chunk)}):
                break  # the link is behind; retry from here on the next round
            delivered += len(chunk)
        info["cursor"] = delivered
        if delivered >= journal_len:
            info.pop("resyncing", None)
        return delivered - start

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
        failures = 0
        while not self._stopped.is_set():
            try:
                self._gossip_round()
                failures = 0
            except Exception as exc:
                # The gossip loop is the only thing that pushes state to peers. It
                # may not die: one unusable member or address entry used to stop
                # this node converging with anyone, silently, until it restarted.
                failures += 1
                self._note("gossip_errors", "gossip round failed (%d in a row): %s: %s",
                           failures, type(exc).__name__, exc)
                self._stopped.wait(min(self.gossip_interval * 2 ** min(failures, 5), 30.0))
                continue
            self._stopped.wait(self.gossip_interval)

    def _gossip_round(self):
        now = time.time()
        for key, info in list(self._links.items()):
            idle = now - info.get("last_seen", now)
            if idle > self.stale_after:
                address = info.get("address")
                if address is not None:
                    with self.lock:
                        self._peer_failures[address] = self._peer_failures.get(address, 0) + 1
                self._note("link_drops", "dropping a link idle for %.1fs (peer=%s)",
                           idle, info.get("peer_id"))
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
        self._heartbeat_links(now)
        self._gossip_tick += 1

    def _heartbeat_links(self, now):
        """Tell every link that has nothing to send what our state root is.

        Two jobs, one small frame. A link that never carries traffic used to be
        reaped as idle and re-dialled, so a converged pair churned - threads,
        registrations and log lines - forever; and a node that silently missed
        gossip (a dropped frame under backpressure, a journal rotation the peer
        raced) had no way to notice. The hash is the node's own canonical
        ``state_hash``, so a mismatch is a *fact* about divergence, not a guess
        about timing.
        """
        # self.hash() rather than a bare state_hash(): the writer thread mutates the
        # map under the lock, and hashing the live OrMap from here could see it change
        # size mid-iteration - an exception in the gossip loop, on the read path's cost
        mine = self.hash()
        journal_len = len(self.replica.journal)
        for key, info in list(self._links.items()):
            if info.get("cursor", 0) < journal_len:
                continue  # it has data coming; a heartbeat would only add noise
            self._enqueue(key, {"type": "heartbeat", "node_id": self.node_id,
                                "hash": mine, "journal": journal_len})

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
                self._push_link(key, info, int(info.get("cursor", 0)))
            except (KeyError, AttributeError):
                continue
            except Exception as exc:
                # a link that cannot be pushed to is dropped, not left half-sent:
                # the cursor would otherwise advance over actions nobody received
                self._note("link_drops", "dropping link %s after a push failure: %s: %s",
                           info.get("peer_id"), type(exc).__name__, exc)
                self._drop_link(key)


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
    parser.add_argument("--frame-bytes", type=int, default=1 << 20,
                        help="byte budget per gossip frame; a journal longer than one "
                             "frame is chunked, never sent oversize and refused")
    parser.add_argument("--max-frame", type=int, default=64 * 1024 * 1024,
                        help="largest frame accepted from, or sent to, a peer")
    parser.add_argument("--max-actions-per-frame", type=int, default=100_000,
                        help="refuse a frame that asks this node to absorb more actions than this")
    parser.add_argument("--max-connections", type=int, default=64,
                        help="inbound links held open at once; beyond it, accept and close")
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
        max_frame=args.max_frame,
        max_connections=args.max_connections,
        frame_batch_bytes=args.frame_bytes,
        max_actions_per_frame=args.max_actions_per_frame,
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