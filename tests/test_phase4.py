import time
from datetime import datetime, timedelta
import ipaddress

import pytest

from conflux import Client, Server
from conflux.protocol import replay
from conflux.storage import JournalStore

crypto = pytest.importorskip("cryptography")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


@pytest.fixture(scope="module")
def tls_files(tmp_path_factory):
    base = tmp_path_factory.mktemp("tls")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "conflux-test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(minutes=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certfile = base / "cert.pem"
    keyfile = base / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    return str(certfile), str(keyfile), str(certfile)


def _converge(server, key, value, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.read(key) == value:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("transport", ["tcp", "ws"])
def test_tls_convergence(tls_files, transport):
    certfile, keyfile, ca = tls_files
    a = Server("a", ("127.0.0.1", 0), transport=transport, tls=(certfile, keyfile),
               gossip_interval=0.1).start()
    b = Server("b", ("127.0.0.1", 0), transport=transport, tls=(certfile, keyfile),
               gossip_interval=0.1).start()
    a.add_peer(b.node_address)
    try:
        with Client("127.0.0.1", b.node_address[1], agent_id="c",
                    transport=transport, tls=(certfile, keyfile, ca)) as c:
            c.counter_inc("orders", 4)
        assert _converge(a, "orders", 4)
        assert a.hash() == b.hash()
        assert a.metrics()["members"] >= 1
    finally:
        a.stop()
        b.stop()


def test_partition_reconnect(tls_files):
    a = Server("a", ("127.0.0.1", 0), gossip_interval=0.05,
               tls=tls_files[:2], stale_after=0.6).start()
    b = Server("b", ("127.0.0.1", 0), gossip_interval=0.05,
               tls=tls_files[:2], stale_after=0.6).start()
    c = Server("c", ("127.0.0.1", 0), gossip_interval=0.05, auto_discover=True,
               tls=tls_files[:2], stale_after=0.6).start()
    a.add_peer(b.node_address)
    b.add_peer(c.node_address)
    client = None
    try:
        with Client("127.0.0.1", a.node_address[1], agent_id="c", tls=tls_files[2]) as cl:
            client = cl
            cl.counter_inc("orders", 1)
            assert _converge(c, "orders", 1)

            b.stop()
            deadline = time.time() + 6
            while time.time() < deadline and a.metrics()["links"] > 0:
                time.sleep(0.05)
            assert a.metrics()["links"] == 0

            cl.counter_inc("orders", 2)
            assert _converge(a, "orders", 3)

            b2 = Server("b", b.node_address, gossip_interval=0.05,
                        tls=tls_files[:2], stale_after=0.6).start()
            a.add_peer(b2.node_address)
            assert _converge(b2, "orders", 3)
            assert _converge(c, "orders", 3)
            assert a.hash() == b2.hash()
    finally:
        if client is not None:
            client.close()
        a.stop()
        for s in (b, c):
            try:
                s.stop()
            except (AttributeError, OSError):
                pass


def test_signed_journal_tamper(tmp_path):
    store = JournalStore(str(tmp_path / "signed"), snapshot_threshold=1000,
                         journal_secret="node-secret")
    from conflux import Replica

    r = Replica("writer")
    r._emit("counter_inc", "orders", {"by": 5})
    store.append(r.journal[-1])

    _, actions = store.load()
    assert len(actions) == 1

    path = store.dir / "journal.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"key": "orders"', '"key": "votes"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()


def test_signed_snapshot_tamper(tmp_path):
    store = JournalStore(str(tmp_path / "snap"), snapshot_threshold=1, journal_secret="k")
    from conflux import Replica

    r = Replica("writer")
    r._emit("counter_inc", "orders", {"by": 1})
    store.append(r.journal[-1])
    store.rotate(r.state._map)

    snap = store.dir / "snapshot.json"
    data = snap.read_text(encoding="utf-8")
    tampered = data.replace('"orders"', '"votes"')
    assert tampered != data
    snap.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()


def test_cumulative_compaction(tmp_path):
    store = JournalStore(str(tmp_path / "compact"), snapshot_threshold=1)
    from conflux import Replica

    r = Replica("writer")
    for _ in range(4):
        r._emit("counter_inc", "orders", {"by": 1})
        store.append(r.journal[-1])
        store.rotate(r.state._map)

    assert store.generations() == 1
    root, actions = store.load()
    assert root is not None
    assert actions == []
    assert root.read("orders") == 4


def test_journal_roundtrip_with_secret(tmp_path):
    store = JournalStore(str(tmp_path / "rt"), snapshot_threshold=1000, journal_secret="k")
    from conflux import Replica

    r = Replica("writer")
    r.counter_inc("orders", 2)
    r.register_set("status", "open")
    for a in r.journal:
        store.append(a)
    store.close()

    store2 = JournalStore(str(tmp_path / "rt"), snapshot_threshold=1000, journal_secret="k")
    _, actions = store2.load()
    rebuilt = replay(actions)
    assert rebuilt.read("orders") == 2
    assert rebuilt.read("status") == "open"
    store3 = JournalStore(str(tmp_path / "rt"), snapshot_threshold=1000, journal_secret="other")
    with pytest.raises(ValueError):
        store3.load()