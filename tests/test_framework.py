from conflux.framework import Cluster, DistributedApp, InMemoryTransport


def test_apps_replicate_across_in_memory_transport():
    transport = InMemoryTransport()
    cluster = Cluster({"a", "b"}, transport=transport)

    cluster.node("a").register_set("profile.name", "alice")
    cluster.gossip_round()

    assert cluster.read("a", "profile.name") == "alice"
    assert cluster.read("b", "profile.name") == "alice"


def test_cluster_can_run_multi_round_gossip():
    transport = InMemoryTransport()
    cluster = Cluster({"a", "b", "c"}, transport=transport)

    cluster.node("a").counter_inc("orders", 5)
    cluster.node("b").counter_inc("orders", 3)
    cluster.gossip_round()
    cluster.gossip_round()

    assert cluster.read("a", "orders") == 8
    assert cluster.read("b", "orders") == 8
    assert cluster.read("c", "orders") == 8


def test_distributed_app_supports_messages_and_event_handlers():
    transport = InMemoryTransport()
    app = DistributedApp("node-1", transport=transport, peers={"node-2"})

    app.emit("set", "status", {"value": "online"})
    app.broadcast()
    assert len(transport.pending("node-2")) == 1

    app2 = DistributedApp("node-2", transport=transport, peers={"node-1"})
    app2.consume_pending()
    assert app2.read("status") == "online"
