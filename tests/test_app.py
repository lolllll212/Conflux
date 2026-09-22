from conflux import (
    EventBus,
    Action,
    Replica,
    TaskStore,
    Watcher,
    Workflow,
    WorkflowRunner,
    WorkflowStep,
    attach_listeners,
    converge_all,
    replay,
    state_hash,
)


def test_event_bus_pattern_filters():
    bus = EventBus()
    hits = []

    bus.subscribe(lambda action, origin: hits.append(("any", action)), op="counter_inc")
    bus.subscribe(lambda action, origin: hits.append(("orders", action)), key_prefix="orders.")
    bus.subscribe(lambda action, origin: hits.append(("status", action)), op="lww_set", key="status")

    bus.publish(Action("a1", "x", 1, "counter_inc", "orders.count", {"by": 1}))
    bus.publish(Action("a2", "x", 1, "lww_set", "status", {"value": "open"}))
    bus.publish(Action("a3", "x", 1, "counter_inc", "inventory.q", {"by": 1}))

    assert [tag for tag, _ in hits] == ["any", "orders", "status", "any"]
    assert bus.subscriptions() == 3
    token = bus.subscribe(lambda a, o: None, op="counter_inc")
    bus.unsubscribe(token)
    assert bus.subscriptions() == 3


def test_replica_emits_flow_through_listeners_local_and_absorb():
    bus = EventBus()
    replica = Replica("agent-a")
    attach_listeners(replica, bus)
    seen = []
    bus.subscribe(lambda action, origin: seen.append((origin, action.op, action.key)), op="counter_inc")

    replica.counter_inc("a", 1)
    remote = Replica("agent-b")
    remote.counter_inc("b", 2)
    replica.absorb([remote.full_journal()[0]])

    assert seen == [("local", "counter_inc", "a"), ("absorb", "counter_inc", "b")]


def test_watcher_fires_on_value_change():
    replica = Replica("w")
    watcher = Watcher(replica, fire_initial=False)
    attach_listeners(replica, watcher)
    changes = []
    watcher.watch("orders", lambda value, key, old: changes.append((value, old)))

    replica.counter_inc("orders", 3)
    assert changes[-1] == (3, None)
    replica.counter_inc("orders", 2)
    assert changes[-1] == (5, 3)
    count_before = len(changes)
    replica.set_add("tags", "x", tag="t1")  # different key -> no event for orders
    assert len(changes) == count_before
    watcher.unwatch("orders")
    replica.counter_inc("orders", 1)
    assert len(changes) == count_before


def test_watcher_initial_fire():
    replica = Replica("w")
    replica.counter_inc("orders", 4)
    watcher = Watcher(replica, fire_initial=True)
    initial = []
    watcher.watch("orders", lambda value, key, old: initial.append((value, old)))
    assert initial == [(4, None)]


def test_task_store_lifecycle_and_replication():
    a = Replica("orchestrator")
    store = TaskStore(a)
    store.create("task-1", "fulfill", agent="orchestrator", inputs={"sku": "A-1"})
    store.start("task-1")
    store.advance("task-1", "pick")
    store.complete("task-1", {"packed": True})

    record = store.read("task-1")
    assert record["status"] == "done"
    assert record["id"] == "task-1"
    assert record["result"] == {"packed": True}
    assert store.list(status="done") == [record]
    assert store.list(status="failed") == []

    b = Replica("other")
    b.absorb(a.full_journal())
    assert b.read("tasks/task-1")["status"] == "done"
    assert state_hash(a.state) == state_hash(b.state)


def test_task_store_fail_and_status_filter():
    a = Replica("orchestrator")
    store = TaskStore(a)
    store.create("t", "job")
    store.fail("t", "timeout")
    assert store.read("t")["status"] == "failed"
    assert store.read("t")["error"] == "timeout"
    assert store.list(status="failed") == [store.read("t")]


def test_workflow_runner_deterministic():
    workflow = Workflow("fulfill", [
        WorkflowStep("reserve", "counter_inc", key="wf/capacity", params={"by": 1}),
        WorkflowStep("pack", "lww_set", key="wf/package", params={"value": {"box": "M"}}),
        WorkflowStep("ship", "orset_add", key="wf/shipments", params={"value": "carrier"}),
    ])
    runner_a = WorkflowRunner(Replica("orch-1"), workflow)
    runner_b = WorkflowRunner(Replica("orch-2"), workflow)

    task_a = runner_a.run("fulfill-1", "fulfill", inputs={"order": 42})
    task_b = runner_b.run("fulfill-1", "fulfill", inputs={"order": 42})

    assert task_a["status"] == "done"
    assert task_b["status"] == "done"
    assert task_a["result"] == task_b["result"]
    assert task_a["step"] is None  # cleared on completion
    assert task_a["result"]["reserve"] == 1
    assert task_a["result"]["pack"] == {"box": "M"}
    assert task_a["result"]["ship"] == ["carrier"]

    rebuilt = Replica("replay")
    rebuilt.absorb(runner_a.store.replica.full_journal())
    assert rebuilt.read("wf/capacity") == 1
    assert rebuilt.read("wf/package") == {"box": "M"}
    assert rebuilt.read("tasks/fulfill-1")["status"] == "done"


def test_workflow_weighted_step_deterministic_across_nodes():
    workflow = Workflow("risk", [
        WorkflowStep("always", "counter_inc", key="wf/rate", params={"by": 1}),
        WorkflowStep("sometimes", "counter_inc", key="wf/rate", params={"by": 2}, weight=0.0),
    ])
    r1, r2 = Replica("a"), Replica("b")
    t1 = WorkflowRunner(r1, workflow).run("risk-1", "risk")
    t2 = WorkflowRunner(r2, workflow).run("risk-1", "risk")
    assert r1.read("wf/rate") == r2.read("wf/rate")
    assert r1.read("wf/rate") == 1
    assert t1["result"] == t2["result"]


def test_workflows_converge_across_cluster():
    workflow = Workflow("batch", [
        WorkflowStep("count", "counter_inc", key="wf/n", params={"by": 1}),
        WorkflowStep("flag", "lww_set", key="wf/done", params={"value": True}),
    ])
    nodes = [Replica(f"n{i}") for i in range(4)]
    for i, node in enumerate(nodes):
        WorkflowRunner(node, workflow).run(f"batch-{i+1}", "batch", inputs={"node": i})
    converge_all(nodes)
    assert all(node.read("wf/n") == 4 for node in nodes)
    assert all(state_hash(node.state) == state_hash(nodes[0].state) for node in nodes)