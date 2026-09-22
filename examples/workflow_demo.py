"""Multi-agent workflow, tasks, events, and watchers end-to-end.

Shows a small replicated crew orchestrating an order-fulfillment workflow:
an orchestrator runs a Workflow (reserve -> pack -> ship with a weighted
skip), TaskStore tracks execution state, EventBus receives action events,
and Watcher notices when the shipment set changes.

Run from the repo root:  python examples/workflow_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import (
    Cluster,
    InMemoryTransport,
    TaskStore,
    Watcher,
    Workflow,
    WorkflowRunner,
    WorkflowStep,
    attach_listeners,
    state_hash,
)


def main():
    cluster = Cluster({"orchestrator", "picker", "packer", "couriercam"}, transport=InMemoryTransport())
    orchestrator = cluster.node("orchestrator").replica

    events = []
    watcher = Watcher(orchestrator, fire_initial=False)
    attach_listeners(orchestrator, watcher,
                     lambda actions, origin: events.extend((a.op, a.key) for a in actions))
    shipments_seen = []
    watcher.watch("wf/shipments", lambda value, key, old: shipments_seen.append(value))

    workflow = Workflow("fulfill", [
        WorkflowStep("reserve", "counter_inc", key="wf/capacity", params={"by": 1}),
        WorkflowStep("pack", "lww_set", key="wf/package", params={"value": {"box": "M", "qty": 2}}),
        WorkflowStep("insurance", "counter_inc", key="wf/capacity", params={"by": 1},
                     weight=0.5),
        WorkflowStep("ship", "orset_add", key="wf/shipments", params={"value": "carrier-17"}),
    ])
    runner = WorkflowRunner(orchestrator, workflow)

    task = runner.run("fulfill-100", "fulfill", inputs={"sku": "A-1", "qty": 2})
    cluster.gossip_round()

    store = TaskStore(cluster.node("couriercam").replica)
    print("task (replicated, read from a different agent):")
    print(" ", task)
    print("  couriercam sees:", store.read("fulfill-100")["status"])
    print("  done tasks:", [t["id"] for t in store.list(status="done")])
    print("  wf/capacity on every node:",
          [cluster.read(a, "wf/capacity") for a in cluster.apps])
    print("  ship events captured on orchestrator:", events[-1] if events else None)
    print("  shipment watcher fired:", shipments_seen)
    print("  all nodes converge:",
          len({state_hash(cluster.node(a).replica.state) for a in cluster.apps}) == 1)


if __name__ == "__main__":
    main()