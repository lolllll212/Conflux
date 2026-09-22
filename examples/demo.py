import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conflux import Cluster, InMemoryTransport, state_hash, replay
from conflux.protocol import journal_from_json, journal_to_json


def main():
    crew = {"agent-alpha", "agent-beta", "agent-gamma"}
    cluster = Cluster(crew, transport=InMemoryTransport())

    cluster.node("agent-alpha").counter_inc("bookings.seats", 2)
    cluster.node("agent-beta").counter_inc("bookings.seats", 1)
    cluster.node("agent-beta").set_add("waitlist", "jane", tag="wl-jane")
    cluster.node("agent-gamma").register_set("flight.eta", "14:05")
    cluster.node("agent-gamma").register_set_weighted(
        "flight.weather-delay", "likely", weight=0.6
    )

    cluster.gossip_round()
    cluster.gossip_round()

    for node_id in sorted(crew):
        print(f"[{node_id}] seats={cluster.read(node_id, 'bookings.seats')} "
              f"eta={cluster.read(node_id, 'flight.eta')} "
              f"waitlist={sorted(cluster.read(node_id, 'waitlist'))}")

    lod = cluster.node("agent-alpha").replica
    actions = journal_from_json(journal_to_json(lod))
    print("Deterministic replay hash matches:",
          state_hash(replay(actions).state) == state_hash(lod.state))


if __name__ == "__main__":
    main()