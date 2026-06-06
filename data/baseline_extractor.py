"""
Baseline extractor: reconstructs DHL's actual route sequence from act_time_min
ordering and computes baseline metrics for each instance.

Output: data/processed/baseline_metrics.csv with columns:
  route, depot, date, n_nodes, vehicle_type, is_electric,
  total_dist_km, tw_violations, capacity_kg, total_demand_kg, capacity_utilization
"""

import os
import csv
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES_FILE = os.path.join(ROOT, "data", "processed", "real_instances.pkl")
OUT_FILE = os.path.join(ROOT, "data", "processed", "baseline_metrics.csv")

FIELDNAMES = [
    "route", "depot", "date", "n_nodes", "vehicle_type", "is_electric",
    "total_dist_km", "tw_violations", "capacity_kg", "total_demand_kg",
    "capacity_utilization",
]


def _route_distance(dist_matrix, sequence):
    """Compute total distance for depot→s0→...→sN→depot (indices into dist_matrix)."""
    total = 0.0
    prev = 0  # depot = index 0
    for idx in sequence:
        total += dist_matrix[prev, idx]
        prev = idx
    total += dist_matrix[prev, 0]  # return to depot
    return total


def _tw_violations(nodes, sequence):
    """Count stops where act_time_min falls outside [open_min, close_min]."""
    violations = 0
    for i in sequence:
        node = nodes[i - 1]  # sequence indices are 1-based (dist_matrix row 0 = depot)
        t = node.get("act_time_min")
        if t is None:
            continue
        if t < node["open_min"] or t > node["close_min"]:
            violations += 1
    return violations


def extract_baselines(save=True):
    with open(INSTANCES_FILE, "rb") as f:
        instances = pickle.load(f)

    rows = []
    for inst in instances:
        nodes = inst["nodes"]
        n = inst["n_nodes"]
        dist_matrix = inst["dist_matrix"]

        # Reconstruct DHL sequence: sort by actual delivery time
        # act_time_min may be None for a few nodes; put those at end
        order = sorted(
            range(n),
            key=lambda i: (nodes[i]["act_time_min"] is None, nodes[i].get("act_time_min", 0))
        )
        # Shift to 1-based indices (dist_matrix row 0 = depot)
        sequence = [i + 1 for i in order]

        total_dist = _route_distance(dist_matrix, sequence)
        tw_viol = _tw_violations(nodes, sequence)
        total_demand = sum(nd["demand_kg"] for nd in nodes)
        cap_util = total_demand / inst["weight_kg"] if inst["weight_kg"] > 0 else 0.0

        rows.append({
            "route": inst["route"],
            "depot": inst["depot_name"],
            "date": inst["date"].date() if hasattr(inst["date"], "date") else inst["date"],
            "n_nodes": n,
            "vehicle_type": inst["vehicle_type"],
            "is_electric": inst["is_electric"],
            "total_dist_km": round(total_dist, 3),
            "tw_violations": tw_viol,
            "capacity_kg": inst["weight_kg"],
            "total_demand_kg": round(total_demand, 3),
            "capacity_utilization": round(cap_util, 4),
        })

    if save:
        os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
        with open(OUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    return rows


if __name__ == "__main__":
    rows = extract_baselines(save=True)

    dists = [r["total_dist_km"] for r in rows]
    tw = [r["tw_violations"] for r in rows]
    utils = [r["capacity_utilization"] for r in rows]

    print(f"Baseline metrics for {len(rows)} instances")
    print(f"Total distance (km): min={min(dists):.1f} mean={sum(dists)/len(dists):.1f} max={max(dists):.1f}")
    print(f"TW violations: mean={sum(tw)/len(tw):.2f} max={max(tw)}")
    print(f"Capacity utilization: mean={sum(utils)/len(utils):.2%}")

    ev_rows = [r for r in rows if r["is_electric"]]
    over_range = [r for r in ev_rows if r["total_dist_km"] > 200]
    print(f"\nEV instances: {len(ev_rows)}")
    print(f"EV routes > 200km (baseline): {len(over_range)}")

    # Per-depot summary
    for depot in ("SAW", "IGA", "CET"):
        subset = [r for r in rows if r["depot"] == depot]
        if not subset:
            continue
        d = [r["total_dist_km"] for r in subset]
        print(f"  {depot}: {len(subset)} instances, mean dist {sum(d)/len(d):.1f} km")
