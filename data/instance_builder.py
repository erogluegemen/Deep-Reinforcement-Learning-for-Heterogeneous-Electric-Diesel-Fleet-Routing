"""
Instance builder: groups stops by (depot, route, date) into VRP instances.

Each instance dict contains:
  depot       - {lat, lng}
  nodes       - list of {lat, lng, demand_kg, open_min, close_min, service_time_min}
  vehicle_type, weight_kg, volume_m3, is_electric, range_km
  dist_matrix - (n+1) × (n+1) numpy array in km (row/col 0 = depot)
  coords_norm - (n+1, 2) normalized lat/lng in [0,1] for model input
  n_nodes     - number of customer stops (excluding depot)
  route, depot_name, date
"""

import os
import pickle
import numpy as np
import pandas as pd
import yaml
from data.fleet_parser import load_fleet_registry
from data.stop_parser import load_stop_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT, "configs", "default.yaml")
OUT_DIR = os.path.join(ROOT, "data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "real_instances.pkl")

SERVICE_TIME_MIN = 3

DEPOT_COORDS = {
    "SAW": (40.8986, 29.3669),
    "IGA": (41.2611, 28.7425),
    "CET": (41.0855, 28.9742),
}

# Unknown/fallback vehicle info when route not found in fleet registry
FALLBACK_VEHICLE = {
    "vehicle_type": "LCV",
    "weight_kg": 1500,
    "volume_m3": 13.5,
    "is_electric": False,
    "range_km": None,
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _build_dist_matrix(lats, lons):
    """Build full pairwise Haversine distance matrix for n+1 points (vectorised)."""
    n = len(lats)
    lat_arr = np.array(lats)
    lon_arr = np.array(lons)
    # Broadcast
    lat1 = lat_arr[:, None]
    lon1 = lon_arr[:, None]
    lat2 = lat_arr[None, :]
    lon2 = lon_arr[None, :]
    dist = haversine_km(lat1, lon1, lat2, lon2)
    np.fill_diagonal(dist, 0.0)
    return dist


def _normalize_coords(lats, lons):
    """Min-max normalize coordinates to [0, 1]."""
    lat_arr = np.array(lats, dtype=np.float32)
    lon_arr = np.array(lons, dtype=np.float32)
    lat_min, lat_max = lat_arr.min(), lat_arr.max()
    lon_min, lon_max = lon_arr.min(), lon_arr.max()

    lat_range = lat_max - lat_min if lat_max > lat_min else 1.0
    lon_range = lon_max - lon_min if lon_max > lon_min else 1.0

    lat_norm = (lat_arr - lat_min) / lat_range
    lon_norm = (lon_arr - lon_min) / lon_range
    return np.stack([lat_norm, lon_norm], axis=1)  # (n, 2)  → x=lat_norm, y=lon_norm


def build_instances(min_stops=3, max_stops=50, save=True):
    """
    Build VRP instances from stop data grouped by (depot, route, date).
    Returns list of instance dicts.
    """
    stops = load_stop_data()
    fleet = load_fleet_registry()

    instances = []
    grouped = stops.groupby(["depot", "route", "date"])

    for (depot_name, route, date), group in grouped:
        n = len(group)
        if n < min_stops or n > max_stops:
            continue

        if depot_name not in DEPOT_COORDS:
            continue

        depot_lat, depot_lng = DEPOT_COORDS[depot_name]

        # Vehicle info from fleet registry; fallback if route not registered
        veh = fleet.get(route, FALLBACK_VEHICLE.copy())

        # Build node list (depot first, then customer stops)
        nodes = []
        for _, row in group.iterrows():
            nodes.append({
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "demand_kg": float(row["demand_kg"]),
                "open_min": int(row["open_min"]),
                "close_min": int(row["close_min"]),
                "service_time_min": SERVICE_TIME_MIN,
                "act_time_min": row["act_time_min"],
            })

        # All lats/lons including depot at index 0
        all_lats = [depot_lat] + [nd["lat"] for nd in nodes]
        all_lons = [depot_lng] + [nd["lng"] for nd in nodes]

        dist_matrix = _build_dist_matrix(all_lats, all_lons)
        coords_norm = _normalize_coords(all_lats, all_lons)

        instance = {
            "route": route,
            "depot_name": depot_name,
            "date": date,
            "depot": {"lat": depot_lat, "lng": depot_lng},
            "nodes": nodes,
            "n_nodes": n,
            "vehicle_type": veh["vehicle_type"],
            "weight_kg": veh["weight_kg"],
            "volume_m3": veh.get("volume_m3"),
            "is_electric": veh["is_electric"],
            "range_km": veh["range_km"],
            "dist_matrix": dist_matrix.astype(np.float32),
            "coords_norm": coords_norm.astype(np.float32),
        }
        instances.append(instance)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_FILE, "wb") as f:
            pickle.dump(instances, f)

    return instances


def load_instances():
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "rb") as f:
            return pickle.load(f)
    return build_instances(save=True)


if __name__ == "__main__":
    instances = build_instances(save=True)

    n_stops = [inst["n_nodes"] for inst in instances]
    ev_count = sum(1 for inst in instances if inst["is_electric"])
    depots = {}
    for inst in instances:
        depots[inst["depot_name"]] = depots.get(inst["depot_name"], 0) + 1
    vtypes = {}
    for inst in instances:
        vt = inst["vehicle_type"]
        vtypes[vt] = vtypes.get(vt, 0) + 1

    print(f"Total instances: {len(instances)}")
    print(f"EV instances: {ev_count}")
    print(f"n_stops — min:{min(n_stops)} mean:{sum(n_stops)/len(n_stops):.1f} max:{max(n_stops)}")
    print(f"By depot: {depots}")
    print(f"By vehicle type: {vtypes}")

    # Sample instance sanity check
    ex = instances[0]
    print(f"\nSample instance: route={ex['route']} depot={ex['depot_name']} "
          f"date={ex['date'].date()} n={ex['n_nodes']} "
          f"vtype={ex['vehicle_type']} ev={ex['is_electric']}")
    print(f"  dist_matrix shape: {ex['dist_matrix'].shape}")
    print(f"  coords_norm shape: {ex['coords_norm'].shape}")
    total_demand = sum(nd["demand_kg"] for nd in ex["nodes"])
    print(f"  total demand: {total_demand:.1f} kg / {ex['weight_kg']} kg capacity")
