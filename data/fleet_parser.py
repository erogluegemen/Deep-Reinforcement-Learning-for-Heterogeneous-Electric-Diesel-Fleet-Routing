"""
Fleet parser: loads DHL_Araclar.xlsx and builds a route-code → vehicle-info dict.

PUD Fac → depot mapping (confirmed by route prefix cross-reference):
  GTW → IGA  (IS* routes)
  EAT → SAW  (EA* routes)
  CET → CET  (CE* routes)

EV detection: Seri Ad contains 'e-Transit' (covers both 350L BEV and 350E BEV).
"""

import os
import pickle
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEET_FILE = os.path.join(ROOT, "data", "raw", "DHL_Araclar.xlsx")
CONFIG_FILE = os.path.join(ROOT, "configs", "default.yaml")
OUT_DIR = os.path.join(ROOT, "data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "fleet_registry.pkl")

DEPOT_SHEETS = {
    "SAW OPS": "SAW",
    "IGA OPS": "IGA",
    "CET OPS": "CET",
}


def _load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def _classify_vehicle(row, config):
    is_ev = "e-Transit" in str(row.get("Seri Ad", ""))
    v_type_raw = str(row.get("Araç Tipi", "")).strip()

    if is_ev:
        v_type = "LCV_BEV"
    elif v_type_raw in ("SCV", "MCV", "LCV"):
        v_type = v_type_raw
    else:
        v_type = "LCV"  # fallback

    caps = config["vehicle_capacities"][v_type]
    return {
        "vehicle_type": v_type,
        "weight_kg": caps["weight_kg"],
        "volume_m3": caps["volume_m3"],
        "is_electric": caps["is_electric"],
        "range_km": caps["range_km"],
    }


def build_fleet_registry(save=True):
    """Returns dict mapping rut_code -> vehicle info + depot."""
    config = _load_config()

    frames = []
    for sheet, depot in DEPOT_SHEETS.items():
        df = pd.read_excel(FLEET_FILE, sheet_name=sheet, engine="openpyxl")
        df["depot"] = depot
        frames.append(df)

    all_vehicles = pd.concat(frames, ignore_index=True)

    registry = {}
    for _, row in all_vehicles.iterrows():
        rut = str(row.get("RUT Code", "")).strip()
        if not rut or rut == "nan":
            continue
        info = _classify_vehicle(row, config)
        info["depot"] = row["depot"]
        info["plate"] = str(row.get("Plaka", "")).strip()
        registry[rut] = info

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_FILE, "wb") as f:
            pickle.dump(registry, f)

    return registry


def load_fleet_registry():
    """Load pre-built registry from disk, or build and cache it."""
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "rb") as f:
            return pickle.load(f)
    return build_fleet_registry(save=True)


if __name__ == "__main__":
    reg = build_fleet_registry(save=True)

    ev_routes = [k for k, v in reg.items() if v["is_electric"]]
    print(f"Total routes in registry: {len(reg)}")
    print(f"EV routes: {len(ev_routes)} → {sorted(ev_routes)}")

    for depot in ("SAW", "IGA", "CET"):
        subset = {k: v for k, v in reg.items() if v["depot"] == depot}
        types = {}
        for v in subset.values():
            types[v["vehicle_type"]] = types.get(v["vehicle_type"], 0) + 1
        print(f"  {depot}: {len(subset)} routes — {types}")
