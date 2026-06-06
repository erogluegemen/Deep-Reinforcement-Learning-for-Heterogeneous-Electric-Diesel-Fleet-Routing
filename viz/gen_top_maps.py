"""
Generate POMO vs DHL comparison maps for the top-performing real instances.
Targets: CET CECD (n=30, -54%), IGA ISCB (n=28 EV, -50%), SAW EADB (n=28, -48%)
"""

import os
import pickle
import sys

import folium
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pomo.inference import solve
from pomo.model import POMOModel
from viz.route_map import _dhl_route_coords, _haversine, DEPOT_COLORS

# (depot, route, date) — exact instances from top-performer analysis
TARGETS = [
    ("CET", "CECD", "2026-01-07"),   # n=30, DHL=19.7 km, POMO=-53.9%
    ("IGA", "ISCB", "2026-01-02"),   # n=28, DHL=54.3 km, POMO=-50.0%, EV
    ("SAW", "EADB", "2026-01-06"),   # n=28, DHL=72.7 km, POMO=-47.7%
]

CKPT = os.path.join(ROOT, "colab_results", "best_cvrp.pt")
OUT_DIR = os.path.join(ROOT, "viz", "maps")


def _dhl_dist_km(inst):
    depot = inst["depot"]
    sorted_nodes = sorted(inst["nodes"], key=lambda nd: nd["act_time_min"])
    coords = [(depot["lat"], depot["lng"])] + [(nd["lat"], nd["lng"]) for nd in sorted_nodes] + [(depot["lat"], depot["lng"])]
    return sum(_haversine(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1))


def make_comparison_map(inst, pomo_tour, dhl_km, pomo_km, out_path):
    depot_lat = inst["depot"]["lat"]
    depot_lng = inst["depot"]["lng"]
    depot_name = inst["depot_name"]
    color = DEPOT_COLORS.get(depot_name, "#8e44ad")
    reduction = 100 * (dhl_km - pomo_km) / dhl_km

    m = folium.Map(location=[depot_lat, depot_lng], zoom_start=13, tiles="CartoDB positron")

    folium.Marker(
        [depot_lat, depot_lng],
        tooltip=f"Depot ({depot_name})",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    for i, nd in enumerate(inst["nodes"]):
        tw = f"{nd['open_min']//60:02d}:{nd['open_min']%60:02d}–{nd['close_min']//60:02d}:{nd['close_min']%60:02d}"
        folium.CircleMarker(
            [nd["lat"], nd["lng"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Stop {i+1} | {nd['demand_kg']:.1f} kg | TW: {tw}",
        ).add_to(m)

    dhl_coords, _ = _dhl_route_coords(inst)
    folium.PolyLine(
        dhl_coords,
        color="#e67e22",
        weight=3,
        opacity=0.7,
        dash_array="8 4",
        tooltip=f"DHL baseline: {dhl_km:.1f} km",
    ).add_to(m)

    depot = inst["depot"]
    nodes = inst["nodes"]
    pomo_coords = [(depot["lat"], depot["lng"])]
    for idx in pomo_tour:
        if idx == 0:
            pomo_coords.append((depot["lat"], depot["lng"]))
        else:
            nd = nodes[idx - 1]
            pomo_coords.append((nd["lat"], nd["lng"]))
    if pomo_coords[-1] != (depot["lat"], depot["lng"]):
        pomo_coords.append((depot["lat"], depot["lng"]))

    folium.PolyLine(
        pomo_coords,
        color="#2980b9",
        weight=3,
        opacity=0.9,
        tooltip=f"POMO: {pomo_km:.1f} km",
    ).add_to(m)

    route = inst["route"]
    date = str(inst["date"])[:10]
    vtype = inst["vehicle_type"]
    ev_tag = " ⚡ EV" if inst["is_electric"] else ""
    n = inst["n_nodes"]

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);
                font-family:Arial;font-size:13px;line-height:1.8">
      <b>{route}</b> · {depot_name} · {date}<br>
      {vtype}{ev_tag} · {n} stops<br>
      <span style="color:#e67e22">━ ━</span> DHL: {dhl_km:.1f} km<br>
      <span style="color:#2980b9">───</span> POMO: {pomo_km:.1f} km<br>
      <b style="color:#27ae60">↓ {reduction:.1f}% reduction</b>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    return reduction


def main():
    with open(os.path.join(ROOT, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    input_dim = ckpt["model_state"]["encoder.input_proj.weight"].shape[1]
    model = POMOModel(config).to("cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {CKPT}  (input_dim={input_dim})")

    with open(os.path.join(ROOT, "data", "processed", "real_instances.pkl"), "rb") as f:
        instances = pickle.load(f)

    # Find instances by exact (depot, route, date)
    target_map = {}
    for inst in instances:
        key = (inst["depot_name"], inst["route"], str(inst["date"])[:10])
        if key in {t for t in TARGETS}:
            target_map[key] = inst

    print(f"\nFound {len(target_map)}/{len(TARGETS)} target instances\n")

    for depot, route, date in TARGETS:
        key = (depot, route, date)
        if key not in target_map:
            print(f"  MISSING: {depot} {route} {date}")
            continue

        inst = target_map[key]
        print(f"  Processing {depot} {route} {date}  n={inst['n_nodes']}  {inst['vehicle_type']} ev={inst['is_electric']}")

        with torch.no_grad():
            pomo_tour, pomo_km = solve(model, inst, use_augmentation=True,
                                       force_cvrp=True, device="cpu")

        dhl_km = _dhl_dist_km(inst)

        fname = f"{depot}_{route}_{date}_pomo_vs_dhl.html"
        out_path = os.path.join(OUT_DIR, fname)
        reduction = make_comparison_map(inst, pomo_tour, dhl_km, pomo_km, out_path)
        print(f"    DHL={dhl_km:.1f} km  POMO={pomo_km:.1f} km  reduction={reduction:.1f}%")
        print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()
