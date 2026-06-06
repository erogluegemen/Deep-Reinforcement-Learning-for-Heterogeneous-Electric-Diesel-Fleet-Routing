"""
Route map visualization: DHL baseline vs POMO route on a folium map.

Usage:
  python viz/route_map.py                          # first 5 instances, auto depot colours
  python viz/route_map.py --n 10                   # first 10 instances
  python viz/route_map.py --route CEAA --depot CET # specific route
  python viz/route_map.py --ev_only                # EV instances only
"""

import argparse
import os
import pickle

import folium
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEPOT_COLORS = {"SAW": "#e74c3c", "IGA": "#2980b9", "CET": "#27ae60"}
VEHICLE_ICONS = {"SCV": "car", "MCV": "truck", "LCV": "truck", "LCV_BEV": "bolt"}


def _pomo_route_coords(inst, tour):
    """
    Convert POMO tour (list of node indices, 1-based customers, 0=depot) to
    a list of (lat, lng) pairs starting and ending at depot.
    """
    depot = (inst["depot"]["lat"], inst["depot"]["lng"])
    nodes = [(nd["lat"], nd["lng"]) for nd in inst["nodes"]]

    coords = [depot]
    for idx in tour:
        if idx == 0:
            coords.append(depot)
        else:
            coords.append(nodes[idx - 1])
    if coords[-1] != depot:
        coords.append(depot)
    return coords


def _dhl_route_coords(inst):
    """
    Reconstruct DHL driver's actual sequence by sorting stops by act_time_min.
    """
    depot = (inst["depot"]["lat"], inst["depot"]["lng"])
    sorted_nodes = sorted(inst["nodes"], key=lambda nd: nd["act_time_min"])
    coords = [depot] + [(nd["lat"], nd["lng"]) for nd in sorted_nodes] + [depot]
    return coords, sorted_nodes


def make_map(inst, pomo_tour, out_path):
    """
    Generate an HTML folium map comparing DHL and POMO routes for one instance.

    Args:
        inst:       real instance dict from real_instances.pkl
        pomo_tour:  list of node indices (0=depot, 1..n=customers) from solve()
        out_path:   path to write the HTML file
    """
    depot_lat = inst["depot"]["lat"]
    depot_lng = inst["depot"]["lng"]
    depot_name = inst["depot_name"]
    color = DEPOT_COLORS.get(depot_name, "#8e44ad")

    m = folium.Map(location=[depot_lat, depot_lng], zoom_start=13, tiles="CartoDB positron")

    # ── Depot marker ──────────────────────────────────────────────────────────
    folium.Marker(
        [depot_lat, depot_lng],
        tooltip=f"Depot ({depot_name})",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    # ── Customer stops ────────────────────────────────────────────────────────
    for i, nd in enumerate(inst["nodes"]):
        tw = f"TW: {nd['open_min']//60:02d}:{nd['open_min']%60:02d}–{nd['close_min']//60:02d}:{nd['close_min']%60:02d}"
        actual = f"Actual: {nd['act_time_min']//60:02d}:{nd['act_time_min']%60:02d}"
        folium.CircleMarker(
            [nd["lat"], nd["lng"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Stop {i+1} | {nd['demand_kg']:.1f} kg | {tw} | {actual}",
        ).add_to(m)

    # ── DHL baseline route (dashed, orange) ───────────────────────────────────
    dhl_coords, _ = _dhl_route_coords(inst)
    folium.PolyLine(
        dhl_coords,
        color="#e67e22",
        weight=3,
        opacity=0.7,
        dash_array="8 4",
        tooltip="DHL baseline (actual driver sequence)",
    ).add_to(m)

    # ── POMO route (solid, blue) ──────────────────────────────────────────────
    if pomo_tour:
        pomo_coords = _pomo_route_coords(inst, pomo_tour)
        folium.PolyLine(
            pomo_coords,
            color="#2980b9",
            weight=3,
            opacity=0.9,
            tooltip="POMO route",
        ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    route = inst["route"]
    date = str(inst["date"])[:10]
    vtype = inst["vehicle_type"]
    ev_tag = " ⚡ EV" if inst["is_electric"] else ""
    n = inst["n_nodes"]
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);
                font-family:Arial;font-size:13px;line-height:1.6">
      <b>{route}</b> · {depot_name} · {date}<br>
      {vtype}{ev_tag} · {n} stops<br>
      <span style="color:#e67e22">━ ━</span> DHL baseline<br>
      <span style="color:#2980b9">───</span> POMO route
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    return out_path


def run(instances, n_maps=5, ev_only=False, route_filter=None, depot_filter=None,
        use_pomo=True, checkpoint_path=None):
    """
    Generate maps for a selection of instances.

    If checkpoint_path is provided, runs POMO inference to get tours.
    Otherwise draws DHL route only.
    """
    import yaml

    subset = instances
    if ev_only:
        subset = [i for i in subset if i["is_electric"]]
    if route_filter:
        subset = [i for i in subset if i["route"] == route_filter]
    if depot_filter:
        subset = [i for i in subset if i["depot_name"] == depot_filter]
    subset = subset[:n_maps]

    model = None
    device = "cpu"
    if use_pomo and checkpoint_path and os.path.exists(checkpoint_path):
        import torch
        from pomo.model import POMOModel
        with open(os.path.join(ROOT, "configs", "default.yaml")) as f:
            config = yaml.safe_load(f)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        input_dim = ckpt["model_state"]["encoder.input_proj.weight"].shape[1]
        model = POMOModel(config).to(device)
        if input_dim != 3:
            model.set_input_dim(input_dim)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}  (input_dim={input_dim})")

    out_dir = os.path.join(ROOT, "viz", "maps")
    generated = []

    for inst in subset:
        pomo_tour = None
        if model is not None:
            from pomo.inference import solve
            with torch.no_grad():
                pomo_tour, pomo_dist = solve(model, inst, use_augmentation=False, device=device)
            dhl_dist = sum(
                _haversine(inst["nodes"][i - 1]["lat"] if i > 0 else inst["depot"]["lat"],
                           inst["nodes"][i - 1]["lng"] if i > 0 else inst["depot"]["lng"],
                           inst["nodes"][j - 1]["lat"] if j > 0 else inst["depot"]["lat"],
                           inst["nodes"][j - 1]["lng"] if j > 0 else inst["depot"]["lng"])
                for i, j in zip([0] + list(pomo_tour), list(pomo_tour) + [0])
            ) if False else None  # skip redundant compute, pomo_dist already correct

        fname = f"{inst['depot_name']}_{inst['route']}_{str(inst['date'])[:10]}.html"
        out_path = os.path.join(out_dir, fname)
        make_map(inst, pomo_tour, out_path)
        generated.append(out_path)
        print(f"  Saved: {out_path}")

    return generated


def _haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1, math.sqrt(a)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of maps to generate")
    parser.add_argument("--route", type=str, default=None)
    parser.add_argument("--depot", type=str, default=None, choices=["SAW", "IGA", "CET"])
    parser.add_argument("--ev_only", action="store_true")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(ROOT, "colab_results", "best_finetune.pt"))
    parser.add_argument("--no_pomo", action="store_true", help="DHL route only, no inference")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "data", "processed", "real_instances.pkl"), "rb") as f:
        instances = pickle.load(f)

    print(f"Generating {args.n} route maps...")
    paths = run(
        instances,
        n_maps=args.n,
        ev_only=args.ev_only,
        route_filter=args.route,
        depot_filter=args.depot,
        use_pomo=not args.no_pomo,
        checkpoint_path=args.checkpoint,
    )
    print(f"\nDone. {len(paths)} maps written to viz/maps/")


if __name__ == "__main__":
    main()
