"""
Synthetic benchmark: POMO vs OR-Tools vs Nearest Neighbour.

Generates Solomon-style CVRP and VRPTW instances and reports:
  - Mean tour length for each method
  - Optimality gap vs OR-Tools (for n=20, where OR-Tools is tractable)
  - EV range sensitivity analysis

Usage:
  python eval/benchmark_eval.py \
      --cvrp_checkpoint  colab_results/best_cvrp.pt \
      --vrptw_checkpoint colab_results/best_vrptw.pt

  # skip OR-Tools (faster, no optimality gap):
  python eval/benchmark_eval.py --no_ortools

  # only CVRP:
  python eval/benchmark_eval.py --problems cvrp
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ORTOOLS_SCALE = 10_000   # float→int scaling for OR-Tools


# ── Instance generators ───────────────────────────────────────────────────────

def _euclidean_dm(coords):
    """coords: (n+1, 2) → (n+1, n+1) float32 distance matrix."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)


def gen_cvrp_batch(B, n, cap=50, base_seed=0):
    """
    Generate B CVRP instances.
    Returns: coords (B, n+1, 2), demand (B, n+1), cap float, dm (B, n+1, n+1)
    """
    rng = np.random.RandomState(base_seed)
    coords = rng.rand(B, n + 1, 2).astype(np.float32)
    demand = np.zeros((B, n + 1), dtype=np.float32)
    demand[:, 1:] = rng.randint(1, 10, (B, n)).astype(np.float32)
    dm = np.stack([_euclidean_dm(coords[b]) for b in range(B)])
    return coords, demand, float(cap), dm


def gen_vrptw_batch(B, n, cap=50, base_seed=1000, T=480.0):
    """
    Generate B VRPTW instances matching the training distribution.

    T=480 matches configs/default.yaml vrptw.time_horizon (minutes).
    Travel times are Euclidean distance in [0,1]^2 (speed=1.0), so max
    travel time ≈ 1.41 << T. Time windows [0.15T, 0.35T] half-width are
    therefore always loose — this matches training, where TW masking is
    rarely triggered.

    Returns: coords, demand, cap, dm, time_windows (B, n+1, 2), T
    """
    rng = np.random.RandomState(base_seed)
    coords = rng.rand(B, n + 1, 2).astype(np.float32)
    demand = np.zeros((B, n + 1), dtype=np.float32)
    demand[:, 1:] = rng.randint(1, 10, (B, n)).astype(np.float32)
    dm = np.stack([_euclidean_dm(coords[b]) for b in range(B)])

    hw      = rng.uniform(0.15, 0.35, (B, n + 1)) * T
    center  = hw + rng.uniform(0, 1, (B, n + 1)) * (T - 2 * hw)
    center  = np.clip(center, hw, T - hw)
    tw      = np.zeros((B, n + 1, 2), dtype=np.float32)
    tw[:, :, 0] = np.clip(center - hw, 0, T)
    tw[:, :, 1] = np.clip(center + hw, 0, T)
    tw[:, 0]    = [0.0, T]   # depot open all day

    return coords, demand, float(cap), dm, tw, float(T)


# ── Nearest-neighbour heuristic ───────────────────────────────────────────────

def nearest_neighbour_cvrp(coords, demand, cap):
    """Single-instance greedy nearest-neighbour for CVRP. Returns tour length."""
    dm = _euclidean_dm(coords)
    n = len(coords) - 1
    visited = [False] * (n + 1)
    visited[0] = True
    cur, remaining, total = 0, cap, 0.0

    while not all(visited[1:]):
        best_j, best_d = -1, float("inf")
        for j in range(1, n + 1):
            if not visited[j] and demand[j] <= remaining and dm[cur, j] < best_d:
                best_d, best_j = dm[cur, j], j
        if best_j == -1:
            total += dm[cur, 0]
            cur, remaining = 0, cap
        else:
            total += dm[cur, best_j]
            remaining -= demand[best_j]
            visited[best_j] = True
            cur = best_j

    return total + dm[cur, 0]


# ── OR-Tools solvers ──────────────────────────────────────────────────────────

def _ortools_available():
    try:
        import ortools.constraint_solver  # noqa
        return True
    except ImportError:
        return False


def _min_vehicles(demand, cap):
    """Minimum vehicles needed = ceil(total_customer_demand / capacity).
    Equivalent to POMO's depot-return trips for multi-trip single-vehicle routing.
    """
    import math
    total = int(sum(demand[1:]))  # exclude depot (demand[0]=0)
    return max(1, math.ceil(total / int(cap)))


def solve_ortools_cvrp(coords, demand, cap, time_limit=30):
    """
    Solve CVRP with OR-Tools using the minimum number of vehicles required.

    POMO solves multi-trip single-vehicle CVRP (one vehicle, multiple depot
    returns). OR-Tools does not support depot revisits natively, so we use
    k vehicles where k = ceil(total_demand / cap). Each OR-Tools vehicle
    corresponds to one trip of POMO's single vehicle. Total distance is
    therefore directly comparable.

    Returns tour length or None if no solution found.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    dm_int = (_euclidean_dm(coords) * ORTOOLS_SCALE).astype(int)
    n = len(coords) - 1
    cap_int = int(cap)
    dem_int = [int(d) for d in demand]
    k = _min_vehicles(demand, cap)

    manager = pywrapcp.RoutingIndexManager(n + 1, k, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dm_int[manager.IndexToNode(i), manager.IndexToNode(j)])

    cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    def dem_cb(i):
        return dem_int[manager.IndexToNode(i)]

    dc = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(dc, 0, [cap_int] * k, True, "Cap")

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = time_limit

    sol = routing.SolveWithParameters(p)
    return sol.ObjectiveValue() / ORTOOLS_SCALE if sol else None


def solve_ortools_vrptw(coords, demand, cap, tw, T, time_limit=30):
    """Solve VRPTW with OR-Tools (multi-vehicle for multi-trip feasibility).
    Returns tour length or None if infeasible / no solution.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    dm_int = (_euclidean_dm(coords) * ORTOOLS_SCALE).astype(int)
    tw_int = (tw * ORTOOLS_SCALE).astype(int)
    n = len(coords) - 1
    cap_int = int(cap)
    dem_int = [int(d) for d in demand]
    T_int = int(T * ORTOOLS_SCALE)
    k = _min_vehicles(demand, cap)

    manager = pywrapcp.RoutingIndexManager(n + 1, k, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dm_int[manager.IndexToNode(i), manager.IndexToNode(j)])

    cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    routing.AddDimension(cb, T_int, T_int, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, n + 1):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(tw_int[node, 0]), int(tw_int[node, 1]))
    for v in range(k):
        time_dim.CumulVar(routing.Start(v)).SetRange(int(tw_int[0, 0]), int(tw_int[0, 1]))
        time_dim.CumulVar(routing.End(v)).SetRange(int(tw_int[0, 0]), int(tw_int[0, 1]))

    def dem_cb(i):
        return dem_int[manager.IndexToNode(i)]

    dc = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(dc, 0, [cap_int] * k, True, "Cap")

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = time_limit

    sol = routing.SolveWithParameters(p)
    return sol.ObjectiveValue() / ORTOOLS_SCALE if sol else None


# ── POMO batched evaluation ───────────────────────────────────────────────────

def _pomo_cvrp(model, coords_np, demand_np, cap, dm_np, device, use_aug, chunk=16):
    """
    Evaluate CVRP instances with POMO. Processes in chunks to avoid OOM on CPU.
    All arrays shape (B, n+1, ...). Returns list of B tour lengths.
    """
    from pomo.inference import greedy_rollout, _augment_coords

    B = coords_np.shape[0]
    results = []

    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        xy     = torch.tensor(coords_np[start:end], dtype=torch.float32, device=device)
        demand = torch.tensor(demand_np[start:end],  dtype=torch.float32, device=device)
        dm     = torch.tensor(dm_np[start:end],      dtype=torch.float32, device=device)
        b      = end - start
        cap_t  = torch.full((b,), cap, device=device)
        d_norm = demand / cap

        best = torch.full((b,), -float("inf"), device=device)
        for aug_idx in (range(8) if use_aug else range(1)):
            aug_xy = _augment_coords(xy, aug_idx)
            nf = torch.cat([aug_xy, d_norm.unsqueeze(-1)], dim=-1)
            rewards, _ = greedy_rollout(model, nf, demand, cap_t, dm, device)
            best = torch.maximum(best, rewards.max(dim=1).values)

        results.extend((-best).cpu().tolist())

    return results


def _pomo_vrptw(model, coords_np, demand_np, cap, dm_np, tw_np, T, device, use_aug,
                chunk=16):
    """
    Evaluate VRPTW instances with POMO. Processes in chunks to avoid OOM on CPU.
    Returns list of B tour lengths.
    """
    from pomo.inference import greedy_rollout, _augment_coords

    B = coords_np.shape[0]
    results = []

    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        xy     = torch.tensor(coords_np[start:end], dtype=torch.float32, device=device)
        demand = torch.tensor(demand_np[start:end],  dtype=torch.float32, device=device)
        dm     = torch.tensor(dm_np[start:end],      dtype=torch.float32, device=device)
        tw     = torch.tensor(tw_np[start:end],      dtype=torch.float32, device=device)
        b      = end - start
        cap_t  = torch.full((b,), cap, device=device)
        d_norm = demand / cap
        tw_norm = tw / T

        best = torch.full((b,), -float("inf"), device=device)
        for aug_idx in (range(8) if use_aug else range(1)):
            aug_xy = _augment_coords(xy, aug_idx)
            nf = torch.cat([aug_xy, d_norm.unsqueeze(-1), tw_norm], dim=-1)
            rewards, _ = greedy_rollout(
                model, nf, demand, cap_t, dm, device,
                time_windows=tw, speed=1.0, service_time=0.0, time_horizon=T,
            )
            best = torch.maximum(best, rewards.max(dim=1).values)

        results.extend((-best).cpu().tolist())

    return results


# ── EV range sensitivity ──────────────────────────────────────────────────────

def ev_range_analysis(model_cvrp, n, B, device, range_vals=(None, 5.0, 3.0, 2.0)):
    """
    Run CVRP with increasing range constraints and report mean tour length.
    range=None means unconstrained (pure CVRP).
    """
    from pomo.inference import greedy_rollout

    coords_np, demand_np, cap, dm_np = gen_cvrp_batch(B, n, base_seed=9999)
    xy     = torch.tensor(coords_np, dtype=torch.float32, device=device)
    demand = torch.tensor(demand_np,  dtype=torch.float32, device=device)
    cap_t  = torch.full((B,), cap, device=device)
    dm     = torch.tensor(dm_np,      dtype=torch.float32, device=device)
    d_norm = demand / cap
    nf     = torch.cat([xy, d_norm.unsqueeze(-1)], dim=-1)

    print("\n── EV Range Sensitivity (CVRP model, n={}, B={}) ──────────────".format(n, B))
    print(f"  {'Range':>12}  {'Mean tour':>12}  {'vs unconstrained':>18}")

    base_mean = None
    rows = []
    for rv in range_vals:
        rewards, _ = greedy_rollout(
            model_cvrp, nf, demand, cap_t, dm, device,
            is_electric=(rv is not None),
            range_km=rv,
        )
        mean_len = (-rewards.max(dim=1).values).mean().item()
        label = "unlimited" if rv is None else f"{rv:.1f} units"
        if base_mean is None:
            base_mean = mean_len
            delta = "—"
        else:
            delta = f"{100*(mean_len - base_mean)/base_mean:+.1f}%"
        print(f"  {label:>12}  {mean_len:>12.4f}  {delta:>18}")
        rows.append({"range": label, "mean_tour": mean_len})

    return rows


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(ckpt_path, config, device):
    from pomo.model import POMOModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = ckpt["model_state"]["encoder.input_proj.weight"].shape[1]
    model = POMOModel(config).to(device)
    if input_dim != 3:
        model.set_input_dim(input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded {ckpt_path}  (input_dim={input_dim})")
    return model


# ── Main benchmark runner ─────────────────────────────────────────────────────

def run_cvrp_benchmark(model, n_list, B, device, use_aug, run_ortools, ort_limit,
                       ort_n_instances):
    print("\n═══ CVRP Benchmark ══════════════════════════════════════════════")
    all_rows = []

    for n in n_list:
        print(f"\n  n={n}, B={B} instances")
        coords, demand, cap, dm = gen_cvrp_batch(B, n, base_seed=0)

        # Nearest neighbour
        nn_lens = [nearest_neighbour_cvrp(coords[b], demand[b], cap) for b in range(B)]

        # POMO greedy
        pomo_lens = _pomo_cvrp(model, coords, demand, cap, dm, device, use_aug=False)

        # POMO 8× aug
        aug_lens = None
        if use_aug:
            aug_lens = _pomo_cvrp(model, coords, demand, cap, dm, device, use_aug=True)

        # OR-Tools (only for small n)
        ort_lens = []
        if run_ortools and n <= 20:
            n_ort = min(ort_n_instances, B)
            print(f"    OR-Tools ({n_ort} instances, {ort_limit}s each) ...", end="", flush=True)
            t0 = time.time()
            for b in range(n_ort):
                res = solve_ortools_cvrp(coords[b], demand[b], cap, ort_limit)
                if res is not None:
                    ort_lens.append(res)
            print(f" done ({time.time()-t0:.0f}s, {len(ort_lens)}/{n_ort} solved)")

        row = {
            "problem": "CVRP", "n": n,
            "nn_mean":       round(float(np.mean(nn_lens)),   4),
            "pomo_mean":     round(float(np.mean(pomo_lens)),  4),
            "pomo_aug_mean": round(float(np.mean(aug_lens)),   4) if aug_lens else None,
            "ort_mean":      round(float(np.mean(ort_lens)),   4) if ort_lens else None,
            "ort_n_solved":  len(ort_lens),
        }
        row["pomo_gap_pct"] = (
            round(100 * (row["pomo_mean"] - row["ort_mean"]) / row["ort_mean"], 2)
            if row["ort_mean"] else None
        )
        row["pomo_aug_gap_pct"] = (
            round(100 * (row["pomo_aug_mean"] - row["ort_mean"]) / row["ort_mean"], 2)
            if (row["ort_mean"] and row["pomo_aug_mean"]) else None
        )
        all_rows.append(row)

        _print_row(row)

    return all_rows


def run_vrptw_benchmark(model, n_list, B, device, use_aug, run_ortools, ort_limit,
                        ort_n_instances):
    print("\n═══ VRPTW Benchmark ═════════════════════════════════════════════")
    all_rows = []

    for n in n_list:
        print(f"\n  n={n}, B={B} instances")
        coords, demand, cap, dm, tw, T = gen_vrptw_batch(B, n, base_seed=1000)

        # POMO greedy
        pomo_lens = _pomo_vrptw(model, coords, demand, cap, dm, tw, T, device, use_aug=False)

        # POMO 8× aug
        aug_lens = None
        if use_aug:
            aug_lens = _pomo_vrptw(model, coords, demand, cap, dm, tw, T, device, use_aug=True)

        # OR-Tools (only for small n)
        ort_lens = []
        if run_ortools and n <= 20:
            n_ort = min(ort_n_instances, B)
            print(f"    OR-Tools ({n_ort} instances, {ort_limit}s each) ...", end="", flush=True)
            t0 = time.time()
            for b in range(n_ort):
                res = solve_ortools_vrptw(coords[b], demand[b], cap, tw[b], T, ort_limit)
                if res is not None:
                    ort_lens.append(res)
            print(f" done ({time.time()-t0:.0f}s, {len(ort_lens)}/{n_ort} solved)")

        row = {
            "problem": "VRPTW", "n": n,
            "nn_mean":       None,
            "pomo_mean":     round(float(np.mean(pomo_lens)),  4),
            "pomo_aug_mean": round(float(np.mean(aug_lens)),   4) if aug_lens else None,
            "ort_mean":      round(float(np.mean(ort_lens)),   4) if ort_lens else None,
            "ort_n_solved":  len(ort_lens),
        }
        row["pomo_gap_pct"] = (
            round(100 * (row["pomo_mean"] - row["ort_mean"]) / row["ort_mean"], 2)
            if row["ort_mean"] else None
        )
        row["pomo_aug_gap_pct"] = (
            round(100 * (row["pomo_aug_mean"] - row["ort_mean"]) / row["ort_mean"], 2)
            if (row["ort_mean"] and row["pomo_aug_mean"]) else None
        )
        all_rows.append(row)

        _print_row(row)

    return all_rows


def _print_row(row):
    if row["nn_mean"] is not None:
        print(f"    Nearest Neighbour:    {row['nn_mean']:.4f}")
    print(f"    POMO (greedy):        {row['pomo_mean']:.4f}", end="")
    if row["pomo_gap_pct"] is not None:
        print(f"  (gap vs OR-Tools: {row['pomo_gap_pct']:+.2f}%)", end="")
    print()
    if row["pomo_aug_mean"] is not None:
        print(f"    POMO (8× aug):        {row['pomo_aug_mean']:.4f}", end="")
        if row["pomo_aug_gap_pct"] is not None:
            print(f"  (gap vs OR-Tools: {row['pomo_aug_gap_pct']:+.2f}%)", end="")
        print()
    if row["ort_mean"] is not None:
        print(f"    OR-Tools ({row['ort_n_solved']:3d} solved): {row['ort_mean']:.4f}")


def print_summary_table(rows):
    print("\n═══ Summary Table ═══════════════════════════════════════════════")
    header = f"{'Problem':>8} {'n':>4} {'NN':>8} {'POMO':>8} {'POMO+8×':>9} {'OR-Tools':>10} {'Gap%':>8} {'Gap+aug%':>10}"
    print(header)
    print("─" * len(header))
    for r in rows:
        nn  = f"{r['nn_mean']:.4f}"   if r["nn_mean"]        else "   —    "
        ort = f"{r['ort_mean']:.4f}"  if r["ort_mean"]        else "    —    "
        aug = f"{r['pomo_aug_mean']:.4f}" if r["pomo_aug_mean"] else "    —    "
        gap = f"{r['pomo_gap_pct']:+.2f}"    if r["pomo_gap_pct"]     else "   —"
        gpa = f"{r['pomo_aug_gap_pct']:+.2f}" if r["pomo_aug_gap_pct"] else "   —"
        print(f"{r['problem']:>8} {r['n']:>4} {nn:>8} {r['pomo_mean']:>8.4f} {aug:>9} {ort:>10} {gap:>8} {gpa:>10}")


def save_csv(rows, out_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--cvrp_checkpoint",
                        default=os.path.join(ROOT, "colab_results", "best_cvrp.pt"))
    parser.add_argument("--vrptw_checkpoint",
                        default=os.path.join(ROOT, "colab_results", "best_vrptw.pt"))
    parser.add_argument("--problems", choices=["cvrp", "vrptw", "both"], default="both")
    parser.add_argument("--n_list",   nargs="+", type=int, default=[20, 50])
    parser.add_argument("--n_instances", type=int, default=100,
                        help="Instances per (problem, n) cell")
    parser.add_argument("--no_aug",   action="store_true",
                        help="Skip 8× augmentation (faster)")
    parser.add_argument("--no_ortools", action="store_true",
                        help="Skip OR-Tools (no optimality gap, much faster)")
    parser.add_argument("--ort_limit", type=int, default=30,
                        help="OR-Tools time limit per instance (seconds)")
    parser.add_argument("--ort_n_instances", type=int, default=50,
                        help="How many instances to run through OR-Tools (<=n_instances)")
    parser.add_argument("--ev_analysis", action="store_true",
                        help="Run EV range sensitivity analysis")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_csv",
                        default=os.path.join(ROOT, "data", "processed", "benchmark_results.csv"))
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or get_device()
    use_aug = not args.no_aug
    run_ortools = not args.no_ortools

    if run_ortools and not _ortools_available():
        print("WARNING: OR-Tools not installed. Skipping OR-Tools comparison.")
        run_ortools = False

    print(f"Device: {device}  |  Aug: {use_aug}  |  OR-Tools: {run_ortools}")

    all_rows = []

    if args.problems in ("cvrp", "both"):
        print("\nLoading CVRP checkpoint...")
        model_cvrp = load_model(args.cvrp_checkpoint, config, device)
        rows = run_cvrp_benchmark(
            model_cvrp, args.n_list, args.n_instances, device,
            use_aug, run_ortools, args.ort_limit, args.ort_n_instances,
        )
        all_rows.extend(rows)

        if args.ev_analysis:
            ev_range_analysis(model_cvrp, n=20, B=100, device=device)

    if args.problems in ("vrptw", "both"):
        print("\nLoading VRPTW checkpoint...")
        model_vrptw = load_model(args.vrptw_checkpoint, config, device)
        rows = run_vrptw_benchmark(
            model_vrptw, args.n_list, args.n_instances, device,
            use_aug, run_ortools, args.ort_limit, args.ort_n_instances,
        )
        all_rows.extend(rows)

    print_summary_table(all_rows)
    save_csv(all_rows, args.out_csv)


if __name__ == "__main__":
    main()
