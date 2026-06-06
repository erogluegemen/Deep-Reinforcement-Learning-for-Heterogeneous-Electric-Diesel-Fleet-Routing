"""
Evaluation entry point.

Usage:
  python evaluate.py --checkpoint checkpoints/best.pt --mode synthetic
  python evaluate.py --checkpoint checkpoints/best.pt --mode real
  python evaluate.py --checkpoint checkpoints/best.pt --mode real --no_aug
"""

import argparse
import os
import yaml
import torch
import pickle
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eval_synthetic(model, config, device, n_instances=100, n_nodes=50, use_aug=True):
    """Evaluate on freshly generated synthetic CVRP instances."""
    from pomo.trainer import generate_cvrp_batch
    from pomo.inference import greedy_rollout

    capacity = config["training"]["capacity"]
    aug_count = config["inference"]["augmentation_count"] if use_aug else 1

    rewards_all = []
    model.eval()

    batch_size = min(n_instances, 32)
    n_batches = (n_instances + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in range(n_batches):
            node_xy, demand, demand_norm, cap_tensor, dist_matrix = generate_cvrp_batch(
                batch_size, n_nodes, capacity, device
            )
            nf = torch.cat([node_xy, demand_norm.unsqueeze(-1)], dim=-1)

            best_rewards = torch.full((batch_size,), -float("inf"), device=device)

            for aug_idx in range(aug_count):
                from pomo.inference import _augment_coords, _recompute_dist_matrix
                aug_xy = _augment_coords(node_xy, aug_idx)
                aug_nf = torch.cat([aug_xy, demand_norm.unsqueeze(-1)], dim=-1)

                rewards, _ = greedy_rollout(model, aug_nf, demand, cap_tensor, dist_matrix, device)
                best_aug = rewards.max(dim=1).values
                best_rewards = torch.maximum(best_rewards, best_aug)

            rewards_all.extend((-best_rewards).cpu().tolist())

    mean_len = sum(rewards_all) / len(rewards_all)
    print(f"\n[Synthetic CVRP n={n_nodes}] {len(rewards_all)} instances")
    print(f"  Mean tour length: {mean_len:.4f}")
    print(f"  Min:  {min(rewards_all):.4f}  Max: {max(rewards_all):.4f}")
    return mean_len


def eval_real(model, config, device, use_aug=True, n_instances=None, force_cvrp=False):
    """Evaluate on real DHL Istanbul instances and compare to baseline."""
    from pomo.inference import solve

    instances_path = os.path.join(ROOT, "data", "processed", "real_instances.pkl")
    baseline_path = os.path.join(ROOT, "data", "processed", "baseline_metrics.csv")

    if not os.path.exists(instances_path):
        print("Real instances not found. Run: PYTHONPATH=. python3 data/instance_builder.py")
        return

    with open(instances_path, "rb") as f:
        instances = pickle.load(f)

    if n_instances:
        instances = instances[:n_instances]

    # Load baseline metrics
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            for row in csv.DictReader(f):
                key = (row["route"], row["depot"], row["date"])
                baseline[key] = float(row["total_dist_km"])

    aug_count = config["inference"]["augmentation_count"] if use_aug else 1

    results = []
    model.eval()

    for inst in instances:
        key = (inst["route"], inst["depot_name"],
               str(inst["date"].date()) if hasattr(inst["date"], "date") else str(inst["date"]))
        baseline_dist = baseline.get(key)

        with torch.no_grad():
            _, pomo_dist = solve(model, inst, use_augmentation=use_aug,
                                 augmentation_count=aug_count, device=device,
                                 force_cvrp=force_cvrp)

        results.append({
            "route": inst["route"],
            "depot": inst["depot_name"],
            "n_nodes": inst["n_nodes"],
            "vehicle_type": inst["vehicle_type"],
            "is_electric": inst["is_electric"],
            "pomo_dist_km": pomo_dist,
            "baseline_dist_km": baseline_dist,
            "reduction_pct": (
                100 * (baseline_dist - pomo_dist) / baseline_dist
                if baseline_dist else None
            ),
        })

    # Summary
    pomo_dists = [r["pomo_dist_km"] for r in results]
    baseline_dists = [r["baseline_dist_km"] for r in results if r["baseline_dist_km"]]
    reductions = [r["reduction_pct"] for r in results if r["reduction_pct"] is not None]

    print(f"\n[Real Instances] {len(results)} instances")
    print(f"  POMO mean dist: {sum(pomo_dists)/len(pomo_dists):.2f} km")
    if baseline_dists:
        print(f"  Baseline mean dist: {sum(baseline_dists)/len(baseline_dists):.2f} km")
    if reductions:
        print(f"  Mean reduction: {sum(reductions)/len(reductions):.2f}%")

    # Per-depot breakdown
    for depot in ("SAW", "IGA", "CET"):
        sub = [r for r in results if r["depot"] == depot]
        if not sub:
            continue
        d = [r["pomo_dist_km"] for r in sub]
        print(f"  {depot}: {len(sub)} instances, mean POMO dist {sum(d)/len(d):.2f} km")

    # EV-specific
    ev = [r for r in results if r["is_electric"]]
    if ev:
        ev_d = [r["pomo_dist_km"] for r in ev]
        over_range = [r for r in ev if r["pomo_dist_km"] > 200]
        print(f"  EV: {len(ev)} instances, mean {sum(ev_d)/len(ev_d):.2f} km, "
              f"{len(over_range)} exceed 200km range")

    # Save results
    out_path = os.path.join(ROOT, "data", "processed", "pomo_results.csv")
    if results:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"  Results saved to {out_path}")

    return results


def load_model(checkpoint_path, config, device):
    """Load a checkpoint, auto-detecting input_dim from the saved weights."""
    from pomo.model import POMOModel
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = ckpt["model_state"]["encoder.input_proj.weight"].shape[1]
    model = POMOModel(config).to(device)
    if input_dim != 3:
        model.set_input_dim(input_dim)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: {checkpoint_path}  (input_dim={input_dim}, "
          f"best_tour={-ckpt.get('best_mean_reward', float('nan')):.4f})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--mode", choices=["synthetic", "real", "both"], default="both")
    # Separate checkpoints: CVRP for synthetic eval, VRPTW for real-instance eval
    parser.add_argument("--cvrp_checkpoint",
                        default=os.path.join(ROOT, "colab_results", "best_cvrp.pt"))
    parser.add_argument("--vrptw_checkpoint",
                        default=os.path.join(ROOT, "colab_results", "best_vrptw.pt"),
                        help="VRPTW checkpoint; swap to best_finetune.pt after fine-tuning")
    parser.add_argument("--n_nodes", type=int, default=50)
    parser.add_argument("--n_instances", type=int, default=256)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--force_cvrp", action="store_true",
                        help="Ignore TW features; use CVRP model on real instances")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or get_device()
    use_aug = not args.no_aug

    if args.mode in ("synthetic", "both"):
        model_cvrp = load_model(args.cvrp_checkpoint, config, device)
        eval_synthetic(model_cvrp, config, device, args.n_instances, args.n_nodes, use_aug)

    if args.mode in ("real", "both"):
        ckpt_path = args.cvrp_checkpoint if args.force_cvrp else args.vrptw_checkpoint
        model_real = load_model(ckpt_path, config, device)
        eval_real(model_real, config, device, use_aug,
                  n_instances=args.n_instances, force_cvrp=args.force_cvrp)


if __name__ == "__main__":
    main()
