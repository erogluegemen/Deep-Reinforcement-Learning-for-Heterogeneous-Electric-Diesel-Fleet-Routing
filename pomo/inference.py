"""
POMO inference: greedy decoding with optional 8× coordinate augmentation.

Augmentations (8 symmetries of the unit square):
  0: identity
  1: flip x (x → 1-x)
  2: flip y (y → 1-y)
  3: flip both (x,y) → (1-x, 1-y)
  4: transpose (x,y) → (y,x)
  5: transpose + flip x → (y, 1-x)
  6: transpose + flip y → (1-y, x)
  7: transpose + flip both → (1-y, 1-x)

For each augmented instance we run n POMO starts, then return the best tour
across all (augmentations × starts) combinations.
"""

import torch
from pomo.env import VRPEnv


def _augment_coords(xy, aug_idx):
    """
    xy: (B, n+1, 2)  — x=col0, y=col1, all in [0,1]
    aug_idx: int 0–7
    Returns augmented (B, n+1, 2).
    """
    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]

    if aug_idx == 0:
        return xy
    elif aug_idx == 1:
        return torch.cat([1 - x, y], dim=-1)
    elif aug_idx == 2:
        return torch.cat([x, 1 - y], dim=-1)
    elif aug_idx == 3:
        return torch.cat([1 - x, 1 - y], dim=-1)
    elif aug_idx == 4:
        return torch.cat([y, x], dim=-1)
    elif aug_idx == 5:
        return torch.cat([y, 1 - x], dim=-1)
    elif aug_idx == 6:
        return torch.cat([1 - y, x], dim=-1)
    elif aug_idx == 7:
        return torch.cat([1 - y, 1 - x], dim=-1)
    else:
        raise ValueError(f"aug_idx must be 0–7, got {aug_idx}")


def _recompute_dist_matrix(xy):
    """Euclidean distance matrix from coordinates (B, n+1, 2) → (B, n+1, n+1)."""
    diff = xy.unsqueeze(2) - xy.unsqueeze(1)
    return diff.norm(dim=-1)


def greedy_rollout(model, node_features, demand, capacity, dist_matrix, device,
                   vehicle_type_id=None, is_electric=False, range_km=None,
                   time_windows=None, speed=1.0, service_time=0.0, time_horizon=1.0):
    """
    Single greedy (argmax) rollout. No sampling — deterministic decoding.

    Supports CVRP (time_windows=None) and VRPTW (time_windows provided).

    Returns:
        rewards:   (B, S)  negative tour lengths
        tours:     list of B best-start tours (list of node indices)
    """
    B = node_features.shape[0]
    n = node_features.shape[1] - 1
    S = n
    BS = B * S

    H, graph_embed = model.encode(node_features)

    H_exp = H.unsqueeze(1).expand(B, S, n + 1, -1).reshape(BS, n + 1, -1)
    ge_exp = graph_embed.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)
    dist_exp = dist_matrix.unsqueeze(1).expand(B, S, n + 1, n + 1).reshape(BS, n + 1, n + 1)
    demand_exp = demand.unsqueeze(1).expand(B, S, n + 1).reshape(BS, n + 1)
    cap_exp = capacity.unsqueeze(1).expand(B, S).reshape(BS)

    tw_exp = None
    if time_windows is not None:
        tw_exp = time_windows.unsqueeze(1).expand(B, S, n + 1, 2).reshape(BS, n + 1, 2)

    env = VRPEnv(device=device)
    env.B = BS
    env.n = n
    env.S = 1
    env.dist_matrix = dist_exp
    env.demand = demand_exp
    env.capacity = cap_exp
    env.is_electric = is_electric
    env.range_km = range_km
    env.speed = speed
    env.service_time = service_time
    env.time_windows = tw_exp

    flat_idx = torch.arange(BS, device=device)
    flat_start = (flat_idx % S) + 1

    env.visited = torch.zeros(BS, 1, n + 1, dtype=torch.bool, device=device)
    env.visited[:, :, 0] = True
    env.visited[flat_idx, 0, flat_start] = True
    env.current_node = flat_start.unsqueeze(1)

    start_dem = demand_exp[flat_idx, flat_start]
    env.remaining_cap = (cap_exp - start_dem).unsqueeze(1)
    env.route_dist = torch.zeros(BS, 1, device=device)
    env.cumul_dist = torch.zeros(BS, 1, device=device)

    if tw_exp is not None:
        start_travel = dist_exp[flat_idx, 0, flat_start] / speed
        env.current_time = (start_travel + service_time).unsqueeze(1)
    else:
        env.current_time = None

    tour_log = [[] for _ in range(BS)]
    max_steps = 3 * n + 1

    for _ in range(max_steps):
        if env.is_done():
            break
        mask = env.get_mask().squeeze(1)
        cur = env.current_node.squeeze(1)
        cap_norm = env.remaining_cap.squeeze(1) / cap_exp

        cur_time_norm = None
        if env.current_time is not None:
            cur_time_norm = (env.current_time.squeeze(1) / time_horizon).clamp(0, 1)

        veh_id_exp = None
        if vehicle_type_id is not None:
            veh_id_exp = vehicle_type_id.unsqueeze(1).expand(B, S).reshape(BS)

        log_probs = model.decode_step(H_exp, ge_exp, cur, cap_norm, mask,
                                      veh_id_exp, cur_time_norm)

        action = log_probs.argmax(dim=-1)

        for k in range(BS):
            tour_log[k].append(action[k].item())

        env.step(action.unsqueeze(1))

    rewards = env.get_total_reward().squeeze(1).reshape(B, S)

    best_start = rewards.argmax(dim=1)
    best_tours = [
        tour_log[b * S + best_start[b].item()]
        for b in range(B)
    ]

    return rewards, best_tours


@torch.no_grad()
def solve(model, instance, use_augmentation=True, augmentation_count=8, device=None,
          force_cvrp=False):
    """
    Solve a single VRP instance dict (as produced by instance_builder).

    Automatically detects CVRP vs VRPTW from instance keys and builds the
    correct node features (3-dim for CVRP, 5-dim for VRPTW).

    Returns:
        best_tour:   list of node indices (1-based, depot=0)
        best_length: float, total route distance in km (real instances) or units (synthetic)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    coords_norm = torch.tensor(instance["coords_norm"], dtype=torch.float32, device=device)
    xy = coords_norm.unsqueeze(0)  # (1, n+1, 2)

    demand_vals = [0.0] + [nd["demand_kg"] for nd in instance["nodes"]]
    demand = torch.tensor(demand_vals, dtype=torch.float32, device=device).unsqueeze(0)
    cap = torch.tensor([float(instance["weight_kg"])], device=device)
    demand_norm = demand / cap.unsqueeze(1)

    dist_matrix = torch.tensor(
        instance["dist_matrix"], dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, n+1, n+1)

    # Detect VRPTW: customer nodes have open_min/close_min
    # force_cvrp=True skips TW features (needed when using a 3-dim CVRP model)
    has_tw = (not force_cvrp
              and len(instance["nodes"]) > 0
              and instance["nodes"][0].get("open_min") is not None)

    time_windows = None
    tw_norm = None
    service_time = 0.0
    time_horizon = 1.0

    if has_tw:
        all_close = [nd["close_min"] for nd in instance["nodes"]]
        # Normalise by actual max close time so features stay in [0,1]
        time_horizon = max(max(all_close), 480.0)
        tw_rows = [[0.0, time_horizon]]  # depot open all day
        for nd in instance["nodes"]:
            tw_rows.append([float(nd["open_min"]), float(nd["close_min"])])
        time_windows = torch.tensor(tw_rows, dtype=torch.float32, device=device).unsqueeze(0)
        tw_norm = time_windows / time_horizon                   # (1, n+1, 2) normalised in [0,1]
        node_features = torch.cat(
            [xy, demand_norm.unsqueeze(-1), tw_norm], dim=-1   # (1, n+1, 5)
        )
        service_time = float(instance["nodes"][0].get("service_time_min", 3.0))
    else:
        node_features = torch.cat([xy, demand_norm.unsqueeze(-1)], dim=-1)  # (1, n+1, 3)

    aug_indices = range(augmentation_count) if use_augmentation else [0]
    best_reward = -float("inf")
    best_tour = None

    for aug_idx in aug_indices:
        aug_xy = _augment_coords(xy, aug_idx)
        if has_tw:
            aug_features = torch.cat([aug_xy, demand_norm.unsqueeze(-1), tw_norm], dim=-1)
        else:
            aug_features = torch.cat([aug_xy, demand_norm.unsqueeze(-1)], dim=-1)

        rewards, tours = greedy_rollout(
            model, aug_features, demand, cap, dist_matrix, device,
            is_electric=instance.get("is_electric", False),
            range_km=instance.get("range_km"),
            time_windows=time_windows,
            speed=1.0,
            service_time=service_time,
            time_horizon=time_horizon,
        )

        best_aug_reward = rewards.max().item()
        if best_aug_reward > best_reward:
            best_reward = best_aug_reward
            best_tour = tours[0]

    return best_tour, -best_reward


if __name__ == "__main__":
    import yaml, os

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(ROOT, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    from pomo.model import POMOModel
    device = "cpu"
    model = POMOModel(config).to(device)

    B, n = 2, 10
    xy = torch.rand(B, n + 1, 2)
    demand = torch.zeros(B, n + 1)
    demand[:, 1:] = torch.randint(1, 10, (B, n)).float()
    cap = torch.full((B,), 50.0)
    demand_norm = demand / 50.0
    nf = torch.cat([xy, demand_norm.unsqueeze(-1)], dim=-1)
    diff = xy.unsqueeze(2) - xy.unsqueeze(1)
    dist_matrix = diff.norm(dim=-1)

    rewards, tours = greedy_rollout(model, nf, demand, cap, dist_matrix, device)
    print(f"Greedy rewards (neg tour length): {rewards.max(dim=1).values}")
    print(f"Best tour (instance 0): {tours[0]}")
    print(f"Aug check (augment idx 3): {_augment_coords(xy[:1], 3).shape}")
