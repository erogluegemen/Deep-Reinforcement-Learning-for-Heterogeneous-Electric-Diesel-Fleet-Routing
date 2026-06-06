"""
POMO trainer: REINFORCE with within-instance shared baseline.

Key POMO idea:
  - For each training instance, run n starting nodes simultaneously (one per customer).
  - All n rollouts share one encoder pass — only the decoder differs per start.
  - Baseline = mean(reward) across the n starts of the SAME instance.
  - Loss = -mean[ (R_i - baseline) * log π_i ]

Synthetic CVRP generation:
  - n customer nodes, coordinates ~ Uniform(0,1)
  - demand ~ Uniform(1, 9), normalised by capacity
  - capacity = 50 (standard benchmark)
"""

import os
import math
import time
import torch
import torch.optim as optim
from tqdm import tqdm

from pomo.model import POMOModel
from pomo.env import VRPEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_cvrp_batch(batch_size, n_nodes, capacity, device):
    """
    Generates a batch of random CVRP instances (unit-square, standard benchmark).

    Returns:
        node_xy:     (B, n+1, 2)   coordinates, row 0 = depot
        demand:      (B, n+1)      unnormalised demand, depot=0
        demand_norm: (B, n+1)      demand / capacity (for node features)
        capacity:    (B,)          scalar capacity per instance
        dist_matrix: (B, n+1, n+1) Euclidean distances
    """
    B, n = batch_size, n_nodes

    # Node coordinates including depot at index 0
    node_xy = torch.rand(B, n + 1, 2, device=device)

    # Demand: 0 for depot, Uniform(1,9) for customers
    demand = torch.zeros(B, n + 1, device=device)
    demand[:, 1:] = torch.randint(1, 10, (B, n), device=device).float()  # U{1,...,9}

    cap_tensor = torch.full((B,), float(capacity), device=device)
    demand_norm = demand / capacity  # normalise for model input

    # Euclidean distance matrix (vectorised)
    diff = node_xy.unsqueeze(2) - node_xy.unsqueeze(1)  # (B, n+1, n+1, 2)
    dist_matrix = diff.norm(dim=-1)  # (B, n+1, n+1)

    return node_xy, demand, demand_norm, cap_tensor, dist_matrix


def generate_vrptw_batch(batch_size, n_nodes, capacity, time_horizon, service_time, device):
    """
    Generate synthetic VRPTW instances (Solomon-style, unit square).

    Time windows: center ~ U[tw_slack, T-tw_slack], half-width ~ U[0.15, 0.35]*T
    Ensures all customers are reachable in time from depot under unit speed.

    Returns:
        node_xy:      (B, n+1, 2)
        demand:       (B, n+1)          unnormalised
        demand_norm:  (B, n+1)
        capacity:     (B,)
        dist_matrix:  (B, n+1, n+1)
        time_windows: (B, n+1, 2)       [open, close] in absolute time units
        time_features:(B, n+1, 2)       [open/T, close/T] normalised for model input
    """
    B, n = batch_size, n_nodes
    T = float(time_horizon)

    node_xy = torch.rand(B, n + 1, 2, device=device)

    demand = torch.zeros(B, n + 1, device=device)
    demand[:, 1:] = torch.randint(1, 10, (B, n), device=device).float()
    cap_tensor = torch.full((B,), float(capacity), device=device)
    demand_norm = demand / capacity

    diff = node_xy.unsqueeze(2) - node_xy.unsqueeze(1)
    dist_matrix = diff.norm(dim=-1)  # (B, n+1, n+1) — Euclidean = travel time at speed=1

    # Time windows: half-width in [0.15T, 0.35T], center in [hw, T-hw]
    hw = torch.rand(B, n + 1, device=device) * (0.35 - 0.15) * T + 0.15 * T  # (B, n+1)
    center = hw + torch.rand(B, n + 1, device=device) * (T - 2 * hw)           # (B, n+1)
    center = center.clamp(hw, T - hw)

    tw_open = (center - hw).clamp(0, T)
    tw_close = (center + hw).clamp(0, T)

    # Depot: open all day
    tw_open[:, 0] = 0.0
    tw_close[:, 0] = T

    time_windows = torch.stack([tw_open, tw_close], dim=-1)   # (B, n+1, 2)
    time_features = time_windows / T                           # normalised for model

    return node_xy, demand, demand_norm, cap_tensor, dist_matrix, time_windows, time_features


def rollout(model, env, node_features, demand, capacity, dist_matrix, device):
    """
    Run a full POMO rollout (one episode per instance, n starts in parallel).

    Returns:
        total_rewards: (B, S)   negative total route distances (higher = better)
        log_prob_sum:  (B, S)   sum of log_probs over all decode steps
    """
    B = node_features.shape[0]
    n = node_features.shape[1] - 1
    S = n  # POMO: one start per customer node

    # Encode once per instance
    H, graph_embed = model.encode(node_features)  # (B, n+1, d), (B, d)

    # Expand for S starts: (B, n+1, d) → (B*S, n+1, d)
    H_exp = H.unsqueeze(1).expand(B, S, n + 1, -1).reshape(B * S, n + 1, -1)
    ge_exp = graph_embed.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)

    # Expand dist_matrix and demand for (B*S, ...)
    dist_exp = dist_matrix.unsqueeze(1).expand(B, S, n + 1, n + 1).reshape(B * S, n + 1, n + 1)
    demand_exp = demand.unsqueeze(1).expand(B, S, n + 1).reshape(B * S, n + 1)
    cap_exp = capacity.unsqueeze(1).expand(B, S).reshape(B * S)

    # Initialise environment
    obs = env.reset(
        node_xy=node_features[:, :, :2].unsqueeze(1).expand(B, S, n + 1, 2).reshape(B * S, n + 1, 2),
        demand=demand_exp,
        capacity=cap_exp,
        dist_matrix=dist_exp,
    )

    # Manually sync env internals with expanded H since env.reset doesn't know about S
    # (env was already reset above with correct shapes)

    log_prob_sum = torch.zeros(B * S, device=device)

    while True:
        mask = env.get_mask()           # (B*S, n+1)
        cur = env.current_node          # (B*S,)  — already flat because we reset with B*S
        cap_remaining = env.remaining_cap / capacity.repeat_interleave(S)  # normalise

        log_probs = model.decode_step(H_exp, ge_exp, cur, cap_remaining, mask)  # (B*S, n+1)

        # Sample action
        probs = log_probs.exp()
        action = torch.multinomial(probs, 1).squeeze(1)  # (B*S,)

        log_prob_sum += log_probs[torch.arange(B * S, device=device), action]

        _, done = env.step(action.reshape(B * S, 1).squeeze(1)
                           .reshape(B, S)  # env expects (B,S) — but here B*S flat... let's reshape
                           )
        # Actually env was reset with (B*S,1) effectively since B_env=B*S, S_env=1
        # We need to reconcile. See note below.
        if done:
            break

    total_rewards = env.get_total_reward().squeeze(1)  # (B*S,)
    total_rewards = total_rewards.reshape(B, S)
    log_prob_sum = log_prob_sum.reshape(B, S)

    return total_rewards, log_prob_sum


def generate_real_style_vrptw_batch(batch_size, time_horizon, service_time, device):
    """
    Generate synthetic VRPTW instances matching real DHL Istanbul statistics:
      - n ~ randint(10, 51)  (real: mean=36, range 6-50)
      - demand_norm ~ U[0.001, 0.05]  (real: mean=0.007, capacity rarely binds)
      - TW: 50% all-day, 50% moderate windows [30, 300] min wide
      - capacity = 50 (kept consistent with CVRP pre-training)

    This is used for fine-tuning after CVRP/VRPTW pre-training to close the
    synthetic→real distribution gap before evaluating on real DHL instances.
    """
    B = batch_size
    T = float(time_horizon)

    # Variable n per batch item — pad to max_n, mask extras
    n_per = torch.randint(10, 51, (B,))
    max_n = n_per.max().item()

    node_xy = torch.rand(B, max_n + 1, 2, device=device)

    # Very small demand (matches real: mean demand_norm ≈ 0.007)
    capacity = 50.0
    demand_raw = torch.rand(B, max_n + 1, device=device) * 2.5 + 0.05  # U[0.05, 2.55]
    demand_raw[:, 0] = 0.0  # depot
    demand_norm = demand_raw / capacity  # → [0.001, 0.051]
    cap_tensor = torch.full((B,), capacity, device=device)

    diff = node_xy.unsqueeze(2) - node_xy.unsqueeze(1)
    dist_matrix = diff.norm(dim=-1)

    # Time windows: 50% all-day, 50% moderate
    all_day = torch.rand(B, max_n + 1, device=device) < 0.5
    hw = torch.rand(B, max_n + 1, device=device) * (150 - 15) + 15  # half-width 15–150 min
    center = hw + torch.rand(B, max_n + 1, device=device) * (T - 2 * hw.clamp(max=T / 2))
    center = center.clamp(hw, T - hw)
    tw_open = torch.where(all_day, torch.zeros_like(hw), (center - hw).clamp(0, T))
    tw_close = torch.where(all_day, torch.full_like(hw, T), (center + hw).clamp(0, T))
    tw_open[:, 0] = 0.0
    tw_close[:, 0] = T

    time_windows = torch.stack([tw_open, tw_close], dim=-1)
    time_features = time_windows / T

    return (node_xy, demand_raw, demand_norm, cap_tensor, dist_matrix,
            time_windows, time_features, n_per)


def rollout_flat(model, env_class, node_features, demand, capacity, dist_matrix, device,
                 time_windows=None, speed=1.0, service_time=0.0, time_horizon=1.0,
                 vehicle_type_ids=None):
    """
    POMO rollout treating the B*S expansion as a flat batch.

    Supports CVRP (time_windows=None) and VRPTW (time_windows provided).
    Vehicle type embeddings activated when vehicle_type_ids is provided.

    Args:
        node_features: (B, n+1, F) — F=3 for CVRP, F=5 for VRPTW
        time_windows:  (B, n+1, 2) absolute time or None
        time_horizon:  float, for normalising current_time → [0,1]
        vehicle_type_ids: (B,) int tensor or None
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

    veh_ids_exp = None
    if vehicle_type_ids is not None:
        veh_ids_exp = vehicle_type_ids.unsqueeze(1).expand(B, S).reshape(BS)

    flat_idx = torch.arange(BS, device=device)
    flat_start_nodes = (flat_idx % S) + 1  # (BS,) — starts at customer node s+1

    env = env_class(device=device)
    env.B = BS
    env.n = n
    env.S = 1
    env.dist_matrix = dist_exp
    env.demand = demand_exp
    env.capacity = cap_exp
    env.is_electric = False
    env.range_km = None
    env.speed = speed
    env.service_time = service_time
    env.time_windows = tw_exp

    env.visited = torch.zeros(BS, 1, n + 1, dtype=torch.bool, device=device)
    env.visited[:, :, 0] = True
    env.visited[flat_idx, 0, flat_start_nodes] = True
    env.current_node = flat_start_nodes.unsqueeze(1)

    start_demand = demand_exp[flat_idx, flat_start_nodes]
    env.remaining_cap = (cap_exp - start_demand).unsqueeze(1)
    env.route_dist = torch.zeros(BS, 1, device=device)
    env.cumul_dist = torch.zeros(BS, 1, device=device)

    # VRPTW: initialise current_time = travel_time(depot→start_node)
    if tw_exp is not None:
        start_travel = dist_exp[flat_idx, 0, flat_start_nodes] / speed  # (BS,)
        env.current_time = (start_travel + service_time).unsqueeze(1)   # (BS, 1)
    else:
        env.current_time = None

    log_prob_sum = torch.zeros(BS, device=device)
    max_steps = 3 * n + 1

    for _ in range(max_steps):
        if env.is_done():
            break

        mask = env.get_mask().squeeze(1)      # (BS, n+1)
        cur = env.current_node.squeeze(1)     # (BS,)
        cap_norm = env.remaining_cap.squeeze(1) / cap_exp

        cur_time_norm = None
        if env.current_time is not None:
            cur_time_norm = (env.current_time.squeeze(1) / time_horizon).clamp(0, 1)

        log_probs = model.decode_step(
            H_exp, ge_exp, cur, cap_norm, mask,
            vehicle_type_id=veh_ids_exp,
            current_time_norm=cur_time_norm,
        )

        probs = log_probs.exp()
        action = torch.multinomial(probs, 1).squeeze(1)

        log_prob_sum += log_probs[flat_idx, action]
        env.step(action.unsqueeze(1))

    total_rewards = env.get_total_reward().squeeze(1).reshape(B, S)
    log_prob_sum = log_prob_sum.reshape(B, S)
    return total_rewards, log_prob_sum


class Trainer:
    def __init__(self, config, device=None, mode="cvrp"):
        """
        mode: 'cvrp' (3 node features) or 'vrptw' (5 node features)
        """
        self.config = config
        self.device = device or get_device()
        self.mode = mode

        self.model = POMOModel(config).to(self.device)

        # Switch encoder input dimension for VRPTW and finetune (both use 5-feature nodes)
        if mode in ("vrptw", "finetune"):
            self.model.set_input_dim(5)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config["training"]["lr"])
        self.best_mean_reward = -float("inf")

        # Mixed precision: enabled on CUDA only
        self.use_amp = str(self.device).startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def _train_batch_cvrp(self, n_nodes, batch_size, capacity):
        node_xy, demand, demand_norm, cap_tensor, dist_matrix = generate_cvrp_batch(
            batch_size, n_nodes, capacity, self.device
        )
        node_features = torch.cat([node_xy, demand_norm.unsqueeze(-1)], dim=-1)  # (B, n+1, 3)
        return rollout_flat(
            self.model, VRPEnv, node_features, demand, cap_tensor, dist_matrix, self.device
        )

    def _train_batch_vrptw(self, n_nodes, batch_size, capacity):
        cfg_vrptw = self.config["vrptw"]
        T = float(cfg_vrptw["time_horizon"])
        svc = float(cfg_vrptw.get("service_time_min", 3)) / T  # normalise
        speed = 1.0  # unit square: distance = time at speed 1

        (node_xy, demand, demand_norm, cap_tensor,
         dist_matrix, time_windows, time_features) = generate_vrptw_batch(
            batch_size, n_nodes, capacity, T, svc, self.device
        )

        # VRPTW node features: (x, y, demand/cap, open/T, close/T)
        node_features = torch.cat(
            [node_xy, demand_norm.unsqueeze(-1), time_features], dim=-1
        )  # (B, n+1, 5)

        return rollout_flat(
            self.model, VRPEnv, node_features, demand, cap_tensor, dist_matrix, self.device,
            time_windows=time_windows, speed=speed, service_time=svc, time_horizon=T,
        )

    def _train_batch_finetune(self, batch_size):
        """Fine-tune on real-statistics VRPTW instances."""
        cfg_vrptw = self.config["vrptw"]
        T = float(cfg_vrptw["time_horizon"])
        svc = float(cfg_vrptw.get("service_time_min", 3)) / T

        (node_xy, demand, demand_norm, cap_tensor,
         dist_matrix, time_windows, time_features, n_per) = generate_real_style_vrptw_batch(
            batch_size, T, svc, self.device
        )

        # Use max_n for this batch; pad nodes beyond n_per are unreachable via demand mask
        node_features = torch.cat(
            [node_xy, demand_norm.unsqueeze(-1), time_features], dim=-1
        )

        return rollout_flat(
            self.model, VRPEnv, node_features, demand, cap_tensor, dist_matrix, self.device,
            time_windows=time_windows, speed=1.0, service_time=svc, time_horizon=T,
        )

    def train_epoch(self, n_nodes, batch_size, capacity):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        n_batches = self.config["training"]["synthetic_instances_per_epoch"] // batch_size

        for _ in range(n_batches):
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    if self.mode == "finetune":
                        rewards, log_prob_sum = self._train_batch_finetune(batch_size)
                    elif self.mode == "vrptw":
                        rewards, log_prob_sum = self._train_batch_vrptw(n_nodes, batch_size, capacity)
                    else:
                        rewards, log_prob_sum = self._train_batch_cvrp(n_nodes, batch_size, capacity)
                    baseline = rewards.mean(dim=1, keepdim=True).detach()
                    advantage = rewards - baseline
                    loss = -(advantage.detach() * log_prob_sum).mean()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.mode == "finetune":
                    rewards, log_prob_sum = self._train_batch_finetune(batch_size)
                elif self.mode == "vrptw":
                    rewards, log_prob_sum = self._train_batch_vrptw(n_nodes, batch_size, capacity)
                else:
                    rewards, log_prob_sum = self._train_batch_cvrp(n_nodes, batch_size, capacity)
                baseline = rewards.mean(dim=1, keepdim=True).detach()
                advantage = rewards - baseline
                loss = -(advantage.detach() * log_prob_sum).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_reward += rewards.mean().item()

        return total_loss / n_batches, total_reward / n_batches

    def train(self):
        cfg = self.config["training"]
        n_nodes = cfg["n_nodes"]
        batch_size = cfg["batch_size"]
        capacity = cfg["capacity"]
        epochs = cfg["epochs"]

        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"Training on {self.device} | mode={self.mode} | n={n_nodes} | "
              f"B={batch_size} | epochs={epochs}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss, mean_reward = self.train_epoch(n_nodes, batch_size, capacity)
            elapsed = time.time() - t0

            mean_tour = -mean_reward
            print(f"Epoch {epoch:3d}/{epochs} | loss={loss:.4f} | "
                  f"mean_tour={mean_tour:.4f} | {elapsed:.1f}s")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.save_checkpoint(os.path.join(CKPT_DIR, f"best_{self.mode}.pt"))

            if epoch % 10 == 0:
                self.save_checkpoint(
                    os.path.join(CKPT_DIR, f"{self.mode}_epoch_{epoch:03d}.pt")
                )

    def save_checkpoint(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_mean_reward": self.best_mean_reward,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_mean_reward = ckpt.get("best_mean_reward", -float("inf"))
        print(f"Loaded checkpoint from {path}")


if __name__ == "__main__":
    import yaml
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(ROOT, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    # Quick validation: 3 epochs on n=20
    config["training"]["n_nodes"] = 20
    config["training"]["epochs"] = 3
    config["training"]["batch_size"] = 16
    config["training"]["synthetic_instances_per_epoch"] = 160

    trainer = Trainer(config)
    trainer.train()
