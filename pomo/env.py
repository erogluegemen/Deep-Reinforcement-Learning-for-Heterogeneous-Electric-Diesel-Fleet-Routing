"""
VRPEnv: batched environment for CVRP and VRPTW.

Batch dimensions: (B, S, ...) where B=batch_size, S=n_starts (POMO).
Depot is always node index 0. Customer nodes are indices 1..n.

CVRP mode (time_windows=None):
  - Mask: visited | capacity_infeasible | (at_depot and no need to reload)
  - State: current_node, remaining_cap, visited, route_dist

VRPTW mode (time_windows provided):
  - Additional mask: earliest_arrival > close[j]
  - Additional state: current_time (minutes or normalized units)
  - earliest_arrival = current_time + dist(cur,j)/speed
  - Going to depot resets capacity and optionally current_time (depot open all day)

EV mode (is_electric=True, range_km set):
  - Additional mask: cumul_dist + dist(cur,j) + dist(j,depot) > range_km
"""

import torch

VEHICLE_TYPE_MAP = {"SCV": 0, "MCV": 1, "LCV": 2, "LCV_BEV": 3}


class VRPEnv:
    def __init__(self, device="cpu"):
        self.device = device
        self.time_windows = None  # None = CVRP mode
        self.speed = 1.0
        self.service_time = 0.0
        self.current_time = None  # set to tensor in reset() for VRPTW; None = CVRP mode

    def reset(self, node_xy, demand, capacity, dist_matrix,
              time_windows=None, speed=1.0, service_time=0.0,
              is_electric=False, range_km=None):
        """
        Initialise a batch of instances with POMO multiple starts.

        Args:
            node_xy:       (B, n+1, 2)     normalised coordinates, row 0 = depot
            demand:        (B, n+1)         demand per node (depot = 0)
            capacity:      (B,)             vehicle capacity
            dist_matrix:   (B, n+1, n+1)   pairwise distances
            time_windows:  (B, n+1, 2) or None
                             [:, j, 0] = open time,  [:, j, 1] = close time
                             None → CVRP mode (no time constraint)
            speed:         float, distance-units per time-unit
            service_time:  float, fixed service time added after each customer visit
            is_electric:   bool, enables EV range constraint
            range_km:      float or None, max cumulative distance for EV
        """
        B, n1, _ = node_xy.shape
        n = n1 - 1
        S = n  # POMO: one start per customer node

        self.B = B
        self.n = n
        self.S = S
        self.dist_matrix = dist_matrix
        self.demand = demand
        self.capacity = capacity
        self.time_windows = time_windows
        self.speed = speed
        self.service_time = service_time
        self.is_electric = is_electric
        self.range_km = range_km

        # --- Visited mask: (B, S, n+1) ---
        self.visited = torch.zeros(B, S, n + 1, dtype=torch.bool, device=self.device)
        self.visited[:, :, 0] = True  # depot marked visited (logic handled via mask)

        # Each start s begins at customer node s+1
        start_nodes = torch.arange(1, n + 1, device=self.device).unsqueeze(0).expand(B, S)
        self.current_node = start_nodes.clone()  # (B, S)

        # Vectorised: mark start node as visited for each start
        # visited[b, s, s+1] = True for all b, s
        s_idx = torch.arange(S, device=self.device)
        self.visited[:, s_idx, s_idx + 1] = True  # (B, S, n+1) via fancy indexing

        # Remaining capacity (after serving the starting node)
        cap_expand = capacity.unsqueeze(1).expand(B, S)           # (B, S)
        start_demand = demand[:, 1:n + 1]                         # (B, S) — demands of nodes 1..n
        self.remaining_cap = cap_expand - start_demand            # (B, S)

        # Accumulated route distance (for reward)
        self.route_dist = torch.zeros(B, S, device=self.device)

        # EV cumulative distance (for range check)
        self.cumul_dist = torch.zeros(B, S, device=self.device)

        # VRPTW: current time (after serving the starting node)
        # Starts at 0 + travel_time(depot→start_node) + service_time
        # But POMO starts at a customer, not depot. We treat starting time as 0
        # (the vehicle departs depot at time 0, arrives at start_node at travel_time).
        if time_windows is not None:
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S)
            start_travel = dist_matrix[b_idx, 0, start_nodes] / speed  # (B, S)
            self.current_time = start_travel + service_time            # (B, S)
        else:
            self.current_time = None

        return self._get_obs()

    def _get_obs(self):
        obs = {
            "current_node": self.current_node,
            "remaining_cap": self.remaining_cap,
            "visited": self.visited,
        }
        if self.current_time is not None:
            obs["current_time"] = self.current_time
        return obs

    def get_mask(self):
        """
        Returns infeasibility mask: (B, S, n+1)  True = infeasible.

        CVRP rules:
          - Visited customer nodes
          - demand[j] > remaining_cap
          - depot→depot blocked (prevent 0-cost self-loop exploit)

        VRPTW additions:
          - current_time + dist(cur,j)/speed > close[j]

        EV addition:
          - cumul_dist + dist(cur,j) + dist(j,depot) > range_km
        """
        B, S, n1 = self.visited.shape

        mask = self.visited.clone()  # (B, S, n+1)

        # --- Capacity constraint ---
        demand_exp = self.demand.unsqueeze(1).expand(B, S, n1)
        cap_exp = self.remaining_cap.unsqueeze(2).expand(B, S, n1)
        cap_infeasible = demand_exp > cap_exp
        cap_infeasible[:, :, 0] = False
        mask = mask | cap_infeasible

        # --- VRPTW time window constraint ---
        if self.time_windows is not None:
            cur = self.current_node  # (B, S)
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S)

            dist_cur_j = self.dist_matrix[b_idx, cur, :]          # (B, S, n+1)
            travel_time = dist_cur_j / self.speed                  # (B, S, n+1)
            arrival = self.current_time.unsqueeze(2) + travel_time  # (B, S, n+1)

            close_times = self.time_windows[:, :, 1]                # (B, n+1)
            close_exp = close_times.unsqueeze(1).expand(B, S, n1)   # (B, S, n+1)
            tw_infeasible = arrival > close_exp
            tw_infeasible[:, :, 0] = False  # depot always reachable in time
            mask = mask | tw_infeasible

        # --- EV range constraint ---
        if self.is_electric and self.range_km is not None:
            cur = self.current_node
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S)
            dist_cur_j = self.dist_matrix[b_idx, cur, :]          # (B, S, n+1)
            dist_j_dep = self.dist_matrix[:, :, 0].unsqueeze(1).expand(B, S, n1)
            cumul_exp = self.cumul_dist.unsqueeze(2).expand(B, S, n1)
            rk = float(self.range_km)
            range_infeasible = (cumul_exp + dist_cur_j + dist_j_dep) > rk
            range_infeasible[:, :, 0] = False
            mask = mask | range_infeasible

        # --- Depot rule: blocked when currently at depot (prevents 0-cost loop) ---
        at_depot = (self.current_node == 0)
        mask[:, :, 0] = at_depot

        # Safety: if all customers are infeasible, force depot (no -inf softmax)
        all_cust_blocked = mask[:, :, 1:].all(dim=-1)
        mask[:, :, 0] = mask[:, :, 0] & ~all_cust_blocked

        return mask

    def step(self, action):
        """
        action: (B, S) int — chosen node indices.

        Returns:
            step_reward: (B, S)  negative travel distance for this step
            done:        bool
        """
        B, S = action.shape
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S)

        dist_step = self.dist_matrix[b_idx, self.current_node, action]  # (B, S)
        self.route_dist += dist_step
        self.cumul_dist += dist_step

        going_to_depot = (action == 0)

        # Capacity update (reload at depot)
        demand_taken = self.demand[b_idx, action]
        new_cap = self.remaining_cap - demand_taken
        new_cap = torch.where(going_to_depot,
                              self.capacity.unsqueeze(1).expand(B, S),
                              new_cap)
        self.remaining_cap = new_cap

        # EV: reset cumulative distance at depot
        self.cumul_dist = torch.where(going_to_depot,
                                      torch.zeros_like(self.cumul_dist),
                                      self.cumul_dist)

        # VRPTW: update current time
        if self.current_time is not None:
            travel_time = dist_step / self.speed
            new_time = self.current_time + travel_time
            # Respect time window open (wait if arriving early)
            if self.time_windows is not None:
                open_times = self.time_windows[b_idx, action, 0]  # (B, S)
                new_time = torch.maximum(new_time, open_times)
            # Add service time (only at customer nodes)
            new_time = new_time + torch.where(going_to_depot,
                                              torch.zeros_like(new_time),
                                              torch.full_like(new_time, self.service_time))
            # At depot: reset time to 0 (depot available all day)
            new_time = torch.where(going_to_depot, torch.zeros_like(new_time), new_time)
            self.current_time = new_time

        # Visited update (vectorised)
        new_visited = self.visited.clone()
        cust_action = action.masked_fill(going_to_depot, 0)  # map depot actions to 0 (no-op)
        new_visited.scatter_(2, cust_action.unsqueeze(2), True)
        new_visited[:, :, 0] = True  # depot stays "visited" sentinel
        self.visited = new_visited

        self.current_node = action

        return -dist_step, self.is_done()

    def is_done(self):
        return self.visited[:, :, 1:].all()

    def get_total_reward(self):
        """Negative total route distance including final return-to-depot."""
        b_idx = torch.arange(self.B, device=self.device).unsqueeze(1).expand(self.B, self.S)
        final_leg = self.dist_matrix[b_idx, self.current_node, 0]
        return -(self.route_dist + final_leg)


if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    device = "cpu"
    B, n = 2, 5

    xy = torch.rand(B, n + 1, 2)
    demand = torch.zeros(B, n + 1)
    demand[:, 1:] = torch.rand(B, n) * 8 + 1
    capacity = torch.full((B,), 50.0)
    diff = xy.unsqueeze(2) - xy.unsqueeze(1)
    dist_matrix = diff.norm(dim=-1)

    # VRPTW: random time windows in [0, 1] time horizon
    T = 1.0
    tw_center = torch.rand(B, n + 1) * T
    tw_half = torch.rand(B, n + 1) * 0.3 + 0.1
    time_windows = torch.stack([
        (tw_center - tw_half).clamp(0, T),
        (tw_center + tw_half).clamp(0, T),
    ], dim=-1)
    time_windows[:, 0, :] = torch.tensor([0.0, T])  # depot open all day

    env = VRPEnv(device=device)
    env.reset(xy, demand, capacity, dist_matrix,
              time_windows=time_windows, speed=1.0, service_time=0.05)

    print(f"VRPTW mode — current_time shape: {env.current_time.shape}")
    print(f"Visited shape: {env.visited.shape}, mask shape: {env.get_mask().shape}")

    for step_i in range(12):
        if env.is_done():
            break
        mask = env.get_mask()
        logits = torch.where(mask, torch.tensor(-1e9), torch.rand(mask.shape))
        action = logits.argmax(dim=-1)
        _, done = env.step(action)
        print(f"Step {step_i+1}: action={action[0].tolist()} time={env.current_time[0].tolist()}")

    print(f"Total reward: {env.get_total_reward()}")
