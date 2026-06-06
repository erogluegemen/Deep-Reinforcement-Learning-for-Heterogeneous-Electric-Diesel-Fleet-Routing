"""
POMO model: AttentionEncoder + PomoDecoder.

Encoder:
  - Input: (B, n+1, input_dim)  [node features, row 0 = depot]
  - 6 × Multi-Head Self-Attention layers (Pre-LN) with FFN
  - Output: node embeddings H  (B, n+1, d_model)

Decoder (per step):
  - Context = [graph_embed, H[current_node], cap_embed] → projected to d_model
  - Single MHA attention over H → compatibility scores
  - tanh(C) clipping → masked softmax → log_probs
  - Optional: vehicle_type embedding (for heterogeneous fleet extension)

Shape convention throughout: (B, S, ...) for POMO multi-start batching.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VEHICLE_TYPE_MAP = {"SCV": 0, "MCV": 1, "LCV": 2, "LCV_BEV": 3}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        """
        q: (..., L_q, d_model)
        k, v: (..., L_k, d_model)
        mask: (..., L_q, L_k) bool — True means BLOCK that position
        """
        *batch, L_q, _ = q.shape
        *_, L_k, _ = k.shape
        h = self.n_heads

        Q = self.W_q(q).reshape(*batch, L_q, h, self.d_k).transpose(-3, -2)  # (..., h, L_q, d_k)
        K = self.W_k(k).reshape(*batch, L_k, h, self.d_k).transpose(-3, -2)
        V = self.W_v(v).reshape(*batch, L_k, h, self.d_k).transpose(-3, -2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (..., h, L_q, L_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3), -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (..., h, L_q, d_k)
        out = out.transpose(-3, -2).reshape(*batch, L_q, h * self.d_k)
        return self.W_o(out)


class TransformerLayer(nn.Module):
    """Pre-LN transformer encoder layer."""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x, mask=None):
        # Self-attention with residual
        h = self.ln1(x)
        x = x + self.mha(h, h, h, mask=mask)
        # FFN with residual
        h = self.ln2(x)
        x = x + self.ff2(F.relu(self.ff1(h)))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class AttentionEncoder(nn.Module):
    """
    Graph attention encoder for VRP node features.
    input_dim: 3 for CVRP (x, y, demand/cap), 5 for VRPTW (+open/T, +close/T)
    """
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=6, d_ff=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, n+1, input_dim)
        returns H: (B, n+1, d_model)
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.ln_out(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class PomoDecoder(nn.Module):
    """
    Single-step POMO decoder.

    Context vector = concat(graph_embed, node_embed[current], capacity_embed)
    → project to d_model → single MHA over node embeddings → tanh(C) clipped logits.

    For heterogeneous fleet: vehicle_type embedding is concatenated before projection.
    """
    def __init__(self, d_model=128, n_heads=8, clip_c=10.0,
                 vehicle_embed_dim=16, n_vehicle_types=4):
        super().__init__()
        self.d_model = d_model
        self.clip_c = clip_c

        # Scalar projections for dynamic context components
        self.cap_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(1, d_model)  # normalised current time (0 in CVRP mode)

        # Context: [graph_embed, cur_node_embed, cap_embed, time_embed, (veh_embed)]
        self.use_vehicle_embed = vehicle_embed_dim > 0
        if self.use_vehicle_embed:
            self.vehicle_embed = nn.Embedding(n_vehicle_types, vehicle_embed_dim)
            context_in = 4 * d_model + vehicle_embed_dim
        else:
            context_in = 4 * d_model

        self.context_proj = nn.Linear(context_in, d_model)

        # Single MHA for compatibility
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.W_compat = nn.Linear(d_model, 1, bias=False)

    def forward(self, H, graph_embed, current_node, remaining_cap, mask,
                vehicle_type_id=None, current_time_norm=None):
        """
        H:                 (B*S, n+1, d_model)
        graph_embed:       (B*S, d_model)
        current_node:      (B*S,)
        remaining_cap:     (B*S,)   normalised [0,1]
        mask:              (B*S, n+1)  True = infeasible
        vehicle_type_id:   (B*S,) int or None
        current_time_norm: (B*S,) float [0,1] or None (→ 0 in CVRP mode)

        Returns log_probs: (B*S, n+1)
        """
        BS, _, d = H.shape

        cur_embed = H[torch.arange(BS, device=H.device), current_node]  # (BS, d)
        cap_embed = self.cap_proj(remaining_cap.unsqueeze(-1))           # (BS, d)

        # Time embedding: zeros for CVRP, normalised current_time for VRPTW
        if current_time_norm is not None:
            time_val = current_time_norm.unsqueeze(-1)
        else:
            time_val = torch.zeros(BS, 1, device=H.device)
        time_embed = self.time_proj(time_val)                            # (BS, d)

        ctx_parts = [graph_embed, cur_embed, cap_embed, time_embed]
        if self.use_vehicle_embed:
            if vehicle_type_id is not None:
                veh_embed = self.vehicle_embed(vehicle_type_id)
            else:
                veh_dim = self.vehicle_embed.embedding_dim
                veh_embed = torch.zeros(BS, veh_dim, device=H.device)
            ctx_parts.append(veh_embed)

        context = torch.cat(ctx_parts, dim=-1)           # (BS, context_in)
        context = self.context_proj(context).unsqueeze(1)  # (BS, 1, d_model)

        # Attention over node embeddings
        attn_out = self.mha(context, H, H)  # (BS, 1, d_model)

        # Compatibility via dot-product with each node key
        compat = (attn_out * H).sum(dim=-1) / math.sqrt(d)  # (BS, n+1)

        # Clip with tanh × C
        compat = self.clip_c * torch.tanh(compat)

        # Apply infeasibility mask
        if mask is not None:
            compat = compat.masked_fill(mask, -1e9)

        log_probs = F.log_softmax(compat, dim=-1)
        return log_probs


# ---------------------------------------------------------------------------
# Full POMO model
# ---------------------------------------------------------------------------

class POMOModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["model"]["d_model"]
        n_heads = config["model"]["n_heads"]
        n_layers = config["model"]["n_layers"]
        d_ff = config["model"]["d_ff"]
        clip_c = config["model"]["clip_c"]
        vehicle_embed_dim = config["model"].get("vehicle_embed_dim", 16)

        # input_dim: 3 for CVRP, set at runtime based on features
        self.encoder = AttentionEncoder(3, d_model, n_heads, n_layers, d_ff)
        self.decoder = PomoDecoder(d_model, n_heads, clip_c, vehicle_embed_dim)
        self.d_model = d_model

    def encode(self, node_features):
        """
        node_features: (B, n+1, input_dim)
        Returns H: (B, n+1, d_model), graph_embed: (B, d_model)
        """
        H = self.encoder(node_features)
        graph_embed = H.mean(dim=1)  # (B, d_model)
        return H, graph_embed

    def decode_step(self, H, graph_embed, current_node, remaining_cap, mask,
                    vehicle_type_id=None, current_time_norm=None):
        """
        Decode one step. All inputs already in (B*S, ...) form.
        Returns log_probs: (B*S, n+1)
        """
        return self.decoder(H, graph_embed, current_node, remaining_cap, mask,
                            vehicle_type_id, current_time_norm)

    def set_input_dim(self, input_dim):
        """Reinitialise encoder input projection for a different feature size."""
        old = self.encoder.input_proj
        if old.in_features == input_dim:
            return
        device = next(self.parameters()).device
        self.encoder.input_proj = nn.Linear(input_dim, self.d_model).to(device)


if __name__ == "__main__":
    import yaml, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(ROOT, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    device = "cpu"
    model = POMOModel(config).to(device)

    B, n = 4, 10  # batch=4, 10 customer nodes
    S = n          # POMO starts

    # Dummy node features: (x, y, demand/cap)
    node_features = torch.rand(B, n + 1, 3)
    node_features[:, :, 2] = node_features[:, :, 2] * 9 / 50  # demand normalised
    node_features[:, 0, 2] = 0.0  # depot demand = 0

    H, graph_embed = model.encode(node_features)
    print(f"H shape: {H.shape}")           # (4, 11, 128)
    print(f"graph_embed shape: {graph_embed.shape}")  # (4, 128)

    # Expand for S starts
    H_exp = H.unsqueeze(1).expand(B, S, n + 1, -1).reshape(B * S, n + 1, -1)
    ge_exp = graph_embed.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)

    current = torch.arange(1, S + 1).unsqueeze(0).expand(B, S).reshape(B * S)
    remaining = torch.ones(B * S)  # full capacity
    mask = torch.zeros(B * S, n + 1, dtype=torch.bool)

    log_probs = model.decode_step(H_exp, ge_exp, current, remaining, mask)
    print(f"log_probs shape: {log_probs.shape}")  # (40, 11)
    print(f"log_probs exp sum (should ≈ 1): {log_probs.exp().sum(dim=-1)[:3]}")
