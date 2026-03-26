"""
Mixture-of-Recursions (MoR) — Self-contained PyTorch implementation.
Paper: https://arxiv.org/abs/2507.10524

Drop-in module: wrap any standard transformer decoder layer to get
parameter-shared recursive blocks with adaptive per-token routing.

Two routing modes:
  1. Expert-choice: router scores every token at each recursion step,
     top-k tokens proceed (hierarchical filtering).
  2. Token-choice: single routing decision assigns each token to a
     recursion depth upfront.

Usage:
    base_layer = YourTransformerDecoderLayer(...)
    # Expert-choice with 3 recursions, 50% capacity
    mor = MoRExpertChoice(base_layer, hidden_dim=768, num_recursions=3, capacity=0.5)
    # Token-choice with 3 recursions
    mor = MoRTokenChoice(base_layer, hidden_dim=768, num_recursions=3)

    # In your forward loop, replace:
    #   h = layer(h, **kwargs)
    # with:
    #   h, aux = mor(h, **kwargs)
    # Add aux["loss"] to your main loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Router architectures
# ---------------------------------------------------------------------------

class LinearRouter(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPRouter(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, out_dim, bias=False),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MoROutput:
    hidden_states: torch.Tensor
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def loss(self) -> torch.Tensor:
        vals = list(self.aux_losses.values())
        if not vals:
            return torch.tensor(0.0, device=self.hidden_states.device)
        return sum(vals)


# ---------------------------------------------------------------------------
# Expert-Choice MoR
# ---------------------------------------------------------------------------

class MoRExpertChoice(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        hidden_dim: int,
        num_recursions: int = 3,
        capacity: float = 0.5,
        router_type: str = "linear",
        router_temp: float = 1.0,
        alpha: float = 1.0,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.0,
        cap_warmup_steps: int = 0,
    ):
        super().__init__()
        self.block = block
        self.num_recursions = num_recursions
        self.capacity = capacity
        self.alpha = alpha
        self.router_temp = router_temp
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.cap_warmup_steps = cap_warmup_steps

        Router = LinearRouter if router_type == "linear" else MLPRouter
        self.routers = nn.ModuleList([Router(hidden_dim, 1) for _ in range(num_recursions)])
        self._step = 0

    def _get_capacity(self) -> float:
        if self.training and self.cap_warmup_steps > 0:
            self._step += 1
            ratio = min(1.0, self._step / self.cap_warmup_steps)
            decay = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return self.capacity + (1.0 - self.capacity) * decay
        return self.capacity

    def forward(self, x: torch.Tensor, **block_kwargs) -> MoROutput:
        B, S, D = x.shape
        cap = self._get_capacity()
        top_k = max(1, int(cap * S))

        total = x
        active = x
        active_indices = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        aux = {}
        total_sampling_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)

        for r in range(self.num_recursions):
            cur_len = active.shape[1]
            k = max(1, min(top_k, cur_len))

            raw_logits = self.routers[r](active / self.router_temp)
            weights = torch.sigmoid(raw_logits) * self.alpha

            _, topk_idx = torch.topk(weights.squeeze(-1), k, dim=1, sorted=False)
            topk_idx, _ = torch.sort(topk_idx, dim=1)

            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
            selected = torch.gather(active, 1, idx_exp)
            sel_weights = torch.gather(weights, 1, topk_idx.unsqueeze(-1))

            processed = self.block(selected, **block_kwargs)
            if isinstance(processed, tuple):
                processed = processed[0]

            global_idx = torch.gather(active_indices, 1, topk_idx)
            global_idx_exp = global_idx.unsqueeze(-1).expand(-1, -1, D)
            total = torch.scatter_add(total, 1, global_idx_exp, processed * sel_weights)

            active = selected
            active_indices = global_idx

            if self.training and self.aux_loss_weight > 0:
                targets = torch.zeros(B, cur_len, 1, device=x.device, dtype=x.dtype)
                targets.scatter_(1, topk_idx.unsqueeze(-1), 1.0)
                sampling_loss = F.binary_cross_entropy_with_logits(
                    raw_logits.view(-1), targets.view(-1), reduction="sum"
                ) / (B * cur_len)
                total_sampling_loss = total_sampling_loss + sampling_loss

            if self.training and self.z_loss_weight > 0:
                z = torch.logsumexp(raw_logits.squeeze(-1), dim=-1)
                total_z_loss = total_z_loss + (z ** 2).mean()

        if self.aux_loss_weight > 0:
            aux["sampling_loss"] = self.aux_loss_weight * total_sampling_loss
        if self.z_loss_weight > 0:
            aux["z_loss"] = self.z_loss_weight * total_z_loss

        return MoROutput(hidden_states=total, aux_losses=aux)


# ---------------------------------------------------------------------------
# Token-Choice MoR
# ---------------------------------------------------------------------------

class MoRTokenChoice(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        hidden_dim: int,
        num_recursions: int = 3,
        router_type: str = "linear",
        router_temp: float = 1.0,
        alpha: float = 1.0,
        balancing: str = "loss",
        balancing_weight: float = 0.01,
        z_loss_weight: float = 0.0,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.block = block
        self.num_recursions = num_recursions
        self.router_temp = router_temp
        self.alpha = alpha
        self.balancing = balancing
        self.balancing_weight = balancing_weight
        self.z_loss_weight = z_loss_weight
        self.warmup_steps = warmup_steps

        Router = LinearRouter if router_type == "linear" else MLPRouter
        self.router = Router(hidden_dim, num_recursions)

        if balancing == "loss_free":
            self.register_buffer("router_bias", torch.zeros(num_recursions))
        self._step = 0

    def forward(self, x: torch.Tensor, **block_kwargs) -> MoROutput:
        B, S, D = x.shape
        self._step += 1

        logits = self.router(x / self.router_temp)
        probs = F.softmax(logits, dim=-1) * self.alpha

        if self.balancing == "loss_free":
            routed_probs = probs + self.router_bias
        else:
            routed_probs = probs

        if self.training and self._step < self.warmup_steps:
            depth_idx = torch.full((B, S), self.num_recursions - 1, device=x.device, dtype=torch.long)
        else:
            depth_idx = routed_probs.argmax(dim=-1)

        gate_weights = torch.gather(probs, -1, depth_idx.unsqueeze(-1))

        initial = x
        final = x.clone()
        current = x.clone()

        for r in range(self.num_recursions):
            active_mask = (depth_idx >= r)
            if not active_mask.any():
                break

            processed = self.block(current, **block_kwargs)
            if isinstance(processed, tuple):
                processed = processed[0]

            finishing = (depth_idx == r)
            finishing_3d = finishing.unsqueeze(-1)
            final = torch.where(finishing_3d, gate_weights * processed + initial, final)

            continuing = active_mask & ~finishing
            continuing_3d = continuing.unsqueeze(-1)
            current = torch.where(continuing_3d, processed, current)

        aux = {}
        if self.training:
            if self.balancing == "loss":
                P = probs.sum(dim=(0, 1)) / (B * S)
                counts = torch.zeros(self.num_recursions, device=x.device)
                for i in range(self.num_recursions):
                    counts[i] = (depth_idx == i).sum()
                f = self.num_recursions * counts / (B * S)
                aux["balancing_loss"] = self.balancing_weight * (P * f).sum()
            elif self.balancing == "loss_free":
                with torch.no_grad():
                    counts = torch.zeros(self.num_recursions, device=x.device)
                    for i in range(self.num_recursions):
                        counts[i] = (depth_idx == i).sum()
                    target = B * S / self.num_recursions
                    self.router_bias -= 0.01 * (counts - target) / (B * S)

            if self.z_loss_weight > 0:
                z = torch.logsumexp(logits, dim=-1)
                aux["z_loss"] = self.z_loss_weight * (z ** 2).mean()

        return MoROutput(hidden_states=final, aux_losses=aux)


# ---------------------------------------------------------------------------
# Convenience: wrap a full stack with Middle-Cycle sharing
# ---------------------------------------------------------------------------

def build_mor_stack(
    layers: nn.ModuleList,
    hidden_dim: int,
    num_recursions: int = 3,
    mode: str = "expert",
    sharing: str = "middle_cycle",
    **mor_kwargs,
) -> nn.ModuleList:
    L = len(layers)
    Cls = MoRExpertChoice if mode == "expert" else MoRTokenChoice

    if sharing == "middle_cycle":
        assert (L - 2) % num_recursions == 0
        base_depth = (L - 2) // num_recursions
        new_layers: list[nn.Module] = [layers[0]]
        for i in range(base_depth):
            idxs = [1 + i + r * base_depth for r in range(num_recursions)]
            shared_block = layers[idxs[0]]
            new_layers.append(Cls(shared_block, hidden_dim, num_recursions, **mor_kwargs))
        new_layers.append(layers[L - 1])
        return nn.ModuleList(new_layers)

    elif sharing == "cycle":
        assert L % num_recursions == 0
        base_depth = L // num_recursions
        new_layers = []
        for i in range(base_depth):
            idxs = [i + r * base_depth for r in range(num_recursions)]
            shared_block = layers[idxs[0]]
            new_layers.append(Cls(shared_block, hidden_dim, num_recursions, **mor_kwargs))
        return nn.ModuleList(new_layers)

    raise ValueError(f"Unknown sharing strategy: {sharing}")
