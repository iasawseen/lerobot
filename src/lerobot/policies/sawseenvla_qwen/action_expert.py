"""ActionExpertDecoder: small independent transformer.

Each layer does:
  1. Self-attention over the action-chunk sequence (with RoPE on positions 0..chunk-1).
  2. Cross-attention to Qwen anchor hidden states (own k_proj/v_proj, no RoPE on prefix).
  3. SwiGLU MLP.

Layer i cross-attends to `anchors[i]` (one per expert layer; matched by index).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def apply_rope_1d(x: Tensor, positions: Tensor, max_wavelength: float = 10000.0) -> Tensor:
    """Apply standard 1D RoPE to (B, L, H, D). Rotates the full head dim (no partial RoPE)."""
    B, L, H, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"head_dim must be even, got {D}")
    dtype = x.dtype
    inv_freq = 1.0 / (max_wavelength ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    sinusoid = positions[:, :, None, None].float() * inv_freq[None, None, None, :]  # (B,L,1,D/2)
    sin = sinusoid.sin().to(dtype)
    cos = sinusoid.cos().to(dtype)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., dim)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)
        positions = torch.arange(L, device=x.device, dtype=torch.long)[None, :].expand(B, L)
        q = apply_rope_1d(q, positions)
        k = apply_rope_1d(k, positions)
        # (B, H, L, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # Full attention within the chunk (action tokens see each other, no causality).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention from action tokens to Qwen hidden states.

    Maintains its own k_proj/v_proj that project Qwen hidden states (dim=vlm_hidden)
    into expert space. No positional encoding on the prefix.
    """

    def __init__(self, expert_hidden: int, n_heads: int, vlm_hidden: int):
        super().__init__()
        assert expert_hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = expert_hidden // n_heads
        self.q_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)
        self.k_proj = nn.Linear(vlm_hidden, expert_hidden, bias=False)
        self.v_proj = nn.Linear(vlm_hidden, expert_hidden, bias=False)
        self.o_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)

    def forward(self, x: Tensor, prefix: Tensor, prefix_mask: Tensor) -> Tensor:
        # x:      (B, L_q, expert_hidden)
        # prefix: (B, L_k, vlm_hidden)
        # prefix_mask: (B, L_k) — 1 = valid
        B, Lq, _ = x.shape
        Lk = prefix.shape[1]
        q = self.q_proj(x).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Lq, D)
        k = self.k_proj(prefix).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(prefix).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        # attn_mask: (B, 1, 1, Lk) — broadcast to (B, H, Lq, Lk), True = valid
        attn_mask = prefix_mask.to(torch.bool)[:, None, None, :]
        # F.scaled_dot_product_attention expects a boolean mask where True allows.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, Lq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class ActionExpertLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        vlm_hidden: int,
        intermediate: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.self_attn = SelfAttention(hidden, n_heads)
        self.norm2 = RMSNorm(hidden)
        self.cross_attn = CrossAttention(hidden, n_heads, vlm_hidden)
        self.norm3 = RMSNorm(hidden)
        self.mlp = SwiGLU(hidden, intermediate)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, prefix: Tensor, prefix_mask: Tensor) -> Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.cross_attn(self.norm2(x), prefix, prefix_mask))
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ActionExpertDecoder(nn.Module):
    """Independent transformer that consumes (noisy action chunk, time, anchor hidden states).

    n_layers == len(anchor_layer_indices) — each layer cross-attends to one anchor.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        n_layers: int,
        hidden: int,
        n_heads: int,
        intermediate: int,
        vlm_hidden: int,
        dropout: float = 0.0,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden = hidden
        self.min_period = min_period
        self.max_period = max_period

        self.action_in_proj = nn.Linear(action_dim, hidden)
        self.time_mlp_in = nn.Linear(hidden * 2, hidden)
        self.time_mlp_out = nn.Linear(hidden, hidden)
        self.layers = nn.ModuleList(
            [
                ActionExpertLayer(hidden, n_heads, vlm_hidden, intermediate, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden)
        self.action_out_proj = nn.Linear(hidden, action_dim)

    def _time_embedding(self, time: Tensor) -> Tensor:
        """time: (B,) → (B, hidden) sinusoidal embedding (log-spaced periods)."""
        B = time.shape[0]
        if self.hidden % 2 != 0:
            raise ValueError(f"hidden must be even for sin/cos time embedding, got {self.hidden}")
        device = time.device
        fraction = torch.linspace(0.0, 1.0, self.hidden // 2, device=device, dtype=torch.float64)
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        scaling = 1.0 / period * 2.0 * math.pi
        sin_input = scaling[None, :] * time[:, None].double()
        emb = torch.cat([sin_input.sin(), sin_input.cos()], dim=-1).to(time.dtype)
        return emb  # (B, hidden)

    def forward(
        self,
        x_t: Tensor,
        time: Tensor,
        anchors: list[Tensor],
        prefix_mask: Tensor,
    ) -> Tensor:
        """
        x_t:        (B, chunk, action_dim)
        time:       (B,)
        anchors:    list of (B, L_prefix, vlm_hidden), one per layer
        prefix_mask: (B, L_prefix)
        returns:    (B, chunk, action_dim) velocity prediction
        """
        if len(anchors) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} anchor hidden states, got {len(anchors)}"
            )
        B, L, _ = x_t.shape
        a = self.action_in_proj(x_t)  # (B, L, hidden)
        t_emb = self._time_embedding(time)  # (B, hidden)
        t_emb_b = t_emb[:, None, :].expand(-1, L, -1)
        h = self.time_mlp_out(F.silu(self.time_mlp_in(torch.cat([a, t_emb_b], dim=-1))))

        for layer, prefix in zip(self.layers, anchors, strict=True):
            h = layer(h, prefix, prefix_mask)
        h = self.final_norm(h)
        return self.action_out_proj(h)
