"""
Cross-attention module for DynaMo Phase 1.

Fuses structure and dynamics via asymmetric cross-attention:
  - Structure path (H_geom) generates queries
  - Dynamics path (H_star) generates keys and values
  - This enforces "structure asks, dynamics answers" design

Multi-head scaled dot-product attention with residual connections.

Usage:
    cross_attn = StructureDynamicsCrossAttention(d_model=256, n_heads=8)
    H_fused = cross_attn(H_geom, H_star)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureDynamicsCrossAttention(nn.Module):
    """
    Cross-attention where structure queries and dynamics provides key-value.
    
    Asymmetry enforced by design:
      - H_geom (geometry path) → Q only
      - H_star (dynamics path) → K, V only
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # ── Asymmetric projections: Q from structure, K/V from dynamics ─────────
        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # structure → query
        self.W_K = nn.Linear(d_model, d_model, bias=False)  # dynamics → key
        self.W_V = nn.Linear(d_model, d_model, bias=False)  # dynamics → value
        self.W_O = nn.Linear(d_model, d_model)              # output projection

        # ── Normalisation and dropout ──────────────────────────────────────────
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # ── Post-attention FFN ─────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        H_geom: torch.Tensor,    # (N, d_model) structure query source
        H_star: torch.Tensor,    # (N, d_model) dynamics key/value source
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention fusion.

        Args:
            H_geom: (N, d_model) geometry path (structure)
            H_star: (N, d_model) conformational pool (dynamics)
            return_attn: if True, return attention weights for interpretability

        Returns:
            H_fused: (N, d_model) fused representation
            attn (optional): (n_heads, N, N) attention weights
        """
        N = H_geom.shape[0]

        # ── Pre-norm: normalise inputs ─────────────────────────────────────────
        H_geom_norm = self.norm1(H_geom)
        H_star_norm = self.norm1(H_star)

        # ── Project to Q, K, V ─────────────────────────────────────────────────
        Q = self.W_Q(H_geom_norm)  # (N, d_model) — structure asks
        K = self.W_K(H_star_norm)  # (N, d_model) — dynamics provides
        V = self.W_V(H_star_norm)  # (N, d_model) — dynamics provides

        # ── Reshape for multi-head attention ───────────────────────────────────
        Q = Q.reshape(N, self.n_heads, self.d_head).transpose(0, 1)  # (h, N, d_head)
        K = K.reshape(N, self.n_heads, self.d_head).transpose(0, 1)  # (h, N, d_head)
        V = V.reshape(N, self.n_heads, self.d_head).transpose(0, 1)  # (h, N, d_head)

        # ── Scaled dot-product attention ───────────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (h, N, N)
        attn_weights = F.softmax(scores, dim=-1)  # (h, N, N)
        attn_weights = self.dropout_attn(attn_weights)

        # ── Attention output ───────────────────────────────────────────────────
        attn_out = torch.matmul(attn_weights, V)  # (h, N, d_head)
        attn_out = attn_out.transpose(0, 1).reshape(N, self.d_model)  # (N, d_model)

        # ── Output projection ──────────────────────────────────────────────────
        attn_out = self.W_O(attn_out)

        # ── Residual connection (add geometry back) ────────────────────────────
        H_attn = self.norm2(H_geom + attn_out)

        # ── FFN sublayer ───────────────────────────────────────────────────────
        H_fused = H_attn + self.ffn(H_attn)

        if return_attn:
            return H_fused, attn_weights
        else:
            return H_fused


class CrossAttentionBlock(nn.Module):
    """Reusable cross-attention block with pre-norm and residuals."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = StructureDynamicsCrossAttention(d_model, n_heads, dropout)

    def forward(
        self,
        x_query: torch.Tensor,   # source for Q
        x_kv: torch.Tensor,      # source for K, V
        return_attn: bool = False,
    ):
        """Apply cross-attention."""
        if return_attn:
            return self.attn(x_query, x_kv, return_attn=True)
        else:
            return self.attn(x_query, x_kv, return_attn=False)
