"""
Conformational attention pool module for DynaMo Phase 1.

Pools T MD snapshots per residue using learned attention with RMSF-adaptive temperature.
Key novelty: temperature τ_r is per-residue and RMSF-dependent, giving flexible regions
broader attention distributions over conformations.

Usage:
    pool = ConformationalAttentionPool(d_model=256, n_heads=8)
    H_star = pool(H_static, H_snapshots, rmsf)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformationalAttentionPool(nn.Module):
    """
    Pool T MD snapshots via RMSF-adaptive multi-head attention.

    For each residue r:
      - Compute RMSF-dependent temperature τ_r
      - Build query q_r from static structure + RMSF embedding
      - Score each snapshot t via dot product q_r · k_r,t
      - Weight by softmax(score / τ_r)
      - Aggregate snapshot values with those weights
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        rmsf_emb_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # ── RMSF → embedding (scalar → vector) ────────────────────────────────
        self.rmsf_emb = nn.Sequential(
            nn.Linear(1, rmsf_emb_dim),
            nn.SiLU(),
            nn.Linear(rmsf_emb_dim, d_model),
        )

        # ── RMSF → temperature (scalar → positive scalar) ──────────────────────
        self.temp_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # ensures τ > 0
        )

        # ── Query, key, value projections ──────────────────────────────────────
        self.W_q = nn.Linear(d_model * 2, d_model, bias=False)  # concat H_static + RMSF_emb
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # ── Output projection ──────────────────────────────────────────────────
        self.W_o = nn.Linear(d_model, d_model)

        # ── Normalisation and dropout ──────────────────────────────────────────
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_static: torch.Tensor,        # (N, d_model) GVP output on reference structure
        H_snapshots: torch.Tensor,     # (T, N, d_model) GVP output on T MD snapshots
        rmsf: torch.Tensor,            # (N, 1) or (N,) per-residue RMSF from MD
    ) -> torch.Tensor:
        """
        Pool ensemble representation.

        Args:
            H_static: (N, d_model) static structure representation
            H_snapshots: (T, N, d_model) T snapshot representations
            rmsf: (N, 1) or (N,) per-residue flexibility

        Returns:
            H_star: (N, d_model) ensemble-pooled representation
        """
        N = H_static.shape[0]
        T = H_snapshots.shape[0]
        device = H_static.device

        # Ensure RMSF is (N, 1)
        if rmsf.dim() == 1:
            rmsf = rmsf.unsqueeze(-1)  # (N,) → (N, 1)

        # ── Build per-residue temperature from RMSF ────────────────────────────
        tau = self.temp_mlp(rmsf)  # (N, 1), always > 0

        # ── Build query: static structure + RMSF embedding ─────────────────────
        rmsf_emb = self.rmsf_emb(rmsf)  # (N, d_model)
        q = self.W_q(torch.cat([H_static, rmsf_emb], dim=-1))  # (N, d_model)

        # ── Build keys and values for all snapshots ────────────────────────────
        k = self.W_k(H_snapshots)  # (T, N, d_model)
        v = self.W_v(H_snapshots)  # (T, N, d_model)

        # ── Compute attention scores: q · k / τ ────────────────────────────────
        # q: (N, d_model),  k: (T, N, d_model)  →  (T, N) scores
        scores = torch.einsum('nd,tnd->tn', q, k) / tau.squeeze(-1)  # (T, N)

        # ── Softmax over snapshots (dim=0) ─────────────────────────────────────
        alpha = F.softmax(scores, dim=0)  # (T, N), each column sums to 1
        alpha = self.dropout(alpha)

        # ── Weighted sum: alpha (T,N) · v (T,N,d) → (N, d) ───────────────────
        H_star = torch.einsum('tn,tnd->nd', alpha, v)  # (N, d_model)

        # ── Residual connection + output projection ────────────────────────────
        H_star = self.W_o(H_star)
        H_star = self.norm(H_star + H_static)  # residual + LayerNorm

        return H_star


class ConformationalAttentionPoolMultiHead(nn.Module):
    """
    Multi-head variant of conformational attention pool.
    Each head learns different attention patterns across conformations.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        rmsf_emb_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # ── RMSF embeddings (shared across heads) ──────────────────────────────
        self.rmsf_emb = nn.Sequential(
            nn.Linear(1, rmsf_emb_dim),
            nn.SiLU(),
            nn.Linear(rmsf_emb_dim, d_model),
        )

        self.temp_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

        # ── Per-head projections ───────────────────────────────────────────────
        self.W_q = nn.ModuleList([
            nn.Linear(d_model * 2, self.d_head, bias=False)
            for _ in range(n_heads)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(d_model, self.d_head, bias=False)
            for _ in range(n_heads)
        ])
        self.W_v = nn.ModuleList([
            nn.Linear(d_model, self.d_head, bias=False)
            for _ in range(n_heads)
        ])

        # ── Output projection ──────────────────────────────────────────────────
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_static: torch.Tensor,        # (N, d_model)
        H_snapshots: torch.Tensor,     # (T, N, d_model)
        rmsf: torch.Tensor,            # (N, 1) or (N,)
    ) -> torch.Tensor:
        """Pool with multi-head attention."""
        N = H_static.shape[0]
        T = H_snapshots.shape[0]

        if rmsf.dim() == 1:
            rmsf = rmsf.unsqueeze(-1)

        tau = self.temp_mlp(rmsf).squeeze(-1)  # (N,)
        rmsf_emb = self.rmsf_emb(rmsf)  # (N, d_model)

        # Concatenate query input
        q_input = torch.cat([H_static, rmsf_emb], dim=-1)  # (N, d_model*2)

        # Multi-head attention: each head independently
        head_outputs = []
        for h in range(self.n_heads):
            q_h = self.W_q[h](q_input)  # (N, d_head)
            k_h = self.W_k[h](H_snapshots)  # (T, N, d_head)
            v_h = self.W_v[h](H_snapshots)  # (T, N, d_head)

            # Score and softmax
            scores_h = torch.einsum('nd,tnd->tn', q_h, k_h) / (self.d_head ** 0.5 * tau)
            alpha_h = F.softmax(scores_h, dim=0)  # (T, N)
            alpha_h = self.dropout(alpha_h)

            # Aggregate
            out_h = torch.einsum('tn,tnd->nd', alpha_h, v_h)  # (N, d_head)
            head_outputs.append(out_h)

        # Concatenate heads
        H_star = torch.cat(head_outputs, dim=-1)  # (N, d_model)
        H_star = self.W_o(H_star)
        H_star = self.norm(H_star + H_static)

        return H_star
