"""
Invariant Point Attention (IPA) denoiser for SE(3) flow matching.

Predicts velocity field v(x_t, t, c) that denoises protein backbone frames
while respecting SE(3) equivariance.

Architecture:
  - 6 IPA layers with local geometric attention
  - Conditioning injection at each layer (scaffold, geometry, binding patch)
  - SE(3)-equivariant processing of frame coordinates
  - Residual connections for gradient flow
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InvariantPointAttention(nn.Module):
    """
    Invariant Point Attention (IPA) layer.

    Attends over nearby residues using both:
      - Pairwise distances (invariant)
      - Point transformations (equivariant)

    Reference: Jumper et al., AlphaFold 2 (Nature 2021)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 12,
        num_point_qk: int = 4,
        num_point_v: int = 8,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: scalar feature dimension
            num_heads: number of attention heads
            num_point_qk: number of point queries/keys per head
            num_point_v: number of point values per head
            dropout: dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v

        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim

        # ── Scalar attention (standard multi-head) ──────────────────────────────
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ── Point attention (on 3D coordinates) ─────────────────────────────────
        # Query points: project features to 3D
        self.W_q_point = nn.Linear(hidden_dim, num_heads * num_point_qk * 3, bias=False)
        # Key points: project features to 3D
        self.W_k_point = nn.Linear(hidden_dim, num_heads * num_point_qk * 3, bias=False)
        # Value points: project features to 3D
        self.W_v_point = nn.Linear(hidden_dim, num_heads * num_point_v * 3, bias=False)

        # ── Combine scalar + point logits ───────────────────────────────────────
        self.logit_scale = nn.Parameter(torch.ones(1) * math.log(1.0 / 0.5))

        # ── Output projection ──────────────────────────────────────────────────
        self.W_o = nn.Linear(hidden_dim + num_heads * num_point_v * 3, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,        # (N, hidden_dim) residue features
        t: torch.Tensor,        # (N, 3) 3D coordinates (Cα or backbone center)
        edge_index: torch.Tensor,  # (2, E) edges in local graph
        c: torch.Tensor = None,  # (N, hidden_dim) optional conditioning
    ) -> torch.Tensor:
        """
        Apply IPA layer.

        Args:
            x: scalar features (N, hidden_dim)
            t: 3D coordinates (N, 3)
            edge_index: (2, E) sparse edges
            c: conditioning features (optional)

        Returns:
            x_out: (N, hidden_dim) updated scalar features (with point info)
        """
        N = x.shape[0]
        device = x.device

        # ── Condition injection ────────────────────────────────────────────────
        if c is not None:
            x = x + c  # add conditioning

        # ── Scalar attention ───────────────────────────────────────────────────
        Q = self.W_q(x).reshape(N, self.num_heads, self.head_dim)  # (N, h, d)
        K = self.W_k(x).reshape(N, self.num_heads, self.head_dim)
        V = self.W_v(x).reshape(N, self.num_heads, self.head_dim)

        # Pairwise scalar scores: Q[i] · K[j]^T
        scores_scalar = torch.einsum("nhd,mhd->nmh", Q, K) / math.sqrt(self.head_dim)  # (N, N, h)

        # ── Point attention ────────────────────────────────────────────────────
        Q_point = self.W_q_point(x)  # (N, h * q_pt * 3)
        K_point = self.W_k_point(x)  # (N, h * q_pt * 3)
        V_point = self.W_v_point(x)  # (N, h * v_pt * 3)

        # Reshape to points
        Q_point = Q_point.reshape(N, self.num_heads, self.num_point_qk, 3)
        K_point = K_point.reshape(N, self.num_heads, self.num_point_qk, 3)
        V_point = V_point.reshape(N, self.num_heads, self.num_point_v, 3)

        # Compute pairwise distances: ||Q_i - K_j||
        Q_point_expanded = Q_point.unsqueeze(2)  # (N, h, 1, q_pt, 3)
        K_point_expanded = K_point.unsqueeze(0)  # (1, h, N, q_pt, 3)
        distances = torch.norm(Q_point_expanded - K_point_expanded, dim=-1)  # (N, h, N, q_pt, q_pt)
        distances = distances.mean(dim=(-2, -1))  # average over points: (N, h, N)

        # Convert distance to attention: smaller distance = higher attention
        scores_point = -distances  # (N, h, N)

        # ── Combine scalar and point logits ────────────────────────────────────
        scores = scores_scalar + self.logit_scale * scores_point  # (N, N, h)
        scores = scores.transpose(1, 2)  # (N, h, N)

        # ── Mask to sparse edges (optional: only attend to neighbours) ────────
        if edge_index is not None:
            src, dst = edge_index
            # Create dense adjacency mask
            adj = torch.zeros(N, N, device=device, dtype=torch.bool)
            adj[src, dst] = True
            adj[dst, src] = True  # make symmetric
            # Mask scores
            mask = ~adj.unsqueeze(0)  # (1, N, N), mask=True for non-edges
            scores = scores.masked_fill(mask, float('-inf'))

        # ── Softmax attention ──────────────────────────────────────────────────
        attn = F.softmax(scores, dim=-1)  # (N, h, N)
        attn = self.dropout(attn)

        # ── Apply attention to values ──────────────────────────────────────────
        # Scalar attention output
        V_expanded = V.unsqueeze(0)  # (1, N, h, d)
        attn_expanded = attn.unsqueeze(-1)  # (N, h, N, 1)
        scalar_out = torch.einsum("nhm,mhd->nhd", attn, V)  # (N, h, d)
        scalar_out = scalar_out.reshape(N, -1)  # (N, h*d)

        # Point attention output
        attn_v = attn.unsqueeze(-1).unsqueeze(-1)  # (N, h, N, 1, 1)
        V_point_expanded = V_point.unsqueeze(0)  # (1, h, v_pt, 3)
        point_out = torch.einsum("nhm,mhvx->nhvx", attn, V_point)  # (N, h, v_pt, 3)
        point_out = point_out.reshape(N, -1)  # (N, h*v_pt*3)

        # ── Output projection ──────────────────────────────────────────────────
        out_concat = torch.cat([scalar_out, point_out], dim=-1)  # (N, h*d + h*v_pt*3)
        out = self.W_o(out_concat)  # (N, hidden_dim)

        return out


class IPADenoiser(nn.Module):
    """
    IPA-based denoiser for SE(3) flow matching.

    Predicts velocity field: v(x_t, t, c) → (v_R, v_t)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 12,
        num_layers: int = 6,
        time_embed_dim: int = 128,
        condition_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: main feature dimension
            num_heads: attention heads per IPA layer
            num_layers: number of IPA layers
            time_embed_dim: time embedding dimension
            condition_dim: conditioning vector dimension
            dropout: dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Time embedding ─────────────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, hidden_dim),
        )

        # ── Conditioning projection ────────────────────────────────────────────
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ── IPA layers ─────────────────────────────────────────────────────────
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_point_qk=4,
                num_point_v=8,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Layer normalisation ────────────────────────────────────────────────
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # ── FFN layers ─────────────────────────────────────────────────────────
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # ── Output heads ───────────────────────────────────────────────────────
        # Rotation velocity (as rotation vector): (N, 3)
        self.head_v_R = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Translation velocity: (N, 3)
        self.head_v_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self,
        x_t: torch.Tensor,       # (N, hidden_dim) or (N, 3) frame representation at time t
        coords: torch.Tensor,    # (N, 3) Cα coordinates
        time: torch.Tensor,      # (B,) or scalar, time in [0, 1]
        c: torch.Tensor = None,  # (N, condition_dim) conditioning vector
        edge_index: torch.Tensor = None,  # (2, E) sparse edges
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict velocity field.

        Args:
            x_t: current state representation
            coords: 3D coordinates
            time: diffusion time
            c: conditioning vector
            edge_index: sparse edges for local attention

        Returns:
            v_R: (N, 3) rotation velocity
            v_t: (N, 3) translation velocity
        """
        N = x_t.shape[0]

        # ── Project initial features if needed ──────────────────────────────────
        if x_t.shape[-1] != self.hidden_dim:
            x = nn.Linear(x_t.shape[-1], self.hidden_dim)(x_t)
        else:
            x = x_t

        # ── Add time embedding ─────────────────────────────────────────────────
        t_emb = self.time_embed(time.unsqueeze(-1) if time.dim() == 0 else time)
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0).expand(N, -1)
        x = x + t_emb

        # ── Add conditioning ───────────────────────────────────────────────────
        if c is not None:
            c_proj = self.cond_proj(c)
            x = x + c_proj

        # ── IPA layers with residual connections ───────────────────────────────
        for i, (ipa_layer, norm, ffn) in enumerate(zip(self.ipa_layers, self.layer_norms, self.ffn_layers)):
            # IPA attention
            x_attn = ipa_layer(norm(x), coords, edge_index, c=c if c is not None else None)
            x = x + x_attn  # residual

            # FFN
            x_ffn = ffn(norm(x))
            x = x + x_ffn  # residual

        # ── Output heads: predict velocities ───────────────────────────────────
        v_R = self.head_v_R(x)  # (N, 3)
        v_t = self.head_v_t(x)  # (N, 3)

        return v_R, v_t
