"""
Feature fusion module for DynaMo.

Fuses:
  - PLM embeddings (ESM-2, 1280-dim)
  - Structural scalar features (backbone, physicochem, exposure, PMP)
  - Into balanced input for GVP-GNN encoder

Key design: independent LayerNorm on each stream before concatenation
to prevent PLM from dominating gradient signal.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Fuse PLM embeddings and structural features for GVP input."""

    def __init__(
        self,
        plm_dim: int = 1280,
        plm_proj_dim: int = 128,
        struct_scalar_dim: int = 19,
        node_vector_dim: int = 6,
        hidden_scalar: int = 256,
        hidden_vector: int = 16,
        gvp_layers: int = 3,
    ):
        super().__init__()

        self.plm_dim = plm_dim
        self.plm_proj_dim = plm_proj_dim
        self.struct_scalar_dim = struct_scalar_dim
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector

        # ── PLM projection with small init std ────────────────────────────────
        self.plm_proj = nn.Linear(plm_dim, plm_proj_dim)
        nn.init.normal_(self.plm_proj.weight, std=0.01)
        nn.init.zeros_(self.plm_proj.bias)

        # ── Independent normalisation per stream ─────────────────────────────
        self.plm_norm = nn.LayerNorm(plm_proj_dim)
        self.struct_norm = nn.LayerNorm(struct_scalar_dim)

        # ── Fused scalar normalisation ───────────────────────────────────────
        total_scalar_in = plm_proj_dim + struct_scalar_dim
        self.fused_norm = nn.LayerNorm(total_scalar_in)

        # ── GVP projection heads ─────────────────────────────────────────────
        # Project from concatenated features to GVP hidden dimensions
        self.W_s = nn.Linear(total_scalar_in, hidden_scalar)
        self.W_v = nn.Linear(node_vector_dim * 3, hidden_vector * node_vector_dim)

        # ── Optional: post-fusion MLP compression ────────────────────────────
        self.compress = nn.Sequential(
            nn.Linear(total_scalar_in, hidden_scalar),
            nn.GELU(),
            nn.LayerNorm(hidden_scalar),
        )

    def forward(
        self,
        plm_emb: torch.Tensor,      # (N, 1280) ESM-2 embedding
        s_struct: torch.Tensor,     # (N, 19) structural scalars
        V_node: torch.Tensor,       # (N, 6, 3) structural vectors
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse features.

        Returns:
            s_fused (N, hidden_scalar): balanced scalar representation
            V_fused (N, hidden_vector, 3): vector representation (copied, not modified)
        """
        N = plm_emb.shape[0]

        # Step 1: Project and normalise each stream independently
        psi = self.plm_norm(self.plm_proj(plm_emb))   # (N, plm_proj_dim)
        s_struct_norm = self.struct_norm(s_struct)     # (N, struct_scalar_dim)

        # Step 2: Concatenate scalar streams
        s_concat = torch.cat([psi, s_struct_norm], dim=-1)  # (N, total_scalar_in)
        s_concat = self.fused_norm(s_concat)

        # Step 3: Compress to hidden dimension via MLP
        s_fused = self.compress(s_concat)  # (N, hidden_scalar)

        # Step 4: Vector features — project and reshape
        # Flatten vector features for projection
        V_flat = V_node.reshape(N, -1)     # (N, 6*3 = 18)
        V_proj = self.W_v(V_flat)          # (N, hidden_vector*3)
        V_fused = V_proj.reshape(N, self.hidden_vector, 3)  # (N, hidden_vector, 3)

        return s_fused, V_fused


class StructuralEmbedding(nn.Module):
    """Lightweight embedding module for structural features."""

    def __init__(self, struct_scalar_dim: int = 19, output_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(struct_scalar_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, output_dim),
        )

    def forward(self, s_struct: torch.Tensor) -> torch.Tensor:
        """Embed structural features."""
        return self.mlp(s_struct)


class PLMEmbedding(nn.Module):
    """PLM embedding projection with small initialization."""

    def __init__(self, plm_dim: int = 1280, output_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(plm_dim, output_dim)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, plm_emb: torch.Tensor) -> torch.Tensor:
        """Project PLM embeddings."""
        return self.norm(self.proj(plm_emb))
