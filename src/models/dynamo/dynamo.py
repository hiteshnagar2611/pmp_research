"""
DynaMo: Dynamic Membrane Oracle for PMP binding residue prediction.

Full Phase 1 model combining GVP encoder, conformational attention,
membrane geometry path, cross-attention fusion, and physicochemical gating.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .fusion import FeatureFusion
from .conf_attention import ConformationalAttentionPool
from .cross_attention import StructureDynamicsCrossAttention
from .geometry_path import MembraneGeometryPath
from .phys_gate import PhysiochemicalGate


class DynaMo(nn.Module):
    """
    Dynamic Membrane Oracle: predict which residues bind the membrane.
    
    Architecture:
      1. GVP-GNN encoder (shared across snapshots)
      2. Conformational attention pool (T snapshots → H*)
      3. Membrane geometry path (H_geom)
      4. Cross-attention fusion (H_geom queries H*)
      5. Physicochemical gate
      6. Per-residue MLP classifier
    """

    def __init__(
        self,
        plm_dim: int = 1280,
        struct_scalar_dim: int = 19,
        node_vector_dim: int = 6,
        hidden_scalar: int = 256,
        hidden_vector: int = 16,
        gvp_layers: int = 3,
        conf_n_heads: int = 8,
        cross_n_heads: int = 8,
        clf_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Feature fusion ─────────────────────────────────────────────────────
        self.fusion = FeatureFusion(
            plm_dim=plm_dim,
            plm_proj_dim=128,
            struct_scalar_dim=struct_scalar_dim,
            node_vector_dim=node_vector_dim,
            hidden_scalar=hidden_scalar,
            hidden_vector=hidden_vector,
        )

        # Placeholder: GVP encoder would go here
        # self.gvp_encoder = GVPEncoder(...)

        # ── Conformational attention pool ──────────────────────────────────────
        self.conf_pool = ConformationalAttentionPool(
            d_model=hidden_scalar,
            n_heads=conf_n_heads,
            dropout=dropout,
        )

        # ── Membrane geometry path ─────────────────────────────────────────────
        self.geom_path = MembraneGeometryPath(hidden_dim=hidden_scalar)

        # ── Cross-attention fusion ─────────────────────────────────────────────
        self.cross_attn = StructureDynamicsCrossAttention(
            d_model=hidden_scalar,
            n_heads=cross_n_heads,
            dropout=dropout,
        )

        # ── Physicochemical gate ───────────────────────────────────────────────
        self.phys_gate = PhysiochemicalGate(d_model=hidden_scalar)

        # ── Per-residue classifier ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_scalar, clf_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1),
        )

    def forward(
        self,
        H_static: torch.Tensor,     # (N, 256) static GVP encoding
        H_snapshots: torch.Tensor,  # (T, N, 256) MD snapshot encodings
        rmsf: torch.Tensor,         # (N, 1) per-residue RMSF
        depth: torch.Tensor,        # (N, 1) membrane depth
        kd: torch.Tensor,           # (N, 1) Kyte-Doolittle score
        charge: torch.Tensor,       # (N, 1) net charge
        sasa: torch.Tensor,         # (N, 1) relative SASA
        normal: torch.Tensor = None,  # (3,) membrane normal
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Returns:
            logits: (N,) per-residue logits for binding classification
            attn_weights: (n_heads, N, N) cross-attention for interpretability
        """
        # Conformational pool: compress T snapshots → H*
        H_star = self.conf_pool(H_static, H_snapshots, rmsf)

        # Membrane geometry: encode depth, charge, etc. → H_geom
        H_geom = self.geom_path(depth, tilt=None, mem_sasa=sasa, amph_score=None, normal=normal)

        # Cross-attention: structure queries dynamics
        H_fused, attn_weights = self.cross_attn(H_geom, H_star, return_attn=True)

        # Physicochemical gate
        H_gated = self.phys_gate(H_fused, kd, charge, sasa)

        # Classifier
        logits = self.classifier(H_gated).squeeze(-1)  # (N,)

        return logits, {"attn_weights": attn_weights, "H_fused": H_fused}

